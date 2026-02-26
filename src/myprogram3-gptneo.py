#!/usr/bin/env python
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.examples = []
        
        print(f'Processing {len(texts)} text samples...')
        for text in tqdm(texts, desc="Tokenizing"):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['input_ids']
        }


class GPTNeoModel:
    def __init__(self):
        self.model_name = 'EleutherAI/gpt-neo-125m'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )
        print(f'Using device: {self.device}')
        self.model.to(self.device)

    @classmethod
    def load_training_data(cls, data_path):
        all_texts = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")
        from datasets import load_dataset
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".arrow"):
                    file_path = os.path.join(root, file)
                    dataset = load_dataset("arrow", data_files=file_path)["train"]
                    if "text" in dataset.column_names:
                        all_texts.extend(dataset["text"])
        return all_texts

    @classmethod
    def load_test_data(cls, fname):
        with open(fname, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, epochs=3, batch_size=8, learning_rate=5e-5):
        if not data:
            print('No training data provided. Skipping training.')
            return
        
        print(f'Starting training for {epochs} epochs...')
        
        dataset = TextDataset(data, self.tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    print(f'Error in batch: {e}')
                    continue
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        print('Training complete!')

    def predict_next_chars(self, text, top_k=3):
        self.model.eval()
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            top_token_ids = torch.topk(probs, k=min(100, len(probs))).indices
            
            seen_chars = set()
            top_chars = []
            
            for token_id in top_token_ids:
                token_str = self.tokenizer.decode([token_id.item()])
                
                if token_str and len(token_str) > 0:
                    first_char = token_str[0]
                    
                    if first_char not in seen_chars:
                        seen_chars.add(first_char)
                        top_chars.append(first_char)
                        
                        if len(top_chars) == top_k:
                            break
            
            common_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n']
            for char in common_chars:
                if len(top_chars) >= top_k:
                    break
                if char not in seen_chars:
                    top_chars.append(char)
            
            while len(top_chars) < top_k:
                top_chars.append(' ')
            
            return top_chars[:top_k]
            
        except Exception as e:
            print(f'Error predicting for text "{text}": {e}')
            return [' ', 'e', 't']

    def run_pred(self, data, batch_size=32):
        print(f'Generating predictions for {len(data)} samples...')
        self.model.eval()
        
        preds = []
        
        for inp in tqdm(data, desc='Predicting'):
            top_chars = self.predict_next_chars(inp, top_k=3)
            preds.append(''.join(top_chars))
        
        return preds

    def save(self, work_dir):
        print(f'Saving model to {work_dir}...')
        model_path = os.path.join(work_dir, 'model_gptneo')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print('Model saved successfully!')

    @classmethod
    def load(cls, work_dir):
        model_path = os.path.join(work_dir, 'model_gptneo')
        
        if not os.path.exists(model_path):
            print(f'Warning: No trained model found at {model_path}')
            print('Using base GPT-Neo 125M model without fine-tuning.')
            return cls()
        
        print(f'Loading model from {model_path}...')
        model = cls.__new__(cls)
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        model.device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if torch.backends.mps.is_available() 
            else 'cpu'
        )
        print(f'Using device: {model.device}')
        model.model.to(model.device)
        model.model.eval()
        
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', help='path to training data', default='data')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred_gptneo.txt')
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=3)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=5e-5)
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            os.makedirs(args.work_dir)
        
        print('Instantiating GPT-Neo 125M model')
        model = GPTNeoModel()
        
        train_data = []
        if os.path.isdir(args.train_data):
            for d in os.listdir(args.train_data):
                f_dir = os.path.join(args.train_data, d)
                if os.path.isdir(f_dir) and d.startswith("fleurs"):
                    try:
                        train_data.extend(GPTNeoModel.load_training_data(f_dir))
                    except Exception as e:
                        print(f'Skipping {f_dir}: {e}')
        
        if train_data:
            print('Training model...')
            model.run_train(train_data, args.work_dir, epochs=args.epochs, 
                           batch_size=args.batch_size, learning_rate=args.learning_rate)
            model.save(args.work_dir)
        else:
            print('No training data found. Saving base model.')
            model.save(args.work_dir)
            
    elif args.mode == 'test':
        print('Loading model')
        model = GPTNeoModel.load(args.work_dir)
        
        print('Loading test data from {}'.format(args.test_data))
        test_data = GPTNeoModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data)
        model.write_pred(pred, args.test_output)
        print('Done!')
        
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
