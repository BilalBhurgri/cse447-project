#!/usr/bin/env python
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import unicodedata

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len=512):
        super().__init__()

        d_model = 384
        nhead = 8
        num_layers = 6
        dim_feedforward = 1024
        dropout = 0.1

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(positions)
        mask = self.causal_mask[:T, :T]

        x = self.transformer(x, mask=mask)
        return self.lm_head(x)



class MyModel:
    def __init__(self, texts=None, seq_len=256):

        self.seq_len = seq_len
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if texts is not None and len(texts) > 0:
            self._build_vocab(texts)
            self._build_model()
            self.model.to(self.device)

        print(f"Using device: {self.device}")

    @classmethod
    def load_training_data(cls, data_path):
        all_texts = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                file_path = os.path.join(data_path, file)
                print(f"Loading {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Normalize text
                        line = unicodedata.normalize("NFKC", line)

                        all_texts.append(line)

        if len(all_texts) == 0:
            raise ValueError("No .txt files found in data directory")

        print(f"Loaded {len(all_texts)} text examples")
        return all_texts

    @classmethod
    def load_training_data_arrow(cls, data_path):
        all_texts = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".arrow"):
                    file_path = os.path.join(root, file)
                    print(f"Loading {file_path}")

                    dataset = load_dataset(
                        "arrow",
                        data_files=file_path
                    )["train"]

                    if "text" not in dataset.column_names:
                        raise ValueError(f"No 'text' column in {file_path}")
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
                f.write(f"{p}\n")


    def _build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.chars = [self.pad_token, self.unk_token] + sorted(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text):
        return [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text]

    def _build_model(self):
        self.model = CharTransformer(self.vocab_size, self.seq_len)


    def _create_dataset(self, texts):
        import random

        all_ids = []
        for text in texts:
            all_ids.extend(self.encode(text))

        if len(all_ids) < self.seq_len + 1:
            raise ValueError(
                f"Dataset too small. Need at least {self.seq_len+1} characters, "
                f"but got {len(all_ids)}"
            )

        num_chunks = len(all_ids) // self.seq_len
        chunks = []

        for _ in range(num_chunks):
            start = random.randint(0, len(all_ids) - self.seq_len - 1)
            chunk = all_ids[start:start+self.seq_len+1]
            chunks.append(chunk)

        print(f"Created {len(chunks)} training sequences")

        class CharDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                chunk = self.data[idx]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                return x, y

        return CharDataset(chunks)


    def run_train(self, texts, work_dir, epochs=10, batch_size=64, lr=2e-4):

        dataset = self._create_dataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)

                loss = criterion(
                    logits.view(-1, self.vocab_size),
                    y.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

        self.save(work_dir)


    def predict_next_chars(self, text, top_k=3):
        self.model.eval()

        encoded = self.encode(text)[-self.seq_len:]
        x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)

            # Remove whitespace predictions
            for ch in [" ", "\n", "\t", "\r"]:
                if ch in self.stoi:
                    probs[self.stoi[ch]] = 0.0

        probs = probs / probs.sum()
        top_ids = torch.topk(probs, k=top_k).indices.tolist()
        return [self.itos[i] for i in top_ids]
                

    def run_pred(self, data):
        preds = []
        for text in tqdm(data, desc="Predicting"):
            chars = self.predict_next_chars(text, top_k=3)
            preds.append("".join(chars))
        return preds


    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len
        }, os.path.join(work_dir, "model_20k.pt"))
        print("Model saved.")

    @classmethod
    def load(cls, work_dir):
        checkpoint = torch.load(
            os.path.join(work_dir, "model_20k.pt"),
            map_location="cpu"
        )

        model = cls(texts=[], seq_len=checkpoint["seq_len"])

        model.stoi = checkpoint["stoi"]
        model.itos = checkpoint["itos"]
        model.vocab_size = checkpoint["vocab_size"]

        model.pad_token = "<PAD>"
        model.unk_token = "<UNK>"

        model._build_model()
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.to(model.device)
        model.model.eval()

        return model

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'))
    parser.add_argument('--work_dir', default='work')
    parser.add_argument('--test_data', default='example/input.txt')
    parser.add_argument('--test_output', default='pred.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    if args.mode == 'train':

        data_dir = "data"

        # fleurs_dirs = [
        #     os.path.join(data_dir, d)
        #     for d in os.listdir(data_dir)
        #     if os.path.isdir(os.path.join(data_dir, d))
        #     and d.startswith("fleurs")
        # ]

        # train_texts = []
        # for f_dir in fleurs_dirs:
        #     train_texts.extend(MyModel.load_training_data(f_dir))
        train_texts = MyModel.load_training_data(data_dir)

        model = MyModel(texts=train_texts)
        model.run_train(
            train_texts,
            args.work_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == 'test':

        model = MyModel.load(args.work_dir)
        test_data = MyModel.load_test_data(args.test_data)

        preds = model.run_pred(test_data)

        assert len(preds) == len(test_data)
        MyModel.write_pred(preds, args.test_output)

        print("Done.")