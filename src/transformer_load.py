from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

text = "Once upon a time"
inputs = tokenizer.encode(text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)