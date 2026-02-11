FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install required dependencies
COPY requirements.txt /job/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download GPT-2 model to speed up inference (optional but recommended)
# This caches the model in the Docker image
RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')"
