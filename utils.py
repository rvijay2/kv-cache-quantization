from transformers import GPT2Tokenizer

def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")