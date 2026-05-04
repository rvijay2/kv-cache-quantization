import torch
from transformers import GPT2LMHeadModel
from config import *
from kv_scheduler import compute_kv_memory
from metrics import get_memory, Timer
from utils import get_tokenizer
from perplexity import compute_perplexity
import json
import os

device = "cpu"

def run_mode(mode, seq_len):
    tokenizer = get_tokenizer()
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    prompt = "Machine learning is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    past_key_values = None
    total_time = 0
    total_kv_memory = 0

    for step in range(MAX_NEW_TOKENS):
        timer = Timer()

        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        kv_mem = compute_kv_memory(
            past_key_values,
            mode,
            step,
            FP32_TOKENS
        )

        total_kv_memory += kv_mem

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = next_token

        total_time += timer.elapsed()

    avg_kv_memory = total_kv_memory / MAX_NEW_TOKENS

    perplexity = compute_perplexity(model, tokenizer, device)

    return {
        "mode": mode,
        "sequence_length": seq_len,
        "latency": total_time,
        "kv_memory_MB": avg_kv_memory,
        "perplexity": perplexity
    }


def main():
    results = []

    for mode in MODES:
        for seq_len in SEQUENCE_LENGTHS:
            print(f"Running {mode} | seq_len={seq_len}")
            res = run_mode(mode, seq_len)
            results.append(res)

    os.makedirs("results", exist_ok=True)

    with open("results/output.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved.")


if __name__ == "__main__":
    main()