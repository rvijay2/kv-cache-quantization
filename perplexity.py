import torch
from datasets import load_dataset

def compute_perplexity(model, tokenizer, device):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    text = " ".join(dataset["text"][:1000])
    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids.to(device)

    max_length = 1024
    stride = 512  

    nlls = []
    total_tokens = 0

    for i in range(0, input_ids.size(1), stride):
        begin_loc = i
        end_loc = min(i + max_length, input_ids.size(1))
        trg_len = end_loc - begin_loc

        input_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_tokens += trg_len

        if end_loc == input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()