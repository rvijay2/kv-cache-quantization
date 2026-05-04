import torch

def quantize_tensor(x):
    max_val = x.abs().max()
    scale = max_val / 127 if max_val != 0 else 1.0
    q = torch.round(x / scale).to(torch.int8)
    return q, scale

def compute_kv_memory(past_key_values, mode, step, threshold):
    total_memory = 0

    for layer in past_key_values:
        k = layer[0]
        v = layer[1]

        if mode == "fp32":
            total_memory += k.element_size() * k.numel()
            total_memory += v.element_size() * v.numel()

        elif mode == "int8":
            k_q, _ = quantize_tensor(k)
            v_q, _ = quantize_tensor(v)

            total_memory += k_q.element_size() * k_q.numel()
            total_memory += v_q.element_size() * v_q.numel()

        elif mode == "scheduled":
            if step > threshold:
                k_q, _ = quantize_tensor(k)
                v_q, _ = quantize_tensor(v)

                total_memory += k_q.element_size() * k_q.numel()
                total_memory += v_q.element_size() * v_q.numel()
            else:
                total_memory += k.element_size() * k.numel()
                total_memory += v.element_size() * v.numel()

    return total_memory / (1024 ** 2)  # MB