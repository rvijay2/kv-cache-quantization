import json
import matplotlib.pyplot as plt

with open("results/output.json") as f:
    data = json.load(f)

modes = ["fp32", "int8", "scheduled"]

def avg(metric):
    return [sum(d[metric] for d in data if d["mode"] == m)/3 for m in modes]

mem = avg("kv_memory_MB")
lat = avg("latency")

# MEMORY
plt.figure()
plt.bar(modes, mem)
plt.title("KV Cache Memory Comparison")
plt.ylabel("Memory (MB)")
plt.savefig("results/memory.png")

# LATENCY
plt.figure()
plt.barh(modes, lat)
plt.title("Latency Comparison")
plt.xlabel("Seconds")
plt.savefig("results/latency.png")

# TRADE-OFF
x = range(len(modes))
w = 0.25

plt.figure()
plt.bar([i - w/2 for i in x], mem, w, label="Memory")
plt.bar([i + w/2 for i in x], lat, w, label="Latency")

plt.xticks(x, modes)
plt.title("Memory vs Latency Trade-off")
plt.legend()

plt.savefig("results/tradeoff.png")

print("Final graphs generated.")