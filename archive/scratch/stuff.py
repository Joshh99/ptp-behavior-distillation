import json 
data = json.load(open('benchmark_results/rulearena/metrics.json')) 
print('Experiments:', list(data.keys())) 
[print(f"{name}: acc={m["accuracy_tolerance"]:.1%}, cost=${m["total_cost_usd"]:.4f}") for name, m in data.items()]