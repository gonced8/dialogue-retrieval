import json

with open("../generation/results/gpt4_retrieval_multiwoz.jsonl", "r") as f:
    results1 = [json.loads(line) for line in f]

with open("../generation/results/gpt4_taskmaster.jsonl", "r") as f:
    results2 = [json.loads(line) for line in f]

with open("../generation/results/gpt4_multiwoz.jsonl", "r") as f:
    results3 = [json.loads(line) for line in f]

multiwoz_ids = [result["id"] for result in results3]
taskmaster_ids = [result["id"] for result in results2]

results_multiwoz = [result for result in results1 if result["id"] in multiwoz_ids]
results_taskmaster = [result for result in results1 if result["id"] in taskmaster_ids]

with open("../generation/results/gpt4_retrieval_multiwoz.jsonl", "w") as f:
    for result in results_multiwoz:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

with open("../generation/results/gpt4_retrieval_taskmaster.jsonl", "w") as f:
    for result in results_taskmaster:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
