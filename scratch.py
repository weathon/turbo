import json

with open("test_prompt.jsonl", "r") as f:
    prompts_ = [json.loads(line) for line in f.readlines()]

prompts = []
for prompt in prompts_:
    for p in prompt["prompts"]:
        prompts.append({
            "pos": p["prompt"],
            "neg": p["missing_element"],
        }) 
    
with open("prompts2.json", "w") as f:
    json.dump(prompts, f, indent=4)
print(f"Generated {len(prompts)} prompts.")