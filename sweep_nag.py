import torch
import sys
sys.path.append("Normalized-Attention-Guidance")
from src.pipeline_sd3_nag import NAGStableDiffusion3Pipeline
import json
import judge
import wandb
import numpy as np

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipe = NAGStableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open("dev_prompts.json", "r") as f:
    dev_prompts = json.load(f)

def run():
    wandb.init(project="nag-sweep")
    nag_scale = wandb.config.nag_scale
    nag_alpha = wandb.config.nag_alpha
    nag_tau = wandb.config.nag_tau
    scores = np.zeros(2)
    for seed in range(2):
        for i in dev_prompts:
            image = pipe(
                i["pos"],
                nag_negative_prompt=i["neg"],
                guidance_scale=0.,
                nag_scale=nag_scale,
                nag_alpha=nag_alpha,
                nag_tau=nag_tau,
                num_inference_steps=8,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            scores += judge.ask_gpt(image, i["pos"], i["neg"])
            wandb.log({"pos_score": scores[0], "neg_score": scores[1], "total_score": np.sum(scores)})
        
        
sweep_configuration = {
    "method": "bayes", 
    "metric": {"goal": "maximize", "name": "total_score"},
    "parameters": {
        "nag_scale": {"min": 4, "max": 10},
        "nag_alpha": {"min": 0.0, "max": 1.0},
        "nag_tau": {"min": 1.0, "max": 10.0},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="nag-sweep")

wandb.agent(sweep_id, function=run, count=64)