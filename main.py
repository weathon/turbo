# %%
import wandb.util
import torch
from diffusers import StableDiffusion3Pipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
# pipe = pipe.to("cuda")
import sys
sys.path.append("Normalized-Attention-Guidance")

import torch
from processor import JointAttnProcessor2_0
from src.pipeline_sd3_nag import NAGStableDiffusion3Pipeline
from pipeline import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
if "pipe" not in locals():
  pipe = NAGStableDiffusion3Pipeline.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
  )
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
import random


import pandas as pd
import asyncio

try:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_path = '/content/drive/MyDrive/VSF_benchmark'
    
    COLAB = True
except ImportError:
    COLAB = False
    


from ours import inference
import numpy as np
from PIL import Image
import json
from judge import ask_gpt

with open("prompts2.json", "r") as f:
    prompts_data = json.load(f)

prompts = pd.read_csv("sampled.csv")
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []
import os
import time

wandb.init(project="VSF_benchmark")

import tqdm


scores = np.zeros((2, 2))
total = 0
lock = asyncio.Lock()

async def judge_async(image_ours, prompt, neg_prompt):
    global scores, total
    delta = await asyncio.to_thread(ask_gpt, image_ours, image_ours, prompt, neg_prompt)

    delta = delta.T
    with lock:
        scores += delta
        total += 1
        df = pd.DataFrame(
            scores / total,
            columns=["positive", "negative", "quality"],
            index=["ours", "vanilla"],
        )
    print("delta:\n", delta)
    print(df)

    wandb.log({
        "step": total,
        "score_board": wandb.Table(data=df),
    })


async def run(run_id, scale, offset):
    run_id = f"{run_id:03d}"
    seed = 42
    os.system("mkdir -p benchmark/" + run_id)
    with open(f"benchmark/{run_id}/config.txt", "w") as f:
        f.write(f"scale: {scale}\n")
        f.write(f"offset: {offset}\n")
        f.write(f"seed: {seed}\n")
    
    with open(f"benchmark/{run_id}/preview.md", "a") as f:
        f.write(f"# Run {run_id}\n")
        f.write(f"**Scale:** {scale}\n")
        f.write(f"**Offset:** {offset}\n")
        f.write(f"**Seed:** {seed}\n\n")
        f.write(f"**WandB Run ID:** {wandb.run.id}\n\n")
        
            
    tasks = []

    for idx, i in enumerate(tqdm.tqdm(prompts_data)):
        prompt = i["pos"]
        neg_prompt = i["neg"]
        image_ours = inference(pipe, prompt, neg_prompt, seed=seed, scale=scale, offset=offset)
        image_ours.save(f"benchmark/{run_id}/ours_{idx:03d}.png")

        tasks.append(asyncio.create_task(judge_async(image_ours, prompt, neg_prompt)))

        wandb.log({
            "ours": wandb.Image(image_ours, caption=f"+: {prompt}\n -: {neg_prompt}"),
        })
        with open(f"benchmark/{run_id}/preview.md", "a") as f:
            f.write(f"### {idx:03d}\n")
            f.write(f"**Prompt:** {prompt}\n")
            f.write(f"**Negative Prompt:** {neg_prompt}\n")
            f.write(f"![ours](ours_{idx:03d}.png)\n\n")


    await asyncio.gather(*tasks)
        
async def main():
    for i in range(36):
        run_id = i
        scale = random.uniform(0.0, 2.0)
        offset = random.uniform(0.0, 0.4)
        print(f"Running {run_id} with scale {scale:.2f} and offset {offset:.2f}")
        await run(run_id, scale, offset)

        if COLAB:
            os.system(f"zip -r {drive_path}/{run_id}.zip benchmark/{run_id}")

if __name__ == "__main__":
    asyncio.run(main())
    

for f in tasks:
    f.result()

loop.call_soon_threadsafe(loop.stop)
loop_thread.join()