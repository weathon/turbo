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
import threading



from ours import inference
import numpy as np
from PIL import Image
import json
from judge import ask_gpt

scores = np.zeros((2, 2))
total = 0
lock = threading.Lock()


with open("prompts2.json", "r") as f:
    prompts_data = json.load(f)

prompts = pd.read_csv("sampled.csv")
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []
import os
import time

# for i in prompts["prompt"]:

futures = []

def run(scale, offset, seed):
    import wandb
    wandb.init(project="VSF", config={"scale": scale, "offset": offset, "seed": seed}, reinit="create_new", name=f"scale_{scale}_offset_{offset}_seed_{seed}")
    os.system("mkdir -p res/" + wandb.run.id)
    run_id = wandb.run.id
    futures = []
    with open("res/" + run_id + "/preview.md", "a") as f:
        f.write(f"# {run_id}\n scale: {scale}, offset: {offset}, seed: {seed}\n")
        
    for idx, i in enumerate(prompts_data):
        prompt = i["pos"]
        neg_prompt = i["neg"]
        image_ours = inference(pipe, prompt, neg_prompt, seed=seed, scale=scale, offset=offset)

        image_ours.save(f"res/{run_id}/{idx}.png")
        with open("res/" + run_id + "/preview.md", "a") as f:
            f.write(f"![{idx}]({idx}.png)\n") 
        
        wandb.log({"image_ours": wandb.Image(image_ours, caption=f"+: {prompt}\n -:{neg_prompt}"), 
                   "scale": scale, "offset": offset, "seed": seed}, step=idx)

for i in range(36):
    seed = 42
    scale = random.uniform(0.0, 2.0)
    offset = random.uniform(0.0, 0.2)
    run(seed=seed, scale=scale, offset=offset)