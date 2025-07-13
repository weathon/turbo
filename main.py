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
# pipe.set_progress_bar_config(disable=True)
import random


import pandas as pd

import asyncio
import threading

import wandb

from ours import inference
import numpy as np
from PIL import Image
import json
from judge import ask_gpt

scores = np.zeros((2, 1))
total = 0
lock = threading.Lock()

loop = asyncio.new_event_loop()
def _run_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop_thread = threading.Thread(target=_run_loop, args=(loop,), daemon=True)
loop_thread.start()

async def judge_async(image_ours, prompt, neg_prompt):
    global scores, total, wandb
    delta = await asyncio.to_thread(ask_gpt, image_ours, prompt, neg_prompt)
    delta = delta.T
    with lock:
        scores += delta
        total += 1
        df = pd.DataFrame(
            scores / total,
            columns=["positive", "negative"],
        )
    print("ID: ", total)
    print("delta:\n", delta)
    print(df)
    wandb.log({
        "step": total,
        "score_board": wandb.Table(
            data=df,
        ),
    })
    

with open("prompts2.json", "r") as f:
    prompts_data = json.load(f)

prompts = pd.read_csv("sampled.csv")
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []
import os
import time

# for i in prompts["prompt"]:


seed = 42
def run(scale, offset):
    wandb.init(project="VSF", config={"scale": scale, "offset": offset})
    os.system("mkdir -p benchmark/" + wandb.run.id)
    run_id = wandb.run.id
    futures = []
    for idx, i in enumerate(prompts_data):
        prompt = i["pos"]
        neg_prompt = i["neg"]
        torch.cuda.reset_peak_memory_stats()
        image_ours = inference(pipe, prompt, neg_prompt, seed=seed, scale=scale, offset=offset)
        
        futures.append(
            asyncio.run_coroutine_threadsafe(
                judge_async(image_ours, prompt, neg_prompt),
                loop,
            )
        )
    
        frame = image_ours
    
        frame.save(f"benchmark/{run_id}/{idx}.png")
        with open("benchmark/" + run_id + "/preview.md", "a") as f:
            f.write(f"![{idx}]({idx}.png)\n")
        
    for f in futures:
        f.result()

    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join()
    

for i in range(36):
    scale = random.uniform(0.0, 2.0)
    offset = random.uniform(0.0, 0.3)
    run(scale, offset)
