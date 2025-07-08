# %%
import torch
from diffusers import StableDiffusion3Pipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
# pipe = pipe.to("cuda") 
import sys
sys.path.append("Normalized-Attention-Guidance")

import torch
from processor import JointAttnProcessor2_0
from src.pipeline_sd3_nag import NAGStableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
if "pipe" not in locals():
  pipe = NAGStableDiffusion3Pipeline.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
  )
pipe = pipe.to("cuda")

import random
seed = 97832#int(round(random.random() * 1000000))


import pandas as pd

import wandb
wandb.init(project="VSF")

from ours import inference
import numpy as np
from PIL import Image
import json
from judge import ask_gpt

with open("prompts.json", "r") as f:
    prompts_data = json.load(f)
    
prompts = pd.read_csv("sampled.csv")
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []

# for i in prompts["prompt"]: 

scores = np.zeros((2, 3))
total = 0
for i in prompts_data:
    prompt = i["pos"]
    neg_prompt = i["neg"]
    # neg_prompt = "low quality, blurry, bad lighting, poor detail"
    image_ours = inference(pipe, prompt, neg_prompt, seed=seed, scale=3.25)
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = NAGJointAttnProcessor2_0()
        
    images = []
    image_nag = pipe(
        prompt,
        nag_negative_prompt=neg_prompt,
        generator=torch.manual_seed(seed),
        guidance_scale=0.,
        nag_scale=6,
        num_inference_steps=8,
        nag_alpha=0.25,
        nag_tau=2.5
    ).images[0]
    delta = ask_gpt(image_ours, image_nag, prompt, neg_prompt).T
    scores += delta
    total += 1
    df = pd.DataFrame(scores/total, columns=["positive", "negative", "quality"], index=["ours", "vanilla"])
    print(delta)
    wandb.log({
        "img": wandb.Image(Image.fromarray(np.concatenate([np.array(image_ours), np.array(image_nag)], axis=1)), caption=f"+: {prompt}\n -: {neg_prompt}"),
        "scores": wandb.Table(dataframe=df),
    })   
    
    


