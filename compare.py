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


from ours import inference
import numpy as np
from PIL import Image
import json

with open("prompts2.json", "r") as f:
    prompts_data = json.load(f)
    
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []

from datasets import load_dataset
ds = load_dataset("sentence-transformers/coco-captions")
seed = 866737681
import hpsv2

import ImageReward
reward_model = ImageReward.load("ImageReward-v1.0")

import wandb
import random
from urllib.request import urlretrieve
import pandas as pd

table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

metadata_df = pd.read_parquet('metadata.parquet')


run_baseline = False
wandb.init(project="compare")


prompts = metadata_df["prompt"].sample(100)
random.seed(10)
with open("prompts2.json", "r") as f:
    prompts_data = json.load(f)

random.shuffle(prompts_data)
for i in prompts_data:
    image_ours = inference(pipe, i["pos"], i["neg"], seed=seed, scale=3, offset=0.1)
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = NAGJointAttnProcessor2_0()
    image_nag = pipe(
        i["pos"],
        nag_negative_prompt=i["neg"],
        generator=torch.manual_seed(seed),
        guidance_scale=0.,
        nag_scale=0,#5,
        num_inference_steps=8,
        nag_alpha=0.25,
        nag_tau=2.5
    ).images[0]

    image_ours.save("ours.png")
    image_nag.save("nag.png")
    
    full = Image.fromarray(np.concatenate([np.array(image_ours), np.array(image_nag)], axis=1))
    full.save("full.png")
    wandb.log({
        "ours": wandb.Image(image_ours, caption=f"+: {i['pos']}\n -:{i['neg']}"),
        "nag": wandb.Image(image_nag, caption=f"+: {i['pos']}\n -:{i['neg']}"),
    })   
    