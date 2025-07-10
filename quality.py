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

with open("prompts.json", "r") as f:
    prompts_data = json.load(f)
    
from src.attention_joint_nag import NAGJointAttnProcessor2_0
score_ours = []
score_nag = []

from datasets import load_dataset
ds = load_dataset("sentence-transformers/coco-captions")
seed = 866737681
import hpsv2

import wandb
import random
wandb.init(project="VSF")
prompts = ds["train"]["caption1"]
random.shuffle(prompts)
for i in prompts:
    prompt = i
    neg_prompt = "low quality"
    image_ours = inference(pipe, prompt, neg_prompt, seed=seed, scale=0.1)
    
    
    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = NAGJointAttnProcessor2_0()
        
    images = []
    image_nag = pipe(
        prompt,
        nag_negative_prompt=neg_prompt,
        generator=torch.manual_seed(seed),
        guidance_scale=0.,
        nag_scale=5,
        num_inference_steps=8,
        nag_alpha=0.25,
        nag_tau=2.5
    ).images[0]
    image_ours.save("ours.png")
    image_nag.save("nag.png")
    result = hpsv2.score("ours.png", prompt, hps_version="v2.1") 
    score_ours.append(result[0])
    result = hpsv2.score("nag.png", prompt, hps_version="v2.1")
    score_nag.append(result[0])
    print(f"Score Ours: {np.mean(score_ours)}, Score NAG: {np.mean(score_nag)}")
    full = Image.fromarray(np.concatenate([np.array(image_ours), np.array(image_nag)], axis=1))
    full.save("full.png")
    wandb.log({
        "img": wandb.Image(full, caption=f"+: {prompt}\n -: {neg_prompt}"),
        "ours_score": np.mean(score_ours),
        "nag_score": np.mean(score_nag),
    })   