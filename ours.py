import torch
from processor import JointAttnProcessor2_0
from torch.nn.attention.flex_attention import create_block_mask
import numpy as np


def inference(pipe, prompt, neg_prompt, seed=0, scale=3):
    (
        pos_prompt_embeds,
        _,
        pos_pooled_prompt_embeds,
        _,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        padding=False
    )
    (
        neg_prompt_embeds,
        _,
        neg_pooled_prompt_embeds,
        _,
    ) = pipe.encode_prompt(
        prompt=neg_prompt,
        prompt_2=neg_prompt,
        prompt_3=neg_prompt,
        padding=False
    )
    neg_len = neg_prompt_embeds.shape[1]
    pos_len = pos_prompt_embeds.shape[1]
    
    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
    attn_mask = np.ones((1, 4096 + prompt_embeds.shape[1], prompt_embeds.shape[1] + neg_len))
    attn_mask[:,-neg_len-pos_prompt_embeds.shape[1]:,-neg_len:] = False
    attn_mask[:,:-neg_len,-2*neg_len:-neg_len] = False
    attn_mask[:,-neg_len:,:pos_prompt_embeds.shape[1]] = False
    def mask_mod(b, h, q_idx, kv_idx):
        if kv_idx < 4096:
            return True
        return attn_mask[0, q_idx, kv_idx-4096].bool()
        
    
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=4096+pos_len+neg_len, KV_LEN=4096+pos_len+neg_len+neg_len)
    for block in pipe.transformer.transformer_blocks: 
        block.attn.processor = JointAttnProcessor2_0(scale=scale, neg_prompt_length=neg_len, score_mod=None, block_mask=block_mask)

    # pipe.transformer = torch.compile(pipe.transformer)
    image_ours = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pos_pooled_prompt_embeds,
        num_inference_steps=8, 
        guidance_scale=0.0,
        nag_scale=0.0,
        generator=torch.manual_seed(seed),
    ).images[0]
    return image_ours