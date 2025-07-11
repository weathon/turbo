import torch
from processor import JointAttnProcessor2_0

def get_score_mod(img_len, pos_len, neg_len):
    q_total_len = img_len + pos_len + neg_len
    kv_total_len = img_len + pos_len + neg_len + neg_len
    def score_mod(score, b, h, q_idx, kv_idx):
        mask1 = torch.logical_and(q_idx >= (q_total_len-neg_len-pos_len), kv_idx >= (kv_total_len-neg_len))
        mask2 = torch.logical_and(q_idx < (q_total_len-neg_len), torch.logical_and(kv_idx>=(kv_total_len-2*neg_len), kv_idx<(kv_total_len-neg_len)))
        mask3 = torch.logical_and(q_idx >= (q_total_len-neg_len), torch.logical_and(kv_idx >= img_len, kv_idx < (img_len + pos_len)))
        mask = torch.logical_or(mask1, torch.logical_or(mask2, mask3))
        return torch.where(mask, -float("inf"), score)
    return score_mod
    
        
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
    # attn_mask = torch.ones((1, 4096 + prompt_embeds.shape[1], 4096 + prompt_embeds.shape[1] + neg_len))
    # attn_mask[:,-neg_len-pos_len:,-neg_len:] = False #prompts cannot see -neg 
    # attn_mask[:,:-neg_len,-2*neg_len:-neg_len] = False # image and positive prompt cannot see neg
    # attn_mask[:,-neg_len:,4096:4096+pos_len] = False # neg cannot see positive prompt
    # attn_mask = attn_mask.cuda()

    for block in pipe.transformer.transformer_blocks: 
        block.attn.processor = JointAttnProcessor2_0(
            attn_mask=get_score_mod(
                img_len=4096, 
                pos_len=pos_len, 
                neg_len=neg_len
            ),
            scale=scale,
            neg_prompt_length=neg_len,
        )

    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
    # pipe.transformer = torch.compile(pipe.transformer)
    image_ours = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pos_pooled_prompt_embeds,
        num_inference_steps=8,
        guidance_scale=0.0,
        # nag_scale=0.0,
        generator=torch.manual_seed(seed),
    ).images[0]
    return image_ours