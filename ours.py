import torch
from processor import JointAttnProcessor2_0
def inference(pipe, prompt, neg_prompt, seed=0, scale=3):
    # (
    #     pos_prompt_embeds,
    #     _,
    #     pos_pooled_prompt_embeds,
    #     _,
    # ) = pipe.encode_prompt(
    #     prompt=prompt,
    #     prompt_2=prompt,
    #     prompt_3=prompt,
    #     max_sequence_length = 77
    # )

    # (
    #     neg_prompt_embeds,
    #     _,
    #     neg_pooled_prompt_embeds,
    #     _,
    # ) = pipe.encode_prompt(
    #     prompt=neg_prompt,
    #     prompt_2=neg_prompt,
    #     prompt_3=neg_prompt,
    #     max_sequence_length = 77
    # )

    # negative_prompt_length = [len(pipe.tokenizer(neg_prompt).input_ids), len(pipe.tokenizer_3(neg_prompt).input_ids)]
    # attn_mask = torch.ones((1, 4404, 4404)).bool()
    # # attn_mask[:,-154*2:,-154*2:] = False #text cannot see each other 
    # # attn_mask[:,-154*2:-154,-154*2:-154] = True # positive prompt can see each other   
    # attn_mask[:,-154:,154:] = False  #missing negative sign
    # # attn_mask[:,-154*2:,-154*2:] = False
    # # attn_mask[:,-154*2:-154,-154*2:-154] = True
    
    # attn_mask[:,-154+negative_prompt_length[0]:-77:,:] = False
    # attn_mask[:,:,-154+negative_prompt_length[0]:-77] = False
    # attn_mask[:,-77+negative_prompt_length[1]:,:] = False
    # attn_mask[:,:,-77+negative_prompt_length[1]:] = False
    
    # # attn_mask[:,-154*2+negative_prompt_length[0]:-154*2+77:,:] = False
    # # attn_mask[:,:,-154*2+negative_prompt_length[0]:-154*2] = False
    # # attn_mask[:,-154-77+negative_prompt_length[1]:-154,:] = False
    # # attn_mask[:,:,-154-77+negative_prompt_length[1]:-154] = False
    # attn_mask = attn_mask.cuda()

    # for block in pipe.transformer.transformer_blocks:
    #     block.attn.processor = JointAttnProcessor2_0(scale=scale, attn_mask=attn_mask, neg_prompt_length=negative_prompt_length)

    # prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
    # image_ours = pipe(
    #     prompt_embeds=prompt_embeds,
    #     pooled_prompt_embeds=pos_pooled_prompt_embeds,
    #     num_inference_steps=8,
    #     guidance_scale=0.0,
    #     nag_scale=0.0,
    #     generator=torch.manual_seed(seed),
    # ).images[0]  

    # return image_ours
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
    attn_mask = torch.ones((1, 4096 + prompt_embeds.shape[1], 4096 + prompt_embeds.shape[1] + neg_len))
    attn_mask[:,-neg_len-pos_len:,-neg_len:] = False #prompts cannot see -neg 
    attn_mask[:,:-neg_len,-2*neg_len:-neg_len] = False # image and positive prompt cannot see neg
    attn_mask[:,-neg_len:,4096:4096+pos_len] = False # neg cannot see positive prompt
    attn_mask = attn_mask.cuda()

    images = []

    for block in pipe.transformer.transformer_blocks:
        block.attn.processor = JointAttnProcessor2_0(scale=scale, attn_mask=attn_mask, neg_prompt_length=neg_len)

    prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], dim=1)
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