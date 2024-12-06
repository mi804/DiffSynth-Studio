import torch
from diffsynth import ModelManager, FluxImageLoraPipeline, download_models, FluxImagePipeline
import json
from PIL import Image
import numpy as np
from PIL import ImageDraw, ImageFont

def visualize_masks(image, masks, mask_prompts, output_path):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Generate random colors for each mask
    colors = [
        (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), 80)  # 255 for full opacity
        for _ in masks
    ]
    
    # Font settings
    try:
        font = ImageFont.truetype("arial", 20)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(20)
    
    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)

        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
        draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)

        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    # Save or display the resulting image
    result.save(output_path)

download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "t2i_models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "t2i_models/FLUX/FLUX.1-dev/text_encoder_2",
    "t2i_models/FLUX/FLUX.1-dev/ae.safetensors",
    "t2i_models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
# pipe1 = FluxImageLoraPipeline.from_model_manager(model_manager)

lora_alpha = 0.6
path = 'workdirs/flux-controlnet/lora_checkpoint/newdata_ep143/pytorch_model.bin'

model_manager.load_lora(path, lora_alpha=lora_alpha)

pipe = FluxImageLoraPipeline.from_model_manager(model_manager)

image_idx = 0
image_shape = 1024
guidance = 3.5
sample_dir = 'workdirs/flux-controlnet/samples'
out_dir = f'workdirs/flux-controlnet/lora_output/newdata100k_ep143_cfg_alpha{lora_alpha}/'
import os
os.makedirs(out_dir, exist_ok=True)
seed = 8
prompt = 'A person standing by a river.'
cfg = 3.0
# for seed in range(10):
#     # # seed = 8

#     torch.manual_seed(seed)
#     image = pipe(
#         prompt=prompt,
#         cfg_scale=cfg,
#         negative_prompt="",
#         num_inference_steps=50, embedded_guidance=guidance, height=image_shape, width=image_shape
#     )
#     image.save(f"{out_dir}image_{image_idx}_prompt_{prompt}_origin_seed{seed}_nomask_cfg{cfg}.jpg")

# seed = 7
for seed in range(2, 9):
    for mask_idx in range(6):
        # mask_idx = 3
        # if mask_idx in [2, 5]:
        #     continue
        torch.manual_seed(seed)
        mask_prompt = 'person'
        print(f'prompt: {prompt}\nmask_prompt: {mask_prompt}')
        mask = Image.open(f"{sample_dir}/mask_{image_idx}_{mask_idx}.png").resize((image_shape, image_shape), resample=Image.NEAREST)
        steps = 50
        image = pipe(
            prompt=prompt,
            cfg_scale=cfg,
            negative_prompt="",
            num_inference_steps=steps, embedded_guidance=guidance, height=image_shape, width=image_shape,
            local_prompts=[mask_prompt], masks=[mask]
        )
        visualize_masks(image, [mask], [mask_prompt], f"{out_dir}image_{image_idx}_prompt_{prompt}_mask_{mask_idx}_{mask_prompt}_seed{seed}_step{steps}_cfg{cfg}.png")

# 双mask
# prompt = 'A cat and a white dog standing by the river'
# seed = 4
# # torch.manual_seed(seed)
# # image = pipe(
# #     prompt=prompt,
# #     cfg_scale=2.0,
# #     negative_prompt="",
# #     num_inference_steps=50, embedded_guidance=guidance, height=image_shape, width=image_shape
# # )
# # image.save(f"{out_dir}image_{image_idx}_prompt_{prompt}_origin_seed{seed}_nomask.jpg")

# mask_prompts = ['white dog', 'yellow cat']
# # # 多mask
# masks = []
# masks.append(Image.open(f"{sample_dir}/mask_{0}_{0}.png").resize((image_shape, image_shape), resample=Image.NEAREST))
# # masks.append(Image.open(f"{sample_dir}/mask_{0}_{3}.png").resize((image_shape, image_shape), resample=Image.NEAREST))
# masks.append(Image.open(f"{sample_dir}/mask_{5}_{3}.png").resize((image_shape, image_shape), resample=Image.NEAREST))
# cfg=2.0
# torch.manual_seed(seed)
# image = pipe(
#     prompt=prompt,
#     cfg_scale=cfg,
#     negative_prompt="",
#     num_inference_steps=30, embedded_guidance=guidance, height=image_shape, width=image_shape,
#     local_prompts=mask_prompts, masks=masks
# )
# visualize_masks(image, masks, mask_prompts, f"{out_dir}_prompt_{prompt}_cfg{cfg}_multi_masks_seed{seed}_1.png")

