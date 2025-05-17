import os
import torch
import argparse

import numpy as np
import torch.nn.functional as F
import utils.ptp_utils as ptp_utils

from PIL import Image 
from datetime import datetime
from diffusers import LCMScheduler
from torchvision.utils import save_image
from utils.attn_control import AttentionStyle
from models.unet_2d_condition import UNet2DConditionModel
from pipelines.unisty_pipeline import UniStyLatentConsistencyModelPipeline

def load_image(image_path, res, device, gray=False):
    image = Image.open(image_path).convert('RGB') if not gray else Image.open(image_path).convert('L')
    image = torch.tensor(np.array(image)).float()
    if gray:
        image = image.unsqueeze(-1).repeat(1,1,3)
    image = image.permute(2, 0, 1)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (res, res))
    image = image.to(device)
    return image

def main():
    args = argparse.ArgumentParser()

    args.add_argument("--start_ac_layer", type=int, default=0)
    args.add_argument("--end_ac_layer", type=int, default=16)
    args.add_argument("--height", type=int, default=512)
    args.add_argument("--width", type=int, default=512)
    args.add_argument("--cfg_guidance", type=float, default=2)
    args.add_argument("--sty_guidance", type=float, default=1.2)
    args.add_argument("--prompt", type=str, default='face')
    args.add_argument("--neg_prompt", type=str, default='')
    args.add_argument("--output", type=str, default='./results/')
    args.add_argument("--style", type=str, default=None)
    # args.add_argument("--model_path", type=str, default='/mnt/hwfile/gcc/maxin/work/pretrained/LCM_Dreamshaper_v7')
    args.add_argument("--model_path", type=str, default='SimianLuo/LCM_Dreamshaper_v7')
    args.add_argument("--num_inference_steps", type=int, default=4)
    args.add_argument("--fix_step_index", type=int, default=99)
    args.add_argument("--seed", type=int, default=None)

    args = args.parse_args()

    if args.seed is not None:
        torch.manual_seed(10)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch_dtype = torch.float16

    cfg_scale = args.cfg_guidance
    unet = UNet2DConditionModel.from_pretrained(args.model_path, 
                                                subfolder='unet', 
                                                torch_dtype=torch_dtype).to(device)
    model = UniStyLatentConsistencyModelPipeline.from_pretrained(args.model_path,
                                                                 unet=unet,
                                                                 torch_dtype=torch_dtype).to(device)
    model.scheduler = LCMScheduler.from_config(model.scheduler.config)

    content_prompts = ['apples', 'oranges', 'forest', 'lion', 'bird']
    for style_dir in os.listdir(args.style):
        style_name = os.path.splitext(os.path.basename(style_dir))[0]
        os.makedirs(os.path.join(args.output, style_name), exist_ok=True)
        os.makedirs(os.path.join(args.output,style_name, 'results_only'), exist_ok=True)
        time_begin = datetime.now()
        style_image = Image.open(os.path.join(args.style, style_dir)).convert('RGB')
        style = style_image.resize((args.height, args.width))
        print(f"Start processing style {style_name} at {time_begin}")
        for prompt in content_prompts:
            prompt_ = prompt.replace(' ', '_')
            save_name = os.path.join(args.output, style_name, f"{prompt_}.jpg")
            controller = AttentionStyle(args.num_inference_steps, 
                                         args.start_ac_layer,
                                         args.end_ac_layer,
                                         sty_guidance=args.sty_guidance,
                                         )

            ptp_utils.register_attention_control(model, controller)

            with torch.no_grad():
                generate_image = model(
                            prompt=['', prompt],
                            negative_prompt=['', ''],
                            height=args.height,
                            width=args.width,
                            style=style,
                            num_inference_steps=args.num_inference_steps,
                            eta=0.0,
                            guidance_scale=cfg_scale,
                            strength=1.0,
                            save_intermediate=False,
                            fix_step_index=args.fix_step_index,
                            callback = None
                ).images
                
            os.makedirs(args.output, exist_ok=True)
            generate_image = torch.from_numpy(generate_image).permute(0, 3, 1, 2)
            save_image(generate_image, save_name, nrow=3, padding=0)
            save_image(generate_image[-1:], os.path.join(args.output, style_name, 'results_only',f"{prompt_}.png"), nrow=1, padding=0)
                    
        time_end = datetime.now()
        print(f"Finish processing style {style_name} at {time_end} \nTime cost: {time_end-time_begin}, \nPer image cost: {(time_end-time_begin)/len(content_prompts)}")
        

if __name__ == "__main__":
    main()