from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from utils.utils import *
import torch


def generate_images(args):
    df = pd.read_csv(args.prompts_path)
    name=args.model_name.split("/")[-1]
    folder_path = f'{args.save_path}/{name}'

    os.makedirs(folder_path, exist_ok=True)
    case_number = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="P"):
        prompt = [str(row.prompt)] * args.num_samples
        seed = row.evaluation_seed
        case_number += 1


        num_inference_steps = args.ddim_steps

        guidance_scale = args.guidance_scale

        generator = torch.manual_seed(seed)

        batch_size = len(prompt)
        if args.finetuner:
            with finetuner:
                pil_images = diffuser(prompt,
                                  img_size=args.image_size,
                                  n_steps=num_inference_steps,
                                  n_imgs=1,
                                  generator=generator,
                                  guidance_scale=guidance_scale,
                                  show_progress = False
                                  )[0]
        else:
            pil_images = diffuser(prompt,
                                  img_size=args.image_size,
                                  n_steps=num_inference_steps,
                                  n_imgs=1,
                                  generator=generator,
                                  guidance_scale=guidance_scale,
                                  show_progress = False
                                  )[0]
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--train_method', help='name of mode', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=50)
                        
    parser.add_argument('--finetuner', choices=['true', 'false'], required=True)
    args = parser.parse_args()
    args.finetuner = args.finetuner == 'true'
    print(args.finetuner)
    diffuser = StableDiffuser(scheduler='DDIM',version='1-4').to(args.device)

    finetuner = FineTunedModel(diffuser, train_method=args.train_method)
    esd_path = f'{args.model_name}'
    finetuner.load_state_dict(torch.load(esd_path))

    generate_images(args)
