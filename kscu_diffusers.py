import argparse
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from utils.utils import *


def generate_iteration_list(start, end, length, full_loops_before_shift=5):
    result = []
    current_start = start
    loop_count = 0

    while len(result) < length:
        result.extend(range(current_start, end))
        loop_count += 1

        if loop_count >= full_loops_before_shift:
            current_start = min(current_start + 1, end - 1)
            loop_count = 0

    return result[:length]
    

def train(args):
    nsteps = 50
    lambda_1, lambda_2, lambda_3 = 2e-5, 1, 1e-4
    start, end, length, full_loops_before_shift=args.start,nsteps,args.iterations,8
    diffuser = StableDiffuser(scheduler='DDIM', version=args.version).to(args.device)
    diffuser.train()
    finetuner = FineTunedModel(diffuser, train_method=args.train_method)
    optimizer = torch.optim.Adam(finetuner.parameters(), lr=args.lr)
    criteria = torch.nn.MSELoss()
    
    erase_concept = [a.strip() for a in args.erase_concept.split(',')]
    advantaged_concept = get_advantaged_concept(erase_concept, args.mode, n_concept=200 // len(erase_concept),
                                                advantage_rate=args.advantage_rate,
                                                advantage_threshold=args.advantage_threshold)
    # timetable = generate_timetable(14,nsteps-1,args.iterations)
    timetable = generate_iteration_list(start,end,length, full_loops_before_shift)
    pbar = tqdm(range(args.iterations))

    for i in pbar:
        with torch.no_grad():
            index = np.random.choice(len(advantaged_concept), 1, replace=False)[0]
            erase_concept_sampled_list = advantaged_concept
            neutral_text_embeddings = diffuser.get_text_embeddings([''], n_imgs=1)
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled_list[0]], n_imgs=1)
            advantage_text_embeddings = diffuser.get_text_embeddings(
                [erase_concept_sampled_list[np.random.choice(len(erase_concept_sampled_list), 1, replace=False)[0]]],
                n_imgs=1) if args.use_augmentation and i > 0.95*args.iterations else diffuser.get_text_embeddings([erase_concept_sampled_list[0]], n_imgs=1)
            if args.use_augmentation and i > 0.95*args.iterations: 
                advantage_text_embeddings = advantage_text_embeddings + args.noise_scale * torch.randn_like(advantage_text_embeddings)
            
            diffuser.set_scheduler_timesteps(nsteps)
            optimizer.zero_grad()
            

            iteration = timetable[i]
            latents = diffuser.get_initial_latents(args.batch_size, 512, 1)
            
            
            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents, positive_text_embeddings, start_iteration=0, end_iteration=iteration,
                    guidance_scale=3, show_progress=False
                )
            
            # diffuser.set_scheduler_timesteps(1000)
            # iteration = int(iteration / nsteps * 1000)
            
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
        
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], advantage_text_embeddings, guidance_scale=1)
            negative_neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            
        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        lambda_3 = iteration / 10000
        loss = criteria(negative_latents,  neutral_latents - (lambda_2*(positive_latents-neutral_latents))) + lambda_1 * criteria(negative_neutral_latents,  neutral_latents - (lambda_3 * (positive_latents-neutral_latents)))
        loss.backward()
        optimizer.step()

    torch.save(finetuner.state_dict(), args.save_path)
    del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train', description='Finetuning stable diffusion to erase the concepts')
    parser.add_argument('--erase_concept', type=str, required=True)
    parser.add_argument('--erase_from', type=str, default=None)
    parser.add_argument('--train_method', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=750)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--noise_scale', type=float, default=1e-5)
    parser.add_argument('--save_path', type=str, default='models/path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_augmentation', choices=['true', 'false'], default='false')

    parser.add_argument('--mode', type=str, required=True)# concept task
    parser.add_argument('--n_concept', type=int, default=5)
    parser.add_argument('--advantage_rate', type=float, default=0.2)
    parser.add_argument('--advantage_threshold', type=float, default=0.2)
    
    parser.add_argument('--start', type=int, default=25)
    parser.add_argument('--version', , choices=['2-1', '1-4','1-5','UnlearnCanvas'], default='1-4')# stable diffusion version
    
    
     
    args = parser.parse_args()
    
    if args.mode == 'class' or args.mode == 'NSFW' :
        args.iterations=750
        args.start = 15
    elif args.mode == 'instance':
        args.iterations=200
        args.start = 40
    else:
        args.iterations=500
        args.start = 24
    
    args.use_augmentation = args.use_augmentation == 'true' 
    if args.erase_from is None:
        args.erase_from = args.erase_concept
        
    name = (f"kscu-{args.erase_concept.lower().replace(' ', '').replace(',', '')}-{args.train_method}" +
            (f"_use_augmentation_{args.advantage_rate}_{args.advantage_threshold}" if args.use_augmentation else f"_{args.advantage_rate}_{args.advantage_threshold}") +
            f"-epochs_{args.iterations}_{args.version}_0")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    i=1
    while os.path.exists(f"{args.save_path}/{name}.pt"):
        name=name[::-1].replace(name[-1], str(i), 1)[::-1]
        i=i+1
        print(f"{args.save_path}/{name}.pt")
        
        
    args.save_path = f"{args.save_path}/{name}.pt"
    print(args.erase_concept)
    print(args.save_path)
    train(args)
    print('save:',args.save_path)
    # python cce_diffusers.py  --version 1-5  --mode style --train_method xattn  --use_augmentation true --device cuda:0 --erase_concept "Dogs"
