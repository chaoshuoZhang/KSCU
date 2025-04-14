from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import random
import string

    
def get_advantaged_concept(erase_concept, mode, n_concept=5, advantage_rate=0.2, advantage_threshold=0.4):


    def generate_description(concept):
        """Generate a description based on mode and concept with multiple templates."""
        concept_descriptions = [
            f"an expression of {concept}",
            f"a portrayal of {concept}",
            f"an illustration of the concept '{concept}'",
            f"{concept} represented in symbolic form",
            f"a thoughtful depiction of the idea of {concept}",
            f"an abstract representation of '{concept}'",
            f"visualization that explores the notion of {concept}",
            f"{concept} brought to life through art",
            f"evocative rendering of the concept '{concept}'",
            f"{concept} expressed in an artistic form"
        ]

        style_descriptions = [
            f"a painting in the style of {concept}",
            f"an artwork reminiscent of {concept}'s style",
            f"a piece inspired by the style of {concept}",
            f"{concept}-inspired painting capturing the essence of their style",
            f"vivid portrayal that echoes the techniques of {concept}",
            f"a canvas that channels the distinct look of {concept}",
            f"an imaginative work in the style of {concept}'s masterpieces",
            f"{concept}'s unique style, recreated in this piece",
            f"artwork reflecting {concept}'s renowned approach",
            f"an iamge of {concept} style"
        ]

        object_descriptions = [
            f"a photo of a {concept}",
            f"an image of a {concept}",
            f"a detailed picture of {concept}",
            f"{concept} captured in natural light",
            f"high-resolution image of a {concept}",
            f"artistic close-up of {concept}",
            f"{concept} in a dynamic composition",
            f"portrait shot of {concept} with sharp details",
            f"scene featuring {concept} with vibrant colors",
            f"snapshot highlighting {concept}'s unique features"
        ]

        if mode == 'concept':
            return random.choice(concept_descriptions)
        elif mode == 'style':
            return random.choice(style_descriptions)
        elif mode == 'object':
            return random.choice(object_descriptions)
        else:
            return f"a depiction of {concept}"

    def apply_augmentations(description, concept, threshold):
        """Apply random augmentations to the description."""
        augmented_descriptions = []

        words = description.split()
        if random.random() < threshold:
            random.shuffle(words)
            augmented_descriptions.append(" ".join(words))

        if random.random() < threshold:
            # Randomly choose one unimportant concept to remove
            word_to_remove = random.choice([word for word in words if word.lower() not in [concept.lower()]])

            # Remove the selected word
            words = [word for word in words if word != word_to_remove]

            # Add the modified description to the augmented list
            augmented_descriptions.append(" ".join(words))

        if random.random() < threshold:
            th = int(threshold * 10)
            rand_number = random.randint(1, th+1)
            random_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=rand_number))
            augmented_descriptions.append(random_prefix + " " + description)

        if random.random() < threshold:
            th = int(threshold * 10)
            rand_number = random.randint(1, th+1)
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=rand_number))
            augmented_descriptions.append(description + " " + random_suffix)

        return augmented_descriptions or [description]

    output_list = []
    target_length = n_concept * len(erase_concept)
    num_augmented = int(target_length * advantage_rate)
    num_original = target_length - num_augmented

    # Add original concepts
    for concept in erase_concept:
        # base_description = generate_description(concept)
        output_list.extend([concept] * (num_original // len(erase_concept)))

    # Add augmented descriptions
    while len(output_list) < target_length:
        concept = random.choice(erase_concept)
        base_description = generate_description(concept)
        augmented_descriptions = apply_augmentations(base_description, concept, advantage_threshold)
        output_list.append(random.choice(augmented_descriptions))

    # Ensure output_list matches the target length
    output_list = output_list[:target_length]
    random.shuffle(output_list)
    return output_list

def to_gif(images, path):
    images[0].save(path, save_all=True,
                   append_images=images[1:], loop=0, duration=len(images) * 20)


def figure_to_image(figure):
    figure.set_dpi(300)

    figure.canvas.draw()

    return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())


def image_grid(images, outpath=None, column_titles=None, row_titles=None):
    n_rows = len(images)
    n_cols = len(images[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols, n_rows), squeeze=False)

    for row, _images in enumerate(images):

        for column, image in enumerate(_images):
            ax = axs[row][column]
            ax.imshow(image)
            if column_titles and row == 0:
                ax.set_title(textwrap.fill(
                    column_titles[column], width=12), fontsize='x-small')
            if row_titles and column == 0:
                ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        image = figure_to_image(plt.gcf())
        plt.close()
        return image


def get_module(module, module_name):
    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 0:
        return module
    else:
        module = getattr(module, module_name[0])
        return get_module(module, module_name[1:])


def set_module(module, module_name, new_module):
    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = True


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


class StableDiffuser(torch.nn.Module):

    def __init__(self, scheduler='LMS', version='1-4'):
        super().__init__()

        self.version = version
        
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")
            
        if self.version == '1-5':
            model_path=f"sd-legacy/stable-diffusion-v1-5"
        elif self.version == "2-1":
            model_path = f"stabilityai/stable-diffusion-2-1"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        elif self.version == "UnlearnCanvas":
            model_path = f"yourpath"
        else:
            model_path = f"CompVis/stable-diffusion-v{self.version}"

        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae")
        print('vae is loaded')
        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet")
        print('unet is loaded')
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model_path,
                                                                      subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path,
        #                                                                    subfolder="safety_checker")

        # Select the scheduler based on input
        
        if scheduler == 'LMS':
            self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                  num_train_timesteps=1000)
        elif scheduler == 'DDIM':
            self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        elif scheduler == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        self.eval()


    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(self,
                      iteration,
                      latents,
                      text_embeddings,
                      guidance_scale=7.5
                      ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration])

        # predict the noise residual
        noise_prediction = self.unet(
            latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
                           (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=False,
                  pred_x0=False,
                  trace_args=None,
                  show_progress=True,
                  **kwargs):

        latents_steps = []
        trace_steps = []

        trace = None

        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:
                trace = TraceDict(self, **trace_args)

            noise_pred = self.predict_noise(
                iteration,
                latents,
                text_embeddings,
                **kwargs)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[iteration], latents)

            if trace_args:
                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps

    @torch.no_grad()
    def __call__(self,
                 prompts,
                 img_size=512,
                 n_steps=50,
                 n_imgs=1,
                 end_iteration=None,
                 generator=None,
                 start_iteration=0,
                 **kwargs
                 ):

        assert 0 <= n_steps <= 1000

        if not isinstance(prompts, list):
            prompts = [prompts]

        self.set_scheduler_timesteps(n_steps)

        latents = self.get_initial_latents(n_imgs, img_size, len(prompts), generator=generator)

        text_embeddings = self.get_text_embeddings(prompts, n_imgs=n_imgs)

        end_iteration = end_iteration or n_steps
        if start_iteration==0:
            latents_steps, trace_steps = self.diffusion(
                latents,
                text_embeddings,
                end_iteration=end_iteration,
                **kwargs
            )
        else:
            latents_steps, trace_steps = self.diffusion(
                latents,
                text_embeddings,
                end_iteration=end_iteration,
                start_iteration=start_iteration,
                **kwargs
            )

        latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
        images_steps = [self.to_image(latents) for latents in latents_steps]

        # for i in range(len(images_steps)):
            # self.safety_checker = self.safety_checker.float()
            # safety_checker_input = self.feature_extractor(images_steps[i], return_tensors="pt").to(
            #     latents_steps[0].device)
            # image, has_nsfw_concept = self.safety_checker(
            #     images=latents_steps[i].float().cpu().numpy(), clip_input=safety_checker_input.pixel_values.float()
            # )

            # images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]

        images_steps = list(zip(*images_steps))

        if trace_steps:
            return images_steps, trace_steps

        return images_steps


class FineTunedModel(torch.nn.Module):

    def __init__(self,
                 model,
                 train_method,
                 ):

        super().__init__()

        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}

        freeze(self.model)

        for module_name, module in model.named_modules():
            if 'unet' not in module_name:
                continue
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'xattn':
                    if 'attn2' not in module_name:
                        continue
                elif train_method == 'xattn-strict':
                    if 'attn2' not in module_name or 'to_q' not in module_name or 'to_k' not in module_name:
                        continue
                elif train_method == 'noxattn':
                    if 'attn2' in module_name:
                        continue
                elif train_method == 'selfattn':
                    if 'attn1' not in module_name:
                        continue
                else:
                    raise NotImplementedError(
                        f"train_method: {train_method} is not implemented."
                    )
                print(module_name)
                ft_module = copy.deepcopy(module)

                self.orig_modules[module_name] = module
                self.ft_modules[module_name] = ft_module

                unfreeze(ft_module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())

    @classmethod
    def from_checkpoint(cls, model, checkpoint, train_method):

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        modules = [f"{key}$" for key in list(checkpoint.keys())]

        ftm = FineTunedModel(model, train_method=train_method)
        ftm.load_state_dict(checkpoint)

        return ftm

    def __enter__(self):

        for key, ft_module in self.ft_modules.items():
            set_module(self.model, key, ft_module)

    def __exit__(self, exc_type, exc_value, tb):

        for key, module in self.orig_modules.items():
            set_module(self.model, key, module)

    def parameters(self):

        parameters = []

        for ft_module in self.ft_modules.values():
            parameters.extend(list(ft_module.parameters()))

        return parameters

    def state_dict(self):

        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}

        return state_dict

    def load_state_dict(self, state_dict):

        for key, sd in state_dict.items():
            self.ft_modules[key].load_state_dict(sd)
    
    # def load_other(self,ckpt):
    #     state_dict=ckpt["state_dict"]
    #     for key, sd in state_dict.items():
    #         print(key)
    #         input()
