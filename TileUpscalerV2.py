import os
import time

import torch

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL

from PIL import Image
import cv2
import numpy as np

from RealESRGAN import RealESRGAN

import random
import math

import gradio as gr
from gradio_imageslider import ImageSlider

USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timer_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_scheduler(scheduler_name, config):
    if scheduler_name == "DDIM":
        return DDIMScheduler.from_config(config)
    elif scheduler_name == "DPM++ 3M SDE Karras":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
    elif scheduler_name == "DPM++ 3M Karras":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

class LazyLoadPipeline:
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        if self.pipe is None:
            print("Starting to load the pipeline...")
            self.pipe = self.setup_pipeline()
            print(f"Moving pipeline to device: {device}")
            self.pipe.to(device)
            if USE_TORCH_COMPILE:
                print("Compiling the model...")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

    @timer_func
    def setup_pipeline(self):
        print("Setting up the pipeline...")
        controlnet = ControlNetModel.from_single_file(
            "models/ControlNet/control_v11f1e_sd15_tile.pth", torch_dtype=torch.float16
        )
        model_path = "models/models/Stable-diffusion/juggernaut_reborn.safetensors"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None
        )
        vae = AutoencoderKL.from_single_file(
            "models/VAE/vae-ft-mse-840000-ema-pruned.safetensors",
            torch_dtype=torch.float16
        )
        pipe.vae = vae
        pipe.load_textual_inversion("models/embeddings/verybadimagenegative_v1.3.pt")
        pipe.load_textual_inversion("models/embeddings/JuggernautNegative-neg.pt")
        pipe.load_lora_weights("models/Lora/SDXLrender_v2.0.safetensors")
        pipe.fuse_lora(lora_scale=0.5)
        pipe.load_lora_weights("models/Lora/more_details.safetensors")
        pipe.fuse_lora(lora_scale=1.)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        return pipe

    def set_scheduler(self, scheduler_name):
        if self.pipe is not None:
            self.pipe.scheduler = get_scheduler(scheduler_name, self.pipe.scheduler.config)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)

class LazyRealESRGAN:
    def __init__(self, device, scale):
        self.device = device
        self.scale = scale
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = RealESRGAN(self.device, scale=self.scale)
            self.model.load_weights(f'models/upscalers/RealESRGAN_x{self.scale}.pth', download=False)
    def predict(self, img):
        self.load_model()
        return self.model.predict(img)

lazy_realesrgan_x2 = LazyRealESRGAN(device, scale=2)
lazy_realesrgan_x4 = LazyRealESRGAN(device, scale=4)

@timer_func
def resize_and_upscale(input_image, resolution):
    scale = 2 if resolution <= 2048 else 4
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(round(H * k / 64.0)) * 64
    W = int(round(W * k / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    if scale == 2:
        img = lazy_realesrgan_x2.predict(img)
    else:
        img = lazy_realesrgan_x4.predict(img)
    return img

@timer_func
def create_hdr_effect(original_image, hdr):
    if hdr == 0:
        return original_image
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    factors = [1.0 - 0.9 * hdr, 1.0 - 0.7 * hdr, 1.0 - 0.45 * hdr,
              1.0 - 0.25 * hdr, 1.0, 1.0 + 0.2 * hdr,
              1.0 + 0.4 * hdr, 1.0 + 0.6 * hdr, 1.0 + 0.8 * hdr]
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

lazy_pipe = LazyLoadPipeline()
lazy_pipe.load()

@timer_func
def progressive_upscale(input_image, target_resolution, steps=3):
    current_image = input_image.convert("RGB")
    current_size = max(current_image.size)
    
    for _ in range(steps):
        if current_size >= target_resolution:
            break
        
        scale_factor = min(2, target_resolution / current_size)
        new_size = (int(current_image.width * scale_factor), int(current_image.height * scale_factor))
        
        if scale_factor <= 1.5:
            current_image = current_image.resize(new_size, Image.LANCZOS)
        else:
            current_image = lazy_realesrgan_x2.predict(current_image)
        
        current_size = max(current_image.size)
    
    # Final resize to exact target resolution
    if current_size != target_resolution:
        aspect_ratio = current_image.width / current_image.height
        if current_image.width > current_image.height:
            new_size = (target_resolution, int(target_resolution / aspect_ratio))
        else:
            new_size = (int(target_resolution * aspect_ratio), target_resolution)
        current_image = current_image.resize(new_size, Image.LANCZOS)
    
    return current_image

def prepare_image(input_image, resolution, hdr):
    upscaled_image = progressive_upscale(input_image, resolution)
    return create_hdr_effect(upscaled_image, hdr)

def create_gaussian_weight(tile_size, sigma=0.3):
    x = np.linspace(-1, 1, tile_size)
    y = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, y)
    gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

def adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1024):
    w, h = image_size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        tile_w = min(w, max_tile_size)
        tile_h = min(int(tile_w / aspect_ratio), max_tile_size)
    else:
        tile_h = min(h, max_tile_size)
        tile_w = min(int(tile_h * aspect_ratio), max_tile_size)
    return max(tile_w, base_tile_size), max(tile_h, base_tile_size)

def process_tile(tile, num_inference_steps, strength, guidance_scale, controlnet_strength):
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": tile,
        "control_image": tile,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": float(controlnet_strength),
        "generator": torch.Generator(device=device).manual_seed(random.randint(0, 2147483647)),
    }
    
    return np.array(lazy_pipe(**options).images[0])


@timer_func
def gradio_process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name):
    print("Starting image processing...")
    torch.cuda.empty_cache()
    lazy_pipe.set_scheduler(scheduler_name)
    
    # Convert input_image to numpy array
    input_array = np.array(input_image)
    
    # Prepare the condition image
    condition_image = prepare_image(input_image, resolution, hdr)
    W, H = condition_image.size
    
    # Adaptive tiling
    tile_width, tile_height = adaptive_tile_size((W, H))
    
    # Calculate the number of tiles
    overlap = min(64, tile_width // 8, tile_height // 8)  # Adaptive overlap
    num_tiles_x = math.ceil((W - overlap) / (tile_width - overlap))
    num_tiles_y = math.ceil((H - overlap) / (tile_height - overlap))
    
    # Create a blank canvas for the result
    result = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)
    
    # Create gaussian weight
    gaussian_weight = create_gaussian_weight(max(tile_width, tile_height))
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Calculate tile coordinates
            left = j * (tile_width - overlap)
            top = i * (tile_height - overlap)
            right = min(left + tile_width, W)
            bottom = min(top + tile_height, H)
            
            # Adjust tile size if it's at the edge
            current_tile_size = (bottom - top, right - left)
            
            tile = condition_image.crop((left, top, right, bottom))
            tile = tile.resize((tile_width, tile_height))
            
            # Process the tile
            result_tile = process_tile(tile, num_inference_steps, strength, guidance_scale, controlnet_strength)
            
            # Apply gaussian weighting
            if current_tile_size != (tile_width, tile_height):
                result_tile = cv2.resize(result_tile, current_tile_size[::-1])
                tile_weight = cv2.resize(gaussian_weight, current_tile_size[::-1])
            else:
                tile_weight = gaussian_weight[:current_tile_size[0], :current_tile_size[1]]
            
            # Add the tile to the result with gaussian weighting
            result[top:bottom, left:right] += result_tile * tile_weight[:, :, np.newaxis]
            weight_sum[top:bottom, left:right] += tile_weight[:, :, np.newaxis]
    
    # Normalize the result
    final_result = (result / weight_sum).astype(np.uint8)
    
    print("Image processing completed successfully")
    
    return [input_array, final_result]

title = """<h1 align="center">Tile Upscaler V2</h1>
<p align="center">Creative version of Tile Upscaler. The main ideas come from</p>
<p><center>
<a href="https://huggingface.co/spaces/gokaygokay/Tile-Upscaler" target="_blank">[Tile Upscaler]</a>
<a href="https://github.com/philz1337x/clarity-upscaler" target="_blank">[philz1337x]</a>
<a href="https://github.com/BatouResearch/controlnet-tile-upscale" target="_blank">[Pau-Lozano]</a>
</center></p>
"""

with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            run_button = gr.Button("Enhance Image")
        with gr.Column():
            output_slider = ImageSlider(label="Before / After", type="numpy")
    with gr.Accordion("Advanced Options", open=False):
        resolution = gr.Slider(minimum=128, maximum=2048, value=1024, step=128, label="Resolution")
        num_inference_steps = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Number of Inference Steps")
        strength = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.01, label="Strength")
        hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
        guidance_scale = gr.Slider(minimum=0, maximum=20, value=6, step=0.5, label="Guidance Scale")
        controlnet_strength = gr.Slider(minimum=0.0, maximum=2.0, value=0.75, step=0.05, label="ControlNet Strength")
        scheduler_name = gr.Dropdown(
            choices=["DDIM", "DPM++ 3M SDE Karras", "DPM++ 3M Karras"],
            value="DDIM",
            label="Scheduler"
        )

    run_button.click(fn=gradio_process_image, 
                     inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name],
                     outputs=output_slider)

demo.launch(debug=True, share=True)