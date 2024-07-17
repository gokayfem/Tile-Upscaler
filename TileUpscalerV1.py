import spaces

import os
import time

import torch

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.models import AutoencoderKL

from PIL import Image
import cv2
import numpy as np

from RealESRGAN import RealESRGAN

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
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        model_path = "models/models/Stable-diffusion/juggernaut_reborn.safetensors"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=safety_checker
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

def prepare_image(input_image, resolution, hdr):
    condition_image = resize_and_upscale(input_image, resolution)
    condition_image = create_hdr_effect(condition_image, hdr)
    return condition_image

@spaces.GPU
@timer_func
def gradio_process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale):
    print("Starting image processing...")
    torch.cuda.empty_cache()
    
    condition_image = prepare_image(input_image, resolution, hdr)
    
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": condition_image,
        "control_image": condition_image,
        "width": condition_image.size[0],
        "height": condition_image.size[1],
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(device=device).manual_seed(0),
    }
    
    print("Running inference...")
    result = lazy_pipe(**options).images[0]
    print("Image processing completed successfully")
    
    # Convert input_image and result to numpy arrays
    input_array = np.array(input_image)
    result_array = np.array(result)
    
    return [input_array, result_array]

title = """<h1 align="center">Image Upscaler with Tile Controlnet</h1>
<p align="center">The main ideas come from</p>
<p><center>
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
        resolution = gr.Slider(minimum=256, maximum=2048, value=512, step=256, label="Resolution")
        num_inference_steps = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Number of Inference Steps")
        strength = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.01, label="Strength")
        hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
        guidance_scale = gr.Slider(minimum=0, maximum=20, value=3, step=0.5, label="Guidance Scale")

    run_button.click(fn=gradio_process_image, 
                     inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale],
                     outputs=output_slider)

demo.launch(share=True)