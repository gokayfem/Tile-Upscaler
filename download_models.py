import os
import requests

def download_file(url, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        print(f"File already exists: {file_path}")
    else:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"File successfully downloaded and saved: {file_path}")
        else:
            print(f"Error downloading the file. Status code: {response.status_code}")

def download_models():
    models = {
        "MODEL": ("https://huggingface.co/dantea1118/juggernaut_reborn/resolve/main/juggernaut_reborn.safetensors?download=true", "models/models/Stable-diffusion", "juggernaut_reborn.safetensors"),
        "UPSCALER_X2": ("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true", "models/upscalers/", "RealESRGAN_x2.pth"),
        "UPSCALER_X4": ("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true", "models/upscalers/", "RealESRGAN_x4.pth"),
        "NEGATIVE_1": ("https://huggingface.co/philz1337x/embeddings/resolve/main/verybadimagenegative_v1.3.pt?download=true", "models/embeddings", "verybadimagenegative_v1.3.pt"),
        "NEGATIVE_2": ("https://huggingface.co/datasets/AddictiveFuture/sd-negative-embeddings/resolve/main/JuggernautNegative-neg.pt?download=true", "models/embeddings", "JuggernautNegative-neg.pt"),
        "LORA_1": ("https://huggingface.co/philz1337x/loras/resolve/main/SDXLrender_v2.0.safetensors?download=true", "models/Lora", "SDXLrender_v2.0.safetensors"),
        "LORA_2": ("https://huggingface.co/philz1337x/loras/resolve/main/more_details.safetensors?download=true", "models/Lora", "more_details.safetensors"),
        "CONTROLNET": ("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth?download=true", "models/ControlNet", "control_v11f1e_sd15_tile.pth"),
        "VAE": ("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true", "models/VAE", "vae-ft-mse-840000-ema-pruned.safetensors"),
    }

    for model, (url, folder, filename) in models.items():
        download_file(url, folder, filename)

download_models()