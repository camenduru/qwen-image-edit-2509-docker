FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libegl1 libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev cmake && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg xvfb && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

USER camenduru

RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir xformers --extra-index-url https://download.pytorch.org/whl/cu129 && \
    git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && cd /content/ComfyUI && pip install -r requirements.txt && \
    git clone --depth 1 --branch master https://github.com/lrzjason/Comfyui-QwenEditUtils /content/ComfyUI/custom_nodes/Comfyui-QwenEditUtils && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python moviepy timm scikit-image matplotlib diffusers accelerate peft torch-scatter runpod boto3 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors -d /content/ComfyUI/models/diffusion_models -o qwen_image_edit_2509_fp8_e4m3fn.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -d /content/ComfyUI/models/clip -o qwen_2.5_vl_7b_fp8_scaled.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -d /content/ComfyUI/models/vae -o qwen_image_vae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors -d /content/ComfyUI/models/loras -o Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime/resolve/main/Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors -d /content/ComfyUI/models/loras -o Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/aldniki_qwen_reality_transform_v01.safetensors -d /content/ComfyUI/models/loras -o aldniki_qwen_reality_transform_v01.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/A2R_2509_Base.safetensors -d /content/ComfyUI/models/loras -o A2R_2509_Base.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/rsshekhawat/Qwen-Edit-3DChibi-LoRA/resolve/main/qwen_3d_chibi_lora_v1_000000820.safetensors -d /content/ComfyUI/models/loras -o qwen_3d_chibi_lora_v1_000000820.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/qwen_image_ahegao.safetensors -d /content/ComfyUI/models/loras -o qwen_image_ahegao.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/PanelPainter_V2.safetensors -d /content/ComfyUI/models/loras -o PanelPainter_V2.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/3d/resolve/main/consistence_edit_v2.safetensors -d /content/ComfyUI/models/loras -o consistence_edit_v2.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/eigen-ai-labs/eigen-banana-qwen-image-edit/resolve/main/eigen-banana-qwen-image-edit-2509-fp16-lora.safetensors -d /content/ComfyUI/models/loras -o eigen-banana-qwen-image-edit-2509-fp16-lora.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/Samsung.safetensors -d /content/ComfyUI/models/loras -o Samsung.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/vafipas663/Qwen-Edit-2509-Upscale-LoRA/resolve/main/qwen-edit-enhance_000004250.safetensors -d /content/ComfyUI/models/loras -o qwen-edit-enhance_000004250.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/next-scene_lora-v2-3000.safetensors -d /content/ComfyUI/models/loras -o next-scene_lora-v2-3000.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles/resolve/main/%E9%95%9C%E5%A4%B4%E8%BD%AC%E6%8D%A2.safetensors -d /content/ComfyUI/models/loras -o Qwen-Edit-2509-Multiple-angles.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion/resolve/main/%E6%BA%B6%E5%9B%BE.safetensors -d /content/ComfyUI/models/loras -o Qwen-Image-Edit-2509-Fusion.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Relight/resolve/main/Qwen-Edit-Relight.safetensors -d /content/ComfyUI/models/loras -o Qwen-Edit-Relight.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/tarn59/apply_texture_qwen_image_edit_2509/resolve/main/apply_texture_v2_qwen_image_edit_2509.safetensors -d /content/ComfyUI/models/loras -o apply_texture_v2_qwen_image_edit_2509.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Qwen-Loras/resolve/main/extract-outfit_v3.safetensors -d /content/ComfyUI/models/loras -o extract-outfit_v3.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v3_qwen_image_edit_2509.safetensors -d /content/ComfyUI/models/loras -o bfs_head_v3_qwen_image_edit_2509.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py