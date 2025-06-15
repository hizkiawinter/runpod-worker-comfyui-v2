# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.30 --cuda-version 12.6 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client opencv-python scikit-image numexpr pandas

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=sdxl

# Change working directory to ComfyUI
WORKDIR /comfyui

WORKDIR /comfyui/custom_nodes 
RUN git clone https://github.com/hizkiawinter/ComfyS3.git
RUN git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

WORKDIR /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved/models
RUN wget -q -O hotshotxl_mm_v1.pth \
    https://huggingface.co/Kosinkadink/HotShot-XL-MotionModels/resolve/150e621dd65b4c9af2c87a287f8502a6e482c97f/hotshotxl_mm_v1.pth
WORKDIR /comfyui

RUN comfy-node-install comfyui-kjnodes comfyui-ic-light comfyui_nnlatentupscale comfyui_essentials ComfyUI_JPS-Nodes images-grid-comfy-plugin ComfyUI_Noise cg-use-everywhere ComfyUI_Comfyroll_CustomNodes comfyui-custom-scripts comfyui-to-python-extension comfyui-frame-interpolation comfyui_ipadapter_plus  comfyui-videohelpersuite comfyui_fizznodes comfyui_controlnet_aux  comfyui-advanced-controlnet ComfyUI-Manager

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip  models/controlnet models/clip_vision models/ipadapter models/loras models/animatediff_models
ADD skripsi-lora.safetensors /comfyui/models/loras 


# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors "https://civitai.com/api/download/models/198530?type=Model&format=SafeTensor&size=full&fp=fp16" && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/controlnet/xinsir-controlnet-canny.safetensors https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true && \ 
      wget -q -O models/controlnet/xinsir-controlnet-depth.safetensors https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true && \
      wget -q -O models/controlnet/xinsir-controlnet-openpose.safetensors https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true && \ 
      wget -q -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors && \ 
      wget -q -O models/ipadapter/ip-adapter_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors && \
      wget -q -O models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
    fi

RUN echo "ClipVision files:" && ls -lh models/clip_vision/

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes