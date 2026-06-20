ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_DIR=/workspace/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

WORKDIR /opt/ComfyUI
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir \
      accelerate \
      av \
      comfy-cli \
      diffusers \
      hf_transfer \
      imageio-ffmpeg \
      librosa \
      opencv-python-headless \
      protobuf \
      sentencepiece \
      soundfile \
      "transformers[timm]==4.56.2"

WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth=1 https://github.com/Comfy-Org/ComfyUI-Manager.git ComfyUI-Manager \
    && git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite \
    && git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git rgthree-comfy \
    && git clone --depth=1 https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit.git comfyui-WhiteRabbit \
    && git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes \
    && git clone --depth=1 https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git ComfyUI-DaSiWa-Nodes \
    && git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo.git ComfyUI-LTXVideo \
    && git clone --depth=1 https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git WhatDreamsCost-ComfyUI \
    && git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git ComfyUI-GGUF \
    && git clone --depth=1 https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git Nvidia_RTX_Nodes_ComfyUI

WORKDIR /opt/ComfyUI
RUN set -eux; \
    for req in custom_nodes/*/requirements.txt; do \
      if [ -f "$req" ]; then python -m pip install --no-cache-dir -r "$req"; fi; \
    done; \
    python -m pip install --no-cache-dir --upgrade "transformers[timm]==4.56.2"

COPY scripts/patch_ltxvideo_kornia.py /tmp/patch_ltxvideo_kornia.py
RUN python /tmp/patch_ltxvideo_kornia.py /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py

ARG INSTALL_SAGEATTENTION=false
RUN if [ "${INSTALL_SAGEATTENTION}" = "true" ]; then \
      python -m pip install --no-cache-dir sageattention; \
    fi

COPY start.sh /start-comfy.sh
COPY download_models.sh /download_models.sh
COPY workflows /opt/workflows

RUN chmod +x /start-comfy.sh /download_models.sh

EXPOSE 8188 8888

CMD ["/start-comfy.sh"]
