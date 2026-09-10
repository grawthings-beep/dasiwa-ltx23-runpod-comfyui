ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
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
ARG WHATDREAMSCOST_COMFYUI_REF=20f639f302c44197c7887d2c677c571e75f3b5cb
RUN git clone --depth=1 https://github.com/Comfy-Org/ComfyUI-Manager.git ComfyUI-Manager \
    && git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite \
    && git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git rgthree-comfy \
    && git clone --depth=1 https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit.git comfyui-WhiteRabbit \
    && git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes \
    && git clone --depth=1 https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes.git ComfyUI-DaSiWa-Nodes \
    && git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo.git ComfyUI-LTXVideo \
    && git clone https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git WhatDreamsCost-ComfyUI \
    && git -C WhatDreamsCost-ComfyUI checkout "${WHATDREAMSCOST_COMFYUI_REF}" \
    && git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git ComfyUI-GGUF \
    && git clone --depth=1 https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git Nvidia_RTX_Nodes_ComfyUI

WORKDIR /opt/ComfyUI
RUN set -eux; \
    for req in custom_nodes/*/requirements.txt; do \
      if [ -f "$req" ]; then python -m pip install --no-cache-dir -r "$req"; fi; \
    done; \
    python -m pip install --no-cache-dir --upgrade "transformers[timm]==4.56.2"

COPY scripts/patch_ltxvideo_kornia.py /tmp/patch_ltxvideo_kornia.py
COPY scripts/patch_dasiwa_scaler_legacy.py /tmp/patch_dasiwa_scaler_legacy.py
COPY scripts/patch_kjnodes_sage_fallback.py /tmp/patch_kjnodes_sage_fallback.py
RUN python /tmp/patch_ltxvideo_kornia.py /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py \
    && python /tmp/patch_dasiwa_scaler_legacy.py /opt/ComfyUI/custom_nodes/ComfyUI-DaSiWa-Nodes \
    && python /tmp/patch_kjnodes_sage_fallback.py /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes

ARG INSTALL_SAGEATTENTION=true
ARG SAGEATTENTION_PACKAGE=sageattention==1.0.6
RUN if [ "${INSTALL_SAGEATTENTION}" = "true" ]; then \
      python -m pip install --no-cache-dir "${SAGEATTENTION_PACKAGE}" || \
      echo "WARNING: SageAttention install failed; KJNodes will fall back to standard attention."; \
    fi

COPY custom_nodes/dasiwa-compat /opt/ComfyUI/custom_nodes/dasiwa-compat

COPY start.sh /start-comfy.sh
COPY download_models.sh /download_models.sh
COPY workflows /opt/workflows

RUN chmod +x /start-comfy.sh /download_models.sh

EXPOSE 8188

CMD ["/start-comfy.sh"]
