ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG COMFYUI_REF=039ed38ed10ad0072a13f6471e06913ed33d5e56

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
RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth=1 origin "${COMFYUI_REF}" \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "${COMFYUI_REF}"

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

COPY custom_nodes.lock.json /tmp/custom_nodes.lock.json
COPY scripts/install_custom_nodes.py /tmp/install_custom_nodes.py
RUN python /tmp/install_custom_nodes.py \
      --manifest /tmp/custom_nodes.lock.json \
      --destination /opt/ComfyUI/custom_nodes

ARG INSTALL_NVIDIA_VFX=false
WORKDIR /opt/ComfyUI
RUN set -eux; \
    for req in custom_nodes/*/requirements.txt; do \
      if [ ! -f "$req" ]; then continue; fi; \
      if [ "${INSTALL_NVIDIA_VFX}" = "true" ]; then \
        python -m pip install --no-cache-dir -r "$req"; \
      else \
        grep -viE '^[[:space:]]*nvidia-vfx([<>=!~].*)?[[:space:]]*$' "$req" \
          > /tmp/custom-node-requirements.txt || true; \
        python -m pip install --no-cache-dir -r /tmp/custom-node-requirements.txt; \
      fi; \
    done; \
    python -m pip install --no-cache-dir --upgrade "transformers[timm]==4.56.2"

COPY scripts/patch_ltxvideo_kornia.py /tmp/patch_ltxvideo_kornia.py
RUN python /tmp/patch_ltxvideo_kornia.py /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py

COPY workflows /opt/workflows
COPY scripts/validate_workflow_nodes.py /tmp/validate_workflow_nodes.py
RUN python /tmp/validate_workflow_nodes.py \
      --comfy-dir /opt/ComfyUI \
      /opt/workflows/DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V39_runpod.json

ARG INSTALL_SAGEATTENTION=false
RUN if [ "${INSTALL_SAGEATTENTION}" = "true" ]; then \
      python -m pip install --no-cache-dir sageattention; \
    fi

COPY start.sh /start-comfy.sh
COPY download_models.sh /download_models.sh

RUN chmod +x /start-comfy.sh /download_models.sh

EXPOSE 8188 8888

CMD ["/start-comfy.sh"]
