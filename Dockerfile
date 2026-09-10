# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 PIP_CONSTRAINT=/opt/wan22/constraints.txt \
    EXPECTED_TORCH_VERSION=2.10.0+cu128 EXPECTED_TORCHVISION_VERSION=0.25.0+cu128 \
    EXPECTED_TORCHAUDIO_VERSION=2.10.0+cu128 EXPECTED_TORCH_CUDA=12.8 \
    HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates aria2 libglib2.0-0 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY config/constraints.txt /opt/wan22/constraints.txt
ARG COMFYUI_REVISION=a7b1d39d342d102f305797fb5ba12dc304d9c1f5
RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth=1 origin ${COMFYUI_REVISION} \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
    && rm -rf /opt/ComfyUI/.git \
    && python -m pip install -r /opt/ComfyUI/requirements.txt
# Isolate Hub 1.x from Transformers 4.x's Hub <1 dependency. Startup only imports
# the new Hub inside the downloader process, never inside ComfyUI.
RUN env -u PIP_CONSTRAINT python -m pip install --target /opt/wan22/downloader-libs \
    huggingface_hub==1.24.0 hf-xet==1.5.2
COPY custom_nodes/DaSiWa-WAN /opt/ComfyUI/custom_nodes/DaSiWa-WAN
COPY scripts /opt/wan22/scripts
COPY config/models.json /opt/wan22/config/models.json
COPY workflows /opt/wan22/workflows
COPY api /opt/wan22/api
COPY tests /opt/wan22/tests
RUN python /opt/wan22/scripts/gpu_preflight.py --stack-only \
    && python /opt/wan22/scripts/container_smoke.py
ARG BUNDLE_REVISION=unknown
ENV BUNDLE_REVISION=${BUNDLE_REVISION}
LABEL org.opencontainers.image.source="https://github.com/grawthings-beep/dasiwa-ltx23-runpod-comfyui" \
      org.opencontainers.image.revision="${BUNDLE_REVISION}" \
      io.grawthings.workflow="wan22-synthseduction-v9" \
      io.grawthings.comfy-revision="${COMFYUI_REVISION}"
WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT []
CMD ["python", "/opt/wan22/scripts/startup.py"]
