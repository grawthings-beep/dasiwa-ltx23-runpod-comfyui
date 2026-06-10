# DaSiWa LTX 2.3 OmniForge on RunPod

This repository builds a reproducible ComfyUI image for the DaSiWa LTX 2.3
OmniForge V39 workflow.

## Recommended workflow

Open this file after the Pod starts:

```text
dasiwa-ltx23/DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V39_runpod.json
```

The RunPod variant keeps the original V39 workflow intact in a separate file
and changes only deployment defaults:

- Uses the public LTX 2.3 distilled FP8 transformer downloaded by the image.
- Uses the downloaded Heretic Gemma encoder and the `LTX/` VAE paths.
- Loads the Gemma encoder on CPU for lower VRAM pressure.
- Disables SageAttention and NVIDIA RTX VFX post-processing by default.
- Keeps the current LTX tiled VAE decoder installed, but leaves tiled decode off.

The unmodified V39 workflow is also included. It expects its original
DaSiWa GoldenLace UNet and model names.

## Included custom nodes

Custom nodes are pinned by commit in `custom_nodes.lock.json`, including:

- WhatDreamsCost-ComfyUI for `LTXDirector` and `LTXDirectorGuide`
- ComfyUI-LTXVideo
- ComfyUI-KJNodes
- ComfyUI-DaSiWa-Nodes
- ComfyUI-VideoHelperSuite
- rgthree-comfy
- comfyui-WhiteRabbit
- ComfyUI-GGUF
- Nvidia_RTX_Nodes_ComfyUI source (its `nvidia-vfx` runtime is optional)
- ComfyUI-Manager

The Docker build imports ComfyUI and verifies that every backend node used by
the recommended V39 workflow is registered. A missing or broken custom node
therefore fails the image build instead of appearing later in the UI.

## Persistent volume behavior

On every container start, image-managed ComfyUI code and pinned custom nodes
are refreshed into `/workspace/ComfyUI`. Existing models, inputs, outputs,
workflows, and user-added custom nodes are not deleted.

The startup script also creates safe placeholders for:

- `placeholder.mp4`
- `placeholder.mp3`
- `example.png`
- `#Watermark-Darksidewalker-Emblem.png`
- `#audio-mark.png`

## Build

```bash
docker build --platform linux/amd64 \
  -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0 .

docker push ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0
```

Optional SageAttention build:

```bash
docker build --platform linux/amd64 \
  --build-arg INSTALL_SAGEATTENTION=true \
  -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0-sage .
```

`nvidia-vfx` is omitted by default because PyPI currently publishes it only as
a source distribution, with no Linux CPython 3.11 wheel. The recommended V39
workflow bypasses that path. To experiment with RTX VFX, add:

```text
--build-arg INSTALL_NVIDIA_VFX=true
```

## RunPod template

`runpod-template.json` uses:

- Image: `ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0`
- Container disk: 80 GB
- Volume disk: 160 GB
- Volume mount: `/workspace`
- ComfyUI: HTTP port 8188

The default template downloads the public LTX 2.3 FP8 transformer and all
models required by the recommended V39 workflow. Optional post-processing
models are disabled because the corresponding workflow paths are disabled by
default.

To use another transformer, set:

```text
DOWNLOAD_DEFAULT_UNET=false
MAIN_UNET_URL=https://...
MAIN_UNET_NAME=your-model.safetensors
```

For a Civitai URL, also set `CIVITAI_TOKEN` through a RunPod secret. The token
is appended at download time and masked in logs.

## Validation

Run before publishing:

```bash
python scripts/audit_workflows.py
python scripts/generate_workflows.py --check
python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on pull requests and before publishing the
Docker image.

## Legacy workflows

The V36 workflows remain available for reference and migration. V39 RunPod is
the maintained default.

The image is based on CUDA 12.4.1 and pins Transformers 4.56.2 for compatibility
with the bundled Torch 2.4 runtime.
