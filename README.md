# DaSiWa LTX 2.3 OmniForge RunPod Template

This folder packages the provided ComfyUI workflow as a RunPod Pod template.

## What is included

- ComfyUI installed under `/opt/ComfyUI` and copied to `/workspace/ComfyUI` on first start.
- Custom nodes required by the workflow:
  - ComfyUI-VideoHelperSuite
  - rgthree-comfy
  - comfyui-WhiteRabbit
  - ComfyUI-KJNodes
  - ComfyUI-DaSiWa-Nodes
  - ComfyUI-LTXVideo
  - WhatDreamsCost-ComfyUI
  - ComfyUI-GGUF
  - Nvidia_RTX_Nodes_ComfyUI
  - ComfyUI-Manager
- The original V36 workflow set and the provided V39 workflow in `workflows/`.
- First-start model downloader for the public Hugging Face assets and the Civitai Golden Lace v3 UNet used by V39.

## Build and push

Recommended GitHub repository:

```text
grawthings-beep/dasiwa-ltx23-runpod-comfyui
```

Recommended container image:

```text
ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0
```

```bash
docker build --platform linux/amd64 -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0 .
docker push ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0
```

Optional SageAttention build:

```bash
docker build --platform linux/amd64 --build-arg INSTALL_SAGEATTENTION=true -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0-sage .
```

## RunPod template settings

Use `runpod-template.json` as the API payload or fill the RunPod UI with these values:

- Container image: `ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.0`
- Container disk: `80 GB`
- Volume disk: `160 GB`
- Volume mount path: `/workspace`
- HTTP ports: `8188`, `8888`
- TCP ports: `22`

ComfyUI runs on HTTP port `8188`.

The Docker base image is pinned to CUDA 12.4.1 to avoid RunPod hosts with older NVIDIA drivers failing before the container starts.
The Transformers package is pinned to `4.56.2` because newer releases import continuous-batching modules that require newer Torch APIs than the CUDA 12.4 base provides.
ComfyUI starts with `--enable-cors-header` so RunPod's proxy does not trigger host/origin 403 responses.
The startup script creates tiny placeholder media and image files so inactive video, audio, watermark, and reference-image branches do not fail validation before you replace them.

## Model behavior

On first start, `DOWNLOAD_MODELS=true` downloads the model files into `/workspace/ComfyUI/models`.
Downloads run in parallel by default. Tune `MAX_PARALLEL_DOWNLOADS`, `ARIA2_CONNECTIONS`, and `ARIA2_SPLIT` if your RunPod host or network is unhappy.

The V39 workflow selects `LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors` as the main UNet. The template downloads Golden Lace v3 FP8 from Civitai by default and saves it with that workflow filename.
The main transformer is stored in `models/diffusion_models` and symlinked into `models/unet` for compatibility with old and new ComfyUI loaders.
The V39 VAE paths use `models/vae/LTX2/`; compatibility symlinks are created under `models/vae/LTX/` for the older V36 workflows.

Create a RunPod secret named `civitai_token`, then set:

```text
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
MAIN_UNET_URL=https://civitai.com/api/download/models/2967331?type=Model&format=SafeTensor&size=full&fp=fp8
MAIN_UNET_NAME=LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors
```

The downloader appends the token at runtime and masks it in logs.

## Workflow files

- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36.json`: original file copied from Downloads.
- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36_runpod-default.json`: same workflow, but main UNet changed to the default public LTX 2.3 FP8 transformer file.
- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36_runpod-no-ltx-tiled-decode.json`: older RunPod-safe V36 variant. It removes the optional tiled decode branch, disables optional SageAttention, disables fp16 accumulation for Torch 2.4, and loads the Gemma text encoder on CPU by default for 24 GB GPUs.
- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V39.json`: provided V39 workflow copied without content changes. This is the default target for the model names and paths above.

On Pod start, both files are copied to:

```text
/workspace/ComfyUI/user/default/workflows/
```

## Notes

- The first startup can take a long time because the LTX 2.3 model and text encoder are large.
- If you do not use the optional LoRA or post-processing paths, set `DOWNLOAD_OPTIONAL_LORA=false` or `DOWNLOAD_OPTIONAL_POST_MODELS=false`.
- V39 has disabled GGUF branches that reference `placeholder.gguf`; the downloader creates inert placeholder files so the workflow can open cleanly. Replace them with real GGUF models before enabling those branches.
