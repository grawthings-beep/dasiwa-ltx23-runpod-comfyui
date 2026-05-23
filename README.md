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
  - ComfyUI-GGUF
  - Nvidia_RTX_Nodes_ComfyUI
  - ComfyUI-Manager
- The original workflow and a RunPod-default workflow in `workflows/`.
- First-start model downloader for the public Hugging Face assets and the Civitai SolsticeCoin v2 UNet.

## Build and push

Recommended GitHub repository:

```text
grawthings-beep/dasiwa-ltx23-runpod-comfyui
```

Recommended container image:

```text
ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.1.1
```

```bash
docker build --platform linux/amd64 -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.1.1 .
docker push ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.1.1
```

Optional SageAttention build:

```bash
docker build --platform linux/amd64 --build-arg INSTALL_SAGEATTENTION=true -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.1.1-sage .
```

## RunPod template settings

Use `runpod-template.json` as the API payload or fill the RunPod UI with these values:

- Container image: `ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.1.1`
- Container disk: `80 GB`
- Volume disk: `160 GB`
- Volume mount path: `/workspace`
- HTTP ports: `8188`, `8888`
- TCP ports: `22`

ComfyUI runs on HTTP port `8188`.

The Docker base image is pinned to CUDA 12.4.1 to avoid RunPod hosts with older NVIDIA drivers failing before the container starts.

## Model behavior

On first start, `DOWNLOAD_MODELS=true` downloads the public model files into `/workspace/ComfyUI/models`.

The original workflow selects `SolsticeCoin_v2_fp8_mixed.safetensors` as the main UNet. The template now downloads SolsticeCoin v2 from Civitai by default and saves it with that workflow filename.
The main transformer is stored in `models/diffusion_models` and symlinked into `models/unet` for compatibility with old and new ComfyUI loaders.

Create a RunPod secret named `civitai_token`, then set:

```text
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
MAIN_UNET_URL=https://civitai.com/api/download/models/2917963?type=Model&format=SafeTensor&size=full&fp=fp8
MAIN_UNET_NAME=SolsticeCoin_v2_fp8_mixed.safetensors
```

The downloader appends the token at runtime and masks it in logs.

## Workflow files

- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36.json`: original file copied from Downloads.
- `DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36_runpod-default.json`: same workflow, but main UNet changed to the default public LTX 2.3 FP8 transformer file.

On Pod start, both files are copied to:

```text
/workspace/ComfyUI/user/default/workflows/
```

## Notes

- The first startup can take a long time because the LTX 2.3 model and text encoder are large.
- If you do not use the optional LoRA or post-processing paths, set `DOWNLOAD_OPTIONAL_LORA=false` or `DOWNLOAD_OPTIONAL_POST_MODELS=false`.
- Watermark images referenced by the workflow are not bundled. Disable those nodes or upload the images to ComfyUI input if you use that path.
