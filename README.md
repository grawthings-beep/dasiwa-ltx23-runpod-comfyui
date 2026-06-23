# DaSiWa LTX 2.3 OmniForge RunPod Template

This folder packages the provided ComfyUI workflow as a RunPod Pod template.

## What is included

- ComfyUI installed under `/opt/ComfyUI` and synced to `/workspace/ComfyUI` on each start.
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
  - dasiwa-compat (bundled here: registers a passthrough `ColorTransfer` so the workflow has no missing node types)
- The original V36 workflow set and the provided V39 workflow in `workflows/`.
- First-start model downloader for the public Hugging Face assets and the Civitai Golden Lace v3 UNet used by V39.

## Build and push

Recommended GitHub repository:

```text
grawthings-beep/dasiwa-ltx23-runpod-comfyui
```

Recommended container image:

```text
ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.1
```

```bash
docker build --platform linux/amd64 -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.1 .
docker push ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.1
```

SageAttention is installed by default for the workflow's active KJNodes SageAttention path. If the install is unavailable on a future base image, the image still builds and KJNodes falls back to standard attention instead of crashing.

```bash
docker build --platform linux/amd64 --build-arg INSTALL_SAGEATTENTION=false -t ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.1-no-sage .
```

## RunPod template settings

Use `runpod-template.json` as the API payload or fill the RunPod UI with these values:

- Container image: `ghcr.io/grawthings-beep/dasiwa-ltx23-runpod-comfyui:0.2.1`
- Container disk: `80 GB`
- Volume disk: `160 GB`
- Volume mount path: `/workspace`
- HTTP ports: `8188`, `8888`
- TCP ports: `22`

ComfyUI runs on HTTP port `8188`.

The Docker base image uses RunPod's PyTorch 2.8 / CUDA 12.9 image so current ComfyUI can use its dynamic VRAM model loader instead of falling back to the older legacy loader.
Select a RunPod host whose NVIDIA driver supports CUDA 12.9, or rebuild with a lower `BASE_IMAGE` only if you are intentionally targeting older hosts.
The Transformers package is pinned to `4.56.2` because newer releases import continuous-batching modules that require newer Torch APIs than this workflow stack currently uses.
ComfyUI starts with `--enable-cors-header` so RunPod's proxy does not trigger host/origin 403 responses.
ComfyUI also starts with `--reserve-vram 2` by default via `COMFYUI_RESERVE_VRAM=2`, leaving headroom for VAE/post-processing and avoiding edge-of-card allocations on 48 GB GPUs.
Startup refreshes the ComfyUI application files from the image while preserving `/workspace/ComfyUI/models`, `input`, `output`, `temp`, and `user`. This repairs stale or partial ComfyUI code left on a persistent RunPod volume.
Model downloads run in the background by default, so ComfyUI can become reachable before the large UNet and text encoder finish downloading.
The startup script creates tiny placeholder media and image files so inactive video, audio, watermark, and reference-image branches do not fail validation before you replace them.
KJNodes is patched at image build time so missing or unsupported SageAttention kernels are treated as a warning and the model continues with standard attention, keeping the provided workflow unchanged.

## Model behavior

On first start, `DOWNLOAD_MODELS=true` downloads the model files into `/workspace/ComfyUI/models`.
Downloads run in parallel by default. Tune `MAX_PARALLEL_DOWNLOADS`, `ARIA2_CONNECTIONS`, and `ARIA2_SPLIT` if your RunPod host or network is unhappy.
Set `DOWNLOAD_MODELS_BACKGROUND=false` only when you want the container to block ComfyUI startup until every model has finished downloading.

The V39 workflow selects `LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors` as the main UNet. The template downloads Golden Lace v3 FP8 from Civitai by default and saves it with that workflow filename.
If an older RunPod template still passes the legacy Solstice Civitai model URL, the downloader automatically switches it to the V39 Golden Lace v3 defaults.

**No-Civitai fallback:** if `CIVITAI_TOKEN` is not set (or is left as the unresolved `{{ RUNPOD_SECRET_civitai_token }}` placeholder), the downloader automatically fetches the public, ungated **LTX 2.3 distilled transformer** from Hugging Face and saves it under the same `LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors` filename. This means generation works out of the box with zero Civitai setup; set `CIVITAI_TOKEN` only if you specifically want the Golden Lace v3 weights.

**Corrupt-download guard:** a gated/failed download often returns a small HTML or JSON error page instead of the model. The downloader now rejects such files (too small, or starting with `<!DOCTYPE`/`<html`/`{"error"`) and fails loudly instead of saving a broken `.safetensors` that would later cause a confusing ComfyUI load error.
The main transformer is stored in `models/diffusion_models` and symlinked into `models/unet` for compatibility with old and new ComfyUI loaders.
The V39 VAE paths use `models/vae/LTX2/`; compatibility symlinks are created under `models/vae/LTX/` for the older V36 workflows.

Create RunPod secrets named `hf_token` and `civitai_token`, then set:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
CIVITAI_TOKEN={{ RUNPOD_SECRET_civitai_token }}
COMFYUI_RESERVE_VRAM=2
MAIN_UNET_URL=https://civitai.com/api/download/models/2967331?type=Model&format=SafeTensor&size=full&fp=fp8
MAIN_UNET_NAME=LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors
```

The downloader appends the token at runtime and masks it in logs.
Optional Hugging Face LoRAs from `uwgm/nikke-loras` are skipped when `HF_TOKEN`
is not set. Optional Civitai LoRAs are skipped when `CIVITAI_TOKEN` is not set.

When `DOWNLOAD_OPTIONAL_LORA=true`, startup also downloads the LoRAs mirrored
from `grawthings-beep/ltx-video-runpod`:

```text
models/loras/LTX/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors
models/loras/LTX/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors
models/loras/LTX/LTX2.3_reasoning_I2V_V3.safetensors
models/loras/LTX/latexturn_10Eros_i2v_v0.5.safetensors
models/loras/LTX/LTXV23_FOOTJOB_V1.safetensors
models/loras/LTX/TentacleMotion_10Eros_i2v_v1.0.safetensors
models/loras/LTX/stomach_bulge_10eros_sulphur_v1.safetensors
models/loras/LTX/nsfw_riding_backshot_frontshot_ltx23_v1.0.safetensors
models/loras/civitai/ltx23_phut_hon_civitai_2806861.safetensors
models/loras/civitai/smoothmix_animations_ltx_civitai_2911845.safetensors
models/loras/civitai/civitai_2849892.safetensors
```

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
- The V39 workflow contains a bypassed `ColorTransfer` node that is not provided by any installed custom-node pack. The image registers a dependency-free compatibility passthrough for it (`custom_nodes/dasiwa-compat`) that returns its input image unchanged, so the workflow no longer reports a missing node type and the (bypassed) branch validates correctly.
