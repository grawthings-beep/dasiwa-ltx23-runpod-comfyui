#!/usr/bin/env bash
set -Eeuo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${DOWNLOAD_DEFAULT_UNET:=true}"
: "${DOWNLOAD_OPTIONAL_LORA:=true}"
: "${DOWNLOAD_OPTIONAL_POST_MODELS:=true}"
: "${MAIN_UNET_NAME:=SolsticeCoin_v2_fp8_mixed.safetensors}"
: "${MAIN_UNET_URL:=}"
: "${CIVITAI_TOKEN:=}"

MODELS_DIR="${COMFYUI_DIR}/models"

download() {
  local url="$1"
  local dest="$2"
  local dir
  local tmp
  local effective_url
  local safe_url
  dir="$(dirname "${dest}")"
  tmp="${dest}.part"
  effective_url="${url}"
  safe_url="${url}"
  mkdir -p "${dir}"

  if [ -s "${dest}" ]; then
    echo "Already present: ${dest}"
    return
  fi

  if [[ -n "${CIVITAI_TOKEN}" && "${url}" == *"civitai."* && "${url}" != *"token="* ]]; then
    if [[ "${url}" == *"?"* ]]; then
      effective_url="${url}&token=${CIVITAI_TOKEN}"
      safe_url="${url}&token=<CIVITAI_TOKEN>"
    else
      effective_url="${url}?token=${CIVITAI_TOKEN}"
      safe_url="${url}?token=<CIVITAI_TOKEN>"
    fi
  fi

  echo "Downloading: ${safe_url}"
  aria2c \
    --continue=true \
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=1M \
    --console-log-level=warn \
    --summary-interval=30 \
    --dir="${dir}" \
    --out="$(basename "${tmp}")" \
    "${effective_url}"
  mv "${tmp}" "${dest}"
}

link_model() {
  local src="$1"
  local link="$2"
  mkdir -p "$(dirname "${link}")"

  if [ -e "${link}" ] || [ -L "${link}" ]; then
    return
  fi

  ln -s "${src}" "${link}"
}

if [ -n "${MAIN_UNET_URL}" ]; then
  download "${MAIN_UNET_URL}" "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}"
  link_model "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}" "${MODELS_DIR}/unet/${MAIN_UNET_NAME}"
elif [ "${DOWNLOAD_DEFAULT_UNET}" = "true" ] || [ "${DOWNLOAD_DEFAULT_UNET}" = "1" ]; then
  download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors?download=true" \
    "${MODELS_DIR}/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
  link_model "${MODELS_DIR}/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors" \
    "${MODELS_DIR}/unet/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
fi

download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX/LTX23_audio_vae_bf16.safetensors"
link_model "${MODELS_DIR}/vae/LTX/LTX23_audio_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX2/LTX23_audio_vae_bf16.safetensors"
download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX/LTX23_video_vae_bf16.safetensors"
link_model "${MODELS_DIR}/vae/LTX/LTX23_video_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX2/LTX23_video_vae_bf16.safetensors"
download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX/taeltx2_3.safetensors"
link_model "${MODELS_DIR}/vae/LTX/taeltx2_3.safetensors" \
  "${MODELS_DIR}/vae/LTX2/taeltx2_3.safetensors"

download "https://huggingface.co/DreamFast/gemma-3-12b-it-heretic-v2/resolve/main/comfyui/gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors?download=true" \
  "${MODELS_DIR}/text_encoders/gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors?download=true" \
  "${MODELS_DIR}/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

download "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors?download=true" \
  "${MODELS_DIR}/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
download "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors?download=true" \
  "${MODELS_DIR}/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

if [ "${DOWNLOAD_OPTIONAL_LORA}" = "true" ] || [ "${DOWNLOAD_OPTIONAL_LORA}" = "1" ]; then
  download "https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
fi

if [ "${DOWNLOAD_OPTIONAL_POST_MODELS}" = "true" ] || [ "${DOWNLOAD_OPTIONAL_POST_MODELS}" = "1" ]; then
  download "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors?download=true" \
    "${MODELS_DIR}/upscale_models/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
  download "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors?download=true" \
    "${MODELS_DIR}/vfi_models/rife_v4.26.safetensors"
fi
