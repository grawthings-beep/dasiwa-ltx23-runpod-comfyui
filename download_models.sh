#!/usr/bin/env bash
set -Eeuo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${DOWNLOAD_DEFAULT_UNET:=true}"
: "${DOWNLOAD_OPTIONAL_LORA:=true}"
: "${DOWNLOAD_OPTIONAL_POST_MODELS:=true}"
: "${CREATE_GGUF_PLACEHOLDERS:=true}"
: "${MAX_PARALLEL_DOWNLOADS:=4}"
: "${ARIA2_CONNECTIONS:=8}"
: "${ARIA2_SPLIT:=8}"
: "${VERIFY_MODEL_HASHES:=false}"
: "${MAIN_UNET_NAME:=LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors}"
: "${MAIN_UNET_URL:=https://civitai.com/api/download/models/2967331?type=Model&format=SafeTensor&size=full&fp=fp8}"
: "${MAIN_UNET_SHA256:=86E14FD4EAF24AE39D3BB2497E9A86C723888A9172CFFECA31C9730DC2C126E2}"
: "${CIVITAI_TOKEN:=}"
: "${HF_TOKEN:=}"

MODELS_DIR="${COMFYUI_DIR}/models"
DOWNLOAD_PIDS=()
DOWNLOAD_FAILED=0

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

download_url() {
  local url="$1"

  if [[ -n "${CIVITAI_TOKEN}" && "${url}" == *"civitai."* && "${url}" != *"token="* ]]; then
    if [[ "${url}" == *"?"* ]]; then
      printf '%s&token=%s' "${url}" "${CIVITAI_TOKEN}"
    else
      printf '%s?token=%s' "${url}" "${CIVITAI_TOKEN}"
    fi
    return
  fi

  printf '%s' "${url}"
}

mask_url() {
  local url="$1"

  if [[ -n "${CIVITAI_TOKEN}" ]]; then
    url="${url//${CIVITAI_TOKEN}/<CIVITAI_TOKEN>}"
  fi
  if [[ -n "${HF_TOKEN}" ]]; then
    url="${url//${HF_TOKEN}/<HF_TOKEN>}"
  fi

  printf '%s' "${url}"
}

verify_hash() {
  local file="$1"
  local sha256="${2:-}"

  if [[ -z "${sha256}" ]] || ! is_truthy "${VERIFY_MODEL_HASHES}"; then
    return
  fi

  echo "${sha256}  ${file}" | sha256sum -c -
}

download() {
  local url="$1"
  local dest="$2"
  local sha256="${3:-}"
  local dir
  local tmp
  local effective_url
  local safe_url
  local -a aria2_args

  dir="$(dirname "${dest}")"
  tmp="${dest}.part"
  effective_url="$(download_url "${url}")"
  safe_url="$(mask_url "${effective_url}")"
  mkdir -p "${dir}"

  if [ -s "${dest}" ]; then
    verify_hash "${dest}" "${sha256}"
    echo "Already present: ${dest}"
    return
  fi

  echo "Downloading: ${safe_url}"
  aria2_args=(
    --continue=true
    --allow-overwrite=true
    --auto-file-renaming=false
    --file-allocation=none
    --max-connection-per-server="${ARIA2_CONNECTIONS}"
    --split="${ARIA2_SPLIT}"
    --min-split-size=1M
    --max-tries=8
    --retry-wait=10
    --timeout=60
    --console-log-level=warn
    --summary-interval=30
    --dir="${dir}"
    --out="$(basename "${tmp}")"
  )

  if [[ -n "${HF_TOKEN}" && "${url}" == *"huggingface.co"* ]]; then
    aria2_args+=(--header="Authorization: Bearer ${HF_TOKEN}")
  fi

  aria2c "${aria2_args[@]}" "${effective_url}"
  verify_hash "${tmp}" "${sha256}"
  mv -f "${tmp}" "${dest}"
}

queue_download() {
  local done_pid

  while [ "${#DOWNLOAD_PIDS[@]}" -ge "${MAX_PARALLEL_DOWNLOADS}" ]; do
    if ! wait -n -p done_pid; then
      DOWNLOAD_FAILED=1
    fi
    remove_pid "${done_pid:-}"
  done

  download "$@" &
  DOWNLOAD_PIDS+=("$!")
}

remove_pid() {
  local done_pid="$1"
  local remaining=()
  local pid

  if [ -z "${done_pid}" ]; then
    return
  fi

  for pid in "${DOWNLOAD_PIDS[@]}"; do
    if [ "${pid}" != "${done_pid}" ]; then
      remaining+=("${pid}")
    fi
  done
  DOWNLOAD_PIDS=("${remaining[@]}")
}

wait_for_downloads() {
  local done_pid

  while [ "${#DOWNLOAD_PIDS[@]}" -gt 0 ]; do
    if ! wait -n -p done_pid; then
      DOWNLOAD_FAILED=1
    fi
    remove_pid "${done_pid:-}"
  done

  if [ "${DOWNLOAD_FAILED}" -ne 0 ]; then
    echo "One or more model downloads failed." >&2
    exit 1
  fi
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

reuse_existing() {
  local existing="$1"
  local wanted="$2"

  if [ ! -e "${wanted}" ] && [ ! -L "${wanted}" ] && [ -s "${existing}" ]; then
    link_model "${existing}" "${wanted}"
  fi
}

create_gguf_placeholders() {
  if ! is_truthy "${CREATE_GGUF_PLACEHOLDERS}"; then
    return
  fi

  mkdir -p "${MODELS_DIR}/unet" "${MODELS_DIR}/text_encoders"
  [ -e "${MODELS_DIR}/unet/placeholder.gguf" ] || : > "${MODELS_DIR}/unet/placeholder.gguf"
  [ -e "${MODELS_DIR}/text_encoders/placeholder.gguf" ] || : > "${MODELS_DIR}/text_encoders/placeholder.gguf"
}

create_gguf_placeholders

reuse_existing \
  "${MODELS_DIR}/text_encoders/gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors" \
  "${MODELS_DIR}/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"
reuse_existing \
  "${MODELS_DIR}/vae/LTX/LTX23_audio_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX2/LTX23_audio_vae_bf16.safetensors"
reuse_existing \
  "${MODELS_DIR}/vae/LTX/LTX23_video_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX2/LTX23_video_vae_bf16.safetensors"
reuse_existing \
  "${MODELS_DIR}/vae/LTX/taeltx2_3.safetensors" \
  "${MODELS_DIR}/vae/LTX2/taeltx2_3.safetensors"

if [ -n "${MAIN_UNET_URL}" ]; then
  queue_download "${MAIN_UNET_URL}" "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}" "${MAIN_UNET_SHA256}"
elif is_truthy "${DOWNLOAD_DEFAULT_UNET}"; then
  MAIN_UNET_NAME="ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
  queue_download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors?download=true" \
    "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}"
fi

queue_download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX2/LTX23_audio_vae_bf16.safetensors"
queue_download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX2/LTX23_video_vae_bf16.safetensors"
queue_download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors?download=true" \
  "${MODELS_DIR}/vae/LTX2/taeltx2_3.safetensors"

queue_download "https://huggingface.co/DreamFast/gemma-3-12b-it-heretic-v2/resolve/main/comfyui/gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors?download=true" \
  "${MODELS_DIR}/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"
queue_download "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors?download=true" \
  "${MODELS_DIR}/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

queue_download "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors?download=true" \
  "${MODELS_DIR}/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
queue_download "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors?download=true" \
  "${MODELS_DIR}/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

if is_truthy "${DOWNLOAD_OPTIONAL_LORA}"; then
  queue_download "https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments/resolve/main/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
fi

if is_truthy "${DOWNLOAD_OPTIONAL_POST_MODELS}"; then
  queue_download "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors?download=true" \
    "${MODELS_DIR}/upscale_models/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
  queue_download "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors?download=true" \
    "${MODELS_DIR}/vfi_models/rife_v4.26.safetensors"
fi

wait_for_downloads

if [ -n "${MAIN_UNET_NAME}" ] && [ -s "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}" ]; then
  link_model "${MODELS_DIR}/diffusion_models/${MAIN_UNET_NAME}" "${MODELS_DIR}/unet/${MAIN_UNET_NAME}"
fi

link_model "${MODELS_DIR}/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors" \
  "${MODELS_DIR}/text_encoders/gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
link_model "${MODELS_DIR}/vae/LTX2/LTX23_audio_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX/LTX23_audio_vae_bf16.safetensors"
link_model "${MODELS_DIR}/vae/LTX2/LTX23_video_vae_bf16.safetensors" \
  "${MODELS_DIR}/vae/LTX/LTX23_video_vae_bf16.safetensors"
link_model "${MODELS_DIR}/vae/LTX2/taeltx2_3.safetensors" \
  "${MODELS_DIR}/vae/LTX/taeltx2_3.safetensors"
