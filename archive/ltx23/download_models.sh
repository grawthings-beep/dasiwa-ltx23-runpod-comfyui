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
: "${CIVITAI_TOKEN:=}"
: "${HF_TOKEN:=}"

# RunPod leaves the literal "{{ RUNPOD_SECRET_* }}" placeholder in env vars when
# the matching secret was never created. Treat those as unset so downloads do
# not silently authenticate with a broken token.
sanitize_token() {
  local value="$1"
  case "${value}" in
    *"{{"*|*"}}"*|*"RUNPOD_SECRET"*) printf '' ;;
    *) printf '%s' "${value}" ;;
  esac
}
CIVITAI_TOKEN="$(sanitize_token "${CIVITAI_TOKEN}")"
HF_TOKEN="$(sanitize_token "${HF_TOKEN}")"

V39_MAIN_UNET_NAME="LTX2/DaSiWa-LTX23-GoldenLace-v3_fp8.safetensors"
V39_MAIN_UNET_URL="https://civitai.com/api/download/models/2967331?type=Model&format=SafeTensor&size=full&fp=fp8"
V39_MAIN_UNET_SHA256="86E14FD4EAF24AE39D3BB2497E9A86C723888A9172CFFECA31C9730DC2C126E2"

# Public, ungated LTX 2.3 distilled transformer used as a drop-in when no
# Civitai token is available. It is saved under the workflow's expected
# filename so UNETLoader still resolves without any Civitai setup.
PUBLIC_FALLBACK_UNET_URL="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors?download=true"

: "${MAIN_UNET_NAME:=${V39_MAIN_UNET_NAME}}"
: "${MAIN_UNET_URL:=${V39_MAIN_UNET_URL}}"
: "${MAIN_UNET_SHA256:=${V39_MAIN_UNET_SHA256}}"

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

normalize_model_defaults() {
  if [[ "${MAIN_UNET_NAME}" == "SolsticeCoin_v2_fp8_mixed.safetensors" || "${MAIN_UNET_URL}" == *"/2917963"* ]]; then
    echo "Detected legacy Solstice MAIN_UNET settings; using V39 Golden Lace v3 defaults."
    MAIN_UNET_NAME="${V39_MAIN_UNET_NAME}"
    MAIN_UNET_URL="${V39_MAIN_UNET_URL}"
    MAIN_UNET_SHA256="${V39_MAIN_UNET_SHA256}"
  fi
}

apply_civitai_fallback() {
  # When the main UNet is a gated Civitai download but no usable CIVITAI_TOKEN
  # is present, swap to the public Hugging Face LTX 2.3 distilled transformer.
  # MAIN_UNET_NAME is kept unchanged so the file lands at the path the workflow
  # widget expects and ComfyUI's UNETLoader resolves without any Civitai setup.
  if [[ "${MAIN_UNET_URL}" == *"civitai."* && -z "${CIVITAI_TOKEN}" ]]; then
    echo "================================================================"
    echo "CIVITAI_TOKEN is not set, but MAIN_UNET_URL points to a gated"
    echo "Civitai model. Falling back to the public LTX 2.3 distilled"
    echo "transformer so generation works out of the box."
    echo "Saving it as: ${MAIN_UNET_NAME}"
    echo "Set CIVITAI_TOKEN (RunPod secret civitai_token) to use the"
    echo "Golden Lace v3 model from Civitai instead."
    echo "================================================================"
    MAIN_UNET_URL="${PUBLIC_FALLBACK_UNET_URL}"
    MAIN_UNET_SHA256=""
  fi
}

# A failed/gated download often returns a small HTML or JSON error page instead
# of the model. Saving that as a .safetensors makes ComfyUI fail later with a
# confusing load error, so reject it here with a clear message.
looks_like_error_page() {
  local file="$1"
  local size
  local head

  size="$(wc -c < "${file}" 2>/dev/null || echo 0)"
  if [ "${size}" -lt 50000 ]; then
    return 0
  fi

  head="$(head -c 512 "${file}" 2>/dev/null | tr -d '\0')"
  case "${head}" in
    *"<!DOCTYPE"*|*"<!doctype"*|*"<html"*|*"<HTML"*|*'{"error"'*|*'{"message"'*)
      return 0 ;;
    *)
      return 1 ;;
  esac
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

  if ! aria2c "${aria2_args[@]}" "${effective_url}"; then
    echo "aria2 failed for ${safe_url}; retrying with single-stream curl." >&2
    rm -f "${tmp}" "${tmp}.aria2"

    local -a curl_args=(
      --fail
      --location
      --retry 8
      --retry-delay 10
      --retry-all-errors
      --connect-timeout 30
      --output "${tmp}"
    )

    if [[ -n "${HF_TOKEN}" && "${url}" == *"huggingface.co"* ]]; then
      curl_args+=(--header "Authorization: Bearer ${HF_TOKEN}")
    fi

    curl "${curl_args[@]}" "${effective_url}"
  fi

  if looks_like_error_page "${tmp}"; then
    echo "ERROR: ${safe_url} returned an error/HTML page, not a model file." >&2
    echo "       Likely a missing or invalid token (CIVITAI_TOKEN / HF_TOKEN) or" >&2
    echo "       no access to a gated model. Not saving ${dest}." >&2
    rm -f "${tmp}"
    return 1
  fi

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

queue_optional_hf_lora() {
  local url="$1"
  local dest="$2"
  local label="${3:-${dest}}"

  if [[ "${url}" == *"huggingface.co/uwgm/"* && -z "${HF_TOKEN}" ]]; then
    echo "Skipping optional gated Hugging Face LoRA because HF_TOKEN is not set: ${label}"
    return
  fi

  queue_download "${url}" "${dest}"
}

queue_optional_civitai_lora() {
  local url="$1"
  local dest="$2"
  local label="${3:-${dest}}"

  if [[ -z "${CIVITAI_TOKEN}" ]]; then
    echo "Skipping optional Civitai LoRA because CIVITAI_TOKEN is not set: ${label}"
    return
  fi

  queue_download "${url}" "${dest}"
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

normalize_model_defaults
apply_civitai_fallback
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
  queue_optional_hf_lora "https://huggingface.co/Alissonerdx/LTX-LoRAs/resolve/main/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors" \
    "Edit anything IC LoRA"
  queue_optional_civitai_lora "https://civitai.com/api/download/models/2806861?type=Model&format=SafeTensor" \
    "${MODELS_DIR}/loras/civitai/ltx23_phut_hon_civitai_2806861.safetensors" \
    "Civitai LTX 2.3 Phut hon LoRA"
  queue_optional_civitai_lora "https://civitai.com/api/download/models/2911845?type=Model&format=SafeTensor" \
    "${MODELS_DIR}/loras/civitai/smoothmix_animations_ltx_civitai_2911845.safetensors" \
    "Civitai SmoothMix Animations LTX LoRA"
  queue_optional_civitai_lora "https://civitai.com/api/download/models/2849892?type=Model&format=SafeTensor" \
    "${MODELS_DIR}/loras/civitai/civitai_2849892.safetensors" \
    "Civitai LTX LoRA 2849892"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/LTX2.3_reasoning_I2V_V3.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/LTX2.3_reasoning_I2V_V3.safetensors" \
    "Reasoning I2V V3 LoRA"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/latexturn_10Eros_i2v_v0.5.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/latexturn_10Eros_i2v_v0.5.safetensors" \
    "Latexturn 10Eros I2V v0.5 LoRA"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/LTXV23_FOOTJOB_V1.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/LTXV23_FOOTJOB_V1.safetensors" \
    "LTXV23 Footjob V1 LoRA"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/TentacleMotion_10Eros_i2v_v1.0.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/TentacleMotion_10Eros_i2v_v1.0.safetensors" \
    "TentacleMotion 10Eros I2V v1.0 LoRA"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/stomach_bulge_10eros_sulphur_v1.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/stomach_bulge_10eros_sulphur_v1.safetensors" \
    "Stomach bulge 10Eros sulphur v1 LoRA"
  queue_optional_hf_lora "https://huggingface.co/uwgm/nikke-loras/resolve/main/nsfw_riding_backshot_frontshot_ltx23_v1.0.safetensors?download=true" \
    "${MODELS_DIR}/loras/LTX/nsfw_riding_backshot_frontshot_ltx23_v1.0.safetensors" \
    "NSFW riding backshot/frontshot LTX2.3 v1.0 LoRA"
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
