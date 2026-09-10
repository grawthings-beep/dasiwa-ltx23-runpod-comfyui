#!/usr/bin/env bash
set -Eeuo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${COMFYUI_HOST:=0.0.0.0}"
: "${COMFYUI_PORT:=8188}"
: "${COMFYUI_PREVIEW_METHOD:=auto}"
: "${COMFYUI_RESERVE_VRAM:=2}"
: "${DOWNLOAD_MODELS:=true}"
: "${DOWNLOAD_MODELS_BACKGROUND:=true}"

DEFAULT_CLI_ARGS="--listen ${COMFYUI_HOST} --port ${COMFYUI_PORT} --preview-method ${COMFYUI_PREVIEW_METHOD} --enable-cors-header"
if [ -n "${COMFYUI_RESERVE_VRAM}" ]; then
  DEFAULT_CLI_ARGS="${DEFAULT_CLI_ARGS} --reserve-vram ${COMFYUI_RESERVE_VRAM}"
fi
: "${CLI_ARGS:=${DEFAULT_CLI_ARGS}}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ -x /start.sh ]; then
  /start.sh &
fi

sync_comfyui_app() {
  local path
  local -a replace_paths=(
    api_server
    assets
    comfy
    comfy_api
    comfy_extras
    custom_nodes
    script_examples
    tests
    web
  )

  mkdir -p "${COMFYUI_DIR}"

  for path in "${replace_paths[@]}"; do
    rm -rf "${COMFYUI_DIR:?}/${path}"
  done

  rsync -a --delete \
    --exclude=/models/*** \
    --exclude=/input/*** \
    --exclude=/output/*** \
    --exclude=/temp/*** \
    --exclude=/user/*** \
    /opt/ComfyUI/ "${COMFYUI_DIR}/"

  if grep -q "ldm.models.autoencoder" "${COMFYUI_DIR}/comfy/sd.py" \
    && [ ! -f "${COMFYUI_DIR}/comfy/ldm/models/autoencoder.py" ]; then
    echo "ComfyUI sync failed: comfy/ldm/models was not copied." >&2
    exit 1
  fi
}

echo "Syncing ComfyUI application files from image..."
sync_comfyui_app

mkdir -p "${COMFYUI_DIR}/input"
if [ ! -s "${COMFYUI_DIR}/input/placeholder.mp4" ]; then
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i color=c=black:s=512x512:r=24:d=1 \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
    -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac \
    "${COMFYUI_DIR}/input/placeholder.mp4" || true
fi

if [ ! -s "${COMFYUI_DIR}/input/placeholder.mp3" ]; then
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000:d=1 \
    -c:a libmp3lame "${COMFYUI_DIR}/input/placeholder.mp3" || true
fi

create_placeholder_png() {
  local file="$1"
  local size="$2"
  local color="$3"

  if [ ! -s "${file}" ]; then
    ffmpeg -hide_banner -loglevel error -y \
      -f lavfi -i "color=c=${color}:s=${size}" \
      -frames:v 1 -pix_fmt rgba "${file}" || true
  fi
}

create_placeholder_png "${COMFYUI_DIR}/input/example.png" "1024x1024" "black"
create_placeholder_png "${COMFYUI_DIR}/input/#Watermark-Darksidewalker-Emblem.png" "64x64" "black@0.0"
create_placeholder_png "${COMFYUI_DIR}/input/#audio-mark.png" "64x64" "black@0.0"

mkdir -p "${COMFYUI_DIR}/user/default/workflows"
cp -f /opt/workflows/*.json "${COMFYUI_DIR}/user/default/workflows/" 2>/dev/null || true

if is_truthy "${DOWNLOAD_MODELS}"; then
  if is_truthy "${DOWNLOAD_MODELS_BACKGROUND}"; then
    (
      echo "Model downloader started in background."
      if /download_models.sh; then
        echo "Model downloader finished."
      else
        echo "Model downloader failed; ComfyUI is still running." >&2
      fi
    ) &
    echo "$!" > "${COMFYUI_DIR}/download_models.pid"
  else
    /download_models.sh
  fi
fi

cd "${COMFYUI_DIR}"
python main.py ${CLI_ARGS}
