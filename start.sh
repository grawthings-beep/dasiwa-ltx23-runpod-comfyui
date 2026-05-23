#!/usr/bin/env bash
set -Eeuo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${COMFYUI_HOST:=0.0.0.0}"
: "${COMFYUI_PORT:=8188}"
: "${CLI_ARGS:=--listen ${COMFYUI_HOST} --port ${COMFYUI_PORT} --preview-method auto --enable-cors-header}"
: "${DOWNLOAD_MODELS:=true}"

if [ -x /start.sh ]; then
  /start.sh &
fi

if [ ! -d "${COMFYUI_DIR}/.git" ]; then
  mkdir -p "$(dirname "${COMFYUI_DIR}")"
  rsync -a /opt/ComfyUI/ "${COMFYUI_DIR}/"
fi

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

mkdir -p "${COMFYUI_DIR}/user/default/workflows"
cp -n /opt/workflows/*.json "${COMFYUI_DIR}/user/default/workflows/" 2>/dev/null || true

if [ "${DOWNLOAD_MODELS}" = "true" ] || [ "${DOWNLOAD_MODELS}" = "1" ] || [ "${DOWNLOAD_MODELS}" = "yes" ]; then
  /download_models.sh
fi

cd "${COMFYUI_DIR}"
python main.py ${CLI_ARGS}
