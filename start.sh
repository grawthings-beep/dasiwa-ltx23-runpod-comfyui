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

mkdir -p "$(dirname "${COMFYUI_DIR}")"
# Refresh image-managed code on every start. rsync does not delete user-added
# models, inputs, outputs, workflows, or custom nodes from the persistent volume.
rsync -a /opt/ComfyUI/ "${COMFYUI_DIR}/"
for managed_node in /opt/ComfyUI/custom_nodes/*; do
  if [ -d "${managed_node}" ]; then
    node_name="$(basename "${managed_node}")"
    mkdir -p "${COMFYUI_DIR}/custom_nodes/${node_name}"
    rsync -a --delete "${managed_node}/" "${COMFYUI_DIR}/custom_nodes/${node_name}/"
  fi
done

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

COMFYUI_INPUT_DIR="${COMFYUI_DIR}/input" python - <<'PY'
import os
from pathlib import Path

from PIL import Image


input_dir = Path(os.environ["COMFYUI_INPUT_DIR"])
images = {
    "example.png": (0, 0, 0, 255),
    "#Watermark-Darksidewalker-Emblem.png": (255, 255, 255, 0),
    "#audio-mark.png": (255, 255, 255, 0),
}
for name, color in images.items():
    path = input_dir / name
    if not path.exists() or path.stat().st_size == 0:
        Image.new("RGBA", (512, 512), color).save(path)
PY

BUNDLED_WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows/dasiwa-ltx23"
mkdir -p "${BUNDLED_WORKFLOW_DIR}"
cp -f /opt/workflows/*.json "${BUNDLED_WORKFLOW_DIR}/" 2>/dev/null || true

if [ "${DOWNLOAD_MODELS}" = "true" ] || [ "${DOWNLOAD_MODELS}" = "1" ] || [ "${DOWNLOAD_MODELS}" = "yes" ]; then
  /download_models.sh
fi

cd "${COMFYUI_DIR}"
python main.py ${CLI_ARGS}
