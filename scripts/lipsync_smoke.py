"""Build-time CPU checks of the REAL isolated runtime. Not a GPU quality test."""
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
from urllib.parse import quote
import hashlib

sys.path.insert(0, "/opt/LatentSync")
import numpy as np
import torch
import onnxruntime as ort
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from lipsync_worker import StrictPipeline, ExactAudio, FaceDetector, UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.image_processor import load_fixed_mask
import cv2


def main():
    torch.set_num_threads(2)
    assert np.__version__ == "1.26.4"
    assert torch.__version__.split("+")[0] == "2.10.0"
    assert "CPUExecutionProvider" in ort.get_available_providers()
    call = inspect.getsource(LipsyncPipeline.__call__)
    assert "np.load(video_path" in call and "np.save(video_out_path" in call
    assert "subprocess.run" not in call and "shutil.rmtree" not in call
    assert "read_video(video_path" not in call and "read_audio(audio_path)" not in call
    # Exercise the real audio-conditioned and temporal attention code, including
    # the meta/assign checkpoint loader, without downloading 5GB or claiming CUDA.
    config = OmegaConf.to_container(OmegaConf.load("/opt/LatentSync/configs/unet/stage2_512.yaml").model)
    config.update(block_out_channels=[32, 64], down_block_types=["CrossAttnDownBlock3D", "DownBlock3D"],
                  up_block_types=["UpBlock3D", "CrossAttnUpBlock3D"], layers_per_block=1, sample_size=8)
    original = UNet3DConditionModel.from_config(config).eval()
    state = original.state_dict()
    with torch.device("meta"):
        model = UNet3DConditionModel.from_config(config)
    torch.nn.Module.load_state_dict(model, state, strict=True, assign=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 13, 2, 8, 8), 1, encoder_hidden_states=torch.randn(2, 10, 384)).sample
    assert output.shape == (1, 4, 2, 8, 8) and torch.isfinite(output).all()
    vae = AutoencoderKL(block_out_channels=(32,), down_block_types=("DownEncoderBlock2D",),
                        up_block_types=("UpDecoderBlock2D",), layers_per_block=1, latent_channels=4, norm_num_groups=8)
    pipeline = StrictPipeline(vae, None, model, DDIMScheduler.from_pretrained("/opt/LatentSync/configs"))
    try:
        pipeline.loop_video([None, None], np.zeros((1, 16, 16, 3), dtype=np.uint8))
        raise AssertionError("Must not reverse or repeat short videos")
    except ValueError:
        pass
    mask = load_fixed_mask(512, "/opt/LatentSync/latentsync/utils/mask.png")
    assert mask.shape == (3, 512, 512)
    # Real face detector/landmarks: download ONLY 22MB into a disposable build
    # directory, hash-check them, and execute both ONNX graphs on a demo face.
    manifest = json.loads(Path("/opt/wan22/config/lipsync-models.json").read_text())
    with tempfile.TemporaryDirectory(prefix="lipsync-face-smoke-") as temp:
        root = Path(temp)
        os.environ["LIPSYNC_MODELS"] = str(root)
        for asset in manifest["assets"]:
            if asset.get("format") != "onnx":
                continue
            path = root / asset["path"].removeprefix("lipsync/")
            path.parent.mkdir(parents=True, exist_ok=True)
            # URL is SHA-revision pinned in the manifest, never an untrusted job URL.
            url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{quote(asset['file'])}"
            urllib.request.urlretrieve(url, path)
            assert path.stat().st_size == asset["size"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        detector = FaceDetector("cuda")  # patched ONNX provider is CPU
        capture = cv2.VideoCapture("/opt/LatentSync/assets/demo1_video.mp4")
        ok, bgr = capture.read()
        capture.release()
        assert ok, "Pinned upstream demo missing"
        box, landmarks = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        assert box is not None and landmarks.shape == (106, 2)
    print("LIPSYNC CPU SMOKE PASS: isolated imports, temporal/audio UNet forward, meta load, no-loop guard, real ONNX detection/landmarks. CUDA inference/visual quality NOT tested.")


if __name__ == "__main__":
    main()
