"""Isolated inference adapter. No Comfy imports, no model downloads, no WAN changes."""
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, "/opt/LatentSync")
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from safetensors.torch import load_file
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.face_detector import FaceDetector
from lipsync_composite import composite
from models import validation_error

BUNDLE = Path(__file__).resolve().parent.parent


def check_models(root):
    data = json.loads((BUNDLE / "config/lipsync-models.json").read_text())
    for a in data["assets"]:
        path = root / a["path"].removeprefix("lipsync/")
        error = validation_error(path, a)
        if error:
            raise RuntimeError(f"Missing/invalid lip-sync asset: {a['id']} ({error}). Restart with DOWNLOAD_MODELS=1.")


class ExactAudio(Audio2Feature):
    def feature2chunks(self, feature_array, fps):
        # The upstream method appends extra end chunks. Match the output timeline
        # exactly instead, preserving all speech and never looping/reversing video.
        return [self.get_sliced_feature(feature_array, i, fps)[0] for i in range(self.output_count)]


class StrictPipeline(LipsyncPipeline):
    def loop_video(self, whisper_chunks, video_frames):
        if len(whisper_chunks) != len(video_frames):
            raise ValueError("Speech/video frame count mismatch; looping is disabled")
        faces, boxes, matrices = self.affine_transform_video(video_frames)
        return video_frames, faces, boxes, matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        out = np.empty_like(video_frames)
        mask = self.image_processor.mask_image[0].cpu().numpy().astype(np.float32)
        for index, face in enumerate(faces):
            # The upstream pastes the whole affine-warped face. Restrict this to
            # its mouth mask so eyes, hair, clothing and background stay intact.
            out[index] = composite(video_frames[index], face.permute(1, 2, 0).float().cpu().numpy(),
                                   affine_matrices[index].float().cpu().numpy(), boxes[index], mask)
        return out


def run(job):
    root = Path(job["models"]).resolve()
    os.environ["LIPSYNC_MODELS"] = str(root)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    check_models(root)
    if job["mode"] == "check-face":
        frame = np.load(job["image"], allow_pickle=False)
        detector = FaceDetector("cuda")  # ONNX deliberately on CPU, no CUDA context.
        box, _ = detector(frame)
        if box is None:
            raise RuntimeError("Face not detected in the input image. Use one front-facing, unobscured face. Anime detection is not guaranteed.")
        print("LIPSYNC PREFLIGHT: weights present / one face detected; GPU visual quality not checked", flush=True)
        return
    if job["mode"] != "sync":
        raise ValueError("Unknown lip-sync job")
    if not torch.cuda.is_available():
        raise RuntimeError("LatentSync inference requires CUDA")
    frames = np.load(job["frames"], mmap_mode="r", allow_pickle=False)
    if len(frames) != job["count"] or not 13 <= len(frames) <= 150:
        raise ValueError("Expected 0.5..6s at 25fps")
    if Path(job["output"]).exists():
        raise FileExistsError(job["output"])
    print(f"LIPSYNC: {len(frames)} frames / 25fps / 512 face crop / {job['steps']} steps; original voice", flush=True)
    del frames
    config = OmegaConf.load("/opt/LatentSync/configs/unet/stage2_512.yaml")
    torch.manual_seed(job["seed"])
    torch.cuda.manual_seed_all(job["seed"])
    encoder = ExactAudio(str(root / "whisper/tiny.pt"), device="cuda", num_frames=16, audio_feat_length=[2, 2])
    encoder.output_count = job["count"]
    # No from_pretrained URL: every byte was provisioned/verified at boot.
    vae = AutoencoderKL.from_config(json.loads((BUNDLE / "config/lipsync-vae.json").read_text()))
    vae.load_state_dict(load_file(str(root / "vae/diffusion_pytorch_model.safetensors")), strict=True)
    vae.register_to_config(scaling_factor=.18215, shift_factor=0)
    vae.to(dtype=torch.float16)
    vae.enable_slicing()
    # Avoid an extra full model allocation and enforce exact checkpoint keys.
    with torch.device("meta"):
        unet = UNet3DConditionModel.from_config(OmegaConf.to_container(config.model))
    ckpt = torch.load(root / "latentsync_unet.pt", map_location="cpu", weights_only=True, mmap=True)
    torch.nn.Module.load_state_dict(unet, ckpt["state_dict"], strict=True, assign=True)
    del ckpt
    unet.to(dtype=torch.float16)
    pipeline = StrictPipeline(vae, encoder, unet, DDIMScheduler.from_pretrained("/opt/LatentSync/configs")).to("cuda")
    completed = 0
    total = math.ceil(job["count"] / 16) * job["steps"]
    def progress(step, timestep, latents):
        nonlocal completed
        completed += 1
        print(f"LIPSYNC_PROGRESS {completed} {total}", flush=True)
    # No DeepCache, no changed WAN sampler/LoRA strengths.
    pipeline(video_path=job["frames"], audio_path=job["audio"], video_out_path=job["output"],
             num_frames=16, video_fps=25, num_inference_steps=job["steps"], guidance_scale=job["guidance"],
             weight_dtype=torch.float16, width=512, height=512,
             mask_image_path="/opt/LatentSync/latentsync/utils/mask.png", callback=progress)
    print("LIPSYNC COMPLETE: output frames ready; original speech will be muxed by ComfyUI", flush=True)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
