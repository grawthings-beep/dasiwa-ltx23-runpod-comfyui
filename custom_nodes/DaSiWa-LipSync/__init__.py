"""WAN stays native. LatentSync runs only in a separate, isolated process."""
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np
import torch
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, io, ui
from comfy.utils import ProgressBar
from . import media


def interrupt():
    mm.throw_exception_if_processing_interrupted()


def models_root():
    return Path(os.environ.get("WORKSPACE_DIR", "/workspace/wan22")).resolve() / "models/lipsync"


def worker(job, folder):
    python = Path("/opt/lipsync-venv/bin/python")
    if os.environ.get("LIPSYNC_PROFILE") != "latentsync-1.6" or not python.is_file():
        raise RuntimeError("Use the lip-sync image and LIPSYNC_PROFILE=latentsync-1.6 (AUDIO_PROFILE=none).")
    job = {**job, "models": str(models_root())}
    spec = Path(folder) / "job.json"
    spec.write_text(json.dumps(job), encoding="utf-8")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "LIPSYNC_MODELS": str(models_root()), "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    log = Path(folder) / "worker.log"
    progress = ProgressBar(100) if job["mode"] == "sync" else None
    last_report = 0
    print(f"LIPSYNC: starting {job['mode']} worker", flush=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([str(python), "/opt/wan22/scripts/lipsync_worker.py", str(spec)],
                                   cwd="/opt/LatentSync", env=env, stdout=output, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                interrupt()
                if progress is not None and time.monotonic() - last_report > 1:
                    output.flush()
                    status = log.read_text(encoding="utf-8", errors="replace")
                    matches = re.findall(r"LIPSYNC_PROGRESS (\d+) (\d+)", status)
                    if matches:
                        done, total = map(int, matches[-1])
                        progress.update_absolute(done, total)
                    last_report = time.monotonic()
                if time.monotonic() - started > 1800:
                    raise TimeoutError("Lip sync exceeded 30 minutes")
                time.sleep(.25)
            if process.returncode:
                detail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-18:])
                raise RuntimeError("Lip sync failed; no unsynced video saved.\n" + detail)
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


def temp_job():
    root = Path(folder_paths.get_temp_directory())
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="dasiwa-lipsync-", dir=root)


class DaSiWaSpeechPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaSpeechPlan", display_name="Speech · Validate Face / Set WAN Length", category="DaSiWa/lipsync",
            inputs=[io.Image.Input("image"), io.Audio.Input("audio")],
            outputs=[io.Image.Output(), io.Audio.Output(), io.Int.Output(display_name="wan_frames")])

    @classmethod
    def execute(cls, image, audio):
        wave, rate = media.waveform(audio)
        length, _ = media.speech_plan(wave.shape[-1], rate)
        if len(image) != 1:
            raise ValueError("Upload one image with one visible face")
        with temp_job() as temp:
            path = Path(temp) / "image.npy"
            np.save(path, (image[0, ..., :3].detach().cpu().clamp(0, 1)*255).round().to(torch.uint8).numpy())
            # Fail on missing weights / undetected anime face BEFORE expensive WAN.
            worker({"mode": "check-face", "image": str(path)}, temp)
        return io.NodeOutput(image, audio, length)


class DaSiWaLipSync(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaLipSync", display_name="LatentSync 1.6 · Original Speech / 25 FPS", category="DaSiWa/lipsync",
            inputs=[io.Image.Input("images"), io.Audio.Input("audio"), io.Float.Input("source_fps", default=32, min=1, max=120),
                    io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=True),
                    io.Int.Input("steps", default=20, min=20, max=50),
                    io.Float.Input("guidance", default=1.5, min=1, max=3, step=.1)],
            outputs=[io.Image.Output(), io.Audio.Output()])

    @classmethod
    def execute(cls, images, audio, source_fps, seed, steps, guidance):
        wave, rate = media.waveform(audio)
        _, count = media.speech_plan(wave.shape[-1], rate)
        if images.ndim != 4 or max(images.shape[1:3]) > 1920:
            raise ValueError("Apply lip sync before SPAN; max source dimension 1920")
        indices = media.frame_indices(len(images), source_fps, count)
        mm.unload_all_models()
        gc.collect(); mm.soft_empty_cache()
        with temp_job() as temp:
            temp = Path(temp)
            path = temp / "frames.npy"
            dest = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(count, *images.shape[1:3], 3))
            for i, index in enumerate(indices):
                interrupt()
                dest[i] = (images[index, ..., :3].detach().cpu().clamp(0, 1)*255).round().to(torch.uint8).numpy()
            dest.flush(); del dest
            media.write_wave(temp / "speech.wav", wave, rate)
            out = temp / "synced.npy"
            worker({"mode": "sync", "frames": str(path), "audio": str(temp / "speech.wav"), "output": str(out),
                    "count": count, "seed": seed, "steps": steps, "guidance": guidance}, temp)
            result = np.load(out, mmap_mode="r", allow_pickle=False)
            if result.shape != (count, *images.shape[1:3], 3) or result.dtype != np.uint8:
                raise RuntimeError("Lip-sync worker output shape/type mismatch")
            output = torch.empty(result.shape, dtype=torch.float32, device="cpu")
            for i in range(count):
                interrupt()
                output[i] = torch.from_numpy(result[i].copy()).float() / 255
            del result
        return io.NodeOutput(output, audio)


class DaSiWaSaveSpeechMP4(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaSaveSpeechMP4", display_name="Save Speech MP4 · 25 FPS / Original Voice", category="DaSiWa/lipsync",
            is_output_node=True, inputs=[io.Image.Input("images"), io.Audio.Input("audio"),
                io.String.Input("filename_prefix", default="DaSiWa/speech")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo])

    @classmethod
    def execute(cls, images, audio, filename_prefix):
        wave, rate = media.waveform(audio)
        _, count = media.speech_plan(wave.shape[-1], rate)
        if len(images) != count:
            raise ValueError("Do not trim or interpolate frames after LatentSync")
        height, width = images.shape[1:3]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), width, height)
        name = f"{filename}_{counter:05}_.mp4"
        metadata = {"prompt": cls.hidden.prompt, **(cls.hidden.extra_pnginfo or {})}
        frames = ((f[..., :3].detach().cpu().clamp(0, 1)*255).round().to(torch.uint8).numpy() for f in images)
        media.save_video(Path(folder)/name, frames, wave, rate, count, width, height, json.dumps(metadata), interrupt)
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [DaSiWaSpeechPlan, DaSiWaLipSync, DaSiWaSaveSpeechMP4]


async def comfy_entrypoint():
    return Extension()
