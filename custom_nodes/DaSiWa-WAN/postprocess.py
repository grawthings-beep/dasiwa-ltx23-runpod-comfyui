"""Bounded GPU work: one RIFE pair / one upscale frame, no runtime downloads."""
import hashlib
import math
from pathlib import Path
import time

RIFE_NAME = "rife49.pth"
UPSCALE_NAME = "2xNomosUni_span_multijpg.safetensors"
ASSETS = {
    RIFE_NAME: (21345274, "e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533"),
    UPSCALE_NAME: (4461056, "bee2a9c082f2b8f6e7f5db504b36593c24a1a959511f587114c399ca58b9c92c"),
}


def verified_path(folder, name):
    import folder_paths
    path = Path(folder_paths.get_full_path_or_raise(folder, name))
    size, digest = ASSETS[name]
    if path.stat().st_size != size:
        raise RuntimeError(f"Invalid {name} size; restart with DOWNLOAD_MODELS=1")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != digest:
        raise RuntimeError(f"Invalid {name} SHA256; refusing to load")
    return path


def output_spec(count, source_fps, enabled, loop):
    if count < 2 or not math.isfinite(source_fps) or not 1 <= source_fps <= 60:
        raise ValueError("RIFE needs at least 2 frames and source_fps in [1, 60]")
    return ((count - 1) * 2 + 1 if enabled else count) - int(loop), source_fps * (2 if enabled else 1)


def _check_images(images):
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 1:
        raise ValueError("Expected nonempty RGB IMAGE [frames,height,width,3]")


def _middle(model, a, b, device, dtype):
    # Locals are released on return/exception, before a potential CPU retry.
    import torch
    a = a.unsqueeze(0).movedim(-1, 1).to(device=device, dtype=dtype)
    b = b.unsqueeze(0).movedim(-1, 1).to(device=device, dtype=dtype)
    result = model(a, b, timestep=0.5, scale_list=[8, 4, 2, 1],
                   training=False, fastmode=True, ensemble=False)
    if not torch.isfinite(result).all():
        raise RuntimeError("RIFE produced non-finite pixels; refusing to save")
    return result[0].movedim(0, -1).clamp(0, 1).to(device="cpu", dtype=torch.float32)


def interpolate(images, enabled=True, source_fps=16.0, loop=True):
    import torch
    from comfy import model_management as mm
    from comfy.utils import ProgressBar
    _check_images(images)
    count, fps = output_spec(len(images), source_fps, enabled, loop)
    if not enabled:
        return images[:-1] if loop else images, fps
    from . import rife_arch
    path = verified_path("rife", RIFE_NAME)
    started = time.monotonic()
    mm.unload_all_models()
    mm.soft_empty_cache()
    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    output = torch.empty((count, *images.shape[1:]), device="cpu", dtype=torch.float32)
    model = rife_arch.IFNet(arch_ver="4.7")  # 4.9 weights use the 4.7 architecture.
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    progress = ProgressBar(len(images) - 1)
    cpu_retry = False
    try:
        try:
            model.to(device=device, dtype=dtype)
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            cpu_retry = True
        if cpu_retry:
            device, dtype = torch.device("cpu"), torch.float32
            model.to(device=device, dtype=dtype)
            mm.soft_empty_cache()
            print("[dasiwa-post] RIFE CUDA OOM: retrying on CPU (slower)", flush=True)
        with torch.inference_mode():
            for i in range(len(images) - 1):
                mm.throw_exception_if_processing_interrupted()
                output[2 * i].copy_(images[i].detach().to(device="cpu", dtype=torch.float32))
                retry = False
                try:
                    middle = _middle(model, images[i], images[i + 1], device, dtype)
                except torch.OutOfMemoryError:
                    if device.type != "cuda":
                        raise
                    retry = True
                if retry:
                    # Exit the except block first: traceback can retain GPU tensors.
                    device, dtype = torch.device("cpu"), torch.float32
                    rife_arch.backwarp_tenGrid.clear()
                    model.to(device=device, dtype=dtype)
                    mm.soft_empty_cache()
                    print(f"[dasiwa-post] RIFE CUDA OOM at pair {i}: CPU retry (slower)", flush=True)
                    middle = _middle(model, images[i], images[i + 1], device, dtype)
                output[2 * i + 1].copy_(middle)
                del middle
                progress.update(1)
            if not loop:
                output[-1].copy_(images[-1].detach().to(device="cpu", dtype=torch.float32))
    finally:
        rife_arch.backwarp_tenGrid.clear()
        del model
        mm.soft_empty_cache()
    print(f"[dasiwa-post] RIFE {len(images)}->{count} frames {fps:g}fps "
          f"device={device} seconds={time.monotonic() - started:.3f}", flush=True)
    return output, fps


def upscale(images, enabled=True):
    import torch
    from comfy import model_management as mm
    from comfy.utils import ProgressBar
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
    _check_images(images)
    if not enabled:
        return images
    verified_path("upscale_models", UPSCALE_NAME)
    started = time.monotonic()
    # Use pinned ComfyUI's V3 NodeOutput contract and native patcher/tiling/OOM retry.
    model = UpscaleModelLoader.execute(UPSCALE_NAME).result[0]
    if model.scale != 2:
        raise RuntimeError("Expected the pinned 2x SPAN model")
    n, h, w, c = images.shape
    output = torch.empty((n, h * 2, w * 2, c), device="cpu", dtype=torch.float32)
    progress = ProgressBar(n)
    try:
        with torch.inference_mode():
            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                frame = ImageUpscaleWithModel.execute(model, images[i:i+1]).result[0]
                if tuple(frame.shape) != (1, h * 2, w * 2, c) or not torch.isfinite(frame).all():
                    raise RuntimeError("Invalid SPAN output; refusing to save")
                output[i:i+1].copy_(frame.detach().to(device="cpu", dtype=torch.float32))
                del frame
                progress.update(1)
    finally:
        mm.unload_model_and_clones(model.patcher)
        del model
        mm.soft_empty_cache()
    print(f"[dasiwa-post] SPAN {n} frames {w}x{h}->{w*2}x{h*2} "
          f"seconds={time.monotonic() - started:.3f}", flush=True)
    return output
