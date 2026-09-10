"""Per-frame contour auto-mosaic, after generation, with bounded GPU use."""

from __future__ import annotations

import pathlib
import threading
import time

import folder_paths
import numpy as np
from .mosaic_model import verify_model


MODEL_FILENAME = "ntd11_anime_nsfw_segm_v5.pt"
MODEL_CLASSES = (
    "nipples",
    "pussy",
    "anus",
    "penis",
    "cross-section",
    "x-ray",
    "testicles",
)
# Keep anal regions visible by default. Users can still opt in by adding
# ``anus`` to the node's target_classes field.
DEFAULT_CLASSES = "pussy,penis,testicles"
COVERAGE_PRESETS = {
    # Ported from the user's WAN bundle. JUST uses
    # only the instance segmentation contour; WIDE/SAFE add an ellipse.
    "JUST": {"pad_ratio": 0.08, "dilate_ratio": 0.04, "ellipse": False},
    "WIDE": {"pad_ratio": 0.35, "dilate_ratio": 0.10, "ellipse": True},
    "SAFE": {"pad_ratio": 0.55, "dilate_ratio": 0.16, "ellipse": True},
}

_MODEL = None
_MODEL_PATH = None
_MODEL_LOCK = threading.Lock()


def _load_model(model_name: str):
    global _MODEL, _MODEL_PATH

    if model_name != MODEL_FILENAME:
        raise ValueError("Only the pinned auto-mosaic detector is supported")
    located = folder_paths.get_full_path("auto_mosaic", model_name)
    model_path = pathlib.Path(located) if located else None
    if model_path is None or not verify_model(model_path):
        raise FileNotFoundError(
            "Auto-mosaic model is missing or unverified. Redeploy the matching "
            "image with DOWNLOAD_MODELS=1, DOWNLOAD_MOSAIC_MODELS=1 and CIVITAI_API_TOKEN."
        )

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != model_path:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "The bundled ultralytics runtime is unavailable. Use the "
                    "matching GHCR image instead of copying only the workflow."
                ) from exc
            _MODEL = YOLO(str(model_path), task="segment")
            _MODEL_PATH = model_path
    return _MODEL


def _inference_device(requested):
    if requested not in ("auto", "cpu"):
        raise ValueError("mosaic device must be auto or cpu")
    if requested == "cpu":
        return "cpu"
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    from comfy import model_management
    # This node executes only after WAN/upscale/RIFE, not alongside sampling.
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    device = model_management.get_torch_device()
    if device.type != "cuda":
        return "cpu"
    free, _total = torch.cuda.mem_get_info(device)
    return str(device) if free >= 2 * 1024 ** 3 else "cpu"


def _selected_class_ids(names, requested: str) -> list[int]:
    if isinstance(names, dict):
        normalized = {int(index): str(name).lower() for index, name in names.items()}
    else:
        normalized = {index: str(name).lower() for index, name in enumerate(names)}

    selected = {
        item.strip().lower()
        for item in str(requested).split(",")
        if item.strip()
    }
    unknown = selected.difference(MODEL_CLASSES)
    if unknown:
        raise ValueError(
            "Unknown auto-mosaic class(es): " + ", ".join(sorted(unknown))
        )
    ids = [index for index, name in normalized.items() if name in selected]
    if not ids:
        raise ValueError("At least one auto-mosaic target class must be selected.")
    return ids


def _expanded_box(box, width: int, height: int, pad_ratio: float):
    """Expand each bbox side by a fraction of that bbox's width/height."""
    x1, y1, x2, y2 = map(float, box)
    pad = max(0.0, float(pad_ratio))
    grow_x = (x2 - x1) * pad
    grow_y = (y2 - y1) * pad
    return (
        max(0, int(np.floor(x1 - grow_x))),
        max(0, int(np.floor(y1 - grow_y))),
        min(width, int(np.ceil(x2 + grow_x))),
        min(height, int(np.ceil(y2 + grow_y))),
    )


def _dilate_mask(mask, radius: int):
    radius = max(0, int(radius))
    if radius == 0 or not mask.any():
        return mask
    import cv2

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _ellipse_mask(height: int, width: int, box):
    import cv2

    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
    axes = (max(1, int(round((x2 - x1) / 2))), max(1, int(round((y2 - y1) / 2))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _segmentation_union(result, height: int, width: int, coverage_preset: str):
    """Convert YOLO instance masks into one full-frame contour mask."""
    preset_name = str(coverage_preset).upper()
    if preset_name not in COVERAGE_PRESETS:
        raise ValueError(f"Unknown mosaic coverage preset: {coverage_preset}")
    preset = COVERAGE_PRESETS[preset_name]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    import cv2

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    union = np.zeros((height, width), dtype=bool)
    for raw_mask, box in zip(masks, boxes):
        if raw_mask.shape != (height, width):
            raw_mask = cv2.resize(
                raw_mask, (width, height), interpolation=cv2.INTER_LINEAR
            )
        instance = raw_mask > 0.5

        # Crop the prototype mask to the
        # detector box before growing it, so stray prototype pixels cannot leak.
        x1 = max(0, min(width, int(np.floor(box[0]))))
        y1 = max(0, min(height, int(np.floor(box[1]))))
        x2 = max(0, min(width, int(np.ceil(box[2]))))
        y2 = max(0, min(height, int(np.ceil(box[3]))))
        clipped = np.zeros_like(instance)
        if x2 > x1 and y2 > y1:
            clipped[y1:y2, x1:x2] = instance[y1:y2, x1:x2]
        instance = clipped

        expanded = _expanded_box(
            box, width, height, float(preset["pad_ratio"])
        )
        if preset["ellipse"]:
            instance |= _ellipse_mask(height, width, expanded)
        radius = max(
            1,
            int(
                round(
                    min(expanded[2] - expanded[0], expanded[3] - expanded[1])
                    * float(preset["dilate_ratio"])
                )
            ),
        )
        union |= _dilate_mask(instance, radius)
    return union if union.any() else None


def _mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _shift_mask(mask, dx: int, dy: int):
    """Translate a boolean mask without wrapping pixels around frame edges."""
    height, width = mask.shape
    output = np.zeros_like(mask)
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(width, width - dx)
    src_y2 = min(height, height - dy)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return output
    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    dst_x2 = src_x2 + dx
    dst_y2 = src_y2 + dy
    output[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return output


def _interpolate_masks(mask0, mask1, alpha: float):
    """Move two endpoint contours to an interpolated center, then union them."""
    center0 = _mask_center(mask0)
    center1 = _mask_center(mask1)
    if center0 is None or center1 is None:
        return np.logical_or(mask0, mask1)
    target_x = center0[0] + (center1[0] - center0[0]) * float(alpha)
    target_y = center0[1] + (center1[1] - center0[1]) * float(alpha)
    shifted0 = _shift_mask(
        mask0, int(round(target_x - center0[0])), int(round(target_y - center0[1]))
    )
    shifted1 = _shift_mask(
        mask1, int(round(target_x - center1[0])), int(round(target_y - center1[1]))
    )
    return np.logical_or(shifted0, shifted1)


def _fill_short_circular_gaps(frame_masks, max_gap_frames: int):
    """Fill only detector misses; never spread a valid mask onto valid frames."""
    output = list(frame_masks)
    frame_count = len(output)
    max_gap = max(0, int(max_gap_frames))
    detected = [index for index, mask in enumerate(output) if mask is not None]
    if max_gap == 0 or len(detected) < 2:
        return output

    for position, left in enumerate(detected):
        right = detected[(position + 1) % len(detected)]
        distance = (right - left) % frame_count
        if distance == 0:
            continue
        gap = distance - 1
        if gap <= 0 or gap > max_gap:
            continue
        left_mask = frame_masks[left]
        right_mask = frame_masks[right]
        for step in range(1, distance):
            index = (left + step) % frame_count
            if output[index] is None:
                output[index] = _interpolate_masks(
                    left_mask, right_mask, step / distance
                )
    return output


def _fixed_grid_mosaic(rgb, block_size):
    """Pixelate on a frame-global grid so moving masks never shift the tiles."""
    block = max(2, int(block_size))
    height, width, channels = rgb.shape
    padded_height = ((height + block - 1) // block) * block
    padded_width = ((width + block - 1) // block) * block
    padded = np.pad(
        rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    reduced = padded.reshape(
        padded_height // block,
        block,
        padded_width // block,
        block,
        channels,
    ).mean(axis=(1, 3), dtype=np.float32)
    return (
        reduced.repeat(block, axis=0)
        .repeat(block, axis=1)[:height, :width]
        .astype(np.uint8)
    )


def _resolve_block_size(block_size, width: int, height: int):
    requested = int(block_size)
    if requested > 0:
        return max(2, requested)
    return max(round(min(width, height) / 50), 10)


class WanAutoMosaicVideo:
    """Segment explicit contours and mosaic an IMAGE batch after interpolation."""

    CATEGORY = "WAN Loop/Post Processing"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mosaicked_images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": ([MODEL_FILENAME],),
                "coverage_preset": (list(COVERAGE_PRESETS),),
                "confidence": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.01},
                ),
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.05, "max": 0.95, "step": 0.01},
                ),
                "block_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 128, "step": 2},
                ),
                "max_gap_frames": (
                    "INT",
                    {"default": 3, "min": 0, "max": 24, "step": 1},
                ),
                "target_classes": (
                    "STRING",
                    {"default": DEFAULT_CLASSES, "multiline": False},
                ),
                "device": (["auto", "cpu"], {"default": "auto"}),
            }
        }

    def apply(
        self,
        images,
        model_name,
        coverage_preset,
        confidence,
        iou_threshold,
        block_size,
        max_gap_frames,
        target_classes,
        device="cpu",
    ):
        import torch
        from comfy import model_management

        started = time.perf_counter()

        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("Auto mosaic expects IMAGE shaped [frames, H, W, C].")

        model = _load_model(model_name)
        class_ids = _selected_class_ids(model.names, target_classes)
        _frame_count, height, width, _channels = images.shape
        frame_masks = []
        inference_device = _inference_device(device)
        print(f"[wan-post] mosaic detection device={inference_device} frames={len(images)}")

        # Inspect EVERY frame, including interpolated ones. Never trade away
        # coverage by detecting only keyframes as a performance shortcut.
        try:
            for frame in images:
                model_management.throw_exception_if_processing_interrupted()
                rgb = (
                    frame[..., :3]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .clamp(0.0, 1.0)
                    .numpy()
                )
                bgr = np.ascontiguousarray(
                    (rgb[:, :, ::-1] * 255.0).round().astype(np.uint8)
                )
                options = dict(source=bgr, imgsz=640, conf=float(confidence),
                               iou=float(iou_threshold), classes=class_ids, max_det=24,
                               retina_masks=True, half=False, verbose=False)
                try:
                    result = model.predict(**options, device=inference_device)[0]
                except torch.cuda.OutOfMemoryError:
                    if inference_device == "cpu":
                        raise
                    print("[wan-post] mosaic GPU OOM; retrying this and remaining frames on CPU")
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    inference_device = "cpu"
                    result = model.predict(**options, device="cpu")[0]
                frame_masks.append(
                    _segmentation_union(
                        result, int(height), int(width), str(coverage_preset)
                    )
                )
                del result

        finally:
            # Release VRAM even on cancellation or a non-OOM detector error.
            # Errors must propagate; never emit an unprocessed video on failure.
            if inference_device != "cpu":
                model.to("cpu")
                torch.cuda.empty_cache()
        detected = time.perf_counter()

        masks = _fill_short_circular_gaps(frame_masks, int(max_gap_frames))
        resolved_block = _resolve_block_size(block_size, int(width), int(height))

        output = torch.empty_like(images, device="cpu")
        for index, frame in enumerate(images):
            model_management.throw_exception_if_processing_interrupted()
            rgb = (
                frame.detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp(0.0, 1.0)
                .numpy()
            )
            mask = masks[index]
            if mask is not None and mask.any():
                uint8_rgb = (rgb[..., :3] * 255.0).round().astype(np.uint8)
                pixelated = _fixed_grid_mosaic(uint8_rgb, resolved_block)
                rgb = rgb.copy()
                rgb[..., :3][mask] = pixelated[mask].astype(np.float32) / 255.0
            output[index] = torch.from_numpy(rgb).to(dtype=images.dtype)

        ended = time.perf_counter()
        print(f"[wan-post] mosaic frames={len(images)} device={inference_device} "
              f"detect_seconds={detected-started:.3f} apply_seconds={ended-detected:.3f} "
              f"seconds={ended-started:.3f}")
        return (output,)


NODE_CLASS_MAPPINGS = {"WanAutoMosaicVideo": WanAutoMosaicVideo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAutoMosaicVideo": "WAN Auto Mosaic JUST Segmentation (Auto GPU / CPU)"
}
