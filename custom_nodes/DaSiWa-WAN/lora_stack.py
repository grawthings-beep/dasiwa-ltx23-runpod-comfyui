"""Three optional model-only LoRAs per stage; no extra node pack or weight cache."""
import json
import math
from pathlib import Path

NONE = "None"
SLOTS = 3


def catalogue():
    path = Path(__file__).with_name("loras.json")
    if not path.is_file():
        path = Path(__file__).resolve().parents[2] / "config/loras.json"
    return {a["path"].split("/", 1)[1]: a for a in json.loads(path.read_text(encoding="utf-8"))["assets"]}


def choices(stage, available):
    records = catalogue()
    return [NONE] + sorted(name for name in available if name in records and records[name]["stage"] == stage)


def selected(stage, values):
    """OFF/zero does not inspect files or patch the input model at all."""
    if stage not in ("high", "low"):
        raise ValueError("Unknown LoRA stage")
    records = catalogue()
    result = []
    seen = set()
    for i in range(1, SLOTS + 1):
        if not values.get(f"enabled_{i}", False):
            continue
        strength = float(values.get(f"strength_{i}", 1.0))
        if not math.isfinite(strength) or not -10 <= strength <= 10:
            raise ValueError(f"LoRA slot {i}: strength must be finite and within -10..10")
        if strength == 0:
            continue
        name = values.get(f"lora_{i}", NONE)
        if name == NONE:
            raise ValueError(f"LoRA slot {i}: select a downloaded {stage.upper()} LoRA or switch the slot OFF")
        if name not in records or records[name]["stage"] != stage:
            raise ValueError(f"LoRA slot {i}: file is not in the {stage.upper()} catalogue")
        if name in seen:
            raise ValueError(f"LoRA slot {i}: duplicate file; use one slot and its strength instead")
        seen.add(name)
        result.append((name, strength, records[name]))
    return result


def validate_patches(patches, state, name):
    """Catch zero matches and ordinary LoRA shape errors before the sampler.

    Uses shapes only, never allocates a full-size LoRA product just to validate.
    Comfy handles other adapter types. This is NOT a visual compatibility test.
    """
    if not patches:
        raise ValueError(f"{name}: no model keys matched; incompatible LoRA")
    for key, adapter in patches.items():
        weight_key = key if isinstance(key, str) else key[0]
        if weight_key not in state:
            raise ValueError(f"{name}: missing model key {weight_key}")
        if not isinstance(key, str) or getattr(adapter, "name", None) != "lora":
            continue
        up, down, _, mid, _, reshape = adapter.weights
        if mid is not None or reshape is not None:
            continue
        target = tuple(state[key].shape)
        if (len(up.shape) < 2 or len(down.shape) < 2
                or math.prod(up.shape[1:]) != down.shape[0]
                or up.shape[0] != target[0]
                or math.prod(down.shape[1:]) != math.prod(target[1:])):
            raise ValueError(f"{name}: LoRA shape mismatch at {key}; not applied")


def apply(model, stage, values):
    entries = selected(stage, values)
    if not entries:
        return model
    import folder_paths
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    current = model
    for name, strength, record in entries:
        path = folder_paths.get_full_path_or_raise("loras", name)
        # The provisioner checks full SHA256 before Comfy starts. This also
        # catches accidental truncation/replacement in a running Pod quickly.
        if Path(path).stat().st_size != record["size"]:
            raise ValueError(f"{name}: unexpected file size; re-provision the LoRA")
        weights, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        converted = comfy.lora_convert.convert_lora(weights)
        keys = comfy.lora.model_lora_keys_unet(current.model, {})
        patches = comfy.lora.load_lora(converted, keys)
        validate_patches(patches, current.model.state_dict(), name)
        # Same clone + add_patches path as Comfy's model-only LoRA loader;
        # no permanent merge, forced dtype, or extra CPU/GPU raw-weight cache.
        patched = current.clone()
        applied = patched.add_patches(patches, strength)
        if len(applied) != len(patches):
            raise ValueError(f"{name}: not all mapped patches were accepted; generation stopped")
        if metadata:
            patched.set_attachments("lora_metadata", metadata)
        experimental = " / WAN2.1 transfer: experimental" if record["trained_base"].startswith("wan2.1") else ""
        print(f"[dasiwa-lora] {stage.upper()} {name} strength={strength:g} mapped_layers={len(applied)}{experimental}", flush=True)
        current = patched
    return current
