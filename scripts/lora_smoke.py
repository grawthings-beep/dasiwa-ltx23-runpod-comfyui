"""Actual Comfy model-only LoRA arithmetic on a tiny CPU model, no WAN weights."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock


def main():
    app = Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI"))
    sys.path.insert(0, str(app))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    import safetensors.torch
    import folder_paths
    import comfy.model_patcher

    package = app / "custom_nodes/DaSiWa-WAN"
    spec = importlib.util.spec_from_file_location("lora_pack_smoke", package / "__init__.py", submodule_search_locations=[str(package)])
    pack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pack
    spec.loader.exec_module(pack)
    stack = pack.lora_stack

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = torch.nn.Module()
            self.diffusion_model.linear = torch.nn.Linear(4, 4, bias=False)
            self.model_config = types.SimpleNamespace(unet_config={})
            with torch.no_grad():
                self.diffusion_model.linear.weight.copy_(torch.eye(4))

    with tempfile.TemporaryDirectory(prefix="dasiwa-lora-smoke-") as temp:
        directory = Path(temp)
        records = {}
        up = torch.ones(4, 1)
        down = torch.ones(1, 4)
        for name, stage in (("high-test.safetensors", "high"), ("low-test.safetensors", "low")):
            safetensors.torch.save_file({"diffusion_model.linear.lora_up.weight": up,
                                        "diffusion_model.linear.lora_down.weight": down}, str(directory / name))
            records[name] = {"stage": stage, "size": (directory / name).stat().st_size, "trained_base": "wan2.2-i2v-a14b"}
        def resolve(category, name):
            assert category == "loras"
            return str(directory / name)
        with mock.patch.object(stack, "catalogue", return_value=records), mock.patch.object(folder_paths, "get_full_path_or_raise", side_effect=resolve):
            model = Tiny()
            base = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
            assert stack.apply(base, "high", {}) is base
            first = pack.DaSiWaLoraHigh.execute(base, enabled_1=True, lora_1="high-test.safetensors", strength_1=.5).result[0]
            assert first is not base and not base.patches
            # Another stack on the same model must clone, not accumulate in base.
            second = pack.DaSiWaLoraLow.execute(first, enabled_1=True, lora_1="low-test.safetensors", strength_1=.25).result[0]
            key = "diffusion_model.linear.weight"
            assert len(first.patches[key]) == 1 and len(second.patches[key]) == 2
            second.patch_model(device_to=torch.device("cpu"), patch_weights=True)
            expected = torch.eye(4) + .75 * (up @ down)
            assert torch.allclose(model.diffusion_model.linear.weight, expected)
            sample = torch.arange(4, dtype=torch.float32)[None]
            assert torch.allclose(model.diffusion_model.linear(sample), sample @ expected.T)
            second.unpatch_model(device_to=torch.device("cpu"))
            assert torch.equal(model.diffusion_model.linear.weight, torch.eye(4))
            assert stack.apply(base, "high", {"enabled_1": False, "lora_1": "high-test.safetensors"}) is base
            assert not base.patches
            # A shape mismatch must fail before Comfy's adapter can silently skip it.
            bad = directory / "high-test.safetensors"
            safetensors.torch.save_file({"diffusion_model.linear.lora_up.weight": up,
                                        "diffusion_model.linear.lora_down.weight": torch.ones(1, 7)}, str(bad))
            records[bad.name]["size"] = bad.stat().st_size
            try:
                stack.apply(base, "high", {"enabled_1": True, "lora_1": bad.name})
            except ValueError as error:
                assert "shape mismatch" in str(error)
            else:
                raise AssertionError("Bad LoRA shape was accepted")
    print("LORA SMOKE PASSED: real Comfy clone / two additive patches / CPU forward / unpatch / OFF identity / bad-shape rejection. No WAN GPU or visual compatibility test.", flush=True)


if __name__ == "__main__":
    main()
