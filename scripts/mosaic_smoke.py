"""CPU detector runtime gate. CI uses RANDOM weights, never claims accuracy."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create_fixture(directory):
    """Only a temporary CPU smoke fixture; not a production detector."""
    from ultralytics import YOLO
    package = Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI")) / "custom_nodes/DaSiWa-WAN"
    assets = module_at("fixture_assets", package / "mosaic_model.py")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / assets.MODEL_FILENAME
    model = YOLO("yolo11s-seg.yaml")  # Local architecture packaged by ultralytics; no network.
    model.model.names.update({0: "nipples", 1: "pussy", 2: "anus", 3: "penis", 4: "cross-section", 5: "x-ray", 6: "testicles"})
    model.save(path)
    assets.marker(path).write_text(json.dumps({"archive_sha256": assets.ARCHIVE["sha256"],
        "sha256": assets.sha256(path), "size": path.stat().st_size}), encoding="utf-8")
    return path


def validate_detector(path):
    import numpy as np
    import torch
    from ultralytics import YOLO
    package = Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI")) / "custom_nodes/DaSiWa-WAN"
    assets = module_at("runtime_assets", package / "mosaic_model.py")
    if not assets.verify_model(path):
        raise RuntimeError("Detector integrity verification failed before loading")
    model = YOLO(str(path), task="segment")
    expected = {"nipples", "pussy", "anus", "penis", "cross-section", "x-ray", "testicles"}
    if set(model.names.values()) != expected:
        raise RuntimeError("Official detector class names differ from the pinned integration")
    with torch.inference_mode():
        model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device="cpu", verbose=False)
    print("MOSAIC MODEL READY: verified official checkpoint, class names and CPU forward", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate-detector", type=Path)
    args = p.parse_args()
    if args.validate_detector:
        validate_detector(args.validate_detector)
        return
    comfy_dir = Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI"))
    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    import numpy as np
    package = types.ModuleType("mosaic_smoke_package")
    package.__path__ = [str(comfy_dir / "custom_nodes/DaSiWa-WAN")]
    sys.modules[package.__name__] = package
    mosaic = module_at(package.__name__ + ".mosaic_nodes", Path(package.__path__[0]) / "mosaic_nodes.py")
    with tempfile.TemporaryDirectory(prefix="mosaic-runtime-") as temp:
        path = create_fixture(Path(temp))
        folder_paths.add_model_folder_path("auto_mosaic", str(path.parent))
        # Exercise the production verified loader and real YOLO11 segmentation forward.
        model = mosaic._load_model(path.name)
        with torch.inference_mode():
            result = model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device="cpu", verbose=False)[0]
        assert result.orig_shape == (64, 64)
        print("SMOKE: random-weight YOLO11s-seg save/load/CPU forward passed; NOT production weights or accuracy")

        image = torch.rand((4, 64, 64, 3), generator=torch.Generator().manual_seed(17))
        mask = torch.zeros((1, 64, 64)); mask[:, 20:40, 20:40] = 1
        prediction = Results(np.zeros((64, 64, 3), dtype=np.uint8), path="synthetic",
            names={0: "pussy"}, boxes=torch.tensor([[20, 20, 40, 40, .9, 0]]), masks=mask)

        class FakeDetector:
            names = {0: "pussy", 1: "anus"}
            def __init__(self): self.calls, self.offloads = [], []
            def predict(self, **kw):
                self.calls.append(kw["device"])
                assert kw["classes"] == [0] and kw["retina_masks"] is True
                if kw["device"] != "cpu": raise torch.cuda.OutOfMemoryError("simulated")
                return [prediction]
            def to(self, device): self.offloads.append(device); return self

        detector = FakeDetector()
        with patch.object(mosaic, "_load_model", return_value=detector), patch.object(mosaic, "_inference_device", return_value="cuda:0"):
            output = mosaic.WanAutoMosaicVideo().apply(image, path.name, "JUST", .3, .5, 10, 3, "pussy", "auto")[0]
        assert detector.calls == ["cuda:0"] + ["cpu"] * 4 and detector.offloads == ["cpu"]
        assert output.shape == image.shape and output.device.type == "cpu"
        torch.testing.assert_close(output[:, :10], image[:, :10], rtol=0, atol=0)
        assert not torch.equal(output[:, 24:36, 24:36], image[:, 24:36, 24:36])
        failed = FakeDetector()
        with patch.object(mosaic, "_load_model", return_value=failed), patch.object(mosaic, "_inference_device", return_value="cuda:0"), patch.object(failed, "predict", side_effect=RuntimeError("simulated failure")):
            try: mosaic.WanAutoMosaicVideo().apply(image, path.name, "JUST", .3, .5, 10, 3, "pussy", "auto")
            except RuntimeError as exc: assert str(exc) == "simulated failure"
            else: raise AssertionError("Detector failure was swallowed")
        assert failed.offloads == ["cpu"]
        print("SMOKE: every-frame masking, unchanged outside contours, simulated GPU OOM retry and fail-closed passed")


if __name__ == "__main__":
    main()
