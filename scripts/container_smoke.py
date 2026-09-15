"""Real CPU ComfyUI server, workflow schemas and MP4; no WAN GPU inference."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request
from models import load_manifest, valid, safe_path
from startup import BUNDLE, comfy_command, prepare, stop


def fetch_post_models(root):
    # Only 25.8 MB of public postprocess weights; never download WAN in CI.
    # root is a TemporaryDirectory removed inside this same Docker RUN layer.
    for asset in load_manifest(BUNDLE / "config/models.json"):
        if asset["id"] not in ("rife", "upscale"):
            continue
        path = safe_path(root / "models", asset["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{asset['file']}"
        import shutil
        with urllib.request.urlopen(url, timeout=90) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        assert valid(path, asset), f"Postprocess weight validation failed: {asset['id']}"


def main():
    with tempfile.TemporaryDirectory(prefix="dasiwa-smoke-") as temp:
        root = Path(temp)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        os.environ.update(PORT=str(port), LISTEN="127.0.0.1", COMFYUI_ARGS="--cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        prepare(root, root / "config")
        fetch_post_models(root)
        from mosaic_smoke import create_fixture
        create_fixture(root / "models/auto_mosaic")
        child = subprocess.Popen(comfy_command(root, root / "config"), cwd=os.environ.get("COMFYUI_APP", "/opt/ComfyUI"))

        def api(path, data=None):
            request = urllib.request.Request(f"http://127.0.0.1:{port}" + path, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        try:
            for _ in range(120):
                if child.poll() is not None:
                    raise RuntimeError("CPU ComfyUI failed to start")
                try:
                    info = api("/object_info")
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("CPU ComfyUI did not start")
            for path in (BUNDLE / "api").glob("*.json"):
                graph = json.loads(path.read_text())
                for key, node in graph.items():
                    schema = info[node["class_type"]]
                    inputs = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
                    assert set(node["inputs"]) <= set(inputs), (path.name, key, "unknown input")
                    assert set(schema["input"].get("required", {})) <= set(node["inputs"]), (path.name, key, "missing input")
                    for name, value in node["inputs"].items():
                        if isinstance(value, list):
                            upstream = info[graph[value[0]]["class_type"]]
                            assert upstream["output"][value[1]] == inputs[name][0], (key, name, "type mismatch")
                        elif isinstance(inputs[name][0], list) and name not in ("unet_name", "clip_name", "vae_name", "image"):
                            assert value in inputs[name][0], (key, name, value)
                print(f"SCHEMA OK: {path.name}", flush=True)
            import av
            for name, rife, scale, loop, expected, fps, width in (
                ("loop-post", True, True, True, 6, 32, 128),
                ("loop-disabled", False, False, True, 3, 16, 64),
                ("i2v-post", True, True, False, 7, 32, 128),
            ):
                prompt = {
                    "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 4, "color": 6724044}},
                    "4": {"class_type": "DaSiWaRIFE2x", "inputs": {"images": ["1", 0], "enabled": rife, "source_fps": 16.0, "loop": loop}},
                    "5": {"class_type": "DaSiWaUpscale2x", "inputs": {"images": ["4", 0], "enabled": scale}},
                    "3": {"class_type": "DaSiWaAutoMosaic", "inputs": {"images": ["5", 0], "coverage_preset": "JUST", "confidence": .95, "iou_threshold": .5, "block_size": 0, "max_gap_frames": 3, "target_classes": "pussy,penis,testicles", "device": "cpu", "trim_last_frame": False}},
                    "2": {"class_type": "DaSiWaSaveMP4", "inputs": {"images": ["3", 0], "fps": ["4", 1], "filename_prefix": "smoke/" + name, "crf": 18, "trim_last_frame": False}},
                }
                result = api("/prompt", {"prompt": prompt, "client_id": "build-smoke"})
                prompt_id = result["prompt_id"]
                for _ in range(120):
                    history = api("/history/" + prompt_id)
                    if prompt_id in history:
                        assert history[prompt_id]["status"]["status_str"] == "success", history
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError("Postprocessed MP4 prompt timed out")
                files = list((root / "output/smoke").glob(name + "_*.mp4"))
                assert len(files) == 1
                with av.open(files[0]) as container:
                    assert float(container.streams.video[0].average_rate) == fps
                    frames = list(container.decode(video=0))
                    assert len(frames) == expected and frames[0].width == width
                print(f"POST SMOKE OK: {name}: {expected} frames, {fps}fps, {width}x{width}", flush=True)
            print("SMOKE PASSED: CPU startup, all schemas, real RIFE/SPAN weights -> random-weight detector -> MP4 encode/decode. ON/OFF FPS + single loop trim verified. Official detector weights and WAN GPU inference NOT RUN.", flush=True)
        finally:
            stop(child)


if __name__ == "__main__":
    main()
