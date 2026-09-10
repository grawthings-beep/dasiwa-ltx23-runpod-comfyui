"""Real CPU ComfyUI server, workflow schemas and MP4; no WAN GPU inference."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request
from startup import BUNDLE, comfy_command, prepare, stop


def main():
    with tempfile.TemporaryDirectory(prefix="dasiwa-smoke-") as temp:
        root = Path(temp)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        os.environ.update(PORT=str(port), LISTEN="127.0.0.1", COMFYUI_ARGS="--cpu")
        prepare(root, root / "config")
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
            prompt = {
                "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 4, "color": 6724044}},
                "2": {"class_type": "DaSiWaSaveMP4", "inputs": {"images": ["1", 0], "fps": 16.0, "filename_prefix": "smoke/test", "crf": 18, "trim_last_frame": True}},
            }
            result = api("/prompt", {"prompt": prompt, "client_id": "build-smoke"})
            prompt_id = result["prompt_id"]
            for _ in range(30):
                history = api("/history/" + prompt_id)
                if prompt_id in history:
                    assert history[prompt_id]["status"]["status_str"] == "success", history
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Tiny MP4 prompt timed out")
            import av
            files = list((root / "output").rglob("*.mp4"))
            assert len(files) == 1
            with av.open(files[0]) as container:
                frames = list(container.decode(video=0))
                assert len(frames) == 3 and frames[0].width == 64
            print("SMOKE PASSED: real CPU startup, both workflow schemas, MP4 encode/decode + loop trim. GPU inference NOT RUN.", flush=True)
        finally:
            stop(child)


if __name__ == "__main__":
    main()
