"""Small PID-1 supervisor: status -> real CUDA -> exact weights -> ComfyUI."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

from bootstrap_status import write_status
from gpu_preflight import sanitize
from models import atomic_json, load_manifest, provision, safe_path, valid

BUNDLE = Path(__file__).resolve().parent.parent


def truth(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def comfy_command(root, config):
    extra = shlex.split(os.environ.get("COMFYUI_ARGS", ""))
    # Preserve CUDA device UUIDs and the native dtype. Never set highvram or
    # cache-none globally; both can be counterproductive on 24/32 GB cards.
    return [sys.executable, str(Path(os.environ.get("COMFYUI_APP", "/opt/ComfyUI")) / "main.py"),
            "--listen", os.environ.get("LISTEN", os.environ.get("COMFYUI_HOST", "0.0.0.0")),
            "--port", os.environ.get("PORT", os.environ.get("COMFYUI_PORT", "8188")),
            "--input-directory", str(root / "input"), "--output-directory", str(root / "output"),
            "--user-directory", str(root / "user"), "--temp-directory", str(root / "temp"),
            "--extra-model-paths-config", str(config / "extra-model-paths.yaml"),
            "--database-url", "sqlite:///" + str(root / "user" / "comfyui.db"),
            "--reserve-vram", os.environ.get("COMFYUI_RESERVE_VRAM", "2"),
            "--preview-method", "none", "--use-pytorch-cross-attention", "--disable-auto-launch",
            "--disable-api-nodes", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "DaSiWa-WAN",
            "--max-upload-size", "300", *extra]


def prepare(root, config):
    for folder in ("input", "output", "temp", "user/default/workflows/DaSiWa-WAN", "models/diffusion_models", "models/text_encoders", "models/vae", "models/loras"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    paths = {"wan_workspace": {"base_path": str(root), **{name: "models/" + name for name in ("diffusion_models", "text_encoders", "vae", "loras")}}}
    # JSON is valid YAML, and quotes special characters in mounted paths safely.
    atomic_json(config / "extra-model-paths.yaml", paths)
    destination = root / "user/default/workflows/DaSiWa-WAN"
    for source in (BUNDLE / "workflows").glob("*.json"):
        target = destination / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            backup = config / "workflow-backups" / (str(int(time.time())) + "-" + source.name)
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.replace(backup)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    from PIL import Image
    example = root / "input" / "upload-your-image.png"
    if not example.exists():
        Image.new("RGB", (720, 960), (90, 130, 165)).save(example)


def stop(child):
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    started = time.monotonic()
    root = Path(os.environ.get("WORKSPACE_DIR", "/workspace/wan22")).resolve()
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    status = config / "bootstrap-status.json"
    diagnostic = config / "gpu-diagnostics.json"
    boot_id = uuid.uuid4().hex
    child = server = None
    phases = {}

    def progress(phase, message):
        phases[phase] = round(time.monotonic() - started, 2)
        print(f"BOOT {phase} {phases[phase]:.2f}s: {message}", flush=True)
        write_status(status, {"phase": phase, "message": message, "state": "initializing"})

    def terminated(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        print(f"BUNDLE REVISION: {os.environ.get('BUNDLE_REVISION', 'unknown')}", flush=True)
        print("PROFILE: WAN 2.2 SynthSeduction v9 / native FP8-mixed / no extra speed LoRAs", flush=True)
        for key in ("MAIN_UNET_URL", "MAIN_UNET_NAME", "DOWNLOAD_MODELS_BACKGROUND", "CUDA_NORMALIZE_VISIBLE_DEVICES", "CLI_ARGS"):
            if key in os.environ:
                print(f"Legacy {key} ignored; use the WAN template settings.", flush=True)
        write_status(status, {"boot_id": boot_id, "state": "initializing", "phase": "starting", "message": "WAN 2.2を準備しています"}, reset=True)
        server = subprocess.Popen([sys.executable, str(BUNDLE / "scripts/bootstrap_status.py"), "serve", "--file", str(status), "--diagnostics-file", str(diagnostic), "--host", os.environ.get("LISTEN", "0.0.0.0"), "--port", os.environ.get("PORT", os.environ.get("COMFYUI_PORT", "8188"))])
        prepare(root, config)
        progress("cuda-preflight", "モデル取得前にCUDA実演算を確認しています")
        command = [sys.executable, str(BUNDLE / "scripts/gpu_preflight.py"), "--python", sys.executable, "--diagnostics-file", str(diagnostic), "--status-file", str(status), "--boot-id", boot_id]
        if not truth(os.environ.get("CUDA_PREFLIGHT", "1")):
            command += ["--stack-only"]
        else:
            command += ["--timeout", os.environ.get("CUDA_READY_TIMEOUT", "90"), "--interval", "10"]
        if subprocess.call(command) != 0:
            # Include the decisive errno in normal Pod logs as well as the JSON.
            if diagnostic.exists():
                report = json.loads(diagnostic.read_text())
                for attempt in report.get("attempts", [])[-1:]:
                    print("GPU LOWLEVEL: " + json.dumps(sanitize(attempt.get("evidence", {}).get("lowlevel", {}))), flush=True)
            raise RuntimeError("CUDA preflight failed. Save the diagnostic JSON from port 8188.")
        if server.poll() is not None:
            raise RuntimeError("Startup status server could not bind port 8188")
        manifest = BUNDLE / "config/models.json"
        results = []
        if truth(os.environ.get("DOWNLOAD_MODELS", "1")):
            sys.path.insert(0, str(BUNDLE / "downloader-libs"))
            os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
            os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
            os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
            os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
            results = provision(manifest, root / "models", workers=int(os.environ.get("DOWNLOAD_WORKERS", "3")), connections=int(os.environ.get("ARIA2_CONNECTIONS", "16")), headroom_gb=float(os.environ.get("MODEL_DISK_HEADROOM_GB", "10")), source=os.environ.get("MODEL_SOURCE", "auto"), progress=progress)
        else:
            for asset in load_manifest(manifest):
                if not valid(safe_path(root / "models", asset["path"]), asset):
                    raise RuntimeError(f"DOWNLOAD_MODELS=0 but {asset['id']} is missing or invalid")
        progress("weights-ready", "必須モデル検証完了。ComfyUIを起動しています")
        write_status(status, {"state": "handoff", "phase": "comfyui-starting", "message": "ComfyUIへ切り替えています"})
        stop(server)
        server = None
        child = subprocess.Popen(comfy_command(root, config), cwd=os.environ.get("COMFYUI_APP", "/opt/ComfyUI"))
        port = os.environ.get("PORT", os.environ.get("COMFYUI_PORT", "8188"))
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f"ComfyUI exited during startup ({child.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI did not become healthy within 120 seconds")
        phases["generation-ready"] = round(time.monotonic() - started, 2)
        atomic_json(config / "startup-metrics.json", {"revision": os.environ.get("BUNDLE_REVISION"), "phases_seconds": phases, "downloads": results, "note": "Does not include image pull/unpack or first generation model loading."})
        print(f"GENERATION READY {phases['generation-ready']}s (excludes image pull and first model load)", flush=True)
        raise SystemExit(child.wait())
    except Exception as exc:
        detail = str(sanitize(str(exc)))
        print("BOOT FAILED: " + detail, file=sys.stderr, flush=True)
        stop(child)
        if server is None or server.poll() is not None:
            server = subprocess.Popen([sys.executable, str(BUNDLE / "scripts/bootstrap_status.py"), "serve", "--file", str(status), "--diagnostics-file", str(diagnostic), "--host", os.environ.get("LISTEN", "0.0.0.0"), "--port", os.environ.get("PORT", os.environ.get("COMFYUI_PORT", "8188"))])
        write_status(status, {"state": "failed", "phase": "failed", "message": "WAN起動失敗", "detail": detail}, preserve_failure=True)
        time.sleep(max(0, min(3600, int(os.environ.get("BOOT_FAILURE_HOLD_SECONDS", "900")))))
        raise SystemExit(1)
    finally:
        stop(child)
        stop(server)


if __name__ == "__main__":
    main()
