#!/usr/bin/env python3
"""Check actual CUDA; record evidence without guessing host failure."""

import argparse
import errno
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field


TORCH_STACK_PROBE = r'''
import json, os, sys
import torch, torchaudio, torchvision
actual = {
    "torch": torch.__version__, "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__, "torch_cuda": torch.version.cuda,
}
expected = {
    "torch": os.environ.get("EXPECTED_TORCH_VERSION", ""),
    "torchvision": os.environ.get("EXPECTED_TORCHVISION_VERSION", ""),
    "torchaudio": os.environ.get("EXPECTED_TORCHAUDIO_VERSION", ""),
    "torch_cuda": os.environ.get("EXPECTED_TORCH_CUDA", ""),
}
mismatches = {k: {"expected": v, "actual": actual.get(k)}
              for k, v in expected.items() if v and actual.get(k) != v}
print(json.dumps({**actual, "python": sys.executable, "torch_path": torch.__file__,
                  "compiled_arch_flags": getattr(torch._C, "_cuda_getArchFlags", lambda: None)(),
                  "mismatches": mismatches}), flush=True)
if mismatches:
    raise RuntimeError("incompatible pinned torch stack")
'''

CUDA_PROBE = r'''
import json, torch
device = torch.device("cuda:0")
# Exercise kernels, not only NVML/is_available.
value = torch.ones(1, device=device)
value.mul_(2)
matrix = torch.ones((32, 32), device=device, dtype=torch.float16)
product = matrix @ matrix
torch.cuda.synchronize(device)
assert value.item() == 2 and torch.all(product == 32).item(), "CUDA arithmetic mismatch"
properties = torch.cuda.get_device_properties(device)
print(json.dumps({
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "device": properties.name,
    "vram_gib": round(properties.total_memory / (1024 ** 3), 2),
    "compute_capability": list(torch.cuda.get_device_capability(device)),
    "operations": ["allocation", "multiply", "fp16_matmul", "synchronize", "readback"],
}))
'''

# Only after failed CUDA, in a separate timed process. No torch import, driver
# installation, device creation, reset or module reload.
DRIVER_PROBE = r'''
import ctypes, glob, json, os, pathlib, stat
data = {"devices": [], "driver": {}}
for name in ["/dev/nvidiactl", "/dev/nvidia-uvm"] + sorted(glob.glob("/dev/nvidia[0-9]*"))[:16]:
    item = {"path": name}
    try:
        s = os.stat(name)
        item.update({"is_char_device": stat.S_ISCHR(s.st_mode),
                     "major": os.major(s.st_rdev), "minor": os.minor(s.st_rdev)})
        if not item["is_char_device"]:
            item["error"] = "not a character device"
        else:
            fd = os.open(name, os.O_RDWR | os.O_NONBLOCK)
            os.close(fd)
            item["open_ok"] = True
    except OSError as exc:
        item.update({"open_ok": False, "errno": exc.errno, "error": str(exc)})
    data["devices"].append(item)
for key, name in [("kernel_driver", "/proc/driver/nvidia/version"),
                  ("uvm_version", "/sys/module/nvidia_uvm/version")]:
    try:
        data[key] = pathlib.Path(name).read_text()[:1024]
    except OSError:
        data[key] = "unavailable"
try:
    data["registered_devices"] = [s.strip() for s in pathlib.Path("/proc/devices").read_text().splitlines()
                                  if "nvidia" in s.lower()][:20]
except OSError:
    data["registered_devices"] = []
# Preserve device evidence if cuInit hangs or the process crashes.
print(json.dumps(data), flush=True)
try:
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version = ctypes.c_int()
    data["driver"]["version_result"] = lib.cuDriverGetVersion(ctypes.byref(version))
    data["driver"]["api_version"] = version.value
    try:
        data["driver"]["mapped_libraries"] = sorted({s.split()[-1] for s in
            pathlib.Path("/proc/self/maps").read_text().splitlines()
            if any(n in s for n in ("libcuda.so", "libnvidia-ptxjitcompiler", "libnvidia-nvvm"))})[:20]
    except OSError:
        pass
    print(json.dumps(data), flush=True)
    result = lib.cuInit(0)
    data["driver"]["cuInit"] = result
    error_name = ctypes.c_char_p()
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    if lib.cuGetErrorName(result, ctypes.byref(error_name)) == 0 and error_name.value:
        data["driver"]["error_name"] = error_name.value.decode("ascii", errors="replace")
except (OSError, AttributeError) as exc:
    data["driver"]["load_error"] = str(exc)
print(json.dumps(data), flush=True)
'''

ADVICE = {
    "ready": "CUDA実演算が成功しました。動画生成全体の動作確認とは別です。",
    "stack-only": "Python依存関係のみ確認済み。GPU実機テストは未実施です。",
    "runtime-stack": "PyTorch依存関係の読み込み・固定版を確認してください。GPU故障とは判定していません。",
    "device-selection": "CUDA_VISIBLE_DEVICESがGPUを非表示にしています。意図した指定か確認してください。自動変更はしていません。",
    "device-access": "GPUデバイスへのアクセスが失敗しました。診断JSONのerrnoとデバイス情報を確認してください。GPU本体の故障とは断定できません。",
    "driver-library": "libcudaの読み込みを確認してください。コンテナへのドライバ公開やライブラリの混在が候補です。",
    "driver-compatibility": "CUDAとドライバの互換性を確認してください。対応イメージ・ホストの選択が必要な可能性があります。",
    "cuda-memory": "最小CUDAテストでメモリ確保に失敗しました。他のGPU処理や空きVRAMを確認してください。",
    "cuda-operation": "CUDA演算に失敗しました。詳細は診断JSONに保存しています。原因をホスト障害と決めつけていません。",
}


def sanitize(value):
    """Allowlisted evidence only; redact secret values echoed by libraries."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    for key, secret in os.environ.items():
        if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY", key, re.I) and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"https?://[^\s\"'<>]+", "[URL REDACTED]", value)
    return value[:12000]


def last_json(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    for line in reversed((text or "").splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    return {}


def run_command(command, runner, timeout=30):
    try:
        result = runner(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout or "", "stderr": result.stderr or ""}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": None, "timeout": True, "partial": last_json(exc.stdout)}
    except OSError as exc:
        return {"returncode": None, "error": str(exc)}


def result_text(result):
    parts = [str(result[k]).strip() for k in ("stdout", "stderr", "error") if result.get(k)]
    return sanitize("\n".join(parts) or ("probe timed out" if result.get("timeout") else "probe failed"))[-4000:]


@dataclass(frozen=True)
class ProbeResult:
    ready: bool
    diagnostic: str
    retryable: bool = True
    category: str = "cuda-operation"
    evidence: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.ready
        yield self.diagnostic


def probe_torch_stack(python_bin, runner=subprocess.run):
    result = run_command([python_bin, "-c", TORCH_STACK_PROBE], runner)
    ready = result["returncode"] == 0
    return ProbeResult(ready, result_text(result), False, "stack-only" if ready else "runtime-stack",
                       {"torch_stack": result})


def classify_failure(cuda, lowlevel):
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return "device-selection", False
    if "out of memory" in result_text(cuda).lower():
        return "cuda-memory", True
    devices = lowlevel.get("devices", [])
    failed = [d for d in devices if d.get("open_ok") is False or d.get("is_char_device") is False]
    if failed:
        retryable = not any(d.get("errno") in (errno.EIO, errno.EPERM, errno.EACCES) for d in failed)
        return "device-access", retryable
    driver = lowlevel.get("driver", {})
    if driver.get("load_error"):
        return "driver-library", True
    if driver.get("cuInit") in (35, 803, 804):
        return "driver-compatibility", False
    if any(s in result_text(cuda).lower() for s in ("driver version is insufficient", "driver on your system is too old")):
        return "driver-compatibility", False
    return "cuda-operation", True


def probe_once(python_bin, runner=subprocess.run):
    nvidia = run_command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id",
                          "--format=csv,noheader"], runner, timeout=15)
    # NVML failure and the legacy WAN_GPU_FAMILY label never veto real CUDA.
    cuda = run_command([python_bin, "-c", CUDA_PROBE], runner)
    evidence = {"nvidia_smi": nvidia, "cuda_operation": cuda}
    if cuda["returncode"] == 0:
        return ProbeResult(True, result_text(cuda), False, "ready", evidence)
    driver = run_command([python_bin, "-c", DRIVER_PROBE], runner, timeout=15)
    lowlevel = last_json(driver.get("stdout")) or driver.get("partial", {})
    evidence.update({"lowlevel": lowlevel, "driver_probe": driver})
    category, retryable = classify_failure(cuda, lowlevel)
    return ProbeResult(False, result_text(cuda), retryable, category, evidence)


def write_report(path, report):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(sanitize(report), stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def report_context():
    # Never dump os.environ or terminal/auth URLs. Share diagnostic IDs privately.
    return sanitize({k: os.environ.get(k, "<unset>") for k in (
        "RUNPOD_POD_ID", "RUNPOD_DC_ID", "BUNDLE_REVISION", "EXPECTED_TORCH_CUDA",
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF",
    )})


def publish(report, diagnostics_path=None, status_path=None):
    saved = False
    if diagnostics_path:
        try:
            write_report(diagnostics_path, report)
            saved = True
        except OSError as exc:
            print(f"[gpu-preflight] Could not save diagnostics: {sanitize(str(exc))}", file=sys.stderr)
    if status_path:
        from bootstrap_status import write_status
        updates = {"diagnostics_available": saved, "diagnostic_category": report["category"]}
        if report["state"] == "failed":
            updates.update({"state": "failed", "phase": report["category"],
                            "message": "GPU起動チェックに失敗しました", "detail": report["advice"]})
        try:
            write_status(pathlib.Path(status_path), updates)
        except OSError as exc:
            print(f"[gpu-preflight] Could not update status: {sanitize(str(exc))}", file=sys.stderr)


def run_preflight(python_bin, timeout_seconds, interval_seconds, runner=subprocess.run,
                  diagnostics_path=None, status_path=None, boot_id=None, stack=None):
    started = time.monotonic()
    deadline = started + max(0, timeout_seconds)
    attempts = []
    while True:
        result = probe_once(python_bin, runner=runner)
        attempts.append({"number": len(attempts) + 1, "category": result.category,
                         "diagnostic": result.diagnostic, "evidence": sanitize(result.evidence)})
        report = {"schema_version": 1, "boot_id": boot_id, "created_at": int(time.time()),
                  "state": "ready" if result.ready else "checking",
                  "category": result.category, "advice": ADVICE[result.category],
                  "context": report_context(), "stack": sanitize(stack.evidence) if stack else {},
                  "elapsed_seconds": round(time.monotonic() - started, 2),
                  "attempts": attempts[-8:], "attempt_count": len(attempts),
                  "validation_scope": "startup-cuda-only; not full WAN video generation"}
        remaining = deadline - time.monotonic()
        finished = result.ready or not result.retryable or remaining <= 0
        if finished and not result.ready:
            report["state"] = "failed"
        publish(report, diagnostics_path, status_path)
        label = "READY" if result.ready else f"{result.category} attempt={len(attempts)}"
        print(f"[gpu-preflight] {label}\n{result.diagnostic}", flush=True)
        if finished:
            if not result.ready:
                print(f"[gpu-preflight] FAILED [{result.category}] {ADVICE[result.category]}\n"
                      "No model download was started. No GPU reset or visibility change was attempted.", file=sys.stderr)
            return result.ready
        time.sleep(min(max(0.1, interval_seconds), remaining))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stack-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=os.environ.get("CUDA_READY_TIMEOUT", "90"))
    parser.add_argument("--interval", type=float, default=os.environ.get("CUDA_READY_INTERVAL", "10"))
    parser.add_argument("--diagnostics-file")
    parser.add_argument("--status-file")
    parser.add_argument("--boot-id", default=None)
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v >= 0 for v in (args.timeout, args.interval)):
        parser.error("timeout and interval must be finite, non-negative seconds")
    stack = probe_torch_stack(args.python)
    if not stack.ready or args.stack_only:
        report = {"schema_version": 1, "boot_id": args.boot_id, "created_at": int(time.time()),
                  "state": "stack-only" if stack.ready else "failed", "category": stack.category,
                  "advice": ADVICE[stack.category], "context": report_context(),
                  "stack": sanitize(stack.evidence), "validation_scope": "no GPU execution"}
        publish(report, args.diagnostics_file, args.status_file)
        print(f"[gpu-preflight] {stack.category}: {stack.diagnostic}")
        return 0 if stack.ready else 87
    print(f"[gpu-preflight] TORCH STACK READY {stack.diagnostic}", flush=True)
    return 0 if run_preflight(args.python, args.timeout, args.interval,
                             diagnostics_path=args.diagnostics_file, status_path=args.status_file,
                             boot_id=args.boot_id, stack=stack) else 86


if __name__ == "__main__":
    sys.exit(main())
