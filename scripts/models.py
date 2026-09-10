"""Exact-identity provisioning. No model substitution, no model weights in OCI."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlencode


def token(name):
    value = os.environ.get(name, "").strip()
    return value if value and not any(s in value for s in ("{{", "${")) else None


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe model path")
    target = Path(root).joinpath(*p.parts).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Model path escapes root")
    return target


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assets = data["assets"]
    seen = set()
    for a in assets:
        safe_path(Path.cwd(), a["path"])
        safe_path(Path.cwd(), a["file"])
        if a["id"] in seen or not re.fullmatch(r"[a-z0-9_-]+", a["id"]):
            raise ValueError("Duplicate or invalid asset id")
        seen.add(a["id"])
        if not re.fullmatch(r"[0-9a-f]{64}", a["sha256"]) or not re.fullmatch(r"[0-9a-f]{40}", a["revision"]) or a["size"] <= 0:
            raise ValueError("Models must have pinned revision, size and SHA256")
    if len({a["path"] for a in assets}) != len(assets):
        raise ValueError("Duplicate model destination")
    return sorted(assets, key=lambda a: -a["size"])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt(path):
    return path.with_name("." + path.name + ".verified.json")


def validation_error(path, asset, record=True):
    if not path.is_file():
        return "downloaded file missing at expected path"
    st = path.stat()
    if st.st_size != asset["size"]:
        return f"size mismatch: expected {asset['size']}, got {st.st_size} bytes"
    marker = {"sha256": asset["sha256"], "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        if record and json.loads(receipt(path).read_text()) == marker:
            return None
    except (OSError, ValueError):
        pass
    actual = digest(path)
    if actual != asset["sha256"]:
        return f"SHA256 mismatch: expected {asset['sha256']}, got {actual}"
    # Refuse error pages, even if a manifest was accidentally generated for one.
    try:
        with path.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= header_size <= min(100 * 1024 * 1024, st.st_size - 8):
                return "invalid safetensors header length"
            header = json.loads(f.read(header_size))
            if not isinstance(header, dict) or not any(k != "__metadata__" for k in header):
                return "invalid safetensors tensor metadata"
    except (ValueError, struct.error):
        return "invalid safetensors header encoding"
    if record:
        atomic_json(receipt(path), marker)
    return None


def valid(path, asset, record=True):
    return validation_error(path, asset, record) is None


def install_verified(stage, final, asset):
    final.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(final)  # Same filesystem; no second 14 GB model copy.
    st = final.stat()
    atomic_json(receipt(final), {"sha256": asset["sha256"], "size": st.st_size, "mtime_ns": st.st_mtime_ns})


def recover_staged(root, asset):
    """Reuse complete files left under CDN names by the old aria2 invocation.

    Only search this exact hash's private staging directory. Never replace an
    existing final file or trust a filename alone; verify every recovered byte.
    """
    final = safe_path(root, asset["path"])
    staging_root = safe_path(root, ".staging/" + asset["sha256"])
    if final.exists() or not staging_root.is_dir():
        return False
    for candidate in staging_root.rglob("*"):
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(staging_root):
            continue
        if not candidate.is_file() or candidate.stat().st_size != asset["size"]:
            continue
        if candidate.with_name(candidate.name + ".aria2").exists():
            continue
        print(f"RECOVER {asset['id']}: checking staged file SHA256", flush=True)
        if valid(candidate, asset, record=False):
            install_verified(candidate, final, asset)
            print(f"RECOVERED {asset['id']}: verified staged file; no download needed", flush=True)
            return True
    return False


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result:
            result.remove_header("Authorization")
        return result


def resolve_source(asset, source="auto"):
    """Only official HF/Civitai. HEAD all assets before starting payload transfers."""
    hf_error = None
    if source != "civitai" or not asset.get("civitai_version"):
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        try:
            meta = get_hf_file_metadata(hf_hub_url(asset["repo"], asset["file"], revision=asset["revision"]), token=token("HF_TOKEN") or False, timeout=20)
            if meta.size != asset["size"] or (meta.etag or "").strip('"') != asset["sha256"]:
                raise ValueError("Official HF file identity differs from pinned model")
            return {"engine": "hf"}
        except ValueError:
            raise
        except Exception as exc:
            hf_error = type(exc).__name__  # Never print credential-bearing URLs.
            if source == "hf" or not asset.get("civitai_version"):
                raise RuntimeError(f"{asset['id']}: HF access failed ({hf_error}). Check HF_TOKEN and accept the author's model access conditions.") from None
    key = token("CIVITAI_API_TOKEN") or token("CIVITAI_TOKEN")
    if not key:
        raise RuntimeError(f"{asset['id']}: official HF access unavailable ({hf_error}); set an authorized HF_TOKEN or CIVITAI_API_TOKEN. No weights downloaded.")
    query = {"token": key}
    if asset.get("civitai_file"):
        query["fileId"] = asset["civitai_file"]
    url = f"https://civitai.com/api/download/models/{asset['civitai_version']}?" + urlencode(query)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DaSiWa-WAN-RunPod/1"})
        with urllib.request.build_opener(SafeRedirect()).open(request, timeout=25) as response:
            content_type = response.headers.get("Content-Type", "")
            length = response.headers.get("Content-Length")
            if "text/" in content_type or "json" in content_type or (length and int(length) != asset["size"]):
                raise ValueError("Unexpected Civitai model response")
            return {"engine": "aria2", "url": response.geturl()}
    except Exception as exc:
        raise RuntimeError(f"{asset['id']}: Civitai access failed ({type(exc).__name__}); check the token's model download access.") from None


def aria_download(url, stage, connections):
    # Signed CDN URLs go over stdin, never in logs or the process command line.
    stage = Path(stage).absolute()
    if any(c in str(value) for value in (url, stage) for c in ("\r", "\n", "\t")):
        raise ValueError("Invalid aria2 input value")
    binary = shutil.which("aria2c")
    if not binary:
        raise RuntimeError("aria2c missing from image")
    args = [binary, "--input-file=-", "--continue=true", "--auto-file-renaming=false",
            "--allow-overwrite=true", "--file-allocation=none", "--summary-interval=0",
            "--console-log-level=error", "--download-result=hide", "--max-tries=4",
            "--retry-wait=2", "--connect-timeout=20", "--timeout=60",
            "--max-connection-per-server=" + str(connections), "--split=" + str(connections),
            "--min-split-size=16M", "--no-conf=true", "--no-netrc=true"]
    # With --input-file, a global --out is IGNORED by aria2. These must be
    # indented per-URI options, including when the input file is stdin.
    payload = f"{url}\n  dir={stage.parent}\n  out={stage.name}\n"
    result = subprocess.run(args, input=payload, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"aria2 transfer failed (exit {result.returncode}); credential-bearing output withheld")
    if not stage.is_file():
        raise RuntimeError("aria2 completed but expected output file is missing")


def transfer(asset, plan, root, connections):
    from huggingface_hub import hf_hub_download, hf_hub_url
    started = time.monotonic()
    final = safe_path(root, asset["path"])
    if valid(final, asset):
        return {"id": asset["id"], "engine": "existing", "seconds": 0, "bytes": 0}
    staging_root = Path(root) / ".staging" / asset["sha256"]
    staging_root.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = staging_root / "model.part"
    engine = plan["engine"]
    print(f"DOWNLOAD {asset['id']} engine={engine} size={asset['size']/1e9:.2f} GB", flush=True)
    if engine == "hf":
        try:
            stage = Path(hf_hub_download(asset["repo"], asset["file"], revision=asset["revision"], local_dir=staging_root, token=token("HF_TOKEN") or False))
        except Exception as exc:
            # A valid identity was already checked. Obtain a short-lived official
            # HF redirect for aria2 if Xet transport fails; never use a new model.
            print(f"XET fallback {asset['id']}: {type(exc).__name__}", flush=True)
            headers = {"Authorization": "Bearer " + token("HF_TOKEN")} if token("HF_TOKEN") else {}
            req = urllib.request.Request(hf_hub_url(asset["repo"], asset["file"], revision=asset["revision"]), headers=headers)
            try:
                with urllib.request.build_opener(SafeRedirect()).open(req, timeout=25) as response:
                    url = response.geturl()
                stage = staging_root / "model.part"
                aria_download(url, stage, connections)
            except Exception as error:
                raise RuntimeError(f"{asset['id']}: official HF fallback failed ({type(error).__name__})") from None
            engine = "aria2-hf"
    else:
        aria_download(plan["url"], stage, connections)
    print(f"VERIFY {asset['id']} SHA256", flush=True)
    failure = validation_error(stage, asset, record=False)
    if failure:
        raise RuntimeError(f"{asset['id']}: {failure}; file not installed")
    install_verified(stage, final, asset)
    elapsed = time.monotonic() - started
    print(f"READY {asset['id']} {elapsed:.1f}s (transfer + verification)", flush=True)
    return {"id": asset["id"], "engine": engine, "seconds": round(elapsed, 2), "bytes": asset["size"]}


def provision(manifest, root, *, workers=3, connections=16, headroom_gb=10, source="auto", progress=None):
    if source not in ("auto", "hf", "civitai") or not 1 <= workers <= 4 or not 1 <= connections <= 16 or not math.isfinite(headroom_gb) or headroom_gb < 0:
        raise ValueError("Invalid provisioning settings")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = load_manifest(manifest)
    if progress:
        progress("recovering-staged", "前回取得済みのモデルがあれば検証して再利用します")
    for asset in assets:
        recover_staged(root, asset)
    missing = [a for a in assets if not valid(safe_path(root, a["path"]), a)]
    for a in missing:
        if safe_path(root, a["path"]).exists():
            raise RuntimeError(f"{a['id']}: existing final file is invalid; move it aside before retrying. Not overwritten.")
    # Conservative: don't rely on stale partial files to pass the disk check.
    required = sum(a["size"] for a in missing) + headroom_gb * 1e9
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f"Need {required/1e9:.1f} GB free for missing weights + headroom")
    if progress:
        progress("checking-sources", "モデルの取得権限を確認しています")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        plans = list(pool.map(lambda a: resolve_source(a, source), missing))
    total = sum(a["size"] for a in missing)
    done = 0
    results = []
    if progress:
        progress("downloading", f"必須モデル {total/1e9:.2f} GBを取得・検証しています")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(transfer, a, p, root, connections): a for a, p in zip(missing, plans)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            done += result["bytes"]
            if progress:
                progress("downloading", f"検証完了 {done/1e9:.2f} / {total/1e9:.2f} GB")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        assets = load_manifest(args.manifest)
        print(json.dumps({"files": len(assets), "bytes": sum(a["size"] for a in assets), "assets": assets}, indent=2))
        return
    try:
        results = provision(args.manifest, args.root, workers=int(os.environ.get("DOWNLOAD_WORKERS", "3")), connections=int(os.environ.get("ARIA2_CONNECTIONS", "16")), source=os.environ.get("MODEL_SOURCE", "auto"))
        print(json.dumps(results))
    except Exception as exc:
        # Full tracebacks from SDKs can include signed URLs. Only our sanitized
        # domain errors are suitable for logs.
        print(str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
