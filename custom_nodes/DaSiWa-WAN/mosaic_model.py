"""Pinned official detector ZIP. Never unpickle unverified downloads."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MODEL_FILENAME = "ntd11_anime_nsfw_segm_v5.pt"
ARCHIVE = {
    "id": "auto-mosaic",
    "civitai_version": 2266294,
    "civitai_file": 2158406,
    "size": 18846815,
    "sha256": "aca92864d30384b8dd7851b32e7ade621a147730bf9710fb4417214e0c61d690",
}
MAX_MODEL_BYTES = 64 * 1024 * 1024


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def marker(path):
    return Path(path).with_suffix(".pt.verified.json")


def verify_model(path):
    path = Path(path)
    try:
        proof = json.loads(marker(path).read_text())
        return (path.is_file() and not path.is_symlink()
                and proof["archive_sha256"] == ARCHIVE["sha256"]
                and 0 < path.stat().st_size == proof["size"] <= MAX_MODEL_BYTES
                and sha256(path) == proof["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def extract_verified(archive, directory):
    """Check publisher hash first, then extract ONLY the expected PT member."""
    archive, directory = Path(archive), Path(directory)
    if archive.stat().st_size != ARCHIVE["size"] or sha256(archive) != ARCHIVE["sha256"]:
        raise RuntimeError("Auto-mosaic archive size/SHA256 mismatch; not installed")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / MODEL_FILENAME
    if final.exists() or final.is_symlink():
        raise RuntimeError("Existing detector is not overwritten; move invalid file aside")
    with zipfile.ZipFile(archive) as package:
        candidates = []
        for member in package.infolist():
            raw_name = member.orig_filename
            name = PurePosixPath(raw_name)
            if (name.is_absolute() or ".." in name.parts or "\\" in raw_name
                    or ":" in raw_name or "\x00" in raw_name or stat.S_ISLNK(member.external_attr >> 16)):
                raise RuntimeError("Unsafe detector ZIP member")
            if name.name == MODEL_FILENAME and not member.is_dir():
                candidates.append(member)
        if len(candidates) != 1 or not 0 < candidates[0].file_size <= MAX_MODEL_BYTES:
            raise RuntimeError("Expected one bounded detector PT file in official ZIP")
        data = package.read(candidates[0])  # ZIP CRC checked by zipfile.
    if not data.startswith(b"PK\x03\x04"):
        raise RuntimeError("Detector is not a PyTorch ZIP checkpoint")
    partial = directory / (MODEL_FILENAME + ".partial")
    if partial.is_symlink() or marker(final).is_symlink():
        raise RuntimeError("Unsafe detector staging path")
    partial.write_bytes(data)
    partial.replace(final)
    proof = {"archive_sha256": ARCHIVE["sha256"], "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
    marker(final).write_text(json.dumps(proof), encoding="utf-8")
    return final


def ensure_model(root, download=None):
    """download(archive_asset, staging_path); None means offline validation."""
    root = Path(root).resolve()
    directory = root / "auto_mosaic"
    if not directory.resolve().is_relative_to(root):
        raise RuntimeError("Detector directory escapes model root")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / MODEL_FILENAME
    if verify_model(final):
        return final
    if final.exists() or final.is_symlink():
        raise RuntimeError("Auto-mosaic detector failed verification; move it aside before retrying")
    archive = directory / "animeNSFWDetection_v50.zip"
    if archive.is_symlink():
        raise RuntimeError("Unsafe detector archive path")
    verified_archive = archive.is_file() and archive.stat().st_size == ARCHIVE["size"] and sha256(archive) == ARCHIVE["sha256"]
    if not verified_archive:
        if download is None:
            raise RuntimeError("Auto-mosaic detector missing. Set DOWNLOAD_MODELS=1 and CIVITAI_API_TOKEN")
        download(ARCHIVE, archive)
    return extract_verified(archive, directory)
