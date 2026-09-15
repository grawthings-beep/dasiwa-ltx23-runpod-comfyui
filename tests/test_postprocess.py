import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import models
import startup
import workflows

spec = importlib.util.spec_from_file_location("postprocess_test", ROOT / "custom_nodes/DaSiWa-WAN/postprocess.py")
post = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post)


class PostprocessTests(unittest.TestCase):
    def test_loop_duration_with_and_without_rife(self):
        self.assertEqual(post.output_spec(81, 16, True, True), (160, 32))
        self.assertEqual(post.output_spec(81, 16, False, True), (80, 16))
        for n in (2, 17, 81, 161):
            for enabled in (False, True):
                frames, fps = post.output_spec(n, 16, enabled, True)
                self.assertEqual(frames / fps, (n - 1) / 16)

    def test_nonloop_retains_last_frame(self):
        self.assertEqual(post.output_spec(81, 16, True, False), (161, 32))
        self.assertEqual(post.output_spec(81, 16, False, False), (81, 16))

    def test_invalid_timing(self):
        for n, fps in ((0, 16), (1, 16), (81, 0), (81, 61), (81, float("nan")), (81, float("inf"))):
            with self.assertRaises(ValueError):
                post.output_spec(n, fps, True, True)

    def test_exact_post_model_identities(self):
        assets = models.load_manifest(ROOT / "config/models.json")
        small = [a for a in assets if a["id"] in ("rife", "upscale")]
        self.assertEqual(sum(a["size"] for a in small), 25806330)
        for a in small:
            self.assertEqual(post.ASSETS[Path(a["path"]).name], (a["size"], a["sha256"]))

    def test_native_resolution_before_upscale_and_mosaic(self):
        for build in (lambda: workflows.build(False), lambda: workflows.build(True), workflows.build_mosaic):
            _, api = build()
            self.assertEqual(api["17"]["inputs"]["images"], ["13", 0])
            self.assertEqual(api["18"]["inputs"]["images"], ["17", 0])
            self.assertEqual(api["14"]["inputs"]["fps"], ["17", 1])
            self.assertFalse(api["14"]["inputs"]["trim_last_frame"])
            if "16" in api:
                self.assertEqual(api["16"]["inputs"]["images"], ["18", 0])
                self.assertFalse(api["16"]["inputs"]["trim_last_frame"])

    def test_model_paths_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            startup.prepare(root, root / "config")
            mapping = json.loads((root / "config/extra-model-paths.yaml").read_text())["wan_workspace"]
            for name in ("rife", "upscale_models"):
                self.assertEqual(mapping[name], "models/" + name)
                self.assertTrue((root / mapping[name]).is_dir())

    def test_runtime_model_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rife49.pth"
            path.write_bytes(b"original")
            paths = types.ModuleType("folder_paths")
            paths.get_full_path_or_raise = lambda *_: str(path)
            asset = (8, hashlib.sha256(b"original").hexdigest())
            with mock.patch.dict(sys.modules, {"folder_paths": paths}), mock.patch.dict(post.ASSETS, {post.RIFE_NAME: asset}):
                self.assertEqual(post.verified_path("rife", post.RIFE_NAME), path)
                path.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    post.verified_path("rife", post.RIFE_NAME)

    def test_torch_archive_without_unpickling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.part"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", b"not executed by downloader")
                archive.writestr("archive/data/0", b"weights")
            asset = {"size": path.stat().st_size, "sha256": models.digest(path), "format": "torch-state-dict"}
            self.assertTrue(models.valid(path, asset, record=False))
            path.write_bytes(b"<html>login</html>")
            asset.update(size=path.stat().st_size, sha256=models.digest(path))
            self.assertIn("archive", models.validation_error(path, asset, record=False))

    def test_no_runtime_fetch_or_compile(self):
        source = (ROOT / "custom_nodes/DaSiWa-WAN/postprocess.py").read_text()
        for forbidden in ("weights_only=False", "torch.compile(", "urlopen(", "hf_hub_download("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
