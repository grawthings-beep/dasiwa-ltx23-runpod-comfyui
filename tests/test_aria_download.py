"""Real aria2 regression coverage; no credentials, models, or internet needed."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import models


HEADER = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
DATA = struct.pack("<Q", len(HEADER)) + HEADER + struct.pack("<f", 1)
ASSET = {"id": "test", "path": "diffusion_models/model.safetensors", "size": len(DATA),
         "sha256": hashlib.sha256(DATA).hexdigest(), "revision": "a" * 40,
         "repo": "test/test", "file": "official-model.safetensors"}


class RecoveryTests(unittest.TestCase):
    def test_missing_size_hash_and_header_failures_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.part"
            self.assertIn("missing", models.validation_error(path, ASSET))
            path.write_bytes(b"short")
            self.assertIn("size mismatch", models.validation_error(path, ASSET))
            path.write_bytes(b"x" * len(DATA))
            self.assertIn("SHA256 mismatch", models.validation_error(path, ASSET))
            broken = dict(ASSET, sha256=models.digest(path))
            self.assertIn("header", models.validation_error(path, broken))

    def test_recovers_old_cdn_filename_before_disk_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / ".staging" / ASSET["sha256"] / "old-CDN-filename.safetensors"
            staged.parent.mkdir(parents=True)
            staged.write_bytes(DATA)
            with mock.patch.object(models, "load_manifest", return_value=[ASSET]), \
                 mock.patch.object(models.shutil, "disk_usage", return_value=mock.Mock(free=0)), \
                 mock.patch.object(models, "resolve_source") as resolve:
                self.assertEqual(models.provision(None, root, headroom_gb=0), [])
                resolve.assert_not_called()
            self.assertFalse(staged.exists())
            self.assertTrue(models.valid(root / ASSET["path"], ASSET))

    def test_corrupt_or_incomplete_staged_file_not_adopted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / ".staging" / ASSET["sha256"] / "remote-name"
            staged.parent.mkdir(parents=True)
            staged.write_bytes(b"x" * len(DATA))
            self.assertFalse(models.recover_staged(root, ASSET))
            self.assertTrue(staged.exists())
            staged.write_bytes(DATA)
            staged.with_name(staged.name + ".aria2").write_bytes(b"incomplete")
            self.assertFalse(models.recover_staged(root, ASSET))
            self.assertFalse((root / ASSET["path"]).exists())

    def test_existing_final_not_overwritten_by_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final = root / ASSET["path"]
            final.parent.mkdir(parents=True)
            final.write_bytes(b"user file")
            staged = root / ".staging" / ASSET["sha256"] / "remote-name"
            staged.parent.mkdir(parents=True)
            staged.write_bytes(DATA)
            self.assertFalse(models.recover_staged(root, ASSET))
            self.assertEqual(final.read_bytes(), b"user file")

    def test_staging_cannot_follow_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            outside = Path(temp) / "outside"
            outside.mkdir()
            (outside / "remote-name").write_bytes(DATA)
            link = root / ".staging" / ASSET["sha256"]
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Creating symlinks requires privileges on this host")
            with self.assertRaisesRegex(ValueError, "escapes"):
                models.recover_staged(root, ASSET)
            self.assertTrue((outside / "remote-name").exists())

    def test_input_file_options_and_injection_guard(self):
        with mock.patch.object(models.shutil, "which", return_value="aria2c"), \
             mock.patch.object(models.subprocess, "run", return_value=mock.Mock(returncode=1)) as run:
            with self.assertRaises(RuntimeError):
                models.aria_download("https://example.test/remote?private=test", Path("model.part"), 16)
            args = run.call_args.args[0]
            text = run.call_args.kwargs["input"]
            self.assertIn("\n  dir=", text)
            self.assertIn("\n  out=model.part\n", text)
            self.assertFalse(any(arg.startswith("--out=") for arg in args))
            self.assertNotIn("private", " ".join(args))
            for url in ("https://example.test/\n  out=evil", "https://example.test/\r", "https://example.test/\t"):
                with self.assertRaises(ValueError):
                    models.aria_download(url, Path("model.part"), 16)


@unittest.skipUnless(shutil.which("aria2c"), "real aria2 required; installed in CI and image")
class RealAriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(DATA)))
                self.send_header("Content-Disposition", 'attachment; filename="server-selected.safetensors"')
                self.end_headers()
                self.wfile.write(DATA)

            def log_message(self, *_args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/remote-model.safetensors?private=test"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def test_old_global_out_is_ignored_reproducing_missing_file(self):
        with tempfile.TemporaryDirectory(prefix="aria legacy ") as temp:
            path = Path(temp) / "model.part"
            result = subprocess.run([shutil.which("aria2c"), "--no-conf=true", "--no-netrc=true",
                                     "--input-file=-", "--dir=" + temp, "--out=" + path.name,
                                     "--console-log-level=error", "--download-result=hide"],
                                    input=self.url + "\n", text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0)
            self.assertFalse(path.exists(), "aria2 ignores global --out with --input-file")
            self.assertTrue(any(p.is_file() and p.read_bytes() == DATA for p in Path(temp).iterdir()))

    def test_real_aria_transfer_verifies_and_installs_exact_file(self):
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_download = mock.Mock(side_effect=AssertionError("HF should not run"))
        sdk.hf_hub_url = mock.Mock(side_effect=AssertionError("HF should not run"))
        with tempfile.TemporaryDirectory(prefix="aria fixed ") as temp, \
             mock.patch.dict(sys.modules, {"huggingface_hub": sdk}):
            root = Path(temp)
            result = models.transfer(ASSET, {"engine": "aria2", "url": self.url}, root, 16)
            self.assertEqual(result["bytes"], len(DATA))
            self.assertEqual((root / ASSET["path"]).read_bytes(), DATA)
            self.assertTrue(models.valid(root / ASSET["path"], ASSET))
            self.assertFalse(any(root.rglob("server-selected.safetensors")))
            self.assertFalse(any(root.rglob("model.part")))


if __name__ == "__main__":
    unittest.main()
