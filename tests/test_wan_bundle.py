import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import models
import workflows
import startup


class ModelTests(unittest.TestCase):
    def test_exact_v9_assets(self):
        assets = models.load_manifest(ROOT / "config/models.json")
        self.assertEqual(len(assets), 4)
        self.assertEqual(sum(a["size"] for a in assets), 36047005223)
        self.assertEqual({a["id"] for a in assets}, {"high", "low", "text", "vae"})
        for a in assets[:2]:
            self.assertIn("Distilled/FP8/v09/", a["file"])
        self.assertNotEqual(assets[0]["sha256"], assets[1]["sha256"])

    def test_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as root:
            for path in ("../escape", "/absolute", "C:/evil", "a\\b"):
                with self.assertRaises(ValueError):
                    models.safe_path(root, path)

    def test_verify_marker_and_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            data = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1)
            path = Path(root) / "test.safetensors"
            path.write_bytes(data)
            asset = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            self.assertTrue(models.valid(path, asset))
            with mock.patch.object(models, "digest", side_effect=AssertionError("should use receipt")):
                self.assertTrue(models.valid(path, asset))
            path.write_bytes(data[:-4] + b"abcd")
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000))
            self.assertFalse(models.valid(path, asset))

    def test_html_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.safetensors"
            path.write_bytes(b"<html>forbidden</html>")
            self.assertFalse(models.valid(path, {"size": path.stat().st_size, "sha256": models.digest(path)}))

    def test_tokens_and_templates(self):
        for value in ("", "{{ RUNPOD_SECRET_HF_TOKEN }}", "${HF_TOKEN}"):
            with mock.patch.dict(os.environ, {"HF_TOKEN": value}):
                self.assertIsNone(models.token("HF_TOKEN"))

    def test_space_fails_before_network(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(models.shutil, "disk_usage", return_value=mock.Mock(free=0)), mock.patch.object(models, "resolve_source") as resolver:
            with self.assertRaisesRegex(RuntimeError, "free"):
                models.provision(ROOT / "config/models.json", root)
            resolver.assert_not_called()

    def test_aria_signed_url_not_in_argv_or_error(self):
        with mock.patch.object(models.shutil, "which", return_value="aria2c"), mock.patch.object(models.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="secret-url")) as run:
            with self.assertRaisesRegex(RuntimeError, "output withheld"):
                models.aria_download("https://cdn.example/model?secret=123", Path("model.part"), 16)
            self.assertNotIn("secret", " ".join(run.call_args.args[0]))
            self.assertIn("secret=123", run.call_args.kwargs["input"])

    def test_hf_identity_mismatch_never_falls_back(self):
        asset = models.load_manifest(ROOT / "config/models.json")[0]
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_url = mock.Mock(return_value="https://huggingface.co/official/file")
        sdk.get_hf_file_metadata = mock.Mock(return_value=mock.Mock(size=asset["size"], etag="different"))
        with mock.patch.dict(sys.modules, {"huggingface_hub": sdk}), mock.patch.object(models.urllib.request, "build_opener") as http:
            with self.assertRaisesRegex(ValueError, "identity"):
                models.resolve_source(asset)
            http.assert_not_called()

    def test_hf_plan_requires_exact_digest(self):
        asset = models.load_manifest(ROOT / "config/models.json")[0]
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_url = mock.Mock(return_value="https://huggingface.co/official/file")
        sdk.get_hf_file_metadata = mock.Mock(return_value=mock.Mock(size=asset["size"], etag=asset["sha256"]))
        with mock.patch.dict(sys.modules, {"huggingface_hub": sdk}), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models.resolve_source(asset), {"engine": "hf"})
            self.assertFalse(sdk.get_hf_file_metadata.call_args.kwargs["token"])

    def test_source_access_fails_before_transfer(self):
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_url = mock.Mock(return_value="https://huggingface.co/official/file")
        sdk.get_hf_file_metadata = mock.Mock(side_effect=OSError("secret-url"))
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"huggingface_hub": sdk}), mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(models.shutil, "disk_usage", return_value=mock.Mock(free=100e9)), mock.patch.object(models, "transfer") as transfer:
            with self.assertRaises(RuntimeError) as failure:
                models.provision(ROOT / "config/models.json", root)
            self.assertNotIn("secret-url", str(failure.exception))
            transfer.assert_not_called()

    def test_nonfinite_headroom_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                models.provision(ROOT / "config/models.json", Path("unused"), headroom_gb=value)


class WorkflowTests(unittest.TestCase):
    def test_generated_files(self):
        workflows.generate(check=True)

    def test_native_precision_and_four_total_steps(self):
        for loop in (False, True):
            ui, api = workflows.build(loop)
            for key in ("2", "3"):
                self.assertEqual(api[key]["inputs"]["weight_dtype"], "default")
            high, low = (api[k]["inputs"] for k in ("11", "12"))
            self.assertEqual((high["steps"], low["steps"]), (4, 4))
            self.assertEqual((high["start_at_step"], high["end_at_step"], low["start_at_step"], low["end_at_step"]), (0, 2, 2, 4))
            self.assertEqual(high["return_with_leftover_noise"], "enable")
            self.assertEqual(low["add_noise"], "disable")
            self.assertEqual(low["latent_image"], ["11", 0])
            self.assertEqual(api["10"]["inputs"]["width"], 720)
            self.assertEqual(api["10"]["inputs"]["height"], 960)
            self.assertFalse(any("Lora" in node["class_type"] for node in api.values()))

    def test_loop_is_conditioned_not_pasted(self):
        _, api = workflows.build(True)
        condition = api["10"]["inputs"]
        self.assertEqual(condition["start_image"], condition["end_image"])
        self.assertTrue(api["14"]["inputs"]["trim_last_frame"])
        self.assertEqual(api["14"]["inputs"]["images"], ["13", 0])

    def test_ui_links_and_nonoverlapping_cards(self):
        for loop in (False, True):
            ui, api = workflows.build(loop)
            by_id = {n["id"]: n for n in ui["nodes"]}
            for number, origin, slot, target, input_slot, typ in ui["links"]:
                self.assertIn(number, by_id[origin]["outputs"][slot]["links"])
                inp = by_id[target]["inputs"][input_slot]
                self.assertEqual(inp["link"], number)
                self.assertEqual(api[str(target)]["inputs"][inp["name"]], [str(origin), slot])
            for i, a in enumerate(ui["nodes"]):
                for b in ui["nodes"][i+1:]:
                    ax, ay = a["pos"]; aw, ah = a["size"]
                    bx, by = b["pos"]; bw, bh = b["size"]
                    self.assertFalse(ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah, (a["id"], b["id"]))


class StartupTests(unittest.TestCase):
    def test_no_runtime_installs_or_copytree(self):
        code = (ROOT / "scripts/startup.py").read_text(encoding="utf-8")
        for forbidden in ("pip install", "rsync", "git pull", "shutil.copytree"):
            self.assertNotIn(forbidden, code)

    def test_quality_command_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            command = startup.comfy_command(Path("/workspace/wan22"), Path("/workspace/wan22/config"))
        self.assertIn("--use-pytorch-cross-attention", command)
        for flag in ("--fast", "--highvram", "--lowvram", "--cache-none", "--fp8_e4m3fn-unet"):
            self.assertNotIn(flag, command)

    def test_prepare_preserves_user_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            startup.prepare(root, root / "config")
            target = next((root / "user/default/workflows/DaSiWa-WAN").glob("*.json"))
            target.write_text("user-edited-workflow")
            (root / "output/my-video.mp4").write_bytes(b"user video")
            startup.prepare(root, root / "config")
            self.assertEqual((root / "output/my-video.mp4").read_bytes(), b"user video")
            backups = list((root / "config/workflow-backups").glob("*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), "user-edited-workflow")


if __name__ == "__main__":
    unittest.main()
