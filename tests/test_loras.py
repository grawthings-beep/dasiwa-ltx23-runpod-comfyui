import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import models

spec = importlib.util.spec_from_file_location("lora_stack", ROOT / "custom_nodes/DaSiWa-WAN/lora_stack.py")
stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack)


class CatalogueTests(unittest.TestCase):
    def test_profiles_and_exact_sizes(self):
        for profile, count, size in (("none", 0, 0), ("core", 5, 2147456040), ("all", 21, 7932101336)):
            data = models.selected_manifest(ROOT / "config/models.json", ROOT / "config/loras.json", profile)
            self.assertEqual(len(data["assets"]), 6 + count)
            self.assertEqual(sum(a["size"] for a in data["assets"]), 36072811553 + size)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "manifest.json"
                models.atomic_json(path, data)
                self.assertEqual(len(models.load_manifest(path)), 6 + count)
        with self.assertRaisesRegex(ValueError, "LORA_PROFILE"):
            models.selected_manifest(None, None, "ALL-typo")

    def test_catalogue_sides_and_accelerators_absent(self):
        records = stack.catalogue()
        self.assertEqual(len(records), 21)
        self.assertEqual(sum(a["stage"] == "high" for a in records.values()), 10)
        self.assertEqual(sum(a["stage"] == "low" for a in records.values()), 11)
        self.assertEqual(records["wind.safetensors"]["stage"], "low")
        self.assertEqual(sum(a["trained_base"].startswith("wan2.1") for a in records.values()), 5)
        self.assertFalse(any("lightning" in n.lower() or "lightx" in n.lower() for n in records))
        self.assertEqual(stack.choices("high", []), ["None"])
        self.assertEqual(stack.choices("low", ["wind.safetensors", "NSFW-22-H-e8.safetensors", "unknown.safetensors"]), ["None", "wind.safetensors"])

    def test_exact_civitai_file_ids(self):
        assets = [a for a in stack.catalogue().values() if "civitai_file" in a]
        self.assertEqual({a["civitai_file"] for a in assets}, {2099355, 2099364, 2405261, 2405303})

    def test_unpinned_main_only_permitted_for_lora(self):
        asset = models.load_manifest(ROOT / "config/models.json")[0]
        asset["revision"] = "main"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            models.atomic_json(path, {"schema_version": 1, "assets": [asset]})
            with self.assertRaisesRegex(ValueError, "pinned revision"):
                models.load_manifest(path)


class SelectionTests(unittest.TestCase):
    def test_disabled_or_zero_returns_identical_input_without_torch_or_file_io(self):
        model = object()
        for values in ({}, {"enabled_1": False, "lora_1": "missing"}, {"enabled_1": True, "strength_1": 0, "lora_1": "missing"}):
            self.assertIs(stack.apply(model, "high", values), model)

    def test_wrong_side_missing_duplicate_and_nonfinite_rejected(self):
        for values in (
            {"enabled_1": True, "lora_1": "wind.safetensors"},
            {"enabled_1": True, "lora_1": "None"},
            {"enabled_1": True, "lora_1": "../evil"},
            {"enabled_1": True, "strength_1": float("nan")},
            {"enabled_1": True, "strength_1": float("inf")},
            {"enabled_1": True, "strength_1": 11},
            {"enabled_1": True, "lora_1": "NSFW-22-H-e8.safetensors", "enabled_2": True, "lora_2": "NSFW-22-H-e8.safetensors"},
        ):
            with self.assertRaises(ValueError):
                stack.selected("high", values)

    def test_pair_strengths_independent(self):
        high = stack.selected("high", {"enabled_1": True, "lora_1": "NSFW-22-H-e8.safetensors", "strength_1": .6})
        low = stack.selected("low", {"enabled_2": True, "lora_2": "NSFW-22-L-e8.safetensors", "strength_2": .4})
        self.assertEqual(high[0][1], .6)
        self.assertEqual(low[0][1], .4)

    def test_no_matches_and_shape_mismatch_fail(self):
        state = {"weight": types.SimpleNamespace(shape=(4, 3))}
        tensor = lambda shape: types.SimpleNamespace(shape=shape)
        adapter = types.SimpleNamespace(name="lora", weights=(tensor((4, 2)), tensor((2, 3)), None, None, None, None))
        stack.validate_patches({"weight": adapter}, state, "test")
        for patches in ({}, {"missing": adapter}):
            with self.assertRaises(ValueError):
                stack.validate_patches(patches, state, "test")
        for shapes in (((4, 2), (2, 5)), ((5, 2), (2, 3)), ((4, 5), (2, 3))):
            adapter.weights = (tensor(shapes[0]), tensor(shapes[1]), None, None, None, None)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                stack.validate_patches({"weight": adapter}, state, "test")


class SourceTests(unittest.TestCase):
    def test_private_main_frozen_after_identity_check(self):
        asset = next(a for a in stack.catalogue().values() if a.get("revision") == "main")
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_url = mock.Mock(return_value="https://huggingface.co/file")
        sdk.get_hf_file_metadata = mock.Mock(return_value=mock.Mock(size=asset["size"], etag=asset["sha256"], commit_hash="f" * 40))
        with mock.patch.dict(sys.modules, {"huggingface_hub": sdk}), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models.resolve_source(asset), {"engine": "hf", "revision": "f" * 40})
            sdk.get_hf_file_metadata.return_value.commit_hash = None
            with self.assertRaisesRegex(ValueError, "immutable"):
                models.resolve_source(asset)
            sdk.get_hf_file_metadata.return_value.etag = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                models.resolve_source(asset)

    def test_civitai_only_does_not_request_nonexistent_hf_source(self):
        asset = next(a for a in stack.catalogue().values() if "civitai_version" in a)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(models.urllib.request, "build_opener") as network:
            with self.assertRaisesRegex(RuntimeError, "Civitai-only"):
                models.resolve_source(asset, "hf")
            with self.assertRaisesRegex(RuntimeError, "CIVITAI_API_TOKEN"):
                models.resolve_source(asset, "auto")
            network.assert_not_called()

    def test_transfer_uses_resolved_revision_not_main(self):
        asset = copy.deepcopy(next(a for a in stack.catalogue().values() if a.get("revision") == "main"))
        sdk = types.ModuleType("huggingface_hub")
        sdk.hf_hub_url = mock.Mock()
        sdk.hf_hub_download = mock.Mock(return_value="staged.safetensors")
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {"huggingface_hub": sdk}), \
                mock.patch.object(models, "valid", return_value=False), mock.patch.object(models, "validation_error", return_value=None), \
                mock.patch.object(models, "install_verified"):
            models.transfer(asset, {"engine": "hf", "revision": "a" * 40}, temp, 16)
            self.assertEqual(sdk.hf_hub_download.call_args.kwargs["revision"], "a" * 40)
            with self.assertRaisesRegex(ValueError, "immutable"):
                models.transfer(asset, {"engine": "hf"}, temp, 16)


if __name__ == "__main__":
    unittest.main()
