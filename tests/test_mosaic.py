import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_nodes/DaSiWa-WAN"


def load_modules():
    package = types.ModuleType("mosaic_test")
    package.__path__ = [str(PACKAGE)]
    with patch.dict(sys.modules, {"mosaic_test": package, "folder_paths": types.ModuleType("folder_paths")}):
        loaded = []
        for name in ("mosaic_model", "mosaic_nodes"):
            spec = importlib.util.spec_from_file_location("mosaic_test." + name, PACKAGE / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        return loaded


class MosaicTests(unittest.TestCase):
    def setUp(self):
        self.asset, self.node = load_modules()

    def test_fixed_grid_and_default_targets(self):
        data = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        output = self.node._fixed_grid_mosaic(data, 3)
        self.assertTrue(np.all(output[:3, :3] == output[0, 0]))
        self.assertEqual(self.node._selected_class_ids(self.node.MODEL_CLASSES, self.node.DEFAULT_CLASSES), [1, 3, 6])
        self.assertEqual(self.node._resolve_block_size(0, 720, 960), 14)

    def test_circular_gaps_preserve_valid_contours(self):
        a = np.zeros((9, 9), dtype=bool); a[4, 3] = True
        b = np.zeros_like(a); b[4, 5] = True
        result = self.node._fill_short_circular_gaps([None, a, None, b], 1)
        self.assertIs(result[1], a)
        self.assertIs(result[3], b)
        self.assertTrue(result[0][4, 4])
        self.assertTrue(result[2][4, 4])
        self.assertEqual(self.node._fill_short_circular_gaps([None] * 3, 3), [None] * 3)

    def test_just_uses_contour_not_entire_box(self):
        class Array:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
            def __len__(self): return len(self.data)
        mask = np.zeros((1, 64, 64)); mask[0, 28:36, 28:36] = 1
        boxes = Array([[8, 8, 56, 56]]); boxes.xyxy = boxes
        result = types.SimpleNamespace(masks=types.SimpleNamespace(data=Array(mask)), boxes=boxes)
        just = self.node._segmentation_union(result, 64, 64, "JUST")
        wide = self.node._segmentation_union(result, 64, 64, "WIDE")
        self.assertTrue(just[32, 32])
        self.assertFalse(just[12, 32])
        self.assertLess(just.sum(), wide.sum())

    def fixture(self, root, name=None, data=b"PK\x03\x04test weights"):
        path = Path(root) / "fixture.zip"
        with zipfile.ZipFile(path, "w") as package:
            member = zipfile.ZipInfo("fixture")
            # ZipInfo.__init__ normalizes backslashes on Windows; keep the
            # literal hostile archive name for the cross-platform regression.
            member.filename = name or "nested/" + self.asset.MODEL_FILENAME
            package.writestr(member, data)
        return path, {**self.asset.ARCHIVE, "size": path.stat().st_size, "sha256": self.asset.sha256(path)}

    def test_verified_extraction_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, asset = self.fixture(root)
            with patch.object(self.asset, "ARCHIVE", asset):
                final = self.asset.extract_verified(path, Path(root) / "out")
                self.assertTrue(self.asset.verify_model(final))
                final.write_bytes(b"PK\x03\x04tampered data")
                self.assertFalse(self.asset.verify_model(final))
                with self.assertRaisesRegex(RuntimeError, "not overwritten"):
                    self.asset.extract_verified(path, final.parent)

    def test_html_and_archive_hash_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.zip"; path.write_bytes(b"<html>Login required</html>")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                self.asset.extract_verified(path, Path(root) / "out")

    def test_zip_traversal_and_wrong_filename_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("../" + self.asset.MODEL_FILENAME, "C:/" + self.asset.MODEL_FILENAME,
                         "wrong.pt", "a\\" + self.asset.MODEL_FILENAME):
                path, asset = self.fixture(root, name)
                with patch.object(self.asset, "ARCHIVE", asset), self.assertRaises(RuntimeError):
                    self.asset.extract_verified(path, Path(root) / "out")

    def test_missing_offline_is_clear(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "CIVITAI_API_TOKEN"):
                self.asset.ensure_model(root)

    def test_startup_registers_workspace_detector_path(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import startup
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            startup.prepare(root, root / "config")
            config = json.loads((root / "config/extra-model-paths.yaml").read_text())
            self.assertEqual(config["wan_workspace"]["auto_mosaic"], "models/auto_mosaic")

    def test_new_workflow_mosaics_only_generated_frames_and_trims_once(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import workflows
        _, old = workflows.build(True)
        _, new = workflows.build_mosaic()
        for key in old:
            if key != "14": self.assertEqual(old[key], new[key])
        self.assertEqual(new["16"]["inputs"]["images"], ["13", 0])
        self.assertEqual(new["14"]["inputs"]["images"], ["16", 0])
        self.assertTrue(new["16"]["inputs"]["trim_last_frame"])
        self.assertFalse(new["14"]["inputs"]["trim_last_frame"])


if __name__ == "__main__":
    unittest.main()
