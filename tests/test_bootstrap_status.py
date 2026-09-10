import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_status", ROOT / "scripts" / "bootstrap_status.py"
)
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


class BootstrapStatusTests(unittest.TestCase):
    def test_atomic_updates_preserve_progress_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "status.json"
            STATUS.write_status(path, {"assets_total": 12, "state": "initializing"})
            STATUS.write_status(path, {"assets_completed": 3, "phase": "models"})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["assets_total"], 12)
            self.assertEqual(data["assets_completed"], 3)
            self.assertEqual(data["phase"], "models")
            self.assertIsInstance(data["updated_at"], int)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_page_escapes_status_content_and_polls_handoff(self):
        page = STATUS.render_page(
            {"state": "initializing", "phase": "models", "message": "<unsafe>"}
        )
        self.assertIn("&lt;unsafe&gt;", page)
        self.assertNotIn("<unsafe>", page)
        self.assertIn("/status.json", page)
        self.assertIn("location.reload()", page)
        self.assertIn("準備が終わるまで", page)

    def test_failure_page_stops_animation_and_offers_mobile_download(self):
        page = STATUS.render_page({"state": "failed", "diagnostics_available": True})
        self.assertIn('body class="failed"', page)
        self.assertIn('href="/diagnostics.json"', page)
        self.assertIn("診断ファイルを保存", page)
        self.assertIn("待つだけでは生成は始まりません", page)
        self.assertIn("box-sizing:border-box", page)

    def test_reset_clears_previous_failure_and_asset_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "status.json"
            STATUS.write_status(path, {"state": "failed", "detail": "old error", "assets_total": 30, "boot_id": "old"})
            data = STATUS.write_status(path, {"boot_id": "new"}, reset=True)
            self.assertEqual(data["state"], "initializing")
            self.assertNotIn("assets_total", data)
            self.assertNotIn("detail", data)

    def test_shell_failure_does_not_overwrite_classified_gpu_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "status.json"
            STATUS.write_status(path, {"state": "failed", "phase": "device-access", "detail": "UVM error"})
            data = STATUS.write_status(path, {"state": "failed", "phase": "failed", "detail": "line 170"}, preserve_failure=True)
            self.assertEqual(data["phase"], "device-access")
            self.assertEqual(data["detail"], "UVM error")

    def test_real_http_download_and_stale_boot_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "status.json"
            diagnostic = pathlib.Path(temporary) / "gpu.json"
            STATUS.write_status(path, {"boot_id": "new", "state": "failed"})
            report = {"schema_version": 1, "boot_id": "new", "category": "device-access"}
            diagnostic.write_text(json.dumps(report), encoding="utf-8")
            server = STATUS.ReusableServer(("127.0.0.1", 0), STATUS.make_handler(path, diagnostic))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/diagnostics.json") as response:
                    self.assertEqual(json.load(response), report)
                    self.assertIn("attachment", response.headers["Content-Disposition"])
                    self.assertIn("no-store", response.headers["Cache-Control"])
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                report["boot_id"] = "old"
                diagnostic.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/diagnostics.json")
                self.assertEqual(caught.exception.code, 404)
                with urllib.request.urlopen(base + "/status.json") as response:
                    self.assertFalse(json.load(response)["diagnostics_available"])
                for invalid in ("not json", "[]", "x" * (1024 * 1024 + 1)):
                    diagnostic.write_text(invalid, encoding="utf-8")
                    self.assertIsNone(STATUS.read_diagnostics(diagnostic, {"boot_id": "new"}))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
