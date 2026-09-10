import contextlib
import errno
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_preflight", ROOT / "scripts/gpu_preflight.py")
GPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU)


def response(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def failing_runner(lowlevel, cuda_error="CUDA unknown error"):
    return mock.Mock(side_effect=[
        response("NVIDIA GeForce RTX 5090, GPU-test, 32000 MiB, 580.65.06, 0000:C1:00.0"),
        response(stderr=cuda_error, code=1),
        response(json.dumps(lowlevel)),
    ])


class GpuPreflightTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_all_gpu_image_combinations_reach_real_cuda(self):
        for family in ("ada", "blackwell", ""):
            for cuda in ("12.8", "13.0"):
                for name in ("RTX 4090", "RTX 5090", "RTX A5000"):
                    with self.subTest(family=family, cuda=cuda, gpu=name), mock.patch.dict(
                        os.environ, {"WAN_GPU_FAMILY": family, "EXPECTED_TORCH_CUDA": cuda}
                    ):
                        runner = mock.Mock(side_effect=[response(f"{name}, GPU-1, 24000, 580.65.06"),
                                                        response('{"device":"test"}')])
                        result = GPU.probe_once("python", runner)
                        self.assertTrue(result.ready)
                        self.assertEqual(runner.call_count, 2)
                        self.assertEqual(runner.call_args_list[1].args[0], ["python", "-c", GPU.CUDA_PROBE])

    def test_nvml_failure_does_not_veto_successful_cuda(self):
        for smi in (FileNotFoundError("nvidia-smi"), response(stderr="NVML failed", code=1)):
            runner = mock.Mock(side_effect=[smi, response('{"device":"test"}')])
            self.assertTrue(GPU.probe_once("python", runner).ready)

    def test_stack_error_is_not_called_a_host_failure(self):
        result = GPU.probe_torch_stack("python", mock.Mock(return_value=response(stderr="bad torch", code=1)))
        self.assertEqual(result.category, "runtime-stack")
        self.assertFalse(result.ready)
        self.assertFalse(result.retryable)

    def test_uvm_eio_is_classified_without_redeploy_advice(self):
        runner = failing_runner({"devices": [{"path": "/dev/nvidia-uvm", "open_ok": False, "errno": errno.EIO}],
                                 "driver": {"cuInit": 999}})
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            path = pathlib.Path(temporary) / "gpu.json"
            self.assertFalse(GPU.run_preflight("python", 90, 10, runner, path, boot_id="boot-a"))
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["category"], "device-access")
            self.assertEqual(report["attempt_count"], 1)
            self.assertEqual(report["boot_id"], "boot-a")
            self.assertEqual(report["state"], "failed")
            self.assertNotIn("Terminate", err.getvalue())
            self.assertNotIn("different physical host", err.getvalue())

    def test_missing_devices_can_retry_but_permissions_do_not_spin(self):
        for number, retryable in ((errno.ENOENT, True), (errno.EPERM, False), (errno.EACCES, False)):
            result = GPU.probe_once("python", failing_runner({"devices": [{"open_ok": False, "errno": number}]}))
            self.assertEqual(result.category, "device-access")
            self.assertEqual(result.retryable, retryable)

    def test_driver_library_and_compatibility_are_distinct(self):
        for driver, category in (({"load_error": "missing libcuda"}, "driver-library"),
                                 ({"cuInit": 803}, "driver-compatibility"),
                                 ({"cuInit": 804}, "driver-compatibility"),
                                 ({"cuInit": 999}, "cuda-operation"),
                                 ({"cuInit": 0}, "cuda-operation")):
            result = GPU.probe_once("python", failing_runner({"driver": driver}))
            self.assertEqual(result.category, category)

    def test_oom_is_not_a_uvm_or_host_error(self):
        result = GPU.probe_once("python", failing_runner({}, "torch.OutOfMemoryError: CUDA out of memory"))
        self.assertEqual(result.category, "cuda-memory")

    def test_hidden_visibility_is_preserved_and_explained(self):
        for value in ("", "-1"):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                result = GPU.probe_once("python", failing_runner({}))
                self.assertEqual(result.category, "device-selection")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], value)

    def test_timeout_keeps_partial_driver_evidence(self):
        partial = {"devices": [{"path": "/dev/nvidia-uvm", "open_ok": False, "errno": errno.EIO}]}
        runner = mock.Mock(side_effect=[response("GPU"), response(stderr="CUDA failed", code=1),
                                       subprocess.TimeoutExpired("driver", 15, output=(json.dumps(partial) + "\n").encode())])
        result = GPU.probe_once("python", runner)
        self.assertEqual(result.category, "device-access")
        self.assertTrue(result.evidence["driver_probe"]["timeout"])

    def test_transient_failure_then_ready_updates_report(self):
        failed = GPU.ProbeResult(False, "not yet", True, "cuda-operation", {})
        ready = GPU.ProbeResult(True, "OK", False, "ready", {})
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(GPU, "probe_once", side_effect=[failed, ready]), mock.patch.object(GPU.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            path = pathlib.Path(temporary) / "gpu.json"
            self.assertTrue(GPU.run_preflight("python", 90, 0, diagnostics_path=path, boot_id="b"))
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["state"], "ready")
            self.assertEqual(report["attempt_count"], 2)

    def test_diagnostics_failure_never_turns_bad_cuda_into_success(self):
        result = GPU.ProbeResult(False, "failed", False, "cuda-operation")
        with mock.patch.object(GPU, "probe_once", return_value=result), mock.patch.object(GPU, "write_report", side_effect=OSError("read-only")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(GPU.run_preflight("python", 0, 0, diagnostics_path="/unwritable"))

    def test_secrets_and_auth_urls_are_redacted(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_secret", "CIVITAI_API_TOKEN": "civ_test_secret",
                                         "SSH_PRIVATE_KEY": "private-test", "UNRELATED_VAR": "unrelated"}):
            context = GPU.report_context()
            self.assertNotIn("HF_TOKEN", context)
            self.assertNotIn("UNRELATED_VAR", context)
            safe = GPU.sanitize({"stderr": "hf_test_secret civ_test_secret private-test https://example.org/token/path?q=secret"})
            self.assertNotIn("secret", safe["stderr"])
            self.assertNotIn("https://", safe["stderr"])

    def test_report_is_atomic_and_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "gpu.json"
            GPU.write_report(path, {"boot_id": "old"})
            GPU.write_report(path, {"boot_id": "new"})
            self.assertEqual(json.loads(path.read_text())["boot_id"], "new")
            self.assertEqual(len(list(path.parent.iterdir())), 1)
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_probe_code_compiles_and_exercises_fp16_kernels(self):
        for code in (GPU.CUDA_PROBE, GPU.DRIVER_PROBE, GPU.TORCH_STACK_PROBE):
            compile(code, "<probe>", "exec")
        self.assertIn("matrix @ matrix", GPU.CUDA_PROBE)
        self.assertIn("torch.cuda.synchronize", GPU.CUDA_PROBE)
        self.assertNotIn('os.environ["CUDA_VISIBLE_DEVICES"] =', GPU.CUDA_PROBE)

    def test_driver_probe_runs_without_torch_or_a_gpu(self):
        result = subprocess.run([sys.executable, "-c", GPU.DRIVER_PROBE], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = GPU.last_json(result.stdout)
        self.assertIsInstance(data["devices"], list)
        self.assertIsInstance(data["driver"], dict)

    def test_nonfinite_retry_settings_are_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    GPU.main(["--timeout", value])
                self.assertEqual(caught.exception.code, 2)

    def test_main_saves_stack_failure_and_updates_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path = pathlib.Path(temporary) / "gpu.json"
            status_path = pathlib.Path(temporary) / "status.json"
            status_path.write_text(json.dumps({"boot_id": "current", "state": "initializing"}))
            failed = GPU.ProbeResult(False, "bad wheel", False, "runtime-stack", {"torch_stack": "bad wheel"})
            with mock.patch.object(GPU, "probe_torch_stack", return_value=failed), mock.patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]), contextlib.redirect_stdout(io.StringIO()):
                code = GPU.main(["--diagnostics-file", str(report_path), "--status-file", str(status_path), "--boot-id", "current"])
            self.assertEqual(code, 87)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["category"], "runtime-stack")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "runtime-stack")
            self.assertEqual(status["boot_id"], "current")
            self.assertTrue(status["diagnostics_available"])

    def test_stack_only_never_claims_gpu_success(self):
        good = GPU.ProbeResult(True, "stack OK", False, "stack-only")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(GPU, "probe_torch_stack", return_value=good), mock.patch.object(GPU, "probe_once") as probe, contextlib.redirect_stdout(io.StringIO()):
            path = pathlib.Path(temporary) / "gpu.json"
            self.assertEqual(GPU.main(["--stack-only", "--diagnostics-file", str(path)]), 0)
            probe.assert_not_called()
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["state"], "stack-only")
            self.assertEqual(report["validation_scope"], "no GPU execution")


if __name__ == "__main__":
    unittest.main()
