import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import models
import startup
import lipsync_workflow
import workflows
from lipsync_composite import composite
from patch_latentsync import replace_once

MEDIA = ROOT / "custom_nodes/DaSiWa-LipSync/media.py"
if not MEDIA.exists():
    MEDIA = Path("/opt/ComfyUI/custom_nodes/DaSiWa-LipSync/media.py")
spec = importlib.util.spec_from_file_location("speech_media_test", MEDIA)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)

ENV = {"LIPSYNC_PROFILE": "latentsync-1.6", "LIPSYNC_RUNTIME": "latentsync-1.6", "AUDIO_PROFILE": "none"}


class SpeechConfigTests(unittest.TestCase):
    def test_opt_in_assets_and_runtime(self):
        assets = models.load_manifest(ROOT / "config/lipsync-models.json")
        self.assertEqual(len(assets), 5)
        self.assertLess(sum(a["size"] for a in assets), 5_600_000_000)
        base = models.selected_manifest(ROOT / "config/models.json", ROOT / "config/loras.json", "none")
        with mock.patch.dict(os.environ, {"LIPSYNC_PROFILE": "none", "AUDIO_PROFILE": "none"}):
            self.assertIs(startup.add_lipsync_assets(base), base)
            self.assertNotIn("DaSiWa-LipSync", startup.comfy_command(ROOT, ROOT))
        with mock.patch.dict(os.environ, ENV):
            self.assertEqual(len(startup.add_lipsync_assets(base)["assets"]), 11)
            self.assertIn("DaSiWa-LipSync", startup.comfy_command(ROOT, ROOT))
        with mock.patch.dict(os.environ, {**ENV, "LIPSYNC_RUNTIME": ""}):
            with self.assertRaises(RuntimeError): startup.lipsync_enabled()
        with mock.patch.dict(os.environ, {**ENV, "AUDIO_PROFILE": "dragonleap-v4", "AUDIO_RUNTIME": "dragonleap-v4"}):
            with self.assertRaises(ValueError): startup.lipsync_enabled()

    def test_separate_install_and_backup(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, ENV):
            root = Path(temp)
            startup.prepare(root, root / "config")
            self.assertEqual(len(list((root / "user/default/workflows/DaSiWa-WAN").glob("*.json"))), 3)
            path = root / "user/default/workflows/DaSiWa-LipSync" / (lipsync_workflow.NAME + ".json")
            self.assertTrue(path.exists())
            path.write_text("user edit")
            startup.prepare(root, root / "config")
            self.assertTrue(any(p.read_text() == "user edit" for p in (root / "config/workflow-backups").glob("*.json")))

    def test_graph_layout_and_unchanged_wan(self):
        ui, api = lipsync_workflow.build()
        _, base = workflows.build(False)
        for key in ("2", "3", "4", "5", "6", "9", "11", "12", "19", "20"):
            self.assertEqual(api[key], base[key])
        self.assertEqual(api["10"]["inputs"]["length"], ["32", 2])
        self.assertEqual(api["18"]["inputs"]["images"], ["33", 0])
        self.assertEqual(api["35"]["inputs"]["images"], ["18", 0])
        self.assertFalse(api["35"]["inputs"]["trim_last_frame"])
        self.assertEqual(api["14"]["inputs"]["audio"], ["33", 1])
        nodes = {n["id"]: n for n in ui["nodes"]}
        for link, src, slot, dst, inp, dtype in ui["links"]:
            self.assertIn(link, nodes[src]["outputs"][slot]["links"])
            self.assertEqual(nodes[dst]["inputs"][inp]["link"], link)
            self.assertEqual(api[str(dst)]["inputs"][nodes[dst]["inputs"][inp]["name"]], [str(src), slot])
        for index, a in enumerate(ui["nodes"]):
            for b in ui["nodes"][index+1:]:
                ax, ay = a["pos"]; aw, ah = a["size"]
                bx, by = b["pos"]; bw, bh = b["size"]
                self.assertFalse(ax < bx+bw and ax+aw > bx and ay < by+bh and ay+ah > by, (a["id"], b["id"]))
        lipsync_workflow.generate(check=True)

    def test_upstream_patch_fail_closed(self):
        self.assertEqual(replace_once("abc", "b", "x"), "axc")
        for text in ("ac", "abbc"):
            with self.assertRaises(ValueError): replace_once(text, "b", "x")

    def test_onnx_pinned_bytes_and_tamper_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detector.onnx"
            data = b"pinned synthetic ONNX fixture"
            asset = {"format": "onnx", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            path.write_bytes(data)
            self.assertTrue(models.valid(path, asset))
            self.assertTrue(models.receipt(path).exists())
            path.write_bytes(b"X" + data[1:])
            self.assertIn("SHA256", models.validation_error(path, asset, record=False))


class SpeechMediaTests(unittest.TestCase):
    def test_duration_and_sampling_no_loop(self):
        for rate in (8000, 22050, 44100, 48000, 192000):
            for samples in (rate//2, rate, 5*rate+1, 6*rate):
                length, count = media.speech_plan(samples, rate)
                self.assertEqual((length-1) % 4, 0)
                self.assertGreaterEqual(count/25, samples/rate)
                self.assertLess(count/25-samples/rate, .040001)
                for fps, n in ((16, length), (32, (length-1)*2+1)):
                    indices = media.frame_indices(n, fps, count)
                    self.assertTrue(np.all(np.diff(indices) >= 0))
                    self.assertLess(indices[-1], n)
        for samples, rate in ((0, 48000), (48000*7, 48000), (100, 0)):
            with self.assertRaises(ValueError): media.speech_plan(samples, rate)
        with self.assertRaisesRegex(ValueError, "shorter"):
            media.frame_indices(5, 32, 125)

    def test_wave_validation(self):
        good = np.ones((1, 2, 48000), np.float32) * .1
        wave, rate = media.waveform({"waveform": good, "sample_rate": 48000})
        np.testing.assert_array_equal(wave, good[0])
        for value in (good*0, good*np.nan, good*20, good[0], np.ones((2, 2, 48000))):
            with self.assertRaises(ValueError): media.waveform({"waveform": value, "sample_rate": 48000})

    def test_float_pcm_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            wave = np.random.default_rng(2).uniform(-.5, .5, (2, 22050)).astype(np.float32)
            path = Path(temp) / "speech.wav"
            media.write_wave(path, wave, 22050)
            with av.open(str(path)) as source:
                decoded = [f.to_ndarray().reshape(-1, 2).T for f in source.decode(audio=0)]
                np.testing.assert_array_equal(np.concatenate(decoded, axis=1), wave)

    def test_save_av_duration_and_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            for rate, channels in ((22050, 1), (44100, 2), (48000, 2)):
                samples = rate + 137
                wave = np.tile(.1*np.sin(np.arange(samples)*2*np.pi*440/rate), (channels, 1)).astype(np.float32)
                _, count = media.speech_plan(samples, rate)
                path = Path(temp) / f"{rate}.mp4"
                frames = lambda: (np.full((64, 96, 3), i*4, dtype=np.uint8) for i in range(count))
                media.save_video(path, frames(), wave, rate, count, 96, 64)
                with av.open(str(path)) as out:
                    self.assertEqual(out.streams.video[0].average_rate, 25)
                    self.assertEqual(len(list(out.decode(video=0))), count)
                with av.open(str(path)) as out:
                    stream = out.streams.audio[0]
                    self.assertAlmostEqual(float(stream.duration*stream.time_base), count/25, delta=.002)
                    decoded = np.concatenate([f.to_ndarray() for f in out.decode(audio=0)], axis=1)
                    self.assertEqual(decoded.shape[0], channels)
                    expected = .1*np.sin(np.arange(48000)*2*np.pi*440/48000)
                    self.assertGreater(np.corrcoef(decoded[0, :48000], expected)[0, 1], .99)
                with self.assertRaises(FileExistsError):
                    media.save_video(path, frames(), wave, rate, count, 96, 64)

    def test_cancel_and_wrong_count_leave_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.mp4"
            wave = np.ones((1, 48000), np.float32)*.1
            def cancel(): raise InterruptedError("cancel")
            with self.assertRaises(InterruptedError):
                media.save_video(path, [], wave, 48000, 25, 96, 64, interrupt=cancel)
            with self.assertRaises(ValueError):
                media.save_video(path, [np.zeros((64, 96, 3), np.uint8)], wave, 48000, 25, 96, 64)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_mouth_composite_preserves_outside(self):
        original = np.full((100, 100, 3), 37, np.uint8)
        generated = np.ones((100, 100, 3), np.float32)
        fixed_mask = np.ones((100, 100), np.float32)
        fixed_mask[50:70, 30:70] = 0
        output = composite(original, generated, [[1, 0, 0], [0, 1, 0]], [0, 0, 100, 100], fixed_mask)
        np.testing.assert_array_equal(output[:45], original[:45])
        np.testing.assert_array_equal(output[75:], original[75:])
        self.assertEqual(output[60, 50, 0], 255)
        shifted = composite(original, generated, [[1, 0, -10], [0, 1, 0]], [0, 0, 100, 100], fixed_mask)
        self.assertEqual(shifted[60, 80, 0], output[60, 70, 0])


if __name__ == "__main__":
    unittest.main()
