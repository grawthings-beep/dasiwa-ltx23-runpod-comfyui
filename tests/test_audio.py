import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from fractions import Fraction

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import startup
import models
import audio_workflow

MEDIA = ROOT / "custom_nodes/DaSiWa-LTX-Audio/media.py"
if not MEDIA.exists():
    MEDIA = Path("/opt/ComfyUI/custom_nodes/DaSiWa-LTX-Audio/media.py")
spec = importlib.util.spec_from_file_location("audio_media_test", MEDIA)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def make_video(path, fps=32, seconds=5, vfr=False):
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 96, 128, "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        if vfr:
            stream.codec_context.time_base = Fraction(1, fps * 2)
            stream.time_base = Fraction(1, fps * 2)
        for i in range(round(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.full((128, 96, 3), i % 200, dtype=np.uint8), format="rgb24")
            # Alternating presentation spacing, same 5s duration.
            frame.pts = i * 2 + (i % 2 if vfr else 0)
            frame.time_base = Fraction(1, fps * 2)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def packets(path):
    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        return [(hashlib.sha256(bytes(p)).hexdigest(), p.pts * p.time_base, p.dts * p.time_base, p.duration * p.time_base)
                for p in source.demux(stream) if p.size]


class AudioConfigTests(unittest.TestCase):
    def test_exact_assets(self):
        assets = models.load_manifest(ROOT / "config/audio-models.json")
        self.assertEqual(len(assets), 5)
        self.assertEqual(sum(a["size"] for a in assets), 44976750982)
        self.assertFalse(any(a["path"].startswith("loras/") for a in assets))
        self.assertEqual(assets[0]["sha256"], "be48b4cc7bc7d38e8fc4b9ef4216b641f58027717fe8cd0724f6f4b83089fdd1")
        base = models.selected_manifest(ROOT / "config/models.json", ROOT / "config/loras.json", "none")
        with mock.patch.dict(os.environ, {"AUDIO_PROFILE": "none"}):
            self.assertIs(startup.add_audio_assets(base), base)
            self.assertNotIn("DaSiWa-LTX-Audio", startup.comfy_command(ROOT, ROOT))
        with mock.patch.dict(os.environ, {"AUDIO_PROFILE": "dragonleap-v4", "AUDIO_RUNTIME": "dragonleap-v4"}):
            selected = startup.add_audio_assets(base)
            self.assertEqual(len(selected["assets"]), 11)
            self.assertIn("DaSiWa-LTX-Audio", startup.comfy_command(ROOT, ROOT))
        with mock.patch.dict(os.environ, {"AUDIO_PROFILE": "dragonleap-v4", "AUDIO_RUNTIME": ""}):
            with self.assertRaisesRegex(RuntimeError, "cu130"):
                startup.add_audio_assets(base)

    def test_install_separate_workflow_and_backup(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"AUDIO_PROFILE": "dragonleap-v4", "AUDIO_RUNTIME": "dragonleap-v4"}):
            root = Path(temp)
            startup.prepare(root, root / "config")
            self.assertEqual(len(list((root / "user/default/workflows/DaSiWa-WAN").glob("*.json"))), 3)
            path = root / "user/default/workflows/DaSiWa-Audio" / (audio_workflow.NAME + ".json")
            self.assertTrue(path.is_file())
            path.write_text("user edit")
            startup.prepare(root, root / "config")
            backup = list((root / "config/workflow-backups").glob("*.json"))
            self.assertEqual(len(backup), 1)
            self.assertEqual(backup[0].read_text(), "user edit")

    def test_graph_and_layout(self):
        ui, api = audio_workflow.build()
        self.assertEqual(api["3"]["inputs"]["video"], ["1", 0])
        self.assertEqual(api["2"]["inputs"]["video"], ["1", 0])
        for index, a in enumerate(ui["nodes"]):
            for b in ui["nodes"][index+1:]:
                ax, ay = a["pos"]; aw, ah = a["size"]
                bx, by = b["pos"]; bw, bh = b["size"]
                self.assertFalse(ax < bx+bw and ax+aw > bx and ay < by+bh and ay+ah > by)
        for folder, data in zip(("workflows_audio", "api_audio"), (ui, api)):
            self.assertEqual(json.loads((ROOT / folder / (audio_workflow.NAME + ".json")).read_text(encoding="utf-8")), data)


class AudioMediaTests(unittest.TestCase):
    def test_reference_16_and_32_same_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            for fps in (16, 32):
                path = Path(temp) / f"source{fps}.mp4"
                make_video(path, fps)
                frames, info = media.decode_reference(path)
                self.assertEqual(info["duration"], 5)
                self.assertEqual(frames.shape, (121, 128, 96, 3))
                self.assertGreater(int(frames[-1].mean()), int(frames[0].mean()))

    def test_padding_plan_bounded(self):
        info = {"width": 1440, "height": 1920, "duration": 5}
        self.assertEqual(media.reference_plan(info), (480, 640, 121))
        self.assertEqual(media.reference_plan({**info, "duration": 7})[-1], 169)
        with self.assertRaises(ValueError):
            media.reference_plan(info, 9999)

    def test_variable_timestamps_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vfr.mp4"
            make_video(path, fps=24, seconds=4, vfr=True)
            frames, info = media.decode_reference(path)
            _, _, expected = media.reference_plan(info)
            self.assertEqual(len(frames), expected)
            self.assertLess(abs(info["duration"] - 4), .05)
            output = Path(temp) / "vfr_audio.mp4"
            media.mux_copy(path, output, np.zeros((2, round(info["duration"] * 48000))), 48000)
            self.assertEqual(packets(path), packets(output))

    def test_overlong_video_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long.mp4"
            make_video(path, seconds=8)
            with self.assertRaisesRegex(ValueError, "up to 7s"):
                media.decode_reference(path)

    def test_audio_length_edges_and_validation(self):
        result = media.fit_audio(np.ones((1, 2, 244000)) * 2, 48000, 5, 20)
        self.assertEqual(result.shape, (2, 240000))
        self.assertTrue(np.all(result[:, [0, -1]] == 0))
        self.assertLessEqual(float(result.max()), .981)
        for audio in (np.ones((2, 1000)), np.full((2, 240000), np.nan), np.zeros((2, 2, 240000))):
            with self.assertRaises(ValueError):
                media.fit_audio(audio, 48000, 5)

    def test_mux_video_packets_and_timestamps_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "original.mp4", Path(temp) / "with_audio.mp4"
            make_video(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            t = np.arange(240000) / 48000
            audio = np.stack([.1 * np.sin(2*np.pi*440*t)] * 2)
            media.mux_copy(source, output, audio, 48000)
            self.assertEqual(packets(source), packets(output))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            with av.open(str(output)) as result:
                self.assertEqual(float(result.streams.video[0].average_rate), 32)
                self.assertAlmostEqual(float(result.streams.audio[0].duration * result.streams.audio[0].time_base), 5, places=2)
                self.assertEqual(len(list(result.decode(video=0))), 160)
            with av.open(str(output)) as result:
                decoded = np.concatenate([frame.to_ndarray() for frame in result.decode(audio=0)], axis=1)
                self.assertGreater(float(np.sqrt(np.mean(decoded ** 2))), .02)
            replaced = Path(temp) / "replaced_audio.mp4"
            media.mux_copy(output, replaced, np.zeros((2, 240000)), 48000)
            self.assertEqual(packets(source), packets(replaced))
            with av.open(str(replaced)) as result:
                self.assertEqual(len(result.streams.audio), 1)
                decoded = np.concatenate([frame.to_ndarray() for frame in result.decode(audio=0)], axis=1)
                self.assertLess(float(np.abs(decoded).max()), .0001)
            self.assertFalse(list(Path(temp).glob(".ltx-audio-*")))
            with self.assertRaises(ValueError):
                media.mux_copy(source, output, audio, 48000)
            with self.assertRaises(ValueError):
                media.mux_copy(source, source, audio, 48000)

    def test_mux_cancel_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "original.mp4", Path(temp) / "out.mp4"
            make_video(source)
            def cancel():
                raise InterruptedError("cancelled")
            with self.assertRaises(InterruptedError):
                media.mux_copy(source, output, np.zeros((2, 240000)), 48000, interrupt=cancel)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(temp).glob(".ltx-audio-*")))


if __name__ == "__main__":
    unittest.main()
