"""Audited adaptations of the PINNED upstream, applied at image build only.

Keep upstream license notices. Never call its shell-based video conversion,
temp-directory deletion or ping-pong looping paths. Array IO belongs to our job.
"""
from pathlib import Path
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Pinned LatentSync source changed: " + old[:90])
    return text.replace(old, new, 1)


def patch(root):
    path = root / "latentsync/pipelines/lipsync_pipeline.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "        audio_samples = read_audio(audio_path)\n        video_frames = read_video(video_path, use_decord=False)",
        '        video_frames = np.load(video_path, mmap_mode="r", allow_pickle=False)')
    marker = "        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)"
    if text.count(marker) != 1:
        raise ValueError("LatentSync output section changed")
    text = text[:text.index(marker)] + '        np.save(video_out_path, synced_video_frames, allow_pickle=False)\n'
    path.write_text(text, encoding="utf-8")
    path = root / "latentsync/utils/face_detector.py"
    text = path.read_text(encoding="utf-8")
    text = "import os\n" + text
    text = replace_once(text, 'root="checkpoints/auxiliary"', 'root=os.path.join(os.environ["LIPSYNC_MODELS"], "insightface")')
    text = replace_once(text, 'providers=["CUDAExecutionProvider"]', 'providers=["CPUExecutionProvider"]')
    text = replace_once(text, 'ctx_id=cuda_to_int(device)', 'ctx_id=-1')
    text = replace_once(text, '        faces = self.app.get(frame)',
        '        faces = self.app.get(frame)\n        if len(faces) > 1:\n            raise RuntimeError("Multiple faces detected; use one visible speaker")')
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
