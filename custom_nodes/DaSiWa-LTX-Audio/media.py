"""Bounded reference decoding and lossless video remux; no inference imports."""
import math
import os
from pathlib import Path
from fractions import Fraction
import tempfile

import av
import numpy as np

REFERENCE_FPS = 24
MAX_SECONDS = 7.0


def probe(path):
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Select an MP4 with exactly one video stream")
        stream = container.streams.video[0]
        if stream.codec_context.name not in ("h264", "hevc"):
            raise ValueError("Video stream copy supports H.264/H.265 MP4 only; no silent re-encode")
        if stream.duration is None or stream.time_base is None:
            raise ValueError("Video has no reliable duration; export it as MP4 first")
        duration = float(stream.duration * stream.time_base)
        if not 0 < duration <= MAX_SECONDS + 1e-6:
            raise ValueError(f"Audio workflow accepts clips up to {MAX_SECONDS:g}s; got {duration:.3f}s")
        if stream.width <= 0 or stream.height <= 0 or stream.width * stream.height > 4096 * 4096:
            raise ValueError("Invalid or oversized video")
        return {"duration": duration, "width": stream.width, "height": stream.height,
                "start": float((stream.start_time or 0) * stream.time_base)}


def reference_plan(info, max_side=640):
    if max_side not in (384, 512, 640, 768):
        raise ValueError("Reference max_side must be 384, 512, 640 or 768")
    ratio = min(1.0, max_side / max(info["width"], info["height"]))
    width = max(32, int(info["width"] * ratio / 32) * 32)
    height = max(32, int(info["height"] * ratio / 32) * 32)
    # Pad the CONDITIONING copy to 1+8n. Original MP4 is never trimmed/padded.
    frames = 1 + 8 * math.ceil(max(0, info["duration"] * REFERENCE_FPS - 1) / 8)
    return width, height, frames


def decode_reference(path, max_side=640, interrupt=lambda: None):
    info = probe(path)
    width, height, count = reference_plan(info, max_side)
    result = np.empty((count, height, width, 3), dtype=np.uint8)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        previous = None
        index = 0
        last_time = -float("inf")
        for frame in container.decode(stream):
            interrupt()
            if frame.pts is None:
                raise ValueError("Missing video timestamp; cannot align sound reliably")
            time = float(frame.pts * frame.time_base) - info["start"]
            if time < last_time:
                raise ValueError("Non-monotonic video timestamps")
            last_time = time
            current = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            # Nearest-previous timestamp resampling handles 16/32 fps and VFR.
            # It does NOT reinterpret 32 fps as 24 fps (which would slow it down).
            if previous is None:
                previous = current
            while index < count and index / REFERENCE_FPS < time - 1e-8:
                result[index] = previous
                index += 1
            previous = current
            if index == count:
                break
        if previous is None:
            raise ValueError("No decodable video frames")
        result[index:] = previous
    return result, info


def fit_audio(waveform, sample_rate, duration, fade_ms=20):
    """Exact duration; short boundary fades reduce clicks, not a seamless-loop claim."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("Expected one nonempty mono/stereo audio batch")
    if not 8000 <= sample_rate <= 192000 or not 0 <= fade_ms <= 100:
        raise ValueError("Invalid sample rate or fade")
    if not np.isfinite(audio).all():
        raise ValueError("Non-finite audio; not saving corrupt output")
    length = round(duration * sample_rate)
    # Padding more than codec/latent rounding would disguise a broken inference.
    if audio.shape[1] < length - round(sample_rate * .1):
        raise ValueError("Generated audio is too short for this video")
    out = np.zeros((audio.shape[0], length), dtype=np.float32)
    n = min(length, audio.shape[1])
    out[:, :n] = audio[:, :n]
    peak = float(np.max(np.abs(out)))
    if peak > 1:
        out *= .98 / peak  # prevent clipping; never amplify near-silent samples
    fade = min(round(fade_ms * sample_rate / 1000), length // 2)
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        out[:, :fade] *= ramp
        out[:, -fade:] *= ramp[::-1]
    return out


def mux_copy(source, target, waveform, sample_rate, fade_ms=20, interrupt=lambda: None):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.exists():
        raise ValueError("Refusing to overwrite source/existing output")
    info = probe(source)
    audio = fit_audio(waveform, sample_rate, info["duration"], fade_ms)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Use the pinned PyAV's FFmpeg rather than a distro's older ffmpeg CLI,
    # which can infer a different final packet duration for VFR input.
    with tempfile.TemporaryDirectory(prefix=".ltx-audio-", dir=target.parent) as temp:
        temp = Path(temp)
        partial = temp / "result.mp4"
        with av.open(str(source)) as original, av.open(str(partial), "w", format="mp4", options={"movflags": "+faststart"}) as output:
            original_video = original.streams.video[0]
            copied_video = output.add_stream_from_template(original_video)
            copied_video.time_base = original_video.time_base
            output.metadata.update(original.metadata)
            sound = output.add_stream("aac", rate=sample_rate)
            sound.layout = "mono" if audio.shape[0] == 1 else "stereo"
            sound.bit_rate = 192000
            sound.time_base = Fraction(1, sample_rate)

            def video_packets():
                for packet in original.demux(original_video):
                    interrupt()
                    if packet.size:
                        if packet.dts is None:
                            raise ValueError("Video packet lacks a decode timestamp")
                        packet.stream = copied_video
                        yield packet

            def audio_packets():
                for pos in range(0, audio.shape[1], 1024):
                    interrupt()
                    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[:, pos:pos+1024]),
                        format="fltp", layout=sound.layout.name)
                    frame.sample_rate = sample_rate
                    frame.time_base = Fraction(1, sample_rate)
                    frame.pts = pos + round(info["start"] * sample_rate)
                    yield from sound.encode(frame)
                yield from sound.encode()

            # Keep at most one compressed video/audio packet in our own queues.
            viter, aiter = iter(video_packets()), iter(audio_packets())
            vp, ap = next(viter, None), next(aiter, None)
            while vp is not None or ap is not None:
                interrupt()
                if ap is None or (vp is not None and vp.dts * vp.time_base <= ap.dts * ap.time_base):
                    output.mux(vp)
                    vp = next(viter, None)
                else:
                    output.mux(ap)
                    ap = next(aiter, None)
        check = probe(partial)
        if abs(check["duration"] - info["duration"]) > .001 or (check["width"], check["height"]) != (info["width"], info["height"]):
            raise RuntimeError(f"Mux unexpectedly changed video dimensions/duration: {info} -> {check}")
        with av.open(str(partial)) as output:
            if len(output.streams.audio) != 1:
                raise RuntimeError("Mux produced no audio stream")
        interrupt()
        # A unique Comfy output filename is chosen upstream; retain fail-closed semantics.
        if target.exists():
            raise FileExistsError(target)
        os.replace(partial, target)
    return target
