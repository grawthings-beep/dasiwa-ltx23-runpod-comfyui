"""CPU media/timing helpers. Input speech is preserved, never synthesized."""
from fractions import Fraction
import math
import os
from pathlib import Path
import tempfile
import av
import numpy as np


FPS = 25
MAX_SECONDS = 6


def speech_plan(samples, rate):
    if not isinstance(rate, int) or not 8000 <= rate <= 192000:
        raise ValueError("Speech sample rate must be 8000..192000 Hz")
    seconds = samples / rate
    if not .5 <= seconds <= MAX_SECONDS:
        raise ValueError("Use one short speech clip, 0.5..6 seconds. No automatic truncation or video looping.")
    # WAN needs 4n+1 frames. Cover the final 25fps output timestamp as well.
    output_count = (FPS * samples + rate - 1) // rate
    length = 1 + 4 * math.ceil((output_count - 1) * 16 / (FPS * 4))
    return length, output_count


def waveform(audio):
    rate = int(audio["sample_rate"])
    wave = audio["waveform"]
    if hasattr(wave, "detach"):
        wave = wave.detach().to(device="cpu", dtype=__import__("torch").float32).numpy()
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2):
        raise ValueError("Expected one mono/stereo speech clip (batch=1)")
    speech_plan(wave.shape[-1], rate)
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.001:
        raise ValueError("Speech has invalid/clipping samples; normalize the source audio first")
    if np.max(np.abs(wave)) < 1e-5:
        raise ValueError("Speech input is silent")
    return wave[0], rate


def frame_indices(count, source_fps, target_count):
    if not math.isfinite(source_fps) or not 1 <= source_fps <= 120:
        raise ValueError("Invalid source FPS")
    indices = np.floor(np.arange(target_count) * source_fps / FPS + .5).astype(int)
    if count < 1 or indices[-1] >= count:
        raise ValueError("Video is shorter than the speech. Regenerate with Speech Plan connected; no ping-pong fallback.")
    return indices


def pad_wave(wave, rate, count):
    target = round(count / FPS * rate)
    padding = target - wave.shape[-1]
    if padding < 0 or padding > math.ceil(rate / FPS):
        raise ValueError("Audio/video lengths differ by more than one output frame")
    return np.pad(wave, ((0, 0), (0, padding)))


def write_wave(path, wave, rate):
    layout = "mono" if wave.shape[0] == 1 else "stereo"
    with av.open(str(path), "w", format="wav") as out:
        stream = out.add_stream("pcm_f32le", rate=rate)
        stream.layout = layout
        for pos in range(0, wave.shape[1], 4096):
            # Packed float PCM: interleave channels along the single row.
            data = np.ascontiguousarray(wave[:, pos:pos+4096].T.reshape(1, -1))
            frame = av.AudioFrame.from_ndarray(data, format="flt", layout=layout)
            frame.sample_rate, frame.pts, frame.time_base = rate, pos, Fraction(1, rate)
            for p in stream.encode(frame):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)


def save_video(target, frames, wave, rate, count, width, height, metadata="", interrupt=lambda: None):
    """Single H264 encode + original speech AAC. Atomic, no silent-success fallback."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    if width % 2 or height % 2 or min(width, height) < 2:
        raise ValueError("H264 requires positive even dimensions")
    interrupt()
    wave = pad_wave(wave, rate, count)
    # AAC supports 48k/44.1k etc; use a deterministic 48k resampler for any upload rate.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lipsync-save-", dir=target.parent) as temp:
        partial = Path(temp) / "result.mp4"
        with av.open(str(partial), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            out.metadata["comment"] = metadata
            video = out.add_stream("libx264", rate=FPS)
            video.width, video.height, video.pix_fmt = width, height, "yuv420p"
            video.options = {"crf": "18", "preset": "veryfast"}
            sound = out.add_stream("aac", rate=48000)
            sound.layout = "mono" if wave.shape[0] == 1 else "stereo"
            sound.bit_rate = 192000
            resampler = av.AudioResampler(format="fltp", layout=sound.layout.name, rate=48000)

            def vp():
                seen = 0
                for index, array in enumerate(frames):
                    interrupt()
                    if array.shape != (height, width, 3) or array.dtype != np.uint8:
                        raise ValueError("Invalid RGB frame")
                    f = av.VideoFrame.from_ndarray(array, format="rgb24")
                    f.pts, f.time_base = index, Fraction(1, FPS)
                    yield from video.encode(f)
                    seen += 1
                if seen != count:
                    raise ValueError("Unexpected lip-sync frame count")
                yield from video.encode()

            def ap():
                for pos in range(0, wave.shape[1], 4096):
                    interrupt()
                    f = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave[:, pos:pos+4096]), format="fltp", layout=sound.layout.name)
                    f.sample_rate, f.pts, f.time_base = rate, pos, Fraction(1, rate)
                    for rf in resampler.resample(f):
                        yield from sound.encode(rf)
                for rf in resampler.resample(None):
                    yield from sound.encode(rf)
                yield from sound.encode()

            vi, ai = iter(vp()), iter(ap())
            v, a = next(vi, None), next(ai, None)
            while v is not None or a is not None:
                interrupt()
                if a is None or (v is not None and v.dts*v.time_base <= a.dts*a.time_base):
                    out.mux(v); v = next(vi, None)
                else:
                    out.mux(a); a = next(ai, None)
        with av.open(str(partial)) as check:
            if len(check.streams.video) != 1 or len(check.streams.audio) != 1:
                raise RuntimeError("Missing video/audio in output")
            stream = check.streams.video[0]
            if stream.frames != count or abs(float(stream.duration * stream.time_base)-count/FPS) > .001:
                raise RuntimeError("Output timing changed")
        interrupt()
        if target.exists():
            raise FileExistsError(target)
        os.replace(partial, target)
