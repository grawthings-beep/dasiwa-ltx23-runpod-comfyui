"""Optional audio-enabled image only. No import-time model loads/downloads."""
import os
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, io, ui
from . import engine, media


def source_path(video):
    path = video.get_stream_source()
    if not isinstance(path, str) or not Path(path).is_file():
        raise ValueError("Use Load Video with a saved local MP4, not an in-memory generated video")
    if hasattr(video, "get_active_trim_window") and video.get_active_trim_window() != (0.0, 0.0):
        raise ValueError("Use the original untrimmed MP4; save a trimmed clip first")
    return path


class DaSiWaLTXAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="LTX 2.3 · DragonLeap V4 · Sound Only", category="DaSiWa/audio",
            description="Experimental frozen-video audio generation. Full LTX model; not official Foley. Original video is unchanged.",
            inputs=[io.Video.Input("video"),
                    io.String.Input("prompt", multiline=True, default="Soft fabric rustling follows the visible movement, with quiet room ambience. No speech is present. No music is present."),
                    io.Int.Input("seed", default=42, min=0, max=2**64-1, control_after_generate=True),
                    io.Combo.Input("reference_max_side", options=[384, 512, 640, 768], default=640,
                                   tooltip="Only the sound-conditioning copy is reduced. Original video keeps its full resolution/FPS."),
                    io.Boolean.Input("refine_audio", default=True, tooltip="8 + 4 + 2 steps, matching source sampling settings. OFF: 8-step trial, quality can differ.")],
            outputs=[io.Audio.Output()])

    @classmethod
    def execute(cls, video, prompt, seed, reference_max_side, refine_audio):
        if os.environ.get("AUDIO_PROFILE", "none").lower() != "dragonleap-v4":
            raise RuntimeError("Use the wan22-v9-ltxaudio-cu130 image with AUDIO_PROFILE=dragonleap-v4")
        return io.NodeOutput(engine.generate(source_path(video), prompt, seed, int(reference_max_side), refine_audio))


class DaSiWaMuxAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Save MP4 · Original Video + LTX Audio", category="DaSiWa/audio", is_output_node=True,
            inputs=[io.Video.Input("video"), io.Audio.Input("audio"),
                    io.String.Input("filename_prefix", default="DaSiWa/audio"),
                    io.Int.Input("edge_fade_ms", default=20, min=0, max=100,
                                 tooltip="Reduces boundary clicks. Does not guarantee seamless sound or invent matching speech.")])

    @classmethod
    def execute(cls, video, audio, filename_prefix, edge_fade_ms):
        path = source_path(video)
        info = media.probe(path)
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix,
            folder_paths.get_output_directory(), info["width"], info["height"])
        name = f"{filename}_{counter:05}_.mp4"
        media.mux_copy(path, Path(folder) / name, audio["waveform"].detach().cpu().numpy(),
                       int(audio["sample_rate"]), edge_fade_ms, mm.throw_exception_if_processing_interrupted)
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [DaSiWaLTXAudio, DaSiWaMuxAudio]


async def comfy_entrypoint():
    return Extension()
