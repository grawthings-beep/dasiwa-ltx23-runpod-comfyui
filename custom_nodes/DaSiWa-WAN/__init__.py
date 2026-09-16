"""One small native extension; no pip installs, model downloads or monkey patches."""
import json
import os
from fractions import Fraction
import av
import torch
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from .mosaic_nodes import WanAutoMosaicVideo, MODEL_FILENAME, DEFAULT_CLASSES
from . import postprocess, lora_stack


class DaSiWaLoraHigh(io.ComfyNode):
    stage = "high"

    @classmethod
    def define_schema(cls):
        options = lora_stack.choices(cls.stage, folder_paths.get_filename_list("loras"))
        inputs = [io.Model.Input("model")]
        for i in range(1, lora_stack.SLOTS + 1):
            inputs += [io.Boolean.Input(f"enabled_{i}", default=False),
                       io.Combo.Input(f"lora_{i}", options=options, default=lora_stack.NONE),
                       io.Float.Input(f"strength_{i}", default=1.0, min=-10, max=10, step=0.05,
                                      tooltip="Neutral starting value, not an author recommendation. Compare one LoRA at a time.")]
        return io.Schema(node_id=cls.__name__, display_name=f"{cls.stage.upper()} LoRA · 3 Optional Slots",
                         category="DaSiWa/loaders", inputs=inputs, outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, model, **values):
        return io.NodeOutput(lora_stack.apply(model, cls.stage, values))


class DaSiWaLoraLow(DaSiWaLoraHigh):
    stage = "low"


class DaSiWaRIFE2x(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaRIFE2x", display_name="RIFE 4.9 · 2x / Loop Timing", category="DaSiWa/video",
            inputs=[io.Image.Input("images"), io.Boolean.Input("enabled", default=True),
                    io.Float.Input("source_fps", default=16, min=1, max=60, step=1),
                    io.Boolean.Input("loop", default=True, tooltip="Remove ONE final generated endpoint AFTER interpolation. Keep trim disabled downstream.")],
            outputs=[io.Image.Output(), io.Float.Output(display_name="output_fps")])

    @classmethod
    def execute(cls, images, enabled, source_fps, loop):
        frames, fps = postprocess.interpolate(images, enabled, source_fps, loop)
        return io.NodeOutput(frames, fps)


class DaSiWaUpscale2x(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaUpscale2x", display_name="SPAN · 2x Framewise Upscale", category="DaSiWa/video",
            inputs=[io.Image.Input("images"), io.Boolean.Input("enabled", default=True)],
            outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, images, enabled):
        return io.NodeOutput(postprocess.upscale(images, enabled))


class DaSiWaAutoMosaic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaAutoMosaic", display_name="Loop Auto Mosaic · JUST", category="DaSiWa/video",
            inputs=[io.Image.Input("images"),
                    io.Combo.Input("coverage_preset", options=["JUST", "WIDE", "SAFE"], default="JUST"),
                    io.Float.Input("confidence", default=0.30, min=0.05, max=0.95, step=0.01),
                    io.Float.Input("iou_threshold", default=0.50, min=0.05, max=0.95, step=0.01),
                    io.Int.Input("block_size", default=0, min=0, max=128, step=2),
                    io.Int.Input("max_gap_frames", default=3, min=0, max=24),
                    io.String.Input("target_classes", default=DEFAULT_CLASSES),
                    io.Combo.Input("device", options=["auto", "cpu"], default="auto"),
                    io.Boolean.Input("trim_last_frame", default=True,
                                     tooltip="Remove repeated loop endpoint BEFORE circular mask gap filling. Disable trim in the MP4 saver.")],
            outputs=[io.Image.Output(display_name="mosaicked_images")])

    @classmethod
    def execute(cls, images, coverage_preset, confidence, iou_threshold, block_size,
                max_gap_frames, target_classes, device, trim_last_frame):
        if images.ndim != 4 or len(images) < (2 if trim_last_frame else 1):
            raise ValueError("Expected a nonempty video (at least 2 frames when trimming)")
        if trim_last_frame:
            images = images[:-1]
        result = WanAutoMosaicVideo().apply(images, MODEL_FILENAME, coverage_preset,
            confidence, iou_threshold, block_size, max_gap_frames, target_classes, device)
        return io.NodeOutput(result[0])


class DaSiWaSaveMP4(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DaSiWaSaveMP4", display_name="Save MP4 · DaSiWa", category="DaSiWa/video", is_output_node=True,
            inputs=[io.Image.Input("images"), io.Float.Input("fps", default=16, min=1, max=120, step=1),
                    io.String.Input("filename_prefix", default="DaSiWa/video"),
                    io.Int.Input("crf", default=18, min=0, max=51),
                    io.Boolean.Input("trim_last_frame", default=False, tooltip="Loop variant: remove the repeated endpoint, not insert a source image.")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo])

    @classmethod
    def execute(cls, images, fps, filename_prefix, crf, trim_last_frame):
        if images.ndim != 4 or len(images) < 1 or (trim_last_frame and len(images) < 2):
            raise ValueError("Expected a nonempty video (at least 2 frames when trimming)")
        if trim_last_frame:
            images = images[:-1]
        height, width = images.shape[1:3]
        if height % 2 or width % 2:
            raise ValueError("H.264 output needs even width and height")
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), width, height)
        name = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(folder, name)
        partial = path + ".partial"
        try:
            with av.open(partial, "w", format="mp4", options={"movflags": "+faststart"}) as container:
                metadata = {}
                if cls.hidden.prompt is not None:
                    metadata["prompt"] = cls.hidden.prompt
                if cls.hidden.extra_pnginfo is not None:
                    metadata.update(cls.hidden.extra_pnginfo)
                container.metadata["comment"] = json.dumps(metadata)
                stream = container.add_stream("libx264", rate=Fraction(str(fps)))
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf), "preset": "veryfast"}
                # Framewise conversion avoids a second full-video uint8 buffer.
                for tensor in images:
                    array = (tensor[..., :3].detach().clamp(0, 1) * 255).round().to(device="cpu", dtype=torch.uint8).numpy()
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            os.replace(partial, path)
        except BaseException:
            if os.path.isfile(partial):
                os.unlink(partial)
            raise
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)]))


class DaSiWaExtension(ComfyExtension):
    async def get_node_list(self):
        return [DaSiWaSaveMP4, DaSiWaAutoMosaic, DaSiWaRIFE2x, DaSiWaUpscale2x, DaSiWaLoraHigh, DaSiWaLoraLow]


async def comfy_entrypoint():
    return DaSiWaExtension()
