"""One small native extension; no pip installs, model downloads or monkey patches."""
import json
import os
from fractions import Fraction
import av
import torch
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui


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
        return [DaSiWaSaveMP4]


async def comfy_entrypoint():
    return DaSiWaExtension()
