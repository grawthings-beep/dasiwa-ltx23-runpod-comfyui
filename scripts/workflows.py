"""Deterministic UI + API workflows. Native WAN; one tiny video saver extension."""
import argparse
import copy
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]


def build(loop=False):
    assets = {a["id"]: a["path"].split("/", 1)[1] for a in json.loads((ROOT / "config/models.json").read_text())["assets"]}
    graph = {}
    canvas = []
    connections = []

    def add(i, kind, title, pos, size, values, outputs, widgets, input_types=None):
        graph[str(i)] = {"class_type": kind, "inputs": copy.deepcopy(values), "_meta": {"title": title}}
        inputs = [{"name": name, "type": typ, "link": None} for name, typ in (input_types or {}).items()]
        canvas.append({"id": i, "type": kind, "title": title, "pos": pos, "size": size, "flags": {}, "order": len(canvas), "mode": 0,
                       "inputs": inputs, "outputs": [{"name": name, "type": typ, "links": []} for name, typ in outputs],
                       "properties": {"Node name for S&R": kind}, "widgets_values": widgets})

    def link(origin, slot, target, name):
        a = next(n for n in canvas if n["id"] == origin)
        b = next(n for n in canvas if n["id"] == target)
        input_slot = next(i for i, n in enumerate(b["inputs"]) if n["name"] == name)
        number = len(connections) + 1
        connections.append([number, origin, slot, target, input_slot, a["outputs"][slot]["type"]])
        a["outputs"][slot]["links"].append(number)
        b["inputs"][input_slot]["link"] = number
        graph[str(target)]["inputs"][name] = [str(origin), slot]

    prompt = "Locked-off camera. A steady gentle breeze moves the fabric and hair in smooth continuous waves. The subject stays in place with a calm, unchanged expression and closed mouth. Natural motion, consistent lighting, stable composition."
    if loop:
        prompt += " The motion completes one gentle cycle and smoothly returns to its starting pose."
    add(1, "LoadImage", "01 · Upload image", [40, 80], [440, 490], {"image": "upload-your-image.png"}, [("IMAGE", "IMAGE"), ("MASK", "MASK")], ["upload-your-image.png", "image"])
    add(2, "UNETLoader", "HIGH · SynthSeduction v9", [560, 80], [440, 90], {"unet_name": assets["high"], "weight_dtype": "default"}, [("MODEL", "MODEL")], [assets["high"], "default"])
    add(3, "UNETLoader", "LOW · SynthSeduction v9", [1060, 80], [440, 90], {"unet_name": assets["low"], "weight_dtype": "default"}, [("MODEL", "MODEL")], [assets["low"], "default"])
    add(4, "ModelSamplingSD3", "HIGH · Shift 5", [560, 220], [440, 90], {"shift": 5.0}, [("MODEL", "MODEL")], [5.0], {"model": "MODEL"})
    add(5, "ModelSamplingSD3", "LOW · Shift 5", [1060, 220], [440, 90], {"shift": 5.0}, [("MODEL", "MODEL")], [5.0], {"model": "MODEL"})
    add(6, "CLIPLoader", "UMT5 · GPU auto/offload", [40, 650], [440, 120], {"clip_name": assets["text"], "type": "wan", "device": "default"}, [("CLIP", "CLIP")], [assets["text"], "wan", "default"])
    add(7, "CLIPTextEncode", "02 · Describe the motion", [560, 390], [440, 240], {"text": prompt}, [("CONDITIONING", "CONDITIONING")], [prompt], {"clip": "CLIP"})
    add(8, "ConditioningZeroOut", "CFG 1 · no negative evaluation", [1060, 390], [440, 70], {}, [("CONDITIONING", "CONDITIONING")], [], {"conditioning": "CONDITIONING"})
    add(9, "VAELoader", "Wan 2.1 VAE · correct for A14B", [40, 850], [440, 80], {"vae_name": assets["vae"]}, [("VAE", "VAE")], [assets["vae"]])
    kind = "WanFirstLastFrameToVideo" if loop else "WanImageToVideo"
    ins = {"positive": "CONDITIONING", "negative": "CONDITIONING", "vae": "VAE"}
    if loop:
        ins.update({"clip_vision_start_image": "CLIP_VISION_OUTPUT", "clip_vision_end_image": "CLIP_VISION_OUTPUT", "start_image": "IMAGE", "end_image": "IMAGE"})
    else:
        ins.update({"clip_vision_output": "CLIP_VISION_OUTPUT", "start_image": "IMAGE"})
    add(10, kind, "03 · Width / Height / Frames · batch = 1", [560, 740], [440, 290], {"width": 720, "height": 960, "length": 81, "batch_size": 1}, [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")], [720, 960, 81, 1], ins)
    for i, title, noise, start, end, leftover, x in [(11, "04 · HIGH · first 2 steps", "enable", 0, 2, "enable", 1060), (12, "05 · LOW · last 2 steps", "disable", 2, 4, "disable", 1560)]:
        values = {"add_noise": noise, "noise_seed": 123456789, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "start_at_step": start, "end_at_step": end, "return_with_leftover_noise": leftover}
        add(i, "KSamplerAdvanced", title, [x, 650], [440, 390], values, [("LATENT", "LATENT")], [noise, 123456789, "fixed", 4, 1.0, "euler", "simple", start, end, leftover], {"model": "MODEL", "positive": "CONDITIONING", "negative": "CONDITIONING", "latent_image": "LATENT"})
    add(13, "VAEDecode", "06 · Decode · native OOM fallback", [1560, 1120], [440, 90], {}, [("IMAGE", "IMAGE")], [], {"samples": "LATENT", "vae": "VAE"})
    prefix = "DaSiWa/loop" if loop else "DaSiWa/i2v"
    add(14, "DaSiWaSaveMP4", "07 · Save MP4 · " + ("80 frames = 5 seconds" if loop else "81 frames / 16 FPS"), [1560, 1300], [440, 360], {"fps": 16.0, "filename_prefix": prefix, "crf": 18, "trim_last_frame": loop}, [], [16.0, prefix, 18, loop], {"images": "IMAGE"})
    for a, slot, b, name in [(2,0,4,"model"),(3,0,5,"model"),(6,0,7,"clip"),(7,0,8,"conditioning"),(7,0,10,"positive"),(8,0,10,"negative"),(9,0,10,"vae"),(1,0,10,"start_image"),(4,0,11,"model"),(5,0,12,"model"),(10,0,11,"positive"),(10,1,11,"negative"),(10,2,11,"latent_image"),(10,0,12,"positive"),(10,1,12,"negative"),(11,0,12,"latent_image"),(12,0,13,"samples"),(9,0,13,"vae"),(13,0,14,"images")]:
        link(a, slot, b, name)
    if loop:
        link(1, 0, 10, "end_image")
    note = "Start: upload an image, describe motion, then Run.\n720 x 960 / 81 frames / 16 FPS / 4 TOTAL steps (2 high + 2 low).\nNative checkpoint precision; do not force fp8_fast or add Lightning/LightX2V.\nThe same seed is kept for comparisons; change HIGH noise_seed for variation.\nCFG=1 does not evaluate a negative prompt. No audio/upscale/interpolation pass."
    if loop:
        note += "\nLoop: the image CONDITIONS both ends; no source image is pasted into the result.\nThe repeated endpoint is dropped (80 frames / 16 FPS = 5s).\nA seamless motion/velocity match is NOT guaranteed. Use cyclic motion and a fixed camera."
    canvas.append({"id": 15, "type": "Note", "pos": [40, 1120], "size": [930, 300], "flags": {}, "order": 14, "mode": 0, "properties": {}, "widgets_values": [note], "title": "READ ME · Quality / speed / loop boundaries"})
    # Unconnected optional sockets are absent from the API, not empty strings.
    ui = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "grawthings/dasiwa/v9/" + str(loop))), "revision": 0, "last_node_id": 15, "last_link_id": len(connections), "nodes": canvas, "links": connections, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.55, "offset": [50, 50]}}, "version": 0.4}
    return ui, graph


def generate(check=False):
    for loop, name in [(False, "01_DaSiWa_v9_I2V"), (True, "02_DaSiWa_v9_Loop")]:
        for folder, data in zip(("workflows", "api"), build(loop)):
            path = ROOT / folder / (name + ".json")
            content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            if check:
                if path.read_text(encoding="utf-8") != content:
                    raise ValueError(f"Stale generated workflow: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    generate(p.parse_args().check)
