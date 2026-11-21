"""Separate one-click speech graph; original WAN workflows remain byte-identical."""
import argparse
import copy
import json
import uuid
from pathlib import Path
from workflows import build as wan_build, build_mosaic

ROOT = Path(__file__).resolve().parents[1]
NAME = "05_DaSiWa_v9_Image_Speech_LipSync"


def build():
    ui, api = wan_build(False)
    ui["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "grawthings/dasiwa/v9/speech-lipsync"))
    nodes = {n["id"]: n for n in ui["nodes"]}

    def add(i, kind, title, pos, size, values, inputs, outputs, widgets):
        api[str(i)] = {"class_type": kind, "inputs": values, "_meta": {"title": title}}
        node = {"id": i, "type": kind, "title": title, "pos": pos, "size": size, "flags": {}, "order": len(ui["nodes"]), "mode": 0,
                "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
                "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets}
        ui["nodes"].append(node); nodes[i] = node

    prompt = "Locked-off camera. The subject faces the viewer with a friendly expression, gentle natural head movement and subtle conversational gestures. The face stays clearly visible and mostly front-facing. Consistent character, hairstyle, outfit and lighting."
    api["7"]["inputs"]["text"] = prompt
    nodes[7]["widgets_values"] = [prompt]
    add(31, "LoadAudio", "02 · Upload recorded dialogue · 0.5–6 seconds", [40, 650], [440, 150],
        {"audio": "upload-your-speech.wav"}, [], [("AUDIO", "AUDIO")], ["upload-your-speech.wav", None, None])
    add(32, "DaSiWaSpeechPlan", "03 · Face preflight / automatic duration", [40, 850], [440, 150],
        {"image": ["1", 0], "audio": ["31", 0]}, [("image", "IMAGE"), ("audio", "AUDIO")],
        [("IMAGE", "IMAGE"), ("AUDIO", "AUDIO"), ("wan_frames", "INT")], [])
    api["10"]["inputs"].update(start_image=["32", 0], length=["32", 2])
    nodes[10]["inputs"].append({"name": "length", "type": "INT", "widget": {"name": "length"}, "link": None})
    nodes[10]["title"] = "WAN · Width / Height · duration follows speech"
    api["10"]["_meta"]["title"] = nodes[10]["title"]
    add(33, "DaSiWaLipSync", "09 · Mouth sync · original speech · 25 FPS", [2060, 1700], [440, 300],
        {"images": ["17", 0], "audio": ["32", 1], "source_fps": ["17", 1], "seed": 42, "steps": 20, "guidance": 1.5},
        [("images", "IMAGE"), ("audio", "AUDIO"), ("source_fps", "FLOAT")], [("IMAGE", "IMAGE"), ("AUDIO", "AUDIO")], [32.0, 42, "fixed", 20, 1.5])
    nodes[33]["inputs"][2]["widget"] = {"name": "source_fps"}
    api["18"]["inputs"]["images"] = ["33", 0]
    nodes[18]["pos"] = [2560, 1700]
    nodes[18]["title"] = "10 · SPAN 2x · after mouth sync"
    mosaic_ui, mosaic_api = build_mosaic()
    mosaic = copy.deepcopy(mosaic_api["16"])
    api["35"] = mosaic
    mosaic_node = copy.deepcopy(next(n for n in mosaic_ui["nodes"] if n["id"] == 16))
    mosaic_node.update(id=35, title="11 · Auto Mosaic · after all image processing", pos=[3060, 1700])
    ui["nodes"].append(mosaic_node); nodes[35] = mosaic_node
    api["14"] = {"class_type": "DaSiWaSaveSpeechMP4", "inputs": {"images": ["35", 0], "audio": ["33", 1], "filename_prefix": "DaSiWa/speech"},
                 "_meta": {"title": "12 · Save original voice + lip-synced MP4"}}
    nodes[14].update(type="DaSiWaSaveSpeechMP4", title=api["14"]["_meta"]["title"], pos=[3560, 1700],
                     inputs=[{"name": "images", "type": "IMAGE", "link": None}, {"name": "audio", "type": "AUDIO", "link": None}],
                     properties={"Node name for S&R": "DaSiWaSaveSpeechMP4"}, widgets_values=["DaSiWa/speech"])
    nodes[15]["widgets_values"] = [
        "IMAGE + RECORDED SPEECH -> WAN v9 -> RIFE -> LatentSync 1.6 -> SPAN -> Mosaic -> MP4\n"
        "Upload one image and one 0.5–6s dialogue file, edit motion prompt, then Run once. No text-to-speech or LTX model.\n"
        "Speech Plan checks a single detectable face BEFORE WAN sampling and sets WAN frame count automatically.\n"
        "Keep face visible, near frontal, not tiny or covered. Anime detection/visual quality requires a real test.\n"
        "This is NOT a seamless loop: speech drives mouth timing. No inserted first image, reverse/ping-pong, speech truncation or speed change.\n"
        "Output: 25 FPS (LatentSync native), speech duration rounded UP by <40ms, silence only at the very end. Original voice, AAC.\n"
        "RIFE resamples source motion before mouth sync. Do NOT interpolate/trim after sync. SPAN does not change timing.\n"
        "WAN checkpoints/sampler/optional LoRAs unchanged. LatentSync uses 20 steps / guidance1.5 / 512 face crop; it unloads WAN GPU models first.\n"
        "No DeepCache. Higher guidance can distort faces; more steps cost time. No claim of GPU validation or unchanged face quality.\n"
        "Lip-sync runtime/models: special lip-sync image, LIPSYNC_PROFILE=latentsync-1.6, AUDIO_PROFILE=none. +5.50 GB weights.\n"
        "Face weights (InsightFace buffalo_l) are restricted to non-commercial research. Check all model licenses before use.\n"
        "Mosaic is output-only, after lip sync and SPAN. Always inspect final frames; automated detection is not guaranteed."
    ]
    # Rebuild every wire from the executable graph, avoiding stale slot IDs.
    ui["links"] = []
    for n in ui["nodes"]:
        for entry in n.get("inputs", []): entry["link"] = None
        for entry in n.get("outputs", []): entry["links"] = []
    for key, node in api.items():
        for name, value in node["inputs"].items():
            if not isinstance(value, list): continue
            source, slot = nodes[int(value[0])], value[1]
            target = nodes[int(key)]
            target_slot = next(i for i, entry in enumerate(target["inputs"]) if entry["name"] == name)
            number = len(ui["links"]) + 1
            ui["links"].append([number, source["id"], slot, target["id"], target_slot, source["outputs"][slot]["type"]])
            source["outputs"][slot]["links"].append(number)
            target["inputs"][target_slot]["link"] = number
    ui.update(last_node_id=35, last_link_id=len(ui["links"]))
    for order, node in enumerate(ui["nodes"]): node["order"] = order
    return ui, api


def generate(check=False):
    for folder, data in zip(("workflows_lipsync", "api_lipsync"), build()):
        path = ROOT / folder / (NAME + ".json")
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if check:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Stale workflow: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)
