"""Small optional post-audio UI/API graph. Leaves all three WAN workflows alone."""
import argparse
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
NAME = "04_DaSiWa_DragonLeap_V4_Add_Audio"
PROMPT = "Soft fabric rustling follows the visible movement, with quiet room ambience. No speech is present. No music is present."


def build():
    api = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "upload-your-finished-video.mp4"}},
        "2": {"class_type": "DaSiWaLTXAudio", "inputs": {"video": ["1", 0], "prompt": PROMPT,
            "seed": 42, "reference_max_side": 640, "refine_audio": True}},
        "3": {"class_type": "DaSiWaMuxAudio", "inputs": {"video": ["1", 0], "audio": ["2", 0],
            "filename_prefix": "DaSiWa/audio", "edge_fade_ms": 20}},
    }

    def node(i, title, pos, size, inputs, outputs, widgets):
        kind = api[str(i)]["class_type"]
        return {"id": i, "type": kind, "title": title, "pos": pos, "size": size,
            "flags": {}, "order": i-1, "mode": 0, "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": kind}, "widgets_values": widgets}

    canvas = [
        node(1, "01 · Upload FINISHED MP4 · <= 7 seconds", [40, 60], [450, 440], [],
             [{"name": "VIDEO", "type": "VIDEO", "links": [1, 2]}], ["upload-your-finished-video.mp4"]),
        node(2, "02 · Describe SOUND · DragonLeap V4", [550, 60], [500, 440],
             [{"name": "video", "type": "VIDEO", "link": 1}],
             [{"name": "AUDIO", "type": "AUDIO", "links": [3]}], [PROMPT, 42, "fixed", 640, True]),
        node(3, "03 · Save · Original pixels / FPS / mosaic kept", [1110, 60], [480, 440],
             [{"name": "video", "type": "VIDEO", "link": 2}, {"name": "audio", "type": "AUDIO", "link": 3}], [],
             ["DaSiWa/audio", 20]),
    ]
    note = ("AUDIO IMAGE ONLY: wan22-v9-ltxaudio-cu130 / AUDIO_PROFILE=dragonleap-v4\n"
            "Finish WAN generation (including RIFE, upscale, mosaic), save the MP4, then upload it here.\n"
            "No WAN re-generation. The saver stream-copies the ORIGINAL video and REPLACES its audio.\n"
            "One H.264/H.265 MP4, up to 7 seconds. Same duration, resolution and FPS; output is a NEW file.\n"
            "Describe audible events and their visible timing, not just the picture. Seed stays fixed for comparison.\n"
            "reference_max_side reduces ONLY the conditioning copy. 640 default; 512/384 reduce audio inference memory.\n"
            "refine_audio ON: 8+4+2 steps / CFG1 / euler_cfg_pp / linear_quadratic. OFF: 8-step trial, quality may differ.\n"
            "CFG1 ignores negatives. State No speech / No music in the positive prompt if needed; not guaranteed.\n"
            "edge_fade_ms=20 reduces boundary clicks; a perceptually seamless sound loop is NOT guaranteed.\n"
            "Uses the exact DragonLeap V4 INT8 ConvRot + Gemma / projection / VAEs from ltx23-i2v-runpod.\n"
            "This NEW frozen-video mode is experimental, NOT official Foley and NOT the original joint-video workflow.\n"
            "Additional models: 44.98 GB. CUDA13-compatible host driver required. GPU quality/speed not validated by CPU CI.")
    canvas.append({"id": 4, "type": "Note", "title": "READ ME · Sound-only postprocessing",
        "pos": [40, 570], "size": [1550, 380], "flags": {}, "order": 3, "mode": 0,
        "properties": {}, "widgets_values": [note]})
    ui = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "grawthings/dasiwa/dragonleap-v4-audio")),
        "revision": 1, "last_node_id": 4, "last_link_id": 3, "nodes": canvas,
        "links": [[1, 1, 0, 2, 0, "VIDEO"], [2, 1, 0, 3, 0, "VIDEO"], [3, 2, 0, 3, 1, "AUDIO"]],
        "groups": [], "config": {}, "extra": {"ds": {"scale": .65, "offset": [30, 30]}}, "version": .4}
    return ui, api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for folder, data in zip(("workflows_audio", "api_audio"), build()):
        path = ROOT / folder / (NAME + ".json")
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"Out of date: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
