import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
SOURCE = WORKFLOWS / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V39.json"
OUTPUT = WORKFLOWS / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V39_runpod.json"

DEFAULT_UNET = "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
DEFAULT_TEXT_ENCODER = "gemma-3-12b-it-heretic-v2_fp8_e4m3fn.safetensors"
DISABLED_NODE_TYPES = {
    "DaSiWa_RTX_UpscalerRefiner",
    "LTX2MemoryEfficientSageAttentionPatch",
    "PathchSageAttentionKJ",
}


def containers(value):
    if isinstance(value, dict):
        if isinstance(value.get("nodes"), list):
            yield value
        for child in value.values():
            yield from containers(child)
    elif isinstance(value, list):
        for child in value:
            yield from containers(child)


def apply_runpod_defaults(container: dict) -> int:
    nodes = container.get("nodes", [])
    links = container.get("links", [])
    nodes_by_id = {node.get("id"): node for node in nodes}
    disabled_ids = set()
    changed = 0

    for node in nodes:
        node_type = node.get("type")
        widgets = node.get("widgets_values")

        if node_type == "UNETLoader" and isinstance(widgets, list) and widgets:
            if widgets[0] != DEFAULT_UNET:
                widgets[0] = DEFAULT_UNET
                changed += 1

        if node_type in {"VAELoader", "VAELoaderKJ"} and isinstance(widgets, list) and widgets:
            if isinstance(widgets[0], str) and widgets[0].startswith("LTX2/"):
                widgets[0] = "LTX/" + widgets[0].split("/", 1)[1]
                changed += 1

        if node_type == "DualCLIPLoader" and isinstance(widgets, list):
            if widgets and widgets[0] != DEFAULT_TEXT_ENCODER:
                widgets[0] = DEFAULT_TEXT_ENCODER
                changed += 1
            if len(widgets) >= 4 and widgets[3] != "cpu":
                widgets[3] = "cpu"
                changed += 1

        if node_type == "ModelPatchTorchSettings" and isinstance(widgets, list) and widgets:
            if widgets[0] is not False:
                widgets[0] = False
                changed += 1

        if node_type in DISABLED_NODE_TYPES:
            disabled_ids.add(node.get("id"))
            if node.get("mode") != 4:
                node["mode"] = 4
                changed += 1

        if node_type == "PrimitiveBoolean" and node.get("title") == "Tiled VAE":
            if isinstance(widgets, list) and widgets and widgets[0] is not False:
                widgets[0] = False
                changed += 1

    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("origin_id") not in disabled_ids:
            continue
        switch = nodes_by_id.get(link.get("target_id"))
        if not switch or switch.get("type") != "DaSiWa_NodeStatusSwitch":
            continue
        widgets = switch.get("widgets_values")
        if isinstance(widgets, list) and widgets and widgets[0] is not False:
            widgets[0] = False
            changed += 1

    return changed


def render() -> str:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    changed = sum(apply_runpod_defaults(container) for container in containers(workflow))
    if changed == 0:
        raise RuntimeError("RunPod workflow generation made no changes")
    return json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{OUTPUT} is stale; run {Path(__file__).name}")
        print(f"{OUTPUT.name} is up to date")
        return

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
