import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "workflows" / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36.json"
DST = ROOT / "workflows" / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36_runpod-default.json"

DEFAULT_UNET = "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    changed = 0

    for node in walk(data):
        if node.get("type") == "UNETLoader":
            widgets = node.get("widgets_values")
            if isinstance(widgets, list) and widgets:
                widgets[0] = DEFAULT_UNET
                changed += 1

    if changed == 0:
        raise SystemExit("No UNETLoader widget was found")

    DST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
