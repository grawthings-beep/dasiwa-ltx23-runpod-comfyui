import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
SRC = WORKFLOWS / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36.json"
DST = WORKFLOWS / "DasiwaLTX23WorkflowsI2VFLF2V_omniforgeCLTX23V36_runpod-no-ltx-tiled-decode.json"

MISSING_NODE_TYPE = "LTXVSpatioTemporalTiledVAEDecode"


def walk_containers(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_containers(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_containers(child)


def remove_missing_tiled_node(container):
    nodes = container.get("nodes")
    links = container.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        return False

    missing_nodes = [node for node in nodes if node.get("type") == MISSING_NODE_TYPE]
    if not missing_nodes:
        return False

    missing_ids = {node["id"] for node in missing_nodes}
    removed_link_ids = {
        link["id"]
        for link in links
        if link.get("origin_id") in missing_ids or link.get("target_id") in missing_ids
    }

    # The workflow's VAE switch defaults to normal VAEDecode. When removing the
    # optional old tiled decode branch, leave that switch on the normal path.
    for node in nodes:
        if node.get("type") == "ComfySwitchNode" and node.get("title") == "Switch VAE":
            widgets = node.get("widgets_values")
            if isinstance(widgets, list) and widgets:
                widgets[0] = False
            fallback_link = None
            for input_ in node.get("inputs", []):
                if input_.get("name") == "on_false":
                    fallback_link = input_.get("link")
            for input_ in node.get("inputs", []):
                if input_.get("link") in removed_link_ids:
                    input_["link"] = fallback_link

    for node in nodes:
        for output in node.get("outputs", []):
            if isinstance(output.get("links"), list):
                output["links"] = [link_id for link_id in output["links"] if link_id not in removed_link_ids]

    container["nodes"] = [node for node in nodes if node.get("id") not in missing_ids]
    container["links"] = [link for link in links if link.get("id") not in removed_link_ids]
    return True


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    changed = 0
    for container in walk_containers(data):
        if remove_missing_tiled_node(container):
            changed += 1

    if changed == 0:
        raise SystemExit(f"No {MISSING_NODE_TYPE} node found")

    DST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
