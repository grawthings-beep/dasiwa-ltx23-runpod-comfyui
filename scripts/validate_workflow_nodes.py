import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


FRONTEND_ONLY_TYPES = {
    "Bookmark (rgthree)",
    "Context (rgthree)",
    "Context Switch (rgthree)",
    "Label (rgthree)",
    "MarkdownNote",
    "Note",
    "PrimitiveBoolean",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveStringMultiline",
    "Reroute",
}


def required_node_types(path: Path) -> set[str]:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    subgraph_ids = {
        subgraph["id"]
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
    }
    node_types = {node["type"] for node in workflow.get("nodes", [])}
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        node_types.update(node["type"] for node in subgraph.get("nodes", []))
    return node_types - subgraph_ids - FRONTEND_ONLY_TYPES


def load_available_nodes(comfy_dir: Path) -> set[str]:
    os.chdir(comfy_dir)
    sys.path.insert(0, str(comfy_dir))
    sys.argv = [sys.argv[0], "--cpu"]

    import nodes

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=True))
    return set(nodes.NODE_CLASS_MAPPINGS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", type=Path, required=True)
    parser.add_argument("workflows", type=Path, nargs="+")
    args = parser.parse_args()

    available = load_available_nodes(args.comfy_dir.resolve())
    missing_by_workflow = {}
    for workflow in args.workflows:
        missing = sorted(required_node_types(workflow) - available)
        if missing:
            missing_by_workflow[workflow.name] = missing

    if missing_by_workflow:
        lines = ["Missing ComfyUI node types:"]
        for workflow, missing in missing_by_workflow.items():
            lines.append(f"- {workflow}: {', '.join(missing)}")
        raise SystemExit("\n".join(lines))

    print(
        f"Validated {len(args.workflows)} workflow(s) against "
        f"{len(available)} registered ComfyUI node types"
    )


if __name__ == "__main__":
    main()
