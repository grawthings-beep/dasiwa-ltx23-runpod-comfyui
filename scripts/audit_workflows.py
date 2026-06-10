import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "custom_nodes.lock.json"
DEFAULT_WORKFLOWS = ROOT / "workflows"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def workflow_nodes(workflow: dict):
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def load_manifest(path: Path) -> tuple[dict, set[str], set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    directories = set()
    cnr_ids = set()
    provided_types = set()

    for node in manifest.get("custom_nodes", []):
        directory = node["directory"]
        if directory in directories:
            raise ValueError(f"Duplicate custom node directory: {directory}")
        directories.add(directory)

        ref = node["ref"]
        if not SHA_PATTERN.fullmatch(ref):
            raise ValueError(f"{node['name']} is not pinned to a full commit SHA: {ref}")

        cnr_ids.update(value.lower() for value in node.get("cnr_ids", []))
        provided_types.update(node.get("provides", []))

    return manifest, cnr_ids, provided_types


def audit_workflow(path: Path, cnr_ids: set[str], provided_types: set[str]) -> list[str]:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    errors = []

    for node in workflow_nodes(workflow):
        node_type = node.get("type", "<unknown>")
        cnr_id = node.get("properties", {}).get("cnr_id")
        if cnr_id and cnr_id.lower() != "comfy-core" and cnr_id.lower() not in cnr_ids:
            errors.append(f"{path.name}: {node_type} requires unlisted package {cnr_id}")

        if node_type.startswith("DaSiWa_") and node_type not in provided_types:
            errors.append(f"{path.name}: {node_type} is not assigned to a locked package")

        if node_type in {"LTXDirector", "LTXDirectorGuide"} and node_type not in provided_types:
            errors.append(f"{path.name}: {node_type} is not assigned to a locked package")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workflows", type=Path, default=DEFAULT_WORKFLOWS)
    args = parser.parse_args()

    _, cnr_ids, provided_types = load_manifest(args.manifest)
    workflow_paths = sorted(args.workflows.glob("*.json"))
    if not workflow_paths:
        raise SystemExit(f"No workflows found in {args.workflows}")

    errors = []
    for path in workflow_paths:
        errors.extend(audit_workflow(path, cnr_ids, provided_types))

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Audited {len(workflow_paths)} workflows against "
        f"{len(cnr_ids)} locked Comfy Registry package IDs"
    )


if __name__ == "__main__":
    main()
