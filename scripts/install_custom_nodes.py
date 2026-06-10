import argparse
import json
import subprocess
from pathlib import Path


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def install_node(node: dict, destination: Path) -> None:
    target = destination / node["directory"]
    if target.exists():
        raise RuntimeError(f"Custom node destination already exists: {target}")

    target.mkdir(parents=True)
    run("git", "-C", str(target), "init")
    run("git", "-C", str(target), "remote", "add", "origin", node["repository"])
    run(
        "git",
        "-C",
        str(target),
        "fetch",
        "--depth=1",
        "origin",
        node["ref"],
    )
    run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")

    installed_ref = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if installed_ref != node["ref"]:
        raise RuntimeError(
            f"{node['name']} resolved to {installed_ref}, expected {node['ref']}"
        )
    print(f"Installed {node['name']} at {installed_ref}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.destination.mkdir(parents=True, exist_ok=True)
    for node in manifest["custom_nodes"]:
        install_node(node, args.destination)


if __name__ == "__main__":
    main()
