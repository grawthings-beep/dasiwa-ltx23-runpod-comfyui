import re
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    if "from kornia.core import pad" in text:
        return

    pattern = r"from kornia\.geometry\.transform\.pyramid import \((.*?)\)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match or "pad" not in match.group(1):
        return

    items = [
        item.strip()
        for item in match.group(1).split(",")
        if item.strip() and item.strip() != "pad"
    ]
    replacement = (
        "from kornia.core import pad\n"
        "from kornia.geometry.transform.pyramid import (\n    "
        + ",\n    ".join(items)
        + ",\n)"
    )
    text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
