"""CLI wrapper for image describer - used by Claude Code when model can't read images directly.

Usage:
    python scripts/describe_image_cli.py <image_path> [prompt]

Example:
    python scripts/describe_image_cli.py data/baselines/golden/notepad.png
    python scripts/describe_image_cli.py screenshot.png "描述这个界面的内容"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from src.perception.image_describer import describe_image


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python describe_image_cli.py <image_path> [prompt]")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    prompt = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        img = Image.open(image_path)
        result = describe_image(img, prompt=prompt)
        if result:
            print(result)
        else:
            print("(no description returned - check API key and config)")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
