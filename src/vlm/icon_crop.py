"""Helpers for icon-crop VLM understanding tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class IconCrop:
    candidate_id: str
    image: Image.Image
    bounds: list[int]
    dhash: str


def crop_candidate_icon(
    screenshot: Image.Image,
    element: Any,
    *,
    padding: int = 4,
) -> IconCrop | None:
    """Crop a candidate from the screenshot with small padding.

    Returned bounds are relative to the crop because icon prompts operate on
    the crop image, not the full original screenshot.
    """
    bounds = getattr(element, "bounds", None)
    if not bounds or len(bounds) != 4:
        return None
    try:
        left, top, right, bottom = [int(round(float(v))) for v in bounds]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None

    width, height = screenshot.size
    crop_left = max(0, left - padding)
    crop_top = max(0, top - padding)
    crop_right = min(width, right + padding)
    crop_bottom = min(height, bottom + padding)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None

    crop = screenshot.crop((crop_left, crop_top, crop_right, crop_bottom))
    rel_bounds = [
        max(0, left - crop_left),
        max(0, top - crop_top),
        min(crop.width, right - crop_left),
        min(crop.height, bottom - crop_top),
    ]
    if rel_bounds[2] <= rel_bounds[0] or rel_bounds[3] <= rel_bounds[1]:
        return None
    return IconCrop(
        candidate_id=str(getattr(element, "element_id", "")),
        image=crop,
        bounds=rel_bounds,
        dhash=image_dhash(crop),
    )


def image_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a small deterministic dHash for crop de-duplication metadata."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    bits: list[str] = []
    for y in range(hash_size):
        for x in range(hash_size):
            bits.append("1" if gray.getpixel((x, y)) > gray.getpixel((x + 1, y)) else "0")
    value = int("".join(bits), 2)
    return f"{value:0{hash_size * hash_size // 4}x}"
