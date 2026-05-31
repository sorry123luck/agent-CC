"""Coordinate-space helpers for VLM prompt construction.

The application canonical coordinate system is the original screenshot. VLM
prompts use the actual image sent to the model, which may be resized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image

from src.vlm.prompt_builder import PromptInput
from src.vlm.schema import PageSemanticModel


@dataclass(frozen=True)
class VLMCoordinateMeta:
    original_size: tuple[int, int]
    vlm_size: tuple[int, int]
    max_width_setting: int
    scale_x: float
    scale_y: float
    coordinate_space: str = "vlm_image"
    canonical_coordinate_space: str = "original_screenshot"
    prompt_mode: str = "candidate_annotation"
    candidate_count_before_filter: int = 0
    candidate_count_after_filter: int = 0
    sent_candidate_count: int = 0
    omitted_candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_size": list(self.original_size),
            "vlm_size": list(self.vlm_size),
            "max_width_setting": self.max_width_setting,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "coordinate_space": self.coordinate_space,
            "canonical_coordinate_space": self.canonical_coordinate_space,
            "prompt_mode": self.prompt_mode,
            "candidate_count_before_filter": self.candidate_count_before_filter,
            "candidate_count_after_filter": self.candidate_count_after_filter,
            "sent_candidate_count": self.sent_candidate_count,
            "omitted_candidate_count": self.omitted_candidate_count,
        }


def original_to_vlm_bounds(
    bounds: list[int | float] | tuple[int | float, ...],
    *,
    scale_x: float,
    scale_y: float,
) -> list[int]:
    # Preserve containment when projecting candidate boxes into the smaller VLM
    # image: upper-left floors, lower-right ceils.
    return [
        int(math.floor(float(bounds[0]) * scale_x)),
        int(math.floor(float(bounds[1]) * scale_y)),
        int(math.ceil(float(bounds[2]) * scale_x)),
        int(math.ceil(float(bounds[3]) * scale_y)),
    ]


def vlm_to_original_bounds(
    bounds: list[int | float] | tuple[int | float, ...],
    *,
    scale_x: float,
    scale_y: float,
) -> list[int]:
    inv_x = 1.0 / scale_x if scale_x else 1.0
    inv_y = 1.0 / scale_y if scale_y else 1.0
    # Mapping VLM-only areas back to canonical screenshot coordinates should
    # keep the visible area covered. Do not use round(), which can shrink boxes.
    return [
        int(math.floor(float(bounds[0]) * inv_x)),
        int(math.floor(float(bounds[1]) * inv_y)),
        int(math.ceil(float(bounds[2]) * inv_x)),
        int(math.ceil(float(bounds[3]) * inv_y)),
    ]


def _resized_image(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image
    new_h = max(1, int(round(image.height * max_width / image.width)))
    return image.resize((max_width, new_h), Image.LANCZOS)


def _coerce_bounds(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError):
        return None


def _clip_bounds(bounds: list[float], width: int, height: int) -> list[float] | None:
    l, t, r, b = bounds
    if l >= r or t >= b:
        return None
    if r <= 0 or b <= 0 or l >= width or t >= height:
        return None
    l = max(0.0, l)
    t = max(0.0, t)
    r = min(float(width), r)
    b = min(float(height), b)
    if l >= r or t >= b:
        return None
    if (r - l) < 2 or (b - t) < 2:
        return None
    return [l, t, r, b]


def _scale_candidate_list(
    candidates: list[dict[str, Any]] | None,
    *,
    original_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[list[dict[str, Any]] | None, int, int]:
    if not candidates:
        return candidates, 0, 0

    width, height = original_size
    result: list[dict[str, Any]] = []
    before = len(candidates)
    for candidate in candidates:
        c = dict(candidate)
        key = "bounds" if "bounds" in c else "bbox" if "bbox" in c else None
        if key is None:
            result.append(c)
            continue
        bounds = _coerce_bounds(c.get(key))
        clipped = _clip_bounds(bounds, width, height) if bounds else None
        if clipped is None:
            continue
        scaled = original_to_vlm_bounds(clipped, scale_x=scale_x, scale_y=scale_y)
        # Rounding can collapse tiny boxes; keep prompt candidates strictly valid.
        if scaled[0] >= scaled[2] or scaled[1] >= scaled[3]:
            continue
        c[key] = scaled
        if key == "bbox" and "bounds" in c:
            c["bounds"] = scaled
        result.append(c)
    return result, before, len(result)


def prepare_vlm_prompt_input(
    prompt_input: PromptInput,
    *,
    max_width: int = 1280,
    prompt_mode: str = "candidate_annotation",
) -> tuple[PromptInput, VLMCoordinateMeta]:
    """Return a prompt input whose screenshot/candidate bounds use VLM coords."""
    original_size = prompt_input.screenshot.size
    vlm_image = _resized_image(prompt_input.screenshot, max_width)
    vlm_size = vlm_image.size
    scale_x = vlm_size[0] / original_size[0] if original_size[0] else 1.0
    scale_y = vlm_size[1] / original_size[1] if original_size[1] else 1.0

    omni, omni_before, omni_after = _scale_candidate_list(
        prompt_input.omni_candidates,
        original_size=original_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    uia, uia_before, uia_after = _scale_candidate_list(
        prompt_input.uia_elements,
        original_size=original_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    before = omni_before + uia_before
    after = omni_after + uia_after
    prepared = PromptInput(
        screenshot=vlm_image,
        numbered_overlay_image=(
            _resized_image(prompt_input.numbered_overlay_image, max_width)
            if prompt_input.numbered_overlay_image is not None else None
        ),
        candidate_atlas_image=prompt_input.candidate_atlas_image,
        omni_candidates=omni,
        ocr_texts=prompt_input.ocr_texts,
        uia_elements=uia,
        current_page_model=prompt_input.current_page_model,
        historical_states=prompt_input.historical_states,
        task_hint=prompt_input.task_hint,
        task_mode=prompt_input.task_mode,
    )
    meta = VLMCoordinateMeta(
        original_size=original_size,
        vlm_size=vlm_size,
        max_width_setting=max_width,
        scale_x=scale_x,
        scale_y=scale_y,
        prompt_mode=prompt_mode,
        candidate_count_before_filter=before,
        candidate_count_after_filter=after,
        sent_candidate_count=after,
        omitted_candidate_count=max(0, before - after),
    )
    return prepared, meta


def _normalize_bounds_in_item(
    item: dict[str, Any],
    *,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    result = dict(item)
    bounds = _coerce_bounds(result.get("bounds"))
    if bounds is not None:
        result["bounds"] = vlm_to_original_bounds(bounds, scale_x=scale_x, scale_y=scale_y)
    return result


def _normalize_rough_area_in_item(
    item: dict[str, Any],
    *,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    result = dict(item)
    bounds = _coerce_bounds(result.get("rough_area"))
    if bounds is not None:
        result["rough_area"] = vlm_to_original_bounds(bounds, scale_x=scale_x, scale_y=scale_y)
    return result


def normalize_model_to_original(
    model: PageSemanticModel,
    meta: VLMCoordinateMeta,
) -> PageSemanticModel:
    """Map a validated VLM-image-space model back to original screenshot coords."""
    data = model.to_dict()
    for key in ("regions", "fixed_controls", "dynamic_zones"):
        data[key] = [
            _normalize_bounds_in_item(item, scale_x=meta.scale_x, scale_y=meta.scale_y)
            for item in (data.get(key) or [])
        ]
    data["missing_suggestions"] = [
        _normalize_rough_area_in_item(item, scale_x=meta.scale_x, scale_y=meta.scale_y)
        for item in (data.get("missing_suggestions") or [])
    ]
    data["image_size"] = [int(meta.original_size[0]), int(meta.original_size[1])]
    data["coordinate_space"] = meta.canonical_coordinate_space
    data.setdefault("metadata", {})
    if isinstance(data["metadata"], dict):
        data["metadata"]["vlm_raw_coordinate_space"] = meta.coordinate_space
        data["metadata"]["normalized_from_vlm_size"] = list(meta.vlm_size)
    return PageSemanticModel.from_dict(data)
