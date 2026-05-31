"""Apply candidate correction overlays to runtime canvases.

The raw InteractionCanvas remains the perception record. This module builds an
effective copy for consumers that should trust manual/agent/VLM corrections.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from src.memory.candidate_override_store import CandidateOverrideData, CandidateOverrideStore
from src.perception.page_compiler_models import Candidate, InteractionCanvas, SemanticRole
from src.storage.db import Session
from sqlalchemy.exc import OperationalError


def _canvas_size(canvas: InteractionCanvas) -> tuple[int, int]:
    width = 0
    height = 0

    if canvas.window:
        if canvas.window.rect_client:
            left, top, right, bottom = canvas.window.rect_client
            width = max(width, right - left)
            height = max(height, bottom - top)
        if (width <= 0 or height <= 0) and canvas.window.rect_screen:
            left, top, right, bottom = canvas.window.rect_screen
            width = max(width, right - left)
            height = max(height, bottom - top)

    if width <= 0 or height <= 0:
        for element in canvas.elements:
            if not element.bounds or len(element.bounds) < 4:
                continue
            width = max(width, int(element.bounds[2]))
            height = max(height, int(element.bounds[3]))

    return max(width, 1), max(height, 1)


def _relative_to_bounds(relative_bounds: list[float], canvas: InteractionCanvas) -> tuple[int, int, int, int] | None:
    if len(relative_bounds) < 4:
        return None
    width, height = _canvas_size(canvas)
    x1, y1, x2, y2 = relative_bounds[:4]
    bounds = (
        round(max(0.0, min(1.0, x1)) * width),
        round(max(0.0, min(1.0, y1)) * height),
        round(max(0.0, min(1.0, x2)) * width),
        round(max(0.0, min(1.0, y2)) * height),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    return bounds


def scoped_stable_key(canvas: InteractionCanvas, stable_key_id: str) -> str:
    """Return the page-scoped stable override key used by the console."""
    page_model_id = getattr(canvas, "page_model_id", None)
    if page_model_id:
        return f"stable:{page_model_id}:{stable_key_id}"
    return f"stable:{stable_key_id}"


def candidate_override_keys(candidate: Candidate, canvas: InteractionCanvas) -> list[str]:
    """Return override scope keys from most specific to legacy-compatible."""
    keys: list[str] = []
    if candidate.stable_key_id:
        scoped = scoped_stable_key(canvas, candidate.stable_key_id)
        keys.append(scoped)
        legacy = f"stable:{candidate.stable_key_id}"
        if legacy != scoped:
            keys.append(legacy)

    if canvas.canvas_id and candidate.element_id:
        keys.append(f"canvas:{canvas.canvas_id}:{candidate.element_id}")

    state_template_id = getattr(canvas, "state_template_id", None)
    if state_template_id and candidate.element_id:
        keys.append(f"state:{state_template_id}:transient_{candidate.element_id}")

    return keys


_SOURCE_PRIORITY: dict[str, int] = {
    "manual": 0,
    "agent": 1,
    "vlm_semantic": 2,
}


def _pick_override(
    candidate: Candidate,
    canvas: InteractionCanvas,
    overrides_by_key: dict[str, CandidateOverrideData],
) -> CandidateOverrideData | None:
    matches: list[CandidateOverrideData] = []
    for key in candidate_override_keys(candidate, canvas):
        override = overrides_by_key.get(key)
        if override and override.status == "active":
            matches.append(override)
    if not matches:
        return None
    # manual > agent > vlm_semantic (lower number = higher priority)
    matches.sort(key=lambda o: _SOURCE_PRIORITY.get(o.source or "manual", 99))
    return matches[0]


def _coerce_semantic_role(value: str) -> SemanticRole | str:
    try:
        return SemanticRole(value)
    except ValueError:
        return value


def _apply_override_to_candidate(
    candidate: Candidate,
    canvas: InteractionCanvas,
    override: CandidateOverrideData,
) -> Candidate | None:
    if override.kind == "ignored":
        return None

    updated = deepcopy(candidate)
    changed = False

    if override.label is not None:
        updated.role_label = override.label
        changed = True

    if override.semantic_role is not None:
        updated.semantic_role = _coerce_semantic_role(override.semantic_role)
        changed = True

    if override.visual_type is not None:
        updated.visual_type = override.visual_type
        changed = True

    if override.region_id is not None:
        updated.region_id = override.region_id
        changed = True

    if override.relative_bounds is not None:
        bounds = _relative_to_bounds(override.relative_bounds, canvas)
        if bounds is not None:
            updated.bounds = bounds
            updated.click_point = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
            changed = True

    if override.absolute_bounds is not None and len(override.absolute_bounds) >= 4:
        left, top, right, bottom = [int(round(v)) for v in override.absolute_bounds[:4]]
        if right > left and bottom > top:
            updated.bounds = (left, top, right, bottom)
            updated.click_point = ((left + right) // 2, (top + bottom) // 2)
            changed = True

    if override.kind is not None:
        updated.attributes = {**updated.attributes, "manual_kind": override.kind}
        changed = True

    if changed:
        updated.refine_status = "refined"
        updated.role_source = override.source or "manual"
        updated.role_confidence = 1.0
        if "manual" not in updated.provider_sources:
            updated.provider_sources = [*updated.provider_sources, "manual"]

    return updated


def apply_candidate_overrides(
    canvas: InteractionCanvas,
    overrides: Iterable[CandidateOverrideData] | None = None,
) -> InteractionCanvas:
    """Return an effective canvas copy with active candidate overrides applied."""
    if overrides is None:
        store = CandidateOverrideStore()
        try:
            with Session() as session:
                overrides = store.list(
                    session,
                    canvas_id=canvas.canvas_id,
                    state_template_id=getattr(canvas, "state_template_id", None),
                    page_model_id=getattr(canvas, "page_model_id", None),
                )
        except OperationalError as exc:
            if "candidate_overrides" not in str(exc):
                raise
            overrides = []

    overrides_by_key = {
        override.scope_key: override
        for override in overrides
        if override.status == "active"
    }
    if not overrides_by_key:
        return canvas

    effective = deepcopy(canvas)
    effective.elements = []
    for candidate in canvas.elements:
        override = _pick_override(candidate, canvas, overrides_by_key)
        if override is None:
            effective.elements.append(deepcopy(candidate))
            continue
        updated = _apply_override_to_candidate(candidate, canvas, override)
        if updated is not None:
            effective.elements.append(updated)
    return effective
