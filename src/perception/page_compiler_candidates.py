"""Core boundary-candidate builders shared by perception, execution, and debug flows."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from src.perception.page_compiler_models import (
    BoundaryCandidate,
    CandidateKind,
    InteractionCanvas,
)

_SEARCH_TOKENS = ("搜索", "查找", "筛选", "search", "find", "filter")
_BUTTON_TOKENS = ("发送", "提交", "确认", "确定", "send", "submit", "confirm", "ok", "apply")
_CANCEL_TOKENS = ("取消", "关闭", "cancel", "close")
_GENERIC_REGION_HINTS = {"", "content_area", "viewport", "container", "layout", "unknown"}


def build_boundary_candidates(
    snapshot: InteractionCanvas | dict[str, Any],
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Return weak-semantic candidates for downstream decision engines."""
    existing = _load_existing_candidates(snapshot)
    if existing is not None:
        return existing[:max_candidates] if max_candidates else existing

    candidates = _collect_boundary_candidates(snapshot)
    return candidates[:max_candidates] if max_candidates else candidates


def populate_boundary_candidate_artifacts(
    snapshot: InteractionCanvas,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Refresh snapshot artifacts/provider details with the latest boundary candidates."""
    candidates = _collect_boundary_candidates(snapshot)
    if max_candidates:
        candidates = candidates[:max_candidates]
    stats = summarize_boundary_candidates(candidates)
    snapshot.artifacts["boundary_candidates"] = candidates
    snapshot.artifacts["boundary_candidate_stats"] = stats
    snapshot.provider_trace.provider_details["boundary_candidates"] = stats
    return candidates


def summarize_boundary_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize boundary candidates for artifacts/provider trace."""
    by_source = Counter(str(candidate.get("source") or "unknown") for candidate in candidates)
    by_kind = Counter(str(candidate.get("candidate_kind") or "unknown") for candidate in candidates)
    by_origin = Counter(
        str(dict(candidate.get("attributes") or {}).get("candidate_origin") or "base")
        for candidate in candidates
    )
    high_confidence_count = sum(
        1 for candidate in candidates if float(candidate.get("confidence") or 0.0) >= 0.7
    )
    text_backed_count = sum(1 for candidate in candidates if str(candidate.get("text") or "").strip())
    return {
        "count": len(candidates),
        "by_source": dict(sorted(by_source.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "high_confidence_count": high_confidence_count,
        "text_backed_count": text_backed_count,
        "omniparser_retained_count": by_source.get("omniparser", 0),
        "vision_retained_count": sum(
            count for source, count in by_source.items() if source in {"vision", "omniparser"}
        ),
    }


def _load_existing_candidates(
    snapshot: InteractionCanvas | dict[str, Any],
) -> list[dict[str, Any]] | None:
    artifacts = snapshot.get("artifacts", {}) if isinstance(snapshot, dict) else snapshot.artifacts
    raw_candidates = artifacts.get("boundary_candidates")
    if not raw_candidates:
        return None
    normalized = [_normalize_existing_candidate(candidate) for candidate in raw_candidates]
    return _dedupe_candidates(normalized)


def _collect_boundary_candidates(
    snapshot: InteractionCanvas | dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(snapshot, dict):
        regions = {region.get("region_id"): region for region in snapshot.get("regions", [])}
        artifacts = snapshot.get("artifacts", {})
        canvas_bounds = _infer_canvas_bounds_from_dict(snapshot)
        candidates: list[dict[str, Any]] = []

        for element in snapshot.get("elements", []):
            bbox = element.get("bounds")
            if not bbox:
                continue
            attrs = dict(element.get("attributes") or {})
            region = regions.get(element.get("region_id")) or {}
            text = attrs.get("ocr_text") or element.get("text") or element.get("name") or ""
            candidates.append(
                _serialize_candidate(
                    BoundaryCandidate(
                        candidate_id=f"elem::{element.get('element_id')}",
                        bbox=tuple(bbox),
                        source=_primary_source(element.get("provider_sources") or []),
                        confidence=_candidate_confidence(element),
                        candidate_kind=_candidate_kind_from_element(
                            element.get("semantic_role"),
                            element.get("control_type"),
                        ),
                        text=_normalized_text(text),
                        control_hint=_control_hint_from_element(
                            element.get("semantic_role"),
                            element.get("control_type"),
                        ),
                        region_hint=_resolve_region_hint(
                            region.get("role"),
                            tuple(bbox),
                            regions.values(),
                            canvas_bounds,
                        ),
                        region_id=element.get("region_id"),
                        attributes={
                            "provider_sources": list(element.get("provider_sources") or []),
                            "content_group_id": element.get("content_group_id"),
                        },
                    )
                )
            )

        for index, block in enumerate(artifacts.get("ocr_blocks", [])):
            bbox = block.get("bbox")
            if not bbox:
                continue
            region_hint = _resolve_region_hint(
                None,
                tuple(bbox),
                regions.values(),
                canvas_bounds,
            )
            candidates.append(
                _serialize_candidate(
                    BoundaryCandidate(
                        candidate_id=f"ocr::{index}",
                        bbox=tuple(bbox),
                        source="ocr",
                        confidence=float(block.get("confidence") or 0.0),
                        candidate_kind=_candidate_kind_from_ocr(block.get("text"), region_hint, tuple(bbox), canvas_bounds),
                        text=_normalized_text(block.get("text")),
                        control_hint=_control_hint_from_ocr(block.get("text"), region_hint, tuple(bbox), canvas_bounds),
                        region_hint=region_hint,
                    )
                )
            )

        for index, candidate in enumerate(artifacts.get("vision_candidates", [])):
            bbox = candidate.get("bbox")
            if not bbox:
                continue
            region_hint = _resolve_region_hint(
                candidate.get("region_role"),
                tuple(bbox),
                regions.values(),
                canvas_bounds,
            )
            candidates.append(
                _serialize_candidate(
                    BoundaryCandidate(
                        candidate_id=str(candidate.get("candidate_id") or f"vision::{index}"),
                        bbox=tuple(bbox),
                        source=str(candidate.get("source") or "vision"),
                        confidence=float(candidate.get("confidence") or 0.0),
                        candidate_kind=_candidate_kind_from_vision(candidate.get("kind"), region_hint),
                        text=_normalized_text(candidate.get("text")),
                        control_hint=_control_hint_from_vision(candidate.get("kind"), region_hint),
                        region_hint=region_hint,
                        attributes={
                            "group_id": candidate.get("group_id"),
                            "icon_type": candidate.get("icon_type"),
                        },
                    )
                )
            )

        _append_layout_region_candidates(candidates, artifacts, regions.values(), canvas_bounds)
        _append_control_group_candidates(candidates, artifacts, regions.values(), canvas_bounds)
        return _dedupe_candidates(candidates)

    snapshot_regions = {region.region_id: region for region in snapshot.regions}
    canvas_bounds = _infer_canvas_bounds_from_snapshot(snapshot)
    candidates: list[dict[str, Any]] = []

    for element in snapshot.elements:
        if not element.bounds:
            continue
        region = snapshot_regions.get(element.region_id or "")
        text = element.attributes.get("ocr_text") or element.text or element.name or ""
        candidates.append(
            _serialize_candidate(
                BoundaryCandidate(
                    candidate_id=f"elem::{element.element_id}",
                    bbox=tuple(element.bounds),
                    source=_primary_source(element.provider_sources),
                    confidence=_candidate_confidence(
                        {
                            "provider_sources": list(element.provider_sources),
                            "locator_ids": list(element.locator_ids),
                            "anchor_ids": list(element.anchor_ids),
                            "attributes": dict(element.attributes),
                        }
                    ),
                    candidate_kind=_candidate_kind_from_element(
                        element.semantic_role.value if element.semantic_role else None,
                        element.control_type,
                    ),
                    text=_normalized_text(text),
                    control_hint=_control_hint_from_element(
                        element.semantic_role.value if element.semantic_role else None,
                        element.control_type,
                    ),
                    region_hint=_resolve_region_hint(
                        region.role if region else None,
                        tuple(element.bounds),
                        snapshot.regions,
                        canvas_bounds,
                    ),
                    region_id=element.region_id,
                    attributes={
                        "provider_sources": list(element.provider_sources),
                        "content_group_id": element.content_group_id,
                    },
                )
            )
        )

    for index, block in enumerate(snapshot.artifacts.get("ocr_blocks", [])):
        bbox = block.get("bbox")
        if not bbox:
            continue
        region_hint = _resolve_region_hint(None, tuple(bbox), snapshot.regions, canvas_bounds)
        candidates.append(
            _serialize_candidate(
                BoundaryCandidate(
                    candidate_id=f"ocr::{index}",
                    bbox=tuple(bbox),
                    source="ocr",
                    confidence=float(block.get("confidence") or 0.0),
                    candidate_kind=_candidate_kind_from_ocr(block.get("text"), region_hint, tuple(bbox), canvas_bounds),
                    text=_normalized_text(block.get("text")),
                    control_hint=_control_hint_from_ocr(block.get("text"), region_hint, tuple(bbox), canvas_bounds),
                    region_hint=region_hint,
                )
            )
        )

    for index, candidate in enumerate(snapshot.artifacts.get("vision_candidates", [])):
        bbox = candidate.get("bbox")
        if not bbox:
            continue
        region_hint = _resolve_region_hint(
            candidate.get("region_role"),
            tuple(bbox),
            snapshot.regions,
            canvas_bounds,
        )
        candidates.append(
            _serialize_candidate(
                BoundaryCandidate(
                    candidate_id=str(candidate.get("candidate_id") or f"vision::{index}"),
                    bbox=tuple(bbox),
                    source=str(candidate.get("source") or "vision"),
                    confidence=float(candidate.get("confidence") or 0.0),
                    candidate_kind=_candidate_kind_from_vision(candidate.get("kind"), region_hint),
                    text=_normalized_text(candidate.get("text")),
                    control_hint=_control_hint_from_vision(candidate.get("kind"), region_hint),
                    region_hint=region_hint,
                    attributes={
                        "group_id": candidate.get("group_id"),
                        "icon_type": candidate.get("icon_type"),
                    },
                )
            )
        )

    _append_layout_region_candidates(candidates, snapshot.artifacts, snapshot.regions, canvas_bounds)
    _append_control_group_candidates(candidates, snapshot.artifacts, snapshot.regions, canvas_bounds)
    return _dedupe_candidates(candidates)


def _normalize_existing_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "bbox": list(candidate.get("bbox") or []),
        "source": str(candidate.get("source") or "unknown"),
        "confidence": float(candidate.get("confidence") or 0.0),
        "candidate_kind": str(candidate.get("candidate_kind") or CandidateKind.UNKNOWN.value),
        "text": _normalized_text(candidate.get("text")),
        "control_hint": str(candidate.get("control_hint") or ""),
        "region_hint": str(candidate.get("region_hint") or ""),
        "region_id": candidate.get("region_id"),
        "attributes": dict(candidate.get("attributes") or {}),
    }


def _serialize_candidate(candidate: BoundaryCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "bbox": list(candidate.bbox),
        "source": candidate.source,
        "confidence": candidate.confidence,
        "candidate_kind": candidate.candidate_kind.value,
        "text": candidate.text,
        "control_hint": candidate.control_hint,
        "region_hint": candidate.region_hint,
        "region_id": candidate.region_id,
        "attributes": dict(candidate.attributes),
    }


def _normalized_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _folded_text(value: Any) -> str:
    return _normalized_text(value).lower()


def _primary_source(provider_sources: list[str]) -> str:
    if not provider_sources:
        return "unknown"
    if "merged" in provider_sources:
        return "merged"
    return provider_sources[0]


def _candidate_confidence(item: dict[str, Any]) -> float:
    provider_sources = list(item.get("provider_sources") or [])
    score = 0.35
    if provider_sources:
        score += 0.15
    if item.get("locator_ids"):
        score += 0.15
    if item.get("anchor_ids"):
        score += 0.1
    attrs = dict(item.get("attributes") or {})
    evidence = attrs.get("structure_evidence_score")
    if isinstance(evidence, (int, float)):
        score += min(max(float(evidence), 0.0), 1.0) * 0.2
    return round(min(score, 0.99), 3)


def _candidate_kind_from_element(
    semantic_role: str | None,
    control_type: str | None,
) -> CandidateKind:
    role = str(semantic_role or "").strip().lower()
    control = str(control_type or "").strip().lower()
    if "input" in role or "edit" in control:
        return CandidateKind.INPUT
    if "button" in role or "button" in control:
        return CandidateKind.BUTTON
    if "item" in role or "listitem" in control:
        return CandidateKind.LIST_ITEM
    if "image" in role:
        return CandidateKind.IMAGE
    if "title" in role or "header" in role:
        return CandidateKind.HEADER
    if "pane" in control or "container" in role or "sidebar" in role:
        return CandidateKind.PANEL
    if "icon" in role:
        return CandidateKind.ICON
    return CandidateKind.TEXT if role == "text" else CandidateKind.UNKNOWN


def _control_hint_from_element(
    semantic_role: str | None,
    control_type: str | None,
) -> str:
    role = str(semantic_role or "").strip().lower()
    control = str(control_type or "").strip().lower()
    if "input" in role or "edit" in control:
        return "edit_like"
    if "button" in role or "button" in control:
        return "button_like"
    if "item" in role or "listitem" in control:
        return "list_item_like"
    if "pane" in control:
        return "pane_like"
    if "image" in role:
        return "image_like"
    if "header" in role or "title" in role:
        return "header_like"
    return "text_like"


def _candidate_kind_from_vision(kind: Any, region_hint: str = "") -> CandidateKind:
    value = str(kind or "").strip().lower()
    region_value = str(region_hint or "").strip().lower()
    if "input" in value or region_value in {"top_left_filter", "bottom_input"}:
        return CandidateKind.INPUT
    if "button" in value or region_value in {"bottom_right_action", "top_header_bar"}:
        return CandidateKind.BUTTON
    if "icon" in value:
        return CandidateKind.ICON
    if "item" in value or "list" in value or "card" in value:
        return CandidateKind.LIST_ITEM
    if "panel" in value or "toolbar" in value or "bar" in value or "viewport" in value:
        return CandidateKind.PANEL
    if "image" in value:
        return CandidateKind.IMAGE
    if "text" in value:
        return CandidateKind.TEXT
    return CandidateKind.UNKNOWN


def _control_hint_from_vision(kind: Any, region_hint: str = "") -> str:
    value = str(kind or "").strip().lower()
    region_value = str(region_hint or "").strip().lower()
    if "input" in value or region_value in {"top_left_filter", "bottom_input"}:
        return "edit_like"
    if "button" in value or region_value in {"bottom_right_action", "top_header_bar"}:
        return "button_like"
    if "item" in value or "list" in value or "card" in value:
        return "list_item_like"
    if "toolbar" in value or "header" in value:
        return "header_like"
    if "panel" in value or "bar" in value or "viewport" in value:
        return "pane_like"
    if "image" in value:
        return "image_like"
    if "icon" in value:
        return "icon"
    return str(kind or "vision_like")


def _append_layout_region_candidates(
    candidates: list[dict[str, Any]],
    artifacts: dict[str, Any],
    regions: Any,
    canvas_bounds: tuple[int, int, int, int] | None,
) -> None:
    provider = _vision_provider_name(artifacts)
    for index, region in enumerate(artifacts.get("vision_layout_regions", [])):
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        role = str(region.get("role") or region.get("region_role") or "vision_region")
        candidates.append(
            _serialize_candidate(
                BoundaryCandidate(
                    candidate_id=str(region.get("region_id") or f"vision_region::{index}"),
                    bbox=tuple(bbox),
                    source=provider,
                    confidence=float(region.get("confidence") or 0.55),
                    candidate_kind=_candidate_kind_from_layout_role(role),
                    text=_normalized_text(region.get("label") or region.get("title") or role),
                    control_hint=f"{role}_like",
                    region_hint=_resolve_region_hint(role, tuple(bbox), regions, canvas_bounds),
                    attributes={
                        "candidate_origin": "vision_layout_region",
                        "raw_role": role,
                    },
                )
            )
        )


def _append_control_group_candidates(
    candidates: list[dict[str, Any]],
    artifacts: dict[str, Any],
    regions: Any,
    canvas_bounds: tuple[int, int, int, int] | None,
) -> None:
    provider = _vision_provider_name(artifacts)
    vision_by_id = {
        str(item.get("candidate_id")): item
        for item in artifacts.get("vision_candidates", [])
        if item.get("candidate_id") and item.get("bbox")
    }
    for index, group in enumerate(artifacts.get("vision_control_groups", [])):
        member_ids = list(group.get("member_ids") or group.get("members") or [])
        member_boxes = []
        for member_id in member_ids:
            candidate = vision_by_id.get(str(member_id))
            if candidate and candidate.get("bbox"):
                member_boxes.append(tuple(candidate.get("bbox")))
        bbox = _union_bbox(member_boxes)
        if bbox is None:
            raw_bbox = group.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                continue
            bbox = tuple(raw_bbox)
        group_id = str(group.get("group_id") or f"vision_group::{index}")
        candidates.append(
            _serialize_candidate(
                BoundaryCandidate(
                    candidate_id=f"vision_group::{group_id}",
                    bbox=bbox,
                    source=provider,
                    confidence=float(group.get("confidence") or 0.58),
                    candidate_kind=CandidateKind.PANEL,
                    text=_normalized_text(group.get("label") or group.get("title") or group_id),
                    control_hint="group_like",
                    region_hint=_resolve_region_hint(None, bbox, regions, canvas_bounds),
                    attributes={
                        "candidate_origin": "vision_control_group",
                        "group_id": group_id,
                        "member_ids": member_ids,
                    },
                )
            )
        )


def _candidate_kind_from_ocr(
    text: Any,
    region_hint: str,
    bbox: tuple[int, int, int, int],
    canvas_bounds: tuple[int, int, int, int] | None,
) -> CandidateKind:
    text_value = _folded_text(text)
    raw_text = _normalized_text(text)
    region_value = str(region_hint or "").strip().lower()
    width_ratio, height_ratio = _bbox_relative_size(bbox, canvas_bounds)
    if any(token in text_value for token in _SEARCH_TOKENS):
        return CandidateKind.INPUT
    if any(token in text_value for token in _BUTTON_TOKENS + _CANCEL_TOKENS):
        return CandidateKind.BUTTON
    if region_value in {"top_left_filter", "filter_bar", "bottom_input", "composer_area"}:
        return CandidateKind.INPUT
    if region_value in {"bottom_right_action", "action_bar"}:
        return CandidateKind.BUTTON
    if region_value in {"left_panel_list", "list_panel"} and raw_text:
        return CandidateKind.LIST_ITEM
    if region_value in {"top_header_bar", "title_bar"} and raw_text and len(raw_text) <= 24:
        return CandidateKind.HEADER
    if region_value in {"main_content", "detail_panel"} and len(raw_text) <= 20 and width_ratio <= 0.25:
        return CandidateKind.LIST_ITEM
    if region_value == "top_header_bar" and len(raw_text) <= 12 and height_ratio <= 0.08:
        return CandidateKind.BUTTON
    return CandidateKind.TEXT


def _control_hint_from_ocr(
    text: Any,
    region_hint: str,
    bbox: tuple[int, int, int, int],
    canvas_bounds: tuple[int, int, int, int] | None,
) -> str:
    text_value = _folded_text(text)
    raw_text = _normalized_text(text)
    region_value = str(region_hint or "").strip().lower()
    _, height_ratio = _bbox_relative_size(bbox, canvas_bounds)
    if any(token in text_value for token in _SEARCH_TOKENS):
        return "edit_like"
    if any(token in text_value for token in _BUTTON_TOKENS + _CANCEL_TOKENS):
        return "button_like"
    if region_value in {"top_left_filter", "filter_bar", "bottom_input", "composer_area"}:
        return "edit_like"
    if region_value in {"bottom_right_action", "action_bar"}:
        return "button_like"
    if region_value in {"left_panel_list", "list_panel"} and raw_text:
        return "list_item_like"
    if region_value in {"top_header_bar", "title_bar"} and raw_text and len(raw_text) <= 24:
        return "header_like"
    if region_value == "top_header_bar" and len(raw_text) <= 12 and height_ratio <= 0.08:
        return "button_like"
    return "text_like"


def _candidate_kind_from_layout_role(role: str) -> CandidateKind:
    value = str(role or "").strip().lower()
    if "toolbar" in value or "header" in value or "title" in value:
        return CandidateKind.HEADER
    if "panel" in value or "viewport" in value or "sidebar" in value or "bar" in value:
        return CandidateKind.PANEL
    if "list" in value or "grid" in value:
        return CandidateKind.LIST_ITEM
    return CandidateKind.PANEL


def _vision_provider_name(artifacts: dict[str, Any]) -> str:
    provider = (artifacts.get("vision_provider") or {}).get("provider")
    if provider:
        return str(provider)
    for candidate in artifacts.get("vision_candidates", []):
        if candidate.get("source"):
            return str(candidate.get("source"))
    return "vision"


def _union_bbox(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return (left, top, right, bottom)


def _infer_canvas_bounds_from_dict(snapshot: dict[str, Any]) -> tuple[int, int, int, int] | None:
    window = dict(snapshot.get("window") or {})
    for key in ("rect_client", "rect_screen"):
        rect = window.get(key)
        if rect and len(rect) == 4:
            return tuple(rect)
    boxes = _collect_bbox_sources(
        [region.get("bounds") for region in snapshot.get("regions", []) if region.get("bounds")],
        [element.get("bounds") for element in snapshot.get("elements", []) if element.get("bounds")],
        [block.get("bbox") for block in (snapshot.get("artifacts", {}) or {}).get("ocr_blocks", []) if block.get("bbox")],
        [candidate.get("bbox") for candidate in (snapshot.get("artifacts", {}) or {}).get("vision_candidates", []) if candidate.get("bbox")],
        [region.get("bbox") for region in (snapshot.get("artifacts", {}) or {}).get("vision_layout_regions", []) if region.get("bbox")],
        [group.get("bbox") for group in (snapshot.get("artifacts", {}) or {}).get("vision_control_groups", []) if group.get("bbox")],
    )
    return _union_bbox(boxes)


def _infer_canvas_bounds_from_snapshot(snapshot: InteractionCanvas) -> tuple[int, int, int, int] | None:
    window = snapshot.window
    if window is not None:
        for rect in (window.rect_client, window.rect_screen):
            if rect and len(rect) == 4:
                return tuple(rect)
    boxes = _collect_bbox_sources(
        [region.bounds for region in snapshot.regions if region.bounds],
        [element.bounds for element in snapshot.elements if element.bounds],
        [block.get("bbox") for block in snapshot.artifacts.get("ocr_blocks", []) if block.get("bbox")],
        [candidate.get("bbox") for candidate in snapshot.artifacts.get("vision_candidates", []) if candidate.get("bbox")],
        [region.get("bbox") for region in snapshot.artifacts.get("vision_layout_regions", []) if region.get("bbox")],
        [group.get("bbox") for group in snapshot.artifacts.get("vision_control_groups", []) if group.get("bbox")],
    )
    return _union_bbox(boxes)


def _collect_bbox_sources(*groups: list[Any]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        for box in group:
            if box and len(box) == 4:
                boxes.append(tuple(int(value) for value in box))
    return boxes


def _resolve_region_hint(
    preferred_hint: Any,
    bbox: tuple[int, int, int, int],
    regions: Any,
    canvas_bounds: tuple[int, int, int, int] | None,
) -> str:
    preferred = str(preferred_hint or "").strip()
    if preferred and preferred.lower() not in _GENERIC_REGION_HINTS:
        return preferred

    generic_match = ""
    left, top, right, bottom = bbox
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    for region in regions:
        bounds = getattr(region, "bounds", None)
        role = getattr(region, "role", None)
        if isinstance(region, dict):
            bounds = region.get("bounds")
            role = region.get("role")
        if not bounds:
            continue
        rl, rt, rr, rb = bounds
        if rl <= center_x <= rr and rt <= center_y <= rb:
            role_value = str(role or "").strip()
            if role_value and role_value.lower() not in _GENERIC_REGION_HINTS:
                return role_value
            generic_match = role_value
            break

    geometry_hint = _coarse_region_hint(bbox, canvas_bounds)
    if geometry_hint:
        return geometry_hint
    return generic_match


def _coarse_region_hint(
    bbox: tuple[int, int, int, int],
    canvas_bounds: tuple[int, int, int, int] | None,
) -> str:
    if not canvas_bounds:
        return ""
    left, top, right, bottom = bbox
    cl, ct, cr, cb = canvas_bounds
    width = max(cr - cl, 1)
    height = max(cb - ct, 1)
    center_x = ((left + right) / 2 - cl) / width
    center_y = ((top + bottom) / 2 - ct) / height
    box_width = max(right - left, 1) / width
    box_height = max(bottom - top, 1) / height

    if center_x <= 0.08:
        return "left_icon_rail"
    if center_y <= 0.12 and center_x > 0.28:
        return "top_header_bar"
    if center_x <= 0.36:
        if center_x <= 0.28 and center_y <= 0.12:
            return "top_left_filter"
        if center_y >= 0.84:
            return "left_bottom_panel"
        return "left_panel_list"
    if center_y <= 0.16:
        if box_width >= 0.18 and box_height <= 0.10:
            return "top_header_bar"
        return "top_header_bar"
    if center_y >= 0.82:
        if center_x >= 0.82:
            return "bottom_right_action"
        return "bottom_input"
    return "main_content"


def _bbox_relative_size(
    bbox: tuple[int, int, int, int],
    canvas_bounds: tuple[int, int, int, int] | None,
) -> tuple[float, float]:
    if not canvas_bounds:
        return (0.0, 0.0)
    left, top, right, bottom = bbox
    cl, ct, cr, cb = canvas_bounds
    width = max(cr - cl, 1)
    height = max(cb - ct, 1)
    return ((right - left) / width, (bottom - top) / height)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        attributes = dict(candidate.get("attributes") or {})
        key = (
            candidate.get("candidate_id") or "",
            tuple(candidate.get("bbox") or []),
            candidate.get("text") or "",
            candidate.get("source") or "",
            candidate.get("candidate_kind") or "",
            candidate.get("control_hint") or "",
            candidate.get("region_id") or "",
            attributes.get("candidate_origin") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
