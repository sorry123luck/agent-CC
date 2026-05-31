"""Formal OpenClaw payload / decision helpers for runtime and tooling."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from src.perception.page_compiler_candidates import build_boundary_candidates
from src.perception.page_compiler_models import (
    DecisionAction,
    DecisionAlternative,
    DecisionRecord,
    InteractionCanvas,
)


def build_openclaw_payload(
    snapshot: InteractionCanvas | dict[str, Any],
    task: str,
    max_candidates: int = 120,
) -> dict[str, Any]:
    """Build the weak-structure payload sent to OpenClaw."""
    all_candidates = build_boundary_candidates(snapshot, max_candidates=None)
    ranked_candidates = _rank_candidates_for_openclaw(all_candidates)
    candidates = _select_candidates_for_openclaw(
        ranked_candidates,
        max_candidates=max_candidates,
    )
    if isinstance(snapshot, dict):
        surface_type = snapshot.get("surface", {}).get("surface_type")
        page_class = snapshot.get("page", {}).get("page_class")
        regions = snapshot.get("regions", [])
        app_name = (((snapshot.get("app") or {}).get("process_name")) or "").replace(".exe", "")
    else:
        surface_type = snapshot.surface.surface_type.value
        page_class = snapshot.page.page_class
        regions = snapshot.regions
        app_name = (snapshot.app.process_name or "").replace(".exe", "")

    return {
        "task": task,
        "app": app_name or "unknown",
        "surface_type": surface_type,
        "page_hint": page_class,
        "window_bounds": _resolve_window_bounds(snapshot),
        "regions": _serialize_regions_for_openclaw(regions),
        "candidates": candidates,
        "candidate_summary": _summarize_candidates_for_openclaw(
            candidates,
            total_candidates=all_candidates,
        ),
        "provider_health": _serialize_provider_health(
            snapshot,
            all_candidates=all_candidates,
            delivered_candidates=candidates,
        ),
        "ocr_blocks": _serialize_ocr_blocks(snapshot),
        "notes": [
            "这是弱清洗输入，只提供候选框、弱区域和弱控件提示。",
            "如果信息不足，可返回 focus_bbox 并要求局部重采样。",
        ],
    }


def normalize_decision_record(decision: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize an OpenClaw decision JSON to the repo's canonical decision record."""
    if isinstance(decision, str):
        decision = json.loads(decision)

    record = DecisionRecord(
        page_state=str(decision.get("page_state") or ""),
        task_understanding=str(decision.get("task_understanding") or ""),
        selected_candidate_id=str(decision.get("selected_candidate_id") or ""),
        selected_role=str(decision.get("selected_role") or ""),
        bbox=_coerce_bbox_list(decision.get("bbox")),
        reason=str(decision.get("reason") or ""),
        alternatives=[
            DecisionAlternative(
                candidate_id=str(item.get("candidate_id") or ""),
                selected_role=str(item.get("selected_role") or ""),
                reject_reason=str(item.get("reject_reason") or ""),
            )
            for item in list(decision.get("alternatives") or [])
        ],
        next_action=DecisionAction(
            type=str((decision.get("next_action") or {}).get("type") or ""),
            text=str((decision.get("next_action") or {}).get("text") or ""),
        ),
        confidence=float(decision.get("confidence") or 0.0),
        decision_status=str(decision.get("decision_status") or "ready"),
        focus_bbox=_coerce_bbox_list(decision.get("focus_bbox")),
    )
    return {
        "page_state": record.page_state,
        "task_understanding": record.task_understanding,
        "selected_candidate_id": record.selected_candidate_id,
        "selected_role": record.selected_role,
        "bbox": list(record.bbox) if record.bbox else None,
        "reason": record.reason,
        "alternatives": [
            {
                "candidate_id": item.candidate_id,
                "selected_role": item.selected_role,
                "reject_reason": item.reject_reason,
            }
            for item in record.alternatives
        ],
        "next_action": {
            "type": record.next_action.type,
            "text": record.next_action.text,
        },
        "confidence": record.confidence,
        "decision_status": record.decision_status,
        "focus_bbox": list(record.focus_bbox) if record.focus_bbox else None,
    }


def _resolve_window_bounds(snapshot: InteractionCanvas | dict[str, Any]) -> list[int] | None:
    if isinstance(snapshot, dict):
        window = snapshot.get("window") or {}
        rect = window.get("rect_client") or window.get("rect_screen")
        if rect:
            return list(rect)
        regions = snapshot.get("regions", [])
        return _window_bounds_from_regions(regions)
    window = snapshot.window
    if window:
        rect = window.rect_client or window.rect_screen
        if rect:
            return list(rect)
    return _window_bounds_from_regions(snapshot.regions)


def _window_bounds_from_regions(regions: Any) -> list[int] | None:
    bounds_list = []
    for region in regions:
        bounds = getattr(region, "bounds", None)
        if isinstance(region, dict):
            bounds = region.get("bounds")
        if bounds:
            bounds_list.append(bounds)
    if not bounds_list:
        return None
    left = min(item[0] for item in bounds_list)
    top = min(item[1] for item in bounds_list)
    right = max(item[2] for item in bounds_list)
    bottom = max(item[3] for item in bounds_list)
    return [left, top, right, bottom]


def _serialize_regions_for_openclaw(regions: Any) -> list[dict[str, Any]]:
    serialized = []
    for region in regions:
        if isinstance(region, dict):
            serialized.append(
                {
                    "region_id": region.get("region_id"),
                    "role_hint": region.get("role"),
                    "subtype_hint": region.get("subtype"),
                    "bounds": region.get("bounds"),
                }
            )
        else:
            serialized.append(
                {
                    "region_id": region.region_id,
                    "role_hint": region.role,
                    "subtype_hint": region.subtype.value if region.subtype else None,
                    "bounds": list(region.bounds) if region.bounds else None,
                }
            )
    return serialized


def _serialize_ocr_blocks(snapshot: InteractionCanvas | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(snapshot, dict):
        return list((snapshot.get("artifacts") or {}).get("ocr_blocks", []))
    return list(snapshot.artifacts.get("ocr_blocks", []))


def _rank_candidates_for_openclaw(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        score = _candidate_priority_score(candidate)
        enriched = dict(candidate)
        enriched["priority_score"] = round(score, 3)
        scored.append((score, index, enriched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def _select_candidates_for_openclaw(
    ranked_candidates: list[dict[str, Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0 or len(ranked_candidates) <= max_candidates:
        return ranked_candidates

    selected = list(ranked_candidates[:max_candidates])
    selected_ids = {str(candidate.get("candidate_id") or "") for candidate in selected}
    omniparser_candidates = [
        candidate
        for candidate in ranked_candidates
        if str(candidate.get("source") or "").lower() == "omniparser"
    ]
    target_omniparser_count = min(
        len(omniparser_candidates),
        max(1, min(8, max_candidates // 5 or 1)),
    )
    current_omniparser_count = sum(
        1
        for candidate in selected
        if str(candidate.get("source") or "").lower() == "omniparser"
    )
    if current_omniparser_count >= target_omniparser_count:
        return selected

    replacement_pool = [
        candidate
        for candidate in omniparser_candidates
        if str(candidate.get("candidate_id") or "") not in selected_ids
    ]
    if not replacement_pool:
        return selected

    selected.sort(
        key=lambda item: (
            str(item.get("source") or "").lower() == "omniparser",
            float(item.get("priority_score") or 0.0),
        )
    )
    for candidate in replacement_pool:
        if current_omniparser_count >= target_omniparser_count:
            break
        drop_index = next(
            (
                index
                for index, item in enumerate(selected)
                if str(item.get("source") or "").lower() != "omniparser"
            ),
            None,
        )
        if drop_index is None:
            break
        selected.pop(drop_index)
        selected.append(candidate)
        current_omniparser_count += 1

    selected.sort(
        key=lambda item: (
            -float(item.get("priority_score") or 0.0),
            str(item.get("candidate_id") or ""),
        )
    )
    return selected[:max_candidates]


def _candidate_priority_score(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("confidence") or 0.0) * 100.0
    source = str(candidate.get("source") or "").lower()
    kind = str(candidate.get("candidate_kind") or "")
    control_hint = str(candidate.get("control_hint") or "")
    region_hint = str(candidate.get("region_hint") or "")
    attributes = dict(candidate.get("attributes") or {})
    origin = str(attributes.get("candidate_origin") or "")
    text = str(candidate.get("text") or "").strip()

    source_bonus = {
        "omniparser": 18.0,
        "vision": 16.0,
        "merged": 14.0,
        "uia": 12.0,
        "native": 12.0,
        "ocr": 10.0,
    }
    kind_bonus = {
        "input_candidate": 26.0,
        "button_candidate": 24.0,
        "list_item_candidate": 20.0,
        "icon_candidate": 18.0,
        "panel_candidate": 16.0,
        "header_candidate": 12.0,
        "image_candidate": 10.0,
        "text_candidate": 8.0,
    }
    origin_bonus = {
        "vision_control_group": 18.0,
        "vision_layout_region": 14.0,
        "decision_bbox_fallback": -20.0,
    }
    hint_bonus = {
        "edit_like": 12.0,
        "button_like": 11.0,
        "list_item_like": 9.0,
        "header_like": 7.0,
        "group_like": 8.0,
        "pane_like": 6.0,
        "image_like": 4.0,
        "vision_like": 6.0,
    }
    region_bonus = {
        "top_left_filter": 8.0,
        "bottom_right_action": 7.0,
        "bottom_input": 7.0,
        "left_panel_list": 6.0,
        "top_header_bar": 4.0,
        "left_icon_rail": 3.0,
        "main_content": 2.0,
    }

    score += source_bonus.get(source, 4.0)
    score += kind_bonus.get(kind, 4.0)
    score += origin_bonus.get(origin, 0.0)
    score += hint_bonus.get(control_hint, 0.0)
    score += region_bonus.get(region_hint, 0.0)
    if region_hint:
        score += 2.0
    if text:
        score += min(len(text), 24) * 0.4
    if source == "omniparser" and kind in {"icon_candidate", "panel_candidate", "list_item_candidate"}:
        score += 6.0
    return score


def _summarize_candidates_for_openclaw(
    candidates: list[dict[str, Any]],
    total_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_source = Counter(str(candidate.get("source") or "unknown") for candidate in candidates)
    by_kind = Counter(str(candidate.get("candidate_kind") or "unknown") for candidate in candidates)
    by_origin = Counter(
        str(dict(candidate.get("attributes") or {}).get("candidate_origin") or "base")
        for candidate in candidates
    )
    total = list(total_candidates) if total_candidates is not None else list(candidates)
    total_by_source = Counter(str(candidate.get("source") or "unknown") for candidate in total)
    return {
        "count": len(candidates),
        "total_count": len(total),
        "by_source": dict(sorted(by_source.items())),
        "by_source_total": dict(sorted(total_by_source.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "omniparser_retained_count": by_source.get("omniparser", 0),
        "omniparser_total_count": total_by_source.get("omniparser", 0),
        "top_candidate_ids": [
            str(candidate.get("candidate_id") or "")
            for candidate in candidates[:12]
        ],
    }


def _serialize_provider_health(
    snapshot: InteractionCanvas | dict[str, Any],
    all_candidates: list[dict[str, Any]],
    delivered_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(snapshot, dict):
        provider_trace = dict(snapshot.get("provider_trace") or {})
        provider_details = dict(provider_trace.get("provider_details") or {})
        artifacts = dict(snapshot.get("artifacts") or {})
    else:
        provider_trace = {
            "uia_used": snapshot.provider_trace.uia_used,
            "ocr_used": snapshot.provider_trace.ocr_used,
            "vision_used": snapshot.provider_trace.vision_used,
            "dom_used": snapshot.provider_trace.dom_used,
        }
        provider_details = dict(snapshot.provider_trace.provider_details or {})
        artifacts = dict(snapshot.artifacts or {})

    vision = _merge_provider_detail(
        provider_details.get("vision"),
        provider_details.get("vision_provider"),
    )
    ocr = _merge_provider_detail(
        provider_details.get("ocr"),
        provider_details.get("ocr_provider"),
    )
    candidate_stats = dict(artifacts.get("boundary_candidate_stats") or {})
    if not candidate_stats:
        candidate_stats = _summarize_candidates_for_openclaw(
            delivered_candidates,
            total_candidates=all_candidates,
        )
    all_sources = Counter(str(candidate.get("source") or "unknown") for candidate in all_candidates)
    omniparser_total_count = all_sources.get("omniparser", 0)
    omniparser_retained_count = sum(
        1
        for candidate in delivered_candidates
        if str(candidate.get("source") or "").lower() == "omniparser"
    )
    vision_sidecar_count = sum(
        count
        for source, count in all_sources.items()
        if source not in {"omniparser", "vision", "ocr", "uia", "native", "merged", "unknown"}
    )
    vision_used = bool(
        provider_trace.get("vision_used")
        or omniparser_total_count
        or vision.get("candidate_count")
        or vision.get("layout_region_count")
        or vision.get("control_group_count")
    )
    ocr_used = bool(
        provider_trace.get("ocr_used")
        or ocr.get("block_count")
    )
    vision_effective = bool(
        omniparser_total_count
        or vision.get("layout_region_count")
        or vision.get("control_group_count")
    )
    return {
        "uia_used": bool(provider_trace.get("uia_used")),
        "ocr_used": ocr_used,
        "vision_used": vision_used,
        "vision_effective": vision_effective,
        "dom_used": bool(provider_trace.get("dom_used")),
        "vision_provider": {
            "provider": vision.get("provider") or "omniparser",
            "success": vision.get("success"),
            "error": vision.get("error"),
            "candidate_count": vision.get("candidate_count", 0),
            "layout_region_count": vision.get("layout_region_count", 0),
            "control_group_count": vision.get("control_group_count", 0),
        },
        "ocr_provider": {
            "provider": ocr.get("provider"),
            "success": ocr.get("success"),
            "error": ocr.get("error"),
            "block_count": ocr.get("block_count", 0),
        },
        "omniparser_available": bool(vision.get("success")),
        "omniparser_total_count": omniparser_total_count,
        "omniparser_retained_count": omniparser_retained_count,
        "vision_sidecar_count": vision_sidecar_count,
        "boundary_candidates": candidate_stats,
    }


def _merge_provider_detail(
    primary: Any,
    secondary: Any,
) -> dict[str, Any]:
    merged = dict(primary or {})
    merged.update(dict(secondary or {}))
    return merged


def _coerce_bbox_list(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    except (TypeError, ValueError):
        return None
