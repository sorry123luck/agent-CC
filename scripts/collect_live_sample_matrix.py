"""Collect a live recognition-chain sample matrix from currently open windows."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SYSTEM_PROCESS_DENYLIST = {
    "applicationframehost.exe",
    "explorer.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "systemsettings.exe",
    "taskmgr.exe",
    "textinputhost.exe",
}

SAMPLE_MATRIX_SCHEMA_VERSION = "2026-05-25.chat-variant.v3"


def select_sample_windows(
    windows: list[dict[str, Any]],
    *,
    include_processes: list[str],
    max_windows: int,
) -> list[dict[str, Any]]:
    """Return visible sample windows suitable for recognition-chain testing."""
    include = {item.lower() for item in include_processes if item}
    selected: list[dict[str, Any]] = []
    for window in windows:
        process = str(window.get("process_name") or "").lower()
        title = str(window.get("title") or "").strip()
        if not title or window.get("is_minimized"):
            continue
        if include and process not in include:
            continue
        if not include and process in SYSTEM_PROCESS_DENYLIST:
            continue
        selected.append(window)
        if len(selected) >= max_windows:
            break
    return selected


def build_matrix_row(
    *,
    window: dict[str, Any],
    observe: dict[str, Any] | None,
    semantic: dict[str, Any] | None,
    detail: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    """Build one compact matrix row from API responses."""
    observe = observe or {}
    semantic = semantic or {}
    detail = detail or {}
    quality = detail.get("perception_quality") or observe.get("perception_quality") or {}
    quality_warnings = list(quality.get("warnings") or [])
    if _is_root_only_detail(detail):
        quality_warnings.append("root_only_detail")
    roi_plan = detail.get("roi_selection_plan") or observe.get("roi_selection_plan") or {}
    rois = list(roi_plan.get("rois") or [])
    stages = [
        f"{stage.get('name')}:{stage.get('status')}"
        for stage in list(semantic.get("stages") or [])
    ]
    supplements = list(detail.get("roi_vlm_semantic_supplements") or [])
    timeouts = list(detail.get("roi_vlm_timeouts") or [])
    late_failures = list(detail.get("roi_vlm_late_failures") or [])
    rejected = list(detail.get("roi_vlm_rejected_responses") or [])
    late_pending_count = _late_pending_count(
        timeouts=timeouts,
        supplements=supplements,
        late_failures=late_failures,
        rejected=rejected,
    )
    roi_jobs = _semantic_roi_jobs(semantic)
    roi_payload = _roi_payload_summary(roi_jobs)
    composer_payload = _composer_payload_summary(roi_jobs, detail=detail)
    input_summary = _input_candidate_summary(detail)
    search_summary = _search_candidate_summary(window=window, detail=detail)
    screenshot_summary = _screenshot_provider_summary(detail)
    mode = str(roi_plan.get("mode") or (detail.get("visual_pattern") or {}).get("mode") or "")
    chat_variant = _chat_variant_detection(window=window, detail=detail, mode=mode)
    rejected_errors = sorted({
        str(error)
        for item in rejected
        for error in list(item.get("errors") or [])
    })
    sample_name = _sample_name(window)
    return {
        "sample": sample_name,
        "hwnd": int(window.get("hwnd") or 0),
        "title": str(window.get("title") or ""),
        "process_name": str(window.get("process_name") or ""),
        "canvas_id": str(observe.get("canvas_id") or detail.get("canvas_id") or ""),
        "processing_state": str(detail.get("processing_state") or observe.get("processing_state") or ""),
        "observe_elapsed_ms": int(observe.get("elapsed_ms") or 0),
        "semantic_elapsed_ms": int(semantic.get("elapsed_ms") or 0),
        "element_count": int(len(detail.get("elements") or []) or observe.get("element_count") or 0),
        "region_count": int(len(detail.get("regions") or []) or observe.get("region_count") or 0),
        "geometric_region_count": int(
            len(detail.get("geometric_regions") or [])
            or observe.get("geometric_region_count")
            or 0
        ),
        "mode": mode,
        "chat_variant": chat_variant["variant"],
        "chat_variant_source": chat_variant["source"],
        "chat_variant_evidence": chat_variant["evidence"],
        "quality_warnings": sorted(set(str(warning) for warning in quality_warnings if str(warning).strip())),
        "screenshot_success": screenshot_summary["success"],
        "screenshot_error": screenshot_summary["error"],
        "unknown_role_ratio": float(quality.get("unknown_role_ratio") or 0),
        "roi_count": len(rois),
        "roi_purposes": [str(roi.get("purpose") or "") for roi in rois],
        "roi_vlm_job_count": len(roi_jobs),
        "roi_vlm_candidate_count": roi_payload["candidate_count"],
        "roi_vlm_local_text_candidate_count": roi_payload["local_text_candidate_count"],
        "roi_vlm_max_candidates_per_job": roi_payload["max_candidates_per_job"],
        "roi_vlm_avg_candidates_per_job": roi_payload["avg_candidates_per_job"],
        "composer_vlm_candidate_count": composer_payload["candidate_count"],
        "composer_vlm_hint_labels": composer_payload["hint_labels"],
        "composer_vlm_send_hint_count": composer_payload["send_hint_count"],
        "composer_send_target_candidate_id": composer_payload["send_target_candidate_id"],
        "composer_send_target_bounds": composer_payload["send_target_bounds"],
        "composer_send_target_width": composer_payload["send_target_width"],
        "composer_send_target_height": composer_payload["send_target_height"],
        "composer_input_candidate_count": input_summary["candidate_count"],
        "composer_input_safe_count": input_summary["safe_count"],
        "composer_input_review_count": input_summary["review_count"],
        "composer_input_review_bounds": input_summary["review_bounds"],
        "composer_input_review_click_points": input_summary["review_click_points"],
        "composer_input_primary_click_point": input_summary["primary_click_point"],
        "composer_input_review_in_composer_count": input_summary["review_in_composer_count"],
        "composer_input_review_min_width": input_summary["review_min_width"],
        "composer_input_review_min_height": input_summary["review_min_height"],
        "search_candidate_count": search_summary["candidate_count"],
        "search_primary_click_point": search_summary["primary_click_point"],
        "search_bounds": search_summary["bounds"],
        "search_source": search_summary["source"],
        "search_evidence": search_summary["evidence"],
        "semantic_status": str(semantic.get("status") or "not_run"),
        "semantic_next_action": str(semantic.get("next_action") or ""),
        "semantic_stage_statuses": stages,
        "roi_vlm_result_status_counts": dict((semantic.get("roi_vlm") or {}).get("result_status_counts") or {}),
        "roi_vlm_supplement_count": len(supplements),
        "late_supplement_count": len([item for item in supplements if str(item.get("status") or "").startswith("timeout_late")]),
        "roi_vlm_timeout_count": len(timeouts),
        "roi_vlm_late_failure_count": len(late_failures),
        "roi_vlm_late_pending_count": late_pending_count,
        "roi_vlm_rejected_count": len(rejected),
        "roi_vlm_rejected_errors": rejected_errors,
        "error": error,
}


def _screenshot_provider_summary(detail: dict[str, Any]) -> dict[str, Any]:
    trace = detail.get("provider_trace") if isinstance(detail, dict) else {}
    provider_details = (trace or {}).get("provider_details") if isinstance(trace, dict) else {}
    screenshot = (provider_details or {}).get("screenshot_provider") if isinstance(provider_details, dict) else {}
    if not isinstance(screenshot, dict):
        screenshot = {}
    return {
        "success": bool(screenshot.get("success")),
        "error": str(screenshot.get("error") or ""),
    }


def _is_root_only_detail(detail: dict[str, Any]) -> bool:
    elements = [element for element in list(detail.get("elements") or []) if isinstance(element, dict)]
    if len(elements) != 1:
        return False
    element = elements[0]
    role = str(element.get("semantic_role") or element.get("role_label") or "").lower()
    control_type = str(element.get("control_type") or "").lower()
    return role in {"layout", "unknown", ""} and control_type in {"panecontrol", "windowcontrol", "groupcontrol", ""}


def _excluded_sample_reason(*, row: dict[str, Any], detail: dict[str, Any] | None) -> str:
    """Return a reason when a collected window is not a useful recognition sample."""
    detail = detail or {}
    if row.get("screenshot_success") is True:
        return ""
    bounds = [
        _rect(element.get("bounds"))
        for element in list(detail.get("elements") or [])
        if isinstance(element, dict)
    ]
    bounds = [item for item in bounds if item]
    width = int(detail.get("screenshot_width") or detail.get("window_width") or 0)
    height = int(detail.get("screenshot_height") or detail.get("window_height") or 0)
    max_right = max((item[2] for item in bounds), default=0)
    max_bottom = max((item[3] for item in bounds), default=0)
    inferred_width = width or max_right
    inferred_height = height or max_bottom
    if inferred_width and inferred_height and inferred_width <= 128 and inferred_height <= 128:
        return "invalid_tiny_window_no_screenshot"
    return ""


def _chat_variant_summary(*, window: dict[str, Any], detail: dict[str, Any], mode: str) -> str:
    return _chat_variant_detection(window=window, detail=detail, mode=mode)["variant"]


def _chat_variant_detection(*, window: dict[str, Any], detail: dict[str, Any], mode: str) -> dict[str, str]:
    process = str(window.get("process_name") or detail.get("process_name") or "").lower()
    if process != "qq.exe" or mode != "chat_workspace":
        return {"variant": "", "source": "", "evidence": ""}
    evidence = [
        str(item)
        for item in list(((detail.get("visual_pattern") or {}).get("evidence") or []))
        if str(item).startswith("qq_variant:")
    ]
    for item in evidence:
        variant = item.split(":", 1)[1]
        if variant in {"private_chat", "group_chat"}:
            return {"variant": variant, "source": "visual_pattern", "evidence": item}
    local = _infer_qq_chat_variant_from_detail(detail)
    if local["variant"] == "group_chat":
        return local
    vlm = _infer_qq_chat_variant_from_roi_supplements(detail)
    if vlm["variant"] in {"private_chat", "group_chat"}:
        return vlm
    if evidence:
        return {"variant": evidence[0].split(":", 1)[1], "source": "visual_pattern", "evidence": evidence[0]}
    return local


def _infer_qq_chat_variant_from_roi_supplements(detail: dict[str, Any]) -> dict[str, str]:
    private_terms = (
        "private_chat_area",
        "private chat",
        "private messages",
        "one-to-one",
        "direct message",
        "dm",
    )
    group_terms = ("group_chat_area", "group chat", "group conversation", "member list", "group member", "群聊", "群成员")
    for supplement in list(detail.get("roi_vlm_semantic_supplements") or []):
        if not isinstance(supplement, dict):
            continue
        haystack_parts: list[str] = []
        region = supplement.get("region_semantics") or {}
        if isinstance(region, dict):
            haystack_parts.extend(str(region.get(key) or "") for key in ("role", "summary", "label"))
        haystack_parts.extend(str(item) for item in list(supplement.get("review_only_hints") or []))
        haystack = " ".join(haystack_parts).lower()
        if any(term in haystack for term in group_terms):
            return {"variant": "group_chat", "source": "roi_vlm", "evidence": haystack[:160]}
        if any(term in haystack for term in private_terms):
            return {"variant": "private_chat", "source": "roi_vlm", "evidence": haystack[:160]}
    return {"variant": "unknown", "source": "", "evidence": ""}


def _infer_qq_chat_variant_from_detail(detail: dict[str, Any]) -> dict[str, str]:
    width = int(detail.get("screenshot_width") or detail.get("window_width") or 0)
    if width <= 0:
        return {"variant": "unknown", "source": "", "evidence": "missing_width"}
    main_left = int(width * 0.30)
    ignored_toolbar_terms = {"发起群聊", "更多", "语音通话", "视频通话", "屏幕共享", "远程协助"}
    group_terms = ("群公告", "群文件", "群相册", "群成员", "群应用", "群课堂", "群聊", "成员", "人在线")
    title_seen = False
    right_member_rows = 0
    for element in list(detail.get("elements") or []):
        if not isinstance(element, dict):
            continue
        bounds = _rect(element.get("bounds"))
        if not bounds:
            continue
        left, top, right, _bottom = bounds
        if left >= int(width * 0.78) and 70 <= top <= 280 and (right - left) >= 100:
            right_member_rows += 1
        text = str(element.get("text") or element.get("name") or element.get("role_label") or "").strip()
        if not text or left < main_left:
            continue
        if text in ignored_toolbar_terms:
            continue
        if any(term in text for term in group_terms):
            return {"variant": "group_chat", "source": "local_text", "evidence": text[:160]}
        if top <= 80 and (right - left) >= 12:
            title_seen = True
    if right_member_rows >= 2:
        return {"variant": "group_chat", "source": "local_geometry", "evidence": "right_member_list_like"}
    if title_seen:
        return {"variant": "private_chat", "source": "local_title", "evidence": "top_title_seen"}
    return {"variant": "unknown", "source": "", "evidence": "no_variant_evidence"}


def _semantic_roi_jobs(semantic: dict[str, Any]) -> list[dict[str, Any]]:
    roi_vlm = semantic.get("roi_vlm") if isinstance(semantic, dict) else {}
    jobs = (roi_vlm or {}).get("jobs") if isinstance(roi_vlm, dict) else []
    return [job for job in list(jobs or []) if isinstance(job, dict)]


def _late_pending_count(
    *,
    timeouts: list[Any],
    supplements: list[Any],
    late_failures: list[Any],
    rejected: list[Any],
) -> int:
    timeout_ids = {
        str(item.get("roi_id") or "")
        for item in timeouts
        if isinstance(item, dict) and str(item.get("roi_id") or "")
    }
    if not timeout_ids:
        return 0
    completed_ids = {
        str(item.get("roi_id") or "")
        for item in supplements
        if isinstance(item, dict)
        and str(item.get("status") or "").startswith("timeout_late")
        and str(item.get("roi_id") or "")
    }
    completed_ids.update(
        str(item.get("roi_id") or "")
        for item in late_failures
        if isinstance(item, dict)
        and str(item.get("status") or "").startswith("timeout_late")
        and str(item.get("roi_id") or "")
    )
    completed_ids.update(
        str(item.get("roi_id") or "")
        for item in rejected
        if isinstance(item, dict)
        and str(item.get("status") or "").startswith("timeout_late")
        and str(item.get("roi_id") or "")
    )
    return len(timeout_ids - completed_ids)


def _roi_payload_summary(jobs: list[dict[str, Any]]) -> dict[str, int | float]:
    candidate_counts = [len(list(job.get("candidate_ids") or [])) for job in jobs]
    local_text_counts = [len(list(job.get("local_text_candidate_ids") or [])) for job in jobs]
    total_candidates = sum(candidate_counts)
    return {
        "candidate_count": total_candidates,
        "local_text_candidate_count": sum(local_text_counts),
        "max_candidates_per_job": max(candidate_counts, default=0),
        "avg_candidates_per_job": round(total_candidates / len(jobs), 2) if jobs else 0.0,
    }


def _composer_payload_summary(jobs: list[dict[str, Any]], *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    composer_jobs = [
        job for job in jobs
        if str(job.get("purpose") or "") == "composer"
    ]
    refs = [
        ref
        for job in composer_jobs
        for ref in list(job.get("candidate_refs") or [])
        if isinstance(ref, dict)
    ]
    hint_labels = sorted({
        str(ref.get("local_hint") or "")
        for ref in refs
        if str(ref.get("local_hint") or "")
    })
    send_ref = next((ref for ref in refs if str(ref.get("local_hint") or "") == "send_button"), {})
    send_ref = _prefer_detail_send_button(send_ref, detail or {})
    send_bounds = _rect(send_ref.get("bounds")) if isinstance(send_ref, dict) else []
    return {
        "candidate_count": sum(len(list(job.get("candidate_ids") or [])) for job in composer_jobs),
        "hint_labels": hint_labels,
        "send_hint_count": len([hint for hint in hint_labels if hint == "send_button"]),
        "send_target_candidate_id": str(send_ref.get("candidate_id") or "") if isinstance(send_ref, dict) else "",
        "send_target_bounds": send_bounds,
        "send_target_width": (send_bounds[2] - send_bounds[0]) if send_bounds else 0,
        "send_target_height": (send_bounds[3] - send_bounds[1]) if send_bounds else 0,
    }


def _prefer_detail_send_button(send_ref: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    detail_ref = _detail_send_button_ref(detail)
    if not detail_ref:
        return send_ref
    send_bounds = _rect(send_ref.get("bounds")) if isinstance(send_ref, dict) else []
    detail_bounds = _rect(detail_ref.get("bounds"))
    if not send_bounds:
        return detail_ref
    send_width = send_bounds[2] - send_bounds[0]
    send_height = send_bounds[3] - send_bounds[1]
    detail_width = detail_bounds[2] - detail_bounds[0]
    detail_height = detail_bounds[3] - detail_bounds[1]
    if send_width < 32 or send_height < 20:
        return detail_ref
    if detail_width * detail_height > send_width * send_height * 1.5:
        return detail_ref
    return send_ref


def _detail_send_button_ref(detail: dict[str, Any]) -> dict[str, Any]:
    composer_rois = [
        _rect(roi.get("bounds"))
        for roi in list(((detail.get("roi_selection_plan") or {}).get("rois") or []))
        if isinstance(roi, dict) and str(roi.get("purpose") or "") == "composer"
    ]
    composer_rois = [bounds for bounds in composer_rois if bounds]
    candidates: list[dict[str, Any]] = []
    for element in list(detail.get("elements") or []):
        if not isinstance(element, dict):
            continue
        bounds = _rect(element.get("bounds"))
        if not bounds or (composer_rois and not any(_rect_inside(bounds, roi) for roi in composer_rois)):
            continue
        role = str(element.get("semantic_role") or element.get("role_label") or "").lower()
        text = str(element.get("text") or element.get("name") or "").strip().lower()
        if role != "send_button" and text not in {"发送", "send"}:
            continue
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if width < 32 or height < 20:
            continue
        candidates.append({
            "candidate_id": str(element.get("element_id") or ""),
            "bounds": bounds,
            "local_hint": "send_button",
        })
    candidates.sort(
        key=lambda item: (item["bounds"][2] - item["bounds"][0]) * (item["bounds"][3] - item["bounds"][1]),
        reverse=True,
    )
    return candidates[0] if candidates else {}


def _input_candidate_summary(detail: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        item for item in list(detail.get("elements") or [])
        if isinstance(item, dict) and _looks_like_composer_input_candidate(item)
    ]
    safe = [
        item for item in candidates
        if item.get("safe_to_type") is True
        or (isinstance(item.get("attributes"), dict) and item["attributes"].get("safe_to_type") is True)
    ]
    review = [
        item for item in candidates
        if str(item.get("actionability") or item.get("refine_status") or "").lower() == "review"
        or (isinstance(item.get("attributes"), dict) and item["attributes"].get("needs_manual_label") is True)
    ]
    review_bounds = [_rect(item.get("bounds")) for item in review]
    review_bounds = [bounds for bounds in review_bounds if bounds]
    review_click_points = [
        _candidate_click_point(item, bounds)
        for item, bounds in zip(review, review_bounds)
    ]
    review_click_points = [point for point in review_click_points if point]
    composer_rois = [
        _input_container_rect(roi)
        for roi in list(((detail.get("roi_selection_plan") or {}).get("rois") or []))
    ]
    composer_rois = [bounds for bounds in composer_rois if bounds]
    widths = [bounds[2] - bounds[0] for bounds in review_bounds]
    heights = [bounds[3] - bounds[1] for bounds in review_bounds]
    return {
        "candidate_count": len(candidates),
        "safe_count": len(safe),
        "review_count": len(review),
        "review_bounds": review_bounds,
        "review_click_points": review_click_points,
        "primary_click_point": review_click_points[0] if review_click_points else [],
        "review_in_composer_count": sum(
            1
            for bounds in review_bounds
            if any(_rect_major_overlap(bounds, composer) for composer in composer_rois)
        ),
        "review_min_width": min(widths, default=0),
        "review_min_height": min(heights, default=0),
    }


def _search_candidate_summary(*, window: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    process = str(window.get("process_name") or detail.get("process_name") or "").lower()
    ocr_match = _search_from_ocr(detail)
    if ocr_match["bounds"]:
        return ocr_match
    if process in {"feishu.exe", "lark.exe"}:
        app_default = _search_from_app_default_geometry(process=process, detail=detail)
        if app_default["bounds"]:
            return app_default
    if process in {"weixin.exe", "wechat.exe", "qq.exe"}:
        geometric = _search_from_chat_sidebar_geometry(process=process, detail=detail)
        if geometric["bounds"]:
            return geometric
    return {"candidate_count": 0, "primary_click_point": [], "bounds": [], "source": "", "evidence": ""}


def _search_from_ocr(detail: dict[str, Any]) -> dict[str, Any]:
    for block in list(detail.get("ocr_blocks") or []):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if "搜索" not in text and "search" not in text.lower():
            continue
        bounds = _rect(block.get("bbox") or block.get("bounds"))
        if not bounds:
            continue
        expanded = [
            max(0, bounds[0] - 22),
            max(0, bounds[1] - 10),
            bounds[2] + 56,
            bounds[3] + 10,
        ]
        return {
            "candidate_count": 1,
            "primary_click_point": _center(expanded),
            "bounds": expanded,
            "source": "ocr",
            "evidence": text[:80],
        }
    return {"candidate_count": 0, "primary_click_point": [], "bounds": [], "source": "", "evidence": ""}


def _search_from_chat_sidebar_geometry(*, process: str, detail: dict[str, Any]) -> dict[str, Any]:
    elements = [item for item in list(detail.get("elements") or []) if isinstance(item, dict)]
    bounds_list = [_rect(item.get("bounds")) for item in elements]
    bounds_list = [bounds for bounds in bounds_list if bounds]
    search_boxes = [
        bounds for bounds in bounds_list
        if 20 <= bounds[1] <= 60
        and 60 <= bounds[0] <= 120
        and 140 <= (bounds[2] - bounds[0]) <= 230
        and 24 <= (bounds[3] - bounds[1]) <= 42
    ]
    if search_boxes:
        box = min(search_boxes, key=lambda bounds: (bounds[1], bounds[0]))
        return {
            "candidate_count": 1,
            "primary_click_point": _center(box),
            "bounds": box,
            "source": "sidebar_geometry",
            "evidence": f"{process}:top_left_search_box",
        }
    search_icons = [
        bounds for bounds in bounds_list
        if 20 <= bounds[1] <= 55
        and 55 <= bounds[0] <= 115
        and 18 <= (bounds[2] - bounds[0]) <= 45
        and 18 <= (bounds[3] - bounds[1]) <= 45
    ]
    plus_buttons = [
        bounds for bounds in bounds_list
        if 20 <= bounds[1] <= 60
        and 240 <= bounds[0] <= 330
        and 18 <= (bounds[2] - bounds[0]) <= 50
    ]
    if not search_icons:
        return {"candidate_count": 0, "primary_click_point": [], "bounds": [], "source": "", "evidence": ""}
    icon = min(search_icons, key=lambda bounds: (bounds[1], bounds[0]))
    right = min((bounds[0] - 8 for bounds in plus_buttons if bounds[0] > icon[2]), default=260)
    left = max(0, icon[0] - 10)
    top = max(0, min(icon[1] - 8, 34 if process in {"weixin.exe", "wechat.exe"} else 22))
    bottom = icon[3] + 8
    search_bounds = [left, top, max(icon[2] + 80, right), bottom]
    return {
        "candidate_count": 1,
        "primary_click_point": _center(search_bounds),
        "bounds": search_bounds,
        "source": "sidebar_geometry",
        "evidence": f"{process}:top_left_search_icon",
    }


def _search_from_app_default_geometry(*, process: str, detail: dict[str, Any]) -> dict[str, Any]:
    width = int(detail.get("screenshot_width") or 0)
    height = int(detail.get("screenshot_height") or 0)
    if process in {"feishu.exe", "lark.exe"} and width >= 500 and height >= 300:
        bounds = [16, 58, 169, 88]
        return {
            "candidate_count": 1,
            "primary_click_point": _center(bounds),
            "bounds": bounds,
            "source": "app_default_geometry",
            "evidence": f"{process}:left_sidebar_search_slot",
        }
    return {"candidate_count": 0, "primary_click_point": [], "bounds": [], "source": "", "evidence": ""}


def _candidate_click_point(item: dict[str, Any], bounds: list[int]) -> list[int]:
    explicit = item.get("click_point")
    if isinstance(explicit, (list, tuple)) and len(explicit) == 2:
        try:
            x, y = [int(value) for value in explicit]
        except (TypeError, ValueError):
            x, y = 0, 0
        else:
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                return [x, y]
    return [int((bounds[0] + bounds[2]) / 2), int((bounds[1] + bounds[3]) / 2)]


def _center(bounds: list[int]) -> list[int]:
    if not bounds:
        return []
    return [int((bounds[0] + bounds[2]) / 2), int((bounds[1] + bounds[3]) / 2)]


def _rect(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if right <= left or bottom <= top:
        return []
    return [left, top, right, bottom]


def _input_container_rect(roi: dict[str, Any]) -> list[int]:
    purpose = str(roi.get("purpose") or "")
    bounds = _rect(roi.get("bounds"))
    if not bounds:
        return []
    if purpose == "composer":
        return bounds
    if purpose == "detail_pane":
        left, top, right, bottom = bounds
        height = bottom - top
        input_top = max(top, bottom - max(96, int(height * 0.22)))
        return [left, input_top, right, bottom]
    return []


def _rect_inside(inner: list[int], outer: list[int], *, tolerance: int = 8) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _rect_major_overlap(inner: list[int], outer: list[int], *, min_ratio: float = 0.5) -> bool:
    if _rect_inside(inner, outer):
        return True
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if inner_area <= 0:
        return False
    overlap_left = max(inner[0], outer[0])
    overlap_top = max(inner[1], outer[1])
    overlap_right = min(inner[2], outer[2])
    overlap_bottom = min(inner[3], outer[3])
    if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
        return False
    overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    return overlap_area / inner_area >= min_ratio


def _looks_like_composer_input_candidate(element: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(element.get(key) or "")
        for key in (
            "element_id",
            "role_label",
            "semantic_role",
            "control_type",
            "text",
            "name",
        )
    ).lower()
    attrs = element.get("attributes")
    if isinstance(attrs, dict):
        haystack = f"{haystack} " + " ".join(
            str(attrs.get(key) or "")
            for key in ("role_label", "semantic_role", "synthetic_kind", "candidate_kind")
        ).lower()
    return any(
        token in haystack
        for token in (
            "message_input",
            "composer_input",
            "chat_input",
            "input_area",
            "synthetic_collaboration_composer_input",
        )
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate sample matrix metrics."""
    semantic_counts = Counter(str(row.get("semantic_status") or "unknown") for row in rows)
    mode_counts = Counter(str(row.get("mode") or "unknown") for row in rows)
    chat_variant_counts = Counter(
        str(row.get("chat_variant") or "none")
        for row in rows
        if str(row.get("mode") or "") == "chat_workspace"
    )
    error_count = len([row for row in rows if row.get("error")])
    return {
        "sample_matrix_schema_version": SAMPLE_MATRIX_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(rows),
        "error_count": error_count,
        "semantic_status_counts": dict(semantic_counts),
        "mode_counts": dict(mode_counts),
        "chat_variant_counts": dict(chat_variant_counts),
        "excluded_count": 0,
        "excluded_rows": [],
        "rows": rows,
    }


def write_matrix_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    """Persist matrix summary JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_matrix_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Live Sample Matrix",
        "",
        f"- Schema: {summary.get('sample_matrix_schema_version', '')}",
        f"- Generated: {summary.get('generated_at', '')}",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Errors: {summary.get('error_count', 0)}",
        f"- Excluded: {summary.get('excluded_count', 0)}",
        f"- Semantic status counts: {_format_counts(summary.get('semantic_status_counts') or {})}",
        f"- Mode counts: {_format_counts(summary.get('mode_counts') or {})}",
        f"- Chat variant counts: {_format_counts(summary.get('chat_variant_counts') or {})}",
        "",
        "| sample | process | mode | variant | variant_source | state | observe_ms | semantic | roi_count | vlm_candidates | local_text | input_safe/review | input_click | search_click | warnings | late | error |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in list(summary.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {mode} | {variant} | {variant_source} | {state} | {observe} | {semantic} | {roi_count} | {vlm_candidates} | {local_text} | {input_state} | {input_click} | {search_click} | {warnings} | {late} | {error} |".format(
                sample=row.get("sample", ""),
                process=row.get("process_name", ""),
                mode=row.get("mode", ""),
                variant=row.get("chat_variant", ""),
                variant_source=row.get("chat_variant_source", ""),
                state=row.get("processing_state", ""),
                observe=row.get("observe_elapsed_ms", 0),
                semantic=row.get("semantic_status", ""),
                roi_count=row.get("roi_count", 0),
                vlm_candidates=row.get("roi_vlm_candidate_count", 0),
                local_text=row.get("roi_vlm_local_text_candidate_count", 0),
                input_state=f"{row.get('composer_input_safe_count', 0)}/{row.get('composer_input_review_count', 0)}",
                input_click=str(row.get("composer_input_primary_click_point") or "").replace("|", "/"),
                search_click=str(row.get("search_primary_click_point") or "").replace("|", "/"),
                warnings=",".join(row.get("quality_warnings") or []),
                late=row.get("late_supplement_count", 0),
                error=(
                    str(row.get("error") or "").replace("|", "/")
                    or ",".join(row.get("roi_vlm_rejected_errors") or [])
                ),
            )
        )
    excluded = list(summary.get("excluded_rows") or [])
    if excluded:
        lines.extend([
            "",
            "## Excluded Samples",
            "",
            "| sample | process | reason |",
            "| --- | --- | --- |",
        ])
        for row in excluded:
            lines.append(
                "| {sample} | {process} | {reason} |".format(
                    sample=str(row.get("sample") or "").replace("|", "/"),
                    process=str(row.get("process_name") or "").replace("|", "/"),
                    reason=str(row.get("excluded_reason") or "").replace("|", "/"),
                )
            )
    (output_dir / "sample_matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, *, clean: bool) -> None:
    """Create output dir and optionally remove stale JSON/Markdown matrix outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean:
        return
    for pattern in ("*.json", "*.md"):
        for path in output_dir.rglob(pattern):
            if path.is_file():
                path.unlink()


def write_detail_output(row: dict[str, Any], detail: dict[str, Any], output_dir: Path) -> None:
    """Persist one detail response next to the matrix summary."""
    write_response_output(row, detail, output_dir, suffix="detail")


def write_response_output(
    row: dict[str, Any],
    response: dict[str, Any],
    output_dir: Path,
    *,
    suffix: str,
) -> None:
    """Persist one raw API response for later diagnosis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = str(row.get("sample") or "sample")
    (output_dir / f"{sample}.{suffix}.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_matrix(
    *,
    base_url: str,
    output_dir: Path,
    include_processes: list[str],
    max_windows: int,
    run_vlm: bool,
    max_rois_per_window: int,
    wait_late_seconds: float,
    wait_enhance_seconds: float,
    deadline_ms: int,
    clean_output: bool = False,
    async_enhance: bool = False,
    roi_purposes: list[str] | None = None,
) -> dict[str, Any]:
    prepare_output_dir(output_dir, clean=clean_output)
    session = requests.Session()
    windows = session.get(f"{base_url}/api/v1/windows", timeout=20).json()
    selected = select_sample_windows(
        windows,
        include_processes=include_processes,
        max_windows=max_windows,
    )
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for window in selected:
        row, observe, semantic, detail = _collect_one_window(
            session=session,
            base_url=base_url,
            output_dir=output_dir,
            window=window,
            run_vlm=run_vlm,
            max_rois_per_window=max_rois_per_window,
            wait_late_seconds=wait_late_seconds,
            wait_enhance_seconds=wait_enhance_seconds,
            deadline_ms=deadline_ms,
            async_enhance=async_enhance,
            roi_purposes=roi_purposes or [],
        )
        excluded_reason = _excluded_sample_reason(row=row, detail=detail)
        if excluded_reason:
            row["excluded_reason"] = excluded_reason
            excluded_rows.append(row)
        else:
            rows.append(row)
        if observe:
            write_response_output(row, observe, output_dir, suffix="observe")
        if semantic:
            write_response_output(row, semantic, output_dir, suffix="semantic")
        if detail:
            write_detail_output(row, detail, output_dir)
    summary = build_summary(rows)
    summary["excluded_count"] = len(excluded_rows)
    summary["excluded_rows"] = excluded_rows
    write_matrix_outputs(summary, output_dir)
    return summary


def _collect_one_window(
    *,
    session: requests.Session,
    base_url: str,
    output_dir: Path,
    window: dict[str, Any],
    run_vlm: bool,
    max_rois_per_window: int,
    wait_late_seconds: float,
    wait_enhance_seconds: float,
    deadline_ms: int,
    async_enhance: bool,
    roi_purposes: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    observe: dict[str, Any] | None = None
    semantic: dict[str, Any] | None = None
    detail: dict[str, Any] | None = None
    error = ""
    try:
        observe_resp = session.post(
            f"{base_url}/api/v1/observe",
            json={
                "hwnd": int(window["hwnd"]),
                "allow_vlm": False,
                "force_vlm": False,
                "async_enhance": bool(async_enhance),
            },
            timeout=90,
        )
        observe_resp.raise_for_status()
        observe = observe_resp.json()
        detail = session.get(f"{base_url}/api/v1/canvases/{observe['canvas_id']}", timeout=30).json()
        if async_enhance and wait_enhance_seconds > 0:
            _wait_for_enhancement(
                session=session,
                base_url=base_url,
                canvas_id=str(observe["canvas_id"]),
                timeout_seconds=wait_enhance_seconds,
            )
            detail = session.get(f"{base_url}/api/v1/canvases/{observe['canvas_id']}", timeout=30).json()
        try:
            dry = session.post(
                f"{base_url}/api/v1/canvases/{observe['canvas_id']}/semantic-completion",
                json={"dry_run": True, "deadline_ms": deadline_ms},
                timeout=30,
            )
            dry.raise_for_status()
            semantic = dry.json()
        except Exception as exc:  # noqa: BLE001 - semantic dry-run is diagnostic unless VLM is requested.
            semantic = _semantic_dry_run_failure(str(exc))
            if run_vlm:
                raise
        if run_vlm:
            allowed_purposes = {
                str(purpose).strip()
                for purpose in list(roi_purposes or [])
                if str(purpose).strip()
            }
            roi_jobs = list(((semantic.get("roi_vlm") or {}).get("jobs") or []))
            if allowed_purposes:
                roi_jobs = [
                    job for job in roi_jobs
                    if str(job.get("purpose") or "") in allowed_purposes
                ]
            roi_ids = [
                str(job.get("roi_id"))
                for job in roi_jobs[:max_rois_per_window]
                if job.get("roi_id")
            ]
            run_resp = session.post(
                f"{base_url}/api/v1/canvases/{observe['canvas_id']}/semantic-completion",
                json={
                    "dry_run": False,
                    "deadline_ms": deadline_ms,
                    "accept_late": True,
                    "roi_ids": roi_ids,
                },
                timeout=max(30, deadline_ms / 1000 + 5),
            )
            run_resp.raise_for_status()
            semantic = run_resp.json()
            if wait_late_seconds > 0:
                _wait_for_late_semantic(
                    session=session,
                    base_url=base_url,
                    canvas_id=str(observe["canvas_id"]),
                    timeout_seconds=wait_late_seconds,
                )
            detail = session.get(f"{base_url}/api/v1/canvases/{observe['canvas_id']}", timeout=30).json()
    except Exception as exc:  # noqa: BLE001 - matrix rows preserve collection failures.
        error = str(exc)
    row = build_matrix_row(window=window, observe=observe, semantic=semantic, detail=detail, error=error)
    return row, observe, semantic, detail


def _semantic_dry_run_failure(error: str) -> dict[str, Any]:
    return {
        "status": "dry_run_failed",
        "next_action": "retry_semantic_completion",
        "stages": [
            {
                "name": "semantic_completion_dry_run",
                "status": "failed",
                "error": error,
            }
        ],
        "roi_vlm": {"jobs": []},
        "error": error,
    }


def _wait_for_enhancement(
    *,
    session: requests.Session,
    base_url: str,
    canvas_id: str,
    timeout_seconds: float,
) -> None:
    terminal = {
        "enhanced_ready",
        "semantic_ready",
        "semantic_timeout",
        "semantic_partial",
        "semantic_failed",
        "semantic_late_merged",
        "semantic_late_failed",
        "failed",
        "cancelled",
    }
    deadline = time.time() + max(0.0, float(timeout_seconds))
    while time.time() < deadline:
        processing = session.get(
            f"{base_url}/api/v1/canvases/{canvas_id}/processing",
            timeout=10,
        ).json()
        state = str(processing.get("processing_state") or "")
        if state in terminal:
            return
        time.sleep(0.25)


def _wait_for_late_semantic(
    *,
    session: requests.Session,
    base_url: str,
    canvas_id: str,
    timeout_seconds: float,
) -> None:
    terminal = {
        "semantic_late_merged",
        "semantic_late_failed",
        "semantic_failed",
        "failed",
        "cancelled",
    }
    deadline = time.time() + max(0.0, float(timeout_seconds))
    while time.time() < deadline:
        processing = session.get(
            f"{base_url}/api/v1/canvases/{canvas_id}/processing",
            timeout=10,
        ).json()
        state = str(processing.get("processing_state") or "")
        if state in terminal:
            return
        time.sleep(0.25)


def _sample_name(window: dict[str, Any]) -> str:
    raw = f"{window.get('process_name') or 'app'}_{window.get('hwnd') or 0}_{window.get('title') or ''}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return slug[:96] or "sample"


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live_sample_matrix_2026-05-21"))
    parser.add_argument("--include-process", action="append", default=[])
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--max-rois-per-window", type=int, default=1)
    parser.add_argument("--wait-late-seconds", type=float, default=0.0)
    parser.add_argument("--wait-enhance-seconds", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=int, default=2000)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--async-enhance", action="store_true", help="Use observe fast path with background enhancement")
    parser.add_argument("--roi-purpose", action="append", default=[], help="Limit run-vlm ROI jobs to matching purpose(s)")
    args = parser.parse_args()
    summary = collect_matrix(
        base_url=args.base_url.rstrip("/"),
        output_dir=args.output_dir,
        include_processes=args.include_process,
        max_windows=args.max_windows,
        run_vlm=args.run_vlm,
        max_rois_per_window=args.max_rois_per_window,
        wait_late_seconds=args.wait_late_seconds,
        wait_enhance_seconds=args.wait_enhance_seconds,
        deadline_ms=args.deadline_ms,
        clean_output=args.clean_output,
        async_enhance=args.async_enhance,
        roi_purposes=args.roi_purpose,
    )
    print(
        f"samples={summary['sample_count']} errors={summary['error_count']} "
        f"semantic={_format_counts(summary['semantic_status_counts'])}"
    )
    return 0 if int(summary["error_count"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
