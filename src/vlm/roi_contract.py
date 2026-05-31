"""Region-first VLM supplement contract helpers.

Local perception owns coordinates and candidate IDs. ROI VLM calls may only
attach semantics to those local IDs, including late responses observed during
timeout tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ALLOWED_ROI_VLM_OUTPUTS = [
    "region_semantics",
    "candidate_annotations",
    "review_only_hints",
]

COORDINATE_KEYS = {
    "bounds",
    "bbox",
    "box",
    "rect",
    "x",
    "y",
    "width",
    "height",
    "left",
    "top",
    "right",
    "bottom",
    "click_point",
}


@dataclass(frozen=True)
class RoiVlmValidation:
    accepted: bool
    roi_id: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_roi_vlm_jobs(canvas: Any) -> list[dict[str, Any]]:
    """Build bounded ROI VLM jobs from the local roi_selection_plan artifact."""
    artifacts = getattr(canvas, "artifacts", None) or {}
    plan = artifacts.get("roi_selection_plan") or {}
    jobs: list[dict[str, Any]] = []
    assigned_candidate_ids: set[str] = set()
    for roi in list(plan.get("rois") or []):
        roi_id = str(roi.get("roi_id") or "")
        bounds = list(roi.get("bounds") or [])
        if not roi_id or len(bounds) != 4:
            continue
        allowed_outputs = [
            item for item in list(roi.get("allowed_outputs") or ALLOWED_ROI_VLM_OUTPUTS)
            if item in ALLOWED_ROI_VLM_OUTPUTS
        ]
        candidate_refs, local_text_candidate_ids = _unassigned_candidate_refs_for_roi(
            canvas,
            roi,
            assigned_candidate_ids,
        )
        job = {
            "canvas_id": str(getattr(canvas, "canvas_id", "")),
            "roi_id": roi_id,
            "bounds": [int(v) for v in bounds[:4]],
            "mode": str(plan.get("mode") or roi.get("mode") or ""),
            "purpose": str(roi.get("purpose") or ""),
            "app_process": str(getattr(getattr(canvas, "app", None), "process_name", "") or ""),
            "window_title": str(getattr(getattr(canvas, "window", None), "title", "") or ""),
            "candidate_ids": [ref["candidate_id"] for ref in candidate_refs],
            "candidate_refs": candidate_refs,
            "local_text_candidate_ids": local_text_candidate_ids,
            "allowed_outputs": allowed_outputs or list(ALLOWED_ROI_VLM_OUTPUTS),
            "vlm_task": str(roi.get("vlm_task") or "region_annotation"),
        }
        assigned_candidate_ids.update(job["candidate_ids"])
        jobs.append(job)
    return jobs


def validate_roi_vlm_response(canvas: Any, response: dict[str, Any]) -> RoiVlmValidation:
    """Validate that a ROI VLM response only references local IDs."""
    roi_id = str(response.get("roi_id") or "")
    local_roi_ids = {job["roi_id"] for job in build_roi_vlm_jobs(canvas)}
    local_candidate_ids = _all_candidate_ids(canvas)
    errors: list[str] = []
    warnings: list[str] = []

    if not roi_id or roi_id not in local_roi_ids:
        errors.append("unknown_roi_id")

    for item in list(response.get("candidate_annotations") or []):
        candidate_id = str(item.get("candidate_id") or item.get("element_id") or "")
        if not candidate_id or candidate_id not in local_candidate_ids:
            errors.append(f"unknown_candidate_id:{candidate_id or '<empty>'}")

    if _contains_coordinate_keys(response):
        warnings.append("coordinate_fields_ignored")

    return RoiVlmValidation(
        accepted=not errors,
        roi_id=roi_id,
        errors=errors,
        warnings=warnings,
    )


def merge_roi_vlm_response(
    canvas: Any,
    response: dict[str, Any],
    *,
    status: str = "success",
) -> RoiVlmValidation:
    """Store accepted ROI VLM semantics as diagnostic supplements.

    Late responses are accepted in the current testing phase as long as they
    reference existing local ROI/candidate IDs. Coordinate fields are stripped.
    """
    validation = validate_roi_vlm_response(canvas, response)
    if not validation.accepted:
        _append_rejected(canvas, response, validation, status=status)
        return validation

    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        canvas.artifacts = {}
        artifacts = canvas.artifacts
    supplements = artifacts.setdefault("roi_vlm_semantic_supplements", [])
    region_semantics = _strip_coordinates(response.get("region_semantics") or {})
    candidate_annotations = [
        _strip_coordinates(item)
        for item in list(response.get("candidate_annotations") or [])
    ]
    supplements.append({
        "roi_id": validation.roi_id,
        "status": status,
        "late_result_accepted": status.startswith("timeout_late") or status.startswith("late"),
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "warnings": list(validation.warnings),
        "region_semantics": region_semantics,
        "candidate_annotations": candidate_annotations,
        "review_only_hints": [
            str(item) for item in list(response.get("review_only_hints") or [])
        ],
    })
    _project_candidate_annotations(canvas, candidate_annotations, region_semantics)
    _refresh_perception_quality(canvas)
    return validation


def record_roi_vlm_timeout(
    canvas: Any,
    job: dict[str, Any],
    *,
    timeout_seconds: float,
) -> None:
    """Record a ROI VLM timeout without invalidating late responses."""
    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        canvas.artifacts = {}
        artifacts = canvas.artifacts
    timeouts = artifacts.setdefault("roi_vlm_timeouts", [])
    timeouts.append({
        "roi_id": str(job.get("roi_id") or ""),
        "candidate_ids": list(job.get("candidate_ids") or []),
        "timeout_seconds": float(timeout_seconds),
        "timed_out_at": datetime.now(timezone.utc).isoformat(),
        "discard_late_response": False,
    })


def record_roi_vlm_late_failure(
    canvas: Any,
    job: dict[str, Any],
    *,
    error: str,
    status: str = "timeout_late_failed",
) -> None:
    """Record a late ROI VLM worker failure for diagnostic retention."""
    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        canvas.artifacts = {}
        artifacts = canvas.artifacts
    failures = artifacts.setdefault("roi_vlm_late_failures", [])
    failures.append({
        "roi_id": str(job.get("roi_id") or ""),
        "candidate_ids": list(job.get("candidate_ids") or []),
        "status": status,
        "error": str(error),
        "received_at": datetime.now(timezone.utc).isoformat(),
    })


def _append_rejected(
    canvas: Any,
    response: dict[str, Any],
    validation: RoiVlmValidation,
    *,
    status: str,
) -> None:
    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        canvas.artifacts = {}
        artifacts = canvas.artifacts
    rejected = artifacts.setdefault("roi_vlm_rejected_responses", [])
    rejected.append({
        "status": status,
        "roi_id": str(response.get("roi_id") or ""),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "received_at": datetime.now(timezone.utc).isoformat(),
    })


def _known_candidate_ids_for_roi(canvas: Any, roi: dict[str, Any]) -> list[str]:
    local_ids = _all_candidate_ids(canvas)
    return [
        candidate_id for candidate_id in [str(item) for item in list(roi.get("candidate_ids") or [])]
        if candidate_id in local_ids
    ]


def _unassigned_candidate_ids_for_roi(
    canvas: Any,
    roi: dict[str, Any],
    assigned_candidate_ids: set[str],
) -> list[str]:
    refs, _local_text_candidate_ids = _unassigned_candidate_refs_for_roi(canvas, roi, assigned_candidate_ids)
    return [
        ref["candidate_id"]
        for ref in refs
    ]


def _unassigned_candidate_refs_for_roi(
    canvas: Any,
    roi: dict[str, Any],
    assigned_candidate_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id = _candidate_by_id(canvas)
    known_ids = [
        candidate_id
        for candidate_id in _known_candidate_ids_for_roi(canvas, roi)
        if candidate_id not in assigned_candidate_ids
    ]
    roi_bounds = _coerce_bounds(roi.get("bounds"))
    roi_purpose = str(roi.get("purpose") or "").strip()
    filtered_ids = [
        candidate_id
        for candidate_id in known_ids
        if _is_roi_vlm_candidate(by_id.get(candidate_id), roi_bounds, roi_purpose=roi_purpose)
    ]
    ocr_blocks = _canvas_ocr_blocks(canvas)
    local_text_candidate_ids = [
        candidate_id
        for candidate_id in filtered_ids
        if _has_local_text_evidence(by_id.get(candidate_id), ocr_blocks=ocr_blocks)
    ]
    sorted_ids = sorted(
        [candidate_id for candidate_id in filtered_ids if candidate_id not in set(local_text_candidate_ids)],
        key=lambda candidate_id: _candidate_sort_key(by_id.get(candidate_id), candidate_id),
    )
    refs = [
        _candidate_ref(marker=f"C{idx + 1}", candidate_id=candidate_id, element=by_id.get(candidate_id))
        for idx, candidate_id in enumerate(sorted_ids)
    ]
    refs = _dedupe_chat_composer_refs(refs, roi, canvas)
    _attach_chat_composer_position_hints(refs, roi, canvas)
    return refs, sorted(local_text_candidate_ids, key=lambda candidate_id: _candidate_sort_key(by_id.get(candidate_id), candidate_id))


def _all_candidate_ids(canvas: Any) -> set[str]:
    return {
        str(getattr(element, "element_id", ""))
        for element in list(getattr(canvas, "elements", []) or [])
        if getattr(element, "element_id", "")
    }


def _candidate_by_id(canvas: Any) -> dict[str, Any]:
    return {
        str(getattr(element, "element_id", "")): element
        for element in list(getattr(canvas, "elements", []) or [])
        if getattr(element, "element_id", "")
    }


def _candidate_sort_key(element: Any, candidate_id: str) -> tuple[int, int, str]:
    bounds = getattr(element, "bounds", None)
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        try:
            return (int(bounds[1]), int(bounds[0]), candidate_id)
        except (TypeError, ValueError):
            pass
    return (10**9, 10**9, candidate_id)


def _is_roi_vlm_candidate(
    element: Any,
    roi_bounds: tuple[int, int, int, int] | None,
    *,
    roi_purpose: str = "",
) -> bool:
    if element is None:
        return True
    role = _semantic_role_value(getattr(element, "semantic_role", ""))
    control_type = str(getattr(element, "control_type", "") or "").strip().lower()
    if roi_purpose.strip().lower() == "composer" and (
        role in {"message_input", "chat_input"}
        or control_type == "syntheticinputcontrol"
        or str(getattr(element, "element_id", "") or "").startswith("synthetic_chat_composer_input")
        or str(getattr(element, "element_id", "") or "").startswith("synthetic_collaboration_composer_input")
    ):
        return False
    if role in {"container", "layout"}:
        return False
    if control_type in {"windowcontrol", "panecontrol"}:
        return False
    bounds = _coerce_bounds(getattr(element, "bounds", None))
    if bounds is None or roi_bounds is None:
        return True
    if _looks_like_titlebar_window_button(bounds, roi_bounds, roi_purpose=roi_purpose, control_type=control_type):
        return False
    roi_area = max(1, (roi_bounds[2] - roi_bounds[0]) * (roi_bounds[3] - roi_bounds[1]))
    candidate_area = max(0, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
    if candidate_area > roi_area * 0.65:
        return False
    return True


def _looks_like_titlebar_window_button(
    bounds: tuple[int, int, int, int],
    roi_bounds: tuple[int, int, int, int],
    *,
    roi_purpose: str,
    control_type: str,
) -> bool:
    purpose = roi_purpose.strip().lower()
    if purpose in {"window_controls", "title_bar", "top_header", "menu_toolbar"}:
        return False
    if control_type not in {"icon", "buttoncontrol", "imagecontrol"}:
        return False
    left, top, right, bottom = bounds
    roi_left, roi_top, roi_right, _roi_bottom = roi_bounds
    height = bottom - top
    width = right - left
    if top > roi_top + 8 or height > 44 or width > 56:
        return False
    return right >= roi_right - 180 and left >= roi_left


def _semantic_role_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _coerce_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        left, top, right, bottom = [int(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _candidate_ref(*, marker: str, candidate_id: str, element: Any) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "marker": marker,
        "candidate_id": candidate_id,
    }
    bounds = getattr(element, "bounds", None)
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        try:
            ref["bounds"] = [int(v) for v in bounds[:4]]
        except (TypeError, ValueError):
            pass
    control_type = str(getattr(element, "control_type", "") or "").strip()
    if control_type:
        ref["control_type"] = control_type
    text = str(getattr(element, "text", "") or getattr(element, "name", "") or "").strip()
    if text:
        ref["text"] = text[:40]
    return ref


def _attach_chat_composer_position_hints(refs: list[dict[str, Any]], roi: dict[str, Any], canvas: Any) -> None:
    artifacts = getattr(canvas, "artifacts", None) or {}
    plan = artifacts.get("roi_selection_plan") or {}
    mode = str(plan.get("mode") or roi.get("mode") or "").strip()
    purpose = str(roi.get("purpose") or "").strip()
    if mode not in {"chat_workspace", "chat_document", "collaboration_inbox"} or purpose != "composer":
        return
    actionable_refs = [
        ref for ref in refs
        if _is_composer_action_candidate_ref(ref) and _coerce_bounds(ref.get("bounds")) is not None
    ]
    if len(actionable_refs) < 3:
        return
    actionable_refs.sort(key=lambda ref: (_coerce_bounds(ref.get("bounds")) or (0, 0, 0, 0))[0])
    for ref in actionable_refs:
        ref.pop("local_hint", None)

    send_ref = _rightmost_send_like_composer_ref(actionable_refs, roi)
    left_toolbar_refs = [
        ref
        for ref in actionable_refs
        if ref is not send_ref and not (send_ref is not None and _is_right_side_composer_ref(ref, roi))
    ]
    for ref, hint in zip(left_toolbar_refs, _CHAT_COMPOSER_LEFT_TOOLBAR_HINTS):
        ref["local_hint"] = hint
    if send_ref is not None:
        send_ref["local_hint"] = "send_button"

    hint_priority = {
        "emoji_button": 0,
        "attachment_or_app_button": 1,
        "file_button": 2,
        "send_button": 3,
        "screenshot_button": 4,
        "voice_input_button": 5,
    }
    refs.sort(
        key=lambda ref: (
            hint_priority.get(str(ref.get("local_hint") or ""), 100),
            _candidate_sort_key_for_ref(ref),
        )
    )
    hinted_refs = [ref for ref in refs if str(ref.get("local_hint") or "")]
    if len(hinted_refs) >= 3:
        refs[:] = hinted_refs
    for idx, ref in enumerate(refs):
        ref["marker"] = f"C{idx + 1}"


def _dedupe_chat_composer_refs(refs: list[dict[str, Any]], roi: dict[str, Any], canvas: Any) -> list[dict[str, Any]]:
    artifacts = getattr(canvas, "artifacts", None) or {}
    plan = artifacts.get("roi_selection_plan") or {}
    mode = str(plan.get("mode") or roi.get("mode") or "").strip()
    purpose = str(roi.get("purpose") or "").strip()
    if mode not in {"chat_workspace", "chat_document", "collaboration_inbox"} or purpose != "composer":
        return refs

    kept: list[dict[str, Any]] = []
    for ref in refs:
        if _is_composer_decorative_ref(ref):
            continue
        duplicate_index = next(
            (
                idx
                for idx, existing in enumerate(kept)
                if _same_or_near_bounds(ref, existing)
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(ref)
            continue
        if _composer_ref_priority(ref) < _composer_ref_priority(kept[duplicate_index]):
            kept[duplicate_index] = ref
    for idx, ref in enumerate(kept):
        ref["marker"] = f"C{idx + 1}"
    return kept


def _is_composer_decorative_ref(ref: dict[str, Any]) -> bool:
    bounds = _coerce_bounds(ref.get("bounds"))
    if bounds is None:
        return False
    control_type = str(ref.get("control_type") or "").strip().lower()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return control_type in {"imagecontrol", "groupcontrol"} and width >= 80 and height <= 22


def _same_or_near_bounds(left_ref: dict[str, Any], right_ref: dict[str, Any]) -> bool:
    left = _coerce_bounds(left_ref.get("bounds"))
    right = _coerce_bounds(right_ref.get("bounds"))
    if left is None or right is None:
        return False
    return all(abs(left[idx] - right[idx]) <= 2 for idx in range(4))


def _composer_ref_priority(ref: dict[str, Any]) -> int:
    control_type = str(ref.get("control_type") or "").strip().lower()
    if control_type in {"icon", "buttoncontrol"}:
        return 0
    if control_type == "imagecontrol":
        return 1
    if control_type == "groupcontrol":
        return 2
    return 3


def _is_composer_action_candidate_ref(ref: dict[str, Any]) -> bool:
    control_type = str(ref.get("control_type") or "").strip().lower()
    return control_type in {"icon", "buttoncontrol", "imagecontrol", "groupcontrol"}


def _rightmost_send_like_composer_ref(
    icon_refs: list[dict[str, Any]],
    roi: dict[str, Any],
) -> dict[str, Any] | None:
    roi_bounds = _coerce_bounds(roi.get("bounds"))
    if roi_bounds is None:
        return icon_refs[-1] if icon_refs else None
    roi_left, roi_top, roi_right, roi_bottom = roi_bounds
    roi_width = max(1, roi_right - roi_left)
    roi_height = max(1, roi_bottom - roi_top)
    right_cluster = [
        ref for ref in icon_refs
        if ((_coerce_bounds(ref.get("bounds")) or (0, 0, 0, 0))[0] - roi_left) / roi_width >= 0.60
    ]
    if not right_cluster:
        return None
    return max(
        right_cluster,
        key=lambda ref: (
            _composer_send_vertical_score(ref, roi_top=roi_top, roi_bottom=roi_bottom, roi_height=roi_height),
            _candidate_width(ref),
            (_coerce_bounds(ref.get("bounds")) or (0, 0, 0, 0))[2],
        ),
    )


def _is_right_side_composer_ref(ref: dict[str, Any], roi: dict[str, Any]) -> bool:
    roi_bounds = _coerce_bounds(roi.get("bounds"))
    bounds = _coerce_bounds(ref.get("bounds"))
    if roi_bounds is None or bounds is None:
        return False
    roi_left, _roi_top, roi_right, _roi_bottom = roi_bounds
    roi_width = max(1, roi_right - roi_left)
    return (bounds[0] - roi_left) / roi_width >= 0.72


def _composer_send_vertical_score(ref: dict[str, Any], *, roi_top: int, roi_bottom: int, roi_height: int) -> int:
    bounds = _coerce_bounds(ref.get("bounds"))
    if bounds is None:
        return 0
    _left, top, _right, bottom = bounds
    if top >= roi_top + int(roi_height * 0.55):
        return 2
    if bottom >= roi_bottom - 32:
        return 1
    return 0


def _candidate_width(ref: dict[str, Any]) -> int:
    bounds = _coerce_bounds(ref.get("bounds"))
    if bounds is None:
        return 0
    return bounds[2] - bounds[0]


def _candidate_sort_key_for_ref(ref: dict[str, Any]) -> tuple[int, int, str]:
    bounds = _coerce_bounds(ref.get("bounds"))
    if bounds is None:
        return (10**9, 10**9, str(ref.get("candidate_id") or ""))
    return (bounds[1], bounds[0], str(ref.get("candidate_id") or ""))


_CHAT_COMPOSER_LEFT_TOOLBAR_HINTS = [
    "emoji_button",
    "attachment_or_app_button",
    "file_button",
    "screenshot_button",
    "voice_input_button",
]


def _has_local_text_evidence(element: Any, *, ocr_blocks: list[dict[str, Any]] | None = None) -> bool:
    if element is None:
        return False
    values = [
        getattr(element, "text", ""),
        getattr(element, "name", ""),
        getattr(element, "value", ""),
    ]
    if any(str(value or "").strip() for value in values):
        return True
    if _is_icon_like_element(element):
        return False
    attributes = getattr(element, "attributes", None)
    if isinstance(attributes, dict):
        if str(attributes.get("ocr_text", "") or "").strip():
            return True
    return _has_overlapping_ocr_text(element, ocr_blocks or [])


def _canvas_ocr_blocks(canvas: Any) -> list[dict[str, Any]]:
    artifacts = getattr(canvas, "artifacts", None)
    if not isinstance(artifacts, dict):
        return []
    blocks = artifacts.get("ocr_blocks")
    if not isinstance(blocks, list):
        return []
    return [block for block in blocks if isinstance(block, dict)]


def _has_overlapping_ocr_text(element: Any, ocr_blocks: list[dict[str, Any]]) -> bool:
    bounds = _coerce_bounds(getattr(element, "bounds", None))
    if bounds is None:
        return False
    for block in ocr_blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if _overlap_score(bounds, _coerce_bounds(block.get("bbox"))) >= 0.35:
            return True
    return False


def _overlap_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int] | None) -> float:
    if b is None:
        return 0.0
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(1, smaller)


def _is_icon_like_element(element: Any) -> bool:
    control_type = str(getattr(element, "control_type", "") or "").strip().lower()
    role = _semantic_role_value(getattr(element, "semantic_role", ""))
    return control_type in {"icon", "imagecontrol"} or role in {"icon", "image"}


def _project_candidate_annotations(
    canvas: Any,
    candidate_annotations: list[dict[str, Any]],
    region_semantics: dict[str, Any],
) -> None:
    """Project accepted ROI VLM candidate semantics onto local candidates."""
    if not candidate_annotations:
        return
    by_id = {
        str(getattr(element, "element_id", "")): element
        for element in list(getattr(canvas, "elements", []) or [])
        if getattr(element, "element_id", "")
    }
    region_role = str(region_semantics.get("role") or "").strip()
    for annotation in candidate_annotations:
        candidate_id = str(annotation.get("candidate_id") or annotation.get("element_id") or "")
        element = by_id.get(candidate_id)
        if element is None or _has_stronger_role_source(element):
            continue
        label = str(annotation.get("label") or annotation.get("role") or "").strip()
        role = str(annotation.get("role") or "").strip()
        projection_label = _projectable_role_label(element=element, label=label, role=role, region_role=region_role)
        if not projection_label:
            continue
        if _is_review_only_message_input_candidate(element):
            continue
        element.role_label = projection_label
        element.role_source = "roi_vlm"
        element.role_confidence = max(float(getattr(element, "role_confidence", 0.0) or 0.0), 0.72)
        element.refine_status = "refined"
        tags = list(getattr(element, "semantic_tags", []) or [])
        for tag in _roi_semantic_tags(role=role, region_role=region_role):
            if tag not in tags:
                tags.append(tag)
        element.semantic_tags = tags
        evidence = list(getattr(element, "role_evidence", []) or [])
        for item in _roi_role_evidence(role=role, label=label, region_role=region_role):
            if item not in evidence:
                evidence.append(item)
        element.role_evidence = evidence


def _has_stronger_role_source(element: Any) -> bool:
    source = str(getattr(element, "role_source", "") or "").strip().lower()
    return source in {"manual", "agent", "memory", "profile"}


def _is_review_only_message_input_candidate(element: Any) -> bool:
    attrs = getattr(element, "attributes", None) or {}
    element_id = str(getattr(element, "element_id", "") or "")
    role = _semantic_role_value(getattr(element, "semantic_role", ""))
    control_type = str(getattr(element, "control_type", "") or "").strip().lower()
    return (
        element_id.startswith("synthetic_chat_composer_input")
        or element_id.startswith("synthetic_collaboration_composer_input")
        or control_type == "syntheticinputcontrol"
        or (
            role in {"message_input", "chat_input"}
            and attrs.get("candidate_kind") == "message_input_candidate"
            and attrs.get("safe_to_type") is False
        )
        or (
            attrs.get("candidate_kind") == "message_input_candidate"
            and str(attrs.get("actionability") or "").strip().lower() == "review"
        )
    )


def _projectable_role_label(*, element: Any, label: str, role: str, region_role: str) -> str:
    label = label.strip()
    role = role.strip()
    if _is_dynamic_content_action_guess(label=label, role=role, region_role=region_role):
        return ""
    if _is_icon_only_composer_guess(element=element, label=label, role=role, region_role=region_role):
        return ""
    if label and label.lower() not in _LOW_INFORMATION_LABELS:
        return label
    if role and role.lower() not in _LOW_INFORMATION_LABELS:
        return role
    return ""


def _is_dynamic_content_action_guess(*, label: str, role: str, region_role: str) -> bool:
    region = region_role.strip().lower()
    if region not in _DYNAMIC_CONTENT_REGIONS:
        return False
    semantic = {label.strip().lower(), role.strip().lower()}
    return bool(semantic & _DYNAMIC_CONTENT_BLOCKED_ACTION_LABELS)


_DYNAMIC_CONTENT_REGIONS = {
    "chat_area",
    "content_feed",
    "detail_pane",
    "document_thread",
    "inbox_list",
    "message_thread",
    "message_stream",
}


_DYNAMIC_CONTENT_BLOCKED_ACTION_LABELS = {
    "action",
    "button",
    "input",
    "icon",
    "image",
    "send",
    "submit",
    "delete",
    "close",
    "minimize",
    "maximize",
    "fullscreen",
}


_LOW_INFORMATION_LABELS = {
    "action",
    "button",
    "control",
    "element",
    "icon",
    "image",
    "item",
    "label",
    "unknown",
}


def _is_icon_only_composer_guess(*, element: Any, label: str, role: str, region_role: str) -> bool:
    if region_role.strip().lower() != "composer":
        return False
    control_type = str(getattr(element, "control_type", "") or "").strip().lower()
    if control_type != "icon":
        return False
    text_like = (
        str(getattr(element, "text", "") or "")
        or str(getattr(element, "name", "") or "")
        or str(getattr(element, "value", "") or "")
    ).strip()
    if text_like:
        return False
    semantic = {label.strip().lower(), role.strip().lower()}
    return bool(semantic & _COMPOSER_ICON_GUESS_LABELS)


_COMPOSER_ICON_GUESS_LABELS = {
    "attachment",
    "cut",
    "emoji",
    "send",
    "voice",
}


def _roi_semantic_tags(*, role: str, region_role: str) -> list[str]:
    tags: list[str] = []
    if role:
        tags.append(f"roi_vlm.role.{role}")
    if region_role:
        tags.append(f"roi_vlm.region.{region_role}")
    return tags


def _roi_role_evidence(*, role: str, label: str, region_role: str) -> list[str]:
    evidence: list[str] = []
    if region_role:
        evidence.append(f"roi_vlm:{region_role}")
    if role:
        evidence.append(f"roi_vlm_role:{role}")
    if label:
        evidence.append(f"roi_vlm_label:{label}")
    return evidence


def _refresh_perception_quality(canvas: Any) -> None:
    try:
        from src.perception.perception_quality import PerceptionQualityEvaluator

        artifacts = getattr(canvas, "artifacts", None)
        if artifacts is None:
            canvas.artifacts = {}
            artifacts = canvas.artifacts
        artifacts["perception_quality"] = PerceptionQualityEvaluator().evaluate(canvas).to_dict()
    except Exception:
        return


def _strip_coordinates(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_coordinates(child)
            for key, child in value.items()
            if str(key) not in COORDINATE_KEYS
        }
    if isinstance(value, list):
        return [_strip_coordinates(item) for item in value]
    return value


def _contains_coordinate_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in COORDINATE_KEYS:
                return True
            if _contains_coordinate_keys(child):
                return True
    elif isinstance(value, list):
        return any(_contains_coordinate_keys(item) for item in value)
    return False
