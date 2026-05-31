"""VLM layout-audit contract helpers.

Layout audit is a low-frequency fallback for screenshots where local fast
observe cannot produce enough coarse regions. It is review-only: VLM may suggest
coarse relative layout regions, but never safe controls or click targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ALLOWED_LAYOUT_AUDIT_OUTPUTS = ["layout_regions", "layout_summary", "review_only_hints"]
LAYOUT_COORDINATE_KEYS = {
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
class LayoutAuditValidation:
    accepted: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_layout_audit_job(canvas: Any) -> dict[str, Any] | None:
    """Build a single full-screenshot layout audit job when quality gates require it."""
    artifacts = getattr(canvas, "artifacts", None) or {}
    quality = artifacts.get("perception_quality") or {}
    warnings = [str(item) for item in list(quality.get("warnings") or [])]
    if "layout_audit_needed" not in warnings:
        return None

    size = artifacts.get("screenshot_size") or []
    screenshot_size = [int(size[0]), int(size[1])] if isinstance(size, list) and len(size) == 2 else [0, 0]
    return {
        "canvas_id": str(getattr(canvas, "canvas_id", "")),
        "mode_guess": str((artifacts.get("visual_pattern") or {}).get("mode") or quality.get("mode_guess") or ""),
        "screenshot_size": screenshot_size,
        "quality_warnings": warnings,
        "existing_geometric_regions": list(artifacts.get("geometric_regions") or [])[:8],
        "allowed_outputs": list(ALLOWED_LAYOUT_AUDIT_OUTPUTS),
        "review_only": True,
        "max_layout_regions": 8,
    }


def validate_layout_audit_response(canvas: Any, response: dict[str, Any]) -> LayoutAuditValidation:
    """Validate a layout-audit response before storing it."""
    errors: list[str] = []
    warnings: list[str] = []
    regions = list(response.get("layout_regions") or [])
    if len(regions) > 8:
        errors.append("too_many_layout_regions")
    if _contains_disallowed_coordinate_keys(response):
        warnings.append("coordinate_fields_ignored")

    for index, region in enumerate(regions[:8]):
        role = str(region.get("role") or "")
        if not role:
            errors.append(f"missing_role:{index}")
        if not _valid_relative_bounds(region.get("relative_bounds")):
            errors.append(f"invalid_relative_bounds:{index}")

    return LayoutAuditValidation(accepted=not errors, errors=errors, warnings=warnings)


def merge_layout_audit_response(
    canvas: Any,
    response: dict[str, Any],
    *,
    status: str = "success",
) -> LayoutAuditValidation:
    """Store accepted layout audit as review-only semantic layout evidence."""
    validation = validate_layout_audit_response(canvas, response)
    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        canvas.artifacts = {}
        artifacts = canvas.artifacts
    if not validation.accepted:
        artifacts.setdefault("vlm_layout_audit_rejected", []).append({
            "status": status,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "received_at": datetime.now(timezone.utc).isoformat(),
        })
        return validation

    artifacts["vlm_layout_audit"] = {
        "status": status,
        "review_only": True,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "warnings": list(validation.warnings),
        "layout_summary": str(response.get("layout_summary") or ""),
        "layout_regions": [
            _strip_disallowed_coordinates(region)
            for region in list(response.get("layout_regions") or [])[:8]
        ],
        "review_only_hints": [str(item) for item in list(response.get("review_only_hints") or [])],
    }
    return validation


def _valid_relative_bounds(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        left, top, right, bottom = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return 0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0


def _strip_disallowed_coordinates(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_disallowed_coordinates(child)
            for key, child in value.items()
            if str(key) not in LAYOUT_COORDINATE_KEYS
        }
    if isinstance(value, list):
        return [_strip_disallowed_coordinates(item) for item in value]
    return value


def _contains_disallowed_coordinate_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in LAYOUT_COORDINATE_KEYS:
                return True
            if _contains_disallowed_coordinate_keys(child):
                return True
    elif isinstance(value, list):
        return any(_contains_disallowed_coordinate_keys(item) for item in value)
    return False
