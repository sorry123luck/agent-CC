from __future__ import annotations

from typing import Any


def full_local_enhancement_strategy(canvas: Any) -> str:
    """Choose the local enhancement strategy for a fast canvas."""
    artifacts = getattr(canvas, "artifacts", None) or {}
    mode = str((artifacts.get("visual_pattern") or {}).get("mode") or "")
    warnings = {str(item) for item in (artifacts.get("perception_quality") or {}).get("warnings") or []}
    if mode == "collaboration_inbox" and "sparse_elements" in warnings:
        return "lightweight_uia_vision"
    return "full"


def full_local_enhancement_skip_reason(canvas: Any) -> str:
    """Backward-compatible skip reason for callers that only support skipping."""
    return ""
