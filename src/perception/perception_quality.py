"""Perception quality scoring from local canvas evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.perception.visual_pattern import VisualPatternClassifier


@dataclass(frozen=True)
class PerceptionQuality:
    """Compact quality summary for a canvas."""

    usable_state: str
    mode_guess: str
    element_count: int
    region_count: int
    geometric_region_count: int
    unknown_role_ratio: float
    root_shell_like_count: int
    local_coordinate_trust: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable_state": self.usable_state,
            "mode_guess": self.mode_guess,
            "element_count": self.element_count,
            "region_count": self.region_count,
            "geometric_region_count": self.geometric_region_count,
            "unknown_role_ratio": self.unknown_role_ratio,
            "root_shell_like_count": self.root_shell_like_count,
            "local_coordinate_trust": self.local_coordinate_trust,
            "warnings": list(self.warnings),
        }


class PerceptionQualityEvaluator:
    """Evaluate whether a canvas is usable and how it should be enhanced."""

    def __init__(self, classifier: VisualPatternClassifier | None = None) -> None:
        self._classifier = classifier or VisualPatternClassifier()

    def evaluate(self, canvas: Any) -> PerceptionQuality:
        elements = list(getattr(canvas, "elements", []) or [])
        regions = list(getattr(canvas, "regions", []) or [])
        geometric_regions = self._geometric_regions(canvas)
        mode = self._classifier.classify(canvas)

        element_count = len(elements)
        region_count = len(regions)
        geometric_count = len(geometric_regions)
        unknown_ratio = self._unknown_ratio(elements)
        root_shell_like_count = self._root_shell_like_count(canvas, elements)
        screenshot_available = self._has_screenshot_size(canvas)

        warnings: list[str] = []
        if not screenshot_available:
            warnings.append("screenshot_missing")
        if element_count <= 1:
            warnings.append("sparse_elements")
        if region_count <= 2:
            warnings.append("coarse_regions")
        if unknown_ratio >= 0.75 and element_count >= 8:
            warnings.append("unknown_role_heavy")
        if geometric_count > 32:
            warnings.append("geometric_overfragmented")
        if root_shell_like_count >= 2:
            warnings.append("root_shell_candidates")
        if element_count <= 1 and geometric_count < 3:
            warnings.append("layout_audit_needed")
            if not screenshot_available:
                warnings.append("layout_audit_blocked_no_screenshot")

        if mode.mode == "loading_state":
            usable_state = "loading"
        elif element_count <= 1:
            usable_state = "unreliable"
        elif warnings:
            usable_state = "usable_with_warnings"
        else:
            usable_state = "usable"

        local_trust = "low" if usable_state in {"unreliable", "loading"} else "medium"
        if usable_state == "usable" and unknown_ratio < 0.5:
            local_trust = "high"

        return PerceptionQuality(
            usable_state=usable_state,
            mode_guess=mode.mode,
            element_count=element_count,
            region_count=region_count,
            geometric_region_count=geometric_count,
            unknown_role_ratio=round(unknown_ratio, 4),
            root_shell_like_count=root_shell_like_count,
            local_coordinate_trust=local_trust,
            warnings=warnings,
        )

    def _geometric_regions(self, canvas: Any) -> list[dict[str, Any]]:
        artifacts = getattr(canvas, "artifacts", None) or {}
        return list(artifacts.get("geometric_regions") or [])

    def _unknown_ratio(self, elements: list[Any]) -> float:
        if not elements:
            return 1.0
        unknown_count = 0
        for elem in elements:
            role = getattr(elem, "semantic_role", "unknown")
            role_value = getattr(role, "value", str(role))
            if role_value == "unknown" and not self._has_refined_semantics(elem):
                unknown_count += 1
        return unknown_count / len(elements)

    def _has_refined_semantics(self, elem: Any) -> bool:
        label = str(getattr(elem, "role_label", "") or "").strip()
        tags = list(getattr(elem, "semantic_tags", []) or [])
        status = str(getattr(elem, "refine_status", "") or "").strip().lower()
        source = str(getattr(elem, "role_source", "") or "").strip().lower()
        if label and source in {"roi_vlm", "heuristic", "agent", "memory", "manual", "profile"}:
            return True
        if tags and status in {"refined", "reviewed"}:
            return True
        return False

    def _root_shell_like_count(self, canvas: Any, elements: list[Any]) -> int:
        width, height = self._canvas_size(canvas)
        if width <= 0 or height <= 0:
            return 0
        count = 0
        for elem in elements:
            bounds = getattr(elem, "bounds", None)
            ctype = str(getattr(elem, "control_type", "") or "")
            if not bounds or not any(t in ctype for t in ("Window", "Pane", "Document")):
                continue
            left, top, right, bottom = [int(v) for v in bounds]
            if (right - left) >= width * 0.85 and (bottom - top) >= height * 0.85:
                count += 1
        return count

    def _canvas_size(self, canvas: Any) -> tuple[int, int]:
        artifacts = getattr(canvas, "artifacts", None) or {}
        size = artifacts.get("screenshot_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
        window = getattr(canvas, "window", None)
        rect = getattr(window, "rect_client", None) if window else None
        if rect:
            return int(rect[2] - rect[0]), int(rect[3] - rect[1])
        return 0, 0

    def _has_screenshot_size(self, canvas: Any) -> bool:
        artifacts = getattr(canvas, "artifacts", None) or {}
        size = artifacts.get("screenshot_size")
        return isinstance(size, (list, tuple)) and len(size) == 2 and int(size[0]) > 0 and int(size[1]) > 0
