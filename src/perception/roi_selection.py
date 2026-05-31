"""ROI selection plan for region-first VLM semantic supplement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.perception.visual_pattern import VisualPatternClassifier


ALLOWED_VLM_OUTPUTS = ["region_semantics", "candidate_annotations", "review_only_hints"]


@dataclass(frozen=True)
class RoiPlanEntry:
    """One bounded ROI selected for optional VLM supplement."""

    roi_id: str
    bounds: tuple[int, int, int, int]
    mode: str
    purpose: str
    priority: int
    source: str
    vlm_task: str = "region_annotation"
    allowed_outputs: list[str] = field(default_factory=lambda: list(ALLOWED_VLM_OUTPUTS))
    candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "bounds": list(self.bounds),
            "mode": self.mode,
            "purpose": self.purpose,
            "priority": self.priority,
            "source": self.source,
            "vlm_task": self.vlm_task,
            "allowed_outputs": list(self.allowed_outputs),
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class RoiSelectionPlan:
    """Selected ROI set for a canvas."""

    mode: str
    rois: list[RoiPlanEntry]
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rois": [roi.to_dict() for roi in self.rois],
            "skipped_reason": self.skipped_reason,
        }


class RoiSelectionPlanner:
    """Select 3-8 coarse ROIs without asking VLM to draw boxes."""

    def __init__(self, classifier: VisualPatternClassifier | None = None) -> None:
        self._classifier = classifier or VisualPatternClassifier()

    def plan(self, canvas: Any) -> RoiSelectionPlan:
        pattern = self._classifier.classify(canvas)
        mode = pattern.mode
        if mode == "loading_state":
            return RoiSelectionPlan(mode=mode, rois=[], skipped_reason="loading_state")

        width, height = self._canvas_size(canvas)
        candidates = list(getattr(canvas, "elements", []) or [])
        rois: list[RoiPlanEntry] = []

        audit_rois = self._layout_audit_rois(canvas, mode, width, height, candidates)
        if audit_rois:
            return RoiSelectionPlan(mode=mode, rois=audit_rois[:8])

        for purpose, bounds in self._mode_skeleton(mode, width, height, candidates):
            rois.append(
                RoiPlanEntry(
                    roi_id=f"roi_{len(rois)}",
                    bounds=self._clamp(bounds, width, height),
                    mode=mode,
                    purpose=purpose,
                    priority=len(rois) + 1,
                    source="layout_heuristic",
                    candidate_ids=self._candidate_ids_in_bounds(candidates, bounds),
                )
            )

        for geo in ([] if self._mode_has_complete_skeleton(mode) else self._geometric_regions(canvas)):
            if len(rois) >= 8:
                break
            bounds = tuple(int(v) for v in geo.get("bounds", []))
            if len(bounds) != 4 or not self._large_enough(bounds, width, height):
                continue
            if self._overlaps_existing(bounds, [r.bounds for r in rois]):
                continue
            rois.append(
                RoiPlanEntry(
                    roi_id=f"roi_{len(rois)}",
                    bounds=self._clamp(bounds, width, height),
                    mode=mode,
                    purpose="candidate_region",
                    priority=len(rois) + 1,
                    source="geometric_region",
                    vlm_task="missing_audit" if len(rois) >= 5 else "region_annotation",
                    candidate_ids=self._candidate_ids_in_bounds(candidates, bounds),
                )
            )

        if not rois and width > 0 and height > 0:
            rois.append(
                RoiPlanEntry(
                    roi_id="roi_0",
                    bounds=(0, 0, width, height),
                    mode=mode,
                    purpose="full_visible_region",
                    priority=1,
                    source="fallback",
                )
            )

        return RoiSelectionPlan(mode=mode, rois=rois[:8])

    def _mode_has_complete_skeleton(self, mode: str) -> bool:
        return mode in {
            "chat_workspace",
            "chat_document",
            "chat_search_results",
            "collaboration_inbox",
            "collaboration_search_overlay",
            "chat_app_browse_page",
            "collaboration_browse_page",
            "security_dashboard",
            "control_matrix",
            "control_dashboard",
            "media_home",
            "browser_profile_manager",
            "list_management",
        }

    def _mode_skeleton(
        self,
        mode: str,
        width: int,
        height: int,
        candidates: list[Any] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        if width <= 0 or height <= 0:
            return []
        side_w = max(72, int(width * 0.18))
        header_h = max(56, int(height * 0.10))
        bottom_h = max(72, int(height * 0.14))
        chat_composer_top = self._chat_composer_top(
            width=width,
            height=height,
            side_w=side_w,
            default_top=height - bottom_h,
            candidates=list(candidates or []),
        )
        plans = {
            "chat_workspace": [
                ("navigation_and_list", (0, 0, side_w + int(width * 0.12), height)),
                ("message_stream", (side_w, header_h, width, chat_composer_top)),
                ("composer", (side_w, chat_composer_top, width, height)),
            ],
            "collaboration_inbox": [
                ("app_rail", (0, 0, side_w, height)),
                ("inbox_list", (side_w, 0, int(width * 0.50), height)),
                ("message_thread", (int(width * 0.50), 0, width, height - bottom_h)),
                ("composer", (int(width * 0.50), height - bottom_h, width, height)),
            ],
            "collaboration_browse_page": [
                ("app_rail", (0, 0, side_w, height)),
                ("workspace_navigation", (side_w, 0, int(width * 0.58), height)),
                ("main_workspace", (int(width * 0.58), 0, width, height)),
            ],
            "collaboration_search_overlay": [
                ("search_overlay", (int(width * 0.10), int(height * 0.06), int(width * 0.90), int(height * 0.94))),
                ("background_context", (0, 0, width, height)),
            ],
            "chat_app_browse_page": [
                ("app_rail", (0, 0, side_w, height)),
                ("item_list", (side_w, 0, int(width * 0.58), height)),
                ("detail_pane", (int(width * 0.58), 0, width, height)),
            ],
            "chat_search_results": [
                ("app_rail", (0, 0, max(60, int(width * 0.07)), height)),
                ("search_results_list", (max(60, int(width * 0.06)), 0, int(width * 0.34), height)),
                ("conversation_context", (int(width * 0.32), 0, width, height)),
            ],
            "list_management": [
                ("left_navigation", (0, 0, side_w, height)),
                ("toolbar_search_filters", (side_w, 0, width, int(height * 0.25))),
                ("table_or_list", (side_w, int(height * 0.18), width, int(height * 0.88))),
                ("pagination_status", (side_w, int(height * 0.84), width, height)),
            ],
            "media_home": [
                ("left_navigation", (0, 0, side_w, height)),
                ("top_search_account", (side_w, 0, width, header_h)),
                ("content_feed", (side_w, header_h, width, height - bottom_h)),
                ("bottom_player", (0, height - bottom_h, width, height)),
            ],
            "video_feed_home": [
                ("left_navigation", (0, 0, side_w, height)),
                ("top_tabs_search", (side_w, 0, width, header_h)),
                ("representative_card_grid", (side_w, header_h, width, height)),
            ],
            "control_matrix": [
                ("left_control_columns", (0, 0, int(width * 0.45), height)),
                ("middle_control_columns", (int(width * 0.35), 0, int(width * 0.72), height)),
                ("right_master_section", (int(width * 0.68), 0, width, height)),
            ],
            "control_dashboard": [
                ("left_navigation", (0, 0, side_w, height)),
                ("mode_tabs", (side_w, 0, int(width * 0.72), header_h + int(height * 0.08))),
                ("dashboard_or_cards", (side_w, header_h, int(width * 0.74), height)),
                ("right_status_actions", (int(width * 0.70), 0, width, height)),
            ],
            "chat_document": [
                ("conversation_sidebar", (0, 0, side_w + int(width * 0.10), height)),
                ("document_thread", (side_w, 0, width, height - bottom_h)),
                ("composer", (side_w, height - bottom_h, width, height)),
            ],
            "security_dashboard": [
                ("security_status", (0, 0, width, int(height * 0.28))),
                ("feature_grid", (0, int(height * 0.20), width, int(height * 0.86))),
                ("bottom_actions", (0, int(height * 0.82), width, height)),
            ],
            "archive_file_manager": [
                ("menu_toolbar", (0, 0, width, int(height * 0.18))),
                ("file_list", (0, int(height * 0.12), width, int(height * 0.92))),
                ("status_bar", (0, int(height * 0.88), width, height)),
            ],
            "file_search": [
                ("search_bar", (0, 0, width, int(height * 0.18))),
                ("result_list", (0, int(height * 0.12), width, int(height * 0.90))),
                ("status_or_filters", (0, int(height * 0.86), width, height)),
            ],
            "local_transfer_dashboard": [
                ("device_status", (0, 0, width, int(height * 0.30))),
                ("send_receive_actions", (0, int(height * 0.22), width, int(height * 0.68))),
                ("transfer_history", (0, int(height * 0.58), width, height)),
            ],
            "account_switcher": [
                ("account_list", (0, int(height * 0.12), int(width * 0.62), int(height * 0.88))),
                ("login_actions", (int(width * 0.42), int(height * 0.18), width, int(height * 0.88))),
                ("window_controls", (int(width * 0.68), 0, width, int(height * 0.18))),
            ],
            "browser_profile_manager": [
                ("left_navigation", (0, 0, side_w, height)),
                ("profile_table_or_cards", (side_w, int(height * 0.14), width, int(height * 0.88))),
                ("toolbar_search_filters", (side_w, 0, width, int(height * 0.22))),
                ("pagination_status", (side_w, int(height * 0.84), width, height)),
            ],
            "remote_access_dashboard": [
                ("connection_panel", (0, 0, int(width * 0.58), height)),
                ("remote_control_actions", (int(width * 0.42), 0, width, int(height * 0.66))),
                ("status_or_recent", (int(width * 0.42), int(height * 0.55), width, height)),
            ],
            "media_video_home": [
                ("top_navigation_search", (0, 0, width, header_h + int(height * 0.06))),
                ("media_feed", (0, header_h, width, int(height * 0.88))),
                ("player_or_bottom_actions", (0, int(height * 0.82), width, height)),
            ],
        }
        return plans.get(
            mode,
            [
                ("top_header", (0, 0, width, header_h)),
                ("main_content", (0, header_h, width, height - bottom_h // 2)),
                ("bottom_status_or_actions", (0, height - bottom_h, width, height)),
            ],
        )

    def _chat_composer_top(
        self,
        *,
        width: int,
        height: int,
        side_w: int,
        default_top: int,
        candidates: list[Any],
    ) -> int:
        toolbar_tops: list[int] = []
        min_left = max(side_w, int(width * 0.30))
        min_top = int(height * 0.66)
        max_bottom = int(height * 0.76)
        for element in candidates:
            bounds = getattr(element, "bounds", None)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                continue
            left, top, right, bottom = [int(v) for v in bounds]
            box_w = right - left
            box_h = bottom - top
            if left < min_left or top < min_top or bottom > max_bottom:
                continue
            if 12 <= box_w <= 120 and 12 <= box_h <= 70:
                toolbar_tops.append(top)
        if len(toolbar_tops) < 3:
            return default_top
        expanded_top = max(header_h := max(56, int(height * 0.10)), min(toolbar_tops) - 8)
        return min(default_top, expanded_top)

    def _canvas_size(self, canvas: Any) -> tuple[int, int]:
        artifacts = getattr(canvas, "artifacts", None) or {}
        size = artifacts.get("screenshot_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
        for source in (getattr(canvas, "window", None),):
            rect = getattr(source, "rect_client", None) if source else None
            if rect:
                return int(rect[2] - rect[0]), int(rect[3] - rect[1])
        return 0, 0

    def _geometric_regions(self, canvas: Any) -> list[dict[str, Any]]:
        artifacts = getattr(canvas, "artifacts", None) or {}
        return list(artifacts.get("geometric_regions") or [])

    def _layout_audit_rois(
        self,
        canvas: Any,
        mode: str,
        width: int,
        height: int,
        candidates: list[Any],
    ) -> list[RoiPlanEntry]:
        if width <= 0 or height <= 0:
            return []
        audit = (getattr(canvas, "artifacts", None) or {}).get("vlm_layout_audit") or {}
        if not audit.get("review_only"):
            return []
        rois: list[RoiPlanEntry] = []
        for region in list(audit.get("layout_regions") or [])[:8]:
            relative = region.get("relative_bounds")
            if not isinstance(relative, list) or len(relative) != 4:
                continue
            try:
                left, top, right, bottom = [float(value) for value in relative]
            except (TypeError, ValueError):
                continue
            bounds = self._clamp(
                (
                    int(round(left * width)),
                    int(round(top * height)),
                    int(round(right * width)),
                    int(round(bottom * height)),
                ),
                width,
                height,
            )
            if not self._large_enough(bounds, width, height):
                continue
            rois.append(
                RoiPlanEntry(
                    roi_id=f"roi_{len(rois)}",
                    bounds=bounds,
                    mode=mode,
                    purpose=str(region.get("role") or "layout_region"),
                    priority=len(rois) + 1,
                    source="vlm_layout_audit",
                    vlm_task="region_annotation",
                    candidate_ids=self._candidate_ids_in_bounds(candidates, bounds),
                )
            )
        return rois

    def _candidate_ids_in_bounds(self, candidates: list[Any], bounds: tuple[int, int, int, int]) -> list[str]:
        ids: list[str] = []
        left, top, right, bottom = bounds
        for candidate in candidates:
            cb = getattr(candidate, "bounds", None)
            if not cb:
                continue
            cx = (int(cb[0]) + int(cb[2])) // 2
            cy = (int(cb[1]) + int(cb[3])) // 2
            if left <= cx <= right and top <= cy <= bottom:
                ids.append(str(getattr(candidate, "element_id", "")))
        return [cid for cid in ids if cid]

    def _clamp(self, bounds: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = bounds
        return (
            max(0, min(width, int(left))),
            max(0, min(height, int(top))),
            max(0, min(width, int(right))),
            max(0, min(height, int(bottom))),
        )

    def _large_enough(self, bounds: tuple[int, int, int, int], width: int, height: int) -> bool:
        if width <= 0 or height <= 0:
            return False
        area = max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
        return area >= width * height * 0.04

    def _overlaps_existing(self, bounds: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]]) -> bool:
        for other in existing:
            inter_left = max(bounds[0], other[0])
            inter_top = max(bounds[1], other[1])
            inter_right = min(bounds[2], other[2])
            inter_bottom = min(bounds[3], other[3])
            inter = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
            area = max(1, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
            if inter / area > 0.80:
                return True
        return False
