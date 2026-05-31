"""Tests for ROI selection heuristics."""

from __future__ import annotations

from src.perception.roi_selection import RoiSelectionPlanner


class Element:
    def __init__(self, element_id: str, bounds: tuple[int, int, int, int], control_type: str = "icon") -> None:
        self.element_id = element_id
        self.bounds = bounds
        self.control_type = control_type
        self.semantic_role = "unknown"


class Canvas:
    def __init__(self, elements: list[Element]) -> None:
        self.elements = elements
        self.page_class = "qq/main"
        self.artifacts = {
            "screenshot_size": [960, 640],
            "visual_pattern": {"mode": "chat_workspace"},
        }
        self.window = None
        self.app = type("App", (), {"process_name": "qq.exe"})()


def test_chat_workspace_composer_roi_expands_to_include_detached_toolbar_row():
    canvas = Canvas(
        [
            Element("emoji", (323, 433, 363, 475)),
            Element("scissors", (364, 434, 409, 475)),
            Element("folder", (408, 435, 452, 474)),
            Element("send", (844, 598, 942, 627)),
        ]
    )

    plan = RoiSelectionPlanner().plan(canvas)
    rois = {roi.purpose: roi for roi in plan.rois}

    assert rois["composer"].bounds[1] <= 433
    assert "emoji" in rois["composer"].candidate_ids
    assert "send" in rois["composer"].candidate_ids
    assert rois["message_stream"].bounds[3] == rois["composer"].bounds[1]
