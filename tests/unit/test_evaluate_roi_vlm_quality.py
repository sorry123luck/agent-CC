"""Tests for ROI VLM semantic quality evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_roi_vlm_quality.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_roi_vlm_quality", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_row_penalizes_generic_wrong_roles_and_failures():
    module = _load_module()
    row = {
        "sample": "Everything",
        "mode": "file_search",
        "roi_purposes": ["search_bar", "result_list", "status_or_filters"],
        "interactive_statuses": ["timeout", "timeout", "timeout"],
        "new_supplements": [
            {
                "roi_id": "roi_0",
                "region_semantics": {"role": "chat_area"},
                "candidate_annotations": [
                    {"candidate_id": "e1", "role": "message", "label": "message_bubble"},
                    {"candidate_id": "e2", "role": "action", "label": "send_button"},
                ],
            }
        ],
        "new_timeouts": [{"roi_id": "roi_0"}, {"roi_id": "roi_1"}],
        "new_late_failures": [{"roi_id": "roi_2"}],
    }

    result = module.score_row(row)

    assert result["sample"] == "Everything"
    assert result["mode"] == "file_search"
    assert result["annotation_count"] == 2
    assert result["generic_region_count"] == 1
    assert result["invalid_region_count"] == 1
    assert result["invalid_annotation_role_count"] == 1
    assert result["late_failure_count"] == 1
    assert result["score"] < 70
    assert "generic_region_role:chat_area" in result["issues"]
    assert "invalid_annotation_role:message" in result["issues"]


def test_score_row_accepts_mode_scoped_roles():
    module = _load_module()
    row = {
        "sample": "Everything",
        "mode": "file_search",
        "roi_purposes": ["search_bar", "result_list", "status_or_filters"],
        "interactive_statuses": ["timeout", "timeout", "timeout"],
        "new_supplements": [
            {
                "roi_id": "roi_0",
                "region_semantics": {"role": "search_bar"},
                "candidate_annotations": [
                    {"candidate_id": "e1", "role": "input", "label": "file search input"},
                    {"candidate_id": "e2", "role": "result", "label": "result row"},
                    {"candidate_id": "e3", "role": "pagination", "label": "next page"},
                ],
            }
        ],
        "new_timeouts": [{"roi_id": "roi_0"}],
        "new_late_failures": [],
    }

    result = module.score_row(row)

    assert result["annotation_count"] == 3
    assert result["generic_region_count"] == 0
    assert result["invalid_region_count"] == 0
    assert result["invalid_annotation_role_count"] == 0
    assert result["score"] >= 85


def test_score_row_accepts_video_card_candidate_role():
    module = _load_module()
    row = {
        "sample": "iQIYI",
        "mode": "media_video_home",
        "new_supplements": [
            {
                "roi_id": "roi_0",
                "region_semantics": {"role": "media_feed"},
                "candidate_annotations": [
                    {"candidate_id": "v1", "role": "video_card", "label": "poster card"},
                ],
            }
        ],
    }

    result = module.score_row(row)

    assert result["invalid_annotation_role_count"] == 0
    assert result["score"] >= 90
