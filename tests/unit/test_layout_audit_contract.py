"""Tests for VLM layout-audit contracts."""

from __future__ import annotations

from src.vlm.layout_audit_contract import (
    build_layout_audit_job,
    merge_layout_audit_response,
    validate_layout_audit_response,
)


class Canvas:
    canvas_id = "canvas_layout_1"
    elements = []

    def __init__(self, warnings=None):
        self.artifacts = {
            "screenshot_size": [1000, 700],
            "visual_pattern": {"mode": "unknown_pattern"},
            "perception_quality": {
                "usable_state": "unreliable",
                "warnings": warnings or ["sparse_elements", "coarse_regions", "layout_audit_needed"],
            },
            "geometric_regions": [],
        }


def test_build_layout_audit_job_only_when_quality_requires_it():
    job = build_layout_audit_job(Canvas())

    assert job is not None
    assert job["canvas_id"] == "canvas_layout_1"
    assert job["mode_guess"] == "unknown_pattern"
    assert job["screenshot_size"] == [1000, 700]
    assert job["allowed_outputs"] == ["layout_regions", "layout_summary", "review_only_hints"]
    assert build_layout_audit_job(Canvas(warnings=["sparse_elements"])) is None


def test_validate_layout_audit_response_rejects_bad_bounds_and_excess_regions():
    canvas = Canvas()
    invalid = validate_layout_audit_response(
        canvas,
        {
            "layout_regions": [
                {"role": "main_content", "relative_bounds": [0, 0, 1.2, 1], "confidence": 0.9}
                for _ in range(9)
            ]
        },
    )

    assert invalid.accepted is False
    assert "too_many_layout_regions" in invalid.errors
    assert "invalid_relative_bounds:0" in invalid.errors


def test_merge_layout_audit_response_stores_review_only_regions():
    canvas = Canvas()

    validation = merge_layout_audit_response(
        canvas,
        {
            "layout_summary": "single app surface with no local segmentation",
            "layout_regions": [
                {
                    "region_id": "main",
                    "role": "main_content",
                    "relative_bounds": [0.0, 0.0, 1.0, 1.0],
                    "confidence": 0.74,
                    "summary": "whole visible app surface",
                    "click_point": [0.5, 0.5],
                }
            ],
            "review_only_hints": ["use OCR/Omni later for controls"],
        },
        status="success",
    )

    assert validation.accepted is True
    saved = canvas.artifacts["vlm_layout_audit"]
    assert saved["status"] == "success"
    assert saved["review_only"] is True
    assert saved["layout_regions"][0]["relative_bounds"] == [0.0, 0.0, 1.0, 1.0]
    assert "click_point" not in saved["layout_regions"][0]
    assert "use OCR/Omni later for controls" in saved["review_only_hints"]
