"""Tests for the combined ROI/VLM regression runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_roi_vlm_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_roi_vlm_regression", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_combined_summary_fails_on_mode_mismatch_and_low_quality():
    module = _load_module()
    plan_summary = {
        "sample_count": 3,
        "expected_count": 2,
        "mode_match_count": 1,
        "mode_mismatch_count": 1,
        "errors": [],
    }
    quality_summary = {
        "sample_count": 2,
        "average_score": 79.5,
        "low_score_count": 1,
        "generic_region_total": 1,
        "invalid_region_total": 0,
        "invalid_annotation_role_total": 0,
        "late_failure_total": 0,
    }

    summary = module.build_combined_summary(
        plan_summary,
        quality_summary,
        min_quality_score=85,
        max_mode_mismatches=0,
        max_plan_errors=0,
    )

    assert summary["status"] == "fail"
    assert summary["plan"]["mode_mismatch_count"] == 1
    assert summary["quality"]["average_score"] == 79.5
    assert "mode_mismatch_count>0" in summary["failures"]
    assert "average_quality_score<85" in summary["failures"]


def test_write_outputs_persists_json_and_markdown(tmp_path):
    module = _load_module()
    summary = {
        "status": "pass",
        "failures": [],
        "plan": {
            "sample_count": 15,
            "expected_count": 15,
            "mode_match_count": 15,
            "mode_mismatch_count": 0,
            "error_count": 0,
        },
        "quality": {
            "source": "summary.json",
            "sample_count": 6,
            "average_score": 100.0,
            "low_score_count": 0,
            "generic_region_total": 0,
            "invalid_region_total": 0,
            "invalid_annotation_role_total": 0,
            "late_failure_total": 0,
        },
        "thresholds": {
            "min_quality_score": 85,
            "max_mode_mismatches": 0,
            "max_plan_errors": 0,
        },
    }

    module.write_outputs(summary, tmp_path)

    saved = json.loads((tmp_path / "regression_summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "regression_summary.md").read_text(encoding="utf-8")
    assert saved["status"] == "pass"
    assert "ROI/VLM Regression Summary" in report
    assert "Status: pass" in report


def test_run_regression_includes_sample_matrix_status_counts(tmp_path):
    module = _load_module()
    samples_root = tmp_path / "samples"
    samples_root.mkdir()
    (samples_root / "sample_matrix_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 2,
                "semantic_status_counts": {"dry_run": 1, "timeout": 1},
                "mode_counts": {"chat_workspace": 1, "unknown_pattern": 1},
                "error_count": 0,
            }
        ),
        encoding="utf-8",
    )

    summary = module.build_combined_summary(
        {
            "sample_count": 0,
            "expected_count": 0,
            "mode_match_count": 0,
            "mode_mismatch_count": 0,
            "errors": [],
        },
        None,
        min_quality_score=85,
        max_mode_mismatches=0,
        max_plan_errors=0,
        sample_matrix_summary=module.load_sample_matrix_summary(samples_root),
    )

    assert summary["sample_matrix"]["available"] is True
    assert summary["sample_matrix"]["semantic_status_counts"] == {"dry_run": 1, "timeout": 1}
    assert summary["sample_matrix"]["mode_counts"]["unknown_pattern"] == 1
