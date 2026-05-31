"""Tests for VLM semantic supplement readiness reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_vlm_supplement_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_vlm_supplement_readiness", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_good_quality_summary_is_semantic_supplement_ready():
    module = _load_module()

    report = module.build_vlm_supplement_readiness_report(
        quality_summary={
            "sample_count": 2,
            "average_score": 95,
            "low_score_count": 0,
            "invalid_region_total": 0,
            "invalid_annotation_role_total": 0,
            "late_failure_total": 0,
            "rows": [
                {"sample": "wechat", "score": 96, "issues": []},
                {"sample": "qq", "score": 94, "issues": []},
            ],
        }
    )

    assert report["overall_status"] == "semantic_supplement_ready"
    assert report["policy"]["allowed_default"] == "semantic_supplement_only"
    assert report["policy"]["blocked_default"] == ["coordinate_override", "safe_to_type_upgrade", "direct_action_projection"]
    assert report["rows"][0]["status"] == "ready"
    assert report["rows"][0]["projection_allowed"] is False


def test_low_score_and_invalid_roles_require_review():
    module = _load_module()

    report = module.build_vlm_supplement_readiness_report(
        quality_summary={
            "sample_count": 1,
            "average_score": 72,
            "low_score_count": 1,
            "invalid_region_total": 1,
            "invalid_annotation_role_total": 2,
            "late_failure_total": 1,
            "rows": [
                {
                    "sample": "voicemeeter",
                    "score": 72,
                    "issues": ["invalid_region_role:toolbar", "invalid_annotation_role:send"],
                }
            ],
        }
    )

    assert report["overall_status"] == "needs_review"
    row = report["rows"][0]
    assert row["status"] == "needs_review"
    assert row["projection_allowed"] is False
    assert "low_score" in row["blockers"]
    assert "invalid_region_or_role" in row["blockers"]
    assert "late_failure" in report["global_blockers"]


def test_empty_quality_summary_is_missing_evidence():
    module = _load_module()

    report = module.build_vlm_supplement_readiness_report(quality_summary={})

    assert report["overall_status"] == "missing_evidence"
    assert report["global_blockers"] == ["missing_quality_summary"]


def test_analyze_vlm_supplement_readiness_dirs_writes_outputs(tmp_path):
    module = _load_module()
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    (quality_dir / "quality_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 1,
                "average_score": 100,
                "rows": [{"sample": "wechat", "score": 100, "issues": []}],
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_vlm_supplement_readiness_dirs(quality_dir=quality_dir, output_dir=tmp_path / "out")

    assert report["overall_status"] == "semantic_supplement_ready"
    assert (tmp_path / "out" / "vlm_supplement_readiness_report.json").exists()
    markdown = (tmp_path / "out" / "vlm_supplement_readiness_report.md").read_text(encoding="utf-8")
    assert "VLM Supplement Readiness Report" in markdown
    assert "semantic_supplement_only" in markdown
