"""Tests for objective-level progress reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_goal_progress.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_goal_progress", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_goal_progress_maps_current_reports_to_six_objectives():
    module = _load_module()

    report = module.build_goal_progress_report(
        agent_operability={
            "overall_status": "closed_for_recognition_chain",
            "act_preflight_status": "usable_with_blocked_actions",
            "input_safety_status": "probe_ready_review_only",
            "transition_readiness_status": "observed_transition_memory_ready",
            "vlm_supplement_status": "semantic_supplement_ready",
            "sample_summary": {"sample_count": 3},
        },
        input_safety={
            "overall_status": "probe_ready_review_only",
            "rows": [
                {
                    "sample": "wechat",
                    "safe_type_upgrade_gate": {
                        "decision": "blocked",
                        "missing": ["controlled_probe_passed", "state_template_input_transition_verified"],
                    },
                }
            ],
        },
        sample_coverage={
            "overall_status": "coverage_incomplete",
            "counts": {"covered": 3, "missing": 5, "invalid": 0},
            "requirements": [
                {"requirement_id": "wechat_chat", "status": "covered"},
                {"requirement_id": "flclash_dashboard", "status": "missing", "reason": "only_minimized_windows"},
            ],
        },
        artifacts_retention={"summary": {"archive_candidate": 20}},
    )

    assert report["overall_status"] == "in_progress"
    assert [item["goal_id"] for item in report["goals"]] == [
        "goal_1_agent_execution",
        "goal_2_input_safety",
        "goal_3_multi_page_exploration",
        "goal_4_vlm_semantic_supplement",
        "goal_5_real_sample_regression",
        "goal_6_model_data_maintenance",
    ]
    goal_1 = report["goals"][0]
    assert goal_1["status"] == "usable_with_blocked_actions"
    assert "type_text/send execution still blocked by design" in goal_1["remaining_work"]
    goal_2 = report["goals"][1]
    assert goal_2["status"] == "probe_ready_review_only"
    assert "controlled_probe_passed" in goal_2["remaining_work"]
    assert "state_template_input_transition_verified" in goal_2["remaining_work"]
    goal_4 = report["goals"][3]
    assert goal_4["status"] == "semantic_supplement_ready"
    assert goal_4["boundary"] == "semantic supplement only; no coordinate/action projection"
    goal_5 = report["goals"][4]
    assert goal_5["status"] == "coverage_incomplete"
    assert "missing:flclash_dashboard" in goal_5["remaining_work"]


def test_goal_progress_marks_missing_evidence_explicitly():
    module = _load_module()

    report = module.build_goal_progress_report(agent_operability={})

    assert report["overall_status"] == "in_progress"
    assert report["goals"][0]["status"] == "missing_evidence"
    assert report["goals"][4]["status"] == "missing_evidence"
    assert "run_agent_operability_regression" in report["goals"][4]["next_step"]


def test_analyze_goal_progress_dirs_writes_outputs(tmp_path: Path):
    module = _load_module()
    input_dir = tmp_path / "regression"
    input_dir.mkdir()
    (input_dir / "agent_operability_regression_summary.json").write_text(
        json.dumps(
            {
                "act_preflight_status": "usable_with_blocked_actions",
                "input_safety_status": "probe_ready_review_only",
                "transition_readiness_status": "observed_transition_memory_ready",
                "vlm_supplement_status": "semantic_supplement_ready",
                "sample_summary": {"sample_count": 3},
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "input_safety_readiness_report.json").write_text(
        json.dumps({"overall_status": "probe_ready_review_only", "rows": []}),
        encoding="utf-8",
    )

    report = module.analyze_goal_progress_dirs(input_dir=input_dir, output_dir=tmp_path / "out")

    assert report["overall_status"] == "in_progress"
    assert (tmp_path / "out" / "goal_progress_report.json").exists()
    markdown = (tmp_path / "out" / "goal_progress_report.md").read_text(encoding="utf-8")
    assert "Goal Progress Report" in markdown
    assert "goal_1_agent_execution" in markdown
