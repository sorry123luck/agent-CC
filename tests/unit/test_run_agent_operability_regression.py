"""Tests for the agent operability regression runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_agent_operability_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_agent_operability_regression", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _complete_act_response(action: str) -> dict:
    return {
        "execution_result": "blocked",
        "warnings": ["type_text_requires_safe_to_type"] if action == "type_text" else ["send_button_disabled"],
        "action_plan": {
            "execution_policy": {
                "executes_desktop_input": False,
                "confirmed_execution_would_touch_desktop": False,
                "requires_execute_confirmed": False,
                "enabled_for_execution": False,
            },
            "verification_plan": {
                "readback_plan": {
                    "expected_text": "OpenClaw matrix probe",
                    "request": {"region_role": "message_stream"},
                },
                "state_probe_plan": {"current": {"input_state": "empty", "send_enabled": False}},
            },
        },
    }


def test_runner_writes_page_act_closure_and_summary(tmp_path: Path):
    module = _load_module()
    matrix_dir = tmp_path / "matrix"
    detail_dir = tmp_path / "details"
    out_dir = tmp_path / "out"
    detail_dir.mkdir()
    _write_json(
        matrix_dir / "sample_matrix_summary.json",
        {
            "sample_matrix_schema_version": "2026-05-28.current",
            "sample_count": 1,
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "roi_purposes": ["message_stream", "composer"],
                    "composer_input_review_count": 1,
                    "composer_input_primary_click_point": [486, 576],
                    "composer_send_target_bounds": [665, 599, 760, 627],
                    "composer_vlm_send_hint_count": 1,
                }
            ],
        },
    )
    _write_json(
        matrix_dir / "recognition_acceptance.json",
        {
            "overall_status": "pass",
            "counts": {"pass": 1},
            "rows": [{"sample": "qq", "status": "pass", "failures": [], "warnings": []}],
        },
    )
    _write_json(
        detail_dir / "qq.detail.json",
        {
            "canvas_id": "canvas_1",
            "app_id": "qq.exe",
            "elements": [
                {
                    "element_id": "input_1",
                    "semantic_role": "message_input",
                    "attributes": {"safe_to_type": False},
                },
                {"element_id": "send_1", "semantic_role": "send_button", "risk_tags": ["send"]},
            ],
        },
    )

    summary = module.run_agent_operability_regression(
        matrix_dir=matrix_dir,
        detail_dir=detail_dir,
        output_dir=out_dir,
        act_client=lambda payload: _complete_act_response(payload["action"]),
    )

    assert summary["overall_status"] == "closed_for_recognition_chain"
    assert summary["page_operability_status"] == "usable"
    assert summary["act_preflight_status"] == "usable_with_blocked_actions"
    assert summary["input_safety_status"] == "probe_ready_review_only"
    assert (out_dir / "page_operability_report.json").exists()
    assert (out_dir / "act_preflight" / "analysis" / "act_preflight_matrix_report.json").exists()
    assert (out_dir / "act_preflight_matrix_report.json").exists()
    assert (out_dir / "input_safety_readiness_report.json").exists()
    assert (out_dir / "sample_coverage_report.json").exists()
    assert (out_dir / "sample_collection_plan.json").exists()
    assert summary["outputs"]["sample_collection_plan"] == "sample_collection_plan.json"
    assert (out_dir / "goal_progress_report.json").exists()
    assert (out_dir / "recognition_closure_report.json").exists()
    assert (out_dir / "agent_operability_regression_summary.md").exists()


def test_runner_includes_transition_readiness_when_search_probe_is_supplied(tmp_path: Path):
    module = _load_module()
    matrix_dir = tmp_path / "matrix"
    detail_dir = tmp_path / "details"
    search_probe_dir = tmp_path / "search_probe"
    out_dir = tmp_path / "out"
    detail_dir.mkdir()
    _write_json(
        matrix_dir / "sample_matrix_summary.json",
        {
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "roi_purposes": ["message_stream", "composer"],
                    "composer_input_review_count": 1,
                    "composer_input_primary_click_point": [486, 576],
                }
            ],
        },
    )
    _write_json(matrix_dir / "recognition_acceptance.json", {"rows": [{"sample": "qq", "status": "pass"}]})
    _write_json(
        detail_dir / "qq.detail.json",
        {
            "canvas_id": "canvas_1",
            "app_id": "qq.exe",
            "elements": [{"element_id": "input_1", "semantic_role": "message_input", "attributes": {"safe_to_type": False}}],
        },
    )
    _write_json(
        search_probe_dir / "search_probe_report.json",
        {
            "execute": True,
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "captured",
                    "state_transition": {
                        "path": "chat_workspace -> chat_search_results -> chat_workspace",
                        "selected_returned_to_before_state": True,
                    },
                }
            ],
        },
    )

    summary = module.run_agent_operability_regression(
        matrix_dir=matrix_dir,
        detail_dir=detail_dir,
        output_dir=out_dir,
        search_probe_dir=search_probe_dir,
        act_client=lambda payload: _complete_act_response(payload["action"]),
    )

    assert summary["transition_readiness_status"] == "observed_transition_memory_ready"
    assert summary["outputs"]["transition_readiness"] == "transition_readiness_report.json"
    assert (out_dir / "transition_readiness_report.json").exists()


def test_runner_includes_vlm_supplement_readiness_when_quality_dir_is_supplied(tmp_path: Path):
    module = _load_module()
    matrix_dir = tmp_path / "matrix"
    detail_dir = tmp_path / "details"
    vlm_quality_dir = tmp_path / "vlm_quality"
    out_dir = tmp_path / "out"
    detail_dir.mkdir()
    _write_json(
        matrix_dir / "sample_matrix_summary.json",
        {
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "roi_purposes": ["message_stream", "composer"],
                    "composer_input_review_count": 1,
                    "composer_input_primary_click_point": [486, 576],
                }
            ],
        },
    )
    _write_json(matrix_dir / "recognition_acceptance.json", {"rows": [{"sample": "qq", "status": "pass"}]})
    _write_json(
        detail_dir / "qq.detail.json",
        {
            "canvas_id": "canvas_1",
            "app_id": "qq.exe",
            "elements": [{"element_id": "input_1", "semantic_role": "message_input", "attributes": {"safe_to_type": False}}],
        },
    )
    _write_json(
        vlm_quality_dir / "quality_summary.json",
        {
            "sample_count": 1,
            "average_score": 100,
            "rows": [{"sample": "qq", "score": 100, "issues": []}],
        },
    )

    summary = module.run_agent_operability_regression(
        matrix_dir=matrix_dir,
        detail_dir=detail_dir,
        output_dir=out_dir,
        vlm_quality_dir=vlm_quality_dir,
        act_client=lambda payload: _complete_act_response(payload["action"]),
    )

    assert summary["vlm_supplement_status"] == "semantic_supplement_ready"
    assert summary["outputs"]["vlm_supplement_readiness"] == "vlm_supplement_readiness_report.json"
    assert (out_dir / "vlm_supplement_readiness_report.json").exists()
