"""Tests for the /act preflight regression runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_act_preflight_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_act_preflight_regression", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_detail(path: Path) -> None:
    path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )


def _complete_response(action: str) -> dict:
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
                    "expected_text": "DeskCanvas matrix probe",
                    "request": {"region_role": "message_stream"},
                },
                "state_probe_plan": {"current": {"input_state": "empty", "send_enabled": False}},
            }
        },
    }


def test_runner_writes_collected_matrix_and_analysis(tmp_path: Path):
    module = _load_module()
    detail_dir = tmp_path / "details"
    out_dir = tmp_path / "out"
    detail_dir.mkdir()
    _write_detail(detail_dir / "qq.detail.json")

    report = module.run_act_preflight_regression(
        detail_dir=detail_dir,
        output_dir=out_dir,
        act_client=lambda payload: _complete_response(payload["action"]),
    )

    assert report["overall_status"] == "usable_with_blocked_actions"
    assert (out_dir / "act_preflight_matrix.json").exists()
    assert (out_dir / "analysis" / "act_preflight_matrix_report.json").exists()
    assert (out_dir / "act_preflight_regression_summary.md").exists()


def test_safe_act_client_records_backend_failures():
    module = _load_module()

    def failing_post(payload: dict) -> dict:
        raise RuntimeError("backend timeout")

    client = module.safe_act_client(failing_post)
    response = client({"action": "type_text"})

    assert response["execution_result"] == "blocked"
    assert "act_preflight_request_failed" in response["warnings"]
    assert "backend timeout" in response["error_message"]
