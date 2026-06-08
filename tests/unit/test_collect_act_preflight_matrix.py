"""Tests for collecting /act preflight matrices from canvas details."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "collect_act_preflight_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_act_preflight_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _detail() -> dict:
    return {
        "canvas_id": "canvas_1",
        "app_id": "qq.exe",
        "window_title": "QQ",
        "page_class": "chat_workspace",
        "elements": [
            {
                "element_id": "input_1",
                "semantic_role": "message_input",
                "role_label": "message_input",
                "attributes": {"safe_to_type": False},
            },
            {
                "element_id": "send_1",
                "semantic_role": "send_button",
                "role_label": "send_button",
                "text": "发送",
                "risk_tags": ["send"],
            },
        ],
    }


def test_collects_type_and_send_preflight_requests_from_canvas_detail():
    module = _load_module()
    calls: list[dict] = []

    def fake_act(payload: dict) -> dict:
        calls.append(payload)
        return {
            "execution_result": "blocked",
            "warnings": ["type_text_requires_safe_to_type"] if payload["action"] == "type_text" else ["send_button_disabled"],
            "action_plan": {"verification_plan": {"readback_plan": {}, "state_probe_plan": {"current": {}}}},
        }

    matrix = module.collect_act_preflight_matrix_from_details(
        details=[_detail()],
        act_client=fake_act,
        probe_text="OpenClaw matrix probe",
    )

    row = matrix["rows"][0]
    assert row["sample"] == "qq.exe_canvas_1"
    assert row["actions"]["type_text"]["execution_result"] == "blocked"
    assert row["actions"]["send"]["execution_result"] == "blocked"
    assert calls == [
        {
            "canvas_id": "canvas_1",
            "candidate_id": "input_1",
            "action": "type_text",
            "params": {"text": "OpenClaw matrix probe"},
            "dry_run": True,
        },
        {
            "canvas_id": "canvas_1",
            "candidate_id": "send_1",
            "action": "send",
            "params": {"expected_text": "OpenClaw matrix probe"},
            "dry_run": True,
        },
    ]


def test_collects_controlled_click_preflight_request_from_canvas_detail():
    module = _load_module()
    detail = _detail()
    detail["elements"].append(
        {
            "element_id": "button_1",
            "semantic_role": "button",
            "role_label": "button",
            "text": "设置",
            "attributes": {"actionability": "controlled_probe"},
        }
    )
    calls: list[dict] = []

    matrix = module.collect_act_preflight_matrix_from_details(
        details=[detail],
        act_client=lambda payload: calls.append(payload) or {"execution_result": "planned"},
    )

    row = matrix["rows"][0]
    assert row["actions"]["click"]["execution_result"] == "planned"
    assert row["candidate_ids"]["click"] == "button_1"
    assert calls[-1] == {
        "canvas_id": "canvas_1",
        "candidate_id": "button_1",
        "action": "click",
        "params": {},
        "dry_run": True,
    }


def test_collects_scroll_preflight_request_from_scroll_context():
    module = _load_module()
    detail = _detail()
    detail["regions"] = [
        {
            "region_id": "messages",
            "role": "message_stream",
            "scroll_context_id": "scroll_messages",
        }
    ]
    detail["scroll_contexts"] = [
        {
            "scroll_context_id": "scroll_messages",
            "region_id": "messages",
            "viewport_height": 480,
        }
    ]
    detail["elements"].append(
        {
            "element_id": "message_row_1",
            "semantic_role": "text",
            "role_label": "text",
            "region_id": "messages",
            "text": "消息测试456",
        }
    )
    calls: list[dict] = []

    matrix = module.collect_act_preflight_matrix_from_details(
        details=[detail],
        act_client=lambda payload: calls.append(payload) or {"execution_result": "planned"},
    )

    row = matrix["rows"][0]
    assert row["actions"]["scroll"]["execution_result"] == "planned"
    assert row["candidate_ids"]["scroll"] == "message_row_1"
    assert calls[-1] == {
        "canvas_id": "canvas_1",
        "candidate_id": "message_row_1",
        "action": "scroll",
        "params": {"direction": "down", "amount": "page"},
        "dry_run": True,
    }


def test_records_missing_candidates_without_act_calls():
    module = _load_module()
    calls: list[dict] = []
    matrix = module.collect_act_preflight_matrix_from_details(
        details=[{"canvas_id": "canvas_2", "app_id": "notepad.exe", "elements": []}],
        act_client=lambda payload: calls.append(payload) or {},
    )

    row = matrix["rows"][0]
    assert row["status"] == "missing_candidates"
    assert row["actions"] == {}
    assert row["warnings"] == ["missing_message_input_candidate", "missing_send_candidate"]
    assert calls == []


def test_uses_sample_matrix_send_target_when_detail_role_is_missing():
    module = _load_module()
    detail = _detail()
    detail["elements"][1]["semantic_role"] = "unknown"
    detail["elements"][1]["role_label"] = None
    detail["elements"][1]["risk_tags"] = []
    detail["elements"][1]["text"] = ""
    calls: list[dict] = []

    matrix = module.collect_act_preflight_matrix_from_details(
        details=[detail],
        sample_rows=[
            {
                "sample": "qq_live",
                "process_name": "qq.exe",
                "canvas_id": "canvas_1",
                "composer_send_target_candidate_id": "send_1",
            }
        ],
        act_client=lambda payload: calls.append(payload) or {"execution_result": "blocked"},
    )

    row = matrix["rows"][0]
    assert row["sample"] == "qq_live"
    assert row["process_name"] == "qq.exe"
    assert row["candidate_ids"]["send"] == "send_1"
    assert calls[-1]["candidate_id"] == "send_1"


def test_sample_matrix_filters_out_excluded_detail_files(tmp_path: Path):
    module = _load_module()
    detail_dir = tmp_path / "details"
    detail_dir.mkdir()
    (detail_dir / "qq.detail.json").write_text(json.dumps(_detail()), encoding="utf-8")
    excluded = _detail()
    excluded["canvas_id"] = "canvas_excluded"
    excluded["app_id"] = "qq.exe"
    (detail_dir / "qq_tiny.detail.json").write_text(json.dumps(excluded), encoding="utf-8")
    sample_matrix = detail_dir / "sample_matrix_summary.json"
    sample_matrix.write_text(
        json.dumps(
            {
                "rows": [{"sample": "qq_live", "process_name": "qq.exe", "canvas_id": "canvas_1"}],
                "excluded_rows": [{"sample": "qq_tiny", "canvas_id": "canvas_excluded"}],
            }
        ),
        encoding="utf-8",
    )

    matrix = module.collect_act_preflight_matrix_dirs(
        detail_dir=detail_dir,
        sample_matrix=sample_matrix,
        act_client=lambda payload: {"execution_result": "blocked", "warnings": [payload["action"]]},
    )

    assert [row["canvas_id"] for row in matrix["rows"]] == ["canvas_1"]


def test_writes_matrix_from_detail_directory(tmp_path: Path):
    module = _load_module()
    input_dir = tmp_path / "details"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "qq_canvas.detail.json").write_text(json.dumps(_detail()), encoding="utf-8")

    matrix = module.collect_act_preflight_matrix_dirs(
        detail_dir=input_dir,
        output_dir=output_dir,
        act_client=lambda payload: {"execution_result": "blocked", "warnings": [payload["action"]]},
    )

    assert matrix["rows"][0]["sample"] == "qq.exe_canvas_1"
    assert (output_dir / "act_preflight_matrix.json").exists()
    assert (output_dir / "act_preflight_matrix.md").exists()


def test_detail_loader_ignores_summary_reports(tmp_path: Path):
    module = _load_module()
    detail_dir = tmp_path / "details"
    detail_dir.mkdir()
    (detail_dir / "qq.detail.json").write_text(json.dumps(_detail()), encoding="utf-8")
    (detail_dir / "sample_matrix_summary.json").write_text(json.dumps({"rows": [{"sample": "summary"}]}), encoding="utf-8")
    (detail_dir / "recognition_acceptance.json").write_text(json.dumps({"rows": [{"sample": "acceptance"}]}), encoding="utf-8")

    matrix = module.collect_act_preflight_matrix_dirs(
        detail_dir=detail_dir,
        act_client=lambda payload: {"execution_result": "blocked", "warnings": [payload["action"]]},
    )

    assert [row["canvas_id"] for row in matrix["rows"]] == ["canvas_1"]
