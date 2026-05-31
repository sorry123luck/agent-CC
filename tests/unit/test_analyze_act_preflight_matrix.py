"""Tests for act preflight matrix reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_act_preflight_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_act_preflight_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _act_response(
    *,
    action: str,
    execution_result: str = "planned",
    warnings: list[str] | None = None,
    input_state: str = "empty",
    send_enabled: bool | None = False,
    expected_text: str = "DeskCanvas probe",
) -> dict:
    return {
        "execution_result": execution_result,
        "warnings": warnings or ["act_execution_adapter_pending"],
        "action_plan": {
            "action": action,
            "action_level": "controlled" if execution_result == "planned" else "blocked",
            "execution_policy": {
                "executes_desktop_input": False,
                "confirmed_execution_would_touch_desktop": False,
                "requires_execute_confirmed": False,
                "enabled_for_execution": False,
            },
            "verification_plan": {
                "readback_after": True,
                "readback_plan": {
                    "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
                    "request": {"region_role": "message_stream", "include_ocr": True},
                    "expected_text": expected_text,
                    "source": "params.text",
                },
                "state_probe_plan": {
                    "current": {"input_state": input_state, "send_enabled": send_enabled},
                    "before_requirements": (
                        [{"field": "send_enabled", "expected": True}] if action == "send" else []
                    ),
                    "after_expectations": (
                        [
                            {"field": "input_state", "expected": "filled", "source": "params.text"},
                            {"field": "send_enabled", "expected": True, "condition": "send_control_present"},
                        ]
                        if action == "type_text"
                        else [
                            {
                                "field": "message_stream_contains",
                                "expected": expected_text,
                                "source": "params.expected_text",
                            }
                        ]
                    ),
                    "failure_policy": "keep_action_blocked_until_probe_passes",
                },
            },
        },
    }


def _controlled_response(
    *,
    action: str,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "execution_result": "planned",
        "warnings": warnings or ["act_execution_adapter_pending"],
        "action_plan": {
            "action": action,
            "action_level": "controlled",
            "execution_policy": {
                "executes_desktop_input": False,
                "confirmed_execution_would_touch_desktop": True,
                "requires_execute_confirmed": True,
                "enabled_for_execution": True,
            },
            "verification_plan": {
                "observe_after": True,
                "diff_after": True,
                "readback_after": True,
                "readback_plan": {
                    "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
                    "request": {"region_role": "content_area", "include_ocr": True},
                    "after_observe": True,
                    "failure_policy": "downgrade_verification_to_review",
                },
            },
        },
    }


def test_report_accepts_controlled_click_when_readback_protocol_is_complete():
    module = _load_module()

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "flclash",
                    "process_name": "flclash.exe",
                    "actions": {"click": _controlled_response(action="click")},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "usable_with_warnings"
    assert row["status"] == "usable_with_warnings"
    assert row["click"]["protocol_complete"] is True
    assert row["click"]["has_readback_plan"] is True
    assert row["click"]["safe_boundary"] == "controlled_requires_confirmation"


def test_report_marks_controlled_scroll_without_readback_plan_as_fail():
    module = _load_module()
    bad = _controlled_response(action="scroll")
    del bad["action_plan"]["verification_plan"]["readback_plan"]

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "voicemeeter",
                    "process_name": "voicemeeter.exe",
                    "actions": {"scroll": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "scroll_missing_readback_plan" in row["failures"]


def test_report_marks_controlled_action_without_execution_policy_as_fail():
    module = _load_module()
    bad = _controlled_response(action="click")
    del bad["action_plan"]["execution_policy"]

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "flclash",
                    "process_name": "flclash.exe",
                    "actions": {"click": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "click_missing_execution_policy" in row["failures"]


def test_report_marks_controlled_action_with_unsafe_execution_policy_as_fail():
    module = _load_module()
    bad = _controlled_response(action="scroll")
    bad["action_plan"]["execution_policy"]["requires_execute_confirmed"] = False
    bad["action_plan"]["execution_policy"]["confirmed_execution_would_touch_desktop"] = False

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "flclash",
                    "process_name": "flclash.exe",
                    "actions": {"scroll": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "scroll_execution_policy_missing_confirmation_requirement" in row["failures"]
    assert "scroll_execution_policy_missing_confirmed_desktop_touch" in row["failures"]


def test_report_marks_type_text_execution_enabled_policy_as_fail():
    module = _load_module()
    bad = _act_response(
        action="type_text",
        execution_result="blocked",
        warnings=["type_text_requires_safe_to_type"],
    )
    bad["action_plan"]["execution_policy"]["enabled_for_execution"] = True

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "actions": {"type_text": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "type_text_execution_policy_must_not_enable_execution" in row["failures"]


def test_report_accepts_review_only_type_text_when_protocol_is_complete():
    module = _load_module()

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "actions": {
                        "type_text": _act_response(
                            action="type_text",
                            execution_result="blocked",
                            warnings=["type_text_requires_safe_to_type"],
                        ),
                        "send": _act_response(
                            action="send",
                            execution_result="blocked",
                            warnings=["send_button_disabled"],
                        ),
                    },
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "usable_with_blocked_actions"
    assert row["status"] == "usable_with_blocked_actions"
    assert row["type_text"]["protocol_complete"] is True
    assert row["type_text"]["safe_boundary"] == "review_only"
    assert row["send"]["protocol_complete"] is True
    assert row["send"]["safe_boundary"] == "disabled_send_blocked"


def test_report_marks_missing_state_probe_as_fail():
    module = _load_module()
    bad = _act_response(action="type_text")
    del bad["action_plan"]["verification_plan"]["state_probe_plan"]

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "actions": {"type_text": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "type_text_missing_state_probe_plan" in row["failures"]


def test_report_marks_send_without_expected_text_as_fail():
    module = _load_module()
    bad = _act_response(action="send", expected_text="")

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "qq_group",
                    "process_name": "qq.exe",
                    "actions": {"send": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "send_missing_expected_text" in row["failures"]


def test_report_marks_send_candidate_rejected_by_act_as_fail():
    module = _load_module()
    bad = _act_response(action="send", warnings=["send_requires_send_candidate"])

    report = module.build_act_preflight_matrix_report(
        {
            "rows": [
                {
                    "sample": "feishu",
                    "process_name": "feishu.exe",
                    "actions": {"send": bad},
                }
            ]
        }
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert "send_candidate_not_accepted_by_act" in row["failures"]


def test_report_marks_rows_without_action_responses_as_fail():
    module = _load_module()

    report = module.build_act_preflight_matrix_report(
        {"rows": [{"sample": "tiny_qq", "process_name": "qq.exe", "actions": {}}]}
    )

    row = report["rows"][0]
    assert report["overall_status"] == "unusable"
    assert row["status"] == "unusable"
    assert row["failures"] == ["missing_action_preflight_responses"]


def test_report_writes_json_and_markdown(tmp_path: Path):
    module = _load_module()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "act_preflight_matrix.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample": "feishu",
                        "process_name": "feishu.exe",
                        "actions": {"type_text": _act_response(action="type_text")},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_act_preflight_matrix_dirs(input_dir=input_dir, output_dir=output_dir)

    assert report["rows"][0]["sample"] == "feishu"
    assert (output_dir / "act_preflight_matrix_report.json").exists()
    assert (output_dir / "act_preflight_matrix_report.md").exists()
