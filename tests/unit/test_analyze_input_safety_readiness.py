"""Tests for input safety readiness reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_input_safety_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_input_safety_readiness", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_readiness_marks_review_only_input_probe_ready_when_contracts_complete():
    module = _load_module()
    report = module.build_input_safety_readiness_report(
        page_operability={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "agent_contract": {
                        "input_safety": {
                            "status": "review_only",
                            "safe_to_type": False,
                            "next_probe_plan": [{"probe": "type_text_preflight"}, {"probe": "send_preflight"}],
                        }
                    },
                }
            ]
        },
        act_preflight_matrix={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "usable_with_blocked_actions",
                    "type_text": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "review_only",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                        "input_state": "empty",
                    },
                    "send": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "disabled_send_blocked",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                        "send_enabled": False,
                        "expected_text": "OpenClaw matrix probe",
                    },
                    "failures": [],
                }
            ]
        },
    )

    assert report["overall_status"] == "probe_ready_review_only"
    row = report["rows"][0]
    assert row["status"] == "probe_ready_review_only"
    assert row["safe_to_type"] is False
    assert row["can_upgrade_safe_to_type"] is False
    assert row["probe_readiness"] == {
        "type_text_preflight": "complete",
        "send_preflight": "complete",
    }
    assert row["remaining_blockers"] == ["safe_to_type_false"]


def test_readiness_marks_missing_send_state_probe_as_incomplete():
    module = _load_module()
    report = module.build_input_safety_readiness_report(
        page_operability={
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "agent_contract": {"input_safety": {"status": "review_only", "safe_to_type": False}},
                }
            ]
        },
        act_preflight_matrix={
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "status": "unusable",
                    "type_text": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "review_only",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                    },
                    "send": {
                        "present": True,
                        "protocol_complete": False,
                        "safe_boundary": "blocked_or_review",
                        "has_state_probe_plan": False,
                        "has_readback_plan": True,
                    },
                    "failures": ["send_missing_state_probe_plan"],
                }
            ]
        },
    )

    assert report["overall_status"] == "incomplete"
    row = report["rows"][0]
    assert row["status"] == "incomplete"
    assert row["probe_readiness"]["send_preflight"] == "missing_state_probe"
    assert "send_missing_state_probe_plan" in row["remaining_blockers"]


def test_readiness_records_controlled_send_probe_pass_without_upgrading_safe_type():
    module = _load_module()
    report = module.build_input_safety_readiness_report(
        page_operability={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "agent_contract": {
                        "input_safety": {
                            "status": "review_only",
                            "safe_to_type": False,
                        }
                    },
                }
            ]
        },
        act_preflight_matrix={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "usable_with_blocked_actions",
                    "type_text": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "review_only",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                    },
                    "send": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "disabled_send_blocked",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                    },
                    "failures": [],
                }
            ]
        },
        send_probe_report={
            "rows": [
                {
                    "app": "qq",
                    "status": "pass",
                    "text_observed_after": True,
                    "readback_observed_after": True,
                }
            ]
        },
    )

    assert report["overall_status"] == "controlled_probe_verified_review_only"
    row = report["rows"][0]
    assert row["status"] == "controlled_probe_verified_review_only"
    assert row["controlled_probe_status"] == "pass"
    assert row["can_upgrade_safe_to_type"] is False
    assert row["safe_type_upgrade_gate"] == {
        "decision": "blocked",
        "satisfied": [
            "type_text_preflight_complete",
            "send_preflight_complete",
            "controlled_probe_passed",
        ],
        "missing": [
            "explicit_safe_type_policy",
            "state_template_input_transition_verified",
            "multi_sample_stability_verified",
        ],
    }
    assert row["remaining_blockers"] == ["safe_to_type_false", "safe_type_upgrade_policy_pending"]


def test_readiness_gate_lists_missing_controlled_probe_when_not_run():
    module = _load_module()
    report = module.build_input_safety_readiness_report(
        page_operability={
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "agent_contract": {"input_safety": {"status": "review_only", "safe_to_type": False}},
                }
            ]
        },
        act_preflight_matrix={
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "status": "usable_with_blocked_actions",
                    "type_text": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "review_only",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                    },
                    "send": {
                        "present": True,
                        "protocol_complete": True,
                        "safe_boundary": "disabled_send_blocked",
                        "has_state_probe_plan": True,
                        "has_readback_plan": True,
                    },
                    "failures": [],
                }
            ]
        },
    )

    gate = report["rows"][0]["safe_type_upgrade_gate"]
    assert gate["decision"] == "blocked"
    assert "controlled_probe_passed" in gate["missing"]
    assert "type_text_preflight_complete" in gate["satisfied"]
