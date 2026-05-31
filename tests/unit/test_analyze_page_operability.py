"""Tests for generic page operability aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_page_operability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_page_operability", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_chat_page_reports_review_input_and_controlled_probe_without_default_typing():
    module = _load_module()
    matrix = {
        "rows": [
            {
                "sample": "qq_group",
                "process_name": "qq.exe",
                "mode": "chat_workspace",
                "chat_variant": "group_chat",
                "observe_elapsed_ms": 520,
                "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [486, 576],
                "composer_send_target_bounds": [665, 599, 760, 627],
                "composer_send_target_width": 95,
                "composer_send_target_height": 28,
                "composer_vlm_send_hint_count": 1,
            }
        ]
    }
    acceptance = {
        "rows": [
            {
                "sample": "qq_group",
                "status": "pass",
                "failures": [],
                "warnings": [],
            }
        ]
    }
    readiness = {
        "rows": [
            {
                "sample": "qq_group",
                "status": "controlled_ready",
                "can_run_controlled_send_probe": True,
                "can_default_agent_type": False,
            }
        ]
    }

    report = module.build_page_operability_report(
        sample_matrix=matrix,
        recognition_acceptance=acceptance,
        chat_readiness=readiness,
    )

    row = report["rows"][0]
    assert row["operability_status"] == "usable"
    assert row["page_class"] == "chat_workspace"
    assert row["capabilities"]["read_regions"] == ["message_stream"]
    assert row["capabilities"]["review_input_candidates"][0]["click_point"] == [486, 576]
    assert row["capabilities"]["controlled_send_probe"] is True
    assert row["capabilities"]["default_agent_type"] is False
    assert row["capabilities"]["input_safety"] == {
        "status": "review_only",
        "safe_to_type": False,
        "default_agent_type": False,
        "blockers": [
            "safe_to_type_false",
            "input_click_requires_probe",
            "send_state_requires_probe",
            "readback_required_before_safe_type",
        ],
        "next_probe_plan": [
            {
                "probe": "type_text_preflight",
                "endpoint": "POST /api/v1/act",
                "request": {
                    "action": "type_text",
                    "candidate_role": "message_input",
                    "dry_run": True,
                    "execute_confirmed": False,
                    "params": {"text": "<probe_text>"},
                },
                "required_evidence": [
                    "state_probe_plan.current.input_state",
                    "readback_plan.expected_text",
                ],
            },
            {
                "probe": "send_preflight",
                "endpoint": "POST /api/v1/act",
                "request": {
                    "action": "send",
                    "candidate_role": "send_button",
                    "dry_run": True,
                    "execute_confirmed": False,
                    "params": {"expected_text": "<probe_text>"},
                },
                "required_evidence": [
                    "state_probe_plan.current.send_enabled",
                    "readback_plan.expected_text",
                ],
            },
        ],
    }
    assert row["agent_contract"]["input_safety"]["status"] == "review_only"
    assert "chat_input_review_only" in row["risk_flags"]


def test_evidence_mismatch_blocks_controlled_actions_but_keeps_readability():
    module = _load_module()
    matrix = {
        "rows": [
            {
                "sample": "feishu_wrong_page",
                "process_name": "feishu.exe",
                "mode": "collaboration_inbox",
                "observe_elapsed_ms": 840,
                "roi_purposes": ["app_rail", "inbox_list", "message_thread", "composer"],
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [740, 690],
            }
        ]
    }
    acceptance = {
        "rows": [
            {
                "sample": "feishu_wrong_page",
                "status": "pass",
                "failures": [],
                "warnings": [],
            }
        ]
    }
    readiness = {
        "rows": [
            {
                "sample": "feishu_wrong_page",
                "status": "blocked_evidence_mismatch",
                "can_run_controlled_send_probe": False,
                "warnings": ["evidence_gate_stop"],
            }
        ]
    }

    report = module.build_page_operability_report(
        sample_matrix=matrix,
        recognition_acceptance=acceptance,
        chat_readiness=readiness,
    )

    row = report["rows"][0]
    assert row["operability_status"] == "usable_with_blocked_actions"
    assert row["capabilities"]["read_regions"] == ["message_thread"]
    assert row["capabilities"]["controlled_send_probe"] is False
    assert "evidence_gate_stop" in row["risk_flags"]


def test_failed_recognition_marks_page_unreliable_even_if_readiness_exists():
    module = _load_module()
    matrix = {
        "rows": [
            {
                "sample": "wechat_bad",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "observe_elapsed_ms": 2400,
                "roi_purposes": ["message_stream"],
            }
        ]
    }
    acceptance = {
        "rows": [
            {
                "sample": "wechat_bad",
                "status": "fail",
                "failures": ["missing_composer_roi"],
                "warnings": ["observe_over_2s"],
            }
        ]
    }
    readiness = {
        "rows": [
            {
                "sample": "wechat_bad",
                "status": "controlled_ready",
                "can_run_controlled_send_probe": True,
            }
        ]
    }

    report = module.build_page_operability_report(
        sample_matrix=matrix,
        recognition_acceptance=acceptance,
        chat_readiness=readiness,
    )

    row = report["rows"][0]
    assert row["operability_status"] == "unreliable"
    assert row["capabilities"]["controlled_send_probe"] is False
    assert "missing_composer_roi" in row["risk_flags"]
    assert "observe_over_2s" in row["risk_flags"]


def test_report_exposes_agent_contract_region_classes_for_non_chat_page():
    module = _load_module()
    matrix = {
        "rows": [
            {
                "sample": "flclash",
                "process_name": "flclash.exe",
                "mode": "control_dashboard",
                "observe_elapsed_ms": 430,
                "roi_purposes": [
                    "app_rail",
                    "navigation_list",
                    "dashboard_cards",
                    "action_panel",
                    "status_panel",
                ],
                "read_region_contexts": {
                    "dashboard_cards": {
                        "region_id": "cards",
                        "scroll_context_id": "scroll_cards",
                        "scroll_type": "vertical",
                        "is_virtual": False,
                    }
                },
                "search_primary_click_point": [92, 54],
            }
        ]
    }
    acceptance = {
        "rows": [
            {
                "sample": "flclash",
                "status": "pass",
                "failures": [],
                "warnings": [],
            }
        ]
    }

    report = module.build_page_operability_report(
        sample_matrix=matrix,
        recognition_acceptance=acceptance,
    )

    row = report["rows"][0]
    assert row["operability_status"] == "usable"
    assert row["agent_contract"]["schema_version"] == "2026-05-28.page-operability.v1"
    assert row["agent_contract"]["navigation_regions"] == ["app_rail", "navigation_list"]
    assert row["agent_contract"]["read_regions"] == ["dashboard_cards", "status_panel"]
    assert row["agent_contract"]["action_regions"] == ["action_panel"]
    assert row["agent_contract"]["input_regions"] == []
    assert row["agent_contract"]["read_plan"] == [
        {
            "region": "dashboard_cards",
            "method": "region_text_harvest",
            "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
            "request": {
                "region_role": "dashboard_cards",
                "include_elements": True,
                "include_ocr": True,
                "allow_crop_ocr": False,
            },
            "read_scope": "current_viewport",
            "scroll_context": {
                "region_id": "cards",
                "scroll_context_id": "scroll_cards",
                "scroll_type": "vertical",
                "is_virtual": False,
            },
            "long_content_strategy": {
                "status": "current_viewport_only",
                "reason": "controlled_scroll_available",
            },
            "next_scroll_probe": {
                "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
                "request": {
                    "region_role": "dashboard_cards",
                    "direction": "down",
                    "amount": "page",
                    "dry_run": True,
                    "execute_confirmed": False,
                    "read_after": True,
                },
            },
            "scroll_supported": True,
        },
        {
            "region": "status_panel",
            "method": "region_text_harvest",
            "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
            "request": {
                "region_role": "status_panel",
                "include_elements": True,
                "include_ocr": True,
                "allow_crop_ocr": False,
            },
            "read_scope": "current_viewport",
            "scroll_context": None,
            "long_content_strategy": {
                "status": "current_viewport_only",
                "reason": "no_scroll_context",
            },
            "next_scroll_probe": None,
            "scroll_supported": False,
        },
    ]
    assert row["capabilities"]["action_targets"][0]["role"] == "search_entry"


def test_agent_contract_never_promotes_review_input_to_safe_action():
    module = _load_module()
    matrix = {
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "observe_elapsed_ms": 500,
                "roi_purposes": ["navigation_and_list", "message_stream", "composer"],
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [570, 664],
            }
        ]
    }
    acceptance = {"rows": [{"sample": "wechat", "status": "pass", "failures": [], "warnings": []}]}

    report = module.build_page_operability_report(
        sample_matrix=matrix,
        recognition_acceptance=acceptance,
    )

    contract = report["rows"][0]["agent_contract"]
    assert contract["input_regions"] == ["composer"]
    assert contract["read_plan"] == [
        {
            "region": "message_stream",
            "method": "chat_crop_ocr_readback",
            "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
            "request": {
                "region_role": "message_stream",
                "include_elements": True,
                "include_ocr": True,
                "allow_crop_ocr": True,
            },
            "read_scope": "current_viewport",
            "scroll_context": None,
            "long_content_strategy": {
                "status": "current_viewport_only",
                "reason": "no_scroll_context",
            },
            "next_scroll_probe": None,
            "scroll_supported": False,
        }
    ]
    assert contract["safe_action_targets"] == []
    assert contract["review_required_targets"][0]["role"] == "message_input"
    assert contract["default_agent_type"] is False
