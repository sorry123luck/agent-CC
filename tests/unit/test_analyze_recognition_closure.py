"""Tests for recognition-chain closure reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_recognition_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_recognition_closure", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_closure_marks_recognition_chain_closed_with_deferred_product_work():
    module = _load_module()
    report = module.build_closure_report(
        sample_matrix={
            "sample_matrix_schema_version": "2026-05-25.chat-variant.v3",
            "sample_count": 3,
            "excluded_count": 1,
            "rows": [
                {"sample": "wechat", "process_name": "weixin.exe", "mode": "chat_workspace"},
                {"sample": "qq", "process_name": "qq.exe", "mode": "chat_workspace"},
                {"sample": "feishu", "process_name": "feishu.exe", "mode": "collaboration_inbox"},
            ],
        },
        recognition_acceptance={
            "overall_status": "pass",
            "counts": {"pass": 3, "warn": 0, "fail": 0},
            "rows": [
                {"sample": "wechat", "status": "pass"},
                {"sample": "qq", "status": "pass"},
                {"sample": "feishu", "status": "pass"},
            ],
        },
        page_operability={
            "overall_status": "usable_with_blocked_actions",
            "counts": {"usable": 2, "usable_with_blocked_actions": 1},
            "rows": [
                {
                    "sample": "wechat",
                    "operability_status": "usable",
                    "agent_contract": {
                        "read_plan": [
                            {"endpoint": "POST /api/v1/canvases/{canvas_id}/read-region"}
                        ],
                        "default_agent_type": False,
                    },
                },
                {
                    "sample": "qq",
                    "operability_status": "usable",
                    "agent_contract": {
                        "read_plan": [
                                {
                                    "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
                                    "next_scroll_probe": {
                                        "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
                                        "request": {"dry_run": True, "execute_confirmed": False},
                                    },
                                }
                        ],
                        "default_agent_type": False,
                    },
                },
                {
                    "sample": "feishu",
                    "operability_status": "usable_with_blocked_actions",
                    "agent_contract": {"read_plan": [], "default_agent_type": False},
                },
            ],
        },
        act_preflight_matrix={
            "overall_status": "usable_with_blocked_actions",
            "rows": [
                {"sample": "wechat", "status": "usable_with_blocked_actions"},
                {"sample": "qq", "status": "usable_with_blocked_actions"},
            ],
        },
    )

    assert report["overall_status"] == "closed_for_recognition_chain"
    assert report["completion"]["real_sample_rerun"]["status"] == "pass"
    assert report["completion"]["page_operability_contract"]["status"] == "pass"
    assert report["completion"]["region_read_api_contract"]["status"] == "pass"
    assert report["completion"]["controlled_scroll_contract"]["status"] == "pass"
    assert report["completion"]["safety_boundary"]["status"] == "pass"
    assert report["completion"]["act_preflight_contract"]["status"] == "pass"
    assert report["deferred_after_closure"][0]["item"] == "default_safe_typing"


def test_closure_fails_when_present_act_preflight_matrix_is_unusable():
    module = _load_module()
    report = module.build_closure_report(
        sample_matrix={"sample_count": 1, "rows": [{"sample": "qq"}]},
        recognition_acceptance={"overall_status": "pass", "counts": {"pass": 1}},
        page_operability={
            "overall_status": "usable",
            "rows": [
                {
                    "sample": "qq",
                    "operability_status": "usable",
                    "agent_contract": {
                        "read_plan": [{"endpoint": "POST /api/v1/canvases/{canvas_id}/read-region"}],
                        "default_agent_type": False,
                    },
                }
            ],
        },
        act_preflight_matrix={
            "overall_status": "unusable",
            "rows": [{"sample": "qq", "status": "unusable", "failures": ["click_missing_execution_policy"]}],
        },
    )

    assert report["overall_status"] == "open"
    assert report["completion"]["act_preflight_contract"]["status"] == "fail"
    assert "click_missing_execution_policy" in report["completion"]["act_preflight_contract"]["evidence"]


def test_closure_requires_passing_recognition_acceptance():
    module = _load_module()
    report = module.build_closure_report(
        sample_matrix={"sample_count": 1, "rows": [{"sample": "bad"}]},
        recognition_acceptance={"overall_status": "fail", "counts": {"fail": 1}, "rows": [{"sample": "bad", "status": "fail"}]},
        page_operability={"overall_status": "usable", "rows": []},
    )

    assert report["overall_status"] == "open"
    assert report["completion"]["real_sample_rerun"]["status"] == "fail"
