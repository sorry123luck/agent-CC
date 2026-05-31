"""Tests for chat action plan generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_chat_action_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_chat_action_plan", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_action_plan_uses_review_input_and_send_target_centers():
    module = _load_module()
    plan = module.build_action_plan({
        "sample_matrix_schema_version": "2026-05-25.chat-variant.v3",
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "composer_input_safe_count": 0,
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [579, 664],
                "composer_input_review_bounds": [[300, 615, 858, 713]],
                "composer_send_target_bounds": [918, 683, 978, 710],
            }
        ],
    })

    assert plan["overall_status"] == "ready"
    assert plan["counts"] == {"ready": 1, "warn": 0, "blocked": 0}
    row = plan["rows"][0]
    assert row["requires_controlled_probe"] is True
    assert row["safe_to_type"] is False
    assert row["input_click"] == [579, 664]
    assert row["send_click"] == [948, 696]
    assert row["status"] == "ready"


def test_build_action_plan_blocks_missing_input_click():
    module = _load_module()
    plan = module.build_action_plan({
        "rows": [
            {
                "sample": "qq",
                "process_name": "qq.exe",
                "mode": "chat_workspace",
                "composer_input_safe_count": 0,
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [],
                "composer_send_target_bounds": [844, 598, 942, 627],
            }
        ],
    })

    assert plan["overall_status"] == "blocked"
    assert plan["rows"][0]["status"] == "blocked"
    assert "missing_input_click" in plan["rows"][0]["reasons"]


def test_build_action_plan_warns_when_input_is_unexpectedly_safe():
    module = _load_module()
    plan = module.build_action_plan({
        "rows": [
            {
                "sample": "feishu",
                "process_name": "feishu.exe",
                "mode": "collaboration_inbox",
                "composer_input_safe_count": 1,
                "composer_input_review_count": 1,
                "composer_input_primary_click_point": [749, 698],
                "composer_send_target_bounds": [890, 702, 980, 728],
            }
        ],
    })

    assert plan["overall_status"] == "warn"
    assert plan["rows"][0]["safe_to_type"] is False
    assert "input_marked_safe_unexpected" in plan["rows"][0]["reasons"]


def test_write_action_plan_outputs_json_and_markdown(tmp_path):
    module = _load_module()
    plan = {
        "generated_at": "2026-05-25T12:00:00",
        "source_schema": "2026-05-25.chat-variant.v3",
        "overall_status": "ready",
        "counts": {"ready": 1, "warn": 0, "blocked": 0},
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "mode": "chat_workspace",
                "status": "ready",
                "input_click": [579, 664],
                "send_click": [948, 696],
                "safe_to_type": False,
                "reasons": [],
            }
        ],
    }

    module.write_action_plan(plan, tmp_path)

    saved = json.loads((tmp_path / "chat_action_plan.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "chat_action_plan.md").read_text(encoding="utf-8")
    assert saved["overall_status"] == "ready"
    assert "Chat Action Plan" in markdown
    assert "| wechat | weixin.exe | chat_workspace | ready | [579, 664] | [948, 696] | false |  |" in markdown
