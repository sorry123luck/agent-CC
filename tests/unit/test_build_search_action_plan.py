"""Tests for search action plan generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_search_action_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_search_action_plan", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_search_plan_uses_detected_search_click_and_bounds():
    module = _load_module()
    plan = module.build_search_plan({
        "sample_matrix_schema_version": "2026-05-25.chat-variant.v3",
        "rows": [
            {
                "sample": "qq_group",
                "hwnd": 123,
                "title": "23333 (3)",
                "process_name": "qq.exe",
                "mode": "chat_workspace",
                "search_candidate_count": 1,
                "search_primary_click_point": [163, 42],
                "search_bounds": [78, 28, 248, 57],
                "search_source": "sidebar_geometry",
                "search_evidence": "top_left_search_icon",
            }
        ],
    })

    assert plan["overall_status"] == "ready"
    assert plan["counts"] == {"ready": 1, "blocked": 0}
    row = plan["rows"][0]
    assert row["requires_controlled_probe"] is True
    assert row["hwnd"] == 123
    assert row["title"] == "23333 (3)"
    assert row["search_click"] == [163, 42]
    assert row["search_bounds"] == [78, 28, 248, 57]
    assert row["source"] == "sidebar_geometry"
    assert row["evidence"] == "top_left_search_icon"


def test_build_search_plan_blocks_missing_search_candidate():
    module = _load_module()
    plan = module.build_search_plan({
        "rows": [
            {
                "sample": "voicemeeter",
                "process_name": "voicemeeter.exe",
                "mode": "mixer_workspace",
                "search_candidate_count": 0,
                "search_primary_click_point": [],
            }
        ],
    })

    assert plan["overall_status"] == "blocked"
    assert plan["rows"][0]["status"] == "blocked"
    assert "missing_search_click" in plan["rows"][0]["reasons"]
    assert "missing_search_candidate" in plan["rows"][0]["reasons"]


def test_build_search_plan_ignores_rows_without_process_name():
    module = _load_module()
    plan = module.build_search_plan({
        "rows": [
            {
                "sample": "empty",
                "process_name": "",
                "search_candidate_count": 1,
                "search_primary_click_point": [10, 10],
            }
        ],
    })

    assert plan["overall_status"] == "blocked"
    assert plan["rows"] == []


def test_write_search_plan_outputs_json_and_markdown(tmp_path):
    module = _load_module()
    plan = {
        "generated_at": "2026-05-25T12:00:00",
        "source_schema": "2026-05-25.chat-variant.v3",
        "overall_status": "ready",
        "counts": {"ready": 1, "blocked": 0},
        "rows": [
            {
                "sample": "feishu",
                "process_name": "feishu.exe",
                "mode": "collaboration_inbox",
                "status": "ready",
                "search_click": [94, 72],
                "source": "ocr",
                "reasons": [],
            }
        ],
    }

    module.write_search_plan(plan, tmp_path)

    saved = json.loads((tmp_path / "search_action_plan.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "search_action_plan.md").read_text(encoding="utf-8")
    assert saved["overall_status"] == "ready"
    assert "Search Action Plan" in markdown
    assert "| feishu | 0 | feishu.exe | collaboration_inbox | ready | [94, 72] | ocr |  |" in markdown
