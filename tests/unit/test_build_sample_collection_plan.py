"""Tests for actionable sample collection plans."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_sample_collection_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_sample_collection_plan", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collection_plan_turns_missing_requirements_into_commands():
    module = _load_module()

    plan = module.build_sample_collection_plan(
        coverage_report={
            "overall_status": "coverage_incomplete",
            "requirements": [
                {"requirement_id": "wechat_chat", "label": "WeChat", "status": "covered"},
                {
                    "requirement_id": "qq_private_chat",
                    "label": "QQ private chat",
                    "status": "missing",
                    "reason": "no matching valid sample",
                    "collection_hint": "Open QQ private chat.",
                },
                {
                    "requirement_id": "netease_music",
                    "label": "NetEase",
                    "status": "missing",
                    "reason": "window_not_running_or_not_visible",
                    "collection_hint": "Open NetEase.",
                },
            ],
        }
    )

    assert plan["overall_status"] == "collection_needed"
    assert plan["counts"] == {"ready_to_collect": 0, "needs_human_setup": 2}
    by_id = {row["requirement_id"]: row for row in plan["rows"]}
    assert by_id["qq_private_chat"]["include_processes"] == ["qq.exe"]
    assert "--include-process qq.exe" in by_id["qq_private_chat"]["suggested_command"]
    assert "private chat" in by_id["qq_private_chat"]["prerequisite"]
    assert by_id["netease_music"]["include_processes"] == ["cloudmusic.exe"]
    assert "--include-process cloudmusic.exe" in by_id["netease_music"]["suggested_command"]


def test_collection_plan_writes_json_and_markdown(tmp_path: Path):
    module = _load_module()
    coverage = tmp_path / "sample_coverage_report.json"
    coverage.write_text(
        json.dumps(
            {
                "overall_status": "coverage_incomplete",
                "requirements": [
                    {
                        "requirement_id": "flclash_dashboard",
                        "label": "FlClash",
                        "status": "invalid",
                        "reason": "only_minimized_windows",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_sample_collection_plan_from_file(
        coverage_report_file=coverage,
        output_dir=tmp_path / "out",
    )

    assert plan["rows"][0]["include_processes"] == ["flclash.exe"]
    assert (tmp_path / "out" / "sample_collection_plan.json").exists()
    markdown = (tmp_path / "out" / "sample_collection_plan.md").read_text(encoding="utf-8")
    assert "Sample Collection Plan" in markdown
    assert "--include-process flclash.exe" in markdown


def test_collection_plan_no_gaps_when_coverage_ready():
    module = _load_module()

    plan = module.build_sample_collection_plan(
        coverage_report={
            "overall_status": "coverage_ready",
            "requirements": [{"requirement_id": "wechat_chat", "status": "covered"}],
        }
    )

    assert plan["overall_status"] == "no_gaps"
    assert plan["rows"] == []
