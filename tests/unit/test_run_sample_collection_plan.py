"""Tests for executing sample collection plans."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_sample_collection_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_sample_collection_plan", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "requirement_id": "qq_private_chat",
                        "include_processes": ["qq.exe"],
                        "prerequisite": "Restore QQ private chat.",
                    },
                    {
                        "requirement_id": "netease_music",
                        "include_processes": ["cloudmusic.exe"],
                        "prerequisite": "Open NetEase.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_plan_runner_dry_run_marks_visible_targets_as_planned(tmp_path: Path):
    module = _load_module()
    plan = tmp_path / "sample_collection_plan.json"
    _write_plan(plan)

    report = module.run_sample_collection_plan(
        plan_file=plan,
        output_dir=tmp_path / "out",
        execute=False,
        window_provider=lambda _base: [
            {"process_name": "qq.exe", "title": "QQ", "hwnd": 100, "is_minimized": False}
        ],
    )

    by_id = {row["requirement_id"]: row for row in report["rows"]}
    assert by_id["qq_private_chat"]["status"] == "planned"
    assert by_id["qq_private_chat"]["selected_window"] == "qq.exe hwnd=100 title=QQ"
    assert by_id["netease_music"]["status"] == "skipped_no_visible_window"
    assert report["overall_status"] == "collection_waiting_for_targets"
    assert (tmp_path / "out" / "sample_collection_run_report.json").exists()


def test_plan_runner_execute_collects_visible_target(tmp_path: Path):
    module = _load_module()
    plan = tmp_path / "sample_collection_plan.json"
    _write_plan(plan)
    calls = []

    def fake_collector(**kwargs):
        calls.append(kwargs)
        return {
            "sample_count": 1,
            "error_count": 0,
            "rows": [
                {
                    "sample": "qq_private",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "chat_variant": "private_chat",
                }
            ],
        }

    report = module.run_sample_collection_plan(
        plan_file=plan,
        output_dir=tmp_path / "out",
        execute=True,
        window_provider=lambda _base: [
            {"process_name": "qq.exe", "title": "QQ", "hwnd": 100, "is_minimized": False}
        ],
        collector=fake_collector,
    )

    by_id = {row["requirement_id"]: row for row in report["rows"]}
    assert by_id["qq_private_chat"]["status"] == "collected"
    assert by_id["qq_private_chat"]["reason"] == "samples=1"
    assert by_id["netease_music"]["status"] == "skipped_no_visible_window"
    assert calls[0]["include_processes"] == ["qq.exe"]
    assert str(calls[0]["output_dir"]).endswith("collections\\qq_private_chat") or str(calls[0]["output_dir"]).endswith("collections/qq_private_chat")


def test_plan_runner_marks_collected_wrong_page_as_review(tmp_path: Path):
    module = _load_module()
    plan = tmp_path / "sample_collection_plan.json"
    _write_plan(plan)

    report = module.run_sample_collection_plan(
        plan_file=plan,
        output_dir=tmp_path / "out",
        execute=True,
        window_provider=lambda _base: [
            {"process_name": "qq.exe", "title": "QQ", "hwnd": 100, "is_minimized": False}
        ],
        collector=lambda **_kwargs: {
            "sample_count": 1,
            "error_count": 0,
            "rows": [
                {
                    "sample": "qq_group",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "chat_variant": "group_chat",
                }
            ],
        },
    )

    by_id = {row["requirement_id"]: row for row in report["rows"]}
    assert by_id["qq_private_chat"]["status"] == "collected_not_matching_requirement"
    assert "requirement_status=missing" in by_id["qq_private_chat"]["reason"]
    assert report["overall_status"] == "collection_review"


def test_plan_runner_reports_collection_failure(tmp_path: Path):
    module = _load_module()
    plan = tmp_path / "sample_collection_plan.json"
    _write_plan(plan)

    report = module.run_sample_collection_plan(
        plan_file=plan,
        output_dir=tmp_path / "out",
        execute=True,
        window_provider=lambda _base: [
            {"process_name": "qq.exe", "title": "QQ", "hwnd": 100, "is_minimized": False}
        ],
        collector=lambda **_kwargs: {"sample_count": 0, "error_count": 1},
    )

    by_id = {row["requirement_id"]: row for row in report["rows"]}
    assert by_id["qq_private_chat"]["status"] == "failed"
    assert report["overall_status"] == "collection_failed"


def test_plan_runner_can_refresh_coverage_and_next_plan_after_collection(tmp_path: Path):
    module = _load_module()
    plan = tmp_path / "sample_collection_plan.json"
    _write_plan(plan)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "sample_matrix_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"sample": "wx", "process_name": "weixin.exe", "mode": "chat_workspace"},
                    {"sample": "qq_group", "process_name": "qq.exe", "mode": "chat_workspace", "chat_variant": "group_chat"},
                    {"sample": "feishu", "process_name": "feishu.exe", "mode": "collaboration_inbox"},
                    {"sample": "flclash", "process_name": "flclash.exe", "mode": "control_dashboard"},
                    {"sample": "voice", "process_name": "voicemeeter8x64.exe", "mode": "control_matrix"},
                    {"sample": "music", "process_name": "cloudmusic.exe", "mode": "media_video_home"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = module.run_sample_collection_plan(
        plan_file=plan,
        output_dir=tmp_path / "out",
        execute=True,
        baseline_matrix_dirs=[baseline],
        window_provider=lambda _base: [
            {"process_name": "qq.exe", "title": "QQ", "hwnd": 100, "is_minimized": False}
        ],
        collector=lambda **_kwargs: {
            "sample_count": 1,
            "error_count": 0,
            "rows": [
                {
                    "sample": "qq_private",
                    "process_name": "qq.exe",
                    "mode": "chat_workspace",
                    "chat_variant": "private_chat",
                }
            ],
        },
    )

    assert report["followup_coverage"]["overall_status"] == "coverage_ready"
    assert report["followup_collection_plan"]["overall_status"] == "no_gaps"
    assert (tmp_path / "out" / "followup_coverage" / "sample_coverage_report.json").exists()
    assert (tmp_path / "out" / "followup_collection_plan" / "sample_collection_plan.json").exists()
