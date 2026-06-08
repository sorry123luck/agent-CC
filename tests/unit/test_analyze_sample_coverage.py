"""Tests for representative real-sample coverage reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_sample_coverage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_sample_coverage", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_coverage_passes_when_required_representatives_are_present():
    module = _load_module()

    report = module.build_sample_coverage_report(
        sample_matrix={
            "rows": [
                {"sample": "wx", "process_name": "weixin.exe", "mode": "chat_workspace"},
                {"sample": "qq_private", "process_name": "qq.exe", "mode": "chat_workspace", "chat_variant": "private_chat"},
                {"sample": "qq_group", "process_name": "qq.exe", "mode": "chat_workspace", "chat_variant": "group_chat"},
                {"sample": "feishu", "process_name": "feishu.exe", "mode": "collaboration_inbox"},
                {"sample": "flclash", "process_name": "flclash.exe", "mode": "control_dashboard"},
                {"sample": "voicemeeter", "process_name": "voicemeeter.exe", "mode": "control_matrix"},
                {"sample": "music", "process_name": "cloudmusic.exe", "mode": "media_video_home"},
            ]
        }
    )

    assert report["overall_status"] == "coverage_ready"
    assert report["counts"] == {"covered": 8, "missing": 0, "invalid": 0}
    assert all(item["status"] == "covered" for item in report["requirements"])


def test_sample_coverage_distinguishes_missing_and_invalid_tiny_windows():
    module = _load_module()

    report = module.build_sample_coverage_report(
        sample_matrix={
            "rows": [
                {"sample": "qq_tiny", "process_name": "qq.exe", "mode": "account_switcher", "chat_variant": "", "screenshot_success": False}
            ],
            "excluded_rows": [
                {"sample": "qq_tiny", "process_name": "qq.exe", "excluded_reason": "invalid_tiny_window_no_screenshot"}
            ],
        },
        coverage_manifest={
            "rows": [
                {"process_name": "flclash.exe", "covered": False, "reason": "only_minimized_windows"},
                {"process_name": "cloudmusic.exe", "covered": False, "reason": "window_not_running_or_not_visible"},
            ]
        },
    )

    assert report["overall_status"] == "coverage_incomplete"
    by_id = {item["requirement_id"]: item for item in report["requirements"]}
    assert by_id["qq_private_chat"]["status"] == "missing"
    assert by_id["qq_private_chat"]["reason"] == "no matching valid sample"
    assert "--include-process qq.exe" in by_id["qq_private_chat"]["collection_hint"]
    assert by_id["flclash_dashboard"]["status"] == "missing"
    assert by_id["flclash_dashboard"]["reason"] == "only_minimized_windows"
    assert "only_minimized_windows" in by_id["flclash_dashboard"]["collection_hint"]
    assert by_id["qq_valid_window"]["status"] == "invalid"
    assert "invalid_tiny_window_no_screenshot" in by_id["qq_valid_window"]["reason"]
    assert "tiny windows are excluded" in by_id["qq_valid_window"]["collection_hint"]


def test_analyze_sample_coverage_dirs_writes_outputs(tmp_path: Path):
    module = _load_module()
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / "sample_matrix_summary.json").write_text(
        json.dumps({"rows": [{"sample": "wx", "process_name": "weixin.exe", "mode": "chat_workspace"}]}),
        encoding="utf-8",
    )

    report = module.analyze_sample_coverage_dirs(matrix_dir=matrix_dir, output_dir=tmp_path / "out")

    assert report["overall_status"] == "coverage_incomplete"
    assert (tmp_path / "out" / "sample_coverage_report.json").exists()
    markdown = (tmp_path / "out" / "sample_coverage_report.md").read_text(encoding="utf-8")
    assert "Sample Coverage Report" in markdown
    assert "wechat_chat" in markdown
    assert "collection_hint" in markdown


def test_analyze_sample_coverage_dirs_can_merge_multiple_matrix_dirs(tmp_path: Path):
    module = _load_module()
    chat_dir = tmp_path / "chat"
    tool_dir = tmp_path / "tools"
    chat_dir.mkdir()
    tool_dir.mkdir()
    (chat_dir / "sample_matrix_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"sample": "wx", "process_name": "weixin.exe", "mode": "chat_workspace"},
                    {"sample": "qq_group", "process_name": "qq.exe", "mode": "chat_workspace", "chat_variant": "group_chat"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tool_dir / "sample_matrix_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"sample": "flclash", "process_name": "flclash.exe", "mode": "control_dashboard"},
                    {"sample": "voice", "process_name": "voicemeeter.exe", "mode": "control_matrix"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = module.analyze_sample_coverage_dirs(matrix_dirs=[chat_dir, tool_dir], output_dir=tmp_path / "out")

    by_id = {item["requirement_id"]: item for item in report["requirements"]}
    assert by_id["wechat_chat"]["status"] == "covered"
    assert by_id["qq_group_chat"]["status"] == "covered"
    assert by_id["flclash_dashboard"]["status"] == "covered"
    assert by_id["voicemeeter_control_matrix"]["status"] == "covered"
    assert report["source_dirs"] == [str(chat_dir), str(tool_dir)]
