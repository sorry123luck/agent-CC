"""Tests for chat readback snapshot acceptance reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_chat_readback_snapshot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_chat_readback_snapshot", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_snapshot_acceptance_marks_clean_chat_rows_pass():
    module = _load_module()
    report = module.build_snapshot_acceptance(
        {
            "rows": [
                {
                    "sample": "qq-group",
                    "process_name": "qq.exe",
                    "status": "captured",
                    "readback_event_count": 2,
                    "readback_has_events": True,
                    "readback_warnings": [],
                    "readback_result": {
                        "events": [
                            {"text": "消息测试123", "sender": "me", "confidence": 0.99},
                            {"text": "消息测试456", "sender": "peer", "confidence": 0.99},
                        ]
                    },
                }
            ]
        }
    )

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["status"] == "pass"
    assert report["rows"][0]["message_text_count"] == 2
    assert report["rows"][0]["warnings"] == []


def test_build_snapshot_acceptance_warns_on_loading_or_empty_rows():
    module = _load_module()
    report = module.build_snapshot_acceptance(
        {
            "rows": [
                {
                    "sample": "feishu-loading",
                    "process_name": "feishu.exe",
                    "status": "captured",
                    "readback_event_count": 1,
                    "readback_has_events": True,
                    "readback_warnings": ["message_stream_loading"],
                    "readback_result": {"events": [{"text": "正在加载...", "sender": "unknown", "confidence": 0.88}]},
                },
                {
                    "sample": "wechat-empty",
                    "process_name": "weixin.exe",
                    "status": "captured",
                    "readback_event_count": 0,
                    "readback_has_events": False,
                    "readback_warnings": [],
                    "readback_result": {"events": []},
                },
            ]
        }
    )

    assert report["overall_status"] == "warn"
    assert report["rows"][0]["status"] == "warn"
    assert "message_stream_loading" in report["rows"][0]["warnings"]
    assert report["rows"][1]["status"] == "warn"
    assert "no_message_events" in report["rows"][1]["warnings"]


def test_analyze_snapshot_dir_loads_nested_readback_results(tmp_path: Path):
    module = _load_module()
    readback_dir = tmp_path / "readback"
    readback_dir.mkdir()
    result_path = readback_dir / "qq_snapshot_readback.json"
    result_path.write_text(
        json.dumps(
            {
                "window_id": "1",
                "app_process": "qq.exe",
                "status": "pass",
                "warnings": [],
                "events": [{"text": "消息测试123", "sender": "me", "confidence": 0.99}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "chat_readback_snapshot_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample": "qq",
                        "process_name": "qq.exe",
                        "status": "captured",
                        "readback_result_path": str(result_path),
                        "readback_event_count": 1,
                        "readback_has_events": True,
                        "readback_warnings": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.analyze_snapshot_dir(tmp_path, output_dir=tmp_path / "analysis")

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["sample"] == "qq"
    assert report["rows"][0]["message_texts"] == ["消息测试123"]
    assert (tmp_path / "analysis" / "chat_readback_snapshot_acceptance.json").exists()


def test_analyze_snapshot_dir_resolves_repo_relative_readback_paths(tmp_path: Path, monkeypatch):
    module = _load_module()
    repo = tmp_path / "repo"
    snapshot_dir = repo / "artifacts" / "snapshot"
    readback_dir = snapshot_dir / "readback"
    readback_dir.mkdir(parents=True)
    result_path = readback_dir / "qq_snapshot_readback.json"
    result_path.write_text(
        json.dumps(
            {
                "window_id": "1",
                "app_process": "qq.exe",
                "status": "pass",
                "warnings": [],
                "events": [{"text": "消息测试456", "sender": "peer", "confidence": 0.99}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "chat_readback_snapshot_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample": "qq",
                        "process_name": "qq.exe",
                        "status": "captured",
                        "readback_result_path": "artifacts\\snapshot\\readback\\qq_snapshot_readback.json",
                        "readback_event_count": 1,
                        "readback_has_events": True,
                        "readback_warnings": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    report = module.analyze_snapshot_dir(snapshot_dir, output_dir=snapshot_dir / "analysis")

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["message_texts"] == ["消息测试456"]
