"""Tests for read-only chat message stream snapshots."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_chat_readback_snapshot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_chat_readback_snapshot", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_snapshot_targets_keeps_ready_rows_without_send_text(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: True)

    targets = module.build_snapshot_targets(
        {
            "rows": [
                {
                    "sample": "wechat-vincci",
                    "process_name": "weixin.exe",
                    "status": "ready",
                    "hwnd": 12,
                    "input_click": [510, 660],
                    "input_bounds": [[300, 615, 858, 713]],
                    "send_click": [945, 696],
                },
                {
                    "sample": "qq-blocked",
                    "process_name": "qq.exe",
                    "status": "blocked",
                    "hwnd": 13,
                    "input_click": [400, 480],
                    "send_click": [900, 610],
                },
            ]
        },
        current_windows=[],
    )

    assert len(targets) == 1
    assert targets[0]["sample"] == "wechat-vincci"
    assert targets[0]["process_name"] == "weixin.exe"
    assert targets[0]["input_bounds"] == [[300, 615, 858, 713]]
    assert "text" not in targets[0]


def test_run_snapshot_plan_dry_run_writes_report_without_actions(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: True)

    report = module.run_snapshot_plan(
        plan={
            "rows": [
                {
                    "sample": "qq-group",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "hwnd": 31,
                    "input_click": [400, 480],
                    "input_bounds": [[310, 430, 780, 585]],
                    "send_click": [900, 610],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        execute=False,
    )

    assert report["overall_status"] == "dry_run"
    assert report["rows"][0]["status"] == "dry_run"
    assert report["rows"][0]["would_send"] is False
    assert (tmp_path / "chat_readback_snapshot_report.json").exists()
    assert (tmp_path / "chat_readback_snapshot.md").exists()


def test_run_snapshot_plan_execute_captures_message_stream_and_readback(monkeypatch, tmp_path):
    module = _load_module()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_ensure_window_actionable", lambda hwnd, process: calls.append(("ensure", (hwnd, process))))
    monkeypatch.setattr(module, "_bring_window_to_front", lambda hwnd: calls.append(("front", hwnd)))
    monkeypatch.setattr(module, "_save_screen", lambda output_dir, sample, stage: f"{sample}_{stage}.png")
    monkeypatch.setattr(module, "_window_rect", lambda hwnd: [100, 200, 1060, 920])
    monkeypatch.setattr(module, "_save_message_stream_crop", lambda **kwargs: "stream_crop.png")
    monkeypatch.setattr(
        module,
        "_run_readback_for_stream_crop",
        lambda **kwargs: {
            "readback_result_path": "readback/qq_group_snapshot_readback.json",
            "readback_event_count": 2,
            "readback_has_events": True,
            "readback_warnings": [],
        },
    )

    report = module.run_snapshot_plan(
        plan={
            "rows": [
                {
                    "sample": "qq-group",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "hwnd": 31,
                    "input_click": [400, 480],
                    "input_bounds": [[310, 430, 780, 585]],
                    "send_click": [900, 610],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        execute=True,
    )

    assert ("ensure", (31, "qq.exe")) in calls
    assert ("front", 31) in calls
    assert report["overall_status"] == "review"
    assert report["rows"][0]["message_stream_crop"] == "stream_crop.png"
    assert report["rows"][0]["readback_event_count"] == 2
    assert report["rows"][0]["would_send"] is False


def test_run_snapshot_plan_retries_loading_readback(monkeypatch, tmp_path):
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_ensure_window_actionable", lambda hwnd, process: None)
    monkeypatch.setattr(module, "_bring_window_to_front", lambda hwnd: None)
    monkeypatch.setattr(module, "_save_screen", lambda output_dir, sample, stage: f"{sample}_{stage}_{len(calls)}.png")
    monkeypatch.setattr(module, "_window_rect", lambda hwnd: [100, 200, 1060, 920])
    monkeypatch.setattr(module, "_save_message_stream_crop", lambda **kwargs: f"stream_crop_{len(calls)}.png")

    def fake_readback(**kwargs):
        calls.append(kwargs["message_stream_crop"])
        if len(calls) == 1:
            return {
                "readback_result_path": "readback/loading.json",
                "readback_event_count": 1,
                "readback_has_events": True,
                "readback_warnings": ["message_stream_loading"],
            }
        return {
            "readback_result_path": "readback/ready.json",
            "readback_event_count": 3,
            "readback_has_events": True,
            "readback_warnings": [],
        }

    monkeypatch.setattr(module, "_run_readback_for_stream_crop", fake_readback)

    report = module.run_snapshot_plan(
        plan={
            "rows": [
                {
                    "sample": "feishu",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "hwnd": 44,
                    "input_click": [730, 690],
                    "input_bounds": [[509, 658, 990, 738]],
                    "send_click": [940, 720],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        execute=True,
        loading_retry_attempts=1,
        loading_retry_seconds=0,
    )

    assert calls == ["stream_crop_0.png", "stream_crop_1.png"]
    assert report["rows"][0]["readback_event_count"] == 3
    assert report["rows"][0]["readback_warnings"] == []
    assert report["rows"][0]["retry_count"] == 1


def test_write_snapshot_report_includes_readback_summary(tmp_path):
    module = _load_module()
    report = {
        "generated_at": "2026-05-26T12:00:00",
        "overall_status": "review",
        "execute": True,
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "status": "captured",
                "message_stream_crop": "crop.png",
                "readback_event_count": 3,
                "readback_has_events": True,
                "readback_warnings": ["attachment_needs_vlm"],
            }
        ],
    }

    module.write_snapshot_report(report, tmp_path)

    data = json.loads((tmp_path / "chat_readback_snapshot_report.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["readback_event_count"] == 3
    md = (tmp_path / "chat_readback_snapshot.md").read_text(encoding="utf-8")
    assert "attachment_needs_vlm" in md
