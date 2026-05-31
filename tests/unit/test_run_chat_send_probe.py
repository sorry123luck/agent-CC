"""Tests for controlled chat send probe execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.chat.message_stream_crop import input_crop_box, sent_message_crop_box
from src.chat.message_stream_crop import message_stream_crop_box


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_chat_send_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_chat_send_probe", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_plan_accepts_utf8_bom(tmp_path):
    module = _load_module()
    path = tmp_path / "plan.json"
    path.write_text("\ufeff" + json.dumps({"rows": []}), encoding="utf-8")

    assert module.load_plan(path) == {"rows": []}


def test_build_probe_targets_keeps_ready_rows_and_attaches_text():
    module = _load_module()
    plan = {
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "status": "ready",
                "hwnd": 12,
                "input_click": [510, 660],
                "send_click": [945, 696],
                "send_bounds": [918, 683, 978, 710],
                "input_bounds": [[300, 615, 858, 713]],
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
    }

    targets = module.build_probe_targets(plan, texts={"weixin.exe": "probe"})

    assert len(targets) == 1
    assert targets[0]["process_name"] == "weixin.exe"
    assert targets[0]["text"] == "probe"
    assert targets[0]["input_click"] == [510, 660]
    assert targets[0]["send_click"] == [945, 696]
    assert targets[0]["input_bounds"] == [[300, 615, 858, 713]]


def test_build_probe_targets_applies_identity_gate_stop():
    module = _load_module()
    plan = {
        "rows": [
            {
                "sample": "feishu.exe_68378",
                "process_name": "feishu.exe",
                "status": "ready",
                "hwnd": 44,
                "input_click": [730, 690],
                "send_click": [940, 720],
            },
            {
                "sample": "qq.exe_4590900_QQ",
                "process_name": "qq.exe",
                "status": "ready",
                "hwnd": 31,
                "input_click": [400, 480],
                "send_click": [900, 610],
            },
        ]
    }
    identity_gate = {
        "checks": [
            {"sample": "feishu.exe_68378", "decision": "stop", "reason": "wrong target"},
            {"sample": "qq.exe_4590900_QQ", "decision": "proceed"},
        ]
    }

    targets = module.build_probe_targets(
        plan,
        texts={"feishu.exe": "blocked", "qq.exe": "allowed"},
        identity_gate=identity_gate,
    )

    assert [target["sample"] for target in targets] == ["qq.exe_4590900_QQ"]
    assert targets[0]["text"] == "allowed"


def test_run_probe_plan_reports_identity_gate_skipped_rows(tmp_path):
    module = _load_module()

    report = module.run_probe_plan(
        plan={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "hwnd": 44,
                    "input_click": [730, 690],
                    "send_click": [940, 720],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"feishu.exe": "should not send"},
        execute=False,
        send=True,
        wait_seconds=0,
        identity_gate={"checks": [{"sample": "feishu.exe_68378", "decision": "stop", "reason": "wrong target"}]},
    )

    assert report["overall_status"] == "blocked"
    assert report["rows"][0]["status"] == "blocked_target_mismatch"
    assert report["rows"][0]["would_send"] is False
    assert report["rows"][0]["notes"] == ["identity_gate_stop"]
    assert report["rows"][0]["error"] == "wrong target"


def test_run_probe_plan_accepts_generic_evidence_gate(tmp_path):
    module = _load_module()

    report = module.run_probe_plan(
        plan={
            "rows": [
                {
                    "sample": "feishu.exe_68378",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "hwnd": 44,
                    "input_click": [730, 690],
                    "send_click": [940, 720],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"feishu.exe": "should not send"},
        execute=False,
        send=True,
        wait_seconds=0,
        identity_gate={
            "gate_type": "page_evidence",
            "checks": [{"sample": "feishu.exe_68378", "decision": "stop", "reason": "required evidence text not observed"}],
        },
    )

    assert report["rows"][0]["status"] == "blocked_evidence_mismatch"
    assert report["rows"][0]["notes"] == ["evidence_gate_stop"]


def test_build_probe_targets_refreshes_missing_hwnd_from_current_windows(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: hwnd == 88)

    targets = module.build_probe_targets(
        {
            "rows": [
                {
                    "sample": "wechat",
                    "process_name": "weixin.exe",
                    "status": "ready",
                    "input_click": [510, 660],
                    "send_click": [945, 696],
                }
            ]
        },
        texts={"weixin.exe": "probe"},
        current_windows=[
            {"process_name": "weixin.exe", "hwnd": 88, "is_minimized": False, "width": 1000, "height": 720}
        ],
    )

    assert targets[0]["hwnd"] == 88
    assert targets[0]["hwnd_refreshed"] is True


def test_build_probe_targets_uses_hwnd_embedded_in_sample_before_process_match(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: False)

    targets = module.build_probe_targets(
        {
            "rows": [
                {
                    "sample": "qq.exe_4590900_QQ",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "input_click": [400, 480],
                    "send_click": [900, 610],
                }
            ]
        },
        texts={"qq.exe": "probe"},
        current_windows=[
            {"process_name": "qq.exe", "hwnd": 2164878, "is_minimized": False},
            {"process_name": "qq.exe", "hwnd": 4590900, "is_minimized": True},
        ],
    )

    assert targets[0]["hwnd"] == 4590900


def test_build_probe_targets_ignores_tiny_current_windows(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "_window_is_actionable", lambda hwnd: hwnd > 0)

    targets = module.build_probe_targets(
        {
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "input_click": [400, 480],
                    "send_click": [900, 610],
                }
            ]
        },
        texts={"qq.exe": "probe"},
        current_windows=[
            {"process_name": "qq.exe", "hwnd": 10, "is_minimized": False, "width": 64, "height": 64},
            {"process_name": "qq.exe", "hwnd": 99, "is_minimized": False, "width": 960, "height": 720},
        ],
    )

    assert targets[0]["hwnd"] == 99


def test_run_probe_plan_dry_run_writes_report_without_actions(tmp_path):
    module = _load_module()
    plan = {
        "rows": [
            {
                "sample": "wechat",
                "process_name": "weixin.exe",
                "status": "ready",
                "hwnd": 12,
                "input_click": [510, 660],
                "send_click": [945, 696],
            }
        ]
    }

    report = module.run_probe_plan(
        plan=plan,
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"weixin.exe": "probe"},
        execute=False,
        send=False,
        wait_seconds=0,
    )

    assert report["overall_status"] == "dry_run"
    assert report["rows"][0]["status"] == "dry_run"
    assert report["rows"][0]["would_send"] is False
    assert (tmp_path / "send_probe_actions.json").exists()
    assert (tmp_path / "chat_send_probe.md").exists()


def test_run_probe_plan_type_only_observes_before_and_after(monkeypatch, tmp_path):
    module = _load_module()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_ensure_window_actionable", lambda hwnd, process: calls.append(("ensure", (hwnd, process))))
    monkeypatch.setattr(module, "_reset_probe_surface", lambda hwnd, process: calls.append(("reset", (hwnd, process))))
    monkeypatch.setattr(module, "_observe", lambda base_url, hwnd: calls.append(("observe", hwnd)) or f"canvas_{len(calls)}")
    monkeypatch.setattr(module, "_save_detail", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_save_screen", lambda output_dir, sample, stage: f"{sample}_{stage}.png")
    monkeypatch.setattr(module, "_window_rect", lambda hwnd: [100, 200, 1060, 920])
    monkeypatch.setattr(module, "_save_input_crop", lambda *args, **kwargs: "typed_crop.png")
    monkeypatch.setattr(module, "_save_sent_message_crop", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_focus_and_type", lambda hwnd, click, text: calls.append(("type", (hwnd, click, text))))
    monkeypatch.setattr(module, "_clear_input", lambda hwnd, click: calls.append(("clear", (hwnd, click))))

    report = module.run_probe_plan(
        plan={
            "rows": [
                {
                    "sample": "qq",
                    "process_name": "qq.exe",
                    "status": "ready",
                    "hwnd": 31,
                    "input_click": [400, 480],
                    "input_bounds": [[310, 430, 780, 585]],
                    "send_click": [900, 610],
                    "send_bounds": [840, 598, 942, 627],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"qq.exe": "DeskCanvas probe"},
        execute=True,
        send=False,
        wait_seconds=0,
    )

    assert report["overall_status"] == "review"
    assert ("type", (31, [400, 480], "DeskCanvas probe")) in calls
    assert ("clear", (31, [400, 480])) in calls
    assert not any(call[0] == "click_send" for call in calls)
    assert report["rows"][0]["before_canvas_id"].startswith("canvas_")
    assert report["rows"][0]["typed_canvas_id"].startswith("canvas_")
    assert report["rows"][0]["typed_input_crop"] == "typed_crop.png"


def test_run_probe_plan_send_click_requires_explicit_send(monkeypatch, tmp_path):
    module = _load_module()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_ensure_window_actionable", lambda hwnd, process: None)
    monkeypatch.setattr(module, "_reset_probe_surface", lambda hwnd, process: None)
    monkeypatch.setattr(module, "_observe", lambda base_url, hwnd: f"canvas_{len(calls)}")
    monkeypatch.setattr(module, "_save_detail", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_save_screen", lambda output_dir, sample, stage: f"{sample}_{stage}.png")
    monkeypatch.setattr(module, "_window_rect", lambda hwnd: [100, 200, 1060, 920])
    monkeypatch.setattr(module, "_save_input_crop", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_save_sent_message_crop", lambda *args, **kwargs: "sent_message_crop.png")
    monkeypatch.setattr(module, "_focus_and_type", lambda hwnd, click, text: None)
    monkeypatch.setattr(module, "_clear_input", lambda hwnd, click: None)
    monkeypatch.setattr(module, "_click_relative", lambda hwnd, click: calls.append(("click_send", (hwnd, click))))

    report = module.run_probe_plan(
        plan={
            "rows": [
                {
                    "sample": "feishu",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "hwnd": 44,
                    "input_click": [730, 690],
                    "send_click": [940, 720],
                    "send_bounds": [900, 700, 975, 735],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"feishu.exe": "DeskCanvas probe"},
        execute=True,
        send=True,
        wait_seconds=0,
    )

    assert report["rows"][0]["would_send"] is True
    assert report["rows"][0]["sent_canvas_id"].startswith("canvas_")
    assert report["rows"][0]["sent_message_crop"] == "sent_message_crop.png"
    assert ("click_send", (44, [940, 720])) in calls


def test_run_probe_plan_send_runs_readback_by_default(monkeypatch, tmp_path):
    module = _load_module()
    readback_calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "_fetch_windows", lambda base_url: [])
    monkeypatch.setattr(module, "_ensure_window_actionable", lambda hwnd, process: None)
    monkeypatch.setattr(module, "_reset_probe_surface", lambda hwnd, process: None)
    monkeypatch.setattr(module, "_observe", lambda base_url, hwnd: "canvas")
    monkeypatch.setattr(module, "_save_detail", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_save_screen", lambda output_dir, sample, stage: f"{sample}_{stage}.png")
    monkeypatch.setattr(module, "_window_rect", lambda hwnd: [100, 200, 1060, 920])
    monkeypatch.setattr(module, "_save_input_crop", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "_save_sent_message_crop", lambda *args, **kwargs: "sent_message_crop.png")
    monkeypatch.setattr(module, "_focus_and_type", lambda hwnd, click, text: None)
    monkeypatch.setattr(module, "_click_relative", lambda hwnd, click: None)

    def fake_readback(**kwargs):
        readback_calls.append(kwargs)
        return {
            "readback_result_path": "readback/feishu_sent_readback.json",
            "readback_observed_after": True,
            "readback_event_count": 1,
        }

    monkeypatch.setattr(module, "_run_readback_for_sent_crop", fake_readback)

    report = module.run_probe_plan(
        plan={
            "rows": [
                {
                    "sample": "feishu",
                    "process_name": "feishu.exe",
                    "status": "ready",
                    "hwnd": 44,
                    "input_click": [730, 690],
                    "send_click": [940, 720],
                }
            ]
        },
        output_dir=tmp_path,
        base_url="http://unused",
        texts={"feishu.exe": "DeskCanvas probe"},
        execute=True,
        send=True,
        wait_seconds=0,
    )

    assert readback_calls[0]["text"] == "DeskCanvas probe"
    assert readback_calls[0]["sent_message_crop"] == "sent_message_crop.png"
    assert report["rows"][0]["readback_observed_after"] is True
    assert report["rows"][0]["readback_event_count"] == 1


def test_input_crop_box_prefers_bounds_and_clamps_to_screenshot():
    assert input_crop_box(
        window_rect=None,
        input_bounds=[[10, 20, 210, 80]],
        input_click=[100, 50],
        screenshot_size=(180, 100),
    ) == (10, 0, 180, 80)


def test_input_crop_box_offsets_window_relative_bounds_to_screen():
    assert input_crop_box(
        window_rect=[100, 200, 400, 360],
        input_bounds=[[10, 20, 210, 80]],
        input_click=[100, 50],
        screenshot_size=(500, 400),
    ) == (110, 180, 310, 280)


def test_input_crop_box_falls_back_to_click_band():
    assert input_crop_box(
        window_rect=None,
        input_bounds=[],
        input_click=[100, 50],
        screenshot_size=(300, 200),
    ) == (20, 0, 260, 90)


def test_sent_message_crop_box_uses_area_above_input():
    assert sent_message_crop_box(
        input_bounds=[[300, 615, 858, 713]],
        screenshot_size=(1400, 900),
        window_rect=[200, 100, 1200, 820],
    ) == (500, 495, 1190, 705)


def test_message_stream_crop_box_uses_input_right_edge_to_exclude_side_panels():
    assert message_stream_crop_box(
        input_bounds=[[315, 530, 656, 622]],
        send_bounds=[664, 598, 762, 627],
        screenshot_size=(1200, 900),
        window_rect=[0, 0, 960, 640],
    ) == (288, 80, 782, 530)


def test_message_stream_crop_box_keeps_right_edge_when_gap_is_not_side_panel():
    assert message_stream_crop_box(
        input_bounds=[[306, 619, 851, 718]],
        screenshot_size=(1200, 900),
        window_rect=[0, 0, 1000, 730],
    ) == (300, 80, 1000, 619)


def test_message_stream_crop_box_uses_input_left_when_chat_pane_has_wide_navigation():
    assert message_stream_crop_box(
        input_bounds=[[509, 658, 990, 738]],
        send_bounds=[890, 702, 980, 728],
        screenshot_size=(1200, 900),
        window_rect=[0, 0, 1015, 760],
    ) == (489, 80, 1015, 658)
