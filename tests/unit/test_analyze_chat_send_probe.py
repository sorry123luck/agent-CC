"""Tests for chat send probe report generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_chat_send_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_chat_send_probe", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_probe_report_marks_text_observed_after_action():
    module = _load_module()
    actions = [
        {
            "app": "qq",
            "ok": True,
            "text": "OpenClaw probe qq",
            "send_bounds": [844, 598, 942, 627],
        }
    ]
    details = [
        {
            "window_title": "QQ",
            "elements": [{"text": "OpenClaw probe qq"}],
            "ocr_blocks": [],
        }
    ]

    report = module.build_probe_report(actions=actions, details=details)

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["status"] == "pass"
    assert report["rows"][0]["text_observed_after"] is True
    assert report["rows"][0]["send_target_width"] == 98


def test_build_probe_report_flags_missing_text_after_action():
    module = _load_module()
    actions = [
        {
            "app": "feishu",
            "ok": True,
            "text": "OpenClaw probe feishu",
            "send_bounds": [958, 704, 982, 732],
        }
    ]
    details = [
        {
            "window_title": "飞书",
            "elements": [{"text": "unrelated message"}],
            "ocr_blocks": [],
        }
    ]

    report = module.build_probe_report(actions=actions, details=details)

    assert report["overall_status"] == "fail"
    assert report["rows"][0]["status"] == "fail"
    assert "probe_text_not_observed_after" in report["rows"][0]["failures"]
    assert "send_target_too_narrow" in report["rows"][0]["warnings"]


def test_build_probe_report_uses_latest_action_for_same_app():
    module = _load_module()
    actions = [
        {"app": "wechat", "ok": False, "text": "old text"},
        {"app": "wechat", "ok": True, "text": "new text", "send_bounds": [900, 600, 960, 630]},
    ]
    details = [{"window_title": "微信", "elements": [{"text": "new text"}], "ocr_blocks": []}]

    report = module.build_probe_report(actions=actions, details=details)

    assert report["overall_status"] == "pass"
    assert len(report["rows"]) == 1
    assert report["rows"][0]["text"] == "new text"


def test_build_probe_report_prefers_matching_detail_that_contains_probe_text():
    module = _load_module()
    actions = [{"app": "qq", "ok": True, "text": "OpenClaw probe qq", "send_bounds": [844, 598, 942, 627]}]
    details = [
        {"window_title": "QQ", "elements": [{"text": "small account window"}], "ocr_blocks": []},
        {"window_title": "QQ", "elements": [{"text": "OpenClaw probe qq"}], "ocr_blocks": []},
    ]

    report = module.build_probe_report(actions=actions, details=details)

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["text_observed_after"] is True


def test_analyze_probe_dirs_writes_report(tmp_path):
    module = _load_module()
    action_dir = tmp_path / "actions"
    after_dir = tmp_path / "after"
    action_dir.mkdir()
    after_dir.mkdir()
    (action_dir / "send_probe_actions.json").write_text(
        json.dumps([{"app": "wechat", "ok": True, "text": "OpenClaw probe wechat", "send_bounds": [900, 600, 960, 630]}]),
        encoding="utf-8",
    )
    (after_dir / "weixin.detail.json").write_text(
        json.dumps({"window_title": "微信", "elements": [], "ocr_blocks": [{"text": "OpenClaw probe wechat"}]}),
        encoding="utf-8",
    )

    report = module.analyze_probe_dirs(action_dir=action_dir, after_dir=after_dir, output_dir=tmp_path)

    assert report["overall_status"] == "pass"
    assert (tmp_path / "chat_send_probe_report.json").exists()
    assert "OpenClaw probe wechat" in (tmp_path / "chat_send_probe_report.md").read_text(encoding="utf-8")


def test_analyze_probe_dirs_loads_visual_review_from_action_dir_by_default(tmp_path):
    module = _load_module()
    action_dir = tmp_path / "actions"
    after_dir = tmp_path / "after"
    action_dir.mkdir()
    after_dir.mkdir()
    (action_dir / "send_probe_actions.json").write_text(
        json.dumps([{"app": "wechat", "ok": True, "text": "OpenClaw probe wechat", "send_bounds": [900, 600, 960, 630]}]),
        encoding="utf-8",
    )
    (action_dir / "chat_send_visual_review.json").write_text(
        json.dumps({"reviews": [{"app": "wechat", "text": "OpenClaw probe wechat", "observed_after": True}]}),
        encoding="utf-8",
    )
    (after_dir / "weixin.detail.json").write_text(
        json.dumps({"window_title": "微信", "elements": [], "ocr_blocks": []}),
        encoding="utf-8",
    )

    report = module.analyze_probe_dirs(action_dir=action_dir, after_dir=after_dir, output_dir=tmp_path)

    assert report["overall_status"] == "warn"
    assert report["rows"][0]["visual_observed_after"] is True
    assert "machine_text_missing_visual_observed" in report["rows"][0]["warnings"]


def test_analyze_probe_dirs_loads_retry_action_files(tmp_path):
    module = _load_module()
    action_dir = tmp_path / "actions"
    after_dir = tmp_path / "after"
    action_dir.mkdir()
    after_dir.mkdir()
    (action_dir / "send_retry_action.json").write_text(
        json.dumps({"app": "feishu", "ok": True, "text": "OpenClaw probe feishu", "send_bounds": [890, 702, 980, 728]}),
        encoding="utf-8",
    )
    (after_dir / "feishu.detail.json").write_text(
        json.dumps({"window_title": "飞书", "elements": [{"text": "OpenClaw probe feishu"}], "ocr_blocks": []}),
        encoding="utf-8",
    )

    report = module.analyze_probe_dirs(action_dir=action_dir, after_dir=after_dir, output_dir=tmp_path)

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["app"] == "feishu"


def test_analyze_probe_dirs_loads_probe_detail_json_without_detail_suffix(tmp_path):
    module = _load_module()
    action_dir = tmp_path / "actions"
    after_dir = tmp_path / "after"
    action_dir.mkdir()
    after_dir.mkdir()
    (action_dir / "send_probe_actions.json").write_text(
        json.dumps([{"app": "qq", "ok": True, "text": "OpenClaw probe qq", "send_bounds": [844, 598, 942, 627]}]),
        encoding="utf-8",
    )
    (after_dir / "qq_typed_snap_123.json").write_text(
        json.dumps({"window_title": "QQ", "elements": [{"text": "OpenClaw probe qq"}], "ocr_blocks": []}),
        encoding="utf-8",
    )

    report = module.analyze_probe_dirs(action_dir=action_dir, after_dir=after_dir, output_dir=tmp_path)

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["text_observed_after"] is True


def test_build_probe_report_warns_when_action_log_has_no_expected_text():
    module = _load_module()
    actions = [{"app": "feishu_retry", "send_bounds": [890, 702, 980, 728]}]
    details = [{"window_title": "飞书", "elements": [{"text": "OpenClaw probe feishu"}], "ocr_blocks": []}]

    report = module.build_probe_report(actions=actions, details=details)

    assert report["overall_status"] == "warn"
    assert report["rows"][0]["status"] == "warn"
    assert "probe_expected_text_missing" in report["rows"][0]["warnings"]
    assert "probe_action_status_missing" in report["rows"][0]["warnings"]


def test_build_probe_report_warns_when_visual_review_observes_missing_machine_text():
    module = _load_module()
    actions = [
        {
            "app": "wechat",
            "ok": True,
            "text": "OpenClaw probe wechat",
            "send_bounds": [918, 683, 978, 710],
        }
    ]
    details = [{"window_title": "微信", "elements": [], "ocr_blocks": []}]
    visual_reviews = [
        {
            "app": "wechat",
            "text": "OpenClaw probe wechat",
            "observed_after": True,
            "source": "screenshots/weixin.png",
        }
    ]

    report = module.build_probe_report(actions=actions, details=details, visual_reviews=visual_reviews)

    assert report["overall_status"] == "warn"
    assert report["rows"][0]["status"] == "warn"
    assert report["rows"][0]["text_observed_after"] is False
    assert report["rows"][0]["visual_observed_after"] is True
    assert "machine_text_missing_visual_observed" in report["rows"][0]["warnings"]
    assert "probe_text_not_observed_after" not in report["rows"][0]["failures"]


def test_build_probe_report_accepts_readback_worker_evidence():
    module = _load_module()
    actions = [
        {
            "app": "wechat",
            "ok": True,
            "text": "OpenClaw probe wechat",
            "send_bounds": [918, 683, 978, 710],
        }
    ]
    details = [{"window_title": "微信", "elements": [], "ocr_blocks": []}]
    readback_results = [
        {
            "app_process": "weixin.exe",
            "window_id": "12",
            "events": [
                {
                    "message_id": "ocr_0",
                    "sender": "me",
                    "message_type": "text",
                    "text": "OpenClaw probe wechat",
                    "bounds": [600, 400, 900, 440],
                    "confidence": 0.91,
                    "source": "ocr_crop",
                    "attachments": [],
                }
            ],
        }
    ]

    report = module.build_probe_report(actions=actions, details=details, readback_results=readback_results)

    assert report["overall_status"] == "pass"
    assert report["rows"][0]["readback_observed_after"] is True
    assert report["rows"][0]["text_evidence_source"] == "readback_crop_ocr_confirmed"
    assert "probe_text_not_observed_after" not in report["rows"][0]["failures"]
