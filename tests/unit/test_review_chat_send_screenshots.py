"""Tests for chat send screenshot visual review helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "review_chat_send_screenshots.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("review_chat_send_screenshots", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_review_rows_marks_requested_apps_observed(tmp_path):
    module = _load_module()
    screenshot = tmp_path / "weixin_typed.png"
    screenshot.write_bytes(b"fake")
    actions = [
        {
            "app": "wechat",
            "text": "OpenClaw send probe",
            "typed_screenshot": str(screenshot),
        },
        {
            "app": "qq",
            "text": "OpenClaw send probe",
            "typed_screenshot": str(tmp_path / "qq_typed.png"),
        },
    ]

    rows = module.build_review_rows(actions, observed_apps={"wechat"}, stage="typed")

    assert rows[0]["app"] == "wechat"
    assert rows[0]["observed_after"] is True
    assert rows[0]["review_method"] == "human_visual_review"
    assert rows[1]["observed_after"] is False
    assert rows[1]["review_method"] == "not_reviewed"


def test_build_review_rows_uses_crop_ocr_text_when_present(tmp_path):
    module = _load_module()
    actions = [
        {
            "app": "wechat",
            "text": "OpenClaw send probe",
            "typed_screenshot": str(tmp_path / "weixin_typed.png"),
            "typed_input_crop": str(tmp_path / "weixin_crop.png"),
        }
    ]

    rows = module.build_review_rows(
        actions,
        observed_apps=set(),
        stage="typed",
        crop_ocr_text_by_app={"wechat": "noise OpenClaw send probe"},
    )

    assert rows[0]["observed_after"] is True
    assert rows[0]["review_method"] == "input_crop_ocr"
    assert rows[0]["source"] == str(tmp_path / "weixin_crop.png")


def test_build_review_rows_uses_sent_message_crop_for_sent_stage(tmp_path):
    module = _load_module()
    actions = [
        {
            "app": "wechat",
            "text": "OpenClaw send probe",
            "sent_screenshot": str(tmp_path / "weixin_sent.png"),
            "sent_message_crop": str(tmp_path / "weixin_message.png"),
        }
    ]

    rows = module.build_review_rows(
        actions,
        observed_apps=set(),
        stage="sent",
        crop_ocr_text_by_app={"wechat": "OpenClaw send probe"},
    )

    assert rows[0]["observed_after"] is True
    assert rows[0]["review_method"] == "message_crop_ocr"
    assert rows[0]["source"] == str(tmp_path / "weixin_message.png")


def test_write_review_outputs_json_and_markdown(tmp_path):
    module = _load_module()
    rows = [
        {
            "app": "wechat",
            "text": "OpenClaw send probe",
            "observed_after": True,
            "source": "screenshots/weixin.png",
            "review_method": "human_visual_review",
        }
    ]

    module.write_review(rows, tmp_path)

    saved = json.loads((tmp_path / "chat_send_visual_review.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "chat_send_visual_review.md").read_text(encoding="utf-8")
    assert saved["reviews"][0]["observed_after"] is True
    assert "| wechat | yes | human_visual_review | OpenClaw send probe |" in markdown


def test_load_actions_accepts_probe_actions_json(tmp_path):
    module = _load_module()
    action_dir = tmp_path / "probe"
    action_dir.mkdir()
    (action_dir / "send_probe_actions.json").write_text(
        json.dumps([{"app": "feishu", "text": "probe", "typed_screenshot": "typed.png"}]),
        encoding="utf-8",
    )

    assert module.load_actions(action_dir)[0]["app"] == "feishu"
