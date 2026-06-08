"""Tests for real-app input safety sample export helpers."""

from __future__ import annotations

import json

from tests.real_app.input_safety_export import (
    CHAT_INPUT_MATRIX_APPS,
    export_observed_input_safety_sample,
    write_real_app_input_safety_report,
)


class _FakeClient:
    def __init__(self, detail: dict):
        self.detail = detail

    def get(self, path: str):
        assert path == "/api/v1/canvases/canvas_1"

        class _Response:
            status_code = 200

            def json(inner_self):
                return self.detail

        return _Response()


def _canvas_detail() -> dict:
    return {
        "canvas_id": "canvas_1",
        "app_id": "wechat",
        "page_class": "wechat/chat/main",
        "window_title": "wechat chat",
        "surface_type": "native_uia",
        "providers_used": ["uia"],
        "elements": [
            {
                "element_id": "msg_input",
                "region_id": "bottom_input",
                "semantic_role": "message_input",
                "control_type": "Edit",
                "bounds": [80, 520, 600, 580],
                "text": "",
                "interactable": True,
                "state": {"enabled": True, "visible": True},
            },
            {
                "element_id": "send_btn",
                "region_id": "bottom_input",
                "semantic_role": "send_button",
                "control_type": "Button",
                "bounds": [620, 520, 700, 580],
                "text": "send",
                "interactable": False,
                "state": {"enabled": False, "visible": True},
            },
        ],
    }


def test_export_observed_input_safety_sample_writes_report_compatible_snapshot(tmp_path):
    out_path = export_observed_input_safety_sample(
        _FakeClient(_canvas_detail()),
        canvas_id="canvas_1",
        app_name="wechat",
        output_root=tmp_path,
        run_id="20260602_120000",
    )

    assert out_path == tmp_path / "wechat" / "wechat_20260602_120000_canvas.json"
    snapshot = json.loads(out_path.read_text(encoding="utf-8"))
    assert snapshot["app"] == {"app_id": "wechat", "process_name": "wechat"}
    assert snapshot["page"] == {"page_class": "wechat/chat/main"}
    assert snapshot["elements"][0]["semantic_role"] == "message_input"
    assert snapshot["elements"][1]["state"]["enabled"] is False


def test_write_real_app_input_safety_report_uses_chat_matrix(tmp_path):
    sample_root = tmp_path / "samples"
    export_observed_input_safety_sample(
        _FakeClient(_canvas_detail()),
        canvas_id="canvas_1",
        app_name="wechat",
        output_root=sample_root,
        run_id="20260602_120000",
    )

    report, report_path = write_real_app_input_safety_report(
        sample_root,
        tmp_path / "reports",
        run_id="20260602_120000",
    )

    assert CHAT_INPUT_MATRIX_APPS == ["wechat", "qq", "feishu"]
    assert report_path == tmp_path / "reports" / "input_safety" / "20260602_120000" / "input_safety_report.json"
    assert report["matrix"]["covered_apps"] == ["wechat"]
    assert report["matrix"]["missing_apps"] == ["qq", "feishu"]
    assert report["samples"][0]["status"] == "passed"
