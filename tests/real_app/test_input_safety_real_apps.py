"""Real-app chat input safety sampling.

These tests attach to already-running chat apps only. They never type text,
click send, or launch messaging clients.
"""

from __future__ import annotations

import pytest

from tests.real_app.app_helpers import resolve_window
from tests.real_app.input_safety_export import export_observed_input_safety_sample
from tests.real_app.quality_report import ObserveResult


CHAT_INPUT_APPS = [
    ("wechat", ["electron_webview", "native_uia", "browser", "canvas_self_drawn"]),
    ("qq", ["native_uia", "electron_webview", "browser", "canvas_self_drawn"]),
    ("feishu", ["electron_webview", "native_uia", "browser", "canvas_self_drawn"]),
]


@pytest.mark.parametrize(("app_name", "expected_surface_types"), CHAT_INPUT_APPS)
def test_chat_input_safety_sample_export(client, quality_report, app_name, expected_surface_types):
    result = resolve_window(app_name, prefer_session=False)
    if not result.candidates:
        quality_report.record_observe(
            app_name,
            ObserveResult(status="skipped", error=f"{app_name} window not found; attach-only input safety sample"),
        )
        pytest.skip(f"{app_name} window not found; attach-only input safety sample")

    hwnd = result.candidates[0].hwnd
    resp = client.post("/api/v1/observe", json={"hwnd": hwnd})
    obs_result = ObserveResult()

    if resp.status_code != 200:
        obs_result.status = "failed"
        obs_result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        quality_report.record_observe(app_name, obs_result)
        pytest.fail(f"Observe failed: {obs_result.error}")

    data = resp.json()
    obs_result.status = "success"
    obs_result.canvas_id = data["canvas_id"]
    obs_result.surface_type = data.get("surface_type", "")
    obs_result.page_class = data.get("page_class", "")
    obs_result.element_count = data.get("element_count", 0)
    obs_result.region_count = data.get("region_count", 0)
    obs_result.providers_used = data.get("providers_used", [])

    issues = ["attached_existing_chat_app", "input_actions_not_executed"]
    if data.get("surface_type") not in expected_surface_types:
        issues.append(f"surface_type '{data.get('surface_type')}' not in expected {expected_surface_types}")
    obs_result.error = "; ".join(issues)
    quality_report.record_observe(app_name, obs_result)

    export_observed_input_safety_sample(client, canvas_id=data["canvas_id"], app_name=app_name)

    assert data["canvas_id"] != ""
    assert data.get("element_count", 0) > 0
