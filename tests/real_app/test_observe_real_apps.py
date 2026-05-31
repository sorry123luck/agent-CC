"""Real-app observe + query validation tests.

Flow per app:
1. resolve_window — find existing windows (never launches)
2. If no window: launch_app (explicit decision) → resolve again
3. observe(hwnd) — generate InteractionCanvas
4. query — find candidates
5. Record quality report

WeChat: only attach to existing window, never launch.

Run with: pytest tests/real_app/test_observe_real_apps.py -v --tb=short
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.real_app.app_helpers import (
    close_app_safely,
    is_chrome_available,
    launch_app,
    resolve_window,
)
from tests.real_app.quality_report import ObserveResult, QueryResult


# ===== Key elements checklist =====

_PROJECT_ROOT = Path(__file__).parents[2]
_CHECKLIST_PATH = _PROJECT_ROOT / "data" / "baselines" / "golden" / "key_elements_checklist.json"


def _load_checklist() -> dict:
    try:
        with open(_CHECKLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ===== Validation helpers =====

def _validate_observe_summary(data: dict, expected_surface_types: list[str]) -> list[str]:
    """Validate observe response summary fields. Returns list of issues."""
    issues = []
    if not data.get("canvas_id"):
        issues.append("canvas_id is empty")
    if data.get("surface_type") not in expected_surface_types:
        issues.append(f"surface_type '{data.get('surface_type')}' not in expected {expected_surface_types}")
    page_class = data.get("page_class", "")
    if not page_class or "/" not in page_class:
        issues.append(f"page_class '{page_class}' is not in app/workflow/state/variant format")
    if data.get("element_count", 0) <= 0:
        issues.append("element_count is 0")
    if data.get("region_count", 0) <= 0:
        issues.append("region_count is 0")
    if not data.get("providers_used"):
        issues.append("providers_used is empty")
    if data.get("canvas_schema_version") != "1.0":
        issues.append(f"canvas_schema_version is '{data.get('canvas_schema_version')}'")
    return issues


def _check_key_elements(client, canvas_id: str, app_name: str, quality_report) -> set[str]:
    """Query for each required semantic_role and record which are found."""
    checklist = _load_checklist()
    app_checklist = checklist.get(app_name, {})
    required_roles = app_checklist.get("required_roles", [])
    found_roles = set()

    for role in required_roles:
        try:
            resp = client.post(
                "/api/v1/query",
                json={
                    "canvas_id": canvas_id,
                    "target": {"semantic_role": role},
                    "max_results": 1,
                    "min_confidence": 0.0,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("total_matched", 0) > 0:
                    found_roles.add(role)
        except Exception as e:
            quality_report.record_query(
                app_name, f"role:{role}", QueryResult(error=str(e))
            )

    quality_report.record_key_elements(app_name, found_roles, required_roles)
    return found_roles


def _run_query(client, canvas_id: str, query_text: str) -> QueryResult:
    """Run a query and return structured result."""
    try:
        resp = client.post(
            "/api/v1/query",
            json={
                "canvas_id": canvas_id,
                "target": {"text": query_text},
                "max_results": 5,
            },
        )
        if resp.status_code != 200:
            return QueryResult(error=f"HTTP {resp.status_code}")
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            top = candidates[0]
            return QueryResult(
                found=True,
                count=data.get("total_matched", 0),
                text_preview=top.get("text", ""),
                top_role=top.get("semantic_role", ""),
            )
        return QueryResult(found=False, count=0)
    except Exception as e:
        return QueryResult(error=str(e))


def _resolve_or_launch(app_name: str, *, allow_launch: bool = True) -> tuple[int | None, str]:
    """Resolve window, optionally launch if not found.

    Returns (hwnd, window_source) where window_source is:
    - "attached_existing": found existing window
    - "launched_new": launched and found window
    - "no_window": no window found (skip test)
    """
    result = resolve_window(app_name)
    if result.candidates:
        return result.candidates[0].hwnd, "attached_existing"

    if not allow_launch:
        return None, "no_window"

    # Launch explicitly, then resolve again
    proc = launch_app(app_name)
    if proc is None:
        return None, "no_window"

    result = resolve_window(app_name, prefer_session=True)
    if result.candidates:
        return result.candidates[0].hwnd, "launched_new"

    return None, "no_window"


# ===== Test cases =====

def test_observe_notepad(client, quality_report):
    """Open Notepad, observe it, query for menu items."""
    hwnd, source = _resolve_or_launch("notepad")
    try:
        if hwnd is None:
            quality_report.record_observe("notepad", ObserveResult(status="skipped", error="Window not found"))
            pytest.skip("Notepad window not found")

        # Observe
        resp = client.post("/api/v1/observe", json={"hwnd": hwnd})
        obs_result = ObserveResult()

        if resp.status_code != 200:
            obs_result.status = "failed"
            obs_result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            quality_report.record_observe("notepad", obs_result)
            pytest.fail(f"Observe failed: {obs_result.error}")

        data = resp.json()
        obs_result.status = "success"
        obs_result.canvas_id = data["canvas_id"]
        obs_result.surface_type = data.get("surface_type", "")
        obs_result.page_class = data.get("page_class", "")
        obs_result.element_count = data.get("element_count", 0)
        obs_result.region_count = data.get("region_count", 0)
        obs_result.providers_used = data.get("providers_used", [])

        issues = _validate_observe_summary(data, ["native_uia", "canvas_self_drawn"])
        if source == "attached_existing":
            issues.append("attached to existing window (not launched by test)")
        for issue in issues:
            obs_result.error = (obs_result.error or "") + f"; {issue}" if obs_result.error else issue
        quality_report.record_observe("notepad", obs_result)

        # Query for expected elements
        canvas_id = data["canvas_id"]
        for query_text in ["文件", "编辑", "格式"]:
            qr = _run_query(client, canvas_id, query_text)
            quality_report.record_query("notepad", query_text, qr)

        # Check key elements from checklist
        _check_key_elements(client, canvas_id, "notepad", quality_report)

        # Assert core expectations
        assert data["canvas_id"] != ""
        assert data["element_count"] > 0
        assert data["region_count"] > 0
        assert len(data["providers_used"]) > 0

    finally:
        close_app_safely("notepad")


def test_observe_chrome(client, quality_report):
    """Open Chrome with temp profile, observe it, query for navigation elements."""
    if not is_chrome_available():
        quality_report.record_observe("chrome", ObserveResult(status="skipped", error="Chrome not installed"))
        pytest.skip("Chrome not found on this system")

    hwnd, source = _resolve_or_launch("chrome")
    try:
        if hwnd is None:
            quality_report.record_observe("chrome", ObserveResult(status="skipped", error="Window not found"))
            pytest.skip("Chrome window not found")

        resp = client.post("/api/v1/observe", json={"hwnd": hwnd})
        obs_result = ObserveResult()

        if resp.status_code != 200:
            obs_result.status = "failed"
            obs_result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            quality_report.record_observe("chrome", obs_result)
            pytest.fail(f"Observe failed: {obs_result.error}")

        data = resp.json()
        obs_result.status = "success"
        obs_result.canvas_id = data["canvas_id"]
        obs_result.surface_type = data.get("surface_type", "")
        obs_result.page_class = data.get("page_class", "")
        obs_result.element_count = data.get("element_count", 0)
        obs_result.region_count = data.get("region_count", 0)
        obs_result.providers_used = data.get("providers_used", [])

        issues = _validate_observe_summary(data, ["browser", "native_uia", "canvas_self_drawn"])
        if source == "attached_existing":
            issues.append("attached to existing window (not launched by test)")
        obs_result.error = "; ".join(issues) if issues else None
        quality_report.record_observe("chrome", obs_result)

        canvas_id = data["canvas_id"]
        for query_text in ["地址", "搜索", "标签"]:
            qr = _run_query(client, canvas_id, query_text)
            quality_report.record_query("chrome", query_text, qr)

        _check_key_elements(client, canvas_id, "chrome", quality_report)

        assert data["canvas_id"] != ""
        assert data["element_count"] > 0

    finally:
        close_app_safely("chrome")


def test_observe_wechat(client, quality_report):
    """Observe WeChat — attach to existing window only, never launch."""
    hwnd, source = _resolve_or_launch("wechat", allow_launch=False)
    if hwnd is None:
        quality_report.record_observe("wechat", ObserveResult(
            status="skipped",
            error="WeChat window not found — must be already running",
        ))
        pytest.skip("WeChat window not found — must be already running")

    resp = client.post("/api/v1/observe", json={"hwnd": hwnd})
    obs_result = ObserveResult()

    if resp.status_code != 200:
        obs_result.status = "failed"
        obs_result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        quality_report.record_observe("wechat", obs_result)
        pytest.fail(f"Observe failed: {obs_result.error}")

    data = resp.json()
    obs_result.status = "success"
    obs_result.canvas_id = data["canvas_id"]
    obs_result.surface_type = data.get("surface_type", "")
    obs_result.page_class = data.get("page_class", "")
    obs_result.element_count = data.get("element_count", 0)
    obs_result.region_count = data.get("region_count", 0)
    obs_result.providers_used = data.get("providers_used", [])

    issues = _validate_observe_summary(data, ["electron_webview", "native_uia", "browser", "canvas_self_drawn"])
    issues.append("attached to existing window (WeChat never auto-launched)")
    obs_result.error = "; ".join(issues) if issues else None
    quality_report.record_observe("wechat", obs_result)

    canvas_id = data["canvas_id"]
    for query_text in ["联系人", "聊天", "搜索"]:
        qr = _run_query(client, canvas_id, query_text)
        quality_report.record_query("wechat", query_text, qr)

    _check_key_elements(client, canvas_id, "wechat", quality_report)

    assert data["canvas_id"] != ""


def test_observe_vscode(client, quality_report):
    """Open VS Code with temp dir, observe it, query for editor elements."""
    hwnd, source = _resolve_or_launch("vscode")
    try:
        if hwnd is None:
            quality_report.record_observe("vscode", ObserveResult(status="skipped", error="Window not found"))
            pytest.skip("VS Code window not found")

        resp = client.post("/api/v1/observe", json={"hwnd": hwnd})
        obs_result = ObserveResult()

        if resp.status_code != 200:
            obs_result.status = "failed"
            obs_result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            quality_report.record_observe("vscode", obs_result)
            pytest.fail(f"Observe failed: {obs_result.error}")

        data = resp.json()
        obs_result.status = "success"
        obs_result.canvas_id = data["canvas_id"]
        obs_result.surface_type = data.get("surface_type", "")
        obs_result.page_class = data.get("page_class", "")
        obs_result.element_count = data.get("element_count", 0)
        obs_result.region_count = data.get("region_count", 0)
        obs_result.providers_used = data.get("providers_used", [])

        issues = _validate_observe_summary(data, ["electron", "native_uia", "canvas_self_drawn"])
        if source == "attached_existing":
            issues.append("attached to existing window (not launched by test)")
        obs_result.error = "; ".join(issues) if issues else None
        quality_report.record_observe("vscode", obs_result)

        canvas_id = data["canvas_id"]
        for query_text in ["文件", "侧边栏", "终端"]:
            qr = _run_query(client, canvas_id, query_text)
            quality_report.record_query("vscode", query_text, qr)

        assert data["canvas_id"] != ""
        assert data["element_count"] > 0

    finally:
        close_app_safely("vscode")
