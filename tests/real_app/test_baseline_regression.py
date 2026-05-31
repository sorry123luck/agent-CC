"""Baseline regression tests: export live observe results and compare vs golden.

Directly calls PerceptionService (not API) to get full InteractionCanvas,
exports to data/baselines/new/, then compares against data/baselines/golden/.

Run with: pytest tests/real_app/test_baseline_regression.py -v --tb=short
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from tests.real_app.app_helpers import close_app_safely, launch_app, resolve_window
from tests.real_app.quality_report import ObserveResult


_PROJECT_ROOT = Path(__file__).parents[2]
_GOLDEN_DIR = _PROJECT_ROOT / "data" / "baselines" / "golden"
_NEW_DIR = _PROJECT_ROOT / "data" / "baselines" / "new"


def _export_canvas_to_json(canvas, app_name: str) -> Path:
    """Export an InteractionCanvas to JSON file in data/baselines/new/<app>/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _NEW_DIR / app_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build snapshot data (matching golden format)
    snapshot = {
        "canvas_id": canvas.canvas_id,
        "app": {
            "app_id": canvas.app.app_id,
            "process_name": canvas.app.process_name,
        },
        "window": {
            "hwnd": canvas.window.hwnd,
            "title": canvas.window.title,
        },
        "surface_type": canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
        "page_class": canvas.page_class,
        "stable": canvas.stable,
        "loading": canvas.loading,
        "partial": canvas.partial,
        "element_count": len(canvas.elements),
        "region_count": len(canvas.regions),
        "providers_used": canvas.providers_used,
        "providers_failed": canvas.providers_failed,
        "canvas_schema_version": canvas.canvas_schema_version,
        "elements": [
            {
                "element_id": e.element_id,
                "semantic_role": e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role),
                "control_type": e.control_type,
                "text": e.text,
                "name": e.name,
                "bounds": list(e.bounds) if e.bounds else None,
                "region_id": e.region_id,
                "confidence_level": e.confidence_level.value if hasattr(e.confidence_level, "value") else str(e.confidence_level),
                "risk_level": e.risk_level.value if hasattr(e.risk_level, "value") else str(e.risk_level),
            }
            for e in canvas.elements
        ],
        "regions": [
            {
                "region_id": r.region_id,
                "role": r.role,
                "bounds": list(r.bounds) if r.bounds else None,
                "element_ids": r.element_ids,
            }
            for r in canvas.regions
        ],
    }

    out_path = out_dir / f"{app_name}_{timestamp}_snapshot.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _run_baseline_compare(app_name: str) -> dict:
    """Run baseline comparison between new and golden.

    Returns dict with check results.
    """
    # Find latest golden snapshot
    golden_files = sorted(_GOLDEN_DIR.glob(f"{app_name}_*_snapshot.json"))
    if not golden_files:
        return {"status": "skipped", "reason": "No golden baseline found"}

    golden_path = golden_files[-1]

    # Find latest new snapshot
    new_dir = _NEW_DIR / app_name
    new_files = sorted(new_dir.glob(f"{app_name}_*_snapshot.json"))
    if not new_files:
        return {"status": "skipped", "reason": "No new snapshot exported"}

    new_path = new_files[-1]

    # Load both
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    with open(new_path, encoding="utf-8") as f:
        new = json.load(f)

    checks = {}

    # 1. element_count (10% tolerance, min 3)
    g_count = golden.get("element_count", len(golden.get("elements", [])))
    n_count = new.get("element_count", len(new.get("elements", [])))
    delta = abs(g_count - n_count)
    checks["element_count"] = {
        "golden": g_count,
        "new": n_count,
        "delta": delta,
        "pass": delta <= max(g_count * 0.1, 3),
    }

    # 2. region_count (15% tolerance, min 1)
    g_regions = golden.get("region_count", len(golden.get("regions", [])))
    n_regions = new.get("region_count", len(new.get("regions", [])))
    delta = abs(g_regions - n_regions)
    checks["region_count"] = {
        "golden": g_regions,
        "new": n_regions,
        "delta": delta,
        "pass": delta <= max(g_regions * 0.15, 1),
    }

    # 3. key_elements (required roles must exist)
    checklist_path = _GOLDEN_DIR / "key_elements_checklist.json"
    if checklist_path.exists():
        with open(checklist_path, encoding="utf-8") as f:
            checklist = json.load(f)
        app_checklist = checklist.get(app_name, {})
        required_roles = app_checklist.get("required_roles", [])

        new_elements = new.get("elements", [])
        new_roles = {e.get("semantic_role", "") for e in new_elements}

        missing = [r for r in required_roles if r not in new_roles]
        checks["key_elements"] = {
            "required": required_roles,
            "found": [r for r in required_roles if r in new_roles],
            "missing": missing,
            "pass": len(missing) == 0,
        }

    # 4. provider_health
    g_providers = set(golden.get("providers_used", []))
    n_providers = set(new.get("providers_used", []))
    lost = g_providers - n_providers
    checks["provider_health"] = {
        "golden": sorted(g_providers),
        "new": sorted(n_providers),
        "lost_providers": sorted(lost),
        "pass": len(lost) == 0,
    }

    all_pass = all(c.get("pass", False) for c in checks.values())
    return {"status": "compared", "all_pass": all_pass, "checks": checks}


def _get_canvas_via_perception(hwnd: int):
    """Call PerceptionService directly to get full InteractionCanvas."""
    from src.perception.perception_service import PerceptionService

    perception = PerceptionService()
    zone_page = perception.analyze(hwnd)
    canvas = perception.create_page_snapshot(zone_page)
    return canvas


def _assert_baseline_result(result: dict, quality_report, app_name: str) -> None:
    """Assert baseline comparison result.

    - element_count / region_count drift → hard fail (regression)
    - key_elements missing → record as issue (identification gap, not regression)
    - provider_health lost → hard fail (regression)
    """
    assert result["status"] == "compared", f"Comparison skipped: {result.get('reason')}"

    checks = result.get("checks", {})
    regressions = []

    # Hard fail: element_count or region_count drift
    for check_name in ("element_count", "region_count"):
        check = checks.get(check_name, {})
        if not check.get("pass", True):
            regressions.append(f"{check_name}: golden={check.get('golden')}, new={check.get('new')}, delta={check.get('delta')}")

    # Hard fail: providers lost
    provider_check = checks.get("provider_health", {})
    if not provider_check.get("pass", True):
        regressions.append(f"lost providers: {provider_check.get('lost_providers')}")

    # Soft record: key_elements missing (identification gap, not regression)
    key_check = checks.get("key_elements", {})
    if not key_check.get("pass", True):
        missing = key_check.get("missing", [])
        quality_report.apps[app_name].issues.append(f"Baseline: missing key_elements: {missing}")

    if regressions:
        assert False, f"Baseline regression: {'; '.join(regressions)}"


def _resolve_or_launch(app_name: str, *, allow_launch: bool = True) -> int | None:
    """Resolve window, optionally launch if not found. Returns hwnd or None."""
    result = resolve_window(app_name)
    if result.candidates:
        return result.candidates[0].hwnd

    if not allow_launch:
        return None

    proc = launch_app(app_name)
    if proc is None:
        return None

    result = resolve_window(app_name, prefer_session=True)
    return result.candidates[0].hwnd if result.candidates else None


# ===== Test cases =====

def test_baseline_notepad(quality_report):
    """Export live notepad observe and compare vs golden baseline."""
    hwnd = _resolve_or_launch("notepad")
    try:
        if hwnd is None:
            pytest.skip("Notepad window not found")

        canvas = _get_canvas_via_perception(hwnd)
        _export_canvas_to_json(canvas, "notepad")
        result = _run_baseline_compare("notepad")

        quality_report.record_observe("notepad", ObserveResult(
            status="success",
            surface_type=canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
            page_class=canvas.page_class,
            element_count=len(canvas.elements),
            region_count=len(canvas.regions),
            providers_used=canvas.providers_used,
            canvas_id=canvas.canvas_id,
        ))

        print(f"\nNotepad baseline comparison: {json.dumps(result, indent=2, ensure_ascii=False)}")
        _assert_baseline_result(result, quality_report, "notepad")

    finally:
        close_app_safely("notepad")


def test_baseline_chrome(quality_report):
    """Export live chrome observe and compare vs golden baseline."""
    hwnd = _resolve_or_launch("chrome")
    try:
        if hwnd is None:
            pytest.skip("Chrome window not found")

        canvas = _get_canvas_via_perception(hwnd)
        _export_canvas_to_json(canvas, "chrome")
        result = _run_baseline_compare("chrome")

        quality_report.record_observe("chrome", ObserveResult(
            status="success",
            surface_type=canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
            page_class=canvas.page_class,
            element_count=len(canvas.elements),
            region_count=len(canvas.regions),
            providers_used=canvas.providers_used,
            canvas_id=canvas.canvas_id,
        ))

        print(f"\nChrome baseline comparison: {json.dumps(result, indent=2, ensure_ascii=False)}")
        _assert_baseline_result(result, quality_report, "chrome")

    finally:
        close_app_safely("chrome")


def test_baseline_wechat(quality_report):
    """Export live wechat observe and compare vs golden baseline.
    WeChat must already be running — never auto-launched.
    """
    hwnd = _resolve_or_launch("wechat", allow_launch=False)
    if hwnd is None:
        pytest.skip("WeChat window not found — must be already running")

    canvas = _get_canvas_via_perception(hwnd)
    _export_canvas_to_json(canvas, "wechat")
    result = _run_baseline_compare("wechat")

    quality_report.record_observe("wechat", ObserveResult(
        status="success",
        surface_type=canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
        page_class=canvas.page_class,
        element_count=len(canvas.elements),
        region_count=len(canvas.regions),
        providers_used=canvas.providers_used,
        canvas_id=canvas.canvas_id,
    ))

    print(f"\nWeChat baseline comparison: {json.dumps(result, indent=2, ensure_ascii=False)}")
    _assert_baseline_result(result, quality_report, "wechat")


def test_baseline_vscode(quality_report):
    """Export live vscode observe and compare vs golden baseline."""
    hwnd = _resolve_or_launch("vscode")
    try:
        if hwnd is None:
            pytest.skip("VS Code window not found")

        canvas = _get_canvas_via_perception(hwnd)
        _export_canvas_to_json(canvas, "vscode")
        result = _run_baseline_compare("vscode")

        quality_report.record_observe("vscode", ObserveResult(
            status="success",
            surface_type=canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
            page_class=canvas.page_class,
            element_count=len(canvas.elements),
            region_count=len(canvas.regions),
            providers_used=canvas.providers_used,
            canvas_id=canvas.canvas_id,
        ))

        print(f"\nVS Code baseline comparison: {json.dumps(result, indent=2, ensure_ascii=False)}")
        _assert_baseline_result(result, quality_report, "vscode")

    finally:
        close_app_safely("vscode")
