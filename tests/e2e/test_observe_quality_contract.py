"""E2E contract tests for perception quality / visual pattern / ROI outputs."""

from __future__ import annotations

from tests.e2e.conftest import _make_test_canvas


def _quality_canvas(hwnd: int):
    from src.canvas.canvas_cache import get_canvas_cache

    canvas = _make_test_canvas(hwnd=hwnd)
    canvas.artifacts["perception_quality"] = {
        "usable_state": "usable",
        "mode_guess": "list_management",
        "element_count": 3,
    }
    canvas.artifacts["visual_pattern"] = {
        "mode": "list_management",
        "confidence": 0.82,
        "evidence": ["dense_row_repetition"],
    }
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "list_management",
        "rois": [
            {
                "roi_id": "roi_primary_list",
                "reason": "table/list candidate region",
                "bounds": [10, 20, 700, 520],
                "element_ids": ["elem_0", "elem_1"],
            }
        ],
    }
    get_canvas_cache().put(canvas)
    return canvas


def test_observe_response_returns_quality_pattern_and_roi_values(client, monkeypatch):
    """Observe response exposes local quality, visual mode, and ROI plan fields."""
    def fake_observe(*args, **kwargs):
        return _quality_canvas(hwnd=args[0]), "new_page", {
            "used": False,
            "status": "skipped",
            "provider": "",
        }

    monkeypatch.setattr("src.integration.api_server._do_observe", fake_observe)
    monkeypatch.setattr(
        "src.windows.screenshot_service.ScreenshotService.normalize_hwnd",
        lambda hwnd: hwnd,
    )
    monkeypatch.setattr(
        "src.windows.screenshot_service.ScreenshotService.is_minimized",
        lambda hwnd: False,
    )

    resp = client.post("/api/v1/observe", json={"hwnd": 12345, "async_enhance": False})

    assert resp.status_code == 200
    data = resp.json()
    assert data["perception_quality"]["usable_state"] == "usable"
    assert data["perception_quality"]["mode_guess"] == "list_management"
    assert data["visual_pattern"]["mode"] == "list_management"
    assert data["visual_pattern"]["confidence"] == 0.82
    assert data["roi_selection_plan"]["mode"] == "list_management"
    assert data["roi_selection_plan"]["rois"][0]["roi_id"] == "roi_primary_list"
    assert data["roi_selection_plan"]["rois"][0]["bounds"] == [10, 20, 700, 520]


def test_canvas_detail_returns_cached_quality_pattern_and_roi_values(client):
    """Canvas detail exposes the same diagnostics for workbench and agents."""
    canvas = _quality_canvas(hwnd=12345)

    resp = client.get(f"/api/v1/canvases/{canvas.canvas_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["perception_quality"]["usable_state"] == "usable"
    assert data["perception_quality"]["mode_guess"] == "list_management"
    assert data["visual_pattern"]["mode"] == "list_management"
    assert data["visual_pattern"]["evidence"] == ["dense_row_repetition"]
    assert data["roi_selection_plan"]["mode"] == "list_management"
    assert data["roi_selection_plan"]["rois"][0]["element_ids"] == ["elem_0", "elem_1"]
