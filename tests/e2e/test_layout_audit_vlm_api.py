"""E2E tests for layout-audit VLM API."""

from __future__ import annotations

import time


def _mark_layout_audit_needed(canvas):
    canvas.artifacts["screenshot_size"] = [800, 600]
    canvas.artifacts["visual_pattern"] = {"mode": "unknown_pattern"}
    canvas.artifacts["perception_quality"] = {
        "usable_state": "unreliable",
        "warnings": ["sparse_elements", "coarse_regions", "layout_audit_needed"],
    }
    canvas.artifacts["geometric_regions"] = []


def test_layout_audit_vlm_dry_run_returns_review_only_job(client, seed_canvas_with_screenshot):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    _mark_layout_audit_needed(canvas)

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/layout-audit-vlm",
        json={"dry_run": True, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "dry_run"
    assert data["job"]["review_only"] is True
    assert data["job"]["allowed_outputs"] == ["layout_regions", "layout_summary", "review_only_hints"]


def test_layout_audit_vlm_non_dry_run_merges_review_only_result(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    _mark_layout_audit_needed(canvas)

    def fake_worker(job, *, screenshot, timeout_seconds=None, image_max_edge=None, max_tokens=None):
        assert timeout_seconds > 1.85
        assert image_max_edge == 512
        assert max_tokens == 384
        return {
            "layout_summary": "single unknown app surface",
            "layout_regions": [
                {
                    "role": "main_content",
                    "relative_bounds": [0.0, 0.0, 1.0, 1.0],
                    "confidence": 0.72,
                }
            ],
            "review_only_hints": ["run OCR/Omni for controls"],
        }

    monkeypatch.setattr(
        "src.vlm.layout_audit_provider_worker.run_layout_audit_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/layout-audit-vlm",
        json={"dry_run": False, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["status"] == "success"
    assert canvas.artifacts["vlm_layout_audit"]["review_only"] is True
    assert canvas.artifacts["vlm_layout_audit"]["layout_regions"][0]["role"] == "main_content"
    assert canvas.artifacts["roi_selection_plan"]["rois"][0]["source"] == "vlm_layout_audit"


def test_layout_audit_vlm_returns_before_late_provider_result(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    _mark_layout_audit_needed(canvas)

    def slow_worker(job, *, screenshot, timeout_seconds=None, image_max_edge=None, max_tokens=None):
        assert timeout_seconds > 0.03
        time.sleep(0.12)
        return {
            "layout_summary": "late layout",
            "layout_regions": [
                {"role": "main_content", "relative_bounds": [0.0, 0.0, 1.0, 1.0], "confidence": 0.7}
            ],
        }

    monkeypatch.setattr(
        "src.vlm.layout_audit_provider_worker.run_layout_audit_provider_job",
        slow_worker,
        raising=False,
    )

    started = time.perf_counter()
    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/layout-audit-vlm",
        json={"dry_run": False, "deadline_ms": 30, "accept_late": True},
    )
    elapsed = time.perf_counter() - started

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "timeout"
    assert data["result"]["late_response_policy"] == "accept_if_review_only"
    assert elapsed < 0.08

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if canvas.artifacts.get("vlm_layout_audit", {}).get("status") == "timeout_late_success":
            break
        time.sleep(0.01)
    assert canvas.artifacts["vlm_layout_audit"]["status"] == "timeout_late_success"
