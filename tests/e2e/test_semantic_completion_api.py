"""E2E tests for semantic completion orchestration."""

from __future__ import annotations


def _mark_layout_audit_needed(canvas):
    canvas.artifacts["screenshot_size"] = [800, 600]
    canvas.artifacts["visual_pattern"] = {"mode": "unknown_pattern"}
    canvas.artifacts["perception_quality"] = {
        "usable_state": "unreliable",
        "warnings": ["sparse_elements", "coarse_regions", "layout_audit_needed"],
    }
    canvas.artifacts["geometric_regions"] = []


def test_semantic_completion_dry_run_plans_layout_then_roi(
    client,
    seed_canvas_with_screenshot,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    _mark_layout_audit_needed(canvas)

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/semantic-completion",
        json={"dry_run": True, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "dry_run"
    assert [stage["name"] for stage in data["stages"]] == ["layout_audit", "roi_vlm"]
    assert data["layout_audit"]["status"] == "dry_run"
    assert data["roi_vlm"]["status"] == "dry_run"


def test_semantic_completion_runs_layout_audit_before_roi(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    _mark_layout_audit_needed(canvas)
    calls = []

    def fake_layout_worker(job, *, screenshot, timeout_seconds=None, image_max_edge=None, max_tokens=None):
        calls.append(("layout", tuple(job["allowed_outputs"])))
        return {
            "layout_summary": "unknown self-drawn page",
            "layout_regions": [
                {
                    "role": "main_content",
                    "relative_bounds": [0.0, 0.0, 1.0, 1.0],
                    "confidence": 0.72,
                }
            ],
        }

    def fake_roi_worker(job, *, screenshot, timeout_seconds=None, image_max_edge=None, max_tokens=None, **_kwargs):
        calls.append(("roi", job["roi_id"], job["purpose"]))
        return {
            "roi_id": job["roi_id"],
            "region_semantics": {"role": "main_content", "summary": "semantic supplement"},
            "candidate_annotations": [],
            "review_only_hints": ["local controls still need confirmation"],
        }

    monkeypatch.setattr(
        "src.vlm.layout_audit_provider_worker.run_layout_audit_provider_job",
        fake_layout_worker,
        raising=False,
    )
    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_roi_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/semantic-completion",
        json={"dry_run": False, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert [stage["name"] for stage in data["stages"]] == ["layout_audit", "roi_vlm"]
    assert calls[0][0] == "layout"
    assert calls[1][0] == "roi"
    assert canvas.artifacts["roi_selection_plan"]["rois"][0]["source"] == "vlm_layout_audit"
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["region_semantics"]["role"] == "main_content"


def test_semantic_completion_blocks_when_screenshot_is_missing(
    client,
    seed_canvas,
):
    canvas_id, canvas = seed_canvas
    canvas.artifacts["perception_quality"] = {
        "usable_state": "unreliable",
        "warnings": ["layout_audit_needed", "layout_audit_blocked_no_screenshot"],
    }

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/semantic-completion",
        json={"dry_run": False, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "blocked"
    assert data["next_action"] == "recapture_with_screenshot"


def test_semantic_completion_reports_skipped_when_no_roi_jobs(
    client,
    seed_canvas_with_screenshot,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["perception_quality"] = {
        "usable_state": "usable_with_warnings",
        "warnings": [],
    }
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "unknown_pattern",
        "rois": [],
        "skipped_reason": "no_usable_regions",
    }

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/semantic-completion",
        json={"dry_run": False, "deadline_ms": 2000},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"
    assert data["next_action"] == "inspect_roi_selection_plan"
