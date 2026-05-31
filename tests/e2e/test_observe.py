"""E2E tests for POST /api/v1/observe."""

from __future__ import annotations


def test_observe_returns_canvas(client, mock_observe):
    """Observe returns a valid InteractionCanvas summary."""
    mock_fn, get_canvas = mock_observe
    resp = client.post("/api/v1/observe", json={"hwnd": 12345})
    assert resp.status_code == 200
    data = resp.json()
    canvas = get_canvas()
    assert data["canvas_id"] == canvas.canvas_id
    assert data["app_id"] == "test_app"
    assert data["page_class"] == "test_app/main/default"
    assert data["surface_type"] == "native_uia"
    assert data["element_count"] == 3
    assert data["region_count"] == 1
    assert data["canvas_schema_version"] == "1.0"
    assert "uia" in data["providers_used"]
    assert data["processing_state"] == "local_ready"
    assert isinstance(data["elapsed_ms"], int)
    assert data["elapsed_ms"] >= 0


def test_observe_default_foreground(client, mock_observe):
    """Observe without hwnd uses foreground window (mocked)."""
    mock_fn, get_canvas = mock_observe
    resp = client.post("/api/v1/observe", json={})
    assert resp.status_code == 200
    canvas = get_canvas()
    assert resp.json()["canvas_id"] == canvas.canvas_id


def test_observe_passes_hwnd(client, mock_observe):
    """Observe passes the hwnd to perception service."""
    mock_fn, get_canvas = mock_observe
    client.post("/api/v1/observe", json={"hwnd": 99999})
    mock_fn.assert_called_once_with(
        99999,
        allow_vlm=False,
        force_vlm=False,
        run_enhancement_phases=False,
        fast_perception=True,
    )
    canvas = get_canvas()
    assert canvas.window.hwnd == 99999


def test_observe_with_vlm_flags(client, mock_observe):
    """Observe accepts allow_vlm and force_vlm flags."""
    mock_fn, _ = mock_observe
    resp = client.post("/api/v1/observe", json={"hwnd": 12345, "allow_vlm": True, "force_vlm": False})
    assert resp.status_code == 200


def test_observe_force_vlm_flag_passed(client, mock_observe):
    """force_vlm starts async enhancement while local observe stays fast."""
    mock_fn, _ = mock_observe
    resp = client.post("/api/v1/observe", json={"hwnd": 12345, "force_vlm": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["processing_state"] in {"local_ready", "vlm_queued", "vlm_running", "failed"}
    assert data["enhancement_job_id"]
    mock_fn.assert_called_once_with(
        12345,
        allow_vlm=False,
        force_vlm=False,
        run_enhancement_phases=False,
        fast_perception=True,
    )


def test_processing_and_job_endpoints_return_persisted_state(client, mock_observe):
    """Observe enhancement jobs are visible through processing/job APIs."""
    resp = client.post("/api/v1/observe", json={"hwnd": 12345, "force_vlm": True})
    assert resp.status_code == 200
    data = resp.json()
    canvas_id = data["canvas_id"]
    job_id = data["enhancement_job_id"]

    processing = client.get(f"/api/v1/canvases/{canvas_id}/processing")
    assert processing.status_code == 200
    assert processing.json()["canvas_id"] == canvas_id
    assert job_id in processing.json()["jobs"]

    job = client.get(f"/api/v1/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["job_id"] == job_id
    assert job.json()["canvas_id"] == canvas_id

    jobs = client.get(f"/api/v1/jobs?canvas_id={canvas_id}")
    assert jobs.status_code == 200
    assert any(item["job_id"] == job_id for item in jobs.json())


def test_job_cancel_endpoint(client, mock_observe):
    """Queued/running jobs can be marked cancelled."""
    resp = client.post("/api/v1/observe", json={"hwnd": 12345, "force_vlm": True})
    assert resp.status_code == 200
    job_id = resp.json()["enhancement_job_id"]

    cancel = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["job_id"] == job_id
    assert cancel.json()["state"] == "cancelled"


def test_background_enhancement_tolerates_duplicate_cleanup_failure(seed_canvas, mock_observe, monkeypatch):
    """Duplicate-canvas cleanup failure must not fail the enhancement job."""
    from src.canvas.canvas_cache import get_canvas_cache
    from src.integration.api_server import _run_canvas_enhancement_sync

    canvas_id, canvas = seed_canvas
    canvas.artifacts["source_hwnd"] = 12345
    get_canvas_cache().put(canvas)

    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr("src.integration.api_server._delete_canvas_db_records", fail_cleanup)

    result = _run_canvas_enhancement_sync(canvas_id, force=False, modes=[])

    assert result.status == "skipped"
    assert get_canvas_cache().get(canvas_id) is not None


def test_background_enhancement_uses_lightweight_vision_for_sparse_collaboration(
    seed_canvas,
    monkeypatch,
):
    """Sparse collaboration inboxes should avoid deep UIA while still running vision."""
    from src.canvas.canvas_cache import get_canvas_cache
    from src.integration.api_server import _run_canvas_enhancement_sync

    canvas_id, canvas = seed_canvas
    canvas.artifacts["source_hwnd"] = 12345
    canvas.artifacts["visual_pattern"] = {"mode": "collaboration_inbox"}
    canvas.artifacts["perception_quality"] = {"warnings": ["sparse_elements", "coarse_regions"]}
    get_canvas_cache().put(canvas)
    observed_kwargs = {}

    def lightweight_observe(hwnd, **kwargs):
        observed_kwargs.update(kwargs)
        return canvas, "reused", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.integration.api_server._do_observe", lightweight_observe)

    result = _run_canvas_enhancement_sync(canvas_id, force=False, modes=[])

    assert result.status == "skipped"
    assert observed_kwargs["lightweight_uia"] is True
    assert observed_kwargs["fast_perception"] is False


def test_background_enhancement_preserves_roi_vlm_artifacts(seed_canvas, monkeypatch):
    """Concurrent semantic completion diagnostics must survive full-canvas replacement."""
    import copy

    from src.canvas.canvas_cache import get_canvas_cache
    from src.integration.api_server import _run_canvas_enhancement_sync

    canvas_id, canvas = seed_canvas
    canvas.artifacts["source_hwnd"] = 12345
    canvas.artifacts["roi_vlm_timeouts"] = [{"roi_id": "roi_0"}]
    canvas.artifacts["roi_vlm_semantic_supplements"] = [{"roi_id": "roi_0", "status": "timeout_late_success"}]
    get_canvas_cache().put(canvas)

    def full_observe(hwnd, **kwargs):
        fresh_canvas = copy.deepcopy(canvas)
        fresh_canvas.canvas_id = "duplicate_canvas"
        fresh_canvas.artifacts = {"source_hwnd": hwnd}
        return fresh_canvas, "new_page", {"used": False, "status": "skipped", "provider": ""}

    monkeypatch.setattr("src.integration.api_server._do_observe", full_observe)

    result = _run_canvas_enhancement_sync(canvas_id, force=False, modes=[])
    updated = get_canvas_cache().get(canvas_id)

    assert result.status == "skipped"
    assert updated is not None
    assert updated.artifacts["roi_vlm_timeouts"] == [{"roi_id": "roi_0"}]
    assert updated.artifacts["roi_vlm_semantic_supplements"] == [
        {"roi_id": "roi_0", "status": "timeout_late_success"}
    ]


def test_observe_increases_cache(client, mock_observe):
    """After observe, canvas cache should have one entry."""
    from src.canvas.canvas_cache import get_canvas_cache

    cache = get_canvas_cache()
    before = cache.size()
    client.post("/api/v1/observe", json={"hwnd": 12345})
    assert cache.size() == before + 1
