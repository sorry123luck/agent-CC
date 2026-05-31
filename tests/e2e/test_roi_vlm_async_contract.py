"""E2E contract tests for ROI VLM dry-run endpoint."""

from __future__ import annotations

import time

from src.perception.page_compiler_models import SemanticRole


def test_roi_vlm_dry_run_returns_local_jobs(client, seed_canvas):
    canvas_id, canvas = seed_canvas
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "chat_workspace",
        "rois": [
            {
                "roi_id": "roi_0",
                "bounds": [0, 0, 400, 300],
                "purpose": "message_stream",
                "candidate_ids": ["elem_0"],
                "allowed_outputs": [
                    "region_semantics",
                    "candidate_annotations",
                    "review_only_hints",
                ],
            }
        ],
    }

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": True, "deadline_ms": 2000, "accept_late": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["canvas_id"] == canvas_id
    assert data["status"] == "dry_run"
    assert data["deadline_ms"] == 2000
    assert data["accept_late"] is True
    assert data["jobs"][0]["roi_id"] == "roi_0"
    assert data["jobs"][0]["candidate_ids"] == []
    assert data["jobs"][0]["local_text_candidate_ids"] == ["elem_0"]
    assert data["jobs"][0]["allowed_outputs"] == [
        "region_semantics",
        "candidate_annotations",
        "review_only_hints",
    ]


def test_roi_vlm_dry_run_can_filter_roi_ids(client, seed_canvas):
    canvas_id, canvas = seed_canvas
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "list_management",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
            {"roi_id": "roi_1", "bounds": [100, 0, 200, 100], "candidate_ids": ["elem_1"]},
        ],
    }

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": True, "roi_ids": ["roi_1"]},
    )

    assert resp.status_code == 200
    assert [job["roi_id"] for job in resp.json()["jobs"]] == ["roi_1"]


def test_roi_vlm_non_dry_run_skips_when_no_roi_jobs(client, seed_canvas_with_screenshot):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "unknown_pattern",
        "rois": [],
        "skipped_reason": "no_usable_regions",
    }

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "deadline_ms": 2000, "accept_late": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"
    assert data["message"] == "No ROI VLM jobs selected"


def test_roi_vlm_non_dry_run_merges_provider_result(client, seed_canvas_with_screenshot, monkeypatch):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "chat_workspace",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
        ],
    }

    def fake_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        assert timeout_seconds > 0.35
        assert profile == "fast"
        assert max_candidate_ids == 4
        assert image_max_edge == 384
        assert max_tokens == 384
        return {
            "roi_id": job["roi_id"],
            "candidate_annotations": [{"candidate_id": "elem_0", "label": "send button"}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "roi_ids": ["roi_0"], "deadline_ms": 500, "accept_late": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["results"][0]["status"] == "success"
    processing = client.get(f"/api/v1/canvases/{canvas_id}/processing").json()
    assert processing["processing_state"] == "semantic_ready"
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["candidate_annotations"] == [
        {"candidate_id": "elem_0", "label": "send button"}
    ]

    detail = client.get(f"/api/v1/canvases/{canvas_id}")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["roi_vlm_semantic_supplements"][0]["candidate_annotations"] == [
        {"candidate_id": "elem_0", "label": "send button"}
    ]


def test_roi_vlm_non_dry_run_expands_dense_control_matrix_profile(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "control_matrix",
        "rois": [
            {
                "roi_id": "roi_0",
                "bounds": [0, 0, 100, 100],
                "purpose": "middle_control_columns",
                "candidate_ids": ["elem_0"],
            },
        ],
    }

    def fake_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        assert profile == "fast"
        assert max_candidate_ids == 8
        assert image_max_edge == 384
        assert max_tokens == 384
        return {
            "roi_id": job["roi_id"],
            "candidate_annotations": [{"candidate_id": "elem_0", "label": "gain slider"}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "roi_ids": ["roi_0"], "deadline_ms": 500, "accept_late": True},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["candidate_annotations"] == [
        {"candidate_id": "elem_0", "label": "gain slider"}
    ]


def test_roi_vlm_merge_refreshes_perception_quality_after_candidate_projection(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    for element in canvas.elements:
        element.semantic_role = SemanticRole.UNKNOWN
    canvas.artifacts["perception_quality"] = {
        "unknown_role_ratio": 1.0,
        "warnings": ["unknown_role_heavy"],
    }
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "security_dashboard",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
        ],
    }

    def fake_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        return {
            "roi_id": job["roi_id"],
            "region_semantics": {"role": "security_status"},
            "candidate_annotations": [
                {"candidate_id": "elem_0", "role": "status", "label": "protection_status"},
            ],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "roi_ids": ["roi_0"], "deadline_ms": 500, "accept_late": True},
    )

    assert resp.status_code == 200
    detail = client.get(f"/api/v1/canvases/{canvas_id}").json()
    assert detail["elements"][0]["role_label"] == "protection_status"
    assert detail["elements"][0]["role_source"] == "roi_vlm"
    assert detail["perception_quality"]["unknown_role_ratio"] < 1.0
    assert "unknown_role_heavy" not in detail["perception_quality"]["warnings"]


def test_canvas_detail_exposes_rejected_roi_vlm_responses(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "chat_workspace",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
        ],
    }

    def fake_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        return {
            "roi_id": "wrong_roi",
            "candidate_annotations": [{"candidate_id": "elem_0", "label": "send button"}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "roi_ids": ["roi_0"], "deadline_ms": 2000, "accept_late": True},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "partial"
    detail = client.get(f"/api/v1/canvases/{canvas_id}").json()
    assert detail["roi_vlm_rejected_responses"][0]["roi_id"] == "wrong_roi"
    assert detail["roi_vlm_rejected_responses"][0]["errors"] == ["unknown_roi_id"]


def test_roi_vlm_non_dry_run_returns_before_late_provider_result(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "chat_workspace",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
        ],
    }

    def slow_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        assert timeout_seconds > 0.01
        assert profile == "fast"
        assert max_candidate_ids == 4
        assert image_max_edge == 384
        assert max_tokens == 384
        time.sleep(0.12)
        return {
            "roi_id": job["roi_id"],
            "candidate_annotations": [{"candidate_id": "elem_0", "label": "late send"}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        slow_worker,
        raising=False,
    )

    started = time.perf_counter()
    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "roi_ids": ["roi_0"], "deadline_ms": 10, "accept_late": True},
    )
    elapsed = time.perf_counter() - started

    assert resp.status_code == 200
    assert resp.json()["status"] == "timeout"
    assert elapsed < 0.10
    processing = client.get(f"/api/v1/canvases/{canvas_id}/processing").json()
    assert processing["processing_state"] == "semantic_timeout"

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if canvas.artifacts.get("roi_vlm_semantic_supplements"):
            break
        time.sleep(0.01)
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["status"] == "timeout_late_success"
    processing = client.get(f"/api/v1/canvases/{canvas_id}/processing").json()
    assert processing["processing_state"] == "semantic_late_merged"

    detail = client.get(f"/api/v1/canvases/{canvas_id}")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["roi_vlm_semantic_supplements"][0]["status"] == "timeout_late_success"


def test_roi_vlm_multi_roi_uses_request_level_deadline(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "list_management",
        "rois": [
            {"roi_id": "roi_0", "bounds": [0, 0, 100, 100], "candidate_ids": ["elem_0"]},
            {"roi_id": "roi_1", "bounds": [100, 0, 200, 100], "candidate_ids": ["elem_0"]},
            {"roi_id": "roi_2", "bounds": [200, 0, 300, 100], "candidate_ids": ["elem_0"]},
        ],
    }

    def slow_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        assert timeout_seconds > 0.03
        time.sleep(0.12)
        return {
            "roi_id": job["roi_id"],
            "candidate_annotations": [{"candidate_id": "elem_0", "label": job["roi_id"]}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        slow_worker,
        raising=False,
    )

    started = time.perf_counter()
    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "deadline_ms": 30, "accept_late": True},
    )
    elapsed = time.perf_counter() - started

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "timeout"
    assert [result["status"] for result in data["results"]] == ["timeout", "timeout", "timeout"]
    assert 1 <= data["elapsed_ms"] < 80
    assert data["result_status_counts"] == {"timeout": 3}
    assert elapsed < 0.08

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if len(canvas.artifacts.get("roi_vlm_semantic_supplements") or []) == 3:
            break
        time.sleep(0.01)
    assert len(canvas.artifacts["roi_vlm_semantic_supplements"]) == 3


def test_roi_vlm_multi_roi_uses_per_roi_provider_profile(
    client,
    seed_canvas_with_screenshot,
    monkeypatch,
):
    canvas_id, canvas, _screenshot = seed_canvas_with_screenshot
    canvas.artifacts["roi_selection_plan"] = {
        "mode": "mixed_probe",
        "rois": [
            {
                "roi_id": "roi_dense",
                "bounds": [0, 0, 100, 100],
                "mode": "control_matrix",
                "purpose": "middle_control_columns",
                "candidate_ids": ["elem_0"],
            },
            {
                "roi_id": "roi_generic",
                "bounds": [100, 0, 200, 100],
                "mode": "security_dashboard",
                "purpose": "feature_grid",
                "candidate_ids": ["elem_0"],
            },
        ],
    }
    seen: dict[str, dict[str, int | str]] = {}

    def fake_worker(
        job,
        *,
        screenshot,
        timeout_seconds=None,
        profile=None,
        max_candidate_ids=None,
        image_max_edge=None,
        max_tokens=None,
    ):
        seen[job["roi_id"]] = {
            "profile": profile,
            "max_candidate_ids": max_candidate_ids,
            "image_max_edge": image_max_edge,
            "max_tokens": max_tokens,
        }
        return {
            "roi_id": job["roi_id"],
            "candidate_annotations": [{"candidate_id": "elem_0", "label": job["roi_id"]}],
        }

    monkeypatch.setattr(
        "src.vlm.roi_provider_worker.run_roi_vlm_provider_job",
        fake_worker,
        raising=False,
    )

    resp = client.post(
        f"/api/v1/canvases/{canvas_id}/roi-vlm",
        json={"dry_run": False, "deadline_ms": 500, "accept_late": True},
    )

    assert resp.status_code == 200
    assert seen["roi_dense"] == {
        "profile": "fast",
        "max_candidate_ids": 8,
        "image_max_edge": 384,
        "max_tokens": 384,
    }
    assert seen["roi_generic"] == {
        "profile": "fast",
        "max_candidate_ids": 4,
        "image_max_edge": 320,
        "max_tokens": 256,
    }


def test_roi_vlm_reserves_internal_margin_for_public_deadline():
    from src.integration.api_server import _roi_vlm_worker_timeout_seconds

    assert _roi_vlm_worker_timeout_seconds(2000) == 1.85
    assert _roi_vlm_worker_timeout_seconds(30) == 0.03
