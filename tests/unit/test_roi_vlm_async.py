"""Tests for async ROI VLM supplement runner."""

from __future__ import annotations

import time

from src.vlm.roi_async import RoiVlmAsyncRunner
from src.vlm.roi_contract import build_roi_vlm_jobs


def _canvas():
    class Canvas:
        canvas_id = "canvas_1"

        def __init__(self):
            self.elements = [type("Element", (), {"element_id": "elem_1"})()]
            self.artifacts = {
                "roi_selection_plan": {
                    "mode": "chat_workspace",
                    "rois": [
                        {
                            "roi_id": "roi_0",
                            "bounds": [0, 0, 400, 300],
                            "purpose": "message_stream",
                            "candidate_ids": ["elem_1"],
                        }
                    ],
                }
            }

    return Canvas()


def test_async_runner_merges_fast_response_immediately():
    canvas = _canvas()
    job = build_roi_vlm_jobs(canvas)[0]
    runner = RoiVlmAsyncRunner(timeout_seconds=0.5)

    result = runner.submit(canvas, job, lambda _job: {
        "roi_id": "roi_0",
        "candidate_annotations": [{"candidate_id": "elem_1", "label": "message"}],
    })

    assert result["status"] == "success"
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["status"] == "success"
    runner.shutdown()


def test_async_runner_accepts_late_response_after_timeout():
    canvas = _canvas()
    job = build_roi_vlm_jobs(canvas)[0]
    persisted = []
    runner = RoiVlmAsyncRunner(
        timeout_seconds=0.01,
        on_late_result=lambda late_canvas, late_job, late_result: persisted.append(
            (late_canvas.canvas_id, late_job["roi_id"], late_result["status"])
        ),
    )

    def slow_worker(_job):
        time.sleep(0.05)
        return {
            "roi_id": "roi_0",
            "candidate_annotations": [{"candidate_id": "elem_1", "label": "late message"}],
        }

    result = runner.submit(canvas, job, slow_worker)
    assert result["status"] == "timeout"
    assert canvas.artifacts["roi_vlm_timeouts"][0]["discard_late_response"] is False

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if canvas.artifacts.get("roi_vlm_semantic_supplements"):
            break
        time.sleep(0.01)

    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["status"] == "timeout_late_success"
    assert canvas.artifacts["roi_vlm_semantic_supplements"][0]["late_result_accepted"] is True
    assert persisted == [("canvas_1", "roi_0", "timeout_late_success")]
    runner.shutdown()


def test_async_runner_records_late_failure_after_timeout():
    canvas = _canvas()
    job = build_roi_vlm_jobs(canvas)[0]
    persisted = []
    runner = RoiVlmAsyncRunner(
        timeout_seconds=0.01,
        on_late_result=lambda late_canvas, late_job, late_result: persisted.append(
            (late_canvas.canvas_id, late_job["roi_id"], late_result["status"])
        ),
    )

    def slow_worker(_job):
        time.sleep(0.05)
        raise RuntimeError("provider failed late")

    result = runner.submit(canvas, job, slow_worker)
    assert result["status"] == "timeout"

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if canvas.artifacts.get("roi_vlm_late_failures"):
            break
        time.sleep(0.01)

    late_failure = canvas.artifacts["roi_vlm_late_failures"][0]
    assert late_failure["roi_id"] == "roi_0"
    assert late_failure["status"] == "timeout_late_failed"
    assert "provider failed late" in late_failure["error"]
    assert persisted == [("canvas_1", "roi_0", "timeout_late_failed")]
    runner.shutdown()
