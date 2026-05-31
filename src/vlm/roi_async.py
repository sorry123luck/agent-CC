"""Async skeleton for ROI-scoped VLM semantic supplement jobs."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Callable

from src.vlm.roi_contract import (
    merge_roi_vlm_response,
    record_roi_vlm_late_failure,
    record_roi_vlm_timeout,
)


logger = logging.getLogger(__name__)
RoiVlmWorker = Callable[[dict[str, Any]], dict[str, Any]]
RoiVlmLateResultCallback = Callable[[Any, dict[str, Any], dict[str, Any]], None]


class RoiVlmAsyncRunner:
    """Run one ROI VLM job with a fast timeout and late-result merge callback."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        executor: ThreadPoolExecutor | None = None,
        on_late_result: RoiVlmLateResultCallback | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self._on_late_result = on_late_result
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="openclaw-roi-vlm",
        )

    def submit(
        self,
        canvas: Any,
        job: dict[str, Any],
        worker: RoiVlmWorker,
    ) -> dict[str, Any]:
        """Submit a worker and wait only up to timeout_seconds for initial result."""
        future = self._executor.submit(worker, dict(job))
        try:
            response = future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            record_roi_vlm_timeout(canvas, job, timeout_seconds=self.timeout_seconds)
            future.add_done_callback(
                lambda done: self._merge_late_future(canvas, job, done),
            )
            return {
                "status": "timeout",
                "canvas_id": str(getattr(canvas, "canvas_id", "")),
                "roi_id": str(job.get("roi_id") or ""),
                "late_response_policy": "accept_if_local_ids_match",
            }
        except Exception as exc:  # noqa: BLE001 - background diagnostics.
            logger.warning("ROI VLM worker failed: %s", exc, exc_info=True)
            return {
                "status": "failed",
                "canvas_id": str(getattr(canvas, "canvas_id", "")),
                "roi_id": str(job.get("roi_id") or ""),
                "error": str(exc),
            }

        validation = merge_roi_vlm_response(canvas, response, status="success")
        return {
            "status": "success" if validation.accepted else "rejected",
            "canvas_id": str(getattr(canvas, "canvas_id", "")),
            "roi_id": validation.roi_id or str(job.get("roi_id") or ""),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        }

    def submit_many(
        self,
        canvas: Any,
        jobs: list[dict[str, Any]],
        worker: RoiVlmWorker,
    ) -> list[dict[str, Any]]:
        """Submit all ROI workers immediately and apply one shared timeout.

        `timeout_seconds` is the interactive response budget for the batch, not
        a per-ROI timeout. Late responses are still accepted through callbacks.
        """
        futures = [
            (dict(job), self._executor.submit(worker, dict(job)))
            for job in jobs
        ]
        deadline = time.monotonic() + self.timeout_seconds
        results: list[dict[str, Any]] = []

        for job, future in futures:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    response = future.result(timeout=remaining)
                    results.append(self._merge_initial_response(canvas, job, response))
                    continue
                except TimeoutError:
                    pass
                except Exception as exc:  # noqa: BLE001 - background diagnostics.
                    logger.warning("ROI VLM worker failed: %s", exc, exc_info=True)
                    results.append(self._failed_result(canvas, job, str(exc)))
                    continue

            if future.done():
                try:
                    response = future.result(timeout=0)
                    results.append(self._merge_initial_response(canvas, job, response))
                except Exception as exc:  # noqa: BLE001 - background diagnostics.
                    logger.warning("ROI VLM worker failed: %s", exc, exc_info=True)
                    results.append(self._failed_result(canvas, job, str(exc)))
                continue

            record_roi_vlm_timeout(canvas, job, timeout_seconds=self.timeout_seconds)
            future.add_done_callback(
                lambda done, late_job=job: self._merge_late_future(canvas, late_job, done),
            )
            results.append(self._timeout_result(canvas, job))

        return results

    def shutdown(self, *, wait: bool = True) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait)

    def _merge_initial_response(
        self,
        canvas: Any,
        job: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        validation = merge_roi_vlm_response(canvas, response, status="success")
        return {
            "status": "success" if validation.accepted else "rejected",
            "canvas_id": str(getattr(canvas, "canvas_id", "")),
            "roi_id": validation.roi_id or str(job.get("roi_id") or ""),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        }

    def _failed_result(
        self,
        canvas: Any,
        job: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "canvas_id": str(getattr(canvas, "canvas_id", "")),
            "roi_id": str(job.get("roi_id") or ""),
            "error": error,
        }

    def _timeout_result(
        self,
        canvas: Any,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "timeout",
            "canvas_id": str(getattr(canvas, "canvas_id", "")),
            "roi_id": str(job.get("roi_id") or ""),
            "late_response_policy": "accept_if_local_ids_match",
        }

    def _merge_late_future(
        self,
        canvas: Any,
        job: dict[str, Any],
        future: Future,
    ) -> None:
        try:
            response = future.result()
            validation = merge_roi_vlm_response(
                canvas,
                response,
                status="timeout_late_success",
            )
            status = "timeout_late_success" if validation.accepted else "timeout_late_rejected"
            if not validation.accepted:
                logger.info(
                    "Late ROI VLM response rejected for canvas=%s roi=%s errors=%s",
                    getattr(canvas, "canvas_id", ""),
                    job.get("roi_id"),
                    validation.errors,
                )
            self._notify_late_result(canvas, job, {
                "status": status,
                "roi_id": validation.roi_id or str(job.get("roi_id") or ""),
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            })
        except Exception as exc:  # noqa: BLE001 - callback diagnostics only.
            record_roi_vlm_late_failure(
                canvas,
                job,
                error=str(exc),
                status="timeout_late_failed",
            )
            self._notify_late_result(canvas, job, {
                "status": "timeout_late_failed",
                "roi_id": str(job.get("roi_id") or ""),
                "error": str(exc),
            })
            logger.warning(
                "Late ROI VLM worker failed for canvas=%s roi=%s: %s",
                getattr(canvas, "canvas_id", ""),
                job.get("roi_id"),
                exc,
                exc_info=True,
            )

    def _notify_late_result(
        self,
        canvas: Any,
        job: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if self._on_late_result is None:
            return
        try:
            self._on_late_result(canvas, job, result)
        except Exception as exc:  # noqa: BLE001 - diagnostics only.
            logger.warning(
                "ROI VLM late-result callback failed for canvas=%s roi=%s: %s",
                getattr(canvas, "canvas_id", ""),
                job.get("roi_id"),
                exc,
                exc_info=True,
            )
