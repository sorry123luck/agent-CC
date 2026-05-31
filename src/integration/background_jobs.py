"""Small in-memory background job runner for API enhancement work."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from src.integration.processing_state import registry

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="openclaw-bg")
_vlm_semaphore = threading.Semaphore(1)


def submit_canvas_job(
    *,
    canvas_id: str,
    job_type: str,
    queued_state: str,
    running_state: str,
    success_state: str,
    fn: Callable[[], object],
    use_vlm_semaphore: bool = False,
    max_attempts: int = 1,
) -> tuple[str, bool]:
    """Submit a deduplicated per-canvas job.

    Returns (job_id, created). If a same canvas/job_type job is already active,
    returns the existing job id with created=False.
    """
    job, created = registry.start_job(canvas_id, job_type, state=queued_state)
    if not created:
        return job.job_id, False

    def _runner() -> None:
        lock = _vlm_semaphore if use_vlm_semaphore else None
        acquired = False
        try:
            if registry.is_cancelled(job.job_id):
                return
            if lock is not None:
                lock.acquire()
                acquired = True
            if registry.is_cancelled(job.job_id):
                return
            attempts = max(1, max_attempts)
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                if registry.is_cancelled(job.job_id):
                    return
                registry.update_job(job.job_id, running_state if attempt == 1 else "retrying")
                try:
                    fn()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= attempts or not _is_transient_error(exc):
                        raise
                    logger.info("Retrying background job %s after transient error: %s", job.job_id, exc)
            if last_exc is not None:
                raise last_exc
            if not registry.is_cancelled(job.job_id):
                registry.update_job(job.job_id, success_state)
        except Exception as exc:
            logger.warning("Background job %s failed: %s", job.job_id, exc, exc_info=True)
            if not registry.is_cancelled(job.job_id):
                registry.update_job(job.job_id, "failed", error=str(exc))
        finally:
            if lock is not None and acquired:
                lock.release()

    _executor.submit(_runner)
    return job.job_id, True


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
        "database is locked",
    )
    return any(marker in text for marker in transient_markers)
