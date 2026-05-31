"""Thread-safe in-memory processing state for canvas enhancement jobs."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


TERMINAL_STATES = {
    "enhanced_ready",
    "semantic_ready",
    "semantic_timeout",
    "semantic_partial",
    "semantic_failed",
    "semantic_late_merged",
    "semantic_late_failed",
    "failed",
    "cancelled",
}
_logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobState:
    job_id: str
    canvas_id: str
    job_type: str
    state: str = "queued"
    error: str = ""
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanvasProcessingState:
    canvas_id: str
    processing_state: str = "local_ready"
    jobs: list[str] = field(default_factory=list)
    last_error: str = ""
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas_id": self.canvas_id,
            "processing_state": self.processing_state,
            "jobs": list(self.jobs),
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }


class ProcessingRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._canvas: dict[str, CanvasProcessingState] = {}
        self._jobs: dict[str, JobState] = {}
        self._active_by_canvas_type: dict[tuple[str, str], str] = {}

    def set_canvas_state(
        self,
        canvas_id: str,
        state: str,
        *,
        error: str = "",
        job_id: str | None = None,
    ) -> CanvasProcessingState:
        with self._lock:
            item = self._canvas.get(canvas_id) or CanvasProcessingState(canvas_id=canvas_id)
            item.processing_state = state
            item.last_error = error
            item.updated_at = _now()
            if job_id and job_id not in item.jobs:
                item.jobs.append(job_id)
            self._canvas[canvas_id] = item
            snapshot = CanvasProcessingState(
                canvas_id=item.canvas_id,
                processing_state=item.processing_state,
                jobs=list(item.jobs),
                last_error=item.last_error,
                updated_at=item.updated_at,
            )
        self._persist_canvas(snapshot)
        return snapshot

    def get_canvas_state(self, canvas_id: str) -> CanvasProcessingState:
        with self._lock:
            item = self._canvas.get(canvas_id)
            if item is not None:
                return CanvasProcessingState(
                    canvas_id=item.canvas_id,
                    processing_state=item.processing_state,
                    jobs=list(item.jobs),
                    last_error=item.last_error,
                    updated_at=item.updated_at,
                )

        persisted = self._load_canvas(canvas_id)
        if persisted is None:
            return CanvasProcessingState(canvas_id=canvas_id)
        with self._lock:
            self._canvas[canvas_id] = persisted
        return CanvasProcessingState(
            canvas_id=persisted.canvas_id,
            processing_state=persisted.processing_state,
            jobs=list(persisted.jobs),
            last_error=persisted.last_error,
            updated_at=persisted.updated_at,
        )

    def recover_incomplete_jobs(self) -> int:
        """Mark stale non-terminal jobs as failed after process restart.

        Phase 1 still executes jobs in-process, so a process restart means queued/running
        work is no longer alive. Persisted failure makes the UI truthful instead of
        showing an endless spinner.
        """
        now = _now()
        recovered = 0
        try:
            from src.storage.db import Session
            from src.storage.schema import CanvasProcessingRecord, ProcessingJobRecord

            with Session() as session:
                rows = (
                    session.query(ProcessingJobRecord)
                    .filter(ProcessingJobRecord.state.notin_(list(TERMINAL_STATES)))
                    .all()
                )
                for row in rows:
                    row.state = "failed"
                    row.error = "server_restarted_before_job_completed"
                    row.updated_at = now
                    canvas = session.get(CanvasProcessingRecord, row.canvas_id)
                    if canvas is not None:
                        canvas.processing_state = "failed"
                        canvas.last_error = row.error
                        canvas.updated_at = now
                    recovered += 1
                session.commit()
        except Exception as exc:
            _logger.debug("Processing job recovery skipped: %s", exc)
        return recovered

    def list_jobs_for_canvas(self, canvas_id: str) -> list[JobState]:
        jobs: list[JobState] = []
        with self._lock:
            for job in self._jobs.values():
                if job.canvas_id == canvas_id:
                    jobs.append(JobState(**job.to_dict()))
        persisted = self._load_jobs_for_canvas(canvas_id)
        by_id = {job.job_id: job for job in jobs}
        for job in persisted:
            by_id.setdefault(job.job_id, job)
        return sorted(by_id.values(), key=lambda item: item.updated_at, reverse=True)

    def list_jobs(self, limit: int = 100) -> list[JobState]:
        persisted = self._load_jobs(limit=limit)
        with self._lock:
            by_id = {job.job_id: job for job in persisted}
            for job in self._jobs.values():
                by_id[job.job_id] = JobState(**job.to_dict())
        return sorted(by_id.values(), key=lambda item: item.updated_at, reverse=True)[:limit]

    def _get_active_job_locked(self, canvas_id: str, job_type: str) -> JobState | None:
        existing_id = self._active_by_canvas_type.get((canvas_id, job_type))
        if existing_id:
            existing = self._jobs.get(existing_id)
            if existing and existing.state not in TERMINAL_STATES:
                return JobState(**existing.to_dict())
        return None

    def _load_active_job(self, canvas_id: str, job_type: str) -> JobState | None:
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                row = (
                    session.query(ProcessingJobRecord)
                    .filter(
                        ProcessingJobRecord.canvas_id == canvas_id,
                        ProcessingJobRecord.job_type == job_type,
                        ProcessingJobRecord.state.notin_(list(TERMINAL_STATES)),
                    )
                    .order_by(ProcessingJobRecord.updated_at.desc())
                    .first()
                )
                return _job_from_row(row) if row is not None else None
        except Exception as exc:
            _logger.debug("Processing active job lookup skipped: %s", exc)
            return None

    def _persist_canvas(self, item: CanvasProcessingState) -> None:
        try:
            from src.storage.db import Session
            from src.storage.schema import CanvasProcessingRecord

            with Session() as session:
                row = session.get(CanvasProcessingRecord, item.canvas_id)
                if row is None:
                    row = CanvasProcessingRecord(canvas_id=item.canvas_id, updated_at=item.updated_at)
                    session.add(row)
                row.processing_state = item.processing_state
                row.jobs_json = json.dumps(item.jobs)
                row.last_error = item.last_error
                row.updated_at = item.updated_at
                session.commit()
        except Exception as exc:
            _logger.debug("Processing canvas persist skipped: %s", exc)

    def _persist_job(self, job: JobState) -> None:
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                row = session.get(ProcessingJobRecord, job.job_id)
                if row is None:
                    row = ProcessingJobRecord(
                        job_id=job.job_id,
                        canvas_id=job.canvas_id,
                        job_type=job.job_type,
                        started_at=job.started_at,
                        updated_at=job.updated_at,
                    )
                    session.add(row)
                row.canvas_id = job.canvas_id
                row.job_type = job.job_type
                row.state = job.state
                row.error = job.error
                row.started_at = job.started_at
                row.updated_at = job.updated_at
                session.commit()
        except Exception as exc:
            _logger.debug("Processing job persist skipped: %s", exc)

    def _load_canvas(self, canvas_id: str) -> CanvasProcessingState | None:
        try:
            from src.storage.db import Session
            from src.storage.schema import CanvasProcessingRecord

            with Session() as session:
                row = session.get(CanvasProcessingRecord, canvas_id)
                if row is None:
                    return None
                return CanvasProcessingState(
                    canvas_id=row.canvas_id,
                    processing_state=row.processing_state or "local_ready",
                    jobs=_parse_jobs(row.jobs_json),
                    last_error=row.last_error or "",
                    updated_at=row.updated_at or _now(),
                )
        except Exception as exc:
            _logger.debug("Processing canvas load skipped: %s", exc)
            return None

    def _load_jobs_for_canvas(self, canvas_id: str) -> list[JobState]:
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                rows = (
                    session.query(ProcessingJobRecord)
                    .filter(ProcessingJobRecord.canvas_id == canvas_id)
                    .order_by(ProcessingJobRecord.updated_at.desc())
                    .all()
                )
                return [_job_from_row(row) for row in rows]
        except Exception as exc:
            _logger.debug("Processing jobs load skipped: %s", exc)
            return []

    def _load_jobs(self, limit: int = 100) -> list[JobState]:
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                rows = (
                    session.query(ProcessingJobRecord)
                    .order_by(ProcessingJobRecord.updated_at.desc())
                    .limit(limit)
                    .all()
                )
                return [_job_from_row(row) for row in rows]
        except Exception as exc:
            _logger.debug("Processing jobs load skipped: %s", exc)
            return []

    def _load_job(self, job_id: str) -> JobState | None:
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                row = session.get(ProcessingJobRecord, job_id)
                return _job_from_row(row) if row is not None else None
        except Exception as exc:
            _logger.debug("Processing job load skipped: %s", exc)
            return None

    def _cache_job_locked(self, job: JobState) -> None:
        self._jobs[job.job_id] = job
        if job.state not in TERMINAL_STATES:
            self._active_by_canvas_type[(job.canvas_id, job.job_type)] = job.job_id

    def _cache_canvas_locked(self, item: CanvasProcessingState) -> None:
        self._canvas[item.canvas_id] = item

    def _canvas_snapshot_locked(self, item: CanvasProcessingState) -> CanvasProcessingState:
        return CanvasProcessingState(
                canvas_id=item.canvas_id,
                processing_state=item.processing_state,
                jobs=list(item.jobs),
                last_error=item.last_error,
                updated_at=item.updated_at,
            )

    def start_job(self, canvas_id: str, job_type: str, *, state: str = "queued") -> tuple[JobState, bool]:
        with self._lock:
            key = (canvas_id, job_type)
            existing = self._get_active_job_locked(canvas_id, job_type)
            if existing is not None:
                return existing, False

        persisted_existing = self._load_active_job(canvas_id, job_type)
        if persisted_existing is not None:
            with self._lock:
                self._cache_job_locked(persisted_existing)
            return persisted_existing, False

        with self._lock:
            key = (canvas_id, job_type)
            job = JobState(job_id=str(uuid.uuid4()), canvas_id=canvas_id, job_type=job_type, state=state)
            self._jobs[job.job_id] = job
            self._active_by_canvas_type[key] = job.job_id
            canvas_state = self._canvas.get(canvas_id) or CanvasProcessingState(canvas_id=canvas_id)
            canvas_state.processing_state = state
            canvas_state.updated_at = job.updated_at
            if job.job_id not in canvas_state.jobs:
                canvas_state.jobs.append(job.job_id)
            self._canvas[canvas_id] = canvas_state
            job_snapshot = JobState(**job.to_dict())
            canvas_snapshot = self._canvas_snapshot_locked(canvas_state)
        self._persist_job(job_snapshot)
        self._persist_canvas(canvas_snapshot)
        return job_snapshot, True

    def update_job(self, job_id: str, state: str, *, error: str = "") -> JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            persisted = self._load_job(job_id)
            if persisted is None:
                return None
            with self._lock:
                self._cache_job_locked(persisted)
                job = self._jobs.get(job_id)
                if job is None:
                    return None
        with self._lock:
            job.state = state
            job.error = error
            job.updated_at = _now()
            canvas_state = self._canvas.get(job.canvas_id) or CanvasProcessingState(canvas_id=job.canvas_id)
            canvas_state.processing_state = state
            canvas_state.last_error = error
            canvas_state.updated_at = job.updated_at
            if job.job_id not in canvas_state.jobs:
                canvas_state.jobs.append(job.job_id)
            self._canvas[job.canvas_id] = canvas_state
            if state in TERMINAL_STATES:
                self._active_by_canvas_type.pop((job.canvas_id, job.job_type), None)
            job_snapshot = JobState(**job.to_dict())
            canvas_snapshot = self._canvas_snapshot_locked(canvas_state)
        self._persist_job(job_snapshot)
        self._persist_canvas(canvas_snapshot)
        return job_snapshot

    def cancel_job(self, job_id: str, *, reason: str = "cancelled_by_user") -> JobState | None:
        """Mark a queued/running job as cancelled.

        ThreadPoolExecutor cannot safely kill running work, so cancellation is
        cooperative: runners check the persisted state before and after work.
        """
        return self.update_job(job_id, "cancelled", error=reason)

    def is_cancelled(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and job.state == "cancelled")

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return JobState(**job.to_dict())
        try:
            from src.storage.db import Session
            from src.storage.schema import ProcessingJobRecord

            with Session() as session:
                row = session.get(ProcessingJobRecord, job_id)
                if row is None:
                    return None
                job = _job_from_row(row)
        except Exception as exc:
            _logger.debug("Processing job load skipped: %s", exc)
            return None
        with self._lock:
            self._cache_job_locked(job)
        return job


def _parse_jobs(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(item) for item in value]
    except Exception:
        return []
    return []


def _job_from_row(row) -> JobState:
    return JobState(
        job_id=row.job_id,
        canvas_id=row.canvas_id,
        job_type=row.job_type,
        state=row.state or "queued",
        error=row.error or "",
        started_at=row.started_at or _now(),
        updated_at=row.updated_at or _now(),
    )


registry = ProcessingRegistry()
