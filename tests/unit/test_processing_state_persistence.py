"""Processing registry persistence tests."""

from __future__ import annotations

import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.integration.processing_state import ProcessingRegistry
from src.storage.schema import Base, CanvasProcessingRecord, ProcessingJobRecord


def _session_factory():
    engine = sqlalchemy.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_canvas_state_persists_and_loads(monkeypatch):
    Session = _session_factory()
    monkeypatch.setattr("src.storage.db.Session", Session)

    registry = ProcessingRegistry()
    state = registry.set_canvas_state("canvas_1", "local_ready")
    assert state.processing_state == "local_ready"

    restored = ProcessingRegistry().get_canvas_state("canvas_1")
    assert restored.canvas_id == "canvas_1"
    assert restored.processing_state == "local_ready"

    with Session() as session:
        row = session.get(CanvasProcessingRecord, "canvas_1")
        assert row is not None
        assert row.processing_state == "local_ready"


def test_job_state_persists_and_deduplicates(monkeypatch):
    Session = _session_factory()
    monkeypatch.setattr("src.storage.db.Session", Session)

    registry = ProcessingRegistry()
    job, created = registry.start_job("canvas_1", "vlm_semantic", state="vlm_queued")
    assert created is True
    assert job.state == "vlm_queued"

    existing, created_again = ProcessingRegistry().start_job("canvas_1", "vlm_semantic", state="vlm_queued")
    assert created_again is False
    assert existing.job_id == job.job_id

    updated = registry.update_job(job.job_id, "enhanced_ready")
    assert updated is not None
    assert updated.state == "enhanced_ready"

    with Session() as session:
        row = session.get(ProcessingJobRecord, job.job_id)
        assert row is not None
        assert row.state == "enhanced_ready"


def test_recover_incomplete_jobs_marks_stale_work_failed(monkeypatch):
    Session = _session_factory()
    monkeypatch.setattr("src.storage.db.Session", Session)

    registry = ProcessingRegistry()
    job, _ = registry.start_job("canvas_1", "vlm_semantic", state="vlm_running")

    recovered = ProcessingRegistry().recover_incomplete_jobs()
    assert recovered == 1

    restored = ProcessingRegistry().get_job(job.job_id)
    assert restored is not None
    assert restored.state == "failed"
    assert restored.error == "server_restarted_before_job_completed"

    canvas_state = ProcessingRegistry().get_canvas_state("canvas_1")
    assert canvas_state.processing_state == "failed"
    assert canvas_state.last_error == "server_restarted_before_job_completed"


def test_cancel_job_persists_cancelled_state(monkeypatch):
    Session = _session_factory()
    monkeypatch.setattr("src.storage.db.Session", Session)

    registry = ProcessingRegistry()
    job, _ = registry.start_job("canvas_1", "vlm_semantic", state="vlm_queued")
    cancelled = registry.cancel_job(job.job_id)

    assert cancelled is not None
    assert cancelled.state == "cancelled"
    assert registry.is_cancelled(job.job_id) is True

    restored = ProcessingRegistry().get_job(job.job_id)
    assert restored is not None
    assert restored.state == "cancelled"
