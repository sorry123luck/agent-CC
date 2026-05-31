"""Tests for model integrity audit script."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from sqlalchemy import text

from src.storage.db import Database


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_model_integrity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_model_integrity", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_model_integrity_reports_orphan_snapshot_refs(tmp_path):
    module = _load_module()
    db_path = str(tmp_path / "openclaw.db")
    db = Database(db_path)
    db.create_all()
    with db.session() as session:
        session.execute(
            text(
                """INSERT INTO canvas_snapshots
                   (snapshot_id, canvas_id, page_model_id, state_template_id,
                    captured_at, element_count, has_screenshot)
                   VALUES (:sid, 'canvas_1', :pm, :st, '2026-01-01T00:00:00', 1, 0)"""
            ),
            {"sid": str(uuid.uuid4()), "pm": str(uuid.uuid4()), "st": str(uuid.uuid4())},
        )

    report = module.audit_model_integrity(db_path)

    assert report["status"] == "warn"
    assert report["counts"]["orphan_canvas_snapshot_state_refs"] == 1
    assert report["counts"]["orphan_canvas_snapshot_page_refs"] == 1
    assert report["samples"]["orphan_canvas_snapshot_state_refs"][0]["canvas_id"] == "canvas_1"


def test_audit_model_integrity_passes_clean_database(tmp_path):
    module = _load_module()
    db_path = str(tmp_path / "openclaw.db")
    db = Database(db_path)
    db.create_all()

    report = module.audit_model_integrity(db_path)

    assert report["status"] == "pass"
    assert all(value == 0 for value in report["counts"].values())


def test_write_report_outputs_json_and_markdown(tmp_path):
    module = _load_module()
    report = {
        "status": "warn",
        "counts": {"orphan_canvas_snapshot_state_refs": 1},
        "samples": {
            "orphan_canvas_snapshot_state_refs": [
                {"canvas_id": "canvas_1", "page_model_id": "pm", "state_template_id": "st", "captured_at": "now"}
            ]
        },
    }

    module.write_report(report, tmp_path)

    assert (tmp_path / "model_integrity_audit.json").exists()
    assert "canvas_1" in (tmp_path / "model_integrity_audit.md").read_text(encoding="utf-8")
