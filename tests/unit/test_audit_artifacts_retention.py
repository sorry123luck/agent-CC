"""Tests for artifact retention auditing."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_artifacts_retention.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_artifacts_retention", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_audit_marks_report_dirs_keep_and_old_raw_dirs_archive_candidate(tmp_path: Path):
    module = _load_module()
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "agent_operability_regression_2026-05-31-current" / "agent_operability_regression_summary.json")
    _write(artifacts / "live_sample_matrix_2026-05-20-old" / "raw.json")
    old_time = datetime.now() - timedelta(days=14)
    old_ts = old_time.timestamp()
    raw_dir = artifacts / "live_sample_matrix_2026-05-20-old"
    for path in raw_dir.rglob("*"):
        os.utime(path, (old_ts, old_ts))
    os.utime(raw_dir, (old_ts, old_ts))

    report = module.audit_artifacts_retention(artifacts_root=artifacts, now=datetime.now(), archive_after_days=7)

    assert report["counts"]["total_dirs"] == 2
    keep = next(row for row in report["rows"] if row["name"].startswith("agent_operability"))
    raw = next(row for row in report["rows"] if row["name"].startswith("live_sample_matrix"))
    assert keep["recommended_action"] == "keep"
    assert "summary_report" in keep["signals"]
    assert raw["recommended_action"] == "archive_candidate"


def test_write_audit_report_outputs_json_and_markdown(tmp_path: Path):
    module = _load_module()
    report = {
        "generated_at": "2026-05-31T00:00:00",
        "counts": {"total_dirs": 1, "keep": 1, "archive_candidate": 0},
        "rows": [
            {
                "name": "sample",
                "size_bytes": 12,
                "age_days": 1,
                "artifact_type": "sample_matrix",
                "recommended_action": "keep",
                "signals": ["sample_matrix"],
            }
        ],
    }

    module.write_artifacts_retention_report(report, tmp_path)

    assert (tmp_path / "artifacts_retention_audit.json").exists()
    assert "sample" in (tmp_path / "artifacts_retention_audit.md").read_text(encoding="utf-8")
