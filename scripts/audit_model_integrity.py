"""Audit persisted page model references without mutating the database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.db import Database


def audit_model_integrity(db_path: str = "data/openclaw.db") -> dict[str, Any]:
    """Return counts and samples for orphaned page/state references."""
    db = Database(db_path)
    with db.session() as session:
        orphan_snapshot_states = _query_dicts(
            session,
            """
            SELECT cs.canvas_id, cs.page_model_id, cs.state_template_id, cs.captured_at
            FROM canvas_snapshots cs
            LEFT JOIN state_templates st ON st.state_template_id = cs.state_template_id
            WHERE cs.state_template_id IS NOT NULL AND st.state_template_id IS NULL
            ORDER BY cs.captured_at DESC
            LIMIT 50
            """,
        )
        orphan_snapshot_pages = _query_dicts(
            session,
            """
            SELECT cs.canvas_id, cs.page_model_id, cs.state_template_id, cs.captured_at
            FROM canvas_snapshots cs
            LEFT JOIN page_models pm ON pm.page_model_id = cs.page_model_id
            WHERE cs.page_model_id IS NOT NULL AND pm.page_model_id IS NULL
            ORDER BY cs.captured_at DESC
            LIMIT 50
            """,
        )
        orphan_candidate_states = _query_dicts(
            session,
            """
            SELECT cs.id, cs.key_id, cs.page_model_id, cs.state_template_id
            FROM candidate_states cs
            LEFT JOIN stable_candidate_keys k ON k.key_id = cs.key_id
            LEFT JOIN state_templates st ON st.state_template_id = cs.state_template_id
            WHERE k.key_id IS NULL OR st.state_template_id IS NULL
            ORDER BY cs.id DESC
            LIMIT 50
            """,
        )
        counts = {
            "orphan_canvas_snapshot_state_refs": _count(
                session,
                """
                SELECT COUNT(*)
                FROM canvas_snapshots cs
                LEFT JOIN state_templates st ON st.state_template_id = cs.state_template_id
                WHERE cs.state_template_id IS NOT NULL AND st.state_template_id IS NULL
                """,
            ),
            "orphan_canvas_snapshot_page_refs": _count(
                session,
                """
                SELECT COUNT(*)
                FROM canvas_snapshots cs
                LEFT JOIN page_models pm ON pm.page_model_id = cs.page_model_id
                WHERE cs.page_model_id IS NOT NULL AND pm.page_model_id IS NULL
                """,
            ),
            "orphan_candidate_states": _count(
                session,
                """
                SELECT COUNT(*)
                FROM candidate_states cs
                LEFT JOIN stable_candidate_keys k ON k.key_id = cs.key_id
                LEFT JOIN state_templates st ON st.state_template_id = cs.state_template_id
                WHERE k.key_id IS NULL OR st.state_template_id IS NULL
                """,
            ),
        }
    status = "pass" if all(value == 0 for value in counts.values()) else "warn"
    return {
        "status": status,
        "counts": counts,
        "samples": {
            "orphan_canvas_snapshot_state_refs": orphan_snapshot_states,
            "orphan_canvas_snapshot_page_refs": orphan_snapshot_pages,
            "orphan_candidate_states": orphan_candidate_states,
        },
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_integrity_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Model Integrity Audit",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
    ]
    for name, rows in (report.get("samples") or {}).items():
        lines.append(f"## {name}")
        lines.append("")
        if not rows:
            lines.append("none")
            lines.append("")
            continue
        lines.append("| canvas/id | page_model_id | state_template_id | extra |")
        lines.append("| --- | --- | --- | --- |")
        for row in rows[:20]:
            ident = row.get("canvas_id") or row.get("id") or ""
            extra = row.get("captured_at") or row.get("key_id") or ""
            lines.append(
                "| {ident} | {pm} | {st} | {extra} |".format(
                    ident=str(ident).replace("|", "/"),
                    pm=str(row.get("page_model_id") or "").replace("|", "/"),
                    st=str(row.get("state_template_id") or "").replace("|", "/"),
                    extra=str(extra).replace("|", "/"),
                )
            )
        lines.append("")
    (output_dir / "model_integrity_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _query_dicts(session: Any, sql: str) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in session.execute(text(sql)).all()]


def _count(session: Any, sql: str) -> int:
    return int(session.execute(text(sql)).scalar() or 0)


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/openclaw.db")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_integrity_audit"))
    args = parser.parse_args()
    report = audit_model_integrity(args.db_path)
    write_report(report, args.output_dir)
    print("status={status} {counts}".format(status=report["status"], counts=_format_counts(report["counts"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
