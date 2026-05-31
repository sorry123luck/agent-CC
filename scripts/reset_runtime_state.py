"""Reset DeskCanvas runtime state after backing up the SQLite database.

This clears observed canvases, page models, processing jobs, candidate evidence,
visual assets/observations, and VLM cached responses. It intentionally keeps
source code, model weights, configuration files, and test fixtures.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "openclaw.db"
WARM_DIR = ROOT / "data" / "visual_assets" / "warm"

RUNTIME_TABLES = [
    "processing_jobs",
    "canvas_processing_states",
    "candidate_confidence_profiles",
    "candidate_evidence",
    "candidate_overrides",
    "visual_observations",
    "visual_assets",
    "vlm_responses",
    "element_templates",
    "region_templates",
    "app_shell_templates",
    "candidate_states",
    "candidate_templates",
    "stable_candidate_keys",
    "canvas_snapshots",
    "state_templates",
    "page_models",
]


def _backup_db(db_path: Path) -> Path | None:
    if not db_path.exists():
        return None
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}_{stamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def _delete_table_rows(conn: sqlite3.Connection) -> dict[str, int]:
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    deleted: dict[str, int] = {}
    for table in RUNTIME_TABLES:
        if table not in existing:
            continue
        before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DELETE FROM {table}")
        deleted[table] = int(before)
    conn.commit()
    return deleted


def _delete_warm_images() -> int:
    if not WARM_DIR.exists():
        return 0
    count = 0
    for path in WARM_DIR.glob("*.jpg"):
        path.unlink(missing_ok=True)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup DB and clear DeskCanvas runtime state.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    backup_path = _backup_db(db_path)
    if not db_path.exists():
        print(f"DB not found, skipped table cleanup: {db_path}")
        warm_count = _delete_warm_images()
        print(f"Deleted warm screenshots: {warm_count}")
        return 0

    with sqlite3.connect(str(db_path)) as conn:
        deleted = _delete_table_rows(conn)
    warm_count = _delete_warm_images()

    print(f"Backup: {backup_path if backup_path else 'none'}")
    print("Deleted rows:")
    for table, count in deleted.items():
        print(f"  {table}: {count}")
    print(f"Deleted warm screenshots: {warm_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
