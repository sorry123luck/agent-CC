"""Repair runtime metadata drift between canvas snapshots and page models.

This script is intentionally conservative: it does not delete screenshots,
models, assets, or VLM responses. It only backs up the DB, then recomputes
summary counters from stored canvas JSON so the workbench tree reflects the
real canvas contents.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import func

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.schema import (  # noqa: E402
    CanvasSnapshotRecord,
    PageModelRecord,
    StateTemplateRecord,
)
from src.storage.db import init_db  # noqa: E402


DB_PATH = ROOT / "data" / "openclaw.db"
BACKUP_DIR = ROOT / "data" / "backups"


def _count_elements(canvas_json: str | None) -> int | None:
    if not canvas_json:
        return None
    try:
        payload = json.loads(canvas_json)
    except json.JSONDecodeError:
        return None
    elements = payload.get("elements")
    if isinstance(elements, list):
        return len(elements)
    return None


def main() -> int:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return 1

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"openclaw_runtime_consistency_fix_{stamp}.db"
    shutil.copy2(DB_PATH, backup_path)
    print(f"backup={backup_path}")

    updated_snapshots = 0
    updated_states = 0
    updated_models = 0

    db = init_db(str(DB_PATH))
    with db.session() as session:
        snapshots = session.query(CanvasSnapshotRecord).all()
        for snapshot in snapshots:
            count = _count_elements(snapshot.canvas_json)
            if count is None:
                continue
            if snapshot.element_count != count:
                snapshot.element_count = count
                updated_snapshots += 1

        session.flush()

        states = session.query(StateTemplateRecord).all()
        for state in states:
            snapshot_count = (
                session.query(func.count(CanvasSnapshotRecord.snapshot_id))
                .filter(CanvasSnapshotRecord.state_template_id == state.state_template_id)
                .scalar()
                or 0
            )
            max_elements = (
                session.query(func.max(CanvasSnapshotRecord.element_count))
                .filter(CanvasSnapshotRecord.state_template_id == state.state_template_id)
                .scalar()
                or 0
            )
            changed = False
            if state.snapshot_count != snapshot_count:
                state.snapshot_count = snapshot_count
                changed = True
            if max_elements and (state.total_element_count or 0) != max_elements:
                state.total_element_count = max_elements
                changed = True
            if changed:
                updated_states += 1

        models = session.query(PageModelRecord).all()
        for model in models:
            state_count = (
                session.query(func.count(StateTemplateRecord.state_template_id))
                .filter(StateTemplateRecord.page_model_id == model.page_model_id)
                .scalar()
                or 0
            )
            if model.state_count != state_count:
                model.state_count = state_count
                updated_models += 1

        session.commit()

    print(
        "updated "
        f"snapshots={updated_snapshots} "
        f"states={updated_states} "
        f"models={updated_models}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
