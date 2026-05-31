"""多源证据融合迁移：创建 4 张新表 + ALTER stable_candidate_keys。

运行方式：
    python -m src.storage.migration_evidence

幂等：已存在的表会被跳过，缺失的列会被 ALTER TABLE 补充。
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from src.storage.db import Database
from src.storage.schema import Base


def migrate(db: Database | None = None) -> list[str]:
    """执行迁移，返回新建/修改的表名列表。"""
    if db is None:
        db = Database()

    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())

    created: list[str] = []

    target_tables = [
        "visual_assets",
        "visual_observations",
        "candidate_evidence",
        "candidate_confidence_profiles",
    ]

    # 创建不存在的表
    tables_to_create = [
        t for t in Base.metadata.sorted_tables
        if t.name in target_tables
        and t.name not in existing
    ]

    if tables_to_create:
        for table in tables_to_create:
            table.create(db.engine, checkfirst=True)
            created.append(table.name)

    # 补充缺失列（ALTER TABLE ... ADD COLUMN）
    _add_missing_columns(db, inspector, existing)

    return created


def _add_missing_columns(db: Database, inspector, existing: set[str]) -> None:
    """检查并补充 stable_candidate_keys 和 canvas_snapshots 的缺失列。"""
    # stable_candidate_keys
    if "stable_candidate_keys" in existing:
        columns = {col["name"] for col in inspector.get_columns("stable_candidate_keys")}
        new_columns = [
            ("permanence_state", "TEXT DEFAULT 'new'"),
            ("state_changed_at", "TEXT"),
            ("fail_count", "INTEGER DEFAULT 0"),
            ("coordinate_drift", "REAL DEFAULT 0.0"),
            ("latest_profile_id", "TEXT"),
        ]
        _VALID_COLUMN_NAMES = {name for name, _ in new_columns}
        with db.engine.begin() as conn:
            for col_name, col_def in new_columns:
                if col_name not in _VALID_COLUMN_NAMES:
                    raise ValueError(f"Unexpected column name: {col_name}")
                if col_name not in columns:
                    conn.execute(text(
                        f"ALTER TABLE stable_candidate_keys ADD COLUMN {col_name} {col_def}"
                    ))

    # canvas_snapshots: add canvas_json for canvas persistence
    if "canvas_snapshots" in existing:
        columns = {col["name"] for col in inspector.get_columns("canvas_snapshots")}
        if "canvas_json" not in columns:
            with db.engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE canvas_snapshots ADD COLUMN canvas_json TEXT"
                ))


def main() -> None:
    created = migrate()
    if created:
        print(f"Created tables: {', '.join(created)}")
    else:
        print("All tables already exist. Nothing to do.")


if __name__ == "__main__":
    main()
