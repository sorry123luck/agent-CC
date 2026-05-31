"""
E Phase 2 迁移：创建 page_models、state_templates、canvas_snapshots、candidate_states 表。

运行方式：
    python -m src.storage.migration_e2

幂等：已存在的表会被跳过。
"""

from __future__ import annotations

from sqlalchemy import inspect

from src.storage.db import Database
from src.storage.schema import Base


def migrate(db: Database | None = None) -> list[str]:
    """执行迁移，返回新建的表名列表。"""
    if db is None:
        db = Database()

    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())

    created: list[str] = []

    target_tables = [
        "page_models",
        "state_templates",
        "canvas_snapshots",
        "candidate_states",
    ]

    tables_to_create = [
        t for t in Base.metadata.sorted_tables
        if t.name in target_tables
        and t.name not in existing
    ]

    if tables_to_create:
        for table in tables_to_create:
            table.create(db.engine, checkfirst=True)
            created.append(table.name)

    return created


def main() -> None:
    created = migrate()
    if created:
        print(f"Created tables: {', '.join(created)}")
    else:
        print("All tables already exist. Nothing to do.")


if __name__ == "__main__":
    main()
