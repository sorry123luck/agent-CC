"""Processing/job 状态持久化迁移。

运行方式：
    python -m src.storage.migration_processing

幂等：只创建缺失表，不改已有数据。
"""

from __future__ import annotations

from sqlalchemy import inspect

from src.storage.db import Database
from src.storage.schema import Base


def migrate(db: Database | None = None) -> list[str]:
    """创建 canvas_processing_states / processing_jobs 表。"""
    if db is None:
        db = Database()

    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())
    target_tables = {"canvas_processing_states", "processing_jobs"}
    created: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name in target_tables and table.name not in existing:
            table.create(db.engine, checkfirst=True)
            created.append(table.name)

    return created


def main() -> None:
    created = migrate()
    if created:
        print(f"Created tables: {', '.join(created)}")
    else:
        print("All processing tables already exist. Nothing to do.")


if __name__ == "__main__":
    main()
