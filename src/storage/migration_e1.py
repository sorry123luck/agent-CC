"""
E Phase 1 迁移：创建 stable_candidate_keys 和 candidate_templates 表。

运行方式：
    python -m src.storage.migration_e1

幂等：已存在的表会被跳过。
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from src.storage.db import Database, Session
from src.storage.schema import Base


def migrate(db: Database | None = None) -> list[str]:
    """执行迁移，返回新建的表名列表。"""
    if db is None:
        db = Database()

    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())

    created: list[str] = []

    # 只创建不存在的表
    tables_to_create = [
        t for t in Base.metadata.sorted_tables
        if t.name in ("stable_candidate_keys", "candidate_templates")
        and t.name not in existing
    ]

    if tables_to_create:
        # 用 metadata.create_all 只创建目标表
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
