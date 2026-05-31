"""VLM Semantic Modeler 迁移：创建 vlm_responses 表。

运行方式：
    python -m src.storage.migration_vlm

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
        "vlm_responses",
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
    """检查并补充缺失的列。"""
    if "vlm_responses" not in existing:
        return

    columns = {col["name"] for col in inspector.get_columns("vlm_responses")}

    # cost_usd 列（v2 迁移新增）
    if "cost_usd" not in columns:
        with db.engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE vlm_responses ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0.0"
            ))


def main() -> None:
    created = migrate()
    if created:
        print(f"Created tables: {', '.join(created)}")
    else:
        print("All tables already exist. Nothing to do.")


if __name__ == "__main__":
    main()
