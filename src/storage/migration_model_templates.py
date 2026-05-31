"""AppShell / Region / Element 模板表迁移。"""

from __future__ import annotations

from sqlalchemy import inspect

from src.storage.db import Database
from src.storage.schema import Base


def migrate(db: Database | None = None) -> list[str]:
    if db is None:
        db = Database()

    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())
    target_tables = {"app_shell_templates", "region_templates", "element_templates"}
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
        print("All model template tables already exist. Nothing to do.")


if __name__ == "__main__":
    main()
