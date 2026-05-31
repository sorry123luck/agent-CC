"""
清理无效候选数据 — 修复稳定身份污染问题。

清理目标：
1. candidate_templates 中 relative_bounds 为全零或无效的记录
2. 孤立的 stable_candidate_keys（没有任何关联 template）
3. 孤立的 candidate_states（引用不存在的 key 或 state_template）

运行方式：
    python -m src.storage.cleanup_invalid_candidates

幂等：可重复运行，不会删除有效数据。
"""

from __future__ import annotations

import json

from sqlalchemy import text

from src.storage.db import Database


def cleanup(db: Database | None = None) -> dict[str, int]:
    """执行清理，返回各类删除数量。"""
    if db is None:
        db = Database()

    stats = {"invalid_templates": 0, "orphan_keys": 0, "orphan_states": 0}

    with db.session() as session:
        # 1. 删除 candidate_templates 中 relative_bounds 为全零或无效的记录
        #    零 bounds: [0,0,0,0] 或 json 解析后全为 0
        #    无效: 零面积 (x2<=x1 或 y2<=y1)
        templates = session.execute(
            text("SELECT id, relative_bounds FROM candidate_templates")
        ).all()

        invalid_template_ids = []
        for row in templates:
            tid, bounds_json = row[0], row[1]
            if not bounds_json:
                invalid_template_ids.append(tid)
                continue
            try:
                bounds = json.loads(bounds_json)
            except (json.JSONDecodeError, TypeError):
                invalid_template_ids.append(tid)
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
                invalid_template_ids.append(tid)
                continue
            x1, y1, x2, y2 = bounds[:4]
            # 全零 → 无效
            if x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0:
                invalid_template_ids.append(tid)
                continue
            # 零面积 → 无效
            if x2 <= x1 or y2 <= y1:
                invalid_template_ids.append(tid)
                continue

        if invalid_template_ids:
            for tid in invalid_template_ids:
                session.execute(
                    text("DELETE FROM candidate_templates WHERE id = :tid"),
                    {"tid": tid},
                )
            stats["invalid_templates"] = len(invalid_template_ids)

        # 2. 删除孤立的 stable_candidate_keys（没有关联 template）
        orphan_keys = session.execute(text("""
            SELECT k.key_id FROM stable_candidate_keys k
            LEFT JOIN candidate_templates t ON t.key_id = k.key_id
            WHERE t.key_id IS NULL
        """)).all()

        if orphan_keys:
            for row in orphan_keys:
                kid = row[0]
                # 先删 candidate_states 引用
                session.execute(
                    text("DELETE FROM candidate_states WHERE key_id = :kid"),
                    {"kid": kid},
                )
                session.execute(
                    text("DELETE FROM stable_candidate_keys WHERE key_id = :kid"),
                    {"kid": kid},
                )
            stats["orphan_keys"] = len(orphan_keys)

        # 3. 删除孤立的 candidate_states（引用不存在的 key 或 state_template）
        orphan_states = session.execute(text("""
            SELECT cs.id FROM candidate_states cs
            LEFT JOIN stable_candidate_keys k ON k.key_id = cs.key_id
            LEFT JOIN state_templates st ON st.state_template_id = cs.state_template_id
            WHERE k.key_id IS NULL OR st.state_template_id IS NULL
        """)).all()

        if orphan_states:
            for row in orphan_states:
                session.execute(
                    text("DELETE FROM candidate_states WHERE id = :id"),
                    {"id": row[0]},
                )
            stats["orphan_states"] = len(orphan_states)

        session.commit()

    return stats


def main() -> None:
    stats = cleanup()
    total = sum(stats.values())
    if total == 0:
        print("No invalid data found. Database is clean.")
    else:
        print(f"Cleanup complete:")
        print(f"  Invalid templates removed: {stats['invalid_templates']}")
        print(f"  Orphan keys removed: {stats['orphan_keys']}")
        print(f"  Orphan states removed: {stats['orphan_states']}")
        print(f"  Total: {total}")


if __name__ == "__main__":
    main()
