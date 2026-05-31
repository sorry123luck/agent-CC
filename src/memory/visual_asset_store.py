"""
VisualAssetStore — 视觉资产元数据 CRUD。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from src.storage.schema import VisualAssetRecord

# Whitelist of fields allowed in create_asset(**kwargs).
# Prevents callers from overriding PK, timestamps, or status.
_ALLOWED_CREATE_FIELDS = frozenset({
    "app_id", "page_model_id", "state_template_id", "stable_key_id",
    "canvas_id", "region_id", "width", "height", "quality", "file_size",
    "bounds_json", "relative_bounds_json", "window_size_json", "dpi_scale",
    "screenshot_hash", "hash_version", "dhash", "phash", "source_provider",
    "parent_asset_id", "extra_metadata",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: VisualAssetRecord) -> dict:
    return {
        "asset_id": row.asset_id,
        "asset_type": row.asset_type,
        "app_id": row.app_id,
        "page_model_id": row.page_model_id,
        "state_template_id": row.state_template_id,
        "stable_key_id": row.stable_key_id,
        "canvas_id": row.canvas_id,
        "region_id": row.region_id,
        "path": row.path,
        "width": row.width,
        "height": row.height,
        "format": row.format,
        "quality": row.quality,
        "file_size": row.file_size,
        "bounds_json": row.bounds_json,
        "relative_bounds_json": row.relative_bounds_json,
        "window_size_json": row.window_size_json,
        "dpi_scale": row.dpi_scale,
        "screenshot_hash": row.screenshot_hash,
        "hash_version": row.hash_version,
        "dhash": row.dhash,
        "phash": row.phash,
        "source_provider": row.source_provider,
        "parent_asset_id": row.parent_asset_id,
        "created_at": row.created_at,
        "last_seen_at": row.last_seen_at,
        "version": row.version,
        "status": row.status,
        "metadata": row.extra_metadata,
    }


class VisualAssetStore:
    """视觉资产元数据存储。"""

    def create_asset(
        self,
        session,
        asset_type: str,
        path: str,
        format: str,
        **kwargs,
    ) -> str:
        # Map API-level 'metadata' to ORM column 'extra_metadata'
        if "metadata" in kwargs:
            kwargs["extra_metadata"] = kwargs.pop("metadata")
        # Whitelist: only allow known data fields, reject PK/timestamps/status
        filtered = {k: v for k, v in kwargs.items() if k in _ALLOWED_CREATE_FIELDS}
        asset_id = uuid.uuid4().hex
        now = _now_iso()
        row = VisualAssetRecord(
            asset_id=asset_id,
            asset_type=asset_type,
            path=path,
            format=format,
            created_at=now,
            last_seen_at=now,
            **filtered,
        )
        session.add(row)
        session.flush()
        return asset_id

    def get_asset(self, session, asset_id: str) -> dict | None:
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        return _row_to_dict(row) if row else None

    def list_by_stable_key(
        self,
        session,
        stable_key_id: str,
        asset_type: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        q = session.query(VisualAssetRecord).filter_by(stable_key_id=stable_key_id)
        if asset_type:
            q = q.filter_by(asset_type=asset_type)
        if status:
            q = q.filter_by(status=status)
        q = q.order_by(VisualAssetRecord.created_at.desc())
        return [_row_to_dict(r) for r in q.all()]

    def list_by_page(
        self,
        session,
        page_model_id: str,
        state_template_id: str | None = None,
        asset_type: str | None = None,
    ) -> list[dict]:
        q = session.query(VisualAssetRecord).filter_by(page_model_id=page_model_id)
        if state_template_id:
            q = q.filter_by(state_template_id=state_template_id)
        if asset_type:
            q = q.filter_by(asset_type=asset_type)
        q = q.order_by(VisualAssetRecord.created_at.desc())
        return [_row_to_dict(r) for r in q.all()]

    def list_by_canvas(self, session, canvas_id: str) -> list[dict]:
        rows = (
            session.query(VisualAssetRecord)
            .filter_by(canvas_id=canvas_id)
            .order_by(VisualAssetRecord.created_at.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def get_by_dhash(
        self,
        session,
        dhash: str,
        *,
        asset_type: str | None = None,
        page_model_id: str | None = None,
        state_template_id: str | None = None,
        stable_key_id: str | None = None,
    ) -> list[dict]:
        # No full table scan: require at least one filter condition
        if not any([asset_type, page_model_id, state_template_id, stable_key_id]):
            raise ValueError(
                "get_by_dhash requires at least one filter condition "
                "(asset_type, page_model_id, state_template_id, or stable_key_id)"
            )
        q = session.query(VisualAssetRecord).filter_by(dhash=dhash)
        if asset_type:
            q = q.filter_by(asset_type=asset_type)
        if page_model_id:
            q = q.filter_by(page_model_id=page_model_id)
        if state_template_id:
            q = q.filter_by(state_template_id=state_template_id)
        if stable_key_id:
            q = q.filter_by(stable_key_id=stable_key_id)
        return [_row_to_dict(r) for r in q.all()]

    def mark_stale(self, session, asset_id: str) -> bool:
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        row.status = "stale"
        session.flush()
        return True

    def archive(self, session, asset_id: str) -> bool:
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        row.status = "archived"
        session.flush()
        return True

    def update_last_seen(self, session, asset_id: str) -> bool:
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        row.last_seen_at = _now_iso()
        session.flush()
        return True

    def next_version(self, session, stable_key_id: str, asset_type: str) -> int:
        """Return the next version number for a given key+type.

        Note: caller must create the new asset row within the same transaction
        to avoid duplicate version numbers under concurrent access.
        """
        rows = (
            session.query(VisualAssetRecord)
            .filter_by(stable_key_id=stable_key_id, asset_type=asset_type, status="active")
            .order_by(VisualAssetRecord.version.desc())
            .all()
        )
        return (rows[0].version + 1) if rows else 1

    def delete_asset(
        self,
        session,
        asset_id: str,
        delete_file: bool = False,
        asset_root: str | None = None,
    ) -> bool:
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        if delete_file:
            if not asset_root:
                raise ValueError("asset_root is required when delete_file=True")
            base = os.path.realpath(asset_root)
            resolved = os.path.realpath(os.path.join(base, row.path))
            if not resolved.startswith(base + os.sep) and resolved != base:
                raise ValueError(f"Path escapes asset root: {row.path}")
            if os.path.exists(resolved):
                os.remove(resolved)
        session.delete(row)
        session.flush()
        return True

    def delete_orphaned_assets(self, session, dry_run: bool = True) -> list[str]:
        # Phase 1: dry_run forced, only returns list
        if not dry_run:
            raise ValueError("delete_orphaned_assets only supports dry_run=True in Phase 1")
        active = (
            session.query(VisualAssetRecord)
            .filter_by(status="active")
            .all()
        )
        orphans = []
        for row in active:
            if not os.path.exists(row.path):
                orphans.append(row.asset_id)
        return orphans
