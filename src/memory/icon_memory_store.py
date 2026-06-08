"""IconMemoryStore — 图标控件语义库 CRUD。

基于 VisualAssetRecord(asset_type="control_crop") + extra_metadata。
不新建表，不修改 schema.py。

Phase 1: 离线索引与匹配验证。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.memory.visual_anchor import compute_dhash, dhash_distance
from src.storage.schema import VisualAssetRecord

logger = logging.getLogger(__name__)

ASSET_TYPE = "control_crop"

# 最小合法 PNG（67 字节，1x1 透明）— 用于 hash_only 占位
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\xb4\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)

# Opaque path: controls/{app}/{dhash8}/{asset_id}.dat
_PATH_PREFIX = "controls"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_app_name(app_process: str) -> str:
    """Sanitize app process name for filesystem path."""
    return app_process.replace("/", "_").replace("\\", "_").replace(":", "_")


class IconMemoryMatch:
    """匹配结果。"""

    def __init__(self, row_dict: dict[str, Any]) -> None:
        self.asset_id: str = row_dict["asset_id"]
        self.dhash: str = row_dict.get("dhash", "")
        self.path: str = row_dict.get("path", "")
        self.meta: dict[str, Any] = {}
        raw = row_dict.get("extra_metadata") or row_dict.get("metadata")
        if isinstance(raw, str):
            try:
                self.meta = json.loads(raw)
            except json.JSONDecodeError:
                self.meta = {}
        elif isinstance(raw, dict):
            self.meta = raw

    @property
    def semantic_state(self) -> str:
        return self.meta.get("semantic_state", "pending")

    @property
    def semantic_role(self) -> str:
        return self.meta.get("semantic_role", "")

    @property
    def semantic_text(self) -> str:
        return self.meta.get("semantic_text", "")

    @property
    def confidence(self) -> float:
        return float(self.meta.get("confidence", 0.0))

    @property
    def privacy_level(self) -> str:
        return self.meta.get("privacy_level", "safe")

    @property
    def app_process(self) -> str:
        return self.meta.get("app_process", "")

    @property
    def source(self) -> str:
        return self.meta.get("source", "")


class IconMemoryStore:
    """图标控件语义库 — 基于 VisualAssetRecord。

    Phase 1: 离线索引与匹配验证。
    Phase 2+: 接入 observe 管线。
    """

    def __init__(self, asset_root: Path | str | None = None) -> None:
        self._asset_root = Path(asset_root) if asset_root else Path("data/visual_assets")

    # -------------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------------

    def store_crop(
        self,
        session: Any,
        crop_image: Image.Image | None,
        app_process: str,
        page_class: str,
        semantic_role: str,
        *,
        precomputed_dhash: str = "",
        semantic_text: str = "",
        control_type: str = "",
        region_role: str = "",
        relative_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        size_ratio: float = 0.0,
        source: str = "index_from_canvas",
        privacy_level: str = "safe",
        state: str = "pending",
        confidence: float = 0.0,
        canvas_id: str = "",
        region_id: str = "",
        bounds_json: str = "",
        window_size_json: str = "",
        dpi_scale: float | None = None,
    ) -> str:
        """Store an icon crop. Returns asset_id.

        Args:
            session: SQLAlchemy session
            crop_image: PIL Image crop (None for hash_only)
            app_process: process name (e.g. "qq.exe")
            page_class: page class
            semantic_role: semantic role
            privacy_level: "safe" | "redacted" | "hash_only"
            state: "pending" | "confirmed"

        Returns:
            asset_id (UUID hex)
        """
        asset_id = uuid.uuid4().hex
        now = _now_iso()

        # Compute dHash: prefer precomputed, then compute from image
        dhash = precomputed_dhash
        if not dhash and crop_image is not None:
            arr = np.array(crop_image.convert("RGB"))
            dhash = compute_dhash(arr)

        # Save .dat file
        app_safe = _safe_app_name(app_process)
        dhash_prefix = dhash[:8] if dhash else "nohash"
        rel_dir = Path(_PATH_PREFIX) / app_safe / dhash_prefix
        abs_dir = self._asset_root / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        dat_filename = f"{asset_id}.dat"
        dat_path = abs_dir / dat_filename

        if privacy_level == "hash_only":
            dat_path.write_bytes(_MINIMAL_PNG)
            width, height = 1, 1
            file_size = len(_MINIMAL_PNG)
        elif crop_image is not None:
            crop_image.save(str(dat_path), format="PNG")
            width, height = crop_image.size
            file_size = dat_path.stat().st_size
        else:
            dat_path.write_bytes(_MINIMAL_PNG)
            width, height = 1, 1
            file_size = len(_MINIMAL_PNG)

        rel_path = str(rel_dir / dat_filename)

        # Build extra_metadata
        extra = {
            "semantic_state": state,
            "semantic_role": semantic_role,
            "semantic_text": semantic_text,
            "control_type": control_type,
            "confidence": confidence,
            "source": source,
            "privacy_level": privacy_level,
            "app_process": app_process,
            "page_class": page_class,
            "region_role": region_role,
            "relative_bounds": list(relative_bounds),
            "size_ratio": size_ratio,
            "match_count": 0,
            "confirmed_at": now if state == "confirmed" else None,
            "rejected_at": None,
            "conflict_role": None,
            "notes": "",
        }

        row = VisualAssetRecord(
            asset_id=asset_id,
            asset_type=ASSET_TYPE,
            app_id=app_process,
            canvas_id=canvas_id or None,
            region_id=region_id or None,
            path=rel_path,
            width=width,
            height=height,
            format="png",
            file_size=file_size,
            bounds_json=bounds_json or None,
            relative_bounds_json=json.dumps(list(relative_bounds)) if any(relative_bounds) else None,
            window_size_json=window_size_json or None,
            dpi_scale=dpi_scale,
            hash_version="dhash_v1",
            dhash=dhash or None,
            source_provider=source,
            created_at=now,
            last_seen_at=now,
            version=1,
            status="active",
            extra_metadata=json.dumps(extra, ensure_ascii=False),
        )
        session.add(row)
        session.flush()

        logger.info("Icon memory stored: %s app=%s role=%s state=%s dhash=%s",
                     asset_id[:8], app_process, semantic_role, state, dhash[:8] if dhash else "none")
        return asset_id

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def list_active_by_app(
        self,
        session: Any,
        app_process: str,
    ) -> list[IconMemoryMatch]:
        """List all active control_crop records for an app.

        This method is NOT on VisualAssetStore — we implement it here.
        """
        rows = (
            session.query(VisualAssetRecord)
            .filter_by(app_id=app_process)
            .filter_by(asset_type=ASSET_TYPE)
            .filter_by(status="active")
            .all()
        )
        return [IconMemoryMatch(self._row_to_dict(r)) for r in rows]

    def list_assets(
        self,
        session: Any,
        app_process: str | None = None,
        state: str | None = None,
        role: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List icon memory assets with optional filters.

        Returns list of metadata dicts (no crop content).
        """
        q = session.query(VisualAssetRecord).filter_by(asset_type=ASSET_TYPE)
        if app_process:
            q = q.filter_by(app_id=app_process)
        q = q.order_by(VisualAssetRecord.created_at.desc()).limit(limit)
        rows = q.all()
        result = []
        for row in rows:
            d = self._row_to_dict(row)
            meta = json.loads(d.get("extra_metadata") or "{}")
            # Apply metadata-level filters
            if state and meta.get("semantic_state") != state:
                continue
            if role and meta.get("semantic_role") != role:
                continue
            if source and meta.get("source") != source:
                continue
            # Strip crop path from response — metadata only
            result.append({
                "asset_id": d["asset_id"],
                "app_id": d["app_id"],
                "dhash": d["dhash"],
                "width": d["width"],
                "height": d["height"],
                "file_size": d["file_size"],
                "created_at": d["created_at"],
                "last_seen_at": d["last_seen_at"],
                "status": d["status"],
                "bounds_json": d["bounds_json"],
                "region_id": d["region_id"],
                **meta,
            })
        return result

    def mark_low_quality(self, session: Any, asset_id: str, reason: str = "low_quality") -> bool:
        """Mark an asset as low quality — excluded from backfill/promotion."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        meta = json.loads(row.extra_metadata or "{}")
        meta["quality_flag"] = reason
        meta["quality_flagged_at"] = _now_iso()
        row.extra_metadata = json.dumps(meta, ensure_ascii=False)
        session.flush()
        return True

    def find_match(
        self,
        session: Any,
        dhash: str,
        app_process: str,
        page_class: str = "",
        max_distance: int = 5,
    ) -> tuple[IconMemoryMatch | None, int]:
        """Find the best dHash match for an app.

        Returns (match, dhash_distance) or (None, -1).
        Phase 1: load all active records for app, compute distance in memory.
        """
        if not dhash:
            return None, -1

        candidates = self.list_active_by_app(session, app_process)
        if not candidates:
            return None, -1

        # Collect all candidates within threshold
        candidates_in_range: list[tuple[int, float, IconMemoryMatch]] = []

        for cand in candidates:
            if not cand.dhash:
                continue
            dist = dhash_distance(dhash, cand.dhash)
            if dist <= max_distance:
                tie_score = self._tie_break_score(cand, page_class)
                candidates_in_range.append((dist, tie_score, cand))

        if not candidates_in_range:
            return None, -1

        # Sort: primary = dhash distance (asc), secondary = tie_score (desc)
        candidates_in_range.sort(key=lambda x: (x[0], -x[1]))
        best_dist, best_score, best_match = candidates_in_range[0]
        return best_match, best_dist

    def get_asset(self, session: Any, asset_id: str) -> IconMemoryMatch | None:
        """Get a single asset by ID."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return None
        return IconMemoryMatch(self._row_to_dict(row))

    # -------------------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------------------

    def confirm(self, session: Any, asset_id: str) -> bool:
        """Transition pending → confirmed."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        meta = json.loads(row.extra_metadata or "{}")
        if meta.get("semantic_state") in ("rejected",):
            return False
        meta["semantic_state"] = "confirmed"
        meta["confirmed_at"] = _now_iso()
        row.extra_metadata = json.dumps(meta, ensure_ascii=False)
        session.flush()
        return True

    def reject(self, session: Any, asset_id: str) -> bool:
        """Transition any → rejected."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        meta = json.loads(row.extra_metadata or "{}")
        meta["semantic_state"] = "rejected"
        meta["rejected_at"] = _now_iso()
        row.extra_metadata = json.dumps(meta, ensure_ascii=False)
        session.flush()
        return True

    def mark_conflict(self, session: Any, asset_id: str, conflicting_role: str) -> bool:
        """Transition pending → conflict."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        meta = json.loads(row.extra_metadata or "{}")
        meta["semantic_state"] = "conflict"
        meta["conflict_role"] = conflicting_role
        row.extra_metadata = json.dumps(meta, ensure_ascii=False)
        session.flush()
        return True

    def increment_match_count(self, session: Any, asset_id: str) -> bool:
        """Increment match_count and update last_seen_at."""
        row = session.query(VisualAssetRecord).filter_by(asset_id=asset_id).first()
        if not row:
            return False
        meta = json.loads(row.extra_metadata or "{}")
        meta["match_count"] = meta.get("match_count", 0) + 1
        row.extra_metadata = json.dumps(meta, ensure_ascii=False)
        row.last_seen_at = _now_iso()
        session.flush()
        return True

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    @staticmethod
    def _tie_break_score(cand: IconMemoryMatch, query_page_class: str = "") -> float:
        """Score for tie-breaking when dHash distances are equal.

        Higher is better. Factors:
        - page_class match: +0.3
        - region_role present: +0.1
        - confirmed state: +0.2
        - higher match_count: +0.1 * log2(match_count+1)
        """
        import math
        score = 0.0
        meta = cand.meta
        if query_page_class and meta.get("page_class") == query_page_class:
            score += 0.3
        if meta.get("region_role"):
            score += 0.1
        if meta.get("semantic_state") == "confirmed":
            score += 0.2
        mc = meta.get("match_count", 0)
        if mc > 0:
            score += 0.1 * math.log2(mc + 1)
        return score

    def stats(self, session: Any, app_process: str = "") -> dict[str, Any]:
        """Aggregate statistics."""
        q = session.query(VisualAssetRecord).filter_by(asset_type=ASSET_TYPE)
        if app_process:
            q = q.filter_by(app_id=app_process)
        rows = q.all()

        total = len(rows)
        by_state: dict[str, int] = {}
        by_app: dict[str, int] = {}
        by_privacy: dict[str, int] = {}

        for r in rows:
            meta = json.loads(r.extra_metadata or "{}")
            st = meta.get("semantic_state", "unknown")
            by_state[st] = by_state.get(st, 0) + 1
            app = r.app_id or "unknown"
            by_app[app] = by_app.get(app, 0) + 1
            pl = meta.get("privacy_level", "unknown")
            by_privacy[pl] = by_privacy.get(pl, 0) + 1

        return {
            "total": total,
            "by_state": by_state,
            "by_app": by_app,
            "by_privacy": by_privacy,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: VisualAssetRecord) -> dict[str, Any]:
        return {
            "asset_id": row.asset_id,
            "asset_type": row.asset_type,
            "app_id": row.app_id,
            "path": row.path,
            "width": row.width,
            "height": row.height,
            "format": row.format,
            "file_size": row.file_size,
            "dhash": row.dhash,
            "source_provider": row.source_provider,
            "created_at": row.created_at,
            "last_seen_at": row.last_seen_at,
            "status": row.status,
            "extra_metadata": row.extra_metadata,
            "bounds_json": row.bounds_json,
            "relative_bounds_json": row.relative_bounds_json,
            "window_size_json": row.window_size_json,
            "canvas_id": row.canvas_id,
            "region_id": row.region_id,
        }
