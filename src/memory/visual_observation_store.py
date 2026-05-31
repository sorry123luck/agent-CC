"""
VisualObservationStore — 视觉观察记录 CRUD。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.storage.schema import VisualObservationRecord

_VO_ALLOWED_CREATE_FIELDS = frozenset({
    "asset_id_b", "stable_key_id", "canvas_id_a", "canvas_id_b",
    "observed_canvas_id", "observed_bounds_json", "observed_crop_path",
    "dhash_distance", "ssim_score", "template_score", "coordinate_drift",
    "roi_bounds_json", "method", "extra_metadata",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: VisualObservationRecord) -> dict:
    return {
        "observation_id": row.observation_id,
        "asset_id_a": row.asset_id_a,
        "asset_id_b": row.asset_id_b,
        "stable_key_id": row.stable_key_id,
        "canvas_id_a": row.canvas_id_a,
        "canvas_id_b": row.canvas_id_b,
        "observed_canvas_id": row.observed_canvas_id,
        "observed_bounds_json": row.observed_bounds_json,
        "observed_crop_path": row.observed_crop_path,
        "dhash_distance": row.dhash_distance,
        "ssim_score": row.ssim_score,
        "template_score": row.template_score,
        "coordinate_drift": row.coordinate_drift,
        "match_status": row.match_status,
        "roi_bounds_json": row.roi_bounds_json,
        "compared_at": row.compared_at,
        "method": row.method,
        "metadata": row.extra_metadata,
    }


class VisualObservationStore:
    """视觉观察记录存储。"""

    def create_observation(
        self,
        session,
        asset_id_a: str,
        match_status: str,
        **kwargs,
    ) -> str:
        # Map API-level 'metadata' to ORM column 'extra_metadata'
        if "metadata" in kwargs:
            kwargs["extra_metadata"] = kwargs.pop("metadata")
        filtered = {k: v for k, v in kwargs.items() if k in _VO_ALLOWED_CREATE_FIELDS}
        observation_id = uuid.uuid4().hex
        row = VisualObservationRecord(
            observation_id=observation_id,
            asset_id_a=asset_id_a,
            match_status=match_status,
            compared_at=_now_iso(),
            **filtered,
        )
        session.add(row)
        session.flush()
        return observation_id

    def get_observation(self, session, observation_id: str) -> dict | None:
        row = session.query(VisualObservationRecord).filter_by(observation_id=observation_id).first()
        return _row_to_dict(row) if row else None

    def list_by_stable_key(self, session, stable_key_id: str, limit: int = 50) -> list[dict]:
        rows = (
            session.query(VisualObservationRecord)
            .filter_by(stable_key_id=stable_key_id)
            .order_by(VisualObservationRecord.compared_at.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def list_by_asset_pair(self, session, asset_id_a: str, asset_id_b: str) -> list[dict]:
        rows = (
            session.query(VisualObservationRecord)
            .filter_by(asset_id_a=asset_id_a, asset_id_b=asset_id_b)
            .order_by(VisualObservationRecord.compared_at.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def get_latest_match(self, session, stable_key_id: str) -> dict | None:
        row = (
            session.query(VisualObservationRecord)
            .filter_by(stable_key_id=stable_key_id)
            .order_by(VisualObservationRecord.compared_at.desc())
            .first()
        )
        return _row_to_dict(row) if row else None

    def delete_old_observations(self, session, before_date: str) -> int:
        count = (
            session.query(VisualObservationRecord)
            .filter(VisualObservationRecord.compared_at < before_date)
            .delete()
        )
        session.flush()
        return count
