"""
Candidate Key Store — SQLite 持久化 StableCandidateKey 和 CandidateTemplate。

Phase 1 核心模块：管理候选的持久身份。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.memory.candidate_identity import (
    CandidateSignature,
    StableCandidateKey,
    compute_signature_similarity,
)
from src.storage.schema import CandidateTemplate, StableCandidateKey as StableCandidateKeyModel, CandidateStateRecord

MATCH_THRESHOLD = 0.75


@dataclass
class CandidateTemplateData:
    """模板数据（从 DB 模型转换）。"""
    key_id: str
    app_id: str
    page_class: str
    region_id: str
    region_role: str
    relative_bounds: tuple[float, float, float, float]
    size_ratio: float
    text_normalized: str
    text_hash: str | None
    visual_type: str
    role_label: str | None
    semantic_tags: list[str]
    crop_hash: str | None
    provider_sources: list[str]
    confidence: float
    is_fixed_control: bool
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CandidateKeyStore:
    """候选持久身份存储。"""

    def find_or_create(
        self,
        session,
        app_id: str,
        page_class: str,
        signature: CandidateSignature,
        force: bool = False,
    ) -> tuple[StableCandidateKey | None, str]:
        """查找或创建 StableCandidateKey。

        Args:
            session: SQLAlchemy session
            app_id: 应用标识
            page_class: 页面类型
            signature: 候选签名
            force: True 时即使 transient 也写入（Agent 明确 feedback 场景）

        Returns:
            (StableCandidateKey | None, "reused" | "new" | "transient")
        """
        # bounds 无效的候选不持久化（防止零坐标污染稳定身份池）
        if not signature.has_valid_bounds:
            return None, "transient"

        # transient candidates 不持久化（除非 force）
        if not signature.is_fixed_control and not force:
            return None, "transient"

        # 查找同 (app_id, page_class) 下所有 key
        rows = session.query(StableCandidateKeyModel).filter(
            StableCandidateKeyModel.app_id == app_id,
            StableCandidateKeyModel.page_class == page_class,
        ).all()

        # 构建签名并计算相似度
        best_key = None
        best_sim = 0.0

        for row in rows:
            existing_sig = self._row_to_signature(row, signature)
            sim = compute_signature_similarity(signature, existing_sig)
            if sim > best_sim:
                best_sim = sim
                best_key = row

        if best_key and best_sim >= MATCH_THRESHOLD:
            # 固定控件二次校验：防止同 role 不同位置的控件被错误归并
            if signature.is_fixed_control:
                existing_sig = self._row_to_signature(best_key, signature)
                # 空文本不算 exact match（否则两个 "" 会误合并）
                has_text = bool(signature.text_normalized and existing_sig.text_normalized)
                text_exact = has_text and signature.text_normalized == existing_sig.text_normalized
                bounds_close = _bounds_close(signature.relative_bounds, existing_sig.relative_bounds)
                if not text_exact and not bounds_close:
                    # 同 role 但不同文本且不同位置 → 创建新 key
                    pass
                    # fall through to create new key below
                else:
                    best_key.last_seen_at = _now_iso()
                    best_key.verify_count = (best_key.verify_count or 0) + 1
                    session.flush()
                    return self._row_to_key(best_key), "reused"
            else:
                # 动态内容：直接复用
                best_key.last_seen_at = _now_iso()
                best_key.verify_count = (best_key.verify_count or 0) + 1
                session.flush()
                return self._row_to_key(best_key), "reused"

        # 创建新 key
        key_id = str(uuid.uuid4())
        now = _now_iso()

        # canonical_text：固定控件存原文，动态内容存 hash
        if signature.is_fixed_control:
            canonical_text = signature.text_normalized
        else:
            canonical_text = signature.text_normalized  # 已经是 hash

        new_row = StableCandidateKeyModel(
            key_id=key_id,
            app_id=app_id,
            page_class=page_class,
            canonical_region=signature.region_id,
            canonical_text=canonical_text,
            canonical_role=signature.region_role,
            created_at=now,
            last_seen_at=now,
            verify_count=1,
            latest_bounds=json.dumps(signature.relative_bounds) if signature.relative_bounds else None,
            latest_crop_hash=signature.crop_hash or None,
            # Phase 6: explicit initial state
            permanence_state="new",
            state_changed_at=now,
            fail_count=0,
            coordinate_drift=0.0,
        )
        session.add(new_row)
        session.flush()

        # 同时创建 candidate_template
        self._create_template(session, key_id, app_id, page_class, signature)

        return self._row_to_key(new_row), "new"

    def get_templates_for_key(
        self,
        session,
        key_id: str,
    ) -> list[CandidateTemplateData]:
        """获取某个 key 的所有模板。"""
        rows = session.query(CandidateTemplate).filter(
            CandidateTemplate.key_id == key_id,
        ).all()
        return [self._template_row_to_data(r) for r in rows]

    def upsert_template(
        self,
        session,
        key_id: str,
        app_id: str,
        page_class: str,
        signature: CandidateSignature,
    ) -> CandidateTemplateData:
        """更新或插入模板（同一 key 只保留最新模板）。"""
        existing = session.query(CandidateTemplate).filter(
            CandidateTemplate.key_id == key_id,
        ).first()

        now = _now_iso()

        if existing:
            self._update_template_row(existing, signature, now)
            session.flush()
            return self._template_row_to_data(existing)

        return self._create_template(session, key_id, app_id, page_class, signature)

    def _create_template(
        self,
        session,
        key_id: str,
        app_id: str,
        page_class: str,
        signature: CandidateSignature,
    ) -> CandidateTemplateData:
        now = _now_iso()

        # 隐私处理
        if signature.is_fixed_control:
            text_norm = signature.text_normalized
            text_hash_val = None
        else:
            text_norm = signature.text_normalized  # 已经是 hash
            text_hash_val = signature.text_normalized

        row = CandidateTemplate(
            key_id=key_id,
            app_id=app_id,
            page_class=page_class,
            region_id=signature.region_id,
            region_role=signature.region_role,
            relative_bounds=json.dumps(signature.relative_bounds),
            size_ratio=signature.size_ratio,
            text_normalized=text_norm,
            text_hash=text_hash_val,
            visual_type=signature.visual_type,
            role_label=signature.role_label,
            semantic_tags=json.dumps(list(signature.semantic_tags)) if signature.semantic_tags else None,
            crop_hash=signature.crop_hash or None,
            provider_sources=json.dumps(list(signature.provider_sources)) if signature.provider_sources else None,
            confidence=signature.confidence,
            is_fixed_control=1 if signature.is_fixed_control else 0,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        return self._template_row_to_data(row)

    def _update_template_row(
        self,
        row: CandidateTemplate,
        signature: CandidateSignature,
        now: str,
    ) -> None:
        row.region_id = signature.region_id
        row.region_role = signature.region_role
        row.relative_bounds = json.dumps(signature.relative_bounds)
        row.size_ratio = signature.size_ratio

        if signature.is_fixed_control:
            row.text_normalized = signature.text_normalized
            row.text_hash = None
        else:
            row.text_normalized = signature.text_normalized
            row.text_hash = signature.text_normalized

        row.visual_type = signature.visual_type
        row.role_label = signature.role_label
        row.semantic_tags = json.dumps(list(signature.semantic_tags)) if signature.semantic_tags else None
        row.crop_hash = signature.crop_hash or None
        row.provider_sources = json.dumps(list(signature.provider_sources)) if signature.provider_sources else None
        row.confidence = signature.confidence
        row.is_fixed_control = 1 if signature.is_fixed_control else 0
        row.updated_at = now

    def _row_to_key(self, row: StableCandidateKeyModel) -> StableCandidateKey:
        return StableCandidateKey(
            key_id=row.key_id,
            app_id=row.app_id,
            page_class=row.page_class,
            created_at=row.created_at,
            canonical_region=row.canonical_region,
            canonical_text=row.canonical_text,
            canonical_role=row.canonical_role,
            permanence_state=getattr(row, "permanence_state", None) or "new",
            state_changed_at=getattr(row, "state_changed_at", None),
            fail_count=getattr(row, "fail_count", None) or 0,
            coordinate_drift=getattr(row, "coordinate_drift", None) or 0.0,
            latest_profile_id=getattr(row, "latest_profile_id", None),
        )

    def _row_to_signature(
        self,
        row: StableCandidateKeyModel,
        ref: CandidateSignature,
    ) -> CandidateSignature:
        """从 DB 行构建签名（用于相似度比较）。

        注意：DB 中没有完整签名信息，用 ref 的部分字段填充。
        """
        rel_bounds = json.loads(row.latest_bounds) if row.latest_bounds else (0.0, 0.0, 0.0, 0.0)
        rel_tuple = tuple(rel_bounds)
        # 验证 DB 中的 bounds 是否有效
        x1, y1, x2, y2 = rel_tuple if len(rel_tuple) >= 4 else (0.0, 0.0, 0.0, 0.0)
        has_valid = not (x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0) and x2 > x1 and y2 > y1
        return CandidateSignature(
            region_id=row.canonical_region,
            region_role=row.canonical_role,
            relative_bounds=rel_tuple,
            size_ratio=ref.size_ratio,  # DB 不存 size_ratio，用 ref
            text_normalized=row.canonical_text,
            visual_type=ref.visual_type,  # DB 不存 visual_type，用 ref
            role_label=ref.role_label,
            semantic_tags=ref.semantic_tags,
            crop_hash=row.latest_crop_hash or "",
            provider_sources=ref.provider_sources,
            confidence=ref.confidence,
            is_fixed_control=_is_fixed_from_role(row.canonical_role),
            has_valid_bounds=has_valid,
        )

    def _template_row_to_data(self, row: CandidateTemplate) -> CandidateTemplateData:
        return CandidateTemplateData(
            key_id=row.key_id,
            app_id=row.app_id,
            page_class=row.page_class,
            region_id=row.region_id,
            region_role=row.region_role,
            relative_bounds=tuple(json.loads(row.relative_bounds)),
            size_ratio=row.size_ratio,
            text_normalized=row.text_normalized,
            text_hash=row.text_hash,
            visual_type=row.visual_type,
            role_label=row.role_label,
            semantic_tags=json.loads(row.semantic_tags) if row.semantic_tags else [],
            crop_hash=row.crop_hash,
            provider_sources=json.loads(row.provider_sources) if row.provider_sources else [],
            confidence=row.confidence,
            is_fixed_control=bool(row.is_fixed_control),
            updated_at=row.updated_at,
        )

    def get_candidates_for_state(
        self,
        session,
        state_template_id: str,
    ) -> list[VirtualCandidateData]:
        """获取某个 StateTemplate 下的所有候选（JOIN candidate_states + candidate_templates + stable_candidate_keys）。

        不依赖 CanvasCache，纯 SQLite 查询。
        """
        rows = (
            session.query(
                StableCandidateKeyModel,
                CandidateTemplate,
                CandidateStateRecord,
            )
            .join(CandidateStateRecord, CandidateStateRecord.key_id == StableCandidateKeyModel.key_id)
            .join(CandidateTemplate, CandidateTemplate.key_id == StableCandidateKeyModel.key_id)
            .filter(CandidateStateRecord.state_template_id == state_template_id)
            .all()
        )

        results = []
        for key_row, tmpl_row, state_row in rows:
            rel_bounds = json.loads(tmpl_row.relative_bounds) if tmpl_row.relative_bounds else [0.0, 0.0, 0.0, 0.0]
            results.append(VirtualCandidateData(
                key_id=key_row.key_id,
                canonical_text=key_row.canonical_text or "",
                canonical_role=key_row.canonical_role or "",
                canonical_region=key_row.canonical_region or "",
                role_label=tmpl_row.role_label,
                semantic_tags=json.loads(tmpl_row.semantic_tags) if tmpl_row.semantic_tags else [],
                visual_type=tmpl_row.visual_type or "",
                relative_bounds=rel_bounds,
                confidence=tmpl_row.confidence or 0.0,
                provider_sources=json.loads(tmpl_row.provider_sources) if tmpl_row.provider_sources else [],
                is_fixed_control=bool(tmpl_row.is_fixed_control),
                verify_count=key_row.verify_count or 1,
                seen_count=state_row.seen_count or 1,
                permanence_state=getattr(key_row, "permanence_state", None) or "new",
            ))

        return results


def _is_fixed_from_role(role: str) -> bool:
    from src.memory.candidate_identity import FIXED_CONTROL_ROLES
    return role in FIXED_CONTROL_ROLES


def _bounds_close(a: tuple[float, ...], b: tuple[float, ...], threshold: float = 0.05) -> bool:
    """判断两个相对坐标是否足够接近（中心点距离 < threshold）。

    全零 bounds 视为无效，不认为是"接近"。
    """
    if not a or not b or len(a) < 4 or len(b) < 4:
        return False

    # 全零 bounds → 无效，不认为接近
    is_a_zero = all(v == 0.0 for v in a[:4])
    is_b_zero = all(v == 0.0 for v in b[:4])
    if is_a_zero or is_b_zero:
        return False

    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return dist < threshold


@dataclass
class VirtualCandidateData:
    """A candidate enriched with key + template + state data for virtual model rendering."""
    key_id: str
    canonical_text: str
    canonical_role: str
    canonical_region: str
    role_label: str | None
    semantic_tags: list[str]
    visual_type: str
    relative_bounds: list[float]
    confidence: float
    provider_sources: list[str]
    is_fixed_control: bool
    verify_count: int
    seen_count: int
    permanence_state: str = "new"
