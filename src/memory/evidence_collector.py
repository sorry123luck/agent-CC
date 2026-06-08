"""
EvidenceCollector — 统一收集 observe 管线中的 evidence。

只在 api_server._do_observe() 中调用。
异常 catch+log，不影响 observe 主流程。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.memory.evidence_store import EvidenceStore

_logger = logging.getLogger(__name__)

# VLM 属性检测常量
# 与 vlm_semantic_bridge.py 的命名约定耦合：
#   - Region attributes: vlm_purpose, vlm_region_id, vlm_role（_merge_regions 写入）
#   - App attributes: vlm_app_name（_apply_app_identity 写入）
# 如果 bridge 改变命名约定，需同步更新此处。
_VLM_ATTR_PREFIX = "vlm_"
_VLM_APP_NAME_KEY = "vlm_app_name"
_PROVIDER_ALIASES = {
    "vision": "omni",
    "omniparser": "omni",
    "omniparser_remote": "omni",
    "paddleocr_bridge": "ocr",
    "ocr_bridge": "ocr",
    "app_layout": "uia",
    "boundary_candidate": "uia",
}


def _normalize_provider(source: str) -> str:
    return _PROVIDER_ALIASES.get(str(source), str(source))


class EvidenceCollector:
    """统一收集 observe 管线中的 evidence。"""

    def __init__(self, evidence_store: EvidenceStore | None = None):
        self._store = evidence_store or EvidenceStore()

    def collect(
        self,
        session: Any,
        canvas: Any,
        *,
        vlm_used: bool = False,
    ) -> int:
        """从 canvas 收集 evidence 写入 candidate_evidence 表。

        Args:
            session: DB session（由调方管理生命周期）
            canvas: InteractionCanvas
            vlm_used: 本次 observe 是否使用了 VLM（由调用方传入）

        Returns:
            写入条数
        """
        count = 0

        # 1. 候选级 evidence：每个 element × 每个 provider_source
        count += self._collect_candidate_evidence(session, canvas)

        # 2. 区域级 evidence：VLM 标记的区域
        if vlm_used:
            count += self._collect_region_evidence(session, canvas)

        # 3. 页面级 evidence：VLM app_identity
        if vlm_used:
            count += self._collect_page_evidence(session, canvas)

        return count

    def _collect_candidate_evidence(self, session: Any, canvas: Any) -> int:
        """收集候选级 evidence（每个 element × 每个 provider_source）。"""
        count = 0
        for element in canvas.elements:
            provider_sources = getattr(element, "provider_sources", None)
            if not provider_sources:
                continue
            for source in provider_sources:
                try:
                    provider = _normalize_provider(source)
                    semantic_role = getattr(element, "semantic_role", None)
                    if semantic_role is not None:
                        semantic_role = getattr(semantic_role, "value", str(semantic_role))

                    self._store.create_evidence(
                        session,
                        provider=provider,
                        evidence_scope="candidate",
                        evidence_event="observe",
                        canvas_id=canvas.canvas_id,
                        element_id=element.element_id,
                        stable_key_id=getattr(element, "stable_key_id", None),
                        bounds_json=json.dumps(element.bounds) if element.bounds else None,
                        text=element.text or None,
                        semantic_role=semantic_role,
                        control_type=element.control_type or None,
                        raw_confidence=element.confidence,
                        region_id=getattr(element, "region_id", None),
                        page_model_id=getattr(canvas, "page_model_id", None),
                        state_template_id=getattr(canvas, "state_template_id", None),
                    )
                    count += 1
                except Exception as exc:
                    _logger.warning(
                        "evidence write failed for element=%s source=%s: %s",
                        element.element_id, source, exc,
                    )
        return count

    def _collect_region_evidence(self, session: Any, canvas: Any) -> int:
        """收集区域级 evidence（VLM 标记的区域）。"""
        count = 0
        for region in canvas.regions:
            attrs = getattr(region, "attributes", None) or {}
            is_vlm_region = any(k.startswith(_VLM_ATTR_PREFIX) for k in attrs)
            if not is_vlm_region:
                continue
            try:
                self._store.create_evidence(
                    session,
                    provider="vlm",
                    evidence_scope="region",
                    evidence_event="vlm_semantic",
                    canvas_id=canvas.canvas_id,
                    region_id=region.region_id,
                    page_model_id=getattr(canvas, "page_model_id", None),
                    state_template_id=getattr(canvas, "state_template_id", None),
                )
                count += 1
            except Exception as exc:
                _logger.warning(
                    "evidence write failed for region=%s: %s",
                    region.region_id, exc,
                )
        return count

    def _collect_page_evidence(self, session: Any, canvas: Any) -> int:
        """收集页面级 evidence（VLM app_identity）。"""
        app = getattr(canvas, "app", None)
        app_attrs = getattr(app, "attributes", None) if app else None
        has_vlm_app = app_attrs and app_attrs.get(_VLM_APP_NAME_KEY)
        if not has_vlm_app:
            return 0
        try:
            self._store.create_evidence(
                session,
                provider="vlm",
                evidence_scope="page",
                evidence_event="vlm_semantic",
                canvas_id=canvas.canvas_id,
                page_model_id=getattr(canvas, "page_model_id", None),
                state_template_id=getattr(canvas, "state_template_id", None),
            )
            return 1
        except Exception as exc:
            _logger.warning("evidence write failed for page-level vlm: %s", exc)
            return 0
