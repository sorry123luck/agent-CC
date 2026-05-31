"""
EvidenceFusionEngine — Evidence → ConfidenceProfile 融合引擎。

在 api_server._do_observe() 中 EvidenceCollector.collect() 之后调用。
异常 catch+log，不影响 observe 主流程。
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.evidence_store import EvidenceStore
from src.memory.confidence_profile_store import ConfidenceProfileStore
from src.memory.evidence_fusion import (
    ConfidenceProfile,
    compute_from_evidence,
    uia_structure_score,
)

_logger = logging.getLogger(__name__)


class EvidenceFusionEngine:
    """Evidence → ConfidenceProfile 融合引擎。"""

    def __init__(
        self,
        evidence_store: EvidenceStore | None = None,
        profile_store: ConfidenceProfileStore | None = None,
    ):
        self._evidence_store = evidence_store or EvidenceStore()
        self._profile_store = profile_store or ConfidenceProfileStore()

    def compute_for_element(
        self,
        session: Any,
        element: Any,
        canvas: Any,
    ) -> ConfidenceProfile | None:
        """为单个 element 计算 + 持久化 ConfidenceProfile。

        无 stable_key_id 或无 evidence 返回 None。
        """
        stable_key_id = getattr(element, "stable_key_id", None)
        if not stable_key_id:
            return None

        summary = self._evidence_store.aggregate_summary(session, stable_key_id)
        if not summary:
            return None

        uia_quality = self._compute_uia_quality(element)
        profile = compute_from_evidence(summary, uia_quality=uia_quality)

        self._profile_store.create_profile(
            session,
            stable_key_id=stable_key_id,
            fused_confidence=profile.fused_confidence,
            formula_version=profile.formula_version,
            canvas_id=canvas.canvas_id,
            page_model_id=getattr(canvas, "page_model_id", None),
            state_template_id=getattr(canvas, "state_template_id", None),
            spatial_confidence=profile.spatial_confidence,
            semantic_confidence=profile.semantic_confidence,
            text_confidence=profile.text_confidence,
            structure_confidence=profile.structure_confidence,
            visual_anchor_confidence=profile.visual_anchor_confidence,
            memory_confidence=profile.memory_confidence,
            action_confidence=profile.action_confidence,
            conflict_penalty=profile.conflict_penalty,
            source_count=profile.source_count,
            source_diversity=profile.source_diversity,
        )
        return profile

    def apply_to_canvas(self, session: Any, canvas: Any) -> int:
        """对所有有 stable_key_id 的 element 执行融合，更新 confidence。

        Returns:
            处理的 element 数
        """
        count = 0
        for element in canvas.elements:
            try:
                profile = self.compute_for_element(session, element, canvas)
                if profile is not None:
                    element.confidence = profile.fused_confidence
                    if not hasattr(element, "attributes") or element.attributes is None:
                        element.attributes = {}
                    element.attributes["confidence_profile"] = profile.to_dict()
                    count += 1
            except Exception as exc:
                _logger.warning(
                    "fusion failed for element=%s: %s",
                    getattr(element, "element_id", "?"), exc,
                )
        return count

    @staticmethod
    def _compute_uia_quality(element: Any) -> float:
        """从 element 属性计算 UIA structure_score。"""
        bounds = getattr(element, "bounds", None)
        bounds_valid = bounds is not None
        width = (bounds[2] - bounds[0]) if bounds_valid else 0
        height = (bounds[3] - bounds[1]) if bounds_valid else 0

        state = getattr(element, "state", None)
        enabled = getattr(state, "enabled", True) if state else True
        visible = getattr(state, "visible", True) if state else True

        has_name = bool(getattr(element, "name", None))
        attrs = getattr(element, "attributes", None) or {}
        has_automation_id = bool(attrs.get("automation_id"))

        return uia_structure_score(
            control_type=element.control_type or "",
            has_automation_id=has_automation_id,
            has_name=has_name,
            is_enabled=enabled,
            bounds_valid=bounds_valid,
            visible=visible,
            width=width,
            height=height,
        )
