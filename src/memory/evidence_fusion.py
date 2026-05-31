"""
Evidence Fusion — ConfidenceProfile 计算。

维度加权公式 v1_dim_weighted：
  0.24×spatial + 0.24×semantic + 0.14×visual_anchor + 0.14×memory
  + 0.09×text + 0.09×structure + 0.06×action - conflict_penalty
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# 权重配置
_WEIGHTS = {
    "spatial": 0.24,
    "semantic": 0.24,
    "visual_anchor": 0.14,
    "memory": 0.14,
    "text": 0.09,
    "structure": 0.09,
    "action": 0.06,
}

# UIA control_type 分层
_CORE_CONTROLS = {"Button", "Edit", "CheckBox", "RadioButton", "ComboBox"}
_CONTAINER_CONTROLS = {"Pane", "Group", "Window", "TabControl"}
_TEXT_CONTROLS = {"Text", "Label", "Hyperlink"}


@dataclass(frozen=True)
class ConfidenceProfile:
    spatial_confidence: float = 0.0
    semantic_confidence: float = 0.0
    text_confidence: float = 0.0
    structure_confidence: float = 0.0
    visual_anchor_confidence: float = 0.0
    memory_confidence: float = 0.0
    action_confidence: float = 0.0
    conflict_penalty: float = 0.0
    formula_version: str = "v1_dim_weighted"
    source_count: int = 0
    source_diversity: int = 0

    @property
    def fused_confidence(self) -> float:
        raw = (
            _WEIGHTS["spatial"] * self.spatial_confidence
            + _WEIGHTS["semantic"] * self.semantic_confidence
            + _WEIGHTS["visual_anchor"] * self.visual_anchor_confidence
            + _WEIGHTS["memory"] * self.memory_confidence
            + _WEIGHTS["text"] * self.text_confidence
            + _WEIGHTS["structure"] * self.structure_confidence
            + _WEIGHTS["action"] * self.action_confidence
            - self.conflict_penalty
        )
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fused_confidence"] = self.fused_confidence
        return d


def uia_structure_score(
    control_type: str,
    has_automation_id: bool,
    has_name: bool,
    is_enabled: bool,
    bounds_valid: bool,
    visible: bool = True,
    width: int = 0,
    height: int = 0,
) -> float:
    if not bounds_valid or not visible or width < 2 or height < 2:
        return 0.0

    if control_type in _CORE_CONTROLS:
        base = 0.85
    elif control_type in _CONTAINER_CONTROLS:
        base = 0.55
    elif control_type in _TEXT_CONTROLS:
        base = 0.70
    else:
        base = 0.30

    factor = 1.0
    if has_automation_id:
        factor *= 1.10
    if has_name:
        factor *= 1.05
    if not is_enabled:
        factor *= 0.70

    return min(1.0, base * factor)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def compute_from_evidence(
    evidence_summary: dict,
    uia_quality: float = 0.0,
    visual_anchor_confidence: float = 0.0,
    memory_confidence: float = 0.0,
    action_confidence: float = 0.0,
    conflict_penalty: float = 0.0,
) -> ConfidenceProfile:
    """从 evidence 聚合摘要计算 ConfidenceProfile。

    Args:
        evidence_summary: EvidenceStore.aggregate_summary() 的输出
        uia_quality: UIA structure_score（从 uia_structure_score() 计算）
        visual_anchor_confidence: 视觉锚点可信度（Phase 5 才有）
        memory_confidence: 历史稳定可信度
        action_confidence: 执行反馈可信度
        conflict_penalty: 冲突惩罚

    Returns:
        ConfidenceProfile（fused_confidence 通过 @property 计算）
    """
    def _best(key: str) -> float:
        entry = evidence_summary.get(key)
        if not isinstance(entry, dict):
            return 0.0
        return _safe_float(entry.get("best_confidence"), 0.0)

    def _count(key: str) -> int:
        entry = evidence_summary.get(key)
        if not isinstance(entry, dict):
            return 0
        return _safe_int(entry.get("count"), 0)

    # spatial: max(uia, omni)
    spatial = max(_best("uia"), _best("omni"))

    # semantic: VLM 为主
    semantic = _best("vlm")

    # text: max(ocr, uia, vlm)
    text = max(_best("ocr"), _best("uia"), _best("vlm"))

    # structure: UIA quality
    structure = _safe_float(uia_quality, 0.0)

    # action: from feedback evidence's best_action_score
    fb_entry = evidence_summary.get("feedback")
    if isinstance(fb_entry, dict):
        fb_action = _safe_float(fb_entry.get("best_action_score"), 0.0)
    else:
        fb_action = 0.0
    resolved_action = _safe_float(action_confidence) if action_confidence else fb_action

    # source counting
    sources = set(evidence_summary.keys())
    source_count = sum(_count(k) for k in evidence_summary)

    return ConfidenceProfile(
        spatial_confidence=spatial,
        semantic_confidence=semantic,
        text_confidence=text,
        structure_confidence=structure,
        visual_anchor_confidence=_safe_float(visual_anchor_confidence),
        memory_confidence=_safe_float(memory_confidence),
        action_confidence=resolved_action,
        conflict_penalty=_safe_float(conflict_penalty),
        source_count=source_count,
        source_diversity=len(sources),
    )
