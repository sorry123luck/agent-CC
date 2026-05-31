"""
Candidate Identity — 签名构建与相似度计算

Phase 1 核心模块：为每个候选构建持久身份。
- CandidateSignature：每次 observe 计算的快照签名
- StableCandidateKey：跨 observe 的持久身份
- build_signature / compute_signature_similarity：签名构建与相似度
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.perception.page_compiler_models import Candidate, InteractionCanvas

# 固定控件角色集合 — 位置和文字稳定，可长期追踪
FIXED_CONTROL_ROLES = frozenset({
    "button",
    "send_button",
    "submit_button",
    "cancel_button",
    "icon_button",
    "toggle_button",
    "message_input",
    "text_input",
    "search_input",
    "password_input",
    "file_input",
    "toolbar",
    "menu_bar",
    "title_bar",
    "status_bar",
    "sidebar",
    "nav_item",
    "tab",
    "menu_item",
    "link",
    "image",
})

# 动态内容角色集合 — 这些角色的原始文字不持久化
DYNAMIC_CONTENT_ROLES = frozenset({
    "message_content",
    "contact_name",
    "group_name",
    "notification_text",
    "search_result",
    "list_item",
    "chat_item",
    "tree_item",
    "text",
    "unknown",
})

# 动态内容区域 subtype：这些区域内的元素不应被视为固定功能控件
_DYNAMIC_CONTENT_SUBTYPES = frozenset({
    "chat",
    "editor",
    "canvas_doc_viewer",
    "timeline_media",
})


def _text_hash(text: str) -> str:
    """SHA-256 前 16 位，用于动态内容去重。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _is_fixed_control(semantic_role: str) -> bool:
    """判断是否为固定控件。

    逻辑：已知固定角色 → 固定；已知动态角色 → 动态；其他 → 动态（安全默认）。
    """
    if semantic_role in FIXED_CONTROL_ROLES:
        return True
    return False


def _normalize_text(text: str) -> str:
    """文本标准化：去首尾空白、转小写。"""
    return text.strip().lower() if text else ""


def _short_text_penalty(text: str) -> float:
    """短文本惩罚系数（0.0-1.0），越小惩罚越重。

    规则：
    - 纯数字 → 0.3
    - 纯符号 → 0.3
    - 长度 <= 3 → 0.4
    - 长度 <= 4 弱语义 → 0.5
    - 其他 → 1.0（无惩罚）
    """
    if not text:
        return 1.0

    # 纯数字
    if text.isdigit():
        return 0.3

    # 纯符号（无字母无数字）
    if not any(c.isalnum() for c in text):
        return 0.3

    # 长度 <= 3
    if len(text) <= 3:
        return 0.4

    # 长度 <= 4 弱语义（简单判断：无动词/名词特征的短文本）
    if len(text) <= 4:
        # 中文 1-2 字、英文 1-4 字母 → 弱语义
        ascii_chars = sum(1 for c in text if c.isascii())
        non_ascii = len(text) - ascii_chars
        if non_ascii >= 1 and len(text) <= 2:
            return 0.5
        if ascii_chars > 0 and non_ascii == 0 and len(text) <= 4:
            return 0.5

    return 1.0


def _crop_hash_similarity(a: str, b: str) -> float:
    """crop_hash 汉明距离相似度（简化：相同=1.0，不同=0.0）。

    Phase 2 可升级为感知哈希汉明距离。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 if a == b else 0.0


def _region_from_bounds(
    bounds: tuple[int, int, int, int],
    window_w: int,
    window_h: int,
) -> str:
    """从 bounds 和窗口尺寸推断区域 ID。"""
    x1, y1, x2, y2 = bounds
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 水平分区
    if cx < window_w * 0.15:
        h = "sidebar"
    elif cx < window_w * 0.4:
        h = "left_panel"
    else:
        h = "right_panel"

    # 垂直分区
    if y1 < 40:
        return "title_bar"
    if y2 < window_h * 0.12:
        v = "top"
    elif y1 > window_h * 0.85:
        v = "bottom"
    else:
        v = "middle"

    return f"{h}_{v}"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CandidateSignature:
    """候选签名 — 每次 observe 计算的快照。

    frozen=True 保证不可变，可安全用作 dict key 或 set member。
    """
    region_id: str
    region_role: str
    relative_bounds: tuple[float, float, float, float]  # (x1/W, y1/H, x2/W, y2/H)
    size_ratio: float               # 面积 / 窗口面积
    text_normalized: str            # 标准化文本（固定控件存原文，动态内容存 hash）
    visual_type: str
    role_label: str | None
    semantic_tags: tuple[str, ...]
    crop_hash: str
    provider_sources: tuple[str, ...]
    confidence: float
    is_fixed_control: bool
    has_valid_bounds: bool = True    # bounds 是否有效（零面积/非法坐标 → False）


@dataclass(frozen=True)
class StableCandidateKey:
    """持久候选身份 — 跨 observe 的稳定标识。"""
    key_id: str                     # UUID
    app_id: str
    page_class: str
    created_at: str                 # ISO format
    canonical_region: str
    canonical_text: str             # 固定控件存原文，动态内容存 text_hash
    canonical_role: str
    # Phase 6: permanence state machine
    permanence_state: str = "new"
    state_changed_at: str | None = None
    fail_count: int = 0
    coordinate_drift: float = 0.0
    latest_profile_id: str | None = None


# =============================================================================
# Signature Builder
# =============================================================================


def build_signature(
    candidate: Candidate,
    canvas: InteractionCanvas,
) -> CandidateSignature:
    """从候选 + 画布构建签名。

    Args:
        candidate: 候选点位
        canvas: 交互画布

    Returns:
        CandidateSignature
    """
    # 窗口尺寸
    w, h = _get_window_size(canvas)

    # 区域
    bounds = candidate.bounds or (0, 0, 0, 0)
    region_id = _region_from_bounds(bounds, w, h) if w and h else "unknown"

    # 相对坐标 + bounds 有效性验证
    has_valid = _validate_bounds(bounds, w, h)
    if w and h and bounds and has_valid:
        rel_bounds = (
            bounds[0] / w,
            bounds[1] / h,
            bounds[2] / w,
            bounds[3] / h,
        )
        # 验证相对坐标是否在合理范围
        if not _validate_relative_bounds(rel_bounds):
            has_valid = False
        size_ratio = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / max(w * h, 1)
    else:
        rel_bounds = (0.0, 0.0, 0.0, 0.0)
        size_ratio = 0.0

    # 语义角色
    role_str = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
    is_fixed = _is_fixed_control(role_str)

    # 动态内容区防护：如果元素在 content_area 且 subtype 是动态类型，
    # 即使 semantic_role 在 FIXED_CONTROL_ROLES 中，也强制为非固定控件。
    # 这防止聊天正文/文档正文里的关键词被当成固定功能按钮。
    if is_fixed and candidate.region_id:
        region = canvas.get_region(candidate.region_id)
        if region and region.role == "content_area":
            subtype_val = region.subtype.value if hasattr(region.subtype, "value") else str(region.subtype)
            if subtype_val in _DYNAMIC_CONTENT_SUBTYPES:
                is_fixed = False

    # 文本（动态内容存 hash）
    raw_text = _normalize_text(candidate.text or "")
    if is_fixed:
        text_for_sig = raw_text
    else:
        text_for_sig = _text_hash(raw_text) if raw_text else ""

    # semantic_tags
    tags = tuple(sorted(candidate.semantic_tags)) if candidate.semantic_tags else ()

    # provider_sources
    sources = tuple(sorted(candidate.provider_sources)) if candidate.provider_sources else ()

    return CandidateSignature(
        region_id=region_id,
        region_role=role_str,
        relative_bounds=rel_bounds,
        size_ratio=size_ratio,
        text_normalized=text_for_sig,
        visual_type=candidate.visual_type or "",
        role_label=candidate.role_label,
        semantic_tags=tags,
        crop_hash="",  # Phase 1: crop_hash 暂未实现，留空
        provider_sources=sources,
        confidence=candidate.confidence,
        is_fixed_control=is_fixed,
        has_valid_bounds=has_valid,
    )


def _validate_bounds(
    bounds: tuple[int, int, int, int],
    window_w: int,
    window_h: int,
) -> bool:
    """验证原始像素 bounds 是否有效。

    无效条件：
    - 窗口尺寸未知（w=0 或 h=0）→ 无法计算相对坐标，视为无效
    - 零面积（x2<=x1 或 y2<=y1）
    - 全零坐标
    - 坐标超出窗口范围过多（允许 20% margin）
    """
    x1, y1, x2, y2 = bounds

    # 窗口尺寸未知 → 无法计算有意义的相对坐标
    if window_w <= 0 or window_h <= 0:
        return False

    # 全零 → 无效
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return False

    # 零面积 → 无效
    if x2 <= x1 or y2 <= y1:
        return False

    margin = max(window_w, window_h) * 0.2

    # 中心点超出窗口过多 → 无效
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < -margin or cx > window_w + margin:
        return False
    if cy < -margin or cy > window_h + margin:
        return False

    return True


def _validate_relative_bounds(
    rel_bounds: tuple[float, float, float, float],
) -> bool:
    """验证相对坐标（0-1）是否有效。

    无效条件：
    - 全零 → 无效
    - 负值过大（<-0.1）→ 无效
    - 超出 1.0 过多（>1.2）→ 无效
    - 零面积（x2<=x1 或 y2<=y1）→ 无效
    """
    x1, y1, x2, y2 = rel_bounds

    # 全零 → 无效
    if x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0:
        return False

    # 零面积
    if x2 <= x1 or y2 <= y1:
        return False

    # 负值过大
    if x1 < -0.1 or y1 < -0.1:
        return False

    # 超出 1.0 过多
    if x2 > 1.2 or y2 > 1.2:
        return False

    return True


def _get_window_size(canvas: InteractionCanvas) -> tuple[int, int]:
    """从 canvas 获取窗口宽高。

    优先级：rect_client → rect_screen → 元素 bounds 并集
    """
    if canvas.window:
        if canvas.window.rect_client:
            x1, y1, x2, y2 = canvas.window.rect_client
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                return w, h
        if canvas.window.rect_screen:
            x1, y1, x2, y2 = canvas.window.rect_screen
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                return w, h

    # 从元素 bounds 推断窗口大小
    if canvas.elements:
        max_x, max_y = 0, 0
        for elem in canvas.elements:
            if elem.bounds and len(elem.bounds) >= 4:
                max_x = max(max_x, elem.bounds[2])
                max_y = max(max_y, elem.bounds[3])
        if max_x > 0 and max_y > 0:
            return max_x, max_y

    return 0, 0


# =============================================================================
# Similarity Computation
# =============================================================================

# 固定控件权重
_FIXED_WEIGHTS = {
    "region_bounds": 0.35,  # region + bounds
    "text": 0.10,           # 文本（降低，固定控件不依赖文本）
    "role": 0.15,
    "vt": 0.10,
    "crop": 0.10,
    "source": 0.05,
    "confidence": 0.05,
    # subtotal = 0.90，剩余 0.10 给 region_bounds（已含）
}
# 重新分配：确保总和 = 1.0
_FIXED_WEIGHTS = {
    "region_bounds": 0.40,
    "text": 0.08,
    "role": 0.15,
    "vt": 0.10,
    "crop": 0.10,
    "source": 0.05,
    "confidence": 0.05,
    "size": 0.07,
}

# 动态内容权重
_DYNAMIC_WEIGHTS = {
    "region_bounds": 0.25,
    "text": 0.30,
    "role": 0.15,
    "vt": 0.05,
    "crop": 0.05,
    "source": 0.05,
    "confidence": 0.05,
    "size": 0.10,
}

# 标准权重（通用）
_STANDARD_WEIGHTS = {
    "region_bounds": 0.30,
    "text": 0.25,
    "role": 0.15,
    "vt": 0.10,
    "crop": 0.10,
    "source": 0.05,
    "confidence": 0.05,
    "size": 0.00,  # 标准模式不使用 size
}


def compute_signature_similarity(
    a: CandidateSignature,
    b: CandidateSignature,
) -> float:
    """计算两个签名的相似度（0.0-1.0）。

    权重策略：
    - 固定控件：region+bounds+crop 权重高，text 权重低
    - 动态内容：text 权重高，但短文本惩罚 + text 不匹配时上限 0.6
    - 混合（一个固定一个动态）：使用标准权重
    """
    # 选择权重策略
    if a.is_fixed_control and b.is_fixed_control:
        weights = _FIXED_WEIGHTS
    elif not a.is_fixed_control and not b.is_fixed_control:
        weights = _DYNAMIC_WEIGHTS
    else:
        weights = _STANDARD_WEIGHTS

    scores: dict[str, float] = {}

    # 1. region + bounds (region 相同 + bounds 距离)
    region_match = 1.0 if a.region_id == b.region_id else 0.0
    ra, rb = a.relative_bounds, b.relative_bounds
    ca = ((ra[0] + ra[2]) / 2, (ra[1] + ra[3]) / 2)
    cb = ((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    dist_sim = max(0, 1.0 - dist * 5)
    scores["region_bounds"] = region_match * 0.5 + dist_sim * 0.5

    # 2. text
    ta, tb = a.text_normalized, b.text_normalized
    if ta and tb:
        text_sim = SequenceMatcher(None, ta, tb).ratio()
        # 短文本惩罚（仅动态内容）
        if not a.is_fixed_control or not b.is_fixed_control:
            # 用原始文本长度判断惩罚（hash 后无意义）
            # 这里用 text_normalized 长度近似
            penalty = min(_short_text_penalty(ta), _short_text_penalty(tb))
            text_sim *= penalty
    elif not ta and not tb:
        text_sim = 1.0
    else:
        text_sim = 0.0
    scores["text"] = text_sim

    # 3. semantic_role
    scores["role"] = 1.0 if a.region_role == b.region_role else 0.0

    # 4. visual_type
    scores["vt"] = 1.0 if a.visual_type == b.visual_type else 0.0

    # 5. crop_hash
    scores["crop"] = _crop_hash_similarity(a.crop_hash, b.crop_hash)

    # 6. provider_sources
    if a.provider_sources and b.provider_sources:
        common = len(set(a.provider_sources) & set(b.provider_sources))
        total = len(set(a.provider_sources) | set(b.provider_sources))
        scores["source"] = common / max(total, 1)
    elif not a.provider_sources and not b.provider_sources:
        scores["source"] = 1.0
    else:
        scores["source"] = 0.0

    # 7. confidence
    scores["confidence"] = max(0, 1 - abs(a.confidence - b.confidence) * 2)

    # 8. size_ratio
    if weights.get("size", 0) > 0:
        size_sim = 1 - min(abs(a.size_ratio - b.size_ratio) * 10, 1.0)
        scores["size"] = size_sim
    else:
        scores["size"] = 0.0

    # 加权求和
    total = sum(scores[k] * weights[k] for k in weights if k in scores)

    # 动态内容 text 不匹配时上限
    if not a.is_fixed_control and not b.is_fixed_control:
        if ta and tb and scores["text"] < 0.8:
            if scores["text"] < 0.2:
                total = min(total, 0.4)
            elif scores["text"] < 0.6:
                total = min(total, 0.6)
            else:
                total = min(total, 0.8)

    return min(total, 1.0)
