"""VLM Semantic Modeler — Prompt 构造。"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from src.vlm.prompt_modes import (
    CANDIDATE_ANNOTATION,
    FULL_PAGE_RECOGNITION,
    ICON_CROP_UNDERSTANDING,
    MISSING_AUDIT,
    REGION_UNDERSTANDING,
)
from src.vlm.provider import VLMSemanticRequest


# ---------------------------------------------------------------------------
# Prompt Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptInput:
    """Prompt 构造的输入。"""
    screenshot: Image.Image
    numbered_overlay_image: Image.Image | None = None
    candidate_atlas_image: Image.Image | None = None
    omni_candidates: list[dict[str, Any]] | None = None
    ocr_texts: list[str] | None = None
    uia_elements: list[dict[str, Any]] | None = None
    current_page_model: dict[str, Any] | None = None
    historical_states: list[dict[str, Any]] | None = None
    task_hint: str = ""
    task_mode: str = CANDIDATE_ANNOTATION


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """你是一个桌面软件页面语义理解引擎。你的任务是尽可能**全面、详细**地分析截图中的所有可见 UI 元素，并输出结构化的页面语义 JSON。

## 重要原则

- **尽可能多地识别控件**：每个可见的按钮、图标、输入框、标签、菜单项、工具栏按钮、状态栏项都应单独列出
- 一个典型桌面应用页面通常有 **10-50 个可见控件**，不要只返回最显眼的几个
- 动态内容区域（聊天消息列表、文章正文、代码块等）标记为 dynamic_zone，但其中的固定控件（如消息操作按钮）仍要列出

## 输出格式

输出一个严格 JSON 对象，包含以下字段：

```json
{
  "schema_version": "1.0",
  "app_identity": {
    "app_name": "应用名称",
    "app_id": "应用标识",
    "surface_type": "native_uia | browser | electron_webview | canvas_self_drawn",
    "display_name": "用户可见的显示名称"
  },
  "page_state": {
    "page_class": "页面大类：home/chat/settings/editor/explorer/file_dialog/login/...",
    "state_label": "具体状态描述，如'聊天窗口-文本输入态'",
    "state_flags": ["has_input_focus", "has_unread", "modal_open", "has_error"]
  },
  "image_size": [{image_width}, {image_height}],
  "regions": [
    {
      "region_id": "r1",
      "role": "navigation | content | input | toolbar | status | sidebar | modal | top_bar | title_bar",
      "bounds": [left, top, right, bottom],
      "purpose": "区域功能描述"
    }
  ],
  "fixed_controls": [
    {
      "control_id": "ctrl_1",
      "region_id": "所在区域ID",
      "control_type": "button | input | tab | menu_item | icon | checkbox | dropdown | link | scrollbar | label | toggle | avatar | badge",
      "text": "控件上的文字（无文字则为空字符串）",
      "bounds": [left, top, right, bottom],
      "semantic_role": "语义角色，如 send_button / search_input / close_button / user_avatar / message_text / emoji_picker / settings_icon",
      "visual_type": "button | icon | text | input | toggle | badge | avatar",
      "interactable": true,
      "confidence": 0.9
    }
  ],
  "dynamic_zones": [
    {
      "zone_id": "dz1",
      "region_id": "所在区域ID",
      "content_type": "chat_messages | article | code | file_list | scrollable_list | message_area",
      "bounds": [left, top, right, bottom],
      "note": "内容类型和特征描述"
    }
  ],
  "candidate_corrections": [],
  "transitions": [],
  "confidence": 0.9,
  "needs_review": false
}
```

## 规则

1. bounds 坐标体系：必须是当前截图的像素坐标 [left, top, right, bottom]，原点=截图左上角。本截图尺寸为 {image_width}×{image_height}
2. right 最大只能等于 {image_width}，bottom 最大只能等于 {image_height}。任何 bounds 超出这个范围都是无效输出
3. 只允许识别当前截图中真实可见的区域和控件，不允许推测滚动列表下方、窗口外、被遮挡或不可见的元素
4. 如果某个控件只露出一部分，只返回可见部分的 bounds，并且必须在截图范围内
5. confidence 取值范围 0.0-1.0
6. regions 列表不能为空，至少要有 2 个区域
7. dynamic_zones 中不要包含原文内容，只描述内容类型
8. 如果有本地候选（source_candidate_ids），尽量匹配并给出修正建议
9. fixed_controls 要尽量全面：顶部导航栏的每个按钮、侧边栏的每个入口、内容区的每个可交互元素、底部状态栏的每个指示器都应列出
10. 只输出 JSON，不要输出其他文字"""


_CANDIDATE_ANNOTATION_PROMPT_TEMPLATE = """你是一个桌面软件 UI 语义标注引擎。你的任务不是重新检测控件坐标，而是在本地候选框证据约束下，为候选元素标注语义。

## 坐标和职责边界

- 你看到的图片尺寸是 {image_width}×{image_height}，坐标空间是 vlm_image。
- 输入候选 bounds 已经是 vlm_image 坐标。
- candidate_annotations 中不要输出 bounds；候选坐标由 candidate_id 对应的本地检测结果提供。
- 只有明确匹配本地 candidate_id 的候选才能标记 actionability=safe。
- 你可以补充 layout_regions / dynamic_zones / inferred_zones，但这些区域默认是 semantic_only 或 review，不能作为 safe 点击坐标。
- 不要推测截图外、滚动列表下方、被遮挡或不可见的控件。

## 输出格式

只输出严格 JSON，不要 Markdown，不要代码块：

{
  "schema_version": "2.0",
  "task_mode": "candidate_annotation",
  "coordinate_space": "vlm_image",
  "image_size": [{image_width}, {image_height}],
  "app_identity": {
    "app_name": "应用名称",
    "app_id": "应用标识",
    "surface_type": "native_uia | browser | electron_webview | canvas_self_drawn",
    "display_name": "用户可见名称"
  },
  "page_state": {
    "page_class": "home/chat/settings/editor/explorer/file_dialog/login/...",
    "state_label": "页面状态描述",
    "state_flags": []
  },
  "layout_regions": [
    {
      "region_id": "r_sidebar",
      "role": "navigation | sidebar | content | input_area | toolbar | title_bar | modal | list | panel",
      "bounds": [left, top, right, bottom],
      "purpose": "区域用途",
      "confidence": 0.9,
      "coordinate_quality": "approximate",
      "source_candidate_ids": ["candidate_id"]
    }
  ],
  "dynamic_zones": [],
  "candidate_annotations": [
    {
      "candidate_id": "本地候选ID",
      "semantic_role": "search_input / send_button / settings_icon / conversation_item / message_input / ...",
      "control_type": "button | input | tab | menu_item | icon | checkbox | dropdown | link | label | avatar",
      "interactable": true,
      "actionability": "safe | review | semantic_only",
      "coordinate_quality": "from_candidate",
      "region_id": "区域ID",
      "confidence": 0.95,
      "reason": "简短依据"
    }
  ],
  "inferred_zones": [],
  "visible_items": [],
  "missing_suggestions": [],
  "confidence": 0.9,
  "needs_review": false
}

## 关键规则

1. candidate_annotations 必须引用输入中真实存在的 candidate_id。
2. candidate_annotations 不允许输出 bounds。
3. actionability=safe 只能用于来自本地候选且语义明确的可交互元素。
4. layout_regions / dynamic_zones / inferred_zones 的 bounds 必须满足 0<=left<right<={image_width}, 0<=top<bottom<={image_height}。
5. 不确定的小图标用 review，不要脑补为 safe。
6. 如果候选明显只是动态内容文本，interactable=false，actionability=semantic_only。
7. 尽量覆盖关键候选，但不要为了凑数量标注低置信度内容。"""


_REGION_UNDERSTANDING_PROMPT_TEMPLATE = """你是一个桌面软件 UI 区域理解引擎。你的任务是理解页面结构、区域用途和动态区域，不是重新画每个小控件。

## 坐标和职责边界

- 你看到的图片尺寸是 {image_width}×{image_height}，坐标空间是 vlm_image。
- 输入候选 bounds 已经是 vlm_image 坐标，可作为区域边界参考。
- 重点输出 layout_regions、dynamic_zones、inferred_zones。
- 区域 bounds 可以是 approximate，但必须在截图范围内。
- inferred_zones 默认 actionability=semantic_only 或 review，不能作为 safe 点击目标。
- 不要推测截图外、滚动列表下方或不可见区域。

## 输出格式

只输出严格 JSON：

{
  "schema_version": "2.0",
  "task_mode": "region_understanding",
  "coordinate_space": "vlm_image",
  "image_size": [{image_width}, {image_height}],
  "app_identity": {
    "app_name": "应用名称",
    "app_id": "应用标识",
    "surface_type": "native_uia | browser | electron_webview | canvas_self_drawn",
    "display_name": "用户可见名称"
  },
  "page_state": {
    "page_class": "home/chat/settings/editor/explorer/file_dialog/login/...",
    "state_label": "页面状态描述",
    "state_flags": []
  },
  "layout_regions": [
    {
      "region_id": "r_left_nav",
      "role": "navigation | sidebar | content | input_area | toolbar | title_bar | modal | list | panel",
      "bounds": [left, top, right, bottom],
      "purpose": "区域用途",
      "confidence": 0.95,
      "coordinate_quality": "approximate",
      "source_candidate_ids": ["candidate_id"]
    }
  ],
  "dynamic_zones": [
    {
      "zone_id": "dz_content_list",
      "region_id": "r_content",
      "content_type": "conversation_list | message_area | article | code | file_list | scrollable_list",
      "bounds": [left, top, right, bottom],
      "scrollable": true,
      "note": "动态内容类型"
    }
  ],
  "candidate_annotations": [],
  "inferred_zones": [
    {
      "zone_id": "z_empty_message_area",
      "bounds": [left, top, right, bottom],
      "zone_type": "message_area",
      "purpose": "区域用途",
      "actionability": "semantic_only",
      "confidence": 0.85
    }
  ],
  "visible_items": [],
  "missing_suggestions": [],
  "confidence": 0.9,
  "needs_review": false
}

## 关键规则

1. layout_regions 至少覆盖主要页面结构：导航、列表/内容、工具栏、输入区、标题栏等。
2. dynamic_zones 用于滚动列表、聊天记录、正文、文件列表等动态内容。
3. inferred_zones 只用于语义理解，不要直接提升为可点击控件。
4. 所有区域 bounds 必须满足 0<=left<right<={image_width}, 0<=top<bottom<={image_height}。
5. 不要输出大量 fixed_controls；这个模式不负责控件检测。"""


_ICON_CROP_UNDERSTANDING_PROMPT_TEMPLATE = """你是一个桌面软件小图标语义识别引擎。你的任务是根据输入的局部 crop 或少量候选，判断图标/小控件的语义。

## 坐标和职责边界

- 你看到的图片尺寸是 {image_width}×{image_height}，坐标空间是 vlm_image。
- 输入候选如果存在，bounds 已经是当前 crop/图片坐标。
- 你的主要输出是 candidate_annotations，不要重新输出 bounds。
- 只有引用输入 candidate_id 的标注才可能 actionability=safe；无法确认时使用 review。
- 不要推测 crop 外的控件，也不要输出 layout_regions / fixed_controls。

## 输出格式

只输出严格 JSON：

{
  "schema_version": "2.0",
  "task_mode": "icon_crop_understanding",
  "coordinate_space": "vlm_image",
  "image_size": [{image_width}, {image_height}],
  "app_identity": {"app_name": "unknown", "surface_type": "native_uia"},
  "page_state": {"page_class": "unknown", "state_label": "unknown", "state_flags": []},
  "layout_regions": [],
  "dynamic_zones": [],
  "candidate_annotations": [
    {
      "candidate_id": "输入候选ID",
      "semantic_role": "search_icon / send_button / settings_icon / menu_button / close_button / attachment_button / voice_button / ...",
      "control_type": "button | icon | menu_item | input | label",
      "interactable": true,
      "actionability": "safe | review | semantic_only",
      "coordinate_quality": "from_candidate",
      "region_id": "",
      "confidence": 0.85,
      "reason": "简短视觉依据"
    }
  ],
  "inferred_zones": [],
  "visible_items": [],
  "missing_suggestions": [],
  "confidence": 0.8,
  "needs_review": false
}

## 关键规则

1. 不输出 bounds。
2. 不确定语义时 actionability=review，confidence 不要高于 0.7。
3. 如果 crop 只是文字或装饰，interactable=false，actionability=semantic_only。
4. 只输出 JSON，不要 Markdown。"""


_MISSING_AUDIT_PROMPT_TEMPLATE = """你是一个桌面软件 UI 漏检审计引擎。你的任务是核对当前语义模型是否遗漏关键区域或关键入口，只输出建议，不直接新增可点击控件。

## 坐标和职责边界

- 你看到的图片尺寸是 {image_width}×{image_height}，坐标空间是 vlm_image。
- 输入候选 bounds 已经是 vlm_image 坐标。
- 只输出 missing_suggestions、visible_items，可补充少量 layout_regions。
- missing_suggestions 默认 actionability=semantic_only 或 review，不能输出 safe。
- 不要推测截图外、滚动列表下方、被遮挡或不可见控件。

## 输出格式

只输出严格 JSON：

{
  "schema_version": "2.0",
  "task_mode": "missing_audit",
  "coordinate_space": "vlm_image",
  "image_size": [{image_width}, {image_height}],
  "app_identity": {"app_name": "应用名称或unknown", "surface_type": "native_uia"},
  "page_state": {"page_class": "页面大类或unknown", "state_label": "状态或unknown", "state_flags": []},
  "layout_regions": [],
  "dynamic_zones": [],
  "candidate_annotations": [],
  "inferred_zones": [],
  "visible_items": [
    {
      "item_id": "item_1",
      "zone_id": "",
      "item_type": "selected_item | current_document | active_tab | list_item",
      "text": "可见文字摘要",
      "matched_candidate_ids": ["candidate_id"],
      "selected": false,
      "interactable": true,
      "confidence": 0.8,
      "actionability": "safe | review | semantic_only"
    }
  ],
  "missing_suggestions": [
    {
      "suggestion_id": "m1",
      "type": "possible_missing_control | possible_missing_region | crop_and_classify",
      "description": "遗漏内容描述",
      "rough_area": [left, top, right, bottom],
      "recommended_next_step": "crop_and_classify | local_detector_retry | manual_review",
      "confidence": 0.7,
      "actionability": "review"
    }
  ],
  "confidence": 0.8,
  "needs_review": true
}

## 关键规则

1. missing_suggestions 是线索，不是最终 safe 控件。
2. rough_area 必须满足 0<=left<right<={image_width}, 0<=top<bottom<={image_height}。
3. 如果没有明显遗漏，返回空 missing_suggestions。
4. 不要输出 fixed_controls。
5. 只输出 JSON，不要 Markdown。"""


def _build_system_prompt(image_width: int, image_height: int) -> str:
    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("{image_width}", str(image_width))
        .replace("{image_height}", str(image_height))
    )


def _build_candidate_annotation_prompt(image_width: int, image_height: int) -> str:
    return (
        _CANDIDATE_ANNOTATION_PROMPT_TEMPLATE
        .replace("{image_width}", str(image_width))
        .replace("{image_height}", str(image_height))
    )


def _build_region_understanding_prompt(image_width: int, image_height: int) -> str:
    return (
        _REGION_UNDERSTANDING_PROMPT_TEMPLATE
        .replace("{image_width}", str(image_width))
        .replace("{image_height}", str(image_height))
    )


def _build_icon_crop_understanding_prompt(image_width: int, image_height: int) -> str:
    return (
        _ICON_CROP_UNDERSTANDING_PROMPT_TEMPLATE
        .replace("{image_width}", str(image_width))
        .replace("{image_height}", str(image_height))
    )


def _build_missing_audit_prompt(image_width: int, image_height: int) -> str:
    return (
        _MISSING_AUDIT_PROMPT_TEMPLATE
        .replace("{image_width}", str(image_width))
        .replace("{image_height}", str(image_height))
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_MAX_OMNI_CANDIDATES = 50
_MAX_OCR_TEXTS = 30
_MAX_UIA_ELEMENTS = 50
_MAX_HISTORICAL_STATES = 5
_JPEG_QUALITY = 85


def _coerce_prompt_bounds(item: dict[str, Any]) -> list[float] | None:
    bounds = item.get("bounds") or item.get("bbox")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
        return None
    try:
        return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
    except (TypeError, ValueError):
        return None


def _prompt_item_id(item: dict[str, Any], fallback: str) -> str:
    return str(item.get("id") or item.get("element_id") or item.get("candidate_id") or fallback)


def _prompt_priority(item: dict[str, Any]) -> float:
    text = str(item.get("text") or item.get("name") or "")
    kind = str(item.get("type") or item.get("control_type") or item.get("visual_type") or "").lower()
    confidence = item.get("confidence", 0.0)
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0
    if text.strip():
        score += 0.25
    if any(token in kind for token in ("button", "input", "edit", "tab", "menu", "checkbox", "combo", "link", "icon")):
        score += 0.35
    bounds = _coerce_prompt_bounds(item)
    if bounds:
        l, t, r, b = bounds
        area = max(0.0, r - l) * max(0.0, b - t)
        if 16 <= area <= 250_000:
            score += 0.10
    return score


def _region_bucket(item: dict[str, Any], image_size: tuple[int, int]) -> str:
    bounds = _coerce_prompt_bounds(item)
    if not bounds:
        return "unknown"
    l, t, r, b = bounds
    cx = (l + r) / 2
    cy = (t + b) / 2
    w, h = max(1, image_size[0]), max(1, image_size[1])
    return f"{min(2, max(0, int(cx / w * 3)))}:{min(2, max(0, int(cy / h * 3)))}"


def _select_prompt_items(
    items: list[dict[str, Any]],
    *,
    limit: int,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Select candidates by validity, de-duplication, priority and region budget."""
    seen: set[tuple[str, tuple[int, int, int, int] | None]] = set()
    valid: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        bounds = _coerce_prompt_bounds(item)
        bounds_key = tuple(int(round(v / 4) * 4) for v in bounds) if bounds else None
        key = (_prompt_item_id(item, f"item_{idx}"), bounds_key)
        if key in seen:
            continue
        seen.add(key)
        valid.append(item)

    sorted_items = sorted(valid, key=_prompt_priority, reverse=True)
    if len(sorted_items) <= limit:
        return sorted_items

    per_bucket = max(3, limit // 9)
    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    overflow: list[dict[str, Any]] = []
    for item in sorted_items:
        bucket = _region_bucket(item, image_size)
        if bucket_counts.get(bucket, 0) < per_bucket:
            selected.append(item)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        else:
            overflow.append(item)
        if len(selected) >= limit:
            return selected
    for item in overflow:
        if len(selected) >= limit:
            break
        selected.append(item)
    return selected


def _image_to_data_url(img: Image.Image) -> str:
    """Encode images compactly without changing pixel dimensions.

    The modeler prepares the VLM-sized image before prompt construction. This
    encoder deliberately avoids resizing so prompt coordinates and pixels stay
    in the same coordinate space.
    """
    if img.mode == "RGBA":
        rgb = Image.new("RGB", img.size, "white")
        rgb.paste(img, mask=img.getchannel("A"))
    elif img.mode in {"RGB", "L"}:
        rgb = img.convert("RGB")
    else:
        rgb = img.convert("RGB")

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_page_understanding_prompt(
    prompt_input: PromptInput,
    *,
    max_tokens: int = 8192,
    provider_options: dict[str, Any] | None = None,
) -> VLMSemanticRequest:
    """构造 VLMSemanticRequest。

    图片输入策略（按优先级）：
    1. 如果有 numbered_overlay_image → 作为主图发送，原图作为第二图
    2. 如果有 candidate_atlas_image → 作为辅助图发送
    3. 只有原图 → 只发原图

    文本线索嵌入 user message。
    """
    # --- 构建 user content ---
    user_content: list[dict[str, Any]] = []

    # Image size is prepared by the modeler. Providers only encode it.

    # --- 文本线索 ---
    text_parts: list[str] = []

    # Omni 候选
    if prompt_input.omni_candidates:
        candidates = _select_prompt_items(
            prompt_input.omni_candidates,
            limit=_MAX_OMNI_CANDIDATES,
            image_size=prompt_input.screenshot.size,
        )
        lines = ["## 本地检测到的候选元素（OmniParser）"]
        for i, c in enumerate(candidates):
            cid = c.get("id") or c.get("element_id") or f"c{i}"
            ctype = c.get("type", "unknown")
            text_val = c.get("text", "")
            bounds = c.get("bounds", [])
            conf = c.get("confidence", 0)
            lines.append(f"- [{cid}] type={ctype} text='{text_val}' bounds={bounds} confidence={conf}")
        text_parts.append("\n".join(lines))

    # OCR 文本
    if prompt_input.ocr_texts:
        texts = prompt_input.ocr_texts[:_MAX_OCR_TEXTS]
        lines = ["## OCR 检测到的文本"]
        for t in texts:
            lines.append(f"- {t}")
        text_parts.append("\n".join(lines))

    # UIA 元素
    if prompt_input.uia_elements:
        elements = _select_prompt_items(
            prompt_input.uia_elements,
            limit=_MAX_UIA_ELEMENTS,
            image_size=prompt_input.screenshot.size,
        )
        lines = ["## UIA 检测到的元素"]
        for e in elements:
            eid = e.get("id") or e.get("element_id") or ""
            name = e.get("name", "")
            etype = e.get("type", "")
            bounds = e.get("bounds", [])
            lines.append(f"- [{eid}] name='{name}' type={etype} bounds={bounds}")
        text_parts.append("\n".join(lines))

    # 当前 PageModel 摘要
    if prompt_input.current_page_model:
        pm = prompt_input.current_page_model
        lines = ["## 当前页面模型摘要"]
        if pm.get("app_id"):
            lines.append(f"- app_id: {pm['app_id']}")
        if pm.get("page_class"):
            lines.append(f"- page_class: {pm['page_class']}")
        if pm.get("state_label"):
            lines.append(f"- state_label: {pm['state_label']}")
        text_parts.append("\n".join(lines))

    # 历史状态
    if prompt_input.historical_states:
        states = prompt_input.historical_states[:_MAX_HISTORICAL_STATES]
        lines = ["## 历史观察到的状态"]
        for s in states:
            label = s.get("state_label", "未知")
            page_class = s.get("page_class", "")
            lines.append(f"- {page_class}: {label}")
        text_parts.append("\n".join(lines))

    # 任务提示
    if prompt_input.task_hint:
        text_parts.append(f"## 任务提示\n{prompt_input.task_hint}")

    # 合并文本
    full_text = "\n\n".join(text_parts) if text_parts else "请分析此页面截图，输出结构化 JSON。"
    user_content.append({"type": "text", "text": full_text})

    # --- messages ---
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_content},
    ]

    # Compute actual screenshot dimensions for coordinate system instruction
    img_w, img_h = prompt_input.screenshot.size

    if prompt_input.task_mode == CANDIDATE_ANNOTATION:
        system_prompt = _build_candidate_annotation_prompt(img_w, img_h)
    elif prompt_input.task_mode == REGION_UNDERSTANDING:
        system_prompt = _build_region_understanding_prompt(img_w, img_h)
    elif prompt_input.task_mode == ICON_CROP_UNDERSTANDING:
        system_prompt = _build_icon_crop_understanding_prompt(img_w, img_h)
    elif prompt_input.task_mode == MISSING_AUDIT:
        system_prompt = _build_missing_audit_prompt(img_w, img_h)
    else:
        system_prompt = _build_system_prompt(img_w, img_h)

    return VLMSemanticRequest(
        screenshot=prompt_input.screenshot,
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        provider_options=provider_options or {},
    )
