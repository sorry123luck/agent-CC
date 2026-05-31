"""Candidate Refine Engine — second-pass semantic review of canvas candidates.

Supports multiple modes:
- heuristic: rule-based inference from existing candidate fields
- vlm: VLM-based visual analysis (future)
- agent: external Agent-based review (future)

Design: the refine engine never overwrites original fields (text, semantic_role,
control_type, provider_sources, confidence). It only writes to refine-specific
fields (visual_type, semantic_tags, role_label, role_confidence, role_source,
role_evidence, refine_status).

Three-layer extensible design:
  Layer 1: visual_type — fixed universal set (icon/text/button/input/...)
  Layer 2: semantic_tags[] + role_label + role_confidence + role_source + role_evidence[]
           — extensible strings, no hardcoded enums
  Layer 3: app-specific knowledge lives in profiles/memory/.ocpack, not in code
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PIL import Image

from src.perception.page_compiler_models import Candidate

logger = logging.getLogger(__name__)


# Visual type classification — Layer 1: fixed universal set
VISUAL_TYPES = {
    "icon", "text", "button", "input", "avatar", "image",
    "list_item", "panel", "toolbar", "menu", "checkbox",
    "tab", "scrollbar", "unknown",
}

# Dynamic content zones — text in these zones is user/document content,
# NOT UI controls. Keyword-based role inference must NOT apply here.
DYNAMIC_CONTENT_ZONES: frozenset[str] = frozenset({
    "message_area",
    "content_stream",
    "document_body",
    "chat_message_area",
    "right_panel_middle",
    "article_body",
    "code_block",
    "list_dynamic_content",
})


@dataclass
class RefineResult:
    """Refine result for a single candidate (three-layer design)."""
    element_id: str
    visual_type: str = "unknown"
    semantic_tags: list[str] = field(default_factory=list)
    role_label: str | None = None
    role_confidence: float = 0.0
    role_source: str = "heuristic"
    role_evidence: list[str] = field(default_factory=list)
    refine_status: str = "unreviewed"


@dataclass
class RefineOutput:
    """Output of a refine pass on a canvas."""
    canvas_id: str
    mode: str = "heuristic"
    results: list[RefineResult] = field(default_factory=list)
    total_refined: int = 0
    total_uncertain: int = 0
    total_unreviewed: int = 0
    errors: list[str] = field(default_factory=list)  # 校验错误（如 element_id 不存在）


class RefineEngine:
    """Second-pass semantic review engine for canvas candidates."""

    def refine(
        self,
        canvas_id: str,
        candidates: list[Candidate],
        screenshots: dict[str, Image.Image] | None = None,
        candidate_ids: list[str] | None = None,
        mode: str = "heuristic",
        agent_results: list[RefineResult] | None = None,
    ) -> RefineOutput:
        """Run refine on canvas candidates.

        Args:
            canvas_id: The canvas ID.
            candidates: List of Candidate objects from the canvas.
            screenshots: Optional dict of element_id -> crop image (not used by heuristic).
            candidate_ids: Optional list of specific element IDs to refine. None = all.
            mode: Refine mode (heuristic / vlm / agent).
            agent_results: External Agent classification results (mode=agent only).

        Returns:
            RefineOutput with results for each candidate.
        """
        errors: list[str] = []

        if mode == "agent" and agent_results is not None:
            # Agent mode: directly use provided results, no heuristic inference
            candidate_id_set = {c.element_id for c in candidates}
            valid_results: list[RefineResult] = []
            for r in agent_results:
                if r.element_id not in candidate_id_set:
                    errors.append(f"element_id '{r.element_id}' not found in canvas")
                    continue
                valid_results.append(r)
            results = valid_results
        else:
            targets = candidates
            if candidate_ids:
                id_set = set(candidate_ids)
                targets = [c for c in candidates if c.element_id in id_set]

            if mode == "heuristic":
                results = [self._refine_heuristic(c) for c in targets]
            else:
                # vlm mode — placeholder, mark all as unreviewed
                results = [
                    RefineResult(element_id=c.element_id, refine_status="unreviewed")
                    for c in targets
                ]

        total_refined = sum(1 for r in results if r.refine_status == "refined")
        total_uncertain = sum(1 for r in results if r.refine_status == "uncertain")
        total_unreviewed = sum(1 for r in results if r.refine_status == "unreviewed")

        return RefineOutput(
            canvas_id=canvas_id,
            mode=mode,
            results=results,
            total_refined=total_refined,
            total_uncertain=total_uncertain,
            total_unreviewed=total_unreviewed,
            errors=errors,
        )

    def _refine_heuristic(self, c: Candidate) -> RefineResult:
        """Heuristic-based refine using existing candidate fields."""
        text = (c.text or "").strip()
        name = (c.name or "").strip()
        ct = (c.control_type or "").lower()
        sources = set(c.provider_sources)
        has_uia = "uia" in sources
        has_ocr = "ocr" in sources
        has_vision = bool(sources & {"vision", "vlm", "omniparser"})
        bounds = c.bounds
        is_small = False
        is_wide = False
        is_tall = False
        if bounds and len(bounds) >= 4:
            w = bounds[2] - bounds[0]
            h = bounds[3] - bounds[1]
            is_small = w < 60 and h < 60
            is_wide = w > 200 and h < 50
            is_tall = h > 200 and w < 50

        # --- Dynamic content zone guard ---
        # Text in dynamic content zones is user/document content, not UI controls.
        # Skip keyword-based role inference to avoid misclassifying chat messages,
        # document text, etc. as functional buttons.
        if c.region_id in DYNAMIC_CONTENT_ZONES and text:
            return self._refine_dynamic_content(c, text)

        # --- Layer 1: Visual type classification ---
        visual_type = self._classify_visual_type(c, ct, text, name, is_small, is_wide, is_tall, sources)

        # --- Layer 2: Semantic tags + role inference ---
        semantic_tags, role_label, role_conf, evidence = self._infer_role(
            c, visual_type, text, name, ct, sources, is_small, is_wide, bounds,
        )

        # --- Determine refine status ---
        if role_conf >= 0.7:
            status = "refined"
        elif role_conf >= 0.3:
            status = "uncertain"
        else:
            status = "uncertain"

        return RefineResult(
            element_id=c.element_id,
            visual_type=visual_type,
            semantic_tags=semantic_tags,
            role_label=role_label,
            role_confidence=role_conf,
            role_source="heuristic",
            role_evidence=evidence,
            refine_status=status,
        )

    def _refine_dynamic_content(self, c: Candidate, text: str) -> RefineResult:
        """Handle candidates in dynamic content zones.

        Text in dynamic content areas (chat messages, document body, etc.)
        is user/document content, NOT UI controls. We assign a safe generic
        label and mark as uncertain. Keyword-based role inference is skipped
        to avoid misclassifying message content as functional buttons.

        refine_status is always 'uncertain' — downstream stable_key_id
        assignment is gated on 'refined', so dynamic content never gets
        a persistent identity key.
        """
        region_id = c.region_id or "unknown"
        # Choose label based on zone type
        if region_id in ("code_block",):
            role_label = "code_text"
        elif region_id in ("document_body", "article_body"):
            role_label = "document_text"
        else:
            role_label = "message_content"

        return RefineResult(
            element_id=c.element_id,
            visual_type="text",
            semantic_tags=["content.dynamic_text"],
            role_label=role_label,
            role_confidence=0.1,
            role_source="heuristic",
            role_evidence=[f"dynamic content zone: {region_id}"],
            refine_status="uncertain",
        )

    def _classify_visual_type(
        self, c: Candidate, ct: str, text: str, name: str,
        is_small: bool, is_wide: bool, is_tall: bool, sources: set[str],
    ) -> str:
        """Classify the visual type of a candidate (Layer 1: fixed universal set)."""
        # Button
        if "button" in ct:
            if is_small:
                return "icon"
            return "button"
        # Input / Edit
        if ct in ("edit", "input", "text") or "edit" in ct:
            return "input"
        # Text / Static
        if ct in ("text", "static", "label"):
            return "text"
        # Checkbox
        if ct in ("checkbox", "check"):
            return "checkbox"
        # Image
        if ct in ("image", "picture"):
            return "image"
        # List item
        if ct in ("listitem", "list_item", "item"):
            return "list_item"
        # Menu
        if ct in ("menu", "menuitem"):
            return "menu"
        # Toolbar
        if ct in ("toolbar", "tool_bar"):
            return "toolbar"
        # Avatar (small square, often from vision)
        if is_small and ct in ("", "unknown") and not text:
            return "icon"
        # Icon: small, no text, from vision
        if is_small and not text and bool(sources & {"vision", "vlm", "omniparser"}):
            return "icon"
        # Wide input-like
        if is_wide and not text:
            return "input"
        # Has text content
        if text:
            return "text"
        return "unknown"

    def _infer_role(
        self, c: Candidate, visual_type: str, text: str, name: str,
        ct: str, sources: set[str], is_small: bool, is_wide: bool,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple[list[str], str | None, float, list[str]]:
        """Infer semantic tags, role label, confidence, and evidence.

        Returns (semantic_tags, role_label, confidence, evidence_list).
        """
        text_lower = text.lower()
        name_lower = name.lower() if name else ""
        combined = f"{text_lower} {name_lower}".strip()

        # --- High-confidence text-based matching ---
        text_result = self._match_text_role(combined, text_lower, name_lower, visual_type)
        if text_result:
            return text_result

        # --- Control-type based ---
        if "button" in ct and text:
            return (
                ["action.button"],
                text,
                0.6,
                [f"control_type={ct}, text='{text}'"],
            )

        if ct in ("edit",) or "edit" in ct:
            if is_wide:
                return (
                    ["input.text", "input.search"],
                    "搜索输入框",
                    0.5,
                    [f"control_type={ct}, wide input area"],
                )
            return (
                ["input.text"],
                "输入框",
                0.4,
                [f"control_type={ct}"],
            )

        # --- Icon heuristics (vision-primary, no text) ---
        if visual_type == "icon" and not text:
            return self._infer_icon_role(c, bounds, sources)

        # --- List items ---
        if visual_type == "list_item" and text:
            return (
                ["list.item"],
                text[:30] if text else "列表项",
                0.4,
                [f"list_item with text='{text[:20]}'"],
            )

        # --- Checkbox ---
        if visual_type == "checkbox":
            return (
                ["input.checkbox"],
                "复选框",
                0.5,
                [f"control_type={ct}"],
            )

        # --- Fallback ---
        return ([], None, 0.1, ["no strong signals"])

    def _match_text_role(
        self, combined: str, text_lower: str, name_lower: str,
        visual_type: str,
    ) -> tuple[list[str], str | None, float, list[str]] | None:
        """Match common UI text patterns to semantic tags and role labels.

        Returns (semantic_tags, role_label, confidence, evidence) or None.
        Tags use generic functional namespaces — no app-specific prefixes.
        App-specific tags (e.g. "wechat.emoji_button") come from
        vlm/agent/memory/profile layers, not from heuristic.
        """
        patterns: list[tuple[list[str], list[str], str | None, float, str]] = [
            # (keywords, semantic_tags, role_label, confidence, evidence)
            (["发送", "send"], ["action.send"], "发送", 0.9, "text matches send"),
            (["表情", "emoji", "😊", "😀"], ["content.emoji"], "表情", 0.85, "text matches emoji"),
            (["麦克风", "mic", "🎤"], ["media.mic"], "麦克风", 0.85, "text matches mic"),
            (["搜索", "search", "🔍"], ["input.search"], "搜索", 0.8, "text matches search"),
            (["设置", "settings", "⚙"], ["action.settings"], "设置", 0.8, "text matches settings"),
            (["关闭", "close", "✕", "×"], ["window.close"], "关闭", 0.9, "text matches close"),
            (["最小化", "minimize", "—"], ["window.minimize"], "最小化", 0.9, "text matches minimize"),
            (["最大化", "maximize", "□"], ["window.maximize"], "最大化", 0.9, "text matches maximize"),
            (["文件", "file", "📁", "文件夹", "folder"], ["content.file"], "文件", 0.7, "text matches file/folder"),
            (["截图", "screenshot", "✂", "剪刀", "scissor"], ["action.screenshot"], "截图", 0.75, "text matches screenshot/scissor"),
            (["电话", "phone", "📞"], ["media.phone"], "电话", 0.8, "text matches phone"),
            (["视频", "video", "📹"], ["media.video"], "视频", 0.8, "text matches video"),
            (["添加", "add", "➕", "+"], ["action.add"], "添加", 0.75, "text matches add"),
            (["刷新", "refresh", "🔄"], ["action.refresh"], "刷新", 0.75, "text matches refresh"),
            (["返回", "back", "←"], ["navigation.back"], "返回", 0.8, "text matches back"),
            (["通知", "notification", "🔔"], ["display.notification"], "通知", 0.7, "text matches notification"),
            (["群", "group"], ["list.group"], "群组", 0.6, "text matches group"),
            (["置顶", "pin"], ["action.pin"], "置顶", 0.7, "text matches pin"),
            (["静音", "mute"], ["action.mute"], "静音", 0.7, "text matches mute"),
            (["附件", "attach", "📎"], ["action.attach"], "附件", 0.75, "text matches attach"),
            (["删除", "delete", "🗑"], ["action.delete"], "删除", 0.75, "text matches delete"),
            (["编辑", "edit", "✏"], ["action.edit"], "编辑", 0.75, "text matches edit"),
            (["保存", "save", "💾"], ["action.save"], "保存", 0.75, "text matches save"),
            (["复制", "copy"], ["action.copy"], "复制", 0.7, "text matches copy"),
            (["粘贴", "paste"], ["action.paste"], "粘贴", 0.7, "text matches paste"),
            (["撤销", "undo"], ["action.undo"], "撤销", 0.7, "text matches undo"),
            (["重做", "redo"], ["action.redo"], "重做", 0.7, "text matches redo"),
            (["登录", "login", "sign in"], ["action.login"], "登录", 0.8, "text matches login"),
            (["注册", "register", "sign up"], ["action.register"], "注册", 0.8, "text matches register"),
            (["确认", "confirm", "确定", "ok"], ["action.confirm"], "确认", 0.75, "text matches confirm"),
            (["取消", "cancel"], ["action.cancel"], "取消", 0.75, "text matches cancel"),
        ]

        for keywords, tags, label, conf, evidence in patterns:
            for kw in keywords:
                if kw in combined:
                    return (tags, label, conf, [evidence])
        return None

    def _infer_icon_role(
        self, c: Candidate, bounds: tuple[int, int, int, int] | None,
        sources: set[str],
    ) -> tuple[list[str], str | None, float, list[str]]:
        """Infer role for icon-type elements with no text.

        Heuristic mode only produces generic tags. App-specific tags
        (e.g. "wechat.send_button") come from vlm/agent/memory/profile.
        """
        if bounds and len(bounds) >= 4:
            left, top, right, bottom = bounds
            w = right - left
            h = bottom - top
            cx = (left + right) / 2
            cy = (top + bottom) / 2

            # Very top-right: likely window controls (close/min/max)
            if top < 50 and right > 800:
                if w < 50 and h < 50:
                    return (
                        ["window.control"],
                        "窗口控制",
                        0.3,
                        ["small icon in top-right area"],
                    )

            # Very bottom: likely input area icons (emoji, mic, send)
            if cy > 600:
                if w < 50 and h < 50:
                    return (
                        ["toolbar.icon"],
                        None,
                        0.2,
                        ["small icon in bottom area — likely input toolbar"],
                    )

        has_vision = bool(sources & {"vision", "vlm", "omniparser"})
        if has_vision:
            return (
                ["display.icon"],
                None,
                0.2,
                ["vision-detected icon, no text, role unknown"],
            )

        return ([], None, 0.1, ["icon with no text or vision context"])
