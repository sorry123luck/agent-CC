"""Canvas Query Engine — query InteractionCanvas for candidates matching a target."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.perception.page_compiler_models import Candidate, InteractionCanvas, SemanticRole


@dataclass
class QueryTarget:
    """Query target specification. Exactly one field should be set."""
    text: str | None = None                    # Fuzzy text match
    semantic_role: str | None = None           # Exact semantic role match
    region: str | None = None                  # Region ID or role
    natural_language: str | None = None        # Keyword extraction + combo
    composite: dict[str, Any] | None = None    # Arbitrary filter dict


@dataclass
class QueryResult:
    """Result of a canvas query."""
    candidates: list[Candidate] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    total_matched: int = 0
    query_target: QueryTarget | None = None


@dataclass(frozen=True)
class QueryIntent:
    """Normalized natural-language intent used for scoring."""
    name: str
    roles: tuple[str, ...] = ()
    negative_roles: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    control_types: tuple[str, ...] = ()
    region_roles: tuple[str, ...] = ()
    require_risk: bool = False


class CanvasQueryEngine:
    """Query an InteractionCanvas for candidates matching a target."""

    def query(
        self,
        canvas: InteractionCanvas,
        target: QueryTarget,
        max_results: int = 5,
        min_confidence: float = 0.2,
    ) -> QueryResult:
        """Query the canvas for candidates matching the target."""
        scored: list[tuple[float, Candidate]] = []

        for candidate in canvas.elements:
            score = self._score_candidate(candidate, target, canvas)
            if score > 0.0 and score >= min_confidence:
                scored.append((score, candidate))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:max_results]]

        suggestions: list[str] = []
        if not top:
            suggestions = self._build_suggestions(canvas, target)

        return QueryResult(
            candidates=top,
            suggestions=suggestions,
            total_matched=len(scored),
            query_target=target,
        )

    def _score_candidate(
        self,
        candidate: Candidate,
        target: QueryTarget,
        canvas: InteractionCanvas,
    ) -> float:
        """Score a candidate against the query target. Returns 0-1."""
        scores: list[float] = []

        if target.text is not None:
            scores.append(self._text_score(candidate, target.text))

        if target.semantic_role is not None:
            role_val = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
            scores.append(1.0 if role_val == target.semantic_role else 0.0)

        if target.region is not None:
            in_region = (candidate.region_id == target.region)
            # Also check region role
            region = canvas.get_region(candidate.region_id) if candidate.region_id else None
            region_match = region is not None and region.role == target.region
            scores.append(1.0 if (in_region or region_match) else 0.0)

        if target.natural_language is not None:
            scores.append(self._nl_score(candidate, target.natural_language, canvas))

        if target.composite is not None:
            scores.append(self._composite_score(candidate, target.composite))

        if not scores:
            return 0.0

        # Relevance: average of all matched dimensions
        relevance_score = sum(scores) / len(scores)

        # Phase 7: reliability weight from fused_confidence
        # Maps fused_confidence [0, 1] → reliability_weight [0.5, 1.0]
        raw_confidence = getattr(candidate, "confidence", 0.0)
        try:
            fused = float(raw_confidence if raw_confidence is not None else 0.0)
        except (TypeError, ValueError):
            fused = 0.0
        fused = max(0.0, min(1.0, fused))
        reliability_weight = 0.5 + 0.5 * fused

        return relevance_score * reliability_weight

    def _text_score(self, candidate: Candidate, query: str) -> float:
        """Fuzzy text match score."""
        query_lower = query.lower()
        texts = [
            candidate.text or "",
            candidate.name or "",
            candidate.placeholder or "",
            candidate.role_label or "",  # refined role_label
        ]
        best = 0.0
        for t in texts:
            t_lower = t.lower()
            if query_lower in t_lower:
                # Exact substring match
                ratio = len(query_lower) / max(len(t_lower), 1)
                best = max(best, 0.5 + 0.5 * ratio)
            else:
                # Fuzzy match
                ratio = SequenceMatcher(None, query_lower, t_lower).ratio()
                best = max(best, ratio * 0.7)
        return best

    def _nl_score(self, candidate: Candidate, query: str, canvas: InteractionCanvas | None = None) -> float:
        """Natural language query score — intent extraction + keyword fallback."""
        intent = self._parse_nl_intent(query)
        if intent is not None:
            return self._intent_score(candidate, intent, canvas)

        keywords = self._query_terms(query)
        if not keywords:
            return 0.0

        texts = self._candidate_search_blob(candidate)

        matched = sum(1 for kw in keywords if kw in texts)
        return matched / len(keywords)

    def _parse_nl_intent(self, query: str) -> QueryIntent | None:
        """Map common agent phrasing to UI-control intents.

        The query API still accepts exact text/role filters; this path handles
        high-frequency natural phrases where plain substring matching creates
        bad false positives, especially in Chinese UI text.
        """
        normalized = self._normalize_query(query)

        def has_any(*phrases: str) -> bool:
            return any(phrase in normalized for phrase in phrases)

        if has_any("搜索框", "搜索栏", "查找框", "搜索输入", "地址栏", "地址搜索"):
            return QueryIntent(
                name="search_input",
                roles=("search_input", "text_input"),
                negative_roles=("button", "send_button", "submit_button"),
                terms=("搜索", "search", "find", "地址", "address"),
                control_types=("edit", "textbox", "input"),
                region_roles=("toolbar", "title_bar", "navigation", "search"),
            )
        if has_any("输入区", "输入框", "文本框", "聊天输入", "消息输入", "回复框"):
            return QueryIntent(
                name="message_input",
                roles=("message_input", "text_input"),
                negative_roles=("search_input", "send_button", "button"),
                terms=("输入", "input", "message", "composer", "回复"),
                control_types=("edit", "textbox", "input"),
                region_roles=("composer", "input_area"),
            )
        if has_any("发送按钮", "发消息按钮", "发送控件") or (has_any("发送", "send") and has_any("按钮", "button")):
            return QueryIntent(
                name="send_button",
                roles=("send_button", "submit_button"),
                negative_roles=("message_input", "text_input", "search_input", "voice_input"),
                terms=("发送", "send", "submit"),
                negative_terms=("输入", "input", "消息内容"),
                control_types=("button", "splitbutton"),
                region_roles=("composer", "action_bar"),
                require_risk=True,
            )
        if has_any("设置菜单", "设置按钮", "设置入口", "偏好设置", "settings"):
            return QueryIntent(
                name="settings_menu",
                roles=("menu_item", "button", "icon_button", "toggle_button"),
                terms=("设置", "settings", "preference", "preferences", "menu", "菜单"),
                control_types=("button", "menuitem"),
                region_roles=("toolbar", "navigation", "title_bar", "sidebar"),
            )
        if has_any("会话列表", "联系人列表", "聊天列表", "对话列表", "conversationlist", "chatlist"):
            return QueryIntent(
                name="conversation_list",
                roles=("chat_item", "list_item", "nav_item", "tree_item"),
                terms=("会话", "联系人", "conversation", "chat", "contact", "列表", "list"),
                region_roles=("sidebar", "navigation", "conversation_list", "item_list"),
            )
        if has_any("页面标题", "当前标题", "窗口标题", "标题"):
            return QueryIntent(
                name="page_title",
                roles=("title_bar", "text", "heading"),
                terms=("标题", "title", "heading"),
                region_roles=("title_bar", "header", "top_bar"),
            )
        if has_any("主内容区域", "主要内容", "内容区域", "正文区域", "主面板"):
            return QueryIntent(
                name="main_content",
                roles=("text", "list_item", "document", "pane"),
                terms=("内容", "content", "main", "正文", "document"),
                region_roles=("main_content", "content_area", "message_stream", "main_workspace", "document"),
            )
        if has_any("危险操作", "危险按钮", "删除按钮", "支付按钮", "提交按钮", "退出按钮"):
            return QueryIntent(
                name="dangerous_action",
                roles=("send_button", "submit_button", "delete_button", "button"),
                terms=("删除", "delete", "支付", "付款", "pay", "提交", "submit", "退出", "发送", "send"),
                control_types=("button",),
                require_risk=True,
            )

        if has_any("菜单按钮", "菜单", "menu", "菜单栏"):
            return QueryIntent(
                name="menu_button",
                roles=("menu_item", "button", "icon_button"),
                terms=("菜单", "menu", "Menu"),
                control_types=("button", "menuitem", "menu"),
                region_roles=("toolbar", "title_bar", "menu_bar", "navigation"),
            )

        # --- P1.1: Chrome / VS Code / QQ / WeChat intents ---

        if has_any("标签页", "标签栏", "标签切换", "tab"):
            return QueryIntent(
                name="tab",
                roles=("tab", "menu_item"),
                terms=("标签", "tab", "标签页"),
                control_types=("tabitem", "tab", "button"),
                region_roles=("tab_bar", "toolbar", "title_bar"),
            )

        if has_any("后退按钮", "后退", "前进按钮", "前进", "刷新按钮", "刷新", "返回按钮", "返回",
                    "back", "forward", "refresh", "reload"):
            return QueryIntent(
                name="navigation_button",
                roles=("button", "icon_button", "nav_item"),
                terms=("后退", "前进", "刷新", "返回", "back", "forward", "refresh", "reload", "重载"),
                control_types=("button", "splitbutton"),
                region_roles=("toolbar", "title_bar", "navigation"),
            )

        if has_any("文件资源管理器", "资源管理器", "文件树", "文件列表", "file explorer", "file explorer"):
            return QueryIntent(
                name="file_explorer",
                roles=("sidebar", "tree_item", "list_item"),
                terms=("文件", "资源管理", "file", "explorer", "workspace", "工作区"),
                control_types=("tree", "treeitem", "list", "listitem"),
                region_roles=("sidebar", "navigation", "side_panel"),
            )

        if has_any("终端", "命令行", "控制台", "terminal", "console", "powershell"):
            return QueryIntent(
                name="terminal",
                roles=("text", "pane", "panel"),
                terms=("终端", "terminal", "console", "powershell", "命令", "shell"),
                control_types=("pane", "document", "text"),
                region_roles=("panel", "bottom_panel", "terminal"),
            )

        if has_any("编辑区域", "编辑器", "代码区", "代码编辑", "editor", "代码窗口"):
            return QueryIntent(
                name="editor",
                roles=("text", "text_input", "document"),
                terms=("编辑", "edit", "代码", "code", "editor"),
                control_types=("edit", "textbox", "document", "text"),
                region_roles=("editor", "main_content", "document"),
            )

        if has_any("联系人", "好友", "通讯录", "contact", "contacts"):
            return QueryIntent(
                name="contact",
                roles=("chat_item", "list_item", "nav_item", "tree_item"),
                terms=("联系人", "contact", "好友", "通讯录", "friend"),
                control_types=("listitem", "treeitem"),
                region_roles=("sidebar", "navigation", "contact_list"),
            )

        return None

    def _intent_score(self, candidate: Candidate, intent: QueryIntent, canvas: InteractionCanvas | None = None) -> float:
        role = self._candidate_role(candidate)
        blob = self._candidate_search_blob(candidate)
        control_type = str(candidate.control_type or "").lower()
        risk_tags = {str(item).lower() for item in list(candidate.risk_tags or [])}
        risk_level = str(getattr(candidate.risk_level, "value", candidate.risk_level) or "").lower()

        if role in intent.negative_roles:
            return 0.0
        if any(term and term in blob for term in intent.negative_terms):
            return 0.0

        # Resolve region role: first from attributes, then from canvas regions
        region_role = self._candidate_region_role(candidate)
        if not region_role and canvas and candidate.region_id:
            region = canvas.get_region(candidate.region_id)
            if region:
                region_role = str(region.role or "")

        score = 0.0
        if role in intent.roles:
            score += 0.50
        if any(term and term in blob for term in intent.terms):
            score += 0.20
        if control_type and any(item in control_type for item in intent.control_types):
            score += 0.10

        # Region matching: bonus for match, penalty for mismatch, neutral for unknown
        if intent.region_roles:
            if region_role and region_role in intent.region_roles:
                score += 0.20  # strong region match bonus
            elif region_role and region_role not in intent.region_roles:
                score -= 0.10  # region mismatch penalty (only when region is known)
            # region_role == "": no bonus, no penalty

        if intent.require_risk and (risk_tags or risk_level not in {"", "l0", "low"}):
            score += 0.08

        # Guard against dangerous-action false positives: text-only matches in
        # non-button controls are not enough for an action intent.
        if intent.require_risk and role not in intent.roles and not risk_tags:
            return 0.0

        return max(0.0, min(score, 1.0))

    def _candidate_search_blob(self, candidate: Candidate) -> str:
        return self._normalize_query(" ".join([
            candidate.text or "",
            candidate.name or "",
            self._candidate_role(candidate),
            candidate.control_type or "",
            candidate.role_label or "",
            " ".join(candidate.semantic_tags),
            candidate.visual_type or "",
            str(getattr(candidate, "placeholder", "") or ""),
        ]))

    def _candidate_role(self, candidate: Candidate) -> str:
        role = candidate.semantic_role
        return str(role.value if hasattr(role, "value") else role or "unknown")

    def _candidate_region_role(self, candidate: Candidate) -> str:
        attrs = getattr(candidate, "attributes", None) or {}
        for key in ("region_role", "parent_region_role"):
            value = str(attrs.get(key) or "")
            if value:
                return value
        return ""

    def _query_terms(self, query: str) -> list[str]:
        label_terms = [
            term.lower()
            for term in re.findall(r"\b[a-zA-Z]+\d+\b", str(query or ""))
        ]
        if label_terms:
            return label_terms
        normalized = self._normalize_query(query)
        spaced = [term for term in normalized.split() if term]
        if spaced:
            return spaced
        return [normalized] if normalized else []

    def _normalize_query(self, query: str) -> str:
        return str(query or "").strip().lower().replace(" ", "")

    def _composite_score(self, candidate: Candidate, filters: dict[str, Any]) -> float:
        """Composite filter score."""
        matches = 0
        total = 0

        if "text" in filters:
            total += 1
            if self._text_score(candidate, filters["text"]) > 0.3:
                matches += 1

        if "semantic_role" in filters:
            total += 1
            role_val = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
            if role_val == filters["semantic_role"]:
                matches += 1

        if "region" in filters:
            total += 1
            if candidate.region_id == filters["region"]:
                matches += 1

        if "interactable" in filters:
            total += 1
            if candidate.interactable == filters["interactable"]:
                matches += 1

        if "control_type" in filters:
            total += 1
            if candidate.control_type == filters["control_type"]:
                matches += 1

        # Refine field filters
        if "role_label" in filters:
            total += 1
            if candidate.role_label == filters["role_label"]:
                matches += 1

        if "semantic_tags" in filters:
            total += 1
            tag_filter = filters["semantic_tags"]
            if isinstance(tag_filter, str):
                tag_filter = [tag_filter]
            if any(t in candidate.semantic_tags for t in tag_filter):
                matches += 1

        if "visual_type" in filters:
            total += 1
            if candidate.visual_type == filters["visual_type"]:
                matches += 1

        if "refine_status" in filters:
            total += 1
            if candidate.refine_status == filters["refine_status"]:
                matches += 1

        return matches / max(total, 1)

    def _build_suggestions(
        self,
        canvas: InteractionCanvas,
        target: QueryTarget,
    ) -> list[str]:
        """Build suggestions when no candidates found."""
        suggestions: list[str] = []

        if target.text:
            suggestions.append(f"No element with text '{target.text}' found. Try scrolling the page or using allow_vlm=true.")

        if target.semantic_role:
            suggestions.append(f"No element with role '{target.semantic_role}' found. The page may need re-observation.")

        if canvas.partial:
            suggestions.append("Canvas is partial (some providers failed). Results may be incomplete.")

        if not canvas.stable:
            suggestions.append("Canvas is not stable (page may be loading). Wait and re-observe.")

        return suggestions
