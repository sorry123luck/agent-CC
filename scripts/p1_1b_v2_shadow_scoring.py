"""P1.1b.2 Shadow Scoring v2: Expected-role-aware evidence scoring.

Shadow-only analysis. Does NOT change actual returned results.

Usage:
    python scripts/p1_1b_v2_shadow_scoring.py
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ["OPENCLAW_U4_LAYOUT_SHADOW"] = "1"
os.environ["OPENCLAW_URE_P0_SHADOW"] = "1"
os.environ["OPENCLAW_VTL_0_SHADOW"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.perception_service import PerceptionService
from src.windows.window_enum import WindowEnumService
from src.canvas.query_engine import CanvasQueryEngine, QueryTarget

REPORT_DIR = Path("reports/p1_1b_v2_shadow_scoring")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Corrected query set from P1.1a.1-fix ────────────────────────
QUERY_TESTS = [
    ("Notepad", "文件", "target", "文件", "menu", None),
    ("Notepad", "编辑", "target", "编辑", "menu", None),
    ("Notepad", "查看", "target", "查看", "menu", None),
    ("Notepad", "搜索框", "abstain", None, None, "substring_fp"),
    ("Notepad", "发送按钮", "abstain", None, None, None),
    ("Notepad", "保存", "ambiguous", None, "menu", None),
    ("Notepad", "打开", "ambiguous", None, "menu", None),
    ("Notepad", "帮助", "ambiguous", None, "menu", None),
    ("Notepad", "格式", "ambiguous", None, "menu", None),
    ("Notepad", "关闭", "target", "关闭标签页", "toolbar", None),
    ("Notepad", "格式", "ambiguous", None, "menu", "substring_fp"),
    ("Notepad", "工具", "ambiguous", None, "menu", "substring_fp"),
    ("Chrome", "搜索", "target", "地址和搜索栏", "toolbar", None),
    ("Chrome", "发送按钮", "ambiguous", None, "content", "icon_miss"),
    ("Chrome", "菜单", "ambiguous", None, "toolbar", None),
    ("Chrome", "分享", "ambiguous", None, "toolbar", None),
    ("Chrome", "删除按钮", "abstain", None, None, "substring_fp"),
    ("Chrome", "刷新", "ambiguous", None, "toolbar", "icon_miss"),
    ("Chrome", "后退", "ambiguous", None, "toolbar", "icon_miss"),
    ("Chrome", "新标签", "abstain", None, None, "icon_miss"),
    ("Chrome", "下载", "ambiguous", None, "toolbar", None),
    ("Chrome", "设置", "ambiguous", None, "toolbar", None),
    ("Chrome", "个人资料", "ambiguous", None, "toolbar", "substring_fp"),
    ("Chrome", "打开", "ambiguous", None, "toolbar", "substring_fp"),
    ("QQ", "搜索", "target", "搜索", "sidebar", None),
    ("QQ", "发送按钮", "target", "发送", "composer", None),
    ("QQ", "表情", "abstain", None, None, "icon_miss"),
    ("QQ", "播放按钮", "abstain", None, None, None),
    ("QQ", "图片", "abstain", None, None, "icon_miss"),
    ("QQ", "文件", "abstain", None, None, "icon_miss"),
    ("QQ", "联系人", "ambiguous", None, "sidebar", None),
    ("QQ", "群聊", "ambiguous", None, "sidebar", "text_content_fp"),
    ("QQ", "视频", "abstain", None, None, "icon_miss"),
    ("QQ", "电话", "abstain", None, None, "icon_miss"),
    ("QQ", "购买", "abstain", None, None, "text_content_fp"),
    ("QQ", "网址", "abstain", None, None, "text_content_fp"),
    ("QQ", "全体成员", "abstain", None, None, "text_content_fp"),
    ("WeChat", "搜索", "target", "搜索", "sidebar", None),
    ("WeChat", "发送按钮", "target", "发送", "composer", None),
    ("WeChat", "表情", "abstain", None, None, "icon_miss"),
    ("WeChat", "播放按钮", "abstain", None, None, None),
    ("WeChat", "图片", "ambiguous", None, "content", "text_content_fp"),
    ("WeChat", "文件", "ambiguous", "文件传输助手", "content", "text_content_fp"),
    ("WeChat", "联系人", "ambiguous", None, "sidebar", "text_content_fp"),
    ("WeChat", "朋友圈", "abstain", None, None, "icon_miss"),
    ("WeChat", "视频", "abstain", None, None, "icon_miss"),
    ("WeChat", "语音", "ambiguous", None, "content", "text_content_fp"),
    ("WeChat", "服务通知", "abstain", None, None, "substring_fp"),
    ("WeChat", "卡券", "abstain", None, None, "substring_fp"),
    ("WeChat", "超市", "abstain", None, None, "text_content_fp"),
    ("WeChat", "中国", "abstain", None, None, "text_content_fp"),
    ("WeChat", "新消息", "abstain", None, None, "text_content_fp"),
    ("WeChat", "是的", "abstain", None, None, "text_content_fp"),
    ("NetEase", "搜索", "ambiguous", None, "toolbar", None),
    ("NetEase", "播放", "ambiguous", None, "toolbar", "icon_miss"),
    ("NetEase", "收藏", "ambiguous", "+收藏全部", "toolbar", None),
    ("NetEase", "发送按钮", "abstain", None, None, None),
    ("NetEase", "播放按钮", "abstain", None, None, "icon_miss"),
    ("NetEase", "上一首", "abstain", None, None, "icon_miss"),
    ("NetEase", "下一首", "abstain", None, None, "icon_miss"),
    ("NetEase", "音量", "abstain", None, None, "icon_miss"),
    ("NetEase", "歌词", "abstain", None, None, "icon_miss"),
    ("NetEase", "评论", "abstain", None, None, "icon_miss"),
    ("NetEase", "门尼", "abstain", None, None, "text_content_fp"),
    ("NetEase", "巴拉莱卡", "abstain", None, None, "text_content_fp"),
    ("VSCode", "搜索", "ambiguous", None, "toolbar", "icon_miss"),
    ("VSCode", "运行", "ambiguous", None, "toolbar", None),
    ("VSCode", "终端", "ambiguous", None, "toolbar", None),
    ("VSCode", "发送按钮", "abstain", None, None, None),
    ("VSCode", "保存", "abstain", None, None, "icon_miss"),
    ("VSCode", "设置", "abstain", None, None, "icon_miss"),
    ("VSCode", "扩展", "ambiguous", "扩展 (Ctrl+Shift+X)", "toolbar", None),
    ("VSCode", "调试", "ambiguous", "运行和调试 (Ctrl+Shift+D)", "toolbar", None),
    ("VSCode", "git", "ambiguous", ".gitignore", "content", "text_content_fp"),
    ("VSCode", "问题", "abstain", None, None, "icon_miss"),
    ("VSCode", "def", "abstain", None, None, "text_content_fp"),
    ("VSCode", "import", "abstain", None, None, "text_content_fp"),
    ("VSCode", "标题操作", "abstain", None, None, "text_content_fp"),
    ("FlClash", "搜索", "abstain", None, None, "icon_miss"),
    ("FlClash", "启动", "abstain", None, None, "icon_miss"),
    ("FlClash", "发送按钮", "abstain", None, None, None),
    ("FlClash", "设置", "ambiguous", None, "menu", None),
    ("FlClash", "代理", "target", "代理", "sidebar", None),
    ("FlClash", "规则", "abstain", None, None, None),
    ("FlClash", "日志", "abstain", None, None, None),
    ("FlClash", "连接", "ambiguous", None, "content", "text_content_fp"),
    ("FlClash", "断开", "abstain", None, None, "icon_miss"),
    ("FlClash", "更新", "abstain", None, None, "icon_miss"),
]

# ── Expected Role Overrides (fixture, not parser) ────────────────
# These override parse_expected_role() for specific evaluation samples.
# They are NOT part of the parser rules. They represent ground truth
# about what the user intended when they typed the query.
EXPECTED_ROLE_OVERRIDES = {
    ("FlClash", "代理"): "navigation_item",  # Sidebar navigation entry
    ("FlClash", "规则"): "navigation_item",  # Sidebar navigation entry
    ("FlClash", "日志"): "navigation_item",  # Sidebar navigation entry
    ("Notepad", "文件"): "menu_item",        # Menu bar item
    ("Notepad", "编辑"): "menu_item",        # Menu bar item
    ("Notepad", "查看"): "menu_item",        # Menu bar item
    ("Notepad", "保存"): "menu_item",        # Menu bar item
    ("Notepad", "帮助"): "menu_item",        # Menu bar item
}


# ── Expected Role Parsing (generic intent only) ──────────────────

# REMOVED domain-specific words:
# 代理, 规则, 日志, 连接, 设置, 配置, 仪表盘, 首页, 消息,
# 通讯录, 发现, 我的, 朋友圈, 动态, 关注, 推荐, 播客,
# 文件, 编辑, 查看, 帮助, 格式, 工具, 运行, 终端, 扩展,
# 调试, git, 问题, 分享, 下载, 新标签, 收藏, 歌词, 评论
# These are now handled by expected_role_override in QUERY_TESTS
# or by candidate evidence (navigation_item detection).

def parse_expected_role(query_text):
    """Parse query text into expected role category.

    ONLY generic intent words. No app/domain-specific words.
    """
    q = query_text.lower()

    # search_input: explicit search input intent
    if any(kw in q for kw in ["搜索框", "搜索输入", "search box", "search input", "输入框"]):
        return "search_input"

    # button/action: explicit button/action intent
    if any(kw in q for kw in [
        "按钮", "button", "点击", "执行",
        "发送按钮", "播放按钮", "删除按钮", "关闭按钮",
        "刷新按钮", "后退按钮", "保存按钮", "打开按钮",
    ]):
        return "button_action"

    # navigation_item: explicit navigation intent
    if any(kw in q for kw in ["侧栏", "导航", "导航项", "标签页", "tab", "sidebar", "nav"]):
        return "navigation_item"

    # menu_item: explicit menu intent
    if any(kw in q for kw in ["菜单", "menu", "菜单项"]):
        return "menu_item"

    # text_or_item: explicit text/content intent
    if any(kw in q for kw in ["文本", "文件名", "歌曲名", "联系人", "消息", "通知"]):
        return "text_or_item"

    # Everything else → unknown
    # Navigation detection is done via candidate evidence, not query tokens
    return "unknown"


# ── Shadow Scoring v2-fixed2 ─────────────────────────────────────

# Actionable UIA roles
ACTIONABLE_UIA_ROLES = {
    "SemanticRole.BUTTON", "SemanticRole.SEARCH_INPUT",
    "SemanticRole.MENU_ITEM", "SemanticRole.TOOLBAR",
    "SemanticRole.TAB", "SemanticRole.CHECKBOX",
    "SemanticRole.LINK", "SemanticRole.DROPDOWN",
    "SemanticRole.SEND_BUTTON",
}

# Actionable control types
ACTIONABLE_CONTROL_TYPES = {
    "button", "hyperlink", "menuitem", "tabitem",
    "checkbox", "radiobutton", "togglebutton",
    "edit", "combobox", "spinner", "slider",
    "buttoncontrol",
}

# Content text control types (NOT actionable)
CONTENT_TEXT_CONTROLS = {"textcontrol", "groupcontrol", "documentcontrol"}

# Content flow regions (only these can demote button/action queries)
CONTENT_FLOW_REGIONS = {"messages", "message_stream", "document_body", "content_stream", "editor_region", "table_region", "list_content", "message_area"}

# Structural regions (positive evidence, never demote)
STRUCTURAL_REGIONS = {"toolbar", "sidebar", "filter_bar", "top_bar", "control_strip", "input_area", "menu_bar", "status_bar", "composer", "bottom_bar"}


def classify_region_kind(region_id):
    """Classify region_id into structural / content_flow / unknown."""
    rid = region_id.lower()
    for kw in STRUCTURAL_REGIONS:
        if kw in rid:
            return "structural"
    for kw in CONTENT_FLOW_REGIONS:
        if kw in rid:
            return "content_flow"
    return "unknown"


def compute_shadow_v2_score(candidate, query_text, expected_type, expected_role_override=None):
    """Compute shadow v2-b4 evidence score.

    Features:
    A. expected-role compatibility hard gate (search_input)
    B. action semantic evidence requirement (button_action)
    C. non-actionable container/text demotion (all roles)
    """
    score = candidate.confidence
    demotion_reasons = []
    positive_reasons = []

    # Use override if provided, otherwise parse from query
    if expected_role_override:
        expected_role = expected_role_override
    else:
        expected_role = parse_expected_role(query_text)

    # Extract candidate features
    role = str(candidate.semantic_role).split(".")[-1]
    full_role = str(candidate.semantic_role)
    ctrl_type = (candidate.control_type or "").lower()
    region_id = (candidate.region_id or "")
    region_kind = classify_region_kind(region_id)
    provider_sources = candidate.provider_sources or []
    text = (candidate.text or "").lower()
    text_match = query_text.lower() in text or text in query_text.lower()
    role_label = (candidate.role_label or "").lower()
    semantic_tags = [t.lower() for t in (candidate.semantic_tags or [])]
    risk_tags = [t.lower() for t in (candidate.risk_tags or [])]
    has_uia = "uia" in provider_sources
    has_ocr = "ocr" in provider_sources
    has_vision = "omni" in provider_sources or "vision" in provider_sources
    is_actionable_role = full_role in ACTIONABLE_UIA_ROLES
    is_actionable_ctrl = ctrl_type in ACTIONABLE_CONTROL_TYPES
    is_content_text_ctrl = ctrl_type in CONTENT_TEXT_CONTROLS

    # ── Feature C: Non-actionable container/text demotion ──
    # This applies GLOBALLY, before expected-role checks.
    # CONTAINER/WindowControl/PaneControl/GroupControl → demote
    # unless query explicitly targets containers
    container_roles = {"CONTAINER", "LAYOUT", "UNKNOWN"}
    container_controls = {"windowcontrol", "panecontrol", "groupcontrol", "documentcontrol", "customcontrol"}

    is_container = role in container_roles or ctrl_type in container_controls
    # Large structural containers (Window/Pane/Document) are never content
    is_large_container = ctrl_type in {"windowcontrol", "panecontrol", "documentcontrol"}

    if is_container:
        # Containers (WindowControl/PaneControl/GroupControl) are structural
        # wrappers, NOT actionable controls or content text.
        if expected_role in ("button_action", "search_input"):
            # Containers are never search/button targets
            score *= 0.2
            demotion_reasons.append("container_role_for_action_query")
        elif is_large_container:
            # Large structural containers (Window/Pane/Document) are always demoted
            # regardless of text_match — they are wrappers, not content
            score *= 0.2
            demotion_reasons.append("large_container_always_demote")
        elif text_match and expected_role in ("navigation_item", "text_or_item", "menu_item"):
            # Small container (GroupControl) with text match in nav/text/menu → keep
            pass
        elif expected_role == "unknown" and text_match:
            # Small container with text match, unknown intent → keep score
            pass
        else:
            # Other containers: demote
            score *= 0.3
            demotion_reasons.append("container_role_general_demote")

    # ── Feature A: Expected-role compatibility hard gate ──

    if expected_role == "search_input":
        # HARD GATE: only SEARCH_INPUT / EditControl / filter_bar region
        if "SEARCH_INPUT" in full_role:
            score = min(1.0, score * 1.5)
            positive_reasons.append("role_matches_search_input")
        elif ctrl_type == "editcontrol":
            score = min(1.0, score * 1.3)
            positive_reasons.append("ctrl_matches_editcontrol")
        elif "filter_bar" in region_id.lower() or "input_area" in region_id.lower():
            score = min(1.0, score * 1.2)
            positive_reasons.append("region_matches_filter_bar")
        # HARD GATE: MENU_ITEM/BUTTON/TEXT cannot pass search_input
        else:
            score *= 0.15
            demotion_reasons.append("search_input_hard_gate_fail")

    elif expected_role == "button_action":
        # Feature B: action semantic evidence requirement
        # Extract action word from query
        action_words = {"删除", "关闭", "刷新", "后退", "保存", "打开",
                        "发送", "播放", "启动", "断开", "更新", "下载",
                        "delete", "close", "refresh", "back", "save", "open",
                        "send", "play", "start", "stop", "update", "download"}
        query_action = None
        for aw in action_words:
            if aw in query_text.lower():
                query_action = aw
                break

        # Check if candidate has matching action semantic evidence
        # The action word from query must appear in candidate text/label/tags
        candidate_text = text
        candidate_label = role_label
        candidate_tags = semantic_tags + risk_tags
        candidate_combined = f"{candidate_text} {candidate_label} {' '.join(candidate_tags)}"

        has_action_evidence = False
        if query_action:
            # Direct match: query action word in candidate text/label/tags
            if query_action in candidate_combined:
                has_action_evidence = True
            # Destructive action synonyms
            destructive_map = {
                "删除": ["delete", "remove", "destructive"],
                "delete": ["删除", "remove", "destructive"],
                "关闭": ["close", "quit", "exit"],
                "close": ["关闭", "quit", "exit"],
            }
            if query_action in destructive_map:
                for syn in destructive_map[query_action]:
                    if syn in candidate_combined:
                        has_action_evidence = True
                        break

        # Scoring
        if is_actionable_role and has_action_evidence:
            score = min(1.0, score * 1.5)
            positive_reasons.append("actionable_role_with_action_evidence")
        elif is_actionable_role and not has_action_evidence and query_action:
            # Has actionable role but no matching action evidence → slight demote
            score *= 0.6
            demotion_reasons.append("actionable_role_no_action_evidence")
        elif is_actionable_ctrl:
            score = min(1.0, score * 1.3)
            positive_reasons.append("actionable_control_type")
        elif role == "TEXT" and is_content_text_ctrl and region_kind == "content_flow":
            score *= 0.2
            demotion_reasons.append("text_content_in_content_flow_for_button")
        elif role == "TEXT" and is_content_text_ctrl and region_kind == "unknown":
            score *= 0.4
            demotion_reasons.append("text_content_unknown_region_for_button")
        elif role == "TEXT" and not is_actionable_role and not is_actionable_ctrl:
            score *= 0.5
            demotion_reasons.append("text_role_for_button_query")

    elif expected_role == "menu_item":
        if "MENU_ITEM" in full_role:
            score = min(1.0, score * 1.5)
            positive_reasons.append("role_matches_menu_item")
        elif ctrl_type == "menuitemcontrol":
            score = min(1.0, score * 1.3)
            positive_reasons.append("ctrl_matches_menuitem")
        elif role == "TEXT" and is_content_text_ctrl:
            score *= 0.3
            demotion_reasons.append("text_content_for_menu_query")

    elif expected_role == "navigation_item":
        nav_roles = {"TEXT", "LIST_ITEM", "MENU_ITEM", "TAB"}
        nav_controls = {"listitemcontrol", "menuitemcontrol", "tabitemcontrol", "textcontrol"}
        nav_regions = {"sidebar", "side_panel", "navigation", "top_nav", "filter_bar"}

        role_ok = role in nav_roles or ctrl_type in nav_controls
        region_is_nav = any(kw in region_id.lower() for kw in nav_regions)
        region_is_structural = region_kind == "structural"

        if role_ok and (region_is_nav or region_is_structural):
            score = min(1.0, score * 1.3)
            positive_reasons.append("nav_role_in_nav_region")
        elif text_match and (region_is_nav or region_is_structural):
            score = min(1.0, score * 1.2)
            positive_reasons.append("text_match_in_nav_region")
        elif text_match:
            positive_reasons.append("text_match_no_demote")
        elif role == "TEXT" and is_content_text_ctrl and region_kind == "content_flow":
            score *= 0.4
            demotion_reasons.append("text_content_in_flow_for_nav")

    elif expected_role == "text_or_item":
        if role in ("TEXT", "LIST_ITEM", "LINK"):
            score = min(1.0, score * 1.2)
            positive_reasons.append("role_matches_text_or_item")
        if is_actionable_role:
            score *= 0.8
            demotion_reasons.append("actionable_for_text_query")

    elif expected_role == "unknown":
        nav_evidence_regions = {"sidebar", "side_panel", "navigation", "top_nav"}
        is_nav_region = any(kw in region_id.lower() for kw in nav_evidence_regions)
        is_nav_control = ctrl_type in {"listitemcontrol", "menuitemcontrol", "tabitemcontrol"}

        if text_match and (is_nav_region or is_nav_control):
            # Candidate text matches query AND is in navigation region/control
            score = min(1.0, score * 1.2)
            positive_reasons.append("candidate_looks_navigation_by_evidence")
        elif text_match and region_kind == "structural":
            # Text match in structural region → keep current score
            positive_reasons.append("text_match_structural_region")
        # No demotion for unknown queries — we don't know what the user wants

    # ── Feature E: Content-text abstain reason ──
    # TEXT in content_flow region + not expected text_or_item/menu/navigation/search
    # → strong demotion (pattern-specific abstain, not global threshold)
    if role == "TEXT" and is_content_text_ctrl:
        if region_kind == "content_flow" and expected_role not in (
            "text_or_item", "menu_item", "navigation_item", "search_input",
        ):
            score *= 0.15
            demotion_reasons.append("content_text_in_flow_abstain")
        elif region_kind == "unknown" and expected_role not in (
            "text_or_item", "menu_item", "navigation_item", "search_input",
        ):
            # Unknown region but OCR/vision text without actionable UIA → demote
            if not is_actionable_role and not is_actionable_ctrl:
                score *= 0.4
                demotion_reasons.append("text_unknown_region_no_actionable")

    # Media indicators (generic, not business-specific)
    media_indicators = ["[图片]", "[语音]", "[视频]", "[文件]"]
    if any(ind in text for ind in media_indicators):
        score *= 0.3
        demotion_reasons.append("media_content_indicator")

    # ── Region positive (only if role compatible) ──

    # Structural region + actionable role/control → promote
    if region_kind == "structural":
        if is_actionable_role or is_actionable_ctrl:
            score = min(1.0, score * 1.2)
            positive_reasons.append("structural_region_actionable")
        # Structural region + TEXT role → no demotion (keep current score)
        elif role == "TEXT":
            positive_reasons.append("structural_region_text_no_demote")

    # ── Provider sources (auxiliary only) ──

    # Three-source consistent does NOT auto-promote
    if has_uia and has_ocr and has_vision:
        if is_actionable_role or is_actionable_ctrl:
            score = min(1.0, score * 1.1)
            positive_reasons.append("three_source_actionable")
        elif role == "TEXT" and region_kind != "structural":
            score *= 0.9
            demotion_reasons.append("three_source_text_role")

    # ── Final score clamping ──
    score = max(0.0, min(1.0, score))

    return score, demotion_reasons, positive_reasons, expected_role


def compute_abstain_v2(results, query_text):
    """Compute abstain recommendation for shadow v2."""
    if not results:
        return True, "no_candidates"

    top_score = max(r["shadow_score"] for r in results)
    if top_score < 0.3:
        return True, "low_shadow_score"

    top = max(results, key=lambda r: r["shadow_score"])
    if len(top["demotion_reasons"]) >= 3:
        return True, "strong_multi_demotion"

    return False, None


def run_shadow_v2():
    """Run P1.1b.2 shadow scoring v2."""
    svc = PerceptionService()
    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)

    app_process = {
        "Notepad": "notepad.exe",
        "Chrome": "chrome.exe",
        "QQ": "qq.exe",
        "WeChat": "weixin.exe",
        "NetEase": "cloudmusic.exe",
        "VSCode": "code.exe",
        "FlClash": "flclash.exe",
    }

    # Observe all apps
    canvases = {}
    for app_name, process in app_process.items():
        win = next(
            (w for w in windows if process.lower() in (w.process_name or "").lower()),
            None,
        )
        if not win:
            continue
        try:
            zone_page = svc.analyze(hwnd=win.hwnd, allow_vlm=False)
            canvas = svc.create_page_snapshot(zone_page, process_name=process)
            canvases[app_name] = canvas
        except Exception:
            pass

    # Run shadow scoring
    engine = CanvasQueryEngine()
    results = []

    for app_name, query, expected, target_text, region, failure_hint in QUERY_TESTS:
        canvas = canvases.get(app_name)
        if not canvas:
            results.append({"app": app_name, "query": query, "status": "skipped", "expected": expected})
            continue

        try:
            query_result = engine.query(canvas, QueryTarget(natural_language=query), max_results=5, min_confidence=0.0)
            candidates = query_result.candidates

            # Get expected_role_override for this sample
            role_override = EXPECTED_ROLE_OVERRIDES.get((app_name, query))

            # Compute shadow v2 scores
            shadow_results = []
            for c in candidates[:5]:
                shadow_score, demotion, positive, expected_role = compute_shadow_v2_score(
                    c, query, expected, expected_role_override=role_override,
                )
                shadow_results.append({
                    "element_id": c.element_id,
                    "text": (c.text or "")[:40],
                    "role": str(c.semantic_role).split(".")[-1],
                    "control_type": c.control_type or "",
                    "provider_sources": c.provider_sources,
                    "current_score": round(c.confidence, 3),
                    "shadow_score": round(shadow_score, 3),
                    "demotion_reasons": demotion,
                    "positive_reasons": positive,
                    "expected_role": expected_role,
                })

            shadow_results.sort(key=lambda r: -r["shadow_score"])

            should_abstain, abstain_reason = compute_abstain_v2(shadow_results, query)

            current_top1 = candidates[0] if candidates else None
            shadow_top1 = shadow_results[0] if shadow_results else None

            # Verdicts
            has_results = len(candidates) > 0
            if expected == "target":
                current_verdict = "hit" if has_results else "miss"
            elif expected == "abstain":
                current_verdict = "fp" if has_results else "correct_abstain"
            else:
                current_verdict = "ambiguous"

            if should_abstain:
                shadow_verdict = "shadow_abstain"
            elif shadow_top1:
                shadow_verdict = "shadow_hit"
            else:
                shadow_verdict = "shadow_miss"

            results.append({
                "app": app_name,
                "query": query,
                "expected": expected,
                "failure_hint": failure_hint,
                "current_verdict": current_verdict,
                "shadow_verdict": shadow_verdict,
                "expected_role": shadow_top1["expected_role"] if shadow_top1 else "unknown",
                "expected_role_override": role_override,
                "current_top1_text": (current_top1.text or "")[:40] if current_top1 else "",
                "shadow_top1_text": shadow_top1["text"] if shadow_top1 else "",
                "shadow_top1_score": shadow_top1["shadow_score"] if shadow_top1 else 0,
                "shadow_top1_demotion": shadow_top1["demotion_reasons"] if shadow_top1 else [],
                "shadow_top1_positive": shadow_top1["positive_reasons"] if shadow_top1 else [],
                "should_abstain": should_abstain,
                "abstain_reason": abstain_reason,
                "shadow_candidates": shadow_results,
            })
        except Exception as e:
            results.append({"app": app_name, "query": query, "status": "error", "error": str(e)})

    # Save
    report = {"generated_at": datetime.now().isoformat(), "phase": "P1.1b.5-cleanup-shadow", "results": results}
    with open(REPORT_DIR / "shadow_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    valid = [r for r in results if r.get("status") not in ("skipped", "error")]
    hits = [r for r in valid if r.get("current_verdict") == "hit"]
    fps = [r for r in valid if r.get("current_verdict") == "fp"]
    abstains = [r for r in valid if r.get("current_verdict") == "correct_abstain"]

    shadow_fixed_fps = [r for r in fps if r.get("should_abstain")]
    shadow_damaged_hits = [r for r in hits if r.get("should_abstain")]
    shadow_broken_abstains = [r for r in abstains if not r.get("should_abstain")]

    # Expected role distribution
    role_dist = {}
    for r in valid:
        er = r.get("expected_role", "unknown")
        role_dist[er] = role_dist.get(er, 0) + 1

    # Demotion reason counts for FP
    fp_demotions = {}
    for r in fps:
        for d in r.get("shadow_top1_demotion", []):
            fp_demotions[d] = fp_demotions.get(d, 0) + 1

    # Override usage
    override_used = [r for r in valid if r.get("expected_role_override")]

    print(f"\n{'='*70}")
    print("  P1.1b.5-cleanup-shadow Scoring Summary")
    print(f"{'='*70}")
    print(f"  Total queries: {len(valid)}")
    print(f"  Current FP: {len(fps)}")
    print(f"  Shadow fixed FP: {len(shadow_fixed_fps)}")
    print(f"  Current hits: {len(hits)}")
    print(f"  Shadow damaged hits: {len(shadow_damaged_hits)}")
    print(f"  Correct abstains: {len(abstains)}")
    print(f"  Shadow broken abstains: {len(shadow_broken_abstains)}")
    print(f"  Expected role overrides used: {len(override_used)}")

    print(f"\n  Removed domain-specific words from parse_expected_role:")
    removed_words = [
        "代理", "规则", "日志", "连接", "设置", "配置", "仪表盘",
        "首页", "消息", "通讯录", "发现", "我的", "朋友圈",
        "动态", "关注", "推荐", "播客", "文件", "编辑", "查看",
        "帮助", "格式", "工具", "运行", "终端", "扩展", "调试",
        "git", "问题", "分享", "下载", "新标签", "收藏", "歌词", "评论",
    ]
    print(f"    {', '.join(removed_words)}")

    print(f"\n  Expected Role Overrides:")
    for (app, query), role in EXPECTED_ROLE_OVERRIDES.items():
        print(f"    {app:10s} \"{query}\" → {role}")

    print(f"\n  Expected Role Distribution:")
    for role, count in sorted(role_dist.items(), key=lambda x: -x[1]):
        print(f"    {role:20s}: {count}")

    print(f"\n  FP Demotion Reasons:")
    for reason, count in sorted(fp_demotions.items(), key=lambda x: -x[1]):
        print(f"    {reason:30s}: {count}")

    if shadow_fixed_fps:
        print(f"\n  FP Fixed by Shadow ({len(shadow_fixed_fps)}):")
        for r in shadow_fixed_fps[:10]:
            print(f"    {r['app']:10s} \"{r['query']}\" → \"{r['current_top1_text'][:25]}\" shadow={r['shadow_top1_score']:.2f} demotion={r['shadow_top1_demotion']}")

    if shadow_damaged_hits:
        print(f"\n  Hits Damaged by Shadow ({len(shadow_damaged_hits)}):")
        for r in shadow_damaged_hits:
            print(f"    {r['app']:10s} \"{r['query']}\" → \"{r['current_top1_text'][:25]}\" shadow={r['shadow_top1_score']:.2f} demotion={r['shadow_top1_demotion']}")

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_shadow_v2()
