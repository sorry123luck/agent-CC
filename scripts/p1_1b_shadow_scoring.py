"""P1.1b Shadow Scoring: Evidence-driven candidate scoring.

Shadow-only analysis that computes shadow_evidence_score alongside
current query results. Does NOT change actual returned results.

Usage:
    python scripts/p1_1b_shadow_scoring.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OPENCLAW_U4_LAYOUT_SHADOW"] = "1"
os.environ["OPENCLAW_URE_P0_SHADOW"] = "1"
os.environ["OPENCLAW_VTL_0_SHADOW"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.perception_service import PerceptionService
from src.windows.window_enum import WindowEnumService
from src.canvas.query_engine import CanvasQueryEngine, QueryTarget

REPORT_DIR = Path("reports/p1_1b_shadow_scoring")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Import corrected query set from P1.1a.1-fix
QUERY_TESTS = [
    # ── Notepad ──
    ("Notepad", "文件", "target", "文件", "menu", "File menu item exists", None),
    ("Notepad", "编辑", "target", "编辑", "menu", "Edit menu item exists", None),
    ("Notepad", "查看", "target", "查看", "menu", "View menu item exists", None),
    ("Notepad", "搜索框", "abstain", None, None, "Notepad has no search box", "substring_fp"),
    ("Notepad", "发送按钮", "abstain", None, None, "Notepad has no send button", None),
    ("Notepad", "保存", "ambiguous", None, "menu", "Save may be under File menu", None),
    ("Notepad", "打开", "ambiguous", None, "menu", "Open may be under File menu", None),
    ("Notepad", "帮助", "ambiguous", None, "menu", "Help menu may or may not exist", None),
    ("Notepad", "格式", "ambiguous", None, "menu", "Format menu may or may not exist", None),
    ("Notepad", "关闭", "target", "关闭标签页", "toolbar", "Close tab button exists", None),
    ("Notepad", "格式", "ambiguous", None, "menu", "Format menu may exist", "substring_fp"),
    ("Notepad", "工具", "ambiguous", None, "menu", "Tools menu may exist", "substring_fp"),
    ("Chrome", "搜索", "target", "地址和搜索栏", "toolbar", "Chrome address bar exists", None),
    ("Chrome", "发送按钮", "ambiguous", None, "content", "ChatGPT send button may be icon-only", "icon_miss"),
    ("Chrome", "菜单", "ambiguous", None, "toolbar", "Menu exists but text is 'wer Plus'", None),
    ("Chrome", "分享", "ambiguous", None, "toolbar", "Share button may exist", None),
    ("Chrome", "删除按钮", "abstain", None, None, "No delete button in current view", "substring_fp"),
    ("Chrome", "刷新", "ambiguous", None, "toolbar", "Refresh may be icon-only", "icon_miss"),
    ("Chrome", "后退", "ambiguous", None, "toolbar", "Back may be icon-only", "icon_miss"),
    ("Chrome", "新标签", "abstain", None, None, "No explicit new tab button text", "icon_miss"),
    ("Chrome", "下载", "ambiguous", None, "toolbar", "Download may exist in menu", None),
    ("Chrome", "设置", "ambiguous", None, "toolbar", "Settings may exist in menu", None),
    ("Chrome", "个人资料", "ambiguous", None, "toolbar", "Profile text in menu description", "substring_fp"),
    ("Chrome", "打开", "ambiguous", None, "toolbar", "Open may be in menu", "substring_fp"),
    ("QQ", "搜索", "target", "搜索", "sidebar", "QQ search input exists", None),
    ("QQ", "发送按钮", "abstain", None, None, "No visible send button in current view", None),
    ("QQ", "表情", "abstain", None, None, "No visible emoji button in current view", "icon_miss"),
    ("QQ", "播放按钮", "abstain", None, None, "QQ has no play button", None),
    ("QQ", "图片", "abstain", None, None, "No visible image button in current view", "icon_miss"),
    ("QQ", "文件", "abstain", None, None, "No visible file button in current view", "icon_miss"),
    ("QQ", "联系人", "ambiguous", None, "sidebar", "Contacts may be in sidebar tab", None),
    ("QQ", "群聊", "ambiguous", None, "sidebar", "Group chat may be in sidebar", "text_content_fp"),
    ("QQ", "视频", "abstain", None, None, "No visible video call button", "icon_miss"),
    ("QQ", "电话", "abstain", None, None, "No visible phone button", "icon_miss"),
    ("QQ", "购买", "abstain", None, None, "Purchase text in message", "text_content_fp"),
    ("QQ", "网址", "abstain", None, None, "URL text in message", "text_content_fp"),
    ("QQ", "全体成员", "abstain", None, None, "@all text in message", "text_content_fp"),
    ("WeChat", "搜索", "target", "搜索", "sidebar", "WeChat search input exists", None),
    ("WeChat", "发送按钮", "abstain", None, None, "No visible send button in current view", None),
    ("WeChat", "表情", "abstain", None, None, "No visible emoji button in current view", "icon_miss"),
    ("WeChat", "播放按钮", "abstain", None, None, "WeChat has no play button", None),
    ("WeChat", "图片", "ambiguous", None, "content", "[图片] in messages, not a button", "text_content_fp"),
    ("WeChat", "文件", "ambiguous", "文件传输助手", "content", "Contact name contains 文件", "text_content_fp"),
    ("WeChat", "联系人", "ambiguous", None, "sidebar", "Contacts tab may exist", "text_content_fp"),
    ("WeChat", "朋友圈", "abstain", None, None, "No visible Moments button", "icon_miss"),
    ("WeChat", "视频", "abstain", None, None, "No visible video button", "icon_miss"),
    ("WeChat", "语音", "ambiguous", None, "content", "[语音] in messages, not a button", "text_content_fp"),
    ("WeChat", "服务通知", "abstain", None, None, "Service notification text, not a button", "substring_fp"),
    ("WeChat", "卡券", "abstain", None, None, "Coupon text in message, not a button", "substring_fp"),
    ("WeChat", "超市", "abstain", None, None, "Supermarket text in contact name", "text_content_fp"),
    ("WeChat", "中国", "abstain", None, None, "Country text in message", "text_content_fp"),
    ("WeChat", "新消息", "abstain", None, None, "New message text in notification", "text_content_fp"),
    ("WeChat", "是的", "abstain", None, None, "Message text 'yes'", "text_content_fp"),
    ("NetEase", "搜索", "ambiguous", None, "toolbar", "Search input exists but text is 'Josh Turner'", None),
    ("NetEase", "播放", "ambiguous", None, "toolbar", "Play button may be icon-only", "icon_miss"),
    ("NetEase", "收藏", "ambiguous", "+收藏全部", "toolbar", "Favorite button exists", None),
    ("NetEase", "发送按钮", "abstain", None, None, "NetEase has no send button", None),
    ("NetEase", "播放按钮", "abstain", None, None, "No visible play button text", "icon_miss"),
    ("NetEase", "上一首", "abstain", None, None, "No visible previous track button", "icon_miss"),
    ("NetEase", "下一首", "abstain", None, None, "No visible next track button", "icon_miss"),
    ("NetEase", "音量", "abstain", None, None, "No visible volume control", "icon_miss"),
    ("NetEase", "歌词", "abstain", None, None, "No visible lyrics button", "icon_miss"),
    ("NetEase", "评论", "abstain", None, None, "No visible comments button", "icon_miss"),
    ("NetEase", "门尼", "abstain", None, None, "Artist name in title", "text_content_fp"),
    ("NetEase", "巴拉莱卡", "abstain", None, None, "Song name in title", "text_content_fp"),
    ("VSCode", "搜索", "ambiguous", None, "toolbar", "Search may exist but no explicit text", "icon_miss"),
    ("VSCode", "运行", "ambiguous", None, "toolbar", "Run/debug may exist", None),
    ("VSCode", "终端", "ambiguous", None, "toolbar", "Terminal may exist", None),
    ("VSCode", "发送按钮", "abstain", None, None, "VSCode has no send button", None),
    ("VSCode", "保存", "abstain", None, None, "No visible save button text", "icon_miss"),
    ("VSCode", "设置", "abstain", None, None, "No visible settings button text", "icon_miss"),
    ("VSCode", "扩展", "ambiguous", "扩展 (Ctrl+Shift+X)", "toolbar", "Extensions button exists", None),
    ("VSCode", "调试", "ambiguous", "运行和调试 (Ctrl+Shift+D)", "toolbar", "Debug button exists", None),
    ("VSCode", "git", "ambiguous", ".gitignore", "content", "Git-related file exists, not a button", "text_content_fp"),
    ("VSCode", "问题", "abstain", None, None, "No visible problems panel button", "icon_miss"),
    ("VSCode", "def", "abstain", None, None, "Python keyword in code", "text_content_fp"),
    ("VSCode", "import", "abstain", None, None, "Python keyword in code", "text_content_fp"),
    ("VSCode", "标题操作", "abstain", None, None, "Toolbar label text", "text_content_fp"),
    ("FlClash", "搜索", "abstain", None, None, "No visible search button", "icon_miss"),
    ("FlClash", "启动", "abstain", None, None, "No visible start button", "icon_miss"),
    ("FlClash", "发送按钮", "abstain", None, None, "FlClash has no send button", None),
    ("FlClash", "设置", "ambiguous", None, "menu", "Settings may be under menu", None),
    ("FlClash", "代理", "target", "代理", "sidebar", "Proxy navigation exists", None),
    ("FlClash", "规则", "abstain", None, None, "No visible rules button", None),
    ("FlClash", "日志", "abstain", None, None, "No visible logs button", None),
    ("FlClash", "连接", "ambiguous", None, "content", "Connection status visible but not a button", "text_content_fp"),
    ("FlClash", "断开", "abstain", None, None, "No visible disconnect button", "icon_miss"),
    ("FlClash", "更新", "abstain", None, None, "No visible update button", "icon_miss"),
]


# ── Shadow Scoring Rules ──────────────────────────────────────

# Actionable UIA roles (strong positive evidence)
ACTIONABLE_UIA_ROLES = {
    "SemanticRole.BUTTON", "SemanticRole.SEARCH_INPUT",
    "SemanticRole.MENU_ITEM", "SemanticRole.TOOLBAR",
    "SemanticRole.TAB", "SemanticRole.CHECKBOX",
    "SemanticRole.LINK", "SemanticRole.DROPDOWN",
}

# Actionable UIA control types
ACTIONABLE_CONTROL_TYPES = {
    "button", "hyperlink", "menuitem", "tabitem",
    "checkbox", "radiobutton", "togglebutton",
    "edit", "combobox", "spinner", "slider",
}

# Toolbar/sidebar region roles (positive evidence)
STRUCTURAL_REGION_ROLES = {
    "toolbar", "sidebar", "top_bar", "control_strip",
    "menu_bar", "status_bar", "input_area",
}

# Content area region roles (negative evidence for actionable queries)
CONTENT_REGION_ROLES = {
    "content", "message_area", "document_body",
    "chat_message_area", "content_stream",
}

# Query intents that expect actionable controls
ACTIONABLE_QUERY_KEYWORDS = {
    "按钮", "button", "发送", "搜索", "播放", "菜单", "输入框",
    "点击", "打开", "关闭", "保存", "删除", "刷新", "后退",
    "下载", "设置", "搜索框", "编辑", "查看",
}


def compute_shadow_score(candidate, query_text, expected_type):
    """Compute shadow evidence score for a candidate.

    Returns: (shadow_score, demotion_reasons, positive_reasons)
    """
    score = candidate.confidence  # Start from current confidence
    demotion_reasons = []
    positive_reasons = []

    # ── Negative evidence ──

    # 1. OCR-only candidate (no UIA)
    provider_sources = candidate.provider_sources or []
    has_uia = "uia" in provider_sources
    has_ocr = "ocr" in provider_sources
    has_omni = "omni" in provider_sources or "vision" in provider_sources

    if has_ocr and not has_uia:
        score *= 0.6
        demotion_reasons.append("ocr_only_no_uia")

    # 2. Text content in messages/files/songs/contacts
    text = (candidate.text or "").lower()
    role = str(candidate.semantic_role)

    # Check if text looks like message/file/song content
    content_indicators = [
        "[图片]", "[语音]", "[视频]", "[文件]",
        "新消息", "服务通知", "卡券",
        "条新消息", "条消息",
    ]
    if any(ind in text for ind in content_indicators):
        score *= 0.3
        demotion_reasons.append("message_content_text")

    # 3. Query expects actionable, candidate is pure text in content area
    query_expects_actionable = any(
        kw in query_text for kw in ACTIONABLE_QUERY_KEYWORDS
    )
    is_content_region = any(
        rr in role.lower() for rr in CONTENT_REGION_ROLES
    )
    if query_expects_actionable and has_ocr and not has_uia and not has_omni:
        score *= 0.5
        demotion_reasons.append("ocr_text_for_actionable_query")

    # 4. Candidate is in dynamic content zone (message/file/song)
    if "DYNAMIC" in role or "CONTENT_STREAM" in role:
        score *= 0.4
        demotion_reasons.append("dynamic_content_zone")

    # 5. Three-source consistent but all text labels
    if has_uia and has_ocr and has_omni:
        # Check if UIA role is just TEXT/LAYOUT, not actionable
        if role not in ACTIONABLE_UIA_ROLES:
            score *= 0.7
            demotion_reasons.append("three_source_but_text_label")

    # ── Positive evidence ──

    # 1. UIA actionable role
    if role in ACTIONABLE_UIA_ROLES:
        score = min(1.0, score * 1.5)
        positive_reasons.append("uia_actionable_role")

    # 2. In toolbar/sidebar/top_bar/control_strip
    if any(rr in role.lower() for rr in STRUCTURAL_REGION_ROLES):
        score = min(1.0, score * 1.3)
        positive_reasons.append("structural_region")

    # 3. Omni control evidence
    if has_omni:
        score = min(1.0, score * 1.2)
        positive_reasons.append("omni_control_evidence")

    # 4. Exact text match for text/list/contact queries
    if expected_type == "target" and query_text in text:
        score = min(1.0, score * 1.4)
        positive_reasons.append("exact_text_match")

    # 5. UIA actionable control type
    ctrl_type = (candidate.control_type or "").lower()
    if ctrl_type in ACTIONABLE_CONTROL_TYPES:
        score = min(1.0, score * 1.3)
        positive_reasons.append("uia_actionable_control_type")

    return score, demotion_reasons, positive_reasons


def compute_abstain_recommendation(results, query_text):
    """Compute whether the query should abstain based on shadow scores."""
    if not results:
        return True, "no_candidates"

    top_score = max(r["shadow_score"] for r in results)
    if top_score < 0.3:
        return True, "low_shadow_score"

    # Check if top candidate has strong demotion
    top = max(results, key=lambda r: r["shadow_score"])
    if len(top["demotion_reasons"]) >= 2:
        return True, "strong_demotion"

    return False, None


def run_shadow_scoring():
    """Run P1.1b shadow scoring analysis."""
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

    # Run shadow scoring for each query
    engine = CanvasQueryEngine()
    results = []

    for app_name, query, expected, target_text, region, notes, failure_hint in QUERY_TESTS:
        canvas = canvases.get(app_name)
        if not canvas:
            results.append({
                "app": app_name, "query": query, "status": "skipped",
                "expected": expected, "failure_hint": failure_hint,
            })
            continue

        try:
            query_result = engine.query(canvas, QueryTarget(natural_language=query), max_results=5, min_confidence=0.0)
            candidates = query_result.candidates

            # Compute shadow scores for top candidates
            shadow_results = []
            for c in candidates[:5]:
                shadow_score, demotion, positive = compute_shadow_score(c, query, expected)
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
                })

            # Sort by shadow_score
            shadow_results.sort(key=lambda r: -r["shadow_score"])

            # Compute abstain recommendation
            should_abstain, abstain_reason = compute_abstain_recommendation(shadow_results, query)

            # Get current top-1 and shadow top-1
            current_top1 = candidates[0] if candidates else None
            shadow_top1 = shadow_results[0] if shadow_results else None

            # Determine current verdict
            has_results = len(candidates) > 0
            if expected == "target":
                current_verdict = "hit" if has_results else "miss"
            elif expected == "abstain":
                current_verdict = "fp" if has_results else "correct_abstain"
            else:
                current_verdict = "ambiguous"

            # Determine shadow verdict
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
                "current_top1_text": (current_top1.text or "")[:40] if current_top1 else "",
                "shadow_top1_text": shadow_top1["text"] if shadow_top1 else "",
                "shadow_top1_score": shadow_top1["shadow_score"] if shadow_top1 else 0,
                "shadow_top1_demotion": shadow_top1["demotion_reasons"] if shadow_top1 else [],
                "should_abstain": should_abstain,
                "abstain_reason": abstain_reason,
                "shadow_candidates": shadow_results,
            })
        except Exception as e:
            results.append({
                "app": app_name, "query": query, "status": "error",
                "error": str(e), "expected": expected,
            })

    # Save results
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "P1.1b-shadow-scoring",
        "total_queries": len(results),
        "results": results,
    }
    with open(REPORT_DIR / "shadow_scoring_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print("  P1.1b Shadow Scoring Summary")
    print(f"{'='*70}")

    valid = [r for r in results if r.get("status") != "skipped" and r.get("status") != "error"]

    # Current vs Shadow comparison
    current_fps = [r for r in valid if r.get("current_verdict") == "fp"]
    shadow_fps = [r for r in valid if r.get("should_abstain") and r.get("current_verdict") == "fp"]
    hits = [r for r in valid if r.get("current_verdict") == "hit"]
    shadow_damaged_hits = [r for r in hits if r.get("should_abstain")]
    correct_abstains = [r for r in valid if r.get("current_verdict") == "correct_abstain"]
    shadow_broken_abstains = [r for r in correct_abstains if not r.get("should_abstain")]

    print(f"  Total queries: {len(valid)}")
    print(f"  Current FP: {len(current_fps)}")
    print(f"  Shadow FP (abstained): {len(shadow_fps)}")
    print(f"  FP reduced: {len(current_fps) - len(shadow_fps) if len(current_fps) > len(shadow_fps) else 0}")
    print(f"  Current hits: {len(hits)}")
    print(f"  Shadow damaged hits: {len(shadow_damaged_hits)}")
    print(f"  Current correct abstains: {len(correct_abstains)}")
    print(f"  Shadow broken abstains: {len(shadow_broken_abstains)}")

    # Show FP that shadow would fix
    if shadow_fps:
        print(f"\n  FP → Abstain (shadow fixed):")
        for r in shadow_fps[:5]:
            print(f"    {r['app']:10s} \"{r['query']}\" → \"{r['current_top1_text'][:25]}\" shadow_score={r['shadow_top1_score']:.2f}")

    # Show hits that shadow would damage
    if shadow_damaged_hits:
        print(f"\n  HIT → Abstain (shadow damaged):")
        for r in shadow_damaged_hits:
            print(f"    {r['app']:10s} \"{r['query']}\" → \"{r['current_top1_text'][:25]}\" shadow_score={r['shadow_top1_score']:.2f}")

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_shadow_scoring()
