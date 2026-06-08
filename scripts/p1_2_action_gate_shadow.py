"""P1.2 Action Semantic Gate Shadow Experiment (corrected).

Shadow-only. Does NOT change src/canvas/query_engine.py.

Key corrections from v1:
1. Covers P1.1c dynamic "删除按钮" FP pattern across all apps
2. Fixes no-candidate abstain bug (shadow_top1=None → abstain)
3. Uses row["verdict"] as current verdict
4. Clarifies base_score = candidate.confidence, not query ranking score
5. Outputs gate coverage metrics
6. Marks NOT COVERED if dynamic FPs aren't reproduced

Usage:
    python scripts/p1_2_action_gate_shadow.py
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OPENCLAW_U4_LAYOUT_SHADOW"] = "1"
os.environ["OPENCLAW_URE_P0_SHADOW"] = "1"
os.environ["OPENCLAW_VTL_0_SHADOW"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.perception_service import PerceptionService
from src.windows.window_enum import WindowEnumService
from src.canvas.query_engine import CanvasQueryEngine, QueryTarget

REPORT_DIR = Path("reports/p1_2_action_gate_shadow")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Action Words (layered) ──────────────────────────────────────

CORE_ACTION_WORDS = {
    "删除", "关闭", "发送", "播放", "刷新", "保存", "打开",
    "后退", "前进", "下载", "启动", "停止",
    "delete", "close", "send", "play", "refresh", "save", "open",
    "back", "forward", "download", "start", "stop",
}

HIGH_RISK_ACTION_WORDS = {
    "支付", "付款", "提交", "断开", "退出",
    "pay", "submit", "disconnect", "quit", "exit",
}

DEFERRED_BUSINESS_WORDS = {
    "分享", "收藏", "评论", "购买",
    "share", "favorite", "comment", "buy",
}

GATE_ACTION_WORDS = CORE_ACTION_WORDS | HIGH_RISK_ACTION_WORDS

ACTION_SYNONYMS = {
    "删除": {"delete", "remove", "destructive"},
    "delete": {"删除", "remove", "destructive"},
    "关闭": {"close", "quit", "exit"},
    "close": {"关闭", "quit", "exit"},
    "发送": {"send", "submit"},
    "send": {"发送", "submit"},
    "播放": {"play", "start"},
    "play": {"播放", "start"},
    "搜索": {"search", "find"},
    "search": {"搜索", "find"},
    "打开": {"open"},
    "open": {"打开"},
}


def contains_action_word(text, action_words):
    """Check if text contains an action word with word boundary matching."""
    if not text:
        return None
    text_lower = text.lower().strip()
    for word in action_words:
        if len(word) >= 4:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                return word
        else:
            if word in text_lower:
                return word
    return None


def extract_action_word(query):
    """Extract action word from query using Core + High-risk words."""
    return contains_action_word(query, GATE_ACTION_WORDS)


def collect_action_evidence(candidate, query_action):
    """Collect action semantic evidence from candidate fields."""
    evidence = []
    if not query_action:
        return evidence

    action_set = {query_action}
    if query_action in ACTION_SYNONYMS:
        action_set.update(ACTION_SYNONYMS[query_action])

    # 1. candidate.text
    text = (candidate.text or "").strip()
    if text and text.lower() not in ("none", ""):
        word = contains_action_word(text, action_set)
        if word:
            evidence.append(("text_match", word))

    # 2. candidate.name
    name = (candidate.name or "").strip()
    if name and name.lower() not in ("none", "") and name != text:
        word = contains_action_word(name, action_set)
        if word:
            evidence.append(("name_match", word))

    # 3. attributes.automation_id
    attrs = getattr(candidate, "attributes", {}) or {}
    auto_id = (attrs.get("automation_id", "") or "").lower()
    if auto_id:
        word = contains_action_word(auto_id, action_set)
        if word:
            evidence.append(("automation_id_match", word))

    # 4. attributes.ocr_text
    ocr_text = (attrs.get("ocr_text", "") or "").strip()
    if ocr_text and ocr_text != text:
        word = contains_action_word(ocr_text, action_set)
        if word:
            evidence.append(("ocr_text_match", word))

    return evidence


def candidate_has_text(candidate):
    """Check if candidate has meaningful text content."""
    text = (candidate.text or "").strip()
    name = (candidate.name or "").strip()
    if text and text.lower() not in ("none", ""):
        return True
    if name and name.lower() not in ("none", "") and name != text:
        return True
    attrs = getattr(candidate, "attributes", {}) or {}
    ocr_text = (attrs.get("ocr_text", "") or "").strip()
    if ocr_text:
        return True
    return False


ACTIONABLE_UIA_ROLES = {
    "SemanticRole.BUTTON", "SemanticRole.SEARCH_INPUT",
    "SemanticRole.MENU_ITEM", "SemanticRole.TOOLBAR",
    "SemanticRole.TAB", "SemanticRole.CHECKBOX",
    "SemanticRole.LINK", "SemanticRole.DROPDOWN",
    "SemanticRole.SEND_BUTTON",
}
ACTIONABLE_CONTROL_TYPES = {
    "button", "hyperlink", "menuitem", "tabitem",
    "checkbox", "radiobutton", "togglebutton",
    "edit", "combobox", "spinner", "slider", "buttoncontrol",
}


def candidate_is_actionable(candidate):
    """Check if candidate has actionable role or control type."""
    full_role = str(candidate.semantic_role)
    ctrl_type = (candidate.control_type or "").lower()
    return full_role in ACTIONABLE_UIA_ROLES or ctrl_type in ACTIONABLE_CONTROL_TYPES


def apply_action_gate(score, query, candidate, expected_role):
    """Apply action semantic gate. Returns (new_score, reasons, extra)."""
    reasons = []
    extra = {}

    # Gate only applies to button_action
    if expected_role != "button_action":
        return score, reasons, extra

    query_action = extract_action_word(query)
    if not query_action:
        return score, reasons, extra

    evidence = collect_action_evidence(candidate, query_action)
    is_actionable = candidate_is_actionable(candidate)
    has_text = candidate_has_text(candidate)

    if is_actionable and evidence:
        score = min(1.0, score * 1.5)
        reasons.append("actionable_with_action_match")
        extra["action_evidence"] = [e[0] for e in evidence]
        extra["matched_word"] = evidence[0][1]

    elif is_actionable and not evidence and has_text:
        score *= 0.15
        reasons.append("action_mismatch_abstain")
        extra["action_evidence"] = []

    elif is_actionable and not evidence and not has_text:
        # Icon-only: no demotion, mark as needs_semantic_hint
        reasons.append("icon_only_needs_semantic_hint")
        extra["action_evidence"] = []

    return score, reasons, extra


def compute_shadow_score(candidate, query_text, expected_role):
    """Compute shadow score with action gate.

    NOTE: base_score = candidate.confidence (not query engine ranking score).
    This is a candidate-level confidence experiment, not a query scoring experiment.
    """
    base_score = candidate.confidence
    gated_score, gate_reasons, gate_extra = apply_action_gate(
        base_score, query_text, candidate, expected_role,
    )
    return {
        "shadow_score": round(gated_score, 3),
        "base_score": round(base_score, 3),
        "gate_reasons": gate_reasons,
        "action_evidence": gate_extra.get("action_evidence", []),
        "matched_word": gate_extra.get("matched_word", None),
        "expected_role": expected_role,
    }


def parse_expected_role(query_text):
    q = query_text.lower()
    if any(kw in q for kw in ["搜索框", "搜索输入", "search box", "search input", "输入框"]):
        return "search_input"
    if q in ["搜索"]:
        return "search_input"
    if any(kw in q for kw in ["按钮", "button", "点击", "执行",
                               "发送按钮", "播放按钮", "删除按钮", "关闭按钮",
                               "刷新按钮", "后退按钮", "保存按钮", "打开按钮"]):
        return "button_action"
    if any(kw in q for kw in ["侧栏", "导航", "导航项", "标签页", "tab", "sidebar", "nav"]):
        return "navigation_item"
    if any(kw in q for kw in ["菜单", "menu", "菜单项"]):
        return "menu_item"
    if any(kw in q for kw in ["文本", "文件名", "歌曲名", "联系人", "消息", "通知"]):
        return "text_or_item"
    return "unknown"


EXPECTED_ROLE_OVERRIDES = {
    ("FlClash", "代理"): "navigation_item",
    ("FlClash", "规则"): "navigation_item",
    ("FlClash", "日志"): "navigation_item",
    ("Notepad", "文件"): "menu_item",
    ("Notepad", "编辑"): "menu_item",
    ("Notepad", "查看"): "menu_item",
    ("Notepad", "保存"): "menu_item",
    ("Notepad", "帮助"): "menu_item",
}


# ── Query tests (P1.1c dynamic set + "删除按钮" across all apps) ──

QUERY_TESTS = [
    # Notepad
    ("Notepad", "文件", "target", "menu"),
    ("Notepad", "编辑", "target", "menu"),
    ("Notepad", "查看", "target", "menu"),
    ("Notepad", "搜索框", "abstain", "no_search_box"),
    ("Notepad", "发送按钮", "abstain", "no_send_button"),
    ("Notepad", "保存", "ambiguous", "maybe_in_menu"),
    ("Notepad", "帮助", "ambiguous", "maybe_in_menu"),
    ("Notepad", "格式", "ambiguous", "maybe_in_menu"),
    ("Notepad", "关闭", "target", "close_tab_button"),
    ("Notepad", "工具", "ambiguous", "maybe_in_menu"),
    ("Notepad", "删除按钮", "abstain", "no_delete_button"),
    # Chrome
    ("Chrome", "搜索", "target", "search_bar"),
    ("Chrome", "发送按钮", "ambiguous", "maybe_icon_only"),
    ("Chrome", "菜单", "ambiguous", "maybe_in_toolbar"),
    ("Chrome", "分享", "ambiguous", "maybe_in_toolbar"),
    ("Chrome", "删除按钮", "abstain", "no_delete_button"),
    ("Chrome", "刷新", "ambiguous", "maybe_icon_only"),
    ("Chrome", "后退", "ambiguous", "maybe_icon_only"),
    ("Chrome", "新标签", "abstain", "no_new_tab_text"),
    ("Chrome", "下载", "ambiguous", "maybe_in_menu"),
    ("Chrome", "设置", "ambiguous", "maybe_in_menu"),
    ("Chrome", "个人资料", "ambiguous", "maybe_profile"),
    ("Chrome", "打开", "ambiguous", "maybe_in_menu"),
    # QQ
    ("QQ", "搜索", "target", "search_input"),
    ("QQ", "发送按钮", "target", "send_button"),
    ("QQ", "表情", "abstain", "no_emoji_button"),
    ("QQ", "播放按钮", "abstain", "no_play_button"),
    ("QQ", "图片", "abstain", "no_image_button"),
    ("QQ", "文件", "abstain", "no_file_button"),
    ("QQ", "联系人", "ambiguous", "maybe_in_sidebar"),
    ("QQ", "群聊", "ambiguous", "maybe_in_sidebar"),
    ("QQ", "视频", "abstain", "no_video_button"),
    ("QQ", "电话", "abstain", "no_phone_button"),
    ("QQ", "购买", "abstain", "message_text_fp"),
    ("QQ", "网址", "abstain", "message_text_fp"),
    ("QQ", "全体成员", "abstain", "message_text_fp"),
    ("QQ", "删除按钮", "abstain", "no_delete_button"),
    # WeChat
    ("WeChat", "搜索", "target", "search_input"),
    ("WeChat", "发送按钮", "target", "send_button"),
    ("WeChat", "表情", "abstain", "no_emoji_button"),
    ("WeChat", "播放按钮", "abstain", "no_play_button"),
    ("WeChat", "图片", "ambiguous", "maybe_in_toolbar"),
    ("WeChat", "文件", "ambiguous", "maybe_contact_name"),
    ("WeChat", "联系人", "ambiguous", "maybe_tab"),
    ("WeChat", "朋友圈", "abstain", "no_moments_button"),
    ("WeChat", "视频", "abstain", "no_video_button"),
    ("WeChat", "语音", "ambiguous", "maybe_in_toolbar"),
    ("WeChat", "服务通知", "abstain", "notification_text"),
    ("WeChat", "卡券", "abstain", "coupon_text"),
    ("WeChat", "超市", "abstain", "contact_name"),
    ("WeChat", "中国", "abstain", "message_text"),
    ("WeChat", "新消息", "abstain", "notification_text"),
    ("WeChat", "是的", "abstain", "message_text"),
    ("WeChat", "删除按钮", "abstain", "no_delete_button"),
    # NetEase
    ("NetEase", "搜索", "ambiguous", "maybe_search"),
    ("NetEase", "播放", "ambiguous", "maybe_play"),
    ("NetEase", "收藏", "ambiguous", "maybe_favorite"),
    ("NetEase", "发送按钮", "abstain", "no_send_button"),
    ("NetEase", "播放按钮", "abstain", "no_play_button_text"),
    ("NetEase", "上一首", "abstain", "no_prev_button"),
    ("NetEase", "下一首", "abstain", "no_next_button"),
    ("NetEase", "音量", "abstain", "no_volume_control"),
    ("NetEase", "歌词", "abstain", "no_lyrics_button"),
    ("NetEase", "评论", "abstain", "no_comments_button"),
    ("NetEase", "门尼", "abstain", "artist_name"),
    ("NetEase", "巴拉莱卡", "abstain", "song_name"),
    ("NetEase", "删除按钮", "abstain", "no_delete_button"),
    # VSCode
    ("VSCode", "搜索", "ambiguous", "maybe_search"),
    ("VSCode", "运行", "ambiguous", "maybe_run"),
    ("VSCode", "终端", "ambiguous", "maybe_terminal"),
    ("VSCode", "发送按钮", "abstain", "no_send_button"),
    ("VSCode", "保存", "abstain", "no_save_button"),
    ("VSCode", "设置", "abstain", "no_settings_button"),
    ("VSCode", "扩展", "ambiguous", "maybe_extensions"),
    ("VSCode", "调试", "ambiguous", "maybe_debug"),
    ("VSCode", "git", "ambiguous", "maybe_git"),
    ("VSCode", "问题", "abstain", "no_problems_button"),
    ("VSCode", "def", "abstain", "code_keyword"),
    ("VSCode", "import", "abstain", "code_keyword"),
    ("VSCode", "标题操作", "abstain", "toolbar_label"),
    ("VSCode", "删除按钮", "abstain", "no_delete_button"),
    # FlClash
    ("FlClash", "搜索", "abstain", "no_search_button"),
    ("FlClash", "启动", "abstain", "no_start_button"),
    ("FlClash", "发送按钮", "abstain", "no_send_button"),
    ("FlClash", "设置", "ambiguous", "maybe_in_menu"),
    ("FlClash", "代理", "target", "navigation_item"),
    ("FlClash", "规则", "abstain", "no_rules_button"),
    ("FlClash", "日志", "abstain", "no_logs_button"),
    ("FlClash", "连接", "ambiguous", "maybe_connection"),
    ("FlClash", "断开", "abstain", "no_disconnect_button"),
    ("FlClash", "更新", "abstain", "no_update_button"),
    ("FlClash", "删除按钮", "abstain", "no_delete_button"),
]


def run_shadow():
    """Run P1.2 action semantic gate shadow experiment."""
    svc = PerceptionService()
    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)
    engine = CanvasQueryEngine()

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

    # Run queries
    all_rows = []
    for app_name, query, expected, reason in QUERY_TESTS:
        canvas = canvases.get(app_name)
        if not canvas:
            all_rows.append({"app": app_name, "query": query, "expected": expected,
                             "reason": reason, "verdict": "skipped", "skip_reason": "no_canvas"})
            continue

        override = EXPECTED_ROLE_OVERRIDES.get((app_name, query))
        expected_role = override if override else parse_expected_role(query)

        try:
            result = engine.query(canvas, QueryTarget(natural_language=query), max_results=3, min_confidence=0.0)
            candidates = result.candidates

            # Shadow scoring with action gate
            shadow_results = []
            for c in candidates[:3]:
                sr = compute_shadow_score(c, query, expected_role)
                shadow_results.append({
                    "element_id": c.element_id,
                    "text": (c.text or "")[:40],
                    "name": (c.name or "")[:30],
                    "role": str(c.semantic_role).split(".")[-1],
                    "control_type": c.control_type or "",
                    "provider_sources": c.provider_sources,
                    "current_score": round(c.confidence, 3),
                    **sr,
                })
            shadow_results.sort(key=lambda r: -r["shadow_score"])

            top1 = candidates[0] if candidates else None
            shadow_top1 = shadow_results[0] if shadow_results else None

            # Current verdict (based on query engine output)
            has_results = len(candidates) > 0
            if expected == "target":
                verdict = "hit" if has_results else "miss"
            elif expected == "abstain":
                verdict = "fp" if has_results else "correct_abstain"
            else:
                verdict = "ambiguous"

            # Shadow verdict (based on shadow scoring)
            # Fix: if no candidates, shadow should abstain
            if shadow_top1 is None:
                should_abstain = True
                abstain_reason = "no_candidates"
            elif shadow_top1["shadow_score"] < 0.3:
                should_abstain = True
                abstain_reason = "low_shadow_score"
            elif "action_mismatch_abstain" in shadow_top1.get("gate_reasons", []):
                should_abstain = True
                abstain_reason = "action_mismatch_abstain"
            elif "icon_only_needs_semantic_hint" in shadow_top1.get("gate_reasons", []):
                should_abstain = True
                abstain_reason = "icon_only_needs_semantic_hint"
            else:
                should_abstain = False
                abstain_reason = None

            all_rows.append({
                "app": app_name,
                "query": query,
                "expected": expected,
                "reason": reason,
                "expected_role": expected_role,
                "expected_role_override": override,
                "verdict": verdict,
                "top1_text": (top1.text or "")[:40] if top1 else "",
                "top1_role": str(top1.semantic_role).split(".")[-1] if top1 else "",
                "top1_ctrl": top1.control_type or "" if top1 else "",
                "shadow_score": shadow_top1["shadow_score"] if shadow_top1 else None,
                "shadow_base_score": shadow_top1["base_score"] if shadow_top1 else None,
                "shadow_verdict": "shadow_abstain" if should_abstain else "shadow_hit",
                "shadow_gate_reasons": shadow_top1.get("gate_reasons", []) if shadow_top1 else [],
                "shadow_action_evidence": shadow_top1.get("action_evidence", []) if shadow_top1 else [],
                "shadow_matched_word": shadow_top1.get("matched_word") if shadow_top1 else None,
                "should_abstain": should_abstain,
                "abstain_reason": abstain_reason,
                "total_candidates": len(candidates),
            })
        except Exception as e:
            all_rows.append({"app": app_name, "query": query, "expected": expected,
                             "reason": reason, "verdict": "error", "error": str(e)})

    # Save
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "P1.2-action-semantic-gate-shadow-v2",
        "total_queries": len(all_rows),
        "gate_config": {
            "core_action_words": sorted(CORE_ACTION_WORDS),
            "high_risk_action_words": sorted(HIGH_RISK_ACTION_WORDS),
            "deferred_business_words": sorted(DEFERRED_BUSINESS_WORDS),
            "gate_uses": "core + high_risk only",
            "deferred_status": "registered only, not used for gate",
            "base_score": "candidate.confidence (not query engine ranking score)",
        },
        "rows": all_rows,
    }
    with open(REPORT_DIR / "p1_2_shadow.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # ── Summary ──
    valid = [r for r in all_rows if r.get("verdict") not in ("error", "skipped")]
    hits = [r for r in valid if r.get("verdict") == "hit"]
    fps = [r for r in valid if r.get("verdict") == "fp"]
    abstains = [r for r in valid if r.get("verdict") == "correct_abstain"]
    misses = [r for r in valid if r.get("verdict") == "miss"]
    amb = [r for r in valid if r.get("verdict") == "ambiguous"]
    skipped = [r for r in all_rows if r.get("verdict") in ("error", "skipped")]

    shadow_fixed = [r for r in fps if r.get("should_abstain")]
    shadow_remaining = [r for r in fps if not r.get("should_abstain")]
    shadow_damaged = [r for r in hits if r.get("should_abstain")]
    shadow_broken = [r for r in abstains if not r.get("should_abstain")]

    # Gate coverage metrics
    button_action_rows = [r for r in valid if r.get("expected_role") == "button_action"]
    ba_with_action = [r for r in button_action_rows if extract_action_word(r.get("query", ""))]
    ba_action_match = [r for r in valid if "actionable_with_action_match" in r.get("shadow_gate_reasons", [])]
    ba_action_mismatch = [r for r in valid if "action_mismatch_abstain" in r.get("shadow_gate_reasons", [])]
    ba_icon_only = [r for r in valid if "icon_only_needs_semantic_hint" in r.get("shadow_gate_reasons", [])]

    # "删除按钮" FP analysis
    delete_fps = [r for r in fps if r.get("query") == "删除按钮"]
    delete_fixed = [r for r in delete_fps if r.get("should_abstain")]
    delete_remaining = [r for r in delete_fps if not r.get("should_abstain")]

    print(f"\n{'='*60}")
    print("  P1.2 Action Semantic Gate Shadow Summary (v2)")
    print(f"{'='*60}")
    print(f"  Base score: candidate.confidence (not query ranking)")
    print(f"  Gate: Core + High-risk only; Deferred registered only")
    print(f"  ---")
    print(f"  Total queries: {len(all_rows)}")
    print(f"  Processed: {len(valid)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  ---")
    print(f"  Hits: {len(hits)}")
    print(f"  FP (row-level): {len(fps)}")
    print(f"  Correct abstains: {len(abstains)}")
    print(f"  Misses: {len(misses)}")
    print(f"  Ambiguous: {len(amb)}")
    print(f"  Sum: {len(hits)+len(fps)+len(abstains)+len(misses)+len(amb)}")
    print(f"  ---")
    print(f"  Shadow fixed FP: {len(shadow_fixed)}")
    print(f"  Shadow remaining FP: {len(shadow_remaining)}")
    print(f"  Shadow hit damage: {len(shadow_damaged)}")
    print(f"  Shadow broken abstain: {len(shadow_broken)}")
    print(f"  ---")
    print(f"  Gate coverage:")
    print(f"    button_action rows: {len(button_action_rows)}")
    print(f"    with query_action: {len(ba_with_action)}")
    print(f"    action_match: {len(ba_action_match)}")
    print(f"    action_mismatch_abstain: {len(ba_action_mismatch)}")
    print(f"    icon_only_needs_semantic_hint: {len(ba_icon_only)}")
    print(f"  ---")
    print(f"  '删除按钮' FP: {len(delete_fps)} total, {len(delete_fixed)} fixed, {len(delete_remaining)} remaining")
    if len(delete_fps) < 8:
        print(f"  WARNING: '删除按钮' FP count {len(delete_fps)} < 8, dynamic pattern NOT FULLY COVERED")
    print(f"  ---")

    # Fixed FP details
    if shadow_fixed:
        print(f"\n  Shadow fixed FP ({len(shadow_fixed)}):")
        for r in shadow_fixed[:10]:
            print(f"    {r['app']:10s} \"{r['query']}\" shadow={r.get('shadow_score',0):.2f} reason={r.get('abstain_reason','?')}")

    # Remaining FP details
    if shadow_remaining:
        print(f"\n  Shadow remaining FP ({len(shadow_remaining)}):")
        for r in shadow_remaining[:10]:
            print(f"    {r['app']:10s} \"{r['query']}\" → \"{r.get('top1_text','')[:20]}\" shadow={r.get('shadow_score',0):.2f}")

    # Damaged hits
    if shadow_damaged:
        print(f"\n  Shadow damaged hits ({len(shadow_damaged)}):")
        for r in shadow_damaged:
            print(f"    {r['app']:10s} \"{r['query']}\" shadow={r.get('shadow_score',0):.2f}")

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_shadow()
