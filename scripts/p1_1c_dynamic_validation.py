"""P1.1c Dynamic Sample Validation (corrected).

Runs shadow scoring from P1.1b.5 on dynamic page states.
Does NOT change main pipeline, does NOT adjust scoring weights.

Usage:
    python scripts/p1_1c_dynamic_validation.py
"""

import json
import os
import sys
import time
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

REPORT_DIR = Path("reports/p1_1c_dynamic_validation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import shadow scoring from P1.1b.5 ──────────────────────────
# Reuse the same scoring logic to ensure consistency

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
CONTENT_TEXT_CONTROLS = {"textcontrol", "groupcontrol", "documentcontrol"}
CONTENT_FLOW_REGIONS = {"messages", "message_stream", "document_body", "content_stream",
                        "editor_region", "table_region", "list_content", "message_area"}
STRUCTURAL_REGIONS = {"toolbar", "sidebar", "filter_bar", "top_bar", "control_strip",
                      "input_area", "menu_bar", "status_bar", "composer", "bottom_bar"}


def classify_region_kind(region_id):
    rid = (region_id or "").lower()
    for kw in STRUCTURAL_REGIONS:
        if kw in rid:
            return "structural"
    for kw in CONTENT_FLOW_REGIONS:
        if kw in rid:
            return "content_flow"
    return "unknown"


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


def compute_shadow_score(candidate, query_text, expected_type, expected_role_override=None):
    """Shadow scoring from P1.1b.5-cleanup."""
    score = candidate.confidence
    demotion_reasons = []
    positive_reasons = []

    if expected_role_override:
        expected_role = expected_role_override
    else:
        expected_role = parse_expected_role(query_text)

    role = str(candidate.semantic_role).split(".")[-1]
    full_role = str(candidate.semantic_role)
    ctrl_type = (candidate.control_type or "").lower()
    region_id = (candidate.region_id or "")
    region_kind = classify_region_kind(region_id)
    provider_sources = candidate.provider_sources or []
    text = (candidate.text or "").lower()
    text_match = query_text.lower() in text or text in query_text.lower()
    role_label = (getattr(candidate, "role_label", None) or "").lower()
    semantic_tags = [t.lower() for t in (getattr(candidate, "semantic_tags", None) or [])]
    risk_tags = [t.lower() for t in (getattr(candidate, "risk_tags", None) or [])]
    has_uia = "uia" in provider_sources
    has_ocr = "ocr" in provider_sources
    has_vision = "omni" in provider_sources or "vision" in provider_sources
    is_actionable_role = full_role in ACTIONABLE_UIA_ROLES
    is_actionable_ctrl = ctrl_type in ACTIONABLE_CONTROL_TYPES
    is_content_text_ctrl = ctrl_type in CONTENT_TEXT_CONTROLS

    # Container demotion
    container_roles = {"CONTAINER", "LAYOUT", "UNKNOWN"}
    container_controls = {"windowcontrol", "panecontrol", "groupcontrol", "documentcontrol", "customcontrol"}
    is_container = role in container_roles or ctrl_type in container_controls
    is_large_container = ctrl_type in {"windowcontrol", "panecontrol", "documentcontrol"}

    if is_container:
        if expected_role in ("button_action", "search_input"):
            score *= 0.2
            demotion_reasons.append("container_role_for_action_query")
        elif is_large_container:
            score *= 0.2
            demotion_reasons.append("large_container_always_demote")
        elif text_match and expected_role in ("navigation_item", "text_or_item", "menu_item"):
            pass
        elif expected_role == "unknown" and text_match:
            pass
        else:
            score *= 0.3
            demotion_reasons.append("container_role_general_demote")

    # Expected role compatibility
    if expected_role == "search_input":
        if "SEARCH_INPUT" in full_role:
            score = min(1.0, score * 1.5)
            positive_reasons.append("role_matches_search_input")
        elif ctrl_type == "editcontrol":
            score = min(1.0, score * 1.3)
            positive_reasons.append("ctrl_matches_editcontrol")
        elif "filter_bar" in region_id.lower() or "input_area" in region_id.lower():
            score = min(1.0, score * 1.2)
            positive_reasons.append("region_matches_filter_bar")
        else:
            score *= 0.15
            demotion_reasons.append("search_input_hard_gate_fail")

    elif expected_role == "button_action":
        action_words = {"删除", "关闭", "刷新", "后退", "保存", "打开",
                        "发送", "播放", "启动", "断开", "更新", "下载",
                        "delete", "close", "refresh", "back", "save", "open",
                        "send", "play", "start", "stop", "update", "download"}
        query_action = None
        for aw in action_words:
            if aw in query_text.lower():
                query_action = aw
                break

        candidate_combined = f"{text} {role_label} {' '.join(semantic_tags)} {' '.join(risk_tags)}"
        has_action_evidence = False
        if query_action:
            if query_action in candidate_combined:
                has_action_evidence = True
            destructive_map = {
                "删除": ["delete", "remove", "destructive"],
                "delete": ["删除", "remove", "destructive"],
                "关闭": ["close", "quit", "exit"],
                "close": ["关闭", "quit", "exit"],
            }
            if not has_action_evidence and query_action in destructive_map:
                for syn in destructive_map[query_action]:
                    if syn in candidate_combined:
                        has_action_evidence = True
                        break

        if is_actionable_role and has_action_evidence:
            score = min(1.0, score * 1.5)
            positive_reasons.append("actionable_role_with_action_evidence")
        elif is_actionable_role and not has_action_evidence and query_action:
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
            score = min(1.0, score * 1.2)
            positive_reasons.append("candidate_looks_navigation_by_evidence")
        elif text_match and region_kind == "structural":
            positive_reasons.append("text_match_structural_region")

    # Feature E: content-text abstain
    if role == "TEXT" and is_content_text_ctrl:
        if region_kind == "content_flow" and expected_role not in (
            "text_or_item", "menu_item", "navigation_item", "search_input",
        ):
            score *= 0.15
            demotion_reasons.append("content_text_in_flow_abstain")
        elif region_kind == "unknown" and expected_role not in (
            "text_or_item", "menu_item", "navigation_item", "search_input",
        ):
            if not is_actionable_role and not is_actionable_ctrl:
                score *= 0.4
                demotion_reasons.append("text_unknown_region_no_actionable")

    # Media indicators
    media_indicators = ["[图片]", "[语音]", "[视频]", "[文件]"]
    if any(ind in text for ind in media_indicators):
        score *= 0.3
        demotion_reasons.append("media_content_indicator")

    score = max(0.0, min(1.0, score))
    return score, demotion_reasons, positive_reasons, expected_role


def compute_abstain(results, query_text):
    if not results:
        return True, "no_candidates"
    top_score = max(r["shadow_score"] for r in results)
    if top_score < 0.3:
        return True, "low_shadow_score"
    top = max(results, key=lambda r: r["shadow_score"])
    if len(top["demotion_reasons"]) >= 3:
        return True, "strong_multi_demotion"
    return False, None


# ── Dynamic negative sample extraction ──────────────────────────

# Excluded roles/controls for negative samples
EXCLUDE_ROLES = {"search_input", "menu_item", "toolbar", "tab", "button", "link",
                 "dropdown", "send_button", "checkbox", "radiobutton"}
EXCLUDE_CONTROLS = {"editcontrol", "buttoncontrol", "menuitemcontrol", "tabitemcontrol",
                     "hyperlink", "checkbox", "radiobutton", "togglebutton"}

# Dirty patterns: window/class names, time/status, app names, navigation items
DIRTY_PATTERNS = [
    # Window/class names
    "rendersubwindow", "mmui", "uiautomation", "panecontrol",
    # Time/status/delay
    ":", "ms", "am", "pm", "上午", "下午",
    # App names / window titles
    "网易云音乐", "google chrome", "visual studio code", "flclash",
    "qq", "微信", "wechat",
    # Navigation/activity bar items
    "资源管理器", "仪表盘", "第 1 个标签", "第 2 个标签", "共",
    "ctrl+shift", "alt+", "ctrl+",
    # Status/system text
    "loading", "error", "warning", "success", "failed",
]


def extract_dynamic_negatives(canvas, max_count=3):
    """Extract content text for dynamic negative samples.

    Strict filtering:
    - Only TEXT role content (not actionable/structural)
    - Exclude window/class names, time/status, app names, navigation items
    - Exclude structural regions
    - Deduplicate
    """
    candidates = []
    rejected = []
    seen_texts = set()

    for c in canvas.elements:
        role = str(c.semantic_role).split(".")[-1].lower()
        ctrl = (c.control_type or "").lower()
        text = (c.text or "").strip()
        region_id = (c.region_id or "").lower()

        # Exclude actionable/structural roles
        if role in EXCLUDE_ROLES:
            rejected.append((text, "actionable_role"))
            continue
        if ctrl in EXCLUDE_CONTROLS:
            rejected.append((text, "structural_control"))
            continue

        # Exclude structural regions
        region_kind = classify_region_kind(c.region_id or "")
        if region_kind == "structural":
            rejected.append((text, "structural_region"))
            continue

        # Exclude short/long text
        if len(text) < 3 or len(text) > 25:
            rejected.append((text, "length_filter"))
            continue

        # Exclude dirty patterns
        text_lower = text.lower()
        if any(pat in text_lower for pat in DIRTY_PATTERNS):
            rejected.append((text, "dirty_pattern"))
            continue

        # Exclude if text looks like a UI label (has parentheses, brackets, shortcuts)
        if "(" in text or ")" in text or "[" in text or "]" in text:
            rejected.append((text, "ui_label"))
            continue

        # Deduplicate
        if text in seen_texts:
            rejected.append((text, "duplicate"))
            continue
        seen_texts.add(text)

        candidates.append(text)

    return candidates[:max_count], rejected


# ── Safe scroll ─────────────────────────────────────────────────

def safe_scroll(hwnd, direction="down", steps=3):
    """Safe read-only scroll."""
    import ctypes
    user32 = ctypes.windll.user32
    for _ in range(steps):
        if direction == "down":
            user32.mouse_event(0x0800, 0, 0, -120, 0)
        else:
            user32.mouse_event(0x0800, 0, 0, 120, 0)
        time.sleep(0.3)


# ── Send button check ───────────────────────────────────────────

def has_send_button(canvas):
    for c in canvas.elements:
        role = str(c.semantic_role).upper()
        ctrl = (c.control_type or "").lower()
        text = (c.text or "").lower()
        if "SEND" in role or "发送" in text:
            if "button" in ctrl or "send" in role:
                return True
    return False


# ── Main ────────────────────────────────────────────────────────

def run_dynamic_validation():
    svc = PerceptionService()
    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)
    engine = CanvasQueryEngine()

    # App configs: (name, process, page_states)
    app_configs = [
        ("Chrome", "chrome.exe", ["baseline", "scroll_down"]),
        ("QQ", "qq.exe", ["baseline", "scroll_down"]),
        ("WeChat", "weixin.exe", ["baseline", "scroll_down"]),
        ("NetEase", "cloudmusic.exe", ["baseline", "scroll_down"]),
        ("VSCode", "code.exe", ["baseline", "scroll_down"]),
        ("FlClash", "flclash.exe", ["baseline", "scroll_down"]),
    ]

    all_rows = []
    all_rejected = []

    for app_name, process, states in app_configs:
        win = next(
            (w for w in windows if process.lower() in (w.process_name or "").lower()),
            None,
        )
        if not win:
            continue

        for state_idx, page_state in enumerate(states):
            # Scroll for changed states
            if state_idx > 0:
                safe_scroll(win.hwnd, "down", 3)
                time.sleep(1)

            # Observe
            try:
                zone_page = svc.analyze(hwnd=win.hwnd, allow_vlm=False)
                canvas = svc.create_page_snapshot(zone_page, process_name=process)
            except Exception as e:
                all_rows.append({
                    "app": app_name, "query": "-", "page_state": page_state,
                    "verdict": "skipped", "reason": f"observe_error: {e}",
                })
                continue

            # Build query set for this state
            queries = []

            # Static queries
            queries.append(("搜索", "target" if app_name in ("QQ", "WeChat") else "ambiguous", "search_input"))
            queries.append(("删除按钮", "abstain", "button_action"))

            # Send button check
            if has_send_button(canvas):
                queries.append(("发送按钮", "target", "button_action"))
            else:
                queries.append(("发送按钮", "ambiguous", "button_action"))

            # Dynamic negatives from content text (strict filtering)
            neg_texts, rejected = extract_dynamic_negatives(canvas, max_count=3)
            all_rejected.extend([(app_name, page_state, text, reason) for text, reason in rejected])
            for text in neg_texts:
                queries.append((f"{text}按钮", "abstain", "dynamic_content_fp"))

            # Override for FlClash navigation
            if app_name == "FlClash":
                queries.append(("代理", "target", "navigation_item"))

            # Run queries
            for query_text, expected, reason in queries:
                # Get override
                override = None
                if app_name == "FlClash" and query_text == "代理":
                    override = "navigation_item"

                try:
                    result = engine.query(canvas, QueryTarget(natural_language=query_text), max_results=3, min_confidence=0.0)
                    candidates = result.candidates

                    # Shadow scoring
                    shadow_results = []
                    for c in candidates[:3]:
                        sc, dem, pos, er = compute_shadow_score(c, query_text, expected, expected_role_override=override)
                        shadow_results.append({
                            "element_id": c.element_id,
                            "text": (c.text or "")[:40],
                            "role": str(c.semantic_role).split(".")[-1],
                            "control_type": c.control_type or "",
                            "provider_sources": c.provider_sources,
                            "current_score": round(c.confidence, 3),
                            "shadow_score": round(sc, 3),
                            "demotion_reasons": dem,
                            "positive_reasons": pos,
                            "expected_role": er,
                        })
                    shadow_results.sort(key=lambda r: -r["shadow_score"])

                    should_abstain, abstain_reason = compute_abstain(shadow_results, query_text)

                    top1 = candidates[0] if candidates else None
                    shadow_top1 = shadow_results[0] if shadow_results else None

                    has_results = len(candidates) > 0
                    if expected == "target":
                        verdict = "hit" if has_results else "miss"
                    elif expected == "abstain":
                        verdict = "fp" if has_results else "correct_abstain"
                    else:
                        verdict = "ambiguous"

                    all_rows.append({
                        "app": app_name,
                        "query": query_text,
                        "expected": expected,
                        "reason": reason,
                        "page_state": page_state,
                        "expected_role_override": override,
                        "current_verdict": verdict,
                        "top1_text": (top1.text or "")[:40] if top1 else "",
                        "top1_role": str(top1.semantic_role).split(".")[-1] if top1 else "",
                        "top1_ctrl": top1.control_type or "" if top1 else "",
                        "shadow_score": shadow_top1["shadow_score"] if shadow_top1 else 0,
                        "shadow_verdict": "shadow_abstain" if should_abstain else "shadow_hit",
                        "shadow_demotion": shadow_top1["demotion_reasons"] if shadow_top1 else [],
                        "shadow_positive": shadow_top1["positive_reasons"] if shadow_top1 else [],
                        "expected_role": shadow_top1["expected_role"] if shadow_top1 else "unknown",
                        "should_abstain": should_abstain,
                        "total_candidates": len(candidates),
                    })
                except Exception as e:
                    all_rows.append({
                        "app": app_name, "query": query_text, "expected": expected,
                        "reason": reason, "page_state": page_state,
                        "verdict": "error", "error": str(e),
                    })

    # Save (will be populated after summary section below)

    # ── Summary ──
    valid = [r for r in all_rows if r.get("current_verdict") not in ("error", "skipped")]
    hits = [r for r in valid if r.get("current_verdict") == "hit"]
    fps = [r for r in valid if r.get("current_verdict") == "fp"]
    abstains = [r for r in valid if r.get("current_verdict") == "correct_abstain"]
    misses = [r for r in valid if r.get("current_verdict") == "miss"]
    amb = [r for r in valid if r.get("current_verdict") == "ambiguous"]
    skipped = [r for r in all_rows if r.get("current_verdict") in ("error", "skipped")]

    shadow_fixed = [r for r in fps if r.get("should_abstain")]
    shadow_remaining = [r for r in fps if not r.get("should_abstain")]
    shadow_damaged = [r for r in hits if r.get("should_abstain")]
    shadow_broken = [r for r in abstains if not r.get("should_abstain")]

    # Pattern-level FP
    fp_patterns = Counter()
    for r in fps:
        reason = r.get("reason", "unknown")
        fp_patterns[reason] += 1

    # Rejected negative counts
    rejected_counts = Counter()
    for _, _, _, reason in all_rejected:
        rejected_counts[reason] += 1

    # Classify FP by type
    title_window_fp = [r for r in fps if "标题" in r.get("query", "") or "工作区" in r.get("query", "") or "窗口" in r.get("query", "")]
    content_text_fp = [r for r in fps if r.get("reason") == "dynamic_content_fp" and r not in title_window_fp]
    button_action_fp = [r for r in fps if r.get("reason") == "button_action"]

    print(f"\n{'='*60}")
    print("  P1.1c Dynamic Validation Summary (corrected v2)")
    print(f"{'='*60}")
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
    print(f"  FP classification:")
    print(f"    button_action_fp: {len(button_action_fp)} (action_mismatch, cross-app stable)")
    print(f"    title_window_fp: {len(title_window_fp)} (title/workspace text)")
    print(f"    valid_content_text_fp: {len(content_text_fp)}")
    print(f"  content_text_in_flow persistence: NOT COVERED (insufficient dynamic evidence)")
    print(f"  ---")
    print(f"  Dynamic negative stats:")
    print(f"    clean_dynamic_negative: {len([r for r in valid if r.get('reason') == 'dynamic_content_fp'])}")
    print(f"    rejected_dynamic_negative: {len(all_rejected)}")
    for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
        print(f"      {reason}: {count}")

    if shadow_fixed:
        print(f"\n  Shadow fixed FP:")
        for r in shadow_fixed:
            print(f"    {r['app']:10s} [{r['page_state']:12s}] \"{r['query']}\" shadow={r['shadow_score']:.2f} demotion={r['shadow_demotion']}")

    if shadow_remaining:
        print(f"\n  Shadow remaining FP:")
        for r in shadow_remaining:
            print(f"    {r['app']:10s} [{r['page_state']:12s}] \"{r['query']}\" → \"{r['top1_text'][:25]}\" shadow={r['shadow_score']:.2f}")

    if shadow_damaged:
        print(f"\n  Shadow damaged hits:")
        for r in shadow_damaged:
            print(f"    {r['app']:10s} [{r['page_state']:12s}] \"{r['query']}\" shadow={r['shadow_score']:.2f}")

    # Save full report to JSON
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "P1.1c-dynamic-validation-corrected-v2",
        "total_queries": len(all_rows),
        "processed": len(valid),
        "skipped": len(skipped),
        "metrics": {
            "hits": len(hits),
            "fp_row_level": len(fps),
            "correct_abstains": len(abstains),
            "misses": len(misses),
            "ambiguous": len(amb),
        },
        "shadow_metrics": {
            "shadow_fixed_fp": len(shadow_fixed),
            "shadow_remaining_fp": len(shadow_remaining),
            "shadow_hit_damage": len(shadow_damaged),
            "shadow_broken_abstain": len(shadow_broken),
        },
        "fp_classification": {
            "button_action_fp": len(button_action_fp),
            "title_window_fp": len(title_window_fp),
            "valid_content_text_fp": len(content_text_fp),
        },
        "dynamic_negative_stats": {
            "clean_dynamic_negative": len([r for r in valid if r.get("reason") == "dynamic_content_fp"]),
            "rejected_dynamic_negative": len(all_rejected),
            "rejected_by_reason": dict(rejected_counts),
            "dirty_examples": [
                {"app": app, "state": st, "text": text[:40], "reason": reason}
                for app, st, text, reason in all_rejected[:20]
                if reason in ("dirty_pattern", "ui_label")
            ],
        },
        "content_text_in_flow_persistence": {
            "status": "not_covered",
            "reason": "insufficient_dynamic_evidence",
            "valid_content_text_fp_count": len(content_text_fp),
            "note": "VSCode '无标题（工作区）' is title/workspace text, not content flow text",
        },
        "rows": all_rows,
    }
    with open(REPORT_DIR / "dynamic_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_dynamic_validation()
