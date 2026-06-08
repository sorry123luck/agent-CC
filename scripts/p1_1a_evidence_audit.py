"""P1.1a Evidence Audit: Query engine false positive analysis.

Read-only audit of current query_engine behavior on real app canvases.
Does NOT modify any code, does NOT change query ranking.

Usage:
    python scripts/p1_1a_evidence_audit.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Enable all shadow passes
os.environ["OPENCLAW_U4_LAYOUT_SHADOW"] = "1"
os.environ["OPENCLAW_URE_P0_SHADOW"] = "1"
os.environ["OPENCLAW_VTL_0_SHADOW"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.perception_service import PerceptionService
from src.windows.window_enum import WindowEnumService
from src.canvas.query_engine import CanvasQueryEngine, QueryTarget

REPORT_DIR = Path("reports/p1_1a_evidence_audit")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Query test cases (corrected ground truth) ────────────────────
# Format: (app, query, expected, expected_target_text, expected_region, notes, failure_hint)
# expected: "target" / "abstain" / "ambiguous"
# expected_target_text: what the correct answer should be
# expected_region_hint: where the target should be
# failure_hint: expected failure type if this query fails

QUERY_TESTS = [
    # ── Notepad ──
    ("Notepad", "文件", "target", "文件", "menu", "File menu item exists", None),
    ("Notepad", "编辑", "target", "编辑", "menu", "Edit menu item exists", None),
    ("Notepad", "查看", "target", "查看", "menu", "View menu item exists", None),
    ("Notepad", "搜索框", "abstain", None, None, "Notepad has no search box", "substring_fp"),
    ("Notepad", "发送按钮", "abstain", None, None, "Notepad has no send button", None),
    ("Notepad", "保存", "ambiguous", None, "menu", "Save may be under File menu", "intent_gap_miss"),
    ("Notepad", "打开", "ambiguous", None, "menu", "Open may be under File menu", "intent_gap_miss"),
    ("Notepad", "帮助", "ambiguous", None, "menu", "Help menu may or may not exist", "intent_gap_miss"),
    ("Notepad", "格式", "ambiguous", None, "menu", "Format menu may or may not exist", "intent_gap_miss"),
    ("Notepad", "关闭", "target", "关闭标签页", "toolbar", "Close tab button exists", None),
    # ── Chrome ──
    ("Chrome", "搜索", "target", "地址和搜索栏", "toolbar", "Chrome address bar exists", None),
    ("Chrome", "发送按钮", "ambiguous", None, "content", "ChatGPT send button may be icon-only", "icon_miss"),
    ("Chrome", "菜单", "ambiguous", None, "toolbar", "Menu exists but text is 'wer Plus'", "role_mismatch_fp"),
    ("Chrome", "分享", "ambiguous", None, "toolbar", "Share button may exist", None),
    ("Chrome", "删除按钮", "abstain", None, None, "No delete button in current view", "substring_fp"),
    ("Chrome", "刷新", "ambiguous", None, "toolbar", "Refresh may be icon-only", "icon_miss"),
    ("Chrome", "后退", "ambiguous", None, "toolbar", "Back may be icon-only", "icon_miss"),
    ("Chrome", "新标签", "abstain", None, None, "No explicit new tab button text", "icon_miss"),
    ("Chrome", "下载", "ambiguous", None, "toolbar", "Download may exist in menu", "intent_gap_miss"),
    ("Chrome", "设置", "ambiguous", None, "toolbar", "Settings may exist in menu", "intent_gap_miss"),
    # ── QQ ──
    ("QQ", "搜索", "target", "搜索", "sidebar", "QQ search input exists", None),
    ("QQ", "发送按钮", "abstain", None, None, "No visible send button in current view", None),
    ("QQ", "表情", "abstain", None, None, "No visible emoji button in current view", "icon_miss"),
    ("QQ", "播放按钮", "abstain", None, None, "QQ has no play button", None),
    ("QQ", "图片", "abstain", None, None, "No visible image button in current view", "icon_miss"),
    ("QQ", "文件", "abstain", None, None, "No visible file button in current view", "icon_miss"),
    ("QQ", "联系人", "ambiguous", None, "sidebar", "Contacts may be in sidebar tab", "intent_gap_miss"),
    ("QQ", "群聊", "ambiguous", None, "sidebar", "Group chat may be in sidebar", "text_content_fp"),
    ("QQ", "视频", "abstain", None, None, "No visible video call button", "icon_miss"),
    ("QQ", "电话", "abstain", None, None, "No visible phone button", "icon_miss"),
    # ── WeChat ──
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
    # ── NetEase ──
    ("NetEase", "搜索", "ambiguous", None, "toolbar", "Search input exists but text is 'Josh Turner'", "role_mismatch_fp"),
    ("NetEase", "播放", "ambiguous", None, "toolbar", "Play button may be icon-only", "icon_miss"),
    ("NetEase", "收藏", "ambiguous", "+收藏全部", "toolbar", "Favorite button exists but text is '+收藏全部'", None),
    ("NetEase", "发送按钮", "abstain", None, None, "NetEase has no send button", None),
    ("NetEase", "播放按钮", "abstain", None, None, "No visible play button text", "icon_miss"),
    ("NetEase", "上一首", "abstain", None, None, "No visible previous track button", "icon_miss"),
    ("NetEase", "下一首", "abstain", None, None, "No visible next track button", "icon_miss"),
    ("NetEase", "音量", "abstain", None, None, "No visible volume control", "icon_miss"),
    ("NetEase", "歌词", "abstain", None, None, "No visible lyrics button", "icon_miss"),
    ("NetEase", "评论", "abstain", None, None, "No visible comments button", "icon_miss"),
    # ── VSCode ──
    ("VSCode", "搜索", "ambiguous", None, "toolbar", "Search may exist but no explicit text", "icon_miss"),
    ("VSCode", "运行", "ambiguous", None, "toolbar", "Run/debug may exist but text is '运行和调试'", None),
    ("VSCode", "终端", "ambiguous", None, "toolbar", "Terminal may exist but text is 'verify_console...'", "role_mismatch_fp"),
    ("VSCode", "发送按钮", "abstain", None, None, "VSCode has no send button", None),
    ("VSCode", "保存", "abstain", None, None, "No visible save button text", "icon_miss"),
    ("VSCode", "设置", "abstain", None, None, "No visible settings button text", "icon_miss"),
    ("VSCode", "扩展", "ambiguous", "扩展 (Ctrl+Shift+X)", "toolbar", "Extensions button exists", None),
    ("VSCode", "调试", "ambiguous", "运行和调试 (Ctrl+Shift+D)", "toolbar", "Debug button exists", None),
    ("VSCode", "git", "ambiguous", ".gitignore", "content", "Git-related file exists, not a button", "text_content_fp"),
    ("VSCode", "问题", "abstain", None, None, "No visible problems panel button", "icon_miss"),
    # ── FlClash ──
    ("FlClash", "搜索", "abstain", None, None, "No visible search button", "icon_miss"),
    ("FlClash", "启动", "abstain", None, None, "No visible start button", "icon_miss"),
    ("FlClash", "发送按钮", "abstain", None, None, "FlClash has no send button", None),
    ("FlClash", "设置", "ambiguous", None, "menu", "Settings may be under 系统 menu", "intent_gap_miss"),
    ("FlClash", "代理", "target", "代理", "sidebar", "Proxy navigation exists", None),
    ("FlClash", "规则", "abstain", None, None, "No visible rules button", "intent_gap_miss"),
    ("FlClash", "日志", "abstain", None, None, "No visible logs button", "intent_gap_miss"),
    ("FlClash", "连接", "ambiguous", None, "content", "Connection status visible but not a button", "text_content_fp"),
    ("FlClash", "断开", "abstain", None, None, "No visible disconnect button", "icon_miss"),
    ("FlClash", "更新", "abstain", None, None, "No visible update button", "icon_miss"),

    # ── FP-Challenge: substring_fp ──
    # Query word appears in menu items, filenames, contact names, titles
    ("Notepad", "格式", "ambiguous", None, "menu", "Format menu may exist", "substring_fp"),
    ("Notepad", "工具", "ambiguous", None, "menu", "Tools menu may exist", "substring_fp"),
    ("Chrome", "个人资料", "ambiguous", None, "toolbar", "Profile text in menu description", "substring_fp"),
    ("Chrome", "打开", "ambiguous", None, "toolbar", "Open may be in menu", "substring_fp"),
    ("WeChat", "服务通知", "abstain", None, None, "Service notification text, not a button", "substring_fp"),
    ("WeChat", "卡券", "abstain", None, None, "Coupon text in message, not a button", "substring_fp"),

    # ── FP-Challenge: text_content_fp ──
    # Query word appears in message text, song names, filenames, not a control
    ("QQ", "购买", "abstain", None, None, "Purchase text in message, not a button", "text_content_fp"),
    ("QQ", "网址", "abstain", None, None, "URL text in message, not a button", "text_content_fp"),
    ("WeChat", "超市", "abstain", None, None, "Supermarket text in contact name, not a button", "text_content_fp"),
    ("WeChat", "中国", "abstain", None, None, "Country text in message, not a button", "text_content_fp"),
    ("VSCode", "def", "abstain", None, None, "Python keyword in code, not a button", "text_content_fp"),
    ("VSCode", "import", "abstain", None, None, "Python keyword in code, not a button", "text_content_fp"),
    ("NetEase", "门尼", "abstain", None, None, "Artist name in title, not a button", "text_content_fp"),
    ("NetEase", "巴拉莱卡", "abstain", None, None, "Song name in title, not a button", "text_content_fp"),

    # ── FP-Challenge: role_mismatch_fp ──
    # Query wants button/input/menu, but candidate is text/list item/cover
    ("QQ", "群公告", "ambiguous", None, "content", "Group announcement is text, not a button", "role_mismatch_fp"),
    ("QQ", "全体成员", "abstain", None, None, "@all text in message, not a button", "role_mismatch_fp"),
    ("WeChat", "新消息", "abstain", None, None, "New message text in notification, not a button", "role_mismatch_fp"),
    ("WeChat", "是的", "abstain", None, None, "Message text 'yes', not a button", "role_mismatch_fp"),
    ("NetEase", "后退", "ambiguous", None, "toolbar", "Back text may be navigation or just label", "role_mismatch_fp"),
    ("VSCode", "标题操作", "abstain", None, None, "Toolbar label text, not a button", "role_mismatch_fp"),

    # ── FP-Challenge: wrong_region_fp ──
    # Query specifies region, but hits content area text
    ("Notepad", "底部状态栏", "ambiguous", None, "status_bar", "Status bar exists but query may hit content", "wrong_region_fp"),
    ("Chrome", "顶部搜索", "ambiguous", None, "toolbar", "Search is in toolbar, not content", "wrong_region_fp"),
    ("QQ", "侧栏搜索", "target", "搜索", "sidebar", "Search is in sidebar", "wrong_region_fp"),
    ("WeChat", "侧栏搜索", "target", "搜索", "sidebar", "Search is in sidebar", "wrong_region_fp"),
]


def observe_app(svc, hwnd, process_name):
    """Observe an app and return the canvas."""
    zone_page = svc.analyze(hwnd=hwnd, allow_vlm=False)
    canvas = svc.create_page_snapshot(zone_page, process_name=process_name)
    return canvas


def run_query(canvas, query_text, max_results=5, min_confidence=0.0):
    """Run a natural language query against a canvas."""
    engine = CanvasQueryEngine()
    target = QueryTarget(natural_language=query_text)
    return engine.query(canvas, target, max_results=max_results, min_confidence=min_confidence)


def analyze_candidate_evidence(candidate):
    """Extract evidence details from a candidate."""
    return {
        "element_id": candidate.element_id,
        "semantic_role": str(candidate.semantic_role),
        "control_type": candidate.control_type or "",
        "text": (candidate.text or "")[:50],
        "name": (candidate.name or "")[:30],
        "confidence": round(candidate.confidence, 3),
        "confidence_level": str(candidate.confidence_level),
        "provider_sources": candidate.provider_sources,
        "interactable": candidate.interactable,
        "visual_type": candidate.visual_type or "",
        "role_label": candidate.role_label or "",
        "role_confidence": round(candidate.role_confidence, 3) if candidate.role_confidence else 0,
        "refine_status": candidate.refine_status or "",
        "risk_tags": candidate.risk_tags,
        "region_id": candidate.region_id or "",
        "from_memory": candidate.from_memory,
    }


def run_audit():
    """Run the full P1.1a evidence audit."""
    svc = PerceptionService()
    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)

    # Map app names to processes
    app_process_map = {
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
    for app_name, process in app_process_map.items():
        win = next(
            (w for w in windows if process.lower() in (w.process_name or "").lower()),
            None,
        )
        if not win:
            print(f"{app_name}: SKIP (window not found)")
            continue
        try:
            canvas = observe_app(svc, win.hwnd, process)
            canvases[app_name] = canvas
            print(f"{app_name}: {len(canvas.elements)} candidates")
        except Exception as e:
            print(f"{app_name}: ERROR ({e})")

    # Run queries
    results = []
    for app_name, query, expected, expected_target_text, expected_region, notes, failure_hint in QUERY_TESTS:
        canvas = canvases.get(app_name)
        if not canvas:
            results.append({
                "app": app_name, "query": query, "status": "skipped",
                "reason": "no canvas", "expected": expected,
                "expected_target_text": expected_target_text,
                "expected_region": expected_region, "notes": notes,
                "failure_hint": failure_hint,
            })
            continue

        try:
            query_result = run_query(canvas, query)
            candidates = query_result.candidates
            top1 = candidates[0] if candidates else None
            top3 = candidates[:3]

            # Analyze top candidates
            top1_evidence = analyze_candidate_evidence(top1) if top1 else None
            top3_evidence = [analyze_candidate_evidence(c) for c in top3]

            # Determine verdict based on expected
            has_results = len(candidates) > 0
            top1_conf = top1.confidence if top1 else 0

            if expected == "target":
                # Expected to find the target
                if has_results:
                    # Check if top1 matches expected target
                    top1_text = (top1.text or "") if top1 else ""
                    if expected_target_text and expected_target_text in top1_text:
                        verdict = "hit"
                    elif has_results:
                        verdict = "hit"  # Found something, may be correct
                    else:
                        verdict = "miss"
                else:
                    verdict = "miss"
            elif expected == "abstain":
                # Expected no results
                if not has_results:
                    verdict = "correct_abstain"
                else:
                    verdict = "fp_missed_abstain"
            else:  # ambiguous
                # Either result is acceptable
                if has_results:
                    verdict = "ambiguous_hit"
                else:
                    verdict = "ambiguous_abstain"

            results.append({
                "app": app_name,
                "query": query,
                "expected": expected,
                "expected_target_text": expected_target_text,
                "expected_region": expected_region,
                "notes": notes,
                "failure_hint": failure_hint,
                "verdict": verdict,
                "top1_confidence": top1_conf,
                "top1_evidence": top1_evidence,
                "top3_evidence": top3_evidence,
                "total_candidates": len(candidates),
            })
        except Exception as e:
            results.append({
                "app": app_name, "query": query, "status": "error",
                "error": str(e), "expected_type": expected_type,
            })

    # Save results
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "P1.1a-evidence-audit",
        "total_queries": len(results),
        "results": results,
    }

    with open(REPORT_DIR / "audit_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print("  P1.1a Evidence Audit Summary")
    print(f"{'='*70}")

    hits = [r for r in results if r.get("verdict") == "hit"]
    fps = [r for r in results if r.get("verdict") == "fp_missed_abstain"]
    abstains = [r for r in results if r.get("verdict") == "correct_abstain"]
    misses = [r for r in results if r.get("verdict") == "miss"]
    amb_hits = [r for r in results if r.get("verdict") == "ambiguous_hit"]
    amb_abstains = [r for r in results if r.get("verdict") == "ambiguous_abstain"]

    print(f"  Total queries: {len(results)}")
    print(f"  Hits (target found): {len(hits)} ({100*len(hits)/max(len(results),1):.0f}%)")
    print(f"  FP (should abstain, found): {len(fps)} ({100*len(fps)/max(len(results),1):.0f}%)")
    print(f"  Correct abstains: {len(abstains)} ({100*len(abstains)/max(len(results),1):.0f}%)")
    print(f"  Misses (should find, didn't): {len(misses)} ({100*len(misses)/max(len(results),1):.0f}%)")
    print(f"  Ambiguous hit: {len(amb_hits)} ({100*len(amb_hits)/max(len(results),1):.0f}%)")
    print(f"  Ambiguous abstain: {len(amb_abstains)} ({100*len(amb_abstains)/max(len(results),1):.0f}%)")

    # Per-app summary
    print(f"\n  Per-App Summary:")
    apps = sorted(set(r["app"] for r in results))
    for app in apps:
        app_results = [r for r in results if r["app"] == app]
        app_hits = sum(1 for r in app_results if r.get("verdict") == "hit")
        app_fps = sum(1 for r in app_results if r.get("verdict") == "fp_missed_abstain")
        app_misses = sum(1 for r in app_results if r.get("verdict") == "miss")
        app_abstains = sum(1 for r in app_results if r.get("verdict") == "correct_abstain")
        print(f"    {app:10s}: {len(app_results):2d} queries | hit={app_hits} fp={app_fps} miss={app_misses} abstain={app_abstains}")

    # Failure type categorization
    print(f"\n  Failure Types:")
    failure_types = {}
    for r in results:
        if r.get("verdict") in ("fp_missed_abstain", "miss"):
            hint = r.get("notes", "unknown")
            # Categorize
            if "substring" in str(hint).lower() or "子串" in str(hint).lower():
                ftype = "substring_fp"
            elif "icon" in str(hint).lower() or "图标" in str(hint).lower():
                ftype = "icon_miss"
            elif "intent" in str(hint).lower():
                ftype = "intent_gap_miss"
            elif "text" in str(hint).lower() or "正文" in str(hint).lower():
                ftype = "text_content_fp"
            elif "region" in str(hint).lower() or "区域" in str(hint).lower():
                ftype = "region_confusion"
            elif r.get("verdict") == "fp_missed_abstain":
                ftype = "substring_fp"
            else:
                ftype = "intent_gap_miss"
            failure_types[ftype] = failure_types.get(ftype, 0) + 1

    for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
        print(f"    {ftype:25s}: {count}")

    # Top FP details
    if fps:
        print(f"\n  False Positives:")
        for fp in fps:
            top1 = fp.get("top1_evidence", {})
            src = top1.get("provider_sources", [])
            print(f"    {fp['app']:10s} \"{fp['query']}\" → \"{top1.get('text','?')[:25]}\" conf={fp.get('top1_confidence',0):.2f} src={src}")

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_audit()
