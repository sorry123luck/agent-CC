"""P1.2c: Delete-button dynamic/attempted-dynamic check (corrected).

Shadow-only. Does NOT change src/canvas/query_engine.py.

Key corrections:
- Records canvas_id, element_count, screenshot_hash per state
- Verifies scroll actually changed page state
- Reports both gate_reason and final abstain_reason
- Marks state_change=not_verified if scroll didn't change anything

Usage:
    python scripts/p1_2c_dynamic_check.py
"""

import hashlib
import json
import os
import re
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

REPORT_DIR = Path("reports/p1_2c_dynamic_delete_button")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Action gate logic ───────────────────────────────────────────

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


def contains_action_word(text, action_words):
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
    return contains_action_word(query, GATE_ACTION_WORDS)


def collect_action_evidence(candidate, query_action):
    evidence = []
    if not query_action:
        return evidence
    action_set = {query_action}
    if query_action in ACTION_SYNONYMS:
        action_set.update(ACTION_SYNONYMS[query_action])

    text = (candidate.text or "").strip()
    if text and text.lower() not in ("none", ""):
        word = contains_action_word(text, action_set)
        if word:
            evidence.append(("text_match", word))

    name = (candidate.name or "").strip()
    if name and name.lower() not in ("none", "") and name != text:
        word = contains_action_word(name, action_set)
        if word:
            evidence.append(("name_match", word))

    attrs = getattr(candidate, "attributes", {}) or {}
    auto_id = (attrs.get("automation_id", "") or "").lower()
    if auto_id:
        word = contains_action_word(auto_id, action_set)
        if word:
            evidence.append(("automation_id_match", word))

    ocr_text = (attrs.get("ocr_text", "") or "").strip()
    if ocr_text and ocr_text != text:
        word = contains_action_word(ocr_text, action_set)
        if word:
            evidence.append(("ocr_text_match", word))

    return evidence


def candidate_has_text(candidate):
    text = (candidate.text or "").strip()
    name = (candidate.name or "").strip()
    if text and text.lower() not in ("none", ""):
        return True
    if name and name.lower() not in ("none", "") and name != text:
        return True
    attrs = getattr(candidate, "attributes", {}) or {}
    if (attrs.get("ocr_text", "") or "").strip():
        return True
    return False


def candidate_is_actionable(candidate):
    full_role = str(candidate.semantic_role)
    ctrl_type = (candidate.control_type or "").lower()
    return full_role in ACTIONABLE_UIA_ROLES or ctrl_type in ACTIONABLE_CONTROL_TYPES


def apply_action_gate(score, query, candidate):
    """Apply action semantic gate. Returns (new_score, gate_reasons)."""
    query_action = extract_action_word(query)
    if not query_action:
        return score, []

    evidence = collect_action_evidence(candidate, query_action)
    is_actionable = candidate_is_actionable(candidate)
    has_text = candidate_has_text(candidate)

    if is_actionable and evidence:
        return min(1.0, score * 1.5), ["actionable_with_action_match"]
    elif is_actionable and not evidence and has_text:
        return score * 0.15, ["action_mismatch_abstain"]
    elif is_actionable and not evidence and not has_text:
        return score, ["icon_only_needs_semantic_hint"]
    return score, []


# ── Window activation and scroll ────────────────────────────────

def activate_and_scroll(hwnd, direction="down", steps=3):
    """Activate window, move mouse to center, scroll."""
    import ctypes
    user32 = ctypes.windll.user32

    # Get window rect
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    center_x = (rect.left + rect.right) // 2
    center_y = (rect.top + rect.bottom) // 2

    # Bring to foreground
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)

    # Move mouse to center of window
    ctypes.windll.user32.SetCursorPos(center_x, center_y)
    time.sleep(0.2)

    # Scroll
    for _ in range(steps):
        if direction == "down":
            user32.mouse_event(0x0800, 0, 0, -120, 0)
        else:
            user32.mouse_event(0x0800, 0, 0, 120, 0)
        time.sleep(0.3)


def screenshot_hash(screenshot):
    """Compute perceptual hash of screenshot for state comparison."""
    if screenshot is None:
        return None
    import numpy as np
    # Resize to 16x16 grayscale and compute hash
    small = screenshot.convert("L").resize((16, 16))
    arr = np.asarray(small)
    median = np.median(arr)
    bits = (arr > median).flatten()
    hash_bytes = bytes(bits.astype(int).tolist())
    return hashlib.md5(hash_bytes).hexdigest()[:16]


# ── Main ────────────────────────────────────────────────────────

def run_dynamic_check():
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

    rows = []
    state_changes_verified = 0
    state_changes_not_verified = 0

    for app_name, process in app_process.items():
        win = next(
            (w for w in windows if process.lower() in (w.process_name or "").lower()),
            None,
        )
        if not win:
            rows.append({"app": app_name, "state": "baseline", "verdict": "skipped",
                         "reason": "no_window", "state_change": "n/a"})
            rows.append({"app": app_name, "state": "scroll_down", "verdict": "skipped",
                         "reason": "no_window", "state_change": "n/a"})
            continue

        baseline_meta = {}

        for state in ["baseline", "scroll_down"]:
            # Scroll for changed states
            if state == "scroll_down":
                activate_and_scroll(win.hwnd, "down", 3)
                time.sleep(1)

            # Observe
            try:
                zone_page = svc.analyze(hwnd=win.hwnd, allow_vlm=False)
                canvas = svc.create_page_snapshot(zone_page, process_name=process)
            except Exception as e:
                rows.append({"app": app_name, "query": "删除按钮", "state": state,
                             "verdict": "error", "error": str(e), "state_change": "error"})
                continue

            # State metadata
            canvas_id = getattr(canvas, "canvas_id", "")
            element_count = len(canvas.elements)
            screenshot = getattr(zone_page, "screenshot", None)
            s_hash = screenshot_hash(screenshot)

            # Verify state change
            state_change = "not_verified"
            if state == "scroll_down" and baseline_meta:
                if s_hash != baseline_meta.get("s_hash"):
                    state_change = "verified"
                    state_changes_verified += 1
                elif element_count != baseline_meta.get("element_count"):
                    state_change = "verified"
                    state_changes_verified += 1
                else:
                    state_change = "not_verified"
                    state_changes_not_verified += 1
            elif state == "baseline":
                state_change = "baseline"
                baseline_meta = {"s_hash": s_hash, "element_count": element_count, "canvas_id": canvas_id}

            # Query "删除按钮"
            try:
                result = engine.query(
                    canvas,
                    QueryTarget(natural_language="删除按钮"),
                    max_results=3,
                    min_confidence=0.0,
                )
                candidates = result.candidates

                shadow_results = []
                for c in candidates[:3]:
                    base_score = c.confidence
                    gated_score, gate_reasons = apply_action_gate(base_score, "删除按钮", c)
                    shadow_results.append({
                        "text": (c.text or "")[:30],
                        "role": str(c.semantic_role).split(".")[-1],
                        "ctrl": c.control_type or "",
                        "base_score": round(base_score, 3),
                        "shadow_score": round(gated_score, 3),
                        "gate_reasons": gate_reasons,
                    })
                shadow_results.sort(key=lambda r: -r["shadow_score"])

                top1 = candidates[0] if candidates else None
                sr1 = shadow_results[0] if shadow_results else None

                has_results = len(candidates) > 0
                verdict = "fp" if has_results else "correct_abstain"

                # Determine final abstain_reason
                if sr1 is None:
                    should_abstain = True
                    final_abstain_reason = "no_candidates"
                elif sr1["shadow_score"] < 0.3:
                    should_abstain = True
                    final_abstain_reason = "low_shadow_score"
                elif "action_mismatch_abstain" in sr1["gate_reasons"]:
                    should_abstain = True
                    final_abstain_reason = "action_mismatch_abstain"
                elif "icon_only_needs_semantic_hint" in sr1["gate_reasons"]:
                    should_abstain = True
                    final_abstain_reason = "icon_only_needs_semantic_hint"
                else:
                    should_abstain = False
                    final_abstain_reason = None

                rows.append({
                    "app": app_name,
                    "query": "删除按钮",
                    "state": state,
                    "verdict": verdict,
                    "canvas_id": canvas_id,
                    "screenshot_hash": s_hash,
                    "element_count": element_count,
                    "state_change": state_change,
                    "top1_text": (top1.text or "")[:30] if top1 else "",
                    "top1_bounds": list(top1.bounds) if top1 and hasattr(top1, "bounds") else [],
                    "top1_role": str(top1.semantic_role).split(".")[-1] if top1 else "",
                    "top1_ctrl": top1.control_type or "" if top1 else "",
                    "base_score": sr1["base_score"] if sr1 else None,
                    "shadow_score": sr1["shadow_score"] if sr1 else None,
                    "gate_reasons": sr1["gate_reasons"] if sr1 else [],
                    "final_abstain_reason": final_abstain_reason,
                    "should_abstain": should_abstain,
                    "total_candidates": len(candidates),
                })
            except Exception as e:
                rows.append({"app": app_name, "query": "删除按钮", "state": state,
                             "verdict": "error", "error": str(e), "state_change": "error"})

    # ── Summary ──
    fps = [r for r in rows if r.get("verdict") == "fp"]
    fixed = [r for r in fps if r.get("should_abstain")]
    remaining = [r for r in fps if not r.get("should_abstain")]
    hits = [r for r in rows if r.get("verdict") == "hit"]
    damaged = [r for r in hits if r.get("should_abstain")]
    abstains = [r for r in rows if r.get("verdict") == "correct_abstain"]
    broken = [r for r in abstains if not r.get("should_abstain")]

    # Gate reason counts (from shadow_results)
    gate_reasons = Counter()
    for r in rows:
        for gr in r.get("gate_reasons", []):
            gate_reasons[gr] += 1

    # Final abstain reason counts
    abstain_reasons = Counter()
    for r in rows:
        reason = r.get("final_abstain_reason")
        if reason:
            abstain_reasons[reason] += 1

    # State change counts
    verified = sum(1 for r in rows if r.get("state_change") == "verified")
    not_verified = sum(1 for r in rows if r.get("state_change") == "not_verified")

    summary = {
        "phase": "P1.2c-dynamic-delete-button-check-corrected",
        "base_score": "candidate.confidence (not query engine ranking)",
        "total": len(rows),
        "fp_total": len(fps),
        "fp_fixed": len(fixed),
        "fp_remaining": len(remaining),
        "hit_damage": len(damaged),
        "broken_abstain": len(broken),
        "state_change_verified": verified,
        "state_change_not_verified": not_verified,
        "gate_reason_counts": dict(gate_reasons),
        "final_abstain_reason_counts": dict(abstain_reasons),
        "rows": rows,
    }
    with open(REPORT_DIR / "p1_2c_dynamic.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  P1.2c Dynamic Delete-Button Check (corrected)")
    print(f"  {'='*55}")
    print(f"  Base score: candidate.confidence")
    print(f"  Total: {len(rows)}")
    print(f"  FP: {len(fps)}")
    print(f"  Fixed: {len(fixed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Hit damage: {len(damaged)}")
    print(f"  Broken abstain: {len(broken)}")
    print(f"  ---")
    print(f"  State change verified: {verified}")
    print(f"  State change not verified: {not_verified}")
    print(f"  ---")
    print(f"  Gate reasons: {dict(gate_reasons)}")
    print(f"  Final abstain reasons: {dict(abstain_reasons)}")
    print()
    for r in rows:
        s = r.get("shadow_score")
        ss = f"{s:.2f}" if s is not None else "None"
        sc = r.get("state_change", "?")
        gr = r.get("gate_reasons", [])
        ar = r.get("final_abstain_reason", "")
        print(f'  {r["app"]:10s} [{r.get("state","?"):12s}] {r.get("verdict","?"):15s} '
              f'top1="{r.get("top1_text","")[:18]}" shadow={ss} '
              f'state={sc:12s} gate={gr} abstain={ar}')
    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_dynamic_check()
