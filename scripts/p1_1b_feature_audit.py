"""P1.1b.1 Feature Audit: Distinguish real controls from content text.

Read-only analysis of candidate features for specific FP/hit/abstain samples.
Does NOT change scoring, query_engine, API, or main pipeline.

Usage:
    python scripts/p1_1b_feature_audit.py
"""

import json
import os
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

REPORT_DIR = Path("reports/p1_1b_feature_audit")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Key samples to analyze ──────────────────────────────────────
KEY_SAMPLES = [
    # (app, query, expected, top1_text_hint, category)
    # Hits (should NOT be demoted)
    ("QQ", "搜索", "hit", "搜索", "hit_misclassified"),
    ("WeChat", "搜索", "hit", "搜索", "hit_misclassified"),
    # FP not fixed (should be demoted)
    ("QQ", "发送按钮", "fp", "发送", "text_content_fp"),
    ("WeChat", "发送按钮", "fp", "发送", "text_content_fp"),
    ("Notepad", "搜索框", "fp", "文件", "substring_fp"),
    ("NetEase", "门尼", "fp", "门尼", "text_content_fp"),
    ("NetEase", "巴拉莱卡", "fp", "巴拉莱卡", "text_content_fp"),
    ("VSCode", "标题操作", "fp", "无标题", "substring_fp"),
    # FP fixed by shadow (should stay demoted)
    ("WeChat", "服务通知", "fp", "服务通知", "text_content_fp_fixed"),
    ("WeChat", "卡券", "fp", "卡券", "text_content_fp_fixed"),
    ("WeChat", "新消息", "fp", "新消息", "text_content_fp_fixed"),
]


def extract_candidate_features(candidate, canvas):
    """Extract detailed features from a candidate."""
    # Basic fields
    text = candidate.text or ""
    role = str(candidate.semantic_role).split(".")[-1]
    ctrl_type = candidate.control_type or ""
    conf = candidate.confidence
    sources = candidate.provider_sources or []
    bounds = candidate.bounds or (0, 0, 0, 0)
    w = bounds[2] - bounds[0] if len(bounds) >= 4 else 0
    h = bounds[3] - bounds[1] if len(bounds) >= 4 else 0
    area = w * h

    # Region info
    region_id = candidate.region_id or ""

    # Refine info
    visual_type = candidate.visual_type or ""
    semantic_tags = candidate.semantic_tags or []
    role_label = candidate.role_label or ""
    role_confidence = candidate.role_confidence or 0
    refine_status = candidate.refine_status or ""

    # Actionable check
    actionable_roles = {"BUTTON", "SEARCH_INPUT", "MENU_ITEM", "TAB",
                        "CHECKBOX", "LINK", "DROPDOWN", "TOOLBAR"}
    actionable_controls = {"button", "hyperlink", "menuitem", "tabitem",
                           "checkbox", "radiobutton", "togglebutton",
                           "edit", "combobox"}
    is_actionable_role = role in actionable_roles
    is_actionable_control = ctrl_type.lower() in actionable_controls

    # Provider analysis
    has_uia = "uia" in sources
    has_ocr = "ocr" in sources
    has_vision = "omni" in sources or "vision" in sources

    return {
        "element_id": candidate.element_id,
        "text": text[:50],
        "semantic_role": role,
        "control_type": ctrl_type,
        "confidence": round(conf, 3),
        "provider_sources": sources,
        "has_uia": has_uia,
        "has_ocr": has_ocr,
        "has_vision": has_vision,
        "bounds": list(bounds),
        "width": w,
        "height": h,
        "area": area,
        "aspect_ratio": round(w / max(h, 1), 2),
        "region_id": region_id,
        "visual_type": visual_type,
        "semantic_tags": semantic_tags,
        "role_label": role_label,
        "role_confidence": round(role_confidence, 3) if role_confidence else 0,
        "refine_status": refine_status,
        "is_actionable_role": is_actionable_role,
        "is_actionable_control": is_actionable_control,
        "interactable": candidate.interactable,
        "from_memory": candidate.from_memory,
        "risk_tags": candidate.risk_tags,
    }


def run_audit():
    """Run P1.1b.1 feature audit."""
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

    # Observe apps
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

    # Analyze key samples
    engine = CanvasQueryEngine()
    analysis = []

    for app_name, query, expected, top1_hint, category in KEY_SAMPLES:
        canvas = canvases.get(app_name)
        if not canvas:
            analysis.append({"app": app_name, "query": query, "status": "no_canvas"})
            continue

        result = engine.query(canvas, QueryTarget(natural_language=query), max_results=3, min_confidence=0.0)
        candidates = result.candidates

        if not candidates:
            analysis.append({
                "app": app_name, "query": query, "expected": expected,
                "category": category, "top1": None, "status": "no_candidates",
            })
            continue

        top1 = candidates[0]
        top1_features = extract_candidate_features(top1, canvas)

        # Also get top-2 and top-3 if available
        top2_features = extract_candidate_features(candidates[1], canvas) if len(candidates) > 1 else None
        top3_features = extract_candidate_features(candidates[2], canvas) if len(candidates) > 2 else None

        analysis.append({
            "app": app_name,
            "query": query,
            "expected": expected,
            "category": category,
            "top1": top1_features,
            "top2": top2_features,
            "top3": top3_features,
            "total_candidates": len(candidates),
        })

    # Save
    report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "P1.1b.1-feature-audit",
        "analysis": analysis,
    }
    with open(REPORT_DIR / "feature_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print analysis
    print(f"\n{'='*80}")
    print("  P1.1b.1 Feature Audit")
    print(f"{'='*80}")

    for a in analysis:
        app = a["app"]
        query = a["query"]
        expected = a["expected"]
        category = a["category"]
        top1 = a.get("top1")

        if not top1:
            print(f"\n  {app} \"{query}\" ({category}): NO CANDIDATES")
            continue

        print(f"\n  {app} \"{query}\" ({category}, expected={expected})")
        print(f"    Top1: \"{top1['text']}\" role={top1['semantic_role']} ctrl={top1['control_type']}")
        print(f"    Sources: {top1['provider_sources']}")
        print(f"    Confidence: {top1['confidence']}")
        print(f"    Actionable: role={top1['is_actionable_role']} ctrl={top1['is_actionable_control']}")
        print(f"    Visual type: {top1['visual_type']} refine={top1['refine_status']}")
        print(f"    Region: {top1['region_id']}")
        print(f"    Bounds: {top1['width']}x{top1['height']} area={top1['area']}")

    # Feature comparison
    print(f"\n{'='*80}")
    print("  Feature Comparison: Hit vs FP")
    print(f"{'='*80}")

    hits = [a for a in analysis if a.get("expected") == "hit" and a.get("top1")]
    fps = [a for a in analysis if a.get("expected") == "fp" and a.get("top1")]

    print(f"\n  Hits ({len(hits)}):")
    for a in hits:
        t = a["top1"]
        print(f"    {a['app']:10s} \"{a['query']}\" → \"{t['text'][:25]}\" role={t['semantic_role']} src={t['provider_sources']} actionable={t['is_actionable_role']}")

    print(f"\n  FP ({len(fps)}):")
    for a in fps:
        t = a["top1"]
        print(f"    {a['app']:10s} \"{a['query']}\" → \"{t['text'][:25]}\" role={t['semantic_role']} src={t['provider_sources']} actionable={t['is_actionable_role']}")

    print(f"\n  Report: {REPORT_DIR}")


if __name__ == "__main__":
    run_audit()
