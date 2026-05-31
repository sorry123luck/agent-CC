"""
Console API contract verification script.

Checks:
1. GET /api/v1/canvases → 200, valid list
2. GET /api/v1/canvases/{id} → 200 (import fix verification)
3. GET /api/v1/state-templates/{id}/virtual-model → 200, candidates >= 1
4. GET /api/v1/windows → 200
5. POST /api/v1/observe → 200
6. POST /api/v1/query → 200
7. POST /api/v1/diff → 200
8. GET /api/v1/page-models/tree → 200

Usage:
    python scripts/verify_console_contract.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify console API contracts")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--hwnd", type=int, default=0, help="Window handle for observe test (0 = skip)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    all_pass = True

    print(f"\n=== Console API Contract Verification ===")
    print(f"Base URL: {base}\n")

    # ── 1. GET /api/v1/canvases ──
    print("[1] GET /api/v1/canvases")
    try:
        r = requests.get(f"{base}/api/v1/canvases", timeout=10)
        ok = r.status_code == 200
        detail = f"status={r.status_code}"
        if ok:
            data = r.json()
            detail += f", count={len(data)}"
            # Check that required fields exist
            if data:
                item = data[0]
                required = ["canvas_id", "app_id", "window_title"]
                missing = [k for k in required if k not in item]
                if missing:
                    ok = False
                    detail += f", missing_fields={missing}"
        all_pass &= check("list_canvases", ok, detail)
    except Exception as e:
        all_pass &= check("list_canvases", False, str(e))

    # ── 2. GET /api/v1/canvases/{id} (import fix) ──
    print("\n[2] GET /api/v1/canvases/{id}")
    try:
        # First get a canvas_id from the list
        r_list = requests.get(f"{base}/api/v1/canvases", timeout=10)
        if r_list.status_code == 200 and r_list.json():
            canvas_id = r_list.json()[0]["canvas_id"]
            r = requests.get(f"{base}/api/v1/canvases/{canvas_id}", timeout=10)
            ok = r.status_code == 200
            detail = f"status={r.status_code}, canvas_id={canvas_id[:8]}..."
            if ok:
                data = r.json()
                # Check CanvasDetail fields
                required = ["canvas_id", "app_id", "elements"]
                missing = [k for k in required if k not in data]
                if missing:
                    ok = False
                    detail += f", missing_fields={missing}"
                else:
                    detail += f", elements={len(data.get('elements', []))}"
        else:
            ok = True
            detail = "no canvases in cache (skipped — run observe first)"
        all_pass &= check("get_canvas_detail", ok, detail)
    except Exception as e:
        all_pass &= check("get_canvas_detail", False, str(e))

    # ── 3. GET /api/v1/state-templates/{id}/virtual-model ──
    print("\n[3] GET /api/v1/state-templates/{id}/virtual-model")
    try:
        r_tree = requests.get(f"{base}/api/v1/page-models/tree", timeout=10)
        if r_tree.status_code == 200:
            tree = r_tree.json()
            templates = tree.get("state_templates", [])
            st_id = templates[0].get("state_template_id") if templates else None

            if st_id:
                r = requests.get(f"{base}/api/v1/state-templates/{st_id}/virtual-model", timeout=10)
                ok = r.status_code == 200
                detail = f"status={r.status_code}"
                if ok:
                    data = r.json()
                    cand_count = len(data.get("candidates", []))
                    has_canvas = data.get("has_available_canvas", False)
                    detail += f", candidates={cand_count}, has_available_canvas={has_canvas}"
                    if cand_count < 1:
                        ok = False
                        detail += " (expected >= 1)"
                    # Check that fixed and dynamic candidates are present
                    fixed = [c for c in data.get("candidates", []) if c.get("is_fixed_control")]
                    dynamic = [c for c in data.get("candidates", []) if not c.get("is_fixed_control")]
                    detail += f", fixed={len(fixed)}, dynamic={len(dynamic)}"
            else:
                ok = True
                detail = "no state_templates found (skipped — run observe first)"
        else:
            ok = True
            detail = "page-models/tree not available (skipped)"
        all_pass &= check("virtual_model", ok, detail)
    except Exception as e:
        all_pass &= check("virtual_model", False, str(e))

    # ── 4. GET /api/v1/windows ──
    print("\n[4] GET /api/v1/windows")
    try:
        r = requests.get(f"{base}/api/v1/windows", timeout=10)
        ok = r.status_code == 200
        detail = f"status={r.status_code}"
        if ok:
            detail += f", count={len(r.json())}"
        all_pass &= check("list_windows", ok, detail)
    except Exception as e:
        all_pass &= check("list_windows", False, str(e))

    # ── 5. POST /api/v1/observe (if --hwnd provided) ──
    if args.hwnd:
        print(f"\n[5] POST /api/v1/observe (hwnd={args.hwnd})")
        try:
            r = requests.post(
                f"{base}/api/v1/observe",
                json={"hwnd": args.hwnd, "allow_vlm": False, "force_vlm": False},
                timeout=30,
            )
            ok = r.status_code == 200
            detail = f"status={r.status_code}"
            if ok:
                data = r.json()
                mms = data.get("model_match_status", "?")
                detail += f", canvas_id={data.get('canvas_id', '?')[:8]}..., model_match_status={mms}"
            all_pass &= check("observe", ok, detail)
        except Exception as e:
            all_pass &= check("observe", False, str(e))
    else:
        print("\n[5] POST /api/v1/observe — SKIPPED (no --hwnd)")

    # ── 6. POST /api/v1/query (needs a canvas_id) ──
    print("\n[6] POST /api/v1/query")
    try:
        r_list = requests.get(f"{base}/api/v1/canvases", timeout=10)
        if r_list.status_code == 200 and r_list.json():
            canvas_id = r_list.json()[0]["canvas_id"]
            r = requests.post(
                f"{base}/api/v1/query",
                json={
                    "canvas_id": canvas_id,
                    "target": {"text": "button"},
                    "max_results": 5,
                },
                timeout=10,
            )
            ok = r.status_code == 200
            detail = f"status={r.status_code}"
            if ok:
                data = r.json()
                detail += f", results={len(data.get('candidates', []))}"
        else:
            ok = True
            detail = "no canvases (skipped)"
        all_pass &= check("query", ok, detail)
    except Exception as e:
        all_pass &= check("query", False, str(e))

    # ── 7. POST /api/v1/diff ──
    if args.hwnd:
        print(f"\n[7] POST /api/v1/diff (hwnd={args.hwnd})")
        try:
            r_list = requests.get(f"{base}/api/v1/canvases", timeout=10)
            if r_list.status_code == 200 and len(r_list.json()) >= 1:
                canvas_id = r_list.json()[0]["canvas_id"]
                r = requests.post(
                    f"{base}/api/v1/diff",
                    json={
                        "hwnd": args.hwnd,
                        "previous_canvas_id": canvas_id,
                        "detail_level": "changes",
                    },
                    timeout=30,
                )
                ok = r.status_code == 200
                detail = f"status={r.status_code}"
                if ok:
                    data = r.json()
                    detail += f", added={len(data.get('added', []))}, removed={len(data.get('removed', []))}"
            else:
                ok = True
                detail = "no canvases (skipped)"
            all_pass &= check("diff", ok, detail)
        except Exception as e:
            all_pass &= check("diff", False, str(e))
    else:
        print("\n[7] POST /api/v1/diff — SKIPPED (no --hwnd)")

    # ── 8. GET /api/v1/page-models/tree ──
    print("\n[8] GET /api/v1/page-models/tree")
    try:
        r = requests.get(f"{base}/api/v1/page-models/tree", timeout=10)
        ok = r.status_code == 200
        detail = f"status={r.status_code}"
        if ok:
            data = r.json()
            page_models = data.get("page_models", [])
            state_templates = data.get("state_templates", [])
            detail += f", page_models={len(page_models)}, state_templates={len(state_templates)}"
        all_pass &= check("page_model_tree", ok, detail)
    except Exception as e:
        all_pass &= check("page_model_tree", False, str(e))

    # ── Summary ──
    print(f"\n{'='*50}")
    if all_pass:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
