"""Verify icon memory matching accuracy.

Usage:
    python scripts/icon_memory_verify.py [--app chrome] [--threshold 5]

Tests:
    1. Self-match: same crop should match with distance 0
    2. Similar-match: slightly different crop should match within threshold
    3. Cross-app: different app should NOT match
    4. Stats report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.memory.icon_memory_store import IconMemoryStore
from src.memory.visual_anchor import compute_dhash, dhash_distance

REPORT_DIR = Path("reports/p12_provider_layers")


def load_crops(app_id: str) -> list[dict]:
    """Load crop info from provider_layers.json."""
    layers_path = REPORT_DIR / app_id / "provider_layers.json"
    screenshot_path = REPORT_DIR / app_id / "raw_screenshot.png"

    if not layers_path.exists():
        return []

    with open(layers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    screenshot = None
    if screenshot_path.exists():
        screenshot = Image.open(screenshot_path)

    results = []
    for layer in data.get("layers", []):
        bounds = layer.get("bounds")
        if not bounds or len(bounds) < 4:
            continue
        if not layer.get("in_canvas", False):
            continue

        role = layer.get("role", layer.get("semantic_role", ""))
        text = (layer.get("text") or "").strip()

        crop = None
        if screenshot is not None:
            l, t, r, b = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
            l = max(0, l)
            t = max(0, t)
            r = min(screenshot.width, r)
            b = min(screenshot.height, b)
            if r > l and b > t:
                crop = screenshot.crop((l, t, r, b))

        if crop is not None:
            arr = np.array(crop.convert("RGB"))
            dhash = compute_dhash(arr)
            results.append({
                "app_id": app_id,
                "role": role,
                "text": text,
                "bounds": bounds,
                "dhash": dhash,
                "crop_size": crop.size,
            })

    return results


def verify_self_match(crops: list[dict]) -> dict:
    """Verify same crop matches itself with distance 0."""
    if not crops:
        return {"test": "self_match", "passed": 0, "total": 0}

    passed = 0
    for c in crops:
        dist = dhash_distance(c["dhash"], c["dhash"])
        if dist == 0:
            passed += 1

    return {"test": "self_match", "passed": passed, "total": len(crops)}


def verify_cross_app(app1_crops: list[dict], app2_crops: list[dict], threshold: int = 5) -> dict:
    """Verify different apps' crops don't match (low false positive rate)."""
    if not app1_crops or not app2_crops:
        return {"test": "cross_app", "passed": 0, "total": 0, "false_positives": []}

    false_positives = []
    total = 0

    for c1 in app1_crops[:20]:  # Sample to avoid O(n²)
        for c2 in app2_crops[:20]:
            total += 1
            dist = dhash_distance(c1["dhash"], c2["dhash"])
            if dist <= threshold:
                false_positives.append({
                    "app1": c1["app_id"], "role1": c1["role"], "text1": c1["text"],
                    "app2": c2["app_id"], "role2": c2["role"], "text2": c2["text"],
                    "distance": dist,
                })

    return {
        "test": "cross_app",
        "passed": total - len(false_positives),
        "total": total,
        "false_positive_count": len(false_positives),
        "false_positive_rate": len(false_positives) / max(total, 1),
        "false_positives": false_positives[:10],  # Show first 10
    }


def verify_intra_app(crops: list[dict], threshold: int = 5) -> dict:
    """Check how many unique dHashes exist within one app."""
    if not crops:
        return {"test": "intra_app", "unique_dhashes": 0, "total": 0}

    unique_dhashes = set(c["dhash"] for c in crops)

    # Check distances between unique hashes
    hash_list = list(unique_dhashes)
    collisions = 0
    for i in range(len(hash_list)):
        for j in range(i + 1, len(hash_list)):
            dist = dhash_distance(hash_list[i], hash_list[j])
            if dist <= threshold:
                collisions += 1

    return {
        "test": "intra_app",
        "total_crops": len(crops),
        "unique_dhashes": len(unique_dhashes),
        "collisions_within_threshold": collisions,
    }


def verify_store_match(app_crops, threshold=5):
    """Verify real store.find_match behavior."""
    from unittest.mock import MagicMock
    from src.memory.icon_memory_store import IconMemoryStore
    import tempfile

    tmp = tempfile.mkdtemp()
    store = IconMemoryStore(asset_root=tmp)
    _store = {}

    class _MQ:
        def __init__(self, s):
            self._s = s; self._f = {}
        def filter_by(self, **kw):
            q = _MQ(self._s); q._f = {**self._f, **kw}; return q
        def filter(self, *a): return self
        def order_by(self, *a): return self
        def all(self):
            r = list(self._s.values())
            if "app_id" in self._f: r = [x for x in r if x.app_id == self._f["app_id"]]
            if "asset_type" in self._f: r = [x for x in r if x.asset_type == self._f["asset_type"]]
            if "status" in self._f: r = [x for x in r if x.status == self._f["status"]]
            return r
        def first(self):
            r = self.all(); return r[0] if r else None

    session = MagicMock()
    session.add = lambda row: _store.__setitem__(row.asset_id, row)
    session.query = MagicMock(side_effect=lambda m: _MQ(_store))
    session.flush = MagicMock()

    total_stored = 0
    for app_id, crops in app_crops.items():
        for c in crops:
            if not c.get("dhash"): continue
            store.store_crop(session, None, app_id, "", c["role"],
                semantic_text=c.get("text", ""), precomputed_dhash=c["dhash"],
                privacy_level="hash_only", state="pending")
            total_stored += 1

    sm_pass = sm_total = 0
    for app_id, crops in app_crops.items():
        for c in crops[:20]:
            if not c.get("dhash"): continue
            match, dist = store.find_match(session, c["dhash"], app_id, max_distance=threshold)
            sm_total += 1
            if match and dist == 0: sm_pass += 1

    fp = fp_total = 0
    apps = list(app_crops.keys())
    for i in range(len(apps)):
        for j in range(i+1, len(apps)):
            for c in app_crops[apps[i]][:10]:
                if not c.get("dhash"): continue
                match, _ = store.find_match(session, c["dhash"], apps[j], max_distance=threshold)
                fp_total += 1
                if match: fp += 1

    return {"total_stored": total_stored,
            "self_match": {"passed": sm_pass, "total": sm_total},
            "cross_app": {"fp": fp, "total": fp_total}}


def main():
    parser = argparse.ArgumentParser(description="Verify icon memory matching")
    parser.add_argument("--apps", default="chrome,codex,netease,voicemeeter,qq,weixin")
    parser.add_argument("--threshold", type=int, default=5)
    args = parser.parse_args()

    apps = [a.strip() for a in args.apps.split(",") if a.strip()]

    # Load all crops
    all_crops: dict[str, list[dict]] = {}
    for app_id in apps:
        crops = load_crops(app_id)
        all_crops[app_id] = crops
        print(f"  {app_id}: {len(crops)} crops loaded")

    print()

    # Test 1: Self-match
    for app_id, crops in all_crops.items():
        result = verify_self_match(crops)
        status = "PASS" if result["passed"] == result["total"] else "FAIL"
        print(f"  [{status}] self_match({app_id}): {result['passed']}/{result['total']}")

    print()

    # Test 2: Intra-app unique dHashes
    for app_id, crops in all_crops.items():
        result = verify_intra_app(crops, args.threshold)
        print(f"  intra_app({app_id}): {result['total_crops']} crops → "
              f"{result['unique_dhashes']} unique dHashes, "
              f"{result['collisions_within_threshold']} collisions")

    print()

    # Test 3: Cross-app false positive rate
    app_list = list(all_crops.keys())
    for i in range(len(app_list)):
        for j in range(i + 1, len(app_list)):
            app1, app2 = app_list[i], app_list[j]
            result = verify_cross_app(all_crops[app1], all_crops[app2], args.threshold)
            if result["total"] > 0:
                rate = result["false_positive_rate"]
                status = "PASS" if rate < 0.02 else "FAIL"
                print(f"  [{status}] cross_app({app1}×{app2}): "
                      f"{result['false_positive_count']}/{result['total']} "
                      f"({rate:.1%} FP)")
                if result["false_positives"]:
                    for fp in result["false_positives"][:3]:
                        print(f"    FP: {fp['role1']}:'{fp['text1']}' vs "
                              f"{fp['role2']}:'{fp['text2']}' dist={fp['distance']}")

    # Test 4: Real store.find_match verification
    print()
    print("--- Store.find_match verification ---")
    sr = verify_store_match(all_crops, args.threshold)
    sm = sr["self_match"]
    sm_s = "PASS" if sm["passed"] == sm["total"] else "FAIL"
    print(f"  [{sm_s}] store self_match: {sm['passed']}/{sm['total']}")
    ci = sr["cross_app"]
    ci_s = "PASS" if ci["fp"] == 0 else "FAIL"
    print(f"  [{ci_s}] store cross_app: {ci['fp']} FP / {ci['total']} total")
    print(f"  Total stored: {sr['total_stored']}")

    # Summary
    print(f"\nThreshold: dHash distance <= {args.threshold}")
    print("PASS = false positive rate < 2%")


if __name__ == "__main__":
    main()
