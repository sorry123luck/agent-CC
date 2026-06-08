"""Offline icon memory indexing from provider_layers.json + raw_screenshot.png.

Usage:
    python scripts/icon_memory_index.py [--apps chrome,qq,voicemeeter] [--dry-run]

Input:
    reports/p12_provider_layers/{app}/provider_layers.json
    reports/p12_provider_layers/{app}/raw_screenshot.png

Output:
    VisualAssetRecord(asset_type="control_crop") entries in SQLite
    data/visual_assets/controls/{app}/{dhash8}/{asset_id}.dat crop files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.memory.icon_memory_store import IconMemoryStore

# Fixed control roles that are eligible for indexing
FIXED_ROLES = frozenset({
    "button", "send_button", "submit_button", "cancel_button",
    "icon_button", "toggle_button", "message_input", "text_input",
    "search_input", "menu_item", "nav_item", "tab", "link",
    "toolbar", "menu_bar", "title_bar", "status_bar", "sidebar",
})

# Dynamic content roles that should NOT be indexed
DYNAMIC_ROLES = frozenset({
    "text", "unknown", "chat_item", "list_item", "tree_item",
    "message_content", "contact_name", "group_name",
})

REPORT_DIR = Path("reports/p12_provider_layers")
DEFAULT_APPS = ["chrome", "codex", "netease", "voicemeeter", "qq", "weixin"]


def _is_indexable(layer: dict) -> bool:
    """Check if a provider_layers.json entry is eligible for indexing."""
    # Must be in final canvas
    if not layer.get("in_canvas", False):
        return False

    # Must have bounds
    bounds = layer.get("bounds")
    if not bounds or len(bounds) < 4:
        return False

    # Area check
    l, t, r, b = bounds[0], bounds[1], bounds[2], bounds[3]
    area = max(0, r - l) * max(0, b - t)
    if area < 36:  # 6x6 minimum
        return False
    if area > 1920 * 1080 * 0.1:  # > 10% of 1080p
        return False

    # Role check
    role = layer.get("role", layer.get("semantic_role", ""))
    if role in DYNAMIC_ROLES:
        return False

    # Text length check
    text = (layer.get("text") or "").strip()
    if len(text) > 20:
        return False

    # URL / date / pure digit check
    lower = text.lower()
    if any(token in lower for token in ("http", "www", ".com", ".cn")):
        return False
    if text.replace("/", "").replace("-", "").replace(":", "").strip().isdigit():
        return False

    return True


def _determine_privacy(layer: dict) -> str:
    """Determine privacy level for an element."""
    text = (layer.get("text") or "").strip()
    if not text:
        return "safe"  # Pure icon
    if len(text) <= 4:
        return "redacted"  # Short text, redact
    return "redacted"


def index_app(
    app_id: str,
    store: IconMemoryStore,
    session: object,
    dry_run: bool = False,
) -> dict:
    """Index one app from provider_layers.json."""
    report_dir = REPORT_DIR / app_id
    layers_path = report_dir / "provider_layers.json"
    screenshot_path = report_dir / "raw_screenshot.png"

    if not layers_path.exists():
        return {"app": app_id, "error": "provider_layers.json not found"}

    with open(layers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get("layers", [])
    app_process = data.get("app", app_id)

    # Try to load screenshot
    screenshot = None
    if screenshot_path.exists():
        screenshot = Image.open(screenshot_path)

    # Get window size from data
    elements_total = data.get("elements_total", 0)
    window_w = 1920  # Default
    window_h = 1080

    indexed = 0
    skipped = 0
    errors = []

    for layer in layers:
        if not _is_indexable(layer):
            skipped += 1
            continue

        bounds = layer["bounds"]
        text = (layer.get("text") or "").strip()
        role = layer.get("role", layer.get("semantic_role", ""))

        # Extract crop from screenshot
        crop = None
        if screenshot is not None:
            l, t, r, b = bounds
            # Clip to screenshot bounds
            l = max(0, int(l))
            t = max(0, int(t))
            r = min(screenshot.width, int(r))
            b = min(screenshot.height, int(b))
            if r > l and b > t:
                crop = screenshot.crop((l, t, r, b))

        privacy = _determine_privacy(layer)

        # Compute relative bounds
        rel_bounds = (
            bounds[0] / window_w if window_w else 0,
            bounds[1] / window_h if window_h else 0,
            bounds[2] / window_w if window_w else 0,
            bounds[3] / window_h if window_h else 0,
        )
        size_ratio = (
            (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            / max(window_w * window_h, 1)
        )

        if dry_run:
            indexed += 1
            continue

        try:
            store.store_crop(
                session=session,
                crop_image=crop,
                app_process=app_process,
                page_class="",
                semantic_role=role,
                semantic_text=text,
                region_role="",  # Phase 1: provider_layers.json has no region_role field
                relative_bounds=rel_bounds,
                size_ratio=size_ratio,
                source="index_from_canvas",
                privacy_level=privacy,
                state="pending",
                confidence=float(layer.get("confidence", 0)),
                bounds_json=json.dumps(bounds),
            )
            indexed += 1
        except Exception as e:
            errors.append(f"{role}/{text}: {e}")

    if not dry_run:
        session.flush()

    return {
        "app": app_id,
        "app_process": app_process,
        "total_layers": len(layers),
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline icon memory indexing")
    parser.add_argument("--apps", default=",".join(DEFAULT_APPS),
                        help="Comma-separated app IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually store, just count")
    parser.add_argument("--db", default="data/openclaw.db",
                        help="SQLite database path")
    args = parser.parse_args()

    apps = [a.strip() for a in args.apps.split(",") if a.strip()]

    store = IconMemoryStore()
    results = []

    if args.dry_run:
        # dry-run: no DB, use mock session
        from unittest.mock import MagicMock
        _dry_store: dict = {}

        class _DryQ:
            def __init__(self, s):
                self._s = s; self._f = {}
            def filter_by(self, **kw):
                q = _DryQ(self._s); q._f = {**self._f, **kw}; return q
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
        session.add = lambda row: _dry_store.__setitem__(row.asset_id, row)
        session.query = MagicMock(side_effect=lambda m: _DryQ(_dry_store))
        session.flush = MagicMock()

        for app_id in apps:
            result = index_app(app_id, store, session, dry_run=True)
            results.append(result)
            print(f"  [DRY] {app_id}: {result['indexed']} indexed, "
                  f"{result['skipped']} skipped, {len(result.get('errors', []))} errors")
    else:
        # real run: init DB at specified path
        from src.storage.db import init_db, Session
        db = init_db(args.db)
        db.create_all()
        with Session() as session:
            for app_id in apps:
                result = index_app(app_id, store, session, dry_run=False)
                results.append(result)
                print(f"  [OK] {app_id}: {result['indexed']} indexed, "
                      f"{result['skipped']} skipped, {len(result.get('errors', []))} errors")

    # Summary
    total_indexed = sum(r["indexed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(len(r.get("errors", [])) for r in results)
    summary_msg = "Total: {} indexed, {} skipped, {} errors".format(total_indexed, total_skipped, total_errors)
    print(chr(10) + summary_msg)

    if not args.dry_run:
        from src.storage.db import Session
        with Session() as session:
            stats = store.stats(session)
            print("DB stats: {}".format(stats))


if __name__ == "__main__":
    main()
