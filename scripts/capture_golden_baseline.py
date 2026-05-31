"""
Phase 0: Golden Baseline 捕获脚本

为指定窗口生成完整的 golden baseline 文件，包括：
- 原始截图 (preview.png)
- InteractionCanvas JSON (snapshot.json)
- boundary_candidates (candidates.json)
- provider_trace / provider_health (provider.json)
- openclaw payload (openclaw_payload.json)
- candidate overlay 图 (candidate_overlay.png)
- 候选来源分布 (source_distribution.json)
- 关键元素清单 (key_elements.json)
- regions, elements, locators, anchors, scroll_contexts, evidence, vision, ocr

用法：
  python scripts/capture_golden_baseline.py --app notepad --hwnd 12345
  python scripts/capture_golden_baseline.py --app chrome --hwnd 67890
  python scripts/capture_golden_baseline.py --foreground --app notepad
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inspector.service import InspectorService
from src.perception.perception_service import PerceptionService
from src.perception.debug_tools import (
    save_debug_bundle,
    overlay_boundary_candidates,
)
from src.perception.page_compiler_candidates import (
    build_boundary_candidates,
    summarize_boundary_candidates,
)
from src.perception.openclaw_protocol import build_openclaw_payload


GOLDEN_DIR = Path("data/baselines/golden")


def capture_baseline(app_name: str, hwnd: int, output_dir: Path) -> dict:
    """Capture complete golden baseline for a window."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{app_name}_{timestamp}"

    print(f"[1/8] Capturing window {hwnd} ({app_name})...")

    # Use inspector service for the full pipeline
    inspector = InspectorService()
    result = inspector.inspect_window(
        hwnd=hwnd,
        output_dir=output_dir,
        include_ocr=True,
    )

    # Rename files to golden format
    bundle_name_old = result["bundle_name"]
    files = result["files"]

    # Rename all files from inspect_{hwnd}_* to {app_name}_{timestamp}_*
    renamed_files = {}
    for key, path_str in files.items():
        old_path = Path(path_str)
        if old_path.exists():
            new_name = old_path.name.replace(bundle_name_old, bundle_name)
            new_path = old_path.parent / new_name
            old_path.rename(new_path)
            renamed_files[key] = str(new_path)
        else:
            renamed_files[key] = path_str

    result["bundle_name"] = bundle_name
    result["files"] = renamed_files

    # Load snapshot for additional artifacts
    snapshot_path = renamed_files.get("snapshot")
    if snapshot_path and Path(snapshot_path).exists():
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)

        # [2/8] Build boundary candidates
        print("[2/8] Building boundary candidates...")
        from src.perception.page_compiler_models import InteractionCanvas
        # Re-load snapshot as dataclass for candidate building
        # We'll use the JSON data directly
        candidates = _build_candidates_from_json(snapshot_data, output_dir, bundle_name)

        # [3/8] Build openclaw payload
        print("[3/8] Building OpenClaw payload...")
        _build_openclaw_payload_from_json(snapshot_data, candidates, output_dir, bundle_name)

        # [4/8] Generate source distribution
        print("[4/8] Computing source distribution...")
        _compute_source_distribution(snapshot_data, candidates, output_dir, bundle_name)

        # [5/8] Extract key elements
        print("[5/8] Extracting key elements...")
        _extract_key_elements(snapshot_data, output_dir, bundle_name)

        # [6/8] Generate candidate overlay
        print("[6/8] Generating candidate overlay...")
        _generate_candidate_overlay(result, output_dir, bundle_name)

    # [7/8] Save summary
    print("[7/8] Saving summary...")
    summary = {
        "app_name": app_name,
        "hwnd": hwnd,
        "timestamp": timestamp,
        "bundle_name": bundle_name,
        "files": renamed_files,
        "summary": result["summary"],
        "captured_at": datetime.now().isoformat(),
    }

    summary_path = output_dir / f"{bundle_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # [8/8] Update capture summary index
    print("[8/8] Updating capture index...")
    _update_capture_index(app_name, summary, output_dir)

    print(f"\nDone! Baseline saved to {output_dir}/")
    print(f"Bundle: {bundle_name}")
    return summary


def _build_candidates_from_json(
    snapshot_data: dict, output_dir: Path, bundle_name: str
) -> list[dict]:
    """Build boundary candidates from snapshot JSON."""
    try:
        # Try to load as InteractionCanvas dataclass
        from src.perception.page_compiler_models import InteractionCanvas
        from dataclasses import asdict

        snapshot = InteractionCanvas(**snapshot_data)
        candidates = build_boundary_candidates(snapshot, max_candidates=120)
    except Exception as e:
        print(f"  Warning: Could not build candidates from dataclass: {e}")
        # Fallback: extract candidates from artifacts
        candidates = snapshot_data.get("artifacts", {}).get("boundary_candidates", [])

    candidates_path = output_dir / f"{bundle_name}_candidates.json"
    candidates_path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Also save summary
    if candidates:
        summary = summarize_boundary_candidates(candidates)
        summary_path = output_dir / f"{bundle_name}_candidate_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"  Candidates: {len(candidates)}")
    return candidates


def _build_openclaw_payload_from_json(
    snapshot_data: dict,
    candidates: list[dict],
    output_dir: Path,
    bundle_name: str,
):
    """Build OpenClaw payload from snapshot JSON."""
    try:
        from src.perception.page_compiler_models import InteractionCanvas

        snapshot = InteractionCanvas(**snapshot_data)
        payload = build_openclaw_payload(
            snapshot=snapshot,
            task="baseline capture",
            max_candidates=120,
        )
    except Exception as e:
        print(f"  Warning: Could not build payload: {e}")
        payload = {"error": str(e)}

    payload_path = output_dir / f"{bundle_name}_openclaw_payload.json"
    payload_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  OpenClaw payload saved")


def _compute_source_distribution(
    snapshot_data: dict,
    candidates: list[dict],
    output_dir: Path,
    bundle_name: str,
):
    """Compute source distribution from snapshot and candidates."""
    elements = snapshot_data.get("elements", [])

    # Source distribution from elements
    source_counts = Counter()
    for elem in elements:
        sources = elem.get("provider_sources", [])
        for src in sources:
            source_counts[src] += 1

    # Source distribution from candidates
    candidate_source_counts = Counter()
    for cand in candidates:
        src = cand.get("source", "unknown")
        if isinstance(src, list):
            for s in src:
                candidate_source_counts[s] += 1
        else:
            candidate_source_counts[src] += 1

    # Locator kind distribution
    locators = snapshot_data.get("locators", [])
    locator_kind_counts = Counter()
    for loc in locators:
        kind = loc.get("kind", "unknown")
        locator_kind_counts[kind] += 1

    distribution = {
        "element_source_distribution": dict(source_counts),
        "candidate_source_distribution": dict(candidate_source_counts),
        "locator_kind_distribution": dict(locator_kind_counts),
        "total_elements": len(elements),
        "total_candidates": len(candidates),
        "total_locators": len(locators),
    }

    dist_path = output_dir / f"{bundle_name}_source_distribution.json"
    dist_path.write_text(
        json.dumps(distribution, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Source distribution: {dict(source_counts)}")


def _extract_key_elements(
    snapshot_data: dict, output_dir: Path, bundle_name: str
):
    """Extract key elements with their semantic roles and bounds."""
    elements = snapshot_data.get("elements", [])

    key_elements = []
    for elem in elements:
        role = elem.get("semantic_role", "")
        if not role or role == "UNKNOWN":
            continue
        key_elements.append({
            "element_id": elem.get("element_id"),
            "semantic_role": role,
            "control_type": elem.get("control_type"),
            "name": elem.get("name"),
            "text": elem.get("text"),
            "bounds": elem.get("bounds"),
            "interactable": elem.get("interactable"),
            "provider_sources": elem.get("provider_sources"),
            "locator_count": len(elem.get("locator_ids", [])),
        })

    key_path = output_dir / f"{bundle_name}_key_elements.json"
    key_path.write_text(
        json.dumps(key_elements, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Key elements: {len(key_elements)}")


def _generate_candidate_overlay(
    result: dict, output_dir: Path, bundle_name: str
):
    """Generate candidate overlay image."""
    overlay_path = result["files"].get("overlay")
    if overlay_path and Path(overlay_path).exists():
        # The overlay already exists from debug_bundle
        # Just copy it as candidate_overlay
        import shutil
        candidate_overlay_path = output_dir / f"{bundle_name}_candidate_overlay.png"
        shutil.copy2(overlay_path, candidate_overlay_path)
        print(f"  Candidate overlay saved")


def _update_capture_index(app_name: str, summary: dict, output_dir: Path):
    """Update the capture summary index."""
    index_path = output_dir / "baseline_capture_summary.json"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}

    index[app_name] = summary

    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Capture golden baseline")
    parser.add_argument("--app", required=True, help="Application name")
    parser.add_argument("--hwnd", type=int, help="Window handle")
    parser.add_argument("--foreground", action="store_true", help="Use foreground window")
    parser.add_argument("--output", default=str(GOLDEN_DIR), help="Output directory")
    args = parser.parse_args()

    if args.foreground:
        from src.windows.window_enum import WindowEnumService
        svc = WindowEnumService()
        fg = svc.get_foreground_window()
        if fg is None:
            print("Error: No foreground window found")
            sys.exit(1)
        hwnd = fg.hwnd
        print(f"Using foreground window: {hwnd} ({fg.title})")
    elif args.hwnd:
        hwnd = args.hwnd
    else:
        print("Error: Must specify --hwnd or --foreground")
        sys.exit(1)

    capture_baseline(args.app, hwnd, Path(args.output))


if __name__ == "__main__":
    main()
