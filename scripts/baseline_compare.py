"""
Phase 0: Golden Baseline 对比脚本

对比重构后的 InteractionCanvas 输出与 golden baseline 的差异。
每个 Phase 完成后运行此脚本验证无能力丢失。

用法：
  python scripts/baseline_compare.py --app notepad
  python scripts/baseline_compare.py --app all
  python scripts/baseline_compare.py --app notepad --new-dir data/baselines/new
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

GOLDEN_DIR = Path("data/baselines/golden")
KEY_ELEMENTS_PATH = Path("data/baselines/golden/key_elements_checklist.json")


@dataclass
class CompareResult:
    app_name: str
    passed: bool
    checks: dict
    details: list[str]

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "passed": self.passed,
            "checks": self.checks,
            "details": self.details,
        }


def find_latest_bundle(app_name: str, search_dir: Path) -> str | None:
    """Find the latest bundle prefix for an app in a directory."""
    bundles = set()
    for f in search_dir.glob(f"{app_name}_*"):
        name = f.stem
        # Extract bundle prefix (app_YYYYMMDD_HHMMSS)
        # Date is 8 digits, time is 6 digits, they may follow multi-part app names
        parts = name.split("_")
        for i, p in enumerate(parts):
            if len(p) == 8 and p.isdigit() and i + 1 < len(parts):
                # Found date, check if next part is time (6 digits)
                if len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                    candidate = "_".join(parts[: i + 2])
                    bundles.add(candidate)
                    break
    if not bundles:
        return None
    return sorted(bundles)[-1]


def load_json(path: Path) -> dict | list | None:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_element_counts(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare element counts between golden and new snapshot."""
    golden_elements = golden_snapshot.get("elements", [])
    new_elements = new_snapshot.get("elements", [])

    golden_count = len(golden_elements)
    new_count = len(new_elements)
    delta = abs(golden_count - new_count)
    threshold = max(3, int(golden_count * 0.10))  # 10% or at least 3

    passed = delta <= threshold
    detail = (
        f"Element count: golden={golden_count}, new={new_count}, "
        f"delta={delta}, threshold={threshold}"
    )
    return passed, detail, {
        "golden_count": golden_count,
        "new_count": new_count,
        "delta": delta,
        "threshold": threshold,
    }


def compare_region_counts(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare region counts."""
    golden_regions = golden_snapshot.get("regions", [])
    new_regions = new_snapshot.get("regions", [])

    golden_count = len(golden_regions)
    new_count = len(new_regions)
    delta = abs(golden_count - new_count)
    threshold = max(1, int(golden_count * 0.15))  # 15% or at least 1

    passed = delta <= threshold
    detail = (
        f"Region count: golden={golden_count}, new={new_count}, "
        f"delta={delta}, threshold={threshold}"
    )
    return passed, detail, {
        "golden_count": golden_count,
        "new_count": new_count,
        "delta": delta,
        "threshold": threshold,
    }


def compare_coordinate_ranges(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare coordinate ranges of elements."""
    golden_elements = golden_snapshot.get("elements", [])
    new_elements = new_snapshot.get("elements", [])

    def get_bounds(elements):
        bounds_list = []
        for e in elements:
            b = e.get("bounds", {})
            if b:
                bounds_list.append(b)
        return bounds_list

    golden_bounds = get_bounds(golden_elements)
    new_bounds = get_bounds(new_elements)

    if not golden_bounds or not new_bounds:
        return True, "No bounds to compare", {}

    def bounds_range(bounds_list):
        # bounds can be dict {"left","top","right","bottom"} or list [left,top,right,bottom]
        lefts, tops, rights, bottoms = [], [], [], []
        for b in bounds_list:
            if isinstance(b, dict):
                lefts.append(b.get("left", 0))
                tops.append(b.get("top", 0))
                rights.append(b.get("right", 0))
                bottoms.append(b.get("bottom", 0))
            elif isinstance(b, (list, tuple)) and len(b) >= 4:
                lefts.append(b[0])
                tops.append(b[1])
                rights.append(b[2])
                bottoms.append(b[3])
        return {
            "min_left": min(lefts) if lefts else 0,
            "max_right": max(rights) if rights else 0,
            "min_top": min(tops) if tops else 0,
            "max_bottom": max(bottoms) if bottoms else 0,
        }

    golden_range = bounds_range(golden_bounds)
    new_range = bounds_range(new_bounds)

    # Check if ranges are similar (within 50px tolerance)
    tolerance = 50
    left_ok = abs(golden_range["min_left"] - new_range["min_left"]) <= tolerance
    top_ok = abs(golden_range["min_top"] - new_range["min_top"]) <= tolerance
    right_ok = abs(golden_range["max_right"] - new_range["max_right"]) <= tolerance
    bottom_ok = abs(golden_range["max_bottom"] - new_range["max_bottom"]) <= tolerance

    passed = left_ok and top_ok and right_ok and bottom_ok
    detail = (
        f"Coordinate range: golden={golden_range}, new={new_range}, "
        f"tolerance={tolerance}"
    )
    return passed, detail, {"golden": golden_range, "new": new_range}


def compare_source_distribution(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare source distribution."""
    golden_elements = golden_snapshot.get("elements", [])
    new_elements = new_snapshot.get("elements", [])

    def source_counts(elements):
        counts = Counter()
        for e in elements:
            for src in e.get("provider_sources", []):
                counts[src] += 1
        return dict(counts)

    golden_sources = source_counts(golden_elements)
    new_sources = source_counts(new_elements)

    # Check that all golden sources are present in new
    missing_sources = set(golden_sources.keys()) - set(new_sources.keys())
    passed = len(missing_sources) == 0
    detail = (
        f"Source distribution: golden={golden_sources}, new={new_sources}, "
        f"missing={missing_sources}"
    )
    return passed, detail, {
        "golden": golden_sources,
        "new": new_sources,
        "missing": list(missing_sources),
    }


def compare_key_elements(
    golden_snapshot: dict, new_snapshot: dict, app_name: str
) -> tuple[bool, str, dict]:
    """Compare key elements (semantic roles) between golden and new."""
    golden_elements = golden_snapshot.get("elements", [])
    new_elements = new_snapshot.get("elements", [])

    # Load key elements checklist
    checklist = load_json(KEY_ELEMENTS_PATH) or {}
    app_checklist = checklist.get(app_name, {})
    required_roles = set(app_checklist.get("required_roles", []))

    # If no checklist, use all non-empty semantic roles from golden
    if not required_roles:
        required_roles = {
            e.get("semantic_role")
            for e in golden_elements
            if e.get("semantic_role") and e.get("semantic_role") != "UNKNOWN"
        }

    new_roles = {
        e.get("semantic_role")
        for e in new_elements
        if e.get("semantic_role") and e.get("semantic_role") != "UNKNOWN"
    }

    missing_roles = required_roles - new_roles
    passed = len(missing_roles) == 0
    detail = (
        f"Key elements: required={sorted(required_roles)}, "
        f"missing={sorted(missing_roles)}"
    )
    return passed, detail, {
        "required_roles": sorted(required_roles),
        "new_roles": sorted(new_roles),
        "missing_roles": sorted(missing_roles),
    }


def compare_provider_health(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare provider trace."""
    golden_trace = golden_snapshot.get("provider_trace", {})
    new_trace = new_snapshot.get("provider_trace", {})

    golden_details = golden_trace.get("provider_details", {})
    new_details = new_trace.get("provider_details", {})

    # Check that providers used in golden are still used in new
    golden_used = set()
    for key, val in golden_details.items():
        if isinstance(val, dict) and val.get("used", False):
            golden_used.add(key)
        elif isinstance(val, bool) and val:
            golden_used.add(key)

    new_used = set()
    for key, val in new_details.items():
        if isinstance(val, dict) and val.get("used", False):
            new_used.add(key)
        elif isinstance(val, bool) and val:
            new_used.add(key)

    # Also check top-level flags
    for flag in ["uia_used", "ocr_used", "vision_used", "dom_used"]:
        if golden_trace.get(flag, False):
            golden_used.add(flag.replace("_used", ""))
        if new_trace.get(flag, False):
            new_used.add(flag.replace("_used", ""))

    missing = golden_used - new_used
    passed = len(missing) == 0
    detail = (
        f"Provider health: golden_used={sorted(golden_used)}, "
        f"new_used={sorted(new_used)}, missing={sorted(missing)}"
    )
    return passed, detail, {
        "golden_used": sorted(golden_used),
        "new_used": sorted(new_used),
        "missing": sorted(missing),
    }


def compare_locator_counts(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare locator counts."""
    golden_locators = golden_snapshot.get("locators", [])
    new_locators = new_snapshot.get("locators", [])

    golden_count = len(golden_locators)
    new_count = len(new_locators)
    delta = abs(golden_count - new_count)
    threshold = max(5, int(golden_count * 0.15))  # 15% or at least 5

    passed = delta <= threshold
    detail = (
        f"Locator count: golden={golden_count}, new={new_count}, "
        f"delta={delta}, threshold={threshold}"
    )
    return passed, detail, {
        "golden_count": golden_count,
        "new_count": new_count,
        "delta": delta,
        "threshold": threshold,
    }


def compare_anchor_counts(
    golden_snapshot: dict, new_snapshot: dict
) -> tuple[bool, str, dict]:
    """Compare anchor counts."""
    golden_anchors = golden_snapshot.get("anchors", [])
    new_anchors = new_snapshot.get("anchors", [])

    golden_count = len(golden_anchors)
    new_count = len(new_anchors)
    delta = abs(golden_count - new_count)
    threshold = max(2, int(golden_count * 0.20))  # 20% or at least 2

    passed = delta <= threshold
    detail = (
        f"Anchor count: golden={golden_count}, new={new_count}, "
        f"delta={delta}, threshold={threshold}"
    )
    return passed, detail, {
        "golden_count": golden_count,
        "new_count": new_count,
        "delta": delta,
        "threshold": threshold,
    }


def compare_app(
    app_name: str, golden_dir: Path, new_dir: Path | None = None
) -> CompareResult:
    """Run all comparisons for one app."""
    if new_dir is None:
        new_dir = golden_dir

    # Find golden bundle
    golden_bundle = find_latest_bundle(app_name, golden_dir)
    if not golden_bundle:
        return CompareResult(
            app_name=app_name,
            passed=False,
            checks={},
            details=[f"No golden baseline found for {app_name}"],
        )

    # Find new bundle
    new_bundle = find_latest_bundle(app_name, new_dir)
    if not new_bundle:
        return CompareResult(
            app_name=app_name,
            passed=False,
            checks={},
            details=[f"No new baseline found for {app_name}"],
        )

    # Load snapshots
    golden_snapshot = load_json(golden_dir / f"{golden_bundle}_snapshot.json")
    new_snapshot = load_json(new_dir / f"{new_bundle}_snapshot.json")

    if not golden_snapshot:
        return CompareResult(
            app_name=app_name,
            passed=False,
            checks={},
            details=[f"Golden snapshot not found: {golden_bundle}"],
        )
    if not new_snapshot:
        return CompareResult(
            app_name=app_name,
            passed=False,
            checks={},
            details=[f"New snapshot not found: {new_bundle}"],
        )

    # Run all checks
    checks = {}
    all_details = []
    all_passed = True

    for check_name, check_fn in [
        ("element_count", lambda: compare_element_counts(golden_snapshot, new_snapshot)),
        ("region_count", lambda: compare_region_counts(golden_snapshot, new_snapshot)),
        ("coordinate_range", lambda: compare_coordinate_ranges(golden_snapshot, new_snapshot)),
        ("source_distribution", lambda: compare_source_distribution(golden_snapshot, new_snapshot)),
        ("key_elements", lambda: compare_key_elements(golden_snapshot, new_snapshot, app_name)),
        ("provider_health", lambda: compare_provider_health(golden_snapshot, new_snapshot)),
        ("locator_count", lambda: compare_locator_counts(golden_snapshot, new_snapshot)),
        ("anchor_count", lambda: compare_anchor_counts(golden_snapshot, new_snapshot)),
    ]:
        passed, detail, data = check_fn()
        checks[check_name] = {"passed": passed, "detail": detail, "data": data}
        all_details.append(f"  {'PASS' if passed else 'FAIL'} {detail}")
        if not passed:
            all_passed = False

    return CompareResult(
        app_name=app_name,
        passed=all_passed,
        checks=checks,
        details=all_details,
    )


def main():
    parser = argparse.ArgumentParser(description="Golden baseline comparison")
    parser.add_argument(
        "--app",
        required=True,
        help="Application name or 'all'",
    )
    parser.add_argument(
        "--golden-dir",
        default=str(GOLDEN_DIR),
        help="Golden baseline directory",
    )
    parser.add_argument(
        "--new-dir",
        default=None,
        help="New baseline directory (default: same as golden)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    new_dir = Path(args.new_dir) if args.new_dir else None

    if args.app == "all":
        # Find all apps in golden dir
        # Filename format: appname_YYYYMMDD_HHMMSS_snapshot.json
        # Date is always 8 digits, time is always 6 digits
        apps = set()
        for f in golden_dir.glob("*_snapshot.json"):
            parts = f.stem.split("_")
            # Find the date part (8 consecutive digits)
            date_idx = None
            for i, p in enumerate(parts):
                if len(p) == 8 and p.isdigit():
                    date_idx = i
                    break
            if date_idx is not None and date_idx > 0:
                app = "_".join(parts[:date_idx])
                apps.add(app)
        apps = sorted(apps)
    else:
        apps = [args.app]

    results = []
    all_passed = True

    for app_name in apps:
        result = compare_app(app_name, golden_dir, new_dir)
        results.append(result)
        if not result.passed:
            all_passed = False

    if args.json:
        output = {
            "all_passed": all_passed,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Golden Baseline Comparison Report")
        print(f"{'='*60}\n")

        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.app_name}")
            for detail in result.details:
                print(detail)
            print()

        print(f"{'='*60}")
        print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
