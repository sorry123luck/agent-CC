"""Evaluate visual mode, quality, and ROI planning across saved samples.

This is an offline regression tool for the region-first VLM design:
local code proposes coarse regions/candidates, while VLM may only add
semantic supplements against those local ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.perception_quality import PerceptionQualityEvaluator
from src.perception.roi_selection import RoiSelectionPlanner
from src.perception.visual_pattern import VisualPatternClassifier


EXPECTED_MODE_HINTS = [
    (("wechat", "微信"), "chat_workspace"),
    (("chatgpt",), "chat_document"),
    (("adspower",), "list_management"),
    (("bitbrowser", "比特"), "list_management"),
    (("cloudmusic", "网易云", "netmusic"), "media_home"),
    (("flclash",), "control_dashboard"),
    (("voicemeeter",), "control_matrix"),
    (("winrar",), "archive_file_manager"),
    (("hipsmain", "huorong", "火绒"), "security_dashboard"),
    (("bilibili", "哔哩"), "video_feed_home"),
    (("baidunetdisk", "百度网盘"), "loading_state"),
    (("feishu", "lark"), "collaboration_inbox"),
    (("everything",), "file_search"),
    (("localsend",), "local_transfer_dashboard"),
    (("qq",), "account_switcher"),
    (("hubstudio",), "browser_profile_manager"),
    (("teamviewer",), "remote_access_dashboard"),
    (("qyclient", "iqiyi", "爱奇艺"), "media_video_home"),
]

EXPECTED_REQUIRED_ROI_PURPOSES = {
    "chat_workspace": ["navigation_and_list", "message_stream", "composer"],
    "collaboration_inbox": ["app_rail", "inbox_list", "message_thread", "composer"],
    "list_management": ["left_navigation", "toolbar_search_filters", "table_or_list"],
    "media_home": ["left_navigation", "content_feed", "bottom_player"],
    "video_feed_home": ["left_navigation", "top_tabs_search", "representative_card_grid"],
    "control_matrix": ["left_control_columns", "middle_control_columns", "right_master_section"],
    "control_dashboard": ["left_navigation", "mode_tabs", "dashboard_or_cards"],
    "chat_document": ["conversation_sidebar", "document_thread", "composer"],
    "security_dashboard": ["security_status", "feature_grid"],
    "archive_file_manager": ["menu_toolbar", "file_list", "status_bar"],
    "file_search": ["search_bar", "result_list", "status_or_filters"],
    "local_transfer_dashboard": ["device_status", "send_receive_actions", "transfer_history"],
    "account_switcher": ["account_list", "login_actions", "window_controls"],
    "browser_profile_manager": ["left_navigation", "profile_table_or_cards", "toolbar_search_filters"],
    "remote_access_dashboard": ["connection_panel", "remote_control_actions", "status_or_recent"],
    "media_video_home": ["top_navigation_search", "media_feed", "player_or_bottom_actions"],
}

SKIP_NAME_PARTS = (
    "observe_summary",
    "roi_selection_summary",
    "latency_summary",
    "background_enhancement_summary",
    "recognition_comparison_summary",
    "summary.json",
)

SKIP_AGGREGATE_SUFFIXES = (
    "_compare.json",
    ".compare.json",
    "_results.json",
    "_probe.json",
)


def expected_mode_for_sample(source_name: str, page_class: str = "", window_title: str = "") -> str:
    """Return human expected coarse mode from stable sample metadata."""
    blob = f"{source_name} {page_class} {window_title}".lower()
    for hints, mode in EXPECTED_MODE_HINTS:
        if any(hint.lower() in blob for hint in hints):
            return mode
    return ""


def canvas_from_sample(sample: dict[str, Any], *, source_name: str) -> SimpleNamespace:
    """Convert a saved API/detail JSON dict to a planner-compatible object."""
    page_class = str(sample.get("page_class") or "")
    window_title = str(sample.get("window_title") or sample.get("title") or "")
    process_name = _process_name_from_metadata(page_class, window_title, source_name)
    width = int(sample.get("screenshot_width") or sample.get("window_width") or 0)
    height = int(sample.get("screenshot_height") or sample.get("window_height") or 0)
    if (width <= 0 or height <= 0) and isinstance(sample.get("artifacts"), dict):
        size = sample["artifacts"].get("screenshot_size")
        if isinstance(size, list) and len(size) == 2:
            width, height = int(size[0]), int(size[1])

    artifacts = dict(sample.get("artifacts") or {})
    if width > 0 and height > 0:
        artifacts["screenshot_size"] = [width, height]
    if "geometric_regions" not in artifacts:
        artifacts["geometric_regions"] = list(sample.get("geometric_regions") or [])

    elements = [_element_from_dict(raw) for raw in list(sample.get("elements") or [])]
    regions = [_region_from_dict(raw) for raw in list(sample.get("regions") or [])]

    return SimpleNamespace(
        canvas_id=str(sample.get("canvas_id") or Path(source_name).stem),
        app=SimpleNamespace(process_name=process_name),
        page=SimpleNamespace(page_class=page_class),
        page_class=page_class,
        window=SimpleNamespace(rect_client=(0, 0, width, height), title=window_title),
        artifacts=artifacts,
        elements=elements,
        regions=regions,
    )


def evaluate_sample(path: Path) -> dict[str, Any]:
    sample = _read_json(path)
    canvas = canvas_from_sample(sample, source_name=path.name)
    pattern = VisualPatternClassifier().classify(canvas)
    quality = PerceptionQualityEvaluator().evaluate(canvas)
    roi_plan = RoiSelectionPlanner().plan(canvas)
    expected_mode = expected_mode_for_sample(
        path.name,
        getattr(canvas, "page_class", ""),
        getattr(getattr(canvas, "window", None), "title", ""),
    )
    roi_dicts = [roi.to_dict() for roi in roi_plan.rois]
    expected_required_purposes = EXPECTED_REQUIRED_ROI_PURPOSES.get(expected_mode or pattern.mode, [])
    roi_quality = _roi_quality_metrics(canvas, roi_dicts, expected_required_purposes)
    vlm_effect = _roi_vlm_effect_metrics(canvas)
    return {
        "sample": path.name,
        "path": str(path),
        "page_class": canvas.page_class,
        "window_title": canvas.window.title,
        "expected_mode": expected_mode,
        "predicted_mode": pattern.mode,
        "mode_match": bool(expected_mode and expected_mode == pattern.mode),
        "mode_confidence": pattern.confidence,
        "mode_evidence": pattern.evidence,
        "usable_state": quality.usable_state,
        "quality_warnings": quality.warnings,
        "element_count": quality.element_count,
        "region_count": quality.region_count,
        "geometric_region_count": quality.geometric_region_count,
        "unknown_role_ratio": quality.unknown_role_ratio,
        "root_shell_like_count": quality.root_shell_like_count,
        "roi_count": len(roi_dicts),
        "roi_purposes": [roi["purpose"] for roi in roi_dicts],
        "roi_sources": sorted({roi["source"] for roi in roi_dicts}),
        "roi_bounds": [roi["bounds"] for roi in roi_dicts],
        "expected_required_purposes": expected_required_purposes,
        "missing_required_purposes": roi_quality["missing_required_purposes"],
        "roi_coverage_ratio": roi_quality["roi_coverage_ratio"],
        "roi_candidate_coverage_ratio": roi_quality["roi_candidate_coverage_ratio"],
        "roi_overlap_ratio": roi_quality["roi_overlap_ratio"],
        "roi_vlm_status": vlm_effect["status"],
        "roi_vlm_supplement_count": vlm_effect["supplement_count"],
        "roi_vlm_annotation_count": vlm_effect["annotation_count"],
        "roi_vlm_timeout_count": vlm_effect["timeout_count"],
        "roi_vlm_late_failure_count": vlm_effect["late_failure_count"],
        "roi_vlm_rejected_count": vlm_effect["rejected_count"],
        "skipped_reason": roi_plan.skipped_reason,
    }


def collect_sample_paths(root: Path) -> list[Path]:
    """Collect saved sample JSON files, excluding observe summaries and aggregate files."""
    paths: list[Path] = []
    for path in root.rglob("*.json"):
        name = path.name.lower()
        if any(part in name for part in SKIP_NAME_PARTS):
            continue
        if name == "results.json" or any(name.endswith(suffix) for suffix in SKIP_AGGREGATE_SUFFIXES):
            continue
        if name.endswith(".observe.json") or name.endswith(".semantic.json"):
            continue
        paths.append(path)
    return sorted(paths)


def evaluate_paths(paths: list[Path]) -> dict[str, Any]:
    rows = []
    errors = []
    for path in paths:
        try:
            rows.append(evaluate_sample(path))
        except Exception as exc:  # noqa: BLE001 - diagnostics should preserve failures.
            errors.append({"path": str(path), "error": repr(exc)})

    expected_rows = [row for row in rows if row["expected_mode"]]
    mismatches = [
        row for row in expected_rows
        if row["predicted_mode"] != row["expected_mode"]
    ]
    return {
        "sample_count": len(rows),
        "expected_count": len(expected_rows),
        "mode_match_count": len(expected_rows) - len(mismatches),
        "mode_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "rows": rows,
        "errors": errors,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# ROI Plan Sample Evaluation",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Expected labels: {summary['expected_count']}",
        f"- Mode matches: {summary['mode_match_count']}",
        f"- Mode mismatches: {summary['mode_mismatch_count']}",
        "",
        "| sample | expected | predicted | usable | warnings | roi_count | coverage | candidate_coverage | missing_required | vlm_status | vlm_annotations | roi_purposes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in summary["rows"]:
        warnings = ",".join(row["quality_warnings"])
        purposes = ",".join(row["roi_purposes"])
        missing = ",".join(row["missing_required_purposes"])
        lines.append(
            "| {sample} | {expected} | {predicted} | {usable} | {warnings} | {roi_count} | {coverage:.2f} | {candidate_coverage:.2f} | {missing} | {vlm_status} | {vlm_annotations} | {purposes} |".format(
                sample=row["sample"],
                expected=row["expected_mode"],
                predicted=row["predicted_mode"],
                usable=row["usable_state"],
                warnings=warnings,
                roi_count=row["roi_count"],
                coverage=float(row["roi_coverage_ratio"]),
                candidate_coverage=float(row["roi_candidate_coverage_ratio"]),
                missing=missing,
                vlm_status=row["roi_vlm_status"],
                vlm_annotations=row["roi_vlm_annotation_count"],
                purposes=purposes,
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _roi_quality_metrics(
    canvas: SimpleNamespace,
    roi_dicts: list[dict[str, Any]],
    expected_required_purposes: list[str],
) -> dict[str, Any]:
    width, height = _canvas_size(canvas)
    bounds = [_valid_bounds(roi.get("bounds"), width, height) for roi in roi_dicts]
    bounds = [item for item in bounds if item]
    canvas_area = max(1, width * height)
    union_area = _rect_union_area(bounds)
    raw_area = sum(max(0, r - l) * max(0, b - t) for l, t, r, b in bounds)
    roi_purposes = {str(roi.get("purpose") or "") for roi in roi_dicts}
    candidate_ids = {
        str(getattr(element, "element_id", ""))
        for element in list(getattr(canvas, "elements", []) or [])
        if str(getattr(element, "element_id", ""))
    }
    covered_candidate_ids = {
        candidate_id
        for roi in roi_dicts
        for candidate_id in list(roi.get("candidate_ids") or [])
        if str(candidate_id)
    }
    return {
        "missing_required_purposes": [
            purpose for purpose in expected_required_purposes
            if purpose not in roi_purposes
        ],
        "roi_coverage_ratio": round(min(1.0, union_area / canvas_area), 4),
        "roi_overlap_ratio": round(max(0.0, (raw_area - union_area) / max(1, raw_area)), 4),
        "roi_candidate_coverage_ratio": round(
            len(candidate_ids & covered_candidate_ids) / max(1, len(candidate_ids)),
            4,
        ),
    }


def _roi_vlm_effect_metrics(canvas: SimpleNamespace) -> dict[str, Any]:
    artifacts = getattr(canvas, "artifacts", None) or {}
    supplements = list(artifacts.get("roi_vlm_semantic_supplements") or [])
    timeouts = list(artifacts.get("roi_vlm_timeouts") or [])
    late_failures = list(artifacts.get("roi_vlm_late_failures") or [])
    rejected = list(artifacts.get("roi_vlm_rejected_responses") or [])
    annotation_count = sum(
        len(list(item.get("candidate_annotations") or []))
        for item in supplements
        if isinstance(item, dict)
    )
    if supplements and (timeouts or late_failures):
        status = "supplemented_with_timeout"
    elif supplements:
        status = "supplemented"
    elif timeouts or late_failures:
        status = "timeout_or_failed"
    elif rejected:
        status = "rejected_only"
    else:
        status = "not_run"
    return {
        "status": status,
        "supplement_count": len(supplements),
        "annotation_count": annotation_count,
        "timeout_count": len(timeouts),
        "late_failure_count": len(late_failures),
        "rejected_count": len(rejected),
    }


def _canvas_size(canvas: SimpleNamespace) -> tuple[int, int]:
    artifacts = getattr(canvas, "artifacts", None) or {}
    size = artifacts.get("screenshot_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return max(0, int(size[0])), max(0, int(size[1]))
    rect = getattr(getattr(canvas, "window", None), "rect_client", None)
    if rect and len(rect) == 4:
        return max(0, int(rect[2]) - int(rect[0])), max(0, int(rect[3]) - int(rect[1]))
    return 0, 0


def _valid_bounds(raw_bounds: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
        return None
    left, top, right, bottom = [int(value) for value in raw_bounds]
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rect_union_area(rects: list[tuple[int, int, int, int]]) -> int:
    if not rects:
        return 0
    xs = sorted({x for left, _, right, _ in rects for x in (left, right)})
    area = 0
    for x1, x2 in zip(xs, xs[1:]):
        if x2 <= x1:
            continue
        intervals = sorted(
            (top, bottom)
            for left, top, right, bottom in rects
            if left < x2 and right > x1
        )
        covered_y = 0
        current_top: int | None = None
        current_bottom = 0
        for top, bottom in intervals:
            if current_top is None:
                current_top, current_bottom = top, bottom
            elif top <= current_bottom:
                current_bottom = max(current_bottom, bottom)
            else:
                covered_y += current_bottom - current_top
                current_top, current_bottom = top, bottom
        if current_top is not None:
            covered_y += current_bottom - current_top
        area += (x2 - x1) * covered_y
    return area


def _element_from_dict(raw: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        element_id=str(raw.get("element_id") or raw.get("id") or ""),
        semantic_role=str(raw.get("semantic_role") or raw.get("role") or "unknown"),
        bounds=tuple(raw.get("bounds") or ()),
        text=str(raw.get("text") or ""),
        name=str(raw.get("name") or ""),
        control_type=str(raw.get("control_type") or ""),
        provider_sources=list(raw.get("provider_sources") or []),
    )


def _region_from_dict(raw: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        region_id=str(raw.get("region_id") or raw.get("id") or ""),
        role=str(raw.get("role") or ""),
        bounds=tuple(raw.get("bounds") or ()),
        element_ids=list(raw.get("element_ids") or []),
    )


def _process_name_from_metadata(page_class: str, window_title: str, source_name: str) -> str:
    blob = f"{page_class} {window_title} {source_name}".lower()
    for hints, mode in EXPECTED_MODE_HINTS:
        if any(hint.lower() in blob for hint in hints):
            return hints[0]
    return page_class.split("/", 1)[0].split(" ", 1)[0] or Path(source_name).stem.lower()


def _read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/roi_plan_eval_2026-05-20"))
    args = parser.parse_args()

    paths = collect_sample_paths(args.samples_root)
    summary = evaluate_paths(paths)
    write_outputs(summary, args.output_dir)
    print(
        f"evaluated={summary['sample_count']} expected={summary['expected_count']} "
        f"matches={summary['mode_match_count']} mismatches={summary['mode_mismatch_count']} "
        f"errors={len(summary['errors'])}"
    )
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
