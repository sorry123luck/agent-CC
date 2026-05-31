"""Evaluate ROI VLM semantic quality from saved comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GENERIC_REGION_ROLES = {"chat_area", "unknown", "main_content", "tab", "action"}

MODE_REGION_OPTIONS = {
    "file_search": {"search_bar", "result_list", "file_list", "filter_bar", "status_bar", "pagination"},
    "local_transfer_dashboard": {"device_status", "send_receive_actions", "transfer_history", "settings"},
    "account_switcher": {"account_list", "login_panel", "login_actions", "window_controls", "account_switcher"},
    "browser_profile_manager": {"left_navigation", "profile_table", "profile_cards", "filter_bar", "pagination"},
    "remote_access_dashboard": {"connection_panel", "remote_control_actions", "status_or_recent", "toolbar"},
    "media_video_home": {"top_navigation_search", "media_feed", "video_card", "player_or_bottom_actions", "media_control"},
    "chat_workspace": {"navigation_list", "chat_area", "message_stream", "composer", "toolbar"},
    "chat_document": {"conversation_sidebar", "document_thread", "composer", "toolbar", "chat_area"},
    "list_management": {"left_navigation", "toolbar", "filter_bar", "table", "list", "pagination"},
    "control_dashboard": {"left_navigation", "mode_tabs", "dashboard", "status_panel", "card_grid"},
    "control_matrix": {"control_columns", "mixer_matrix", "master_section", "status_panel", "dashboard"},
    "security_dashboard": {"security_status", "feature_grid", "scan_actions", "bottom_actions", "status"},
    "archive_file_manager": {"menu_toolbar", "file_list", "status_bar", "path_bar", "toolbar", "table"},
}

MODE_ROLE_OPTIONS = {
    "file_search": {"action", "input", "file", "folder", "result", "filter", "pagination", "status", "unknown"},
    "local_transfer_dashboard": {"action", "device", "file", "transfer", "status", "toggle", "unknown"},
    "account_switcher": {"action", "account", "input", "login", "status", "toggle", "unknown"},
    "browser_profile_manager": {"action", "input", "profile", "row", "filter", "pagination", "status", "unknown"},
    "remote_access_dashboard": {"action", "input", "connection_id", "password", "status", "recent_item", "unknown"},
    "media_video_home": {"action", "input", "video", "video_card", "media_item", "media_control", "tab", "status", "unknown"},
    "chat_workspace": {"action", "input", "item", "message", "status", "tab", "unknown"},
    "chat_document": {"action", "input", "item", "message", "status", "copy", "unknown"},
    "list_management": {"action", "input", "item", "row", "filter", "pagination", "status", "unknown"},
    "control_dashboard": {"action", "toggle", "tab", "status", "item", "unknown"},
    "control_matrix": {"action", "toggle", "slider", "status", "item", "unknown"},
    "security_dashboard": {"action", "status", "security_feature", "toggle", "item", "unknown"},
    "archive_file_manager": {"action", "file", "folder", "item", "status", "unknown"},
}


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode") or row.get("new_mode") or "")
    supplements = list(row.get("new_supplements") or row.get("roi_vlm_semantic_supplements") or [])
    timeouts = list(row.get("new_timeouts") or row.get("roi_vlm_timeouts") or [])
    late_failures = list(row.get("new_late_failures") or row.get("roi_vlm_late_failures") or [])
    interactive_statuses = list(row.get("interactive_statuses") or [])
    allowed_regions = MODE_REGION_OPTIONS.get(mode, set())
    allowed_roles = MODE_ROLE_OPTIONS.get(mode, set())
    issues: list[str] = []
    region_roles: list[str] = []
    annotation_roles: list[str] = []
    annotation_count = 0

    for supplement in supplements:
        role = str((supplement.get("region_semantics") or {}).get("role") or "")
        if role:
            region_roles.append(role)
            if role in GENERIC_REGION_ROLES and mode not in {"chat_workspace", "chat_document"}:
                issues.append(f"generic_region_role:{role}")
            if allowed_regions and role not in allowed_regions:
                issues.append(f"invalid_region_role:{role}")
        for annotation in list(supplement.get("candidate_annotations") or []):
            ann_role = str(annotation.get("role") or "")
            annotation_count += 1
            if ann_role:
                annotation_roles.append(ann_role)
                if allowed_roles and ann_role not in allowed_roles:
                    issues.append(f"invalid_annotation_role:{ann_role}")

    generic_region_count = sum(1 for role in region_roles if role in GENERIC_REGION_ROLES and mode not in {"chat_workspace", "chat_document"})
    invalid_region_count = sum(1 for role in region_roles if allowed_regions and role not in allowed_regions)
    invalid_annotation_role_count = sum(1 for role in annotation_roles if allowed_roles and role not in allowed_roles)
    late_failure_count = len(late_failures)
    timeout_count = len(timeouts)
    initial_timeout_count = sum(1 for status in interactive_statuses if status == "timeout")

    score = 100
    score -= generic_region_count * 12
    score -= invalid_region_count * 10
    score -= invalid_annotation_role_count * 8
    score -= late_failure_count * 15
    score -= max(0, 3 - annotation_count) * 5
    score -= max(0, timeout_count - len(supplements)) * 2
    score = max(0, min(100, score))

    return {
        "sample": str(row.get("sample") or ""),
        "canvas_id": str(row.get("canvas_id") or ""),
        "mode": mode,
        "score": score,
        "region_roles": region_roles,
        "annotation_roles": annotation_roles,
        "annotation_count": annotation_count,
        "supplement_count": len(supplements),
        "timeout_count": timeout_count,
        "initial_timeout_count": initial_timeout_count,
        "late_failure_count": late_failure_count,
        "generic_region_count": generic_region_count,
        "invalid_region_count": invalid_region_count,
        "invalid_annotation_role_count": invalid_annotation_role_count,
        "issues": sorted(set(issues)),
    }


def evaluate_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [score_row(row) for row in list(data.get("rows") or [])]
    rows.sort(key=lambda item: (item["score"], item["sample"]))
    return {
        "source": str(path),
        "sample_count": len(rows),
        "average_score": round(sum(row["score"] for row in rows) / len(rows), 2) if rows else 0,
        "low_score_count": sum(1 for row in rows if row["score"] < 80),
        "generic_region_total": sum(row["generic_region_count"] for row in rows),
        "invalid_region_total": sum(row["invalid_region_count"] for row in rows),
        "invalid_annotation_role_total": sum(row["invalid_annotation_role_count"] for row in rows),
        "late_failure_total": sum(row["late_failure_count"] for row in rows),
        "rows": rows,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# ROI VLM Quality Summary",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Average score: {summary['average_score']}",
        f"- Low score count: {summary['low_score_count']}",
        f"- Generic region total: {summary['generic_region_total']}",
        f"- Invalid region total: {summary['invalid_region_total']}",
        f"- Invalid annotation role total: {summary['invalid_annotation_role_total']}",
        f"- Late failure total: {summary['late_failure_total']}",
        "",
        "| sample | mode | score | annotations | regions | issues |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {sample} | {mode} | {score} | {annotations} | {regions} | {issues} |".format(
                sample=row["sample"],
                mode=row["mode"],
                score=row["score"],
                annotations=row["annotation_count"],
                regions=",".join(row["region_roles"]),
                issues=",".join(row["issues"]),
            )
        )
    (output_dir / "quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/roi_vlm_quality_eval"))
    args = parser.parse_args()
    summary = evaluate_summary(args.summary)
    write_outputs(summary, args.output_dir)
    print(
        f"samples={summary['sample_count']} average_score={summary['average_score']} "
        f"low_score={summary['low_score_count']} generic={summary['generic_region_total']} "
        f"invalid_region={summary['invalid_region_total']} invalid_annotation={summary['invalid_annotation_role_total']} "
        f"late_failures={summary['late_failure_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
