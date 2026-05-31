"""Build a read-only report for controlled chat input/send probes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


APP_TITLE_HINTS = {
    "wechat": ("微信", "weixin", "wechat"),
    "weixin": ("微信", "weixin", "wechat"),
    "feishu": ("飞书", "feishu", "lark"),
    "lark": ("飞书", "feishu", "lark"),
    "qq": ("QQ", "qq"),
}


def build_probe_report(
    *,
    actions: list[dict[str, Any]],
    details: list[dict[str, Any]],
    visual_reviews: list[dict[str, Any]] | None = None,
    readback_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact report from send action logs and post-action canvas details."""
    rows = [
        _evaluate_action(
            action,
            details,
            visual_reviews=visual_reviews or [],
            readback_results=readback_results or [],
        )
        for action in _latest_actions_by_app(actions)
    ]
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "counts": counts,
        "rows": rows,
    }


def _latest_actions_by_app(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        app = str(action.get("app") or "").strip().lower()
        if not app:
            app = f"action_{len(order)}"
        if app not in latest:
            order.append(app)
        latest[app] = action
    return [latest[key] for key in order]


def analyze_probe_dirs(
    *,
    action_dir: Path,
    after_dir: Path,
    output_dir: Path | None = None,
    visual_review_file: Path | None = None,
) -> dict[str, Any]:
    """Load probe actions and post-action details from directories and write a report."""
    actions = _load_actions(action_dir)
    details = _load_details(after_dir)
    visual_reviews = _load_visual_reviews(
        visual_review_file
        or _first_existing_path(
            action_dir / "chat_send_visual_review.json",
            after_dir / "chat_send_visual_review.json",
        )
    )
    readback_results = _load_readback_results(
        _first_existing_path(
            action_dir / "chat_readback_report.json",
            after_dir / "chat_readback_report.json",
            (output_dir or after_dir) / "chat_readback_report.json",
        )
    )
    report = build_probe_report(
        actions=actions,
        details=details,
        visual_reviews=visual_reviews,
        readback_results=readback_results,
    )
    write_report(report, output_dir or after_dir)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat_send_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Send Probe Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| app | status | machine observed | visual observed | send target | failures | warnings | text |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {app} | {status} | {observed} | {visual_observed} | {target} | {failures} | {warnings} | {text} |".format(
                app=str(row.get("app") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                observed="yes" if row.get("text_observed_after") else "no",
                visual_observed="yes" if row.get("visual_observed_after") else "no",
                target=str(row.get("send_target_bounds") or "").replace("|", "/"),
                failures=",".join(row.get("failures") or []),
                warnings=",".join(row.get("warnings") or []),
                text=str(row.get("text") or "").replace("|", "/"),
            )
        )
    (output_dir / "chat_send_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_action(
    action: dict[str, Any],
    details: list[dict[str, Any]],
    *,
    visual_reviews: list[dict[str, Any]],
    readback_results: list[dict[str, Any]],
) -> dict[str, Any]:
    app = str(action.get("app") or "")
    raw_text = action.get("text")
    text = str(raw_text or "")
    send_bounds = _rect(action.get("send_bounds"))
    width = send_bounds[2] - send_bounds[0] if send_bounds else 0
    height = send_bounds[3] - send_bounds[1] if send_bounds else 0
    detail = _detail_for_action(app, details, text=text)
    detail_observed = bool(text) and _text_in_detail(text, detail or {})
    readback_result = _readback_for_action(app, text, readback_results)
    readback_observed = bool(readback_result)
    observed = detail_observed or readback_observed
    visual_review = _visual_review_for_action(app, text, visual_reviews)
    visual_observed = visual_review.get("observed_after") is True if visual_review else False
    failures: list[str] = []
    warnings: list[str] = []
    if action.get("ok") is False:
        failures.append("probe_action_failed")
    elif "ok" not in action:
        warnings.append("probe_action_status_missing")
    if raw_text is None:
        warnings.append("probe_expected_text_missing")
    elif detail_observed:
        pass
    elif readback_observed:
        pass
    elif not observed and visual_observed:
        warnings.append("machine_text_missing_visual_observed")
    elif not observed:
        failures.append("probe_text_not_observed_after")
    if width and width < 32:
        warnings.append("send_target_too_narrow")
    if height and height < 20:
        warnings.append("send_target_too_short")
    return {
        "app": app,
        "status": "fail" if failures else "warn" if warnings else "pass",
        "text": text,
        "action_ok": action.get("ok") is True,
        "text_observed_after": observed,
        "detail_observed_after": detail_observed,
        "readback_observed_after": readback_observed,
        "text_evidence_source": _text_evidence_source(
            detail_observed=detail_observed,
            readback_observed=readback_observed,
            visual_observed=visual_observed,
        ),
        "visual_observed_after": visual_observed,
        "visual_review_source": visual_review.get("source") if visual_review else "",
        "send_target_bounds": send_bounds,
        "send_target_width": width,
        "send_target_height": height,
        "failures": failures,
        "warnings": warnings,
    }


def _load_actions(action_dir: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for path in sorted(action_dir.glob("send*action*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            actions.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            actions.append(data)
    return actions


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_details(after_dir: Path) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    paths = list(after_dir.glob("*.detail.json"))
    for path in after_dir.glob("*.json"):
        if path in paths:
            continue
        if path.name in {"chat_send_probe_report.json", "send_probe_actions.json", "chat_send_visual_review.json"}:
            continue
        paths.append(path)
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            details.append(data)
    return details


def _load_visual_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        reviews = data.get("reviews")
        if isinstance(reviews, list):
            return [item for item in reviews if isinstance(item, dict)]
        return [data]
    return []


def _load_readback_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        if "events" in data:
            return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _detail_for_action(app: str, details: list[dict[str, Any]], *, text: str = "") -> dict[str, Any] | None:
    hints = APP_TITLE_HINTS.get(app.strip().lower(), (app.strip().lower(),))
    matches: list[dict[str, Any]] = []
    for detail in details:
        title = str(detail.get("window_title") or "").lower()
        sample = str(detail.get("sample") or detail.get("canvas_id") or "").lower()
        blob = f"{title} {sample}"
        if any(str(hint).lower() in blob for hint in hints if str(hint).strip()):
            matches.append(detail)
    if text:
        for detail in matches:
            if _text_in_detail(text, detail):
                return detail
    if matches:
        return matches[0]
    return details[0] if len(details) == 1 else None


def _visual_review_for_action(app: str, text: str, visual_reviews: list[dict[str, Any]]) -> dict[str, Any] | None:
    app_key = app.strip().lower()
    for review in visual_reviews:
        review_app = str(review.get("app") or "").strip().lower()
        review_text = str(review.get("text") or "")
        if review_app and review_app != app_key:
            continue
        if text and review_text and review_text != text:
            continue
        return review
    return None


def _readback_for_action(app: str, text: str, readback_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not text:
        return None
    hints = APP_TITLE_HINTS.get(app.strip().lower(), (app.strip().lower(),))
    for result in readback_results:
        app_process = str(result.get("app_process") or "").lower()
        if app_process and not any(str(hint).lower() in app_process for hint in hints if str(hint).strip()):
            continue
        for event in list(result.get("events") or []):
            if not isinstance(event, dict):
                continue
            if text in str(event.get("text") or ""):
                return result
    return None


def _text_evidence_source(*, detail_observed: bool, readback_observed: bool, visual_observed: bool) -> str:
    if detail_observed:
        return "machine_detail_confirmed"
    if readback_observed:
        return "readback_crop_ocr_confirmed"
    if visual_observed:
        return "visual_review_confirmed"
    return ""


def _text_in_detail(text: str, detail: dict[str, Any]) -> bool:
    values: list[str] = []
    for element in list(detail.get("elements") or []):
        if not isinstance(element, dict):
            continue
        values.extend(str(element.get(key) or "") for key in ("text", "name", "role_label"))
    for block in list(detail.get("ocr_blocks") or []):
        if isinstance(block, dict):
            values.append(str(block.get("text") or ""))
    return any(text in value for value in values)


def _rect(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if right <= left or bottom <= top:
        return []
    return [left, top, right, bottom]


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visual-review-file", type=Path)
    args = parser.parse_args()
    report = analyze_probe_dirs(
        action_dir=args.action_dir,
        after_dir=args.after_dir,
        output_dir=args.output_dir,
        visual_review_file=args.visual_review_file,
    )
    print(
        "overall={overall} pass={pass_count} warn={warn_count} fail={fail_count}".format(
            overall=report["overall_status"],
            pass_count=report["counts"].get("pass", 0),
            warn_count=report["counts"].get("warn", 0),
            fail_count=report["counts"].get("fail", 0),
        )
    )
    return 0 if report["overall_status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
