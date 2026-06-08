"""Generate input safety reports from InteractionCanvas snapshot JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "input_safety_report.v1"
INPUT_ROLES = {"message_input", "text_input", "search_input", "password_input"}
SEND_ROLES = {"send_button"}
INPUT_REGION_HINTS = ("bottom_input", "composer", "input", "message")
SEND_BUTTON_LABEL_HINTS = ("send", "发送")


def _role(element: dict[str, Any] | None) -> str:
    if not element:
        return ""
    value = element.get("semantic_role") or ""
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or ""
    return str(value).lower()


def _control_type(element: dict[str, Any] | None) -> str:
    if element is None:
        return ""
    return str(element.get("control_type") or "").strip().lower()


def _label(element: dict[str, Any] | None) -> str:
    if element is None:
        return ""
    return f"{element.get('text') or ''} {element.get('name') or ''}".strip().lower()


def _is_chat_context(snapshot: dict[str, Any]) -> bool:
    app = snapshot.get("app") or {}
    page = snapshot.get("page") or {}
    app_id = str(app.get("app_id") or "").lower()
    page_class = str(page.get("page_class") or "").lower()
    return app_id in {"wechat", "qq", "feishu"} or "chat" in page_class


def _in_input_region(element: dict[str, Any] | None) -> bool:
    if element is None:
        return False
    region_id = str(element.get("region_id") or "").lower()
    return any(hint in region_id for hint in INPUT_REGION_HINTS)


def _is_input_candidate(snapshot: dict[str, Any], element: dict[str, Any] | None) -> bool:
    if element is None:
        return False
    if _role(element) in INPUT_ROLES:
        return True
    control_type = _control_type(element)
    return (
        _is_chat_context(snapshot)
        and _in_input_region(element)
        and ("edit" in control_type or "textbox" in control_type or control_type == "text")
    )


def _is_send_button_candidate(snapshot: dict[str, Any], element: dict[str, Any] | None) -> bool:
    if element is None:
        return False
    if _role(element) in SEND_ROLES:
        return True
    control_type = _control_type(element)
    role = _role(element)
    label = _label(element)
    return (
        _is_chat_context(snapshot)
        and _in_input_region(element)
        and any(hint in label for hint in SEND_BUTTON_LABEL_HINTS)
        and ("button" in control_type or role in {"button", "icon_button"})
    )


def _detection_source(element: dict[str, Any] | None, semantic_roles: set[str]) -> str | None:
    if element is None:
        return None
    return "semantic_role" if _role(element) in semantic_roles else "fallback"


def _enabled(element: dict[str, Any] | None) -> bool | None:
    if element is None:
        return None
    state = element.get("state") or {}
    if isinstance(state, dict) and "enabled" in state:
        return bool(state.get("enabled"))
    if "interactable" in element:
        return bool(element.get("interactable"))
    return None


def _text_value(element: dict[str, Any] | None) -> str:
    if element is None:
        return ""
    attrs = element.get("attributes") or {}
    for key in ("value", "text", "uia_value", "current_value"):
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for key in ("value", "text"):
        value = element.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _input_state(input_candidate: dict[str, Any] | None) -> str:
    if input_candidate is None:
        return "unknown"
    return "has_text" if _text_value(input_candidate) else "empty"


def _bounds_center(bounds: Any) -> list[int] | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
        return None
    left, top, right, bottom = bounds[:4]
    return [int((left + right) / 2), int((top + bottom) / 2)]


def _click_point_summary(element: dict[str, Any] | None) -> dict[str, Any]:
    if element is None:
        return {
            "click_point": None,
            "click_point_source": "missing",
            "click_point_status": "missing",
        }
    click_point = element.get("click_point")
    if isinstance(click_point, (list, tuple)) and len(click_point) >= 2:
        return {
            "click_point": [int(click_point[0]), int(click_point[1])],
            "click_point_source": "candidate_click_point",
            "click_point_status": "stable",
        }
    center = _bounds_center(element.get("bounds"))
    if center:
        return {
            "click_point": center,
            "click_point_source": "bounds_center",
            "click_point_status": "stable",
        }
    return {
        "click_point": None,
        "click_point_source": "missing",
        "click_point_status": "missing",
    }


def _find_first(elements: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for element in elements:
        if predicate(element):
            return element
    return None


def _candidate_summary(element: dict[str, Any] | None) -> dict[str, Any] | None:
    if element is None:
        return None
    return {
        "candidate_id": element.get("element_id"),
        "semantic_role": _role(element),
        "region_id": element.get("region_id"),
        "enabled": _enabled(element),
        **_click_point_summary(element),
    }


def _send_button_state(send_button: dict[str, Any] | None, input_state: str) -> dict[str, Any]:
    enabled = _enabled(send_button)
    expected_enabled = None
    if input_state == "empty":
        expected_enabled = False
    elif input_state == "has_text":
        expected_enabled = True

    matches = None
    mismatch_reason = None
    if enabled is not None and expected_enabled is not None:
        matches = enabled == expected_enabled
        if not matches and input_state == "empty" and enabled is True:
            mismatch_reason = "send_button_enabled_while_input_empty"
        elif not matches and input_state == "has_text" and enabled is False:
            mismatch_reason = "send_button_disabled_while_input_has_text"

    return {
        "candidate_id": send_button.get("element_id") if send_button else None,
        "enabled": enabled,
        "expected_enabled": expected_enabled,
        "matches_input_state": matches,
        "mismatch_reason": mismatch_reason,
    }


def _input_gate_evaluation(
    input_candidate: dict[str, Any] | None,
    send_button: dict[str, Any] | None,
    input_state: str,
    send_state: dict[str, Any],
) -> dict[str, Any]:
    input_click = _click_point_summary(input_candidate)
    send_click = _click_point_summary(send_button)
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    def record(name: str, passed: bool) -> None:
        (passed_checks if passed else failed_checks).append(name)

    record("target_input", input_candidate is not None)
    record("input_state", input_state in {"empty", "has_text"})
    record("input_click_point", input_click["click_point_status"] == "stable")
    record("send_button", send_button is not None)
    record("send_button_click_point", send_click["click_point_status"] == "stable")
    record("send_button_state", send_state.get("matches_input_state") is True)

    base_ready = not failed_checks
    return {
        "gate_status": "blocked_pending_target2_acceptance",
        "eligible_for_safe_to_type": base_ready and input_state == "empty",
        "eligible_for_safe_to_send": base_ready and input_state == "has_text",
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "blocking_reasons": ["input_actions_remain_blocked_pending_target2_gate"],
    }


def _sample_id(snapshot: dict[str, Any], index: int) -> str:
    return str(snapshot.get("canvas_id") or f"sample_{index}")


def assess_input_safety_sample(snapshot: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    app = snapshot.get("app") or {}
    page = snapshot.get("page") or {}
    elements = [element for element in snapshot.get("elements", []) if isinstance(element, dict)]
    input_candidate = _find_first(elements, lambda element: _is_input_candidate(snapshot, element))
    send_button = _find_first(elements, lambda element: _is_send_button_candidate(snapshot, element))
    input_state = _input_state(input_candidate)
    send_state = _send_button_state(send_button, input_state)
    gate_evaluation = _input_gate_evaluation(input_candidate, send_button, input_state, send_state)
    input_summary = _candidate_summary(input_candidate)
    send_summary = _candidate_summary(send_button)

    issues: list[str] = []
    if input_candidate is None:
        issues.append("missing_input_candidate")
    if send_button is None:
        issues.append("missing_send_button")
    if input_summary and input_summary["click_point_status"] != "stable":
        issues.append("input_click_point_unstable")
    if send_summary and send_summary["click_point_status"] != "stable":
        issues.append("send_button_click_point_unstable")
    if send_state["mismatch_reason"]:
        issues.append(send_state["mismatch_reason"])

    return {
        "sample_id": _sample_id(snapshot, index),
        "source_path": snapshot.get("_source_path"),
        "status": "passed" if not issues else "failed",
        "profile": {
            "app_id": app.get("app_id"),
            "page_class": page.get("page_class"),
            "input_region_id": input_candidate.get("region_id") if input_candidate else None,
            "send_button_region_id": send_button.get("region_id") if send_button else None,
        },
        "input_state": input_state,
        "target_input": input_summary,
        "associated_send_button": send_summary,
        "send_button_state": send_state,
        "input_gate_evaluation": gate_evaluation,
        "matrix_detection": {
            "input_source": _detection_source(input_candidate, INPUT_ROLES),
            "send_button_source": _detection_source(send_button, SEND_ROLES),
        },
        "safe_to_type": False,
        "safe_to_send": False,
        "issues": issues,
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    apps: dict[str, dict[str, int]] = {}
    passed = 0
    failed = 0
    for sample in samples:
        app_id = sample["profile"].get("app_id") or "unknown"
        apps.setdefault(app_id, {"total": 0, "passed": 0, "failed": 0})
        apps[app_id]["total"] += 1
        if sample["status"] == "passed":
            passed += 1
            apps[app_id]["passed"] += 1
        else:
            failed += 1
            apps[app_id]["failed"] += 1
    return {
        "total_samples": len(samples),
        "passed": passed,
        "failed": failed,
        "apps": apps,
    }


def _matrix_summary(samples: list[dict[str, Any]], matrix_apps: list[str] | None) -> dict[str, Any] | None:
    if matrix_apps is None:
        return None
    expected = list(matrix_apps)
    covered = sorted(
        {
            sample["profile"].get("app_id")
            for sample in samples
            if sample["profile"].get("app_id") in expected
        }
    )
    missing = [app_id for app_id in expected if app_id not in covered]
    return {
        "expected_apps": expected,
        "covered_apps": covered,
        "missing_apps": missing,
        "coverage_status": "complete" if not missing else "incomplete",
    }


def _target2_acceptance(samples: list[dict[str, Any]], matrix: dict[str, Any] | None) -> dict[str, Any]:
    required_states = ["empty", "has_text"]
    expected_apps = (matrix or {}).get("expected_apps") or sorted(
        {
            sample["profile"].get("app_id")
            for sample in samples
            if sample["profile"].get("app_id")
        }
    )
    failed_apps = sorted(
        {
            sample["profile"].get("app_id") or "unknown"
            for sample in samples
            if sample.get("status") != "passed"
        }
    )
    state_coverage: dict[str, set[str]] = {app_id: set() for app_id in expected_apps}
    for sample in samples:
        app_id = sample["profile"].get("app_id")
        input_state = sample.get("input_state")
        if app_id in state_coverage and input_state in required_states and sample.get("status") == "passed":
            state_coverage[app_id].add(input_state)

    missing_state_coverage = {
        app_id: [state for state in required_states if state not in covered_states]
        for app_id, covered_states in state_coverage.items()
        if any(state not in covered_states for state in required_states)
    }

    blocking_reasons: list[str] = []
    if matrix is not None and matrix.get("coverage_status") != "complete":
        blocking_reasons.append("matrix_coverage_incomplete")
    if missing_state_coverage:
        blocking_reasons.append("state_coverage_incomplete")
    if failed_apps:
        blocking_reasons.append("sample_failures")

    ready = not blocking_reasons
    return {
        "status": "ready_for_manual_review" if ready else "not_ready",
        "safe_to_type_can_be_considered": ready,
        "safe_to_send_can_be_considered": ready,
        "required_states": required_states,
        "missing_state_coverage": missing_state_coverage,
        "failed_apps": failed_apps,
        "blocking_reasons": blocking_reasons,
    }


def build_input_safety_report(
    snapshots: list[dict[str, Any]],
    *,
    matrix_apps: list[str] | None = None,
) -> dict[str, Any]:
    samples = [assess_input_safety_sample(snapshot, index=index) for index, snapshot in enumerate(snapshots)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": _summary(samples),
        "samples": samples,
    }
    matrix = _matrix_summary(samples, matrix_apps)
    if matrix is not None:
        report["matrix"] = matrix
    report["target2_acceptance"] = _target2_acceptance(samples, matrix)
    return report


def load_snapshots(input_dir: Path) -> list[dict[str, Any]]:
    snapshots = []
    for path in sorted(input_dir.rglob("*.json")):
        if not path.name.endswith(("_snapshot.json", "_canvas.json")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = dict(data)
            data["_source_path"] = str(path)
            snapshots.append(data)
    return snapshots


def write_input_safety_report(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    matrix_apps: list[str] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output = Path(output_path)
    report = build_input_safety_report(load_snapshots(input_path), matrix_apps=matrix_apps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_archived_input_safety_report(
    input_dir: str | Path,
    archive_root: str | Path,
    *,
    matrix_apps: list[str] | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    run = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(archive_root) / "input_safety" / run / "input_safety_report.json"
    report = write_input_safety_report(input_dir, output_path, matrix_apps=matrix_apps)
    return report, output_path


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate input safety report from canvas snapshots")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_snapshot.json or *_canvas.json files")
    parser.add_argument("--output", help="Output report JSON path")
    parser.add_argument("--archive-root", help="Archive root; writes input_safety/<run-id>/input_safety_report.json")
    parser.add_argument("--run-id", help="Stable run id for archive output")
    parser.add_argument("--matrix-apps", help="Comma-separated expected app matrix, e.g. wechat,qq,feishu")
    args = parser.parse_args()

    matrix_apps = _csv(args.matrix_apps)
    if args.archive_root:
        report, output_path = write_archived_input_safety_report(
            args.input_dir,
            args.archive_root,
            matrix_apps=matrix_apps,
            run_id=args.run_id,
        )
    elif args.output:
        output_path = Path(args.output)
        report = write_input_safety_report(args.input_dir, output_path, matrix_apps=matrix_apps)
    else:
        parser.error("one of --output or --archive-root is required")

    summary = report["summary"]
    print(
        "Input safety report: "
        f"{summary['passed']}/{summary['total_samples']} passed, "
        f"{summary['failed']} failed -> {output_path}"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
