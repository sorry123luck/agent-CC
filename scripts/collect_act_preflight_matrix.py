"""Collect /act dry-run preflight responses for canvas input/send candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests


ActClient = Callable[[dict[str, Any]], dict[str, Any]]
MATRIX_JSON = "act_preflight_matrix.json"
MATRIX_MD = "act_preflight_matrix.md"
INPUT_ROLES = {"message_input", "text_input"}
SEND_ROLES = {"send_button"}
CLICK_ROLES = {"button", "menu_item", "tab", "checkbox", "radio_button", "link"}
SCROLL_REGION_ROLES = {"message_stream", "message_thread", "chat_area", "content_area", "detail_pane", "list", "viewport"}
NON_SCROLL_CANDIDATE_ROLES = INPUT_ROLES | SEND_ROLES | {"send", "send_button", "message_input", "text_input"}


def collect_act_preflight_matrix_from_details(
    *,
    details: list[dict[str, Any]],
    act_client: ActClient,
    sample_rows: list[dict[str, Any]] | None = None,
    probe_text: str = "OpenClaw matrix probe",
) -> dict[str, Any]:
    """Build a matrix by calling /act dry-run for each detail's input/send candidates."""
    row_index = _sample_row_index(sample_rows or [])
    rows = [
        _build_row(
            detail=detail,
            sample_row=_matching_sample_row(detail, row_index),
            act_client=act_client,
            probe_text=probe_text,
        )
        for detail in details
        if isinstance(detail, dict)
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "act_preflight_matrix_source",
        "policy": (
            "Collects dry-run /act responses for likely message input and send "
            "candidates. It does not execute typing or sending."
        ),
        "probe_text": probe_text,
        "counts": counts,
        "rows": rows,
    }


def collect_act_preflight_matrix_dirs(
    *,
    detail_dir: Path,
    output_dir: Path | None = None,
    act_client: ActClient | None = None,
    sample_matrix: Path | None = None,
    api_base: str = "http://127.0.0.1:8000",
    probe_text: str = "OpenClaw matrix probe",
) -> dict[str, Any]:
    """Load canvas detail JSON files, collect preflight responses, and write outputs."""
    details = _load_detail_files(detail_dir)
    sample_rows = _load_sample_rows(sample_matrix or (detail_dir / "sample_matrix_summary.json"))
    if sample_rows:
        allowed_canvas_ids = {str(row.get("canvas_id") or "") for row in sample_rows if str(row.get("canvas_id") or "")}
        if allowed_canvas_ids:
            details = [detail for detail in details if str(detail.get("canvas_id") or "") in allowed_canvas_ids]
    client = act_client or _requests_act_client(api_base)
    matrix = collect_act_preflight_matrix_from_details(
        details=details,
        sample_rows=sample_rows,
        act_client=client,
        probe_text=probe_text,
    )
    write_act_preflight_matrix(matrix, output_dir or detail_dir)
    return matrix


def write_act_preflight_matrix(matrix: dict[str, Any], output_dir: Path) -> None:
    """Persist raw matrix JSON and a compact Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MATRIX_JSON).write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Act Preflight Matrix",
        "",
        f"- Generated: {matrix.get('generated_at', '')}",
        f"- Probe text: {matrix.get('probe_text', '')}",
        f"- Counts: {_format_counts(matrix.get('counts') or {})}",
        "",
        "| sample | app | canvas | status | type_text | send | click | scroll | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(matrix.get("rows") or []):
        actions = row.get("actions") if isinstance(row.get("actions"), dict) else {}
        lines.append(
            "| {sample} | {app} | {canvas} | {status} | {type_text} | {send} | {click} | {scroll} | {warnings} |".format(
                sample=_md(row.get("sample")),
                app=_md(row.get("process_name")),
                canvas=_md(row.get("canvas_id")),
                status=_md(row.get("status")),
                type_text=_action_status(actions.get("type_text")),
                send=_action_status(actions.get("send")),
                click=_action_status(actions.get("click")),
                scroll=_action_status(actions.get("scroll")),
                warnings=_md(",".join(row.get("warnings") or [])),
            )
        )
    (output_dir / MATRIX_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_row(
    *,
    detail: dict[str, Any],
    sample_row: dict[str, Any] | None,
    act_client: ActClient,
    probe_text: str,
) -> dict[str, Any]:
    canvas_id = str(detail.get("canvas_id") or "")
    sample_data = sample_row or {}
    process_name = str(sample_data.get("process_name") or detail.get("app_id") or detail.get("process_name") or "")
    sample = str(sample_data.get("sample") or detail.get("sample") or f"{process_name or 'unknown'}_{canvas_id or 'canvas'}")
    elements = list(detail.get("elements") or [])
    input_candidate = _first_candidate(elements, roles=INPUT_ROLES)
    send_candidate = _candidate_by_id(elements, str(sample_data.get("composer_send_target_candidate_id") or ""))
    if not send_candidate:
        send_candidate = _first_candidate(elements, roles=SEND_ROLES, fallback_send_text=True)
    click_candidate = _first_click_candidate(elements)
    scroll_candidate = _first_scroll_candidate(detail, elements)
    warnings: list[str] = []
    actions: dict[str, Any] = {}
    if input_candidate:
        actions["type_text"] = act_client(
            {
                "canvas_id": canvas_id,
                "candidate_id": str(input_candidate.get("element_id") or ""),
                "action": "type_text",
                "params": {"text": probe_text},
                "dry_run": True,
            }
        )
    else:
        warnings.append("missing_message_input_candidate")
    if send_candidate:
        actions["send"] = act_client(
            {
                "canvas_id": canvas_id,
                "candidate_id": str(send_candidate.get("element_id") or ""),
                "action": "send",
                "params": {"expected_text": probe_text},
                "dry_run": True,
            }
        )
    else:
        warnings.append("missing_send_candidate")
    if click_candidate:
        actions["click"] = act_client(
            {
                "canvas_id": canvas_id,
                "candidate_id": str(click_candidate.get("element_id") or ""),
                "action": "click",
                "params": {},
                "dry_run": True,
            }
        )
    if scroll_candidate:
        actions["scroll"] = act_client(
            {
                "canvas_id": canvas_id,
                "candidate_id": str(scroll_candidate.get("element_id") or ""),
                "action": "scroll",
                "params": {"direction": "down", "amount": "page"},
                "dry_run": True,
            }
        )
    return {
        "sample": sample,
        "process_name": process_name,
        "canvas_id": canvas_id,
        "window_title": str(detail.get("window_title") or ""),
        "page_class": str(detail.get("page_class") or ""),
        "status": "collected" if actions else "missing_candidates",
        "candidate_ids": {
            "type_text": str(input_candidate.get("element_id") or "") if input_candidate else "",
            "send": str(send_candidate.get("element_id") or "") if send_candidate else "",
            "click": str(click_candidate.get("element_id") or "") if click_candidate else "",
            "scroll": str(scroll_candidate.get("element_id") or "") if scroll_candidate else "",
        },
        "actions": actions,
        "warnings": warnings,
    }


def _first_candidate(
    elements: list[Any],
    *,
    roles: set[str],
    fallback_send_text: bool = False,
) -> dict[str, Any] | None:
    for item in elements:
        if not isinstance(item, dict):
            continue
        role = _role(item)
        text = str(item.get("text") or "").strip().lower()
        risk_tags = {str(tag).lower() for tag in list(item.get("risk_tags") or [])}
        if role in roles:
            return item
        if fallback_send_text and ("send" in risk_tags or text in {"发送", "send"}):
            return item
    return None


def _first_click_candidate(elements: list[Any]) -> dict[str, Any] | None:
    for item in elements:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        if str(attrs.get("actionability") or "") != "controlled_probe":
            continue
        role = _role(item)
        if role in CLICK_ROLES:
            return item
    return None


def _first_scroll_candidate(detail: dict[str, Any], elements: list[Any]) -> dict[str, Any] | None:
    scroll_region_ids = _scroll_region_ids(detail)
    if not scroll_region_ids:
        return None
    for item in elements:
        if not isinstance(item, dict):
            continue
        if str(item.get("element_id") or "") == "":
            continue
        if str(item.get("region_id") or "") not in scroll_region_ids:
            continue
        if _role(item) in NON_SCROLL_CANDIDATE_ROLES:
            continue
        return item
    return None


def _scroll_region_ids(detail: dict[str, Any]) -> set[str]:
    region_ids = {
        str(item.get("region_id") or "")
        for item in list(detail.get("scroll_contexts") or [])
        if isinstance(item, dict) and str(item.get("region_id") or "")
    }
    regions = [item for item in list(detail.get("regions") or []) if isinstance(item, dict)]
    scroll_context_region_ids = {
        str(region.get("region_id") or "")
        for region in regions
        if str(region.get("scroll_context_id") or "") and str(region.get("region_id") or "")
    }
    role_region_ids = {
        str(region.get("region_id") or "")
        for region in regions
        if _region_role(region) in SCROLL_REGION_ROLES and str(region.get("region_id") or "")
    }
    return {region_id for region_id in region_ids | scroll_context_region_ids | role_region_ids if region_id}


def _region_role(region: dict[str, Any]) -> str:
    for key in ("role", "region_role", "purpose", "semantic_role"):
        value = str(region.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _candidate_by_id(elements: list[Any], candidate_id: str) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    for item in elements:
        if isinstance(item, dict) and str(item.get("element_id") or "") == candidate_id:
            return item
    return None


def _role(item: dict[str, Any]) -> str:
    for key in ("semantic_role", "role_label", "visual_type"):
        value = str(item.get(key) or "").strip().lower()
        if value:
            return value
    attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    return str(attrs.get("role") or attrs.get("semantic_role") or "").strip().lower()


def _load_detail_files(detail_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for path in detail_dir.glob("*.detail.json")
        if path.name not in {MATRIX_JSON, "act_preflight_matrix_report.json"}
    )
    details = []
    for path in paths:
        data = _load_json(path)
        if isinstance(data, dict):
            details.append(data)
    return details


def _load_sample_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _load_json(path)
    return [row for row in list(data.get("rows") or []) if isinstance(row, dict)]


def _sample_row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in _sample_keys(row):
            index.setdefault(key, row)
    return index


def _matching_sample_row(detail: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in _sample_keys(detail):
        if key in index:
            return index[key]
    canvas_id = str(detail.get("canvas_id") or "")
    if canvas_id:
        for row in index.values():
            if str(row.get("canvas_id") or "") == canvas_id:
                return row
    return None


def _sample_keys(row: dict[str, Any]) -> list[str]:
    keys = []
    sample = str(row.get("sample") or "")
    if sample:
        keys.append(sample.lower())
    process = str(row.get("process_name") or row.get("app_id") or "")
    canvas_id = str(row.get("canvas_id") or "")
    if process and canvas_id:
        keys.append(f"{process}_{canvas_id}".lower())
    return keys


def _requests_act_client(api_base: str) -> ActClient:
    base = api_base.rstrip("/")

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{base}/api/v1/act", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    return call


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _action_status(response: Any) -> str:
    if not isinstance(response, dict):
        return "missing"
    result = str(response.get("execution_result") or "")
    warnings = ",".join(str(item) for item in list(response.get("warnings") or []) if str(item))
    return _md(f"{result}:{warnings}" if warnings else result)


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-matrix", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--probe-text", default="OpenClaw matrix probe")
    args = parser.parse_args()
    matrix = collect_act_preflight_matrix_dirs(
        detail_dir=args.detail_dir,
        output_dir=args.output_dir,
        sample_matrix=args.sample_matrix,
        api_base=args.api_base,
        probe_text=args.probe_text,
    )
    print(f"rows={len(matrix.get('rows') or [])} counts={_format_counts(matrix.get('counts') or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
