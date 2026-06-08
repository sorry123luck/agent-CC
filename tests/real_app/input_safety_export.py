"""Real-app input safety sample export helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.input_safety_report import write_archived_input_safety_report

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_SAMPLE_ROOT = PROJECT_ROOT / "data" / "baselines" / "new" / "input_safety"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports"
CHAT_INPUT_MATRIX_APPS = ["wechat", "qq", "feishu"]


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def canvas_detail_to_input_safety_snapshot(detail: dict[str, Any], *, app_name: str) -> dict[str, Any]:
    elements = []
    for element in detail.get("elements", []):
        if not isinstance(element, dict):
            continue
        elements.append(
            {
                "element_id": element.get("element_id"),
                "region_id": element.get("region_id"),
                "semantic_role": element.get("semantic_role"),
                "control_type": element.get("control_type", ""),
                "bounds": element.get("bounds"),
                "text": element.get("text", ""),
                "name": element.get("name"),
                "value": element.get("value"),
                "placeholder": element.get("placeholder"),
                "interactable": element.get("interactable", True),
                "state": element.get("state") or {},
                "attributes": element.get("attributes") or {},
                "click_point": element.get("click_point"),
                "provider_sources": element.get("provider_sources") or [],
            }
        )

    return {
        "canvas_id": detail.get("canvas_id", ""),
        "app": {
            "app_id": detail.get("app_id") or app_name,
            "process_name": detail.get("process_name") or app_name,
        },
        "window": {
            "title": detail.get("window_title", ""),
        },
        "surface_type": detail.get("surface_type", ""),
        "page": {
            "page_class": detail.get("page_class", ""),
        },
        "providers_used": detail.get("providers_used") or [],
        "elements": elements,
    }


def export_observed_input_safety_sample(
    client,
    *,
    canvas_id: str,
    app_name: str,
    output_root: str | Path = DEFAULT_SAMPLE_ROOT,
    run_id: str | None = None,
) -> Path:
    response = client.get(f"/api/v1/canvases/{canvas_id}")
    if response.status_code != 200:
        raise RuntimeError(f"Canvas detail export failed: HTTP {response.status_code}")

    run = run_id or _run_id()
    snapshot = canvas_detail_to_input_safety_snapshot(response.json(), app_name=app_name)
    output_dir = Path(output_root) / app_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{app_name}_{run}_canvas.json"
    output_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_real_app_input_safety_report(
    sample_root: str | Path = DEFAULT_SAMPLE_ROOT,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    *,
    run_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    return write_archived_input_safety_report(
        sample_root,
        report_root,
        matrix_apps=CHAT_INPUT_MATRIX_APPS,
        run_id=run_id,
    )
