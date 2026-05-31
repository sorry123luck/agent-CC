"""Run a fixed /act preflight regression from saved canvas details.

This is the product-level glue for the act preflight matrix: collect dry-run
responses, analyze protocol completeness, and write a compact summary. It never
executes desktop typing or sending.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_act_preflight_matrix
import collect_act_preflight_matrix


ActClient = Callable[[dict[str, Any]], dict[str, Any]]


def run_act_preflight_regression(
    *,
    detail_dir: Path,
    output_dir: Path,
    sample_matrix: Path | None = None,
    act_client: ActClient | None = None,
    api_base: str = "http://127.0.0.1:8000",
    probe_text: str = "DeskCanvas matrix probe",
) -> dict[str, Any]:
    """Collect and analyze /act preflight responses for a detail directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = act_client or safe_act_client(_requests_act_post(api_base))
    collect_act_preflight_matrix.collect_act_preflight_matrix_dirs(
        detail_dir=detail_dir,
        output_dir=output_dir,
        sample_matrix=sample_matrix,
        act_client=client,
        probe_text=probe_text,
    )
    analysis_dir = output_dir / "analysis"
    report = analyze_act_preflight_matrix.analyze_act_preflight_matrix_dirs(
        input_dir=output_dir,
        output_dir=analysis_dir,
    )
    write_regression_summary(report=report, output_dir=output_dir)
    return report


def safe_act_client(act_post: ActClient) -> ActClient:
    """Wrap an act client so backend failures become report rows."""

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return act_post(payload)
        except Exception as exc:  # noqa: BLE001 - report the integration failure as data.
            return {
                "execution_result": "blocked",
                "can_execute": False,
                "warnings": ["act_preflight_request_failed"],
                "error_message": str(exc),
                "action_plan": {
                    "action": str(payload.get("action") or ""),
                    "candidate_id": str(payload.get("candidate_id") or ""),
                    "action_level": "blocked",
                    "verification_plan": {},
                },
            }

    return call


def write_regression_summary(*, report: dict[str, Any], output_dir: Path) -> None:
    """Write a human-readable top-level summary."""
    rows = list(report.get("rows") or [])
    lines = [
        "# Act Preflight Regression Summary",
        "",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Rows: {len(rows)}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| sample | process | status | warnings | failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {process} | {status} | {warnings} | {failures} |".format(
                sample=_md(row.get("sample")),
                process=_md(row.get("process_name")),
                status=_md(row.get("status")),
                warnings=_md(",".join(row.get("warnings") or [])),
                failures=_md(",".join(row.get("failures") or [])),
            )
        )
    (output_dir / "act_preflight_regression_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _requests_act_post(api_base: str) -> ActClient:
    base = api_base.rstrip("/")

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{base}/api/v1/act", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    return post


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-matrix", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--probe-text", default="DeskCanvas matrix probe")
    args = parser.parse_args()
    report = run_act_preflight_regression(
        detail_dir=args.detail_dir,
        output_dir=args.output_dir,
        sample_matrix=args.sample_matrix,
        api_base=args.api_base,
        probe_text=args.probe_text,
    )
    print(f"overall={report.get('overall_status')} rows={len(report.get('rows') or [])}")
    return 1 if report.get("overall_status") == "unusable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
