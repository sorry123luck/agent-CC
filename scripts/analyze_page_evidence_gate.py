"""Build generic page evidence gates from readback snapshots.

This script is intentionally not a task planner.  It only verifies whether the
current page/snapshot contains caller-provided evidence text before a later
controlled probe is allowed to proceed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_page_evidence_gate(
    *,
    snapshot_report: dict[str, Any],
    requirements: dict[str, list[str]],
) -> dict[str, Any]:
    """Return proceed/stop decisions by matching required evidence texts."""
    rows = [_evaluate_row(row, requirements) for row in list(snapshot_report.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        overall = "stop"
    elif all(row.get("decision") == "proceed" for row in rows):
        overall = "proceed"
    elif any(row.get("decision") == "proceed" for row in rows):
        overall = "partial_proceed"
    else:
        overall = "stop"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate_type": "page_evidence",
        "overall_decision": overall,
        "policy": "Rows must match all required_texts before controlled actions; missing evidence stops that row.",
        "checks": rows,
    }


def analyze_page_evidence_dir(
    *,
    snapshot_dir: Path,
    requirements_file: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load snapshot/readback artifacts and write page evidence gate reports."""
    snapshot_report = _load_json(snapshot_dir / "chat_readback_snapshot_report.json")
    for row in list(snapshot_report.get("rows") or []):
        if not isinstance(row, dict):
            continue
        row["readback_result"] = _load_readback_result(row, base_dir=snapshot_dir)
    report = build_page_evidence_gate(
        snapshot_report=snapshot_report,
        requirements=_load_requirements(requirements_file),
    )
    write_page_evidence_gate(report, output_dir or snapshot_dir)
    return report


def write_page_evidence_gate(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown page evidence gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_evidence_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Page Evidence Gate",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall decision: {report.get('overall_decision', '')}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| sample | process | decision | matched | missing | evidence | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("checks") or []):
        lines.append(
            "| {sample} | {process} | {decision} | {matched} | {missing} | {evidence} | {reason} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                decision=str(row.get("decision") or "").replace("|", "/"),
                matched=" / ".join(row.get("matched_texts") or []).replace("|", "/"),
                missing=" / ".join(row.get("missing_texts") or []).replace("|", "/"),
                evidence=" / ".join(row.get("evidence_texts") or []).replace("|", "/"),
                reason=str(row.get("reason") or "").replace("|", "/"),
            )
        )
    (output_dir / "page_evidence_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_row(row: dict[str, Any], requirements: dict[str, list[str]]) -> dict[str, Any]:
    sample = str(row.get("sample") or "")
    process = str(row.get("process_name") or "")
    readback = row.get("readback_result") if isinstance(row.get("readback_result"), dict) else {}
    texts = _readback_texts(readback)
    required = requirements.get(sample) or requirements.get(process.lower()) or requirements.get(_app_key(process)) or []
    matched = [text for text in required if _contains_text(text, texts)]
    missing = [text for text in required if text not in matched]
    if not required:
        decision = "stop"
        reason = "required evidence text missing"
    elif missing:
        decision = "stop"
        reason = "required evidence text not observed"
    else:
        decision = "proceed"
        reason = "required evidence text observed"
    return {
        "gate_type": "page_evidence",
        "sample": sample,
        "process_name": process,
        "required_texts": required,
        "expected_texts": required,
        "decision": decision,
        "matched_texts": matched,
        "missing_texts": missing,
        "evidence_texts": texts[:8],
        "reason": reason,
    }


def _readback_texts(readback: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for event in list(readback.get("events") or []):
        if not isinstance(event, dict):
            continue
        text = str(event.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _contains_text(expected: str, observed_texts: list[str]) -> bool:
    expected_norm = _normalize_text(expected)
    return any(expected_norm in _normalize_text(text) for text in observed_texts)


def _normalize_text(value: str) -> str:
    return "".join(str(value).split()).lower()


def _load_requirements(path: Path) -> dict[str, list[str]]:
    data = _load_json(path)
    requirements: dict[str, list[str]] = {}
    for item in list(data.get("requirements") or data.get("targets") or []):
        if not isinstance(item, dict):
            continue
        required = [
            str(text)
            for text in list(item.get("required_texts") or item.get("expected_texts") or [])
            if str(text)
        ]
        if not required:
            continue
        for key_name in ("sample", "process_name", "app"):
            key = str(item.get(key_name) or "").strip()
            if key:
                requirements[key if key_name == "sample" else key.lower()] = required
    for key, value in data.items():
        if key in {"requirements", "targets"} or not isinstance(value, list):
            continue
        required = [str(text) for text in value if str(text)]
        if required:
            requirements[str(key)] = required
    return requirements


def _load_readback_result(row: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    path_text = str(row.get("readback_result_path") or "")
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_absolute():
        candidates = [
            base_dir / path,
            Path.cwd() / path,
            base_dir.parent / path,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        return {}
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _app_key(process: str) -> str:
    normalized = process.strip().lower()
    if normalized in {"weixin.exe", "wechat.exe", "weixin", "wechat"}:
        return "wechat"
    if normalized in {"feishu.exe", "lark.exe", "feishu", "lark"}:
        return "feishu"
    if normalized in {"qq.exe", "qq"}:
        return "qq"
    return normalized.removesuffix(".exe")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True, help="JSON file with page evidence requirements")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_page_evidence_dir(
        snapshot_dir=args.snapshot_dir,
        requirements_file=args.requirements,
        output_dir=args.output_dir,
    )
    print(
        "overall={overall} proceed={proceed} stop={stop}".format(
            overall=report["overall_decision"],
            proceed=sum(1 for row in report["checks"] if row.get("decision") == "proceed"),
            stop=sum(1 for row in report["checks"] if row.get("decision") != "proceed"),
        )
    )
    return 0 if any(row.get("decision") == "proceed" for row in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
