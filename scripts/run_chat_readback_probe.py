"""Run OCR-first chat readback probes over saved chat message crops."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.readback_models import ChatReadbackResult
from src.chat.readback_worker import ChatReadbackWorker


def find_sent_message_crops(send_probe_dir: Path) -> list[dict[str, Any]]:
    """Return send probe rows that have a saved sent message crop."""
    actions_path = send_probe_dir / "send_probe_actions.json"
    if not actions_path.exists():
        return []
    try:
        rows = json.loads(actions_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []
    crops: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        crop = str(row.get("sent_message_crop") or "")
        if not crop or not Path(crop).exists():
            continue
        crops.append(
            {
                "sample": str(row.get("sample") or ""),
                "process_name": str(row.get("process_name") or ""),
                "hwnd": int(row.get("hwnd") or 0),
                "text": str(row.get("text") or ""),
                "sent_message_crop": crop,
            }
        )
    return crops


def run_from_send_probe(send_probe_dir: Path, output_dir: Path) -> list[ChatReadbackResult]:
    """Read saved sent-message crops from a send probe artifact."""
    worker = ChatReadbackWorker()
    results: list[ChatReadbackResult] = []
    for item in find_sent_message_crops(send_probe_dir):
        crop_path = Path(item["sent_message_crop"])
        with Image.open(crop_path) as image:
            width, height = image.size
            result = worker.read_crop(
                image=image.copy(),
                image_path=crop_path,
                app_process=item["process_name"],
                window_id=str(item["hwnd"]),
                stream_bounds=(0, 0, width, height),
                crop_origin=(0, 0),
                expected_self_texts=[str(item.get("text") or "")],
            )
        results.append(result)
        _copy_crop_reference(output_dir, item)
    return results


def write_readback_report(output_dir: Path, results: list[ChatReadbackResult]) -> None:
    """Persist JSON and Markdown reports for chat readback results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": _overall(results),
        "results": [asdict(result) for result in results],
    }
    (output_dir / "chat_readback_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Readback Probe",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Overall: {payload['overall']}",
        f"- Results: {len(results)}",
        "",
        "| app | window | events | text | attachments | warnings |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for result in results:
        text = " / ".join(event.text or "" for event in result.events if event.text)
        attachments = sum(len(event.attachments) for event in result.events)
        lines.append(
            "| {app} | {window} | {events} | {text} | {attachments} | {warnings} |".format(
                app=result.app_process.replace("|", "/"),
                window=result.window_id.replace("|", "/"),
                events=len(result.events),
                text=text.replace("|", "/"),
                attachments=attachments,
                warnings=",".join(result.warnings).replace("|", "/"),
            )
        )
    (output_dir / "chat_readback_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_crop_reference(output_dir: Path, item: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_path = output_dir / "crop_manifest.json"
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else []
    except json.JSONDecodeError:
        existing = []
    existing.append(item)
    existing_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _overall(results: list[ChatReadbackResult]) -> str:
    if not results:
        return "blocked"
    if any(result.status == "fail" for result in results):
        return "fail"
    if any(result.status == "warn" or result.warnings for result in results):
        return "warn"
    return "pass"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-send-probe", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, required=True)
    args = parser.parse_args()
    results = run_from_send_probe(args.from_send_probe, args.output_dir)
    write_readback_report(args.output_dir, results)
    print(f"overall={_overall(results)} results={len(results)}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
