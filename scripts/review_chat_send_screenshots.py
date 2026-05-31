"""Build a visual review file for controlled chat send probe screenshots.

This script does not infer truth by itself. It turns explicit visual review
decisions into `chat_send_visual_review.json`, which `analyze_chat_send_probe.py`
can consume to separate "typed text is visible but OCR missed it" from true
input-coordinate failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_actions(action_dir: Path) -> list[dict[str, Any]]:
    """Load send probe actions from a probe output directory."""
    actions: list[dict[str, Any]] = []
    for path in sorted(action_dir.glob("send*action*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            actions.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            actions.append(data)
    return actions


def build_review_rows(
    actions: list[dict[str, Any]],
    *,
    observed_apps: set[str],
    stage: str,
    crop_ocr_text_by_app: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build visual review rows from explicit observed app decisions."""
    rows: list[dict[str, Any]] = []
    crop_ocr_text_by_app = crop_ocr_text_by_app or {}
    for action in actions:
        app = str(action.get("app") or "").strip().lower()
        text = str(action.get("text") or "")
        screenshot_source = str(action.get(f"{stage}_screenshot") or "")
        crop_key = f"{stage}_message_crop" if stage == "sent" else f"{stage}_input_crop"
        crop_source = str(action.get(crop_key) or "")
        crop_ocr_text = crop_ocr_text_by_app.get(app, "")
        observed_by_ocr = bool(text and crop_ocr_text and text in crop_ocr_text)
        observed_by_human = app in observed_apps
        observed = observed_by_ocr or observed_by_human
        source = crop_source if observed_by_ocr and crop_source else screenshot_source
        method = (
            "message_crop_ocr"
            if observed_by_ocr and stage == "sent"
            else "input_crop_ocr"
            if observed_by_ocr
            else "human_visual_review"
            if observed_by_human
            else "not_reviewed"
        )
        rows.append(
            {
                "app": app,
                "text": text,
                "observed_after": observed,
                "source": source,
                "stage": stage,
                "review_method": method,
                "reviewed_at": datetime.now().isoformat(timespec="seconds") if observed else "",
                "ocr_text": crop_ocr_text,
                "notes": ["typed_text_visible_in_input_crop"] if observed_by_ocr else ["typed_text_visible_in_screenshot"] if observed_by_human else [],
            }
        )
    return rows


def collect_input_crop_ocr_texts(actions: list[dict[str, Any]], *, stage: str) -> dict[str, str]:
    """Run OCR on input crop screenshots and return merged text by app."""
    from PIL import Image

    from src.perception.ocr_service import OCRService

    service = OCRService()
    texts: dict[str, str] = {}
    for action in actions:
        app = str(action.get("app") or "").strip().lower()
        crop_key = f"{stage}_message_crop" if stage == "sent" else f"{stage}_input_crop"
        path = Path(str(action.get(crop_key) or ""))
        if not app or not path.exists():
            continue
        result = service.extract_with_metadata(
            Image.open(path),
            min_text_length=1,
            filter_pure_digits=False,
            filter_pure_symbols=False,
        )
        texts[app] = " ".join(block.text for block in result.blocks)
    return texts


def write_review(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Persist visual review JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reviews": rows,
    }
    (output_dir / "chat_send_visual_review.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Send Visual Review",
        "",
        f"- Generated: {payload['generated_at']}",
        "",
        "| app | observed | method | text | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {app} | {observed} | {method} | {text} | {source} |".format(
                app=str(row.get("app") or "").replace("|", "/"),
                observed="yes" if row.get("observed_after") is True else "no",
                method=str(row.get("review_method") or "").replace("|", "/"),
                text=str(row.get("text") or "").replace("|", "/"),
                source=str(row.get("source") or "").replace("|", "/"),
            )
        )
    (output_dir / "chat_send_visual_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_observed_apps(items: list[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--observed-app", action="append", default=[], help="App whose expected text is visibly present")
    parser.add_argument("--stage", choices=["typed", "sent"], default="typed")
    parser.add_argument("--auto-ocr-crops", action="store_true", help="Run OCR on *_input_crop screenshots before manual review")
    args = parser.parse_args()
    actions = load_actions(args.action_dir)
    crop_ocr_texts = collect_input_crop_ocr_texts(actions, stage=args.stage) if args.auto_ocr_crops else {}
    rows = build_review_rows(
        actions,
        observed_apps=_parse_observed_apps(args.observed_app),
        stage=args.stage,
        crop_ocr_text_by_app=crop_ocr_texts,
    )
    write_review(rows, args.output_dir or args.action_dir)
    observed_count = sum(1 for row in rows if row.get("observed_after") is True)
    print(f"reviews={len(rows)} observed={observed_count}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
