from __future__ import annotations

import base64
import copy
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TRACE_ROOT = Path(os.getenv("OPENCLAW_VLM_TRACE_DIR", "data/vlm_traces"))
_CLEANUP_INTERVAL_SECONDS = 3600
_last_cleanup_ts = 0.0


def vlm_trace_enabled() -> bool:
    return os.getenv("OPENCLAW_VLM_TRACE", "1") != "0"


def save_full_payload_enabled() -> bool:
    return os.getenv("OPENCLAW_VLM_TRACE_FULL_PAYLOAD", "0") == "1"


def cleanup_old_traces(*, retention_hours: int | None = None, max_dirs: int | None = None) -> int:
    """Best-effort trace retention cleanup.

    Full provider payloads remain opt-in via OPENCLAW_VLM_TRACE_FULL_PAYLOAD=1.
    Trace directories are still useful by default, but bounded by age/count.
    """
    if retention_hours is None:
        retention_hours = int(os.getenv("OPENCLAW_VLM_TRACE_RETENTION_HOURS", "24") or "24")
    if max_dirs is None:
        max_dirs = int(os.getenv("OPENCLAW_VLM_TRACE_MAX_DIRS", "200") or "200")
    if not TRACE_ROOT.exists():
        return 0

    import shutil
    import time

    now = time.time()
    cutoff = now - max(1, retention_hours) * 3600
    dirs = [p for p in TRACE_ROOT.iterdir() if p.is_dir()]
    removed = 0

    for path in dirs:
        try:
            if path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue

    remaining = sorted(
        [p for p in TRACE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in remaining[max(0, max_dirs):]:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def _safe_name(value: str) -> str:
    value = value or "unknown"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value[:80]


def create_vlm_trace_dir(provider_name: str, model_name: str) -> Path | None:
    if not vlm_trace_enabled():
        return None

    global _last_cleanup_ts
    import time

    now_ts = time.time()
    if now_ts - _last_cleanup_ts > _CLEANUP_INTERVAL_SECONDS:
        cleanup_old_traces()
        _last_cleanup_ts = now_ts

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    trace_id = uuid.uuid4().hex[:8]

    dirname = f"{ts}_{_safe_name(provider_name)}_{_safe_name(model_name)}_{trace_id}"
    trace_dir = TRACE_ROOT / dirname
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def write_json(trace_dir: str | Path | None, filename: str, data: Any) -> None:
    if not trace_dir:
        return

    path = Path(trace_dir) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_text(trace_dir: str | Path | None, filename: str, text: str) -> None:
    if not trace_dir:
        return

    path = Path(trace_dir) / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def save_image(trace_dir: str | Path | None, filename: str, image: Image.Image) -> dict[str, Any] | None:
    if not trace_dir:
        return None

    path = Path(trace_dir) / filename
    image.save(path)

    return {
        "filename": filename,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }


def save_data_url_image(
    trace_dir: str | Path | None,
    filename_stem: str,
    data_url: str,
) -> dict[str, Any] | None:
    if not trace_dir or not data_url:
        return None

    if "," not in data_url:
        return {
            "error": "not_data_url",
            "preview": data_url[:200],
        }

    header, b64 = data_url.split(",", 1)

    if "image/png" in header:
        ext = "png"
        mime = "image/png"
    elif "image/jpeg" in header or "image/jpg" in header:
        ext = "jpg"
        mime = "image/jpeg"
    else:
        ext = "bin"
        mime = "unknown"

    raw = base64.b64decode(b64)
    filename = f"{filename_stem}.{ext}"
    path = Path(trace_dir) / filename

    with open(path, "wb") as f:
        f.write(raw)

    meta: dict[str, Any] = {
        "filename": filename,
        "mime": mime,
        "bytes": len(raw),
        "data_url_header": header,
    }

    try:
        with Image.open(path) as img:
            meta.update({
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
            })
    except Exception as exc:
        meta["image_open_error"] = str(exc)

    return meta


def sanitize_payload_for_trace(payload: dict[str, Any], sent_image_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    cloned = copy.deepcopy(payload)

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "image_url" in obj and isinstance(obj["image_url"], dict):
                url = str(obj["image_url"].get("url", ""))
                if url.startswith("data:image/"):
                    obj["image_url"]["url"] = {
                        "omitted": True,
                        "reason": "base64 image saved separately",
                        "saved_image": sent_image_meta,
                    }
            for k, v in list(obj.items()):
                obj[k] = walk(v)
            return obj

        if isinstance(obj, list):
            return [walk(v) for v in obj]

        return obj

    return walk(cloned)
