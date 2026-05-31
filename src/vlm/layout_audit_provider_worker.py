"""Provider worker for full-screenshot VLM layout audit."""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from src.vlm.provider import VLMSemanticProvider, VLMSemanticRequest
from src.vlm.roi_provider_worker import create_configured_roi_provider


LAYOUT_AUDIT_SYSTEM_PROMPT = """Layout audit fallback.

You receive one full screenshot only when local fast observe failed to produce
coarse layout regions. Return compact JSON only. Do not identify every control.
Do not return click points or safe action targets. You may suggest up to 8 coarse
review-only layout regions using relative_bounds in 0..1 screenshot space.
"""


def run_layout_audit_provider_job(
    job: dict[str, Any],
    *,
    screenshot: Image.Image,
    provider: VLMSemanticProvider | None = None,
    timeout_seconds: float | None = None,
    image_max_edge: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    provider = provider or create_configured_roi_provider()
    if provider is None or not provider.is_available():
        raise RuntimeError("layout_audit_provider_unavailable")

    image = _resize_max_edge(screenshot, image_max_edge or 512)
    provider_options: dict[str, Any] = {"response_contract": "layout_audit"}
    if timeout_seconds is not None:
        provider_options["timeout"] = max(0.1, float(timeout_seconds))
    request = VLMSemanticRequest(
        screenshot=image,
        system_prompt=LAYOUT_AUDIT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"type": "text", "text": _build_prompt(job)}]}],
        max_tokens=min(int(max_tokens or 384), provider.capabilities.max_tokens_limit),
        provider_options=provider_options,
    )
    raw = provider.analyze_page(request)
    if raw.finish_reason == "error":
        raise RuntimeError(raw.error or "layout_audit_provider_error")
    parsed = _parse_json(raw.raw_text)
    parsed["_provider"] = {
        "provider": raw.provider_name,
        "model": raw.model_name,
        "latency_ms": raw.latency_ms,
        "token_input": raw.token_input,
        "token_output": raw.token_output,
    }
    return parsed


def _build_prompt(job: dict[str, Any]) -> str:
    payload = {
        "task": "Describe coarse app layout only. Do not enumerate controls.",
        "canvas_id": str(job.get("canvas_id") or ""),
        "mode_guess": str(job.get("mode_guess") or ""),
        "quality_warnings": list(job.get("quality_warnings") or []),
        "existing_geometric_regions": list(job.get("existing_geometric_regions") or []),
        "rules": [
            "Return compact JSON only",
            "Use relative_bounds [left,top,right,bottom] in 0..1",
            "At most 8 layout_regions",
            "No click points",
            "No safe action targets",
        ],
        "json_shape": {
            "layout_summary": "short summary",
            "layout_regions": [
                {
                    "region_id": "optional local label",
                    "role": "left_navigation|top_bar|main_content|list_area|input_area|status_area|unknown",
                    "relative_bounds": [0.0, 0.0, 1.0, 1.0],
                    "confidence": 0.0,
                    "summary": "short review-only note",
                }
            ],
            "review_only_hints": ["optional"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    max_edge = int(max_edge)
    if max_edge <= 0:
        return image
    current_max = max(image.size)
    if current_max <= max_edge:
        return image
    scale = max_edge / current_max
    return image.resize(
        (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def _parse_json(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise RuntimeError("layout_audit_empty_response")
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("layout_audit_json_parse_failed")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("layout_audit_json_not_object")
    return parsed
