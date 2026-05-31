"""Provider worker for ROI-scoped VLM semantic supplement."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.vlm.provider import (
    VLMSemanticProvider,
    VLMSemanticRequest,
    create_modeler_provider,
)


ROI_SYSTEM_PROMPT = """ROI semantic supplement.

You receive one local ROI crop plus local candidate IDs. Return compact JSON
only. Do not create coordinates. Do not return bounds, boxes, click points, or
new candidate IDs. Only attach semantics to the given roi_id and candidate_ids.
"""


def create_configured_roi_provider() -> VLMSemanticProvider | None:
    """Create the configured semantic-modeler provider for ROI VLM calls."""
    from src.common.config_manager import load_config

    sm = load_config().semantic_modeler
    if not sm.enabled:
        return None
    return create_modeler_provider(
        provider=sm.provider,
        api_key=sm.api_key,
        endpoint=sm.endpoint,
        model=sm.model,
        provider_variant=sm.provider_variant,
        free_model_only=sm.free_model_only,
        thinking_mode="off",
        image_max_width=sm.image_max_width,
        proxy_url=sm.proxy_url,
        proxy_port=sm.proxy_port,
    )


def run_roi_vlm_provider_job(
    job: dict[str, Any],
    *,
    screenshot: Image.Image,
    provider: VLMSemanticProvider | None = None,
    timeout_seconds: float | None = None,
    profile: str = "generic",
    max_candidate_ids: int | None = None,
    image_max_edge: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Call a VLM provider for one ROI job and return parsed ROI JSON."""
    provider = provider or create_configured_roi_provider()
    if provider is None or not provider.is_available():
        raise RuntimeError("roi_vlm_provider_unavailable")

    profiled_job = _profile_job(job, profile=profile, max_candidate_ids=max_candidate_ids)
    crop = _crop_roi(screenshot, profiled_job.get("bounds") or [])
    if profiled_job.get("draw_markers", True):
        crop = _draw_candidate_markers(crop, profiled_job)
    if image_max_edge:
        crop = _resize_max_edge(crop, image_max_edge)
    provider_options: dict[str, Any] = {
        "response_contract": "roi_semantic_supplement",
        "roi_profile": profile,
    }
    if timeout_seconds is not None:
        provider_options["timeout"] = max(0.1, float(timeout_seconds))

    prompt_text = _build_user_prompt(profiled_job, profile=profile)
    request = VLMSemanticRequest(
        screenshot=crop,
        system_prompt=ROI_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
            }
        ],
        max_tokens=min(int(max_tokens or 1024), provider.capabilities.max_tokens_limit),
        provider_options=provider_options,
    )
    raw = provider.analyze_page(request)
    if raw.finish_reason == "error":
        raise RuntimeError(raw.error or "roi_vlm_provider_error")

    try:
        parsed = _parse_json(raw.raw_text)
    except Exception as exc:
        debug_artifacts = _write_debug_bundle(
            profiled_job,
            crop=crop,
            prompt_text=prompt_text,
            raw_text=raw.raw_text,
            parsed={
                "error": "roi_vlm_json_parse_failed",
                "detail": str(exc),
                "roi_id": str(profiled_job.get("roi_id") or ""),
            },
        )
        debug_dir = str(debug_artifacts.get("dir") or "") if debug_artifacts else ""
        suffix = f"; debug_dir={debug_dir}" if debug_dir else ""
        raise RuntimeError(f"roi_vlm_json_parse_failed{suffix}") from exc
    parsed.setdefault("roi_id", str(profiled_job.get("roi_id") or ""))
    _resolve_marker_annotations(parsed, profiled_job)
    debug_artifacts = _write_debug_bundle(profiled_job, crop=crop, prompt_text=prompt_text, raw_text=raw.raw_text, parsed=parsed)
    parsed["_provider"] = {
        "provider": raw.provider_name,
        "model": raw.model_name,
        "latency_ms": raw.latency_ms,
        "token_input": raw.token_input,
        "token_output": raw.token_output,
    }
    if debug_artifacts:
        parsed["_debug_artifacts"] = debug_artifacts
    return parsed


def _build_user_prompt(job: dict[str, Any], profile: str = "generic") -> str:
    if profile == "fast":
        return _build_fast_prompt(job)
    if profile == "scene":
        return _build_scene_prompt(job)

    payload = {
        "roi_id": str(job.get("roi_id") or ""),
        "purpose": str(job.get("purpose") or ""),
        "mode": str(job.get("mode") or ""),
        "candidate_ids": list(job.get("candidate_ids") or []),
        "candidate_refs": _prompt_candidate_refs(job),
        "allowed_outputs": list(job.get("allowed_outputs") or []),
        "required_json_shape": {
            "roi_id": "same local roi_id",
            "region_semantics": {"role": "short semantic role", "summary": "optional"},
            "candidate_annotations": [
                {"marker": "visible marker", "candidate_id": "one of candidate_ids", "label": "semantic label", "role": "optional"}
            ],
            "review_only_hints": ["optional hints"],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_fast_prompt(job: dict[str, Any]) -> str:
    region_options, role_options = _semantic_options(job)
    candidate_ids = list(job.get("candidate_ids") or [])
    max_candidate_annotations = len(candidate_ids)
    is_qq_message_stream = (
        str(job.get("app_process") or "").lower() == "qq.exe"
        and str(job.get("mode") or "") == "chat_workspace"
        and str(job.get("purpose") or "") == "message_stream"
    )
    json_keys = [
        "roi_id",
        "region_semantics.role",
    ]
    if is_qq_message_stream:
        json_keys.append("region_semantics.summary")
    if candidate_ids:
        json_keys.extend([
            "candidate_annotations[].marker",
            "candidate_annotations[].candidate_id",
            "candidate_annotations[].role",
            "candidate_annotations[].label",
        ])
    payload = {
        "task": _task_hint(job),
        "roi_id": str(job.get("roi_id") or ""),
        "mode": str(job.get("mode") or ""),
        "purpose": str(job.get("purpose") or ""),
        "app_process": str(job.get("app_process") or ""),
        "window_title": str(job.get("window_title") or ""),
        "candidate_ids": candidate_ids,
        "candidate_refs": _prompt_candidate_refs(job),
        "candidate_annotation_policy": (
            f"annotate_up_to_{max_candidate_annotations}_candidate_ids"
            if candidate_ids
            else "return_empty_array"
        ),
        "max_candidate_annotations": max_candidate_annotations,
        "rules": [
            "Return compact JSON only",
            "Copy no placeholder values",
            "Use only given candidate_ids",
            "Use visible marker labels, not list order",
            "If a candidate has a marker, copy marker and matching candidate_id from candidate_refs",
            "local_hint is only a weak prior; override it when visual evidence disagrees",
            "Do not annotate a candidate if the best label is only icon/action/item/button",
            "If candidate_ids is empty, candidate_annotations must be []",
            "Return up to max_candidate_annotations candidate annotations",
            "No coordinates",
            "Use region_options and role_options when possible",
        ] + (
            [
                "For QQ message_stream, region_semantics.role must be one of private_chat_area, group_chat_area, unknown_chat_area",
                "For QQ message_stream, set summary to the visible evidence for private/group/unknown",
            ]
            if is_qq_message_stream
            else []
        ),
        "json_keys": json_keys,
        "region_options": region_options,
        "role_options": role_options,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_scene_prompt(job: dict[str, Any]) -> str:
    payload = {
        "scene": f"{job.get('mode')}/{job.get('purpose')}",
        "goal": _task_hint(job),
        "rules": [
            "Use only candidate_ids",
            "No coordinates",
            "Return <=8 annotations",
        ],
        "roi_id": str(job.get("roi_id") or ""),
        "candidate_ids": list(job.get("candidate_ids") or []),
        "candidate_refs": _prompt_candidate_refs(job),
        "json": {
            "roi_id": "same",
            "region_semantics": {"role": "scene role"},
            "candidate_annotations": [
                {"marker": "visible marker", "candidate_id": "id", "role": "action|input|item|label|status", "label": "short"}
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _task_hint(job: dict[str, Any]) -> str:
    mode = str(job.get("mode") or "")
    purpose = str(job.get("purpose") or "")
    app_process = str(job.get("app_process") or "").lower()
    if app_process == "qq.exe" and mode == "chat_workspace" and purpose == "message_stream":
        return (
            "Identify QQ chat content area and classify active chat variant as "
            "private_chat_area, group_chat_area, or unknown_chat_area."
        )
    if mode in {"chat_workspace", "chat_document"} and purpose in {"navigation_and_list", "conversation_sidebar"}:
        return "Identify the conversation/sidebar navigation list and key chat/contact items."
    if mode in {"chat_workspace", "chat_document", "collaboration_inbox"} and purpose == "composer":
        return "Identify the message composer/input area and send/attachment toolbar candidates."
    if purpose in {"message_stream", "document_thread"} or mode in {"chat_workspace", "chat_document"}:
        return "Identify chat/read area and key message/copy/send/input candidates."
    if purpose in {"table_or_list", "pagination_status"} or mode == "list_management":
        return "Identify table/list rows, filters, pagination, and row action candidates."
    if mode in {"control_dashboard", "control_matrix"}:
        return "Identify dashboard cards, switches, status, and main control candidates."
    if mode == "media_home":
        return "Identify media list/player/search action candidates."
    if mode == "security_dashboard":
        return "Identify security status, scan/protect feature, and action candidates."
    if mode == "file_search":
        return "Identify file search input, result list rows, filters, pagination, and status candidates."
    if mode == "local_transfer_dashboard":
        return "Identify local transfer device status, send/receive actions, and transfer history candidates."
    if mode == "account_switcher":
        return "Identify account entries, login/switch actions, and window control candidates."
    if mode == "browser_profile_manager":
        return "Identify profile/environment rows, navigation, filters, and row action candidates."
    if mode == "remote_access_dashboard":
        return "Identify remote connection panel, ID/password/status, and connect action candidates."
    if mode == "media_video_home":
        return "Identify video navigation/search, media cards, playback, and bottom action candidates."
    return "Identify the ROI role and annotate key actionable candidates."


def _semantic_options(job: dict[str, Any]) -> tuple[list[str], list[str]]:
    mode = str(job.get("mode") or "")
    purpose = str(job.get("purpose") or "")
    app_process = str(job.get("app_process") or "").lower()
    generic_roles = ["action", "input", "item", "status", "tab", "filter", "pagination", "unknown"]
    if mode == "chat_workspace":
        if purpose == "navigation_and_list":
            return (
                ["navigation_list", "conversation_sidebar", "contact_list", "search_bar", "toolbar"],
                ["action", "input", "conversation", "contact", "item", "status", "tab", "unknown"],
            )
        if purpose == "message_stream":
            if app_process == "qq.exe":
                return (
                    ["private_chat_area", "group_chat_area", "chat_area", "message_stream", "unknown_chat_area"],
                    ["message", "link", "media", "status", "unknown"],
                )
            return (
                ["chat_area", "message_stream", "document_preview", "timeline"],
                ["action", "message", "link", "media", "status", "unknown"],
            )
        if purpose == "composer":
            return (
                ["composer", "toolbar"],
                ["action", "input", "send", "attachment", "emoji", "voice", "unknown"],
            )
    if mode == "chat_document":
        if purpose == "navigation_and_list":
            return (
                ["conversation_sidebar", "navigation_list", "search_bar", "toolbar"],
                ["action", "input", "conversation", "item", "status", "unknown"],
            )
        if purpose == "composer":
            return (
                ["composer", "toolbar"],
                ["action", "input", "send", "attachment", "unknown"],
            )
    if mode == "collaboration_inbox":
        if purpose == "app_rail":
            return (
                ["app_rail", "workspace_navigation", "toolbar"],
                ["action", "tab", "workspace", "status", "unknown"],
            )
        if purpose == "inbox_list":
            return (
                ["inbox_list", "conversation_list", "notification_list", "search_bar"],
                ["action", "input", "conversation", "notification", "item", "status", "unknown"],
            )
        if purpose == "message_thread":
            return (
                ["message_thread", "document_preview", "detail_pane"],
                ["action", "message", "link", "attachment", "status", "unknown"],
            )
        if purpose == "composer":
            return (
                ["composer", "toolbar"],
                ["action", "input", "send", "attachment", "emoji", "voice", "unknown"],
            )
    if mode == "security_dashboard":
        if purpose == "security_status":
            return (
                ["security_status", "protection_summary", "scan_status"],
                ["action", "status", "security_feature", "toggle", "unknown"],
            )
        if purpose == "feature_grid":
            return (
                ["feature_grid", "security_feature_list"],
                ["action", "security_feature", "toggle", "status", "item", "unknown"],
            )
        if purpose == "bottom_actions":
            return (
                ["bottom_actions", "toolbar"],
                ["action", "status", "security_feature", "unknown"],
            )
    if mode == "control_matrix":
        if purpose in {"left_control_columns", "middle_control_columns"}:
            return (
                ["control_columns", "mixer_matrix", "input_strip", "bus_strip"],
                ["action", "toggle", "slider", "status", "item", "unknown"],
            )
        if purpose == "right_master_section":
            return (
                ["master_section", "output_bus_controls", "status_panel"],
                ["action", "toggle", "slider", "status", "item", "unknown"],
            )
    by_mode: dict[str, tuple[list[str], list[str]]] = {
        "chat_workspace": (
            ["navigation_list", "chat_area", "message_stream", "composer", "toolbar"],
            ["action", "input", "item", "message", "status", "tab", "unknown"],
        ),
        "chat_document": (
            ["conversation_sidebar", "document_thread", "composer", "toolbar"],
            ["action", "input", "item", "message", "status", "copy", "unknown"],
        ),
        "collaboration_inbox": (
            ["app_rail", "inbox_list", "detail_pane", "message_thread", "composer", "toolbar"],
            ["action", "input", "conversation", "message", "notification", "status", "unknown"],
        ),
        "list_management": (
            ["left_navigation", "toolbar", "filter_bar", "table", "list", "pagination"],
            ["action", "input", "item", "row", "filter", "pagination", "status", "unknown"],
        ),
        "browser_profile_manager": (
            ["left_navigation", "profile_table", "profile_cards", "filter_bar", "pagination"],
            ["action", "input", "profile", "row", "filter", "pagination", "status", "unknown"],
        ),
        "file_search": (
            ["search_bar", "result_list", "file_list", "filter_bar", "status_bar", "pagination"],
            ["action", "input", "file", "folder", "result", "filter", "pagination", "status", "unknown"],
        ),
        "local_transfer_dashboard": (
            ["device_status", "send_receive_actions", "transfer_history", "settings"],
            ["action", "device", "file", "transfer", "status", "toggle", "unknown"],
        ),
        "account_switcher": (
            ["account_list", "login_panel", "login_actions", "window_controls"],
            ["action", "account", "input", "login", "status", "toggle", "unknown"],
        ),
        "remote_access_dashboard": (
            ["connection_panel", "remote_control_actions", "status_or_recent", "toolbar"],
            ["action", "input", "connection_id", "password", "status", "recent_item", "unknown"],
        ),
        "media_home": (
            ["left_navigation", "search_bar", "content_feed", "playlist_grid", "bottom_player"],
            ["action", "input", "media_item", "playlist", "media_control", "status", "unknown"],
        ),
        "media_video_home": (
            ["top_navigation_search", "media_feed", "video_card", "player_or_bottom_actions"],
            ["action", "input", "video", "media_item", "media_control", "tab", "status", "unknown"],
        ),
        "control_dashboard": (
            ["left_navigation", "mode_tabs", "dashboard", "status_panel", "card_grid"],
            ["action", "toggle", "tab", "status", "item", "unknown"],
        ),
        "control_matrix": (
            ["control_columns", "mixer_matrix", "master_section", "status_panel"],
            ["action", "toggle", "slider", "status", "item", "unknown"],
        ),
        "security_dashboard": (
            ["security_status", "feature_grid", "scan_actions", "bottom_actions"],
            ["action", "status", "security_feature", "toggle", "item", "unknown"],
        ),
        "archive_file_manager": (
            ["menu_toolbar", "file_list", "status_bar", "path_bar"],
            ["action", "file", "folder", "item", "status", "unknown"],
        ),
    }
    return by_mode.get(mode, (["toolbar", "main_content", "list", "status_bar"], generic_roles))


def _profile_job(
    job: dict[str, Any],
    *,
    profile: str,
    max_candidate_ids: int | None,
) -> dict[str, Any]:
    next_job = dict(job)
    candidate_ids = list(job.get("candidate_ids") or [])
    if profile in {"fast", "scene"} and max_candidate_ids is not None:
        limit = max(0, int(max_candidate_ids))
        candidate_ids = candidate_ids[:limit]
    next_job["candidate_ids"] = candidate_ids
    next_job["candidate_refs"] = _candidate_refs_for_ids(job, candidate_ids)
    _tighten_chat_composer_icon_bounds(next_job)
    return next_job


def _tighten_chat_composer_icon_bounds(job: dict[str, Any]) -> None:
    mode = str(job.get("mode") or "")
    purpose = str(job.get("purpose") or "")
    if mode not in {"chat_workspace", "chat_document", "collaboration_inbox"} or purpose != "composer":
        return
    refs = list(job.get("candidate_refs") or [])
    if len(refs) < 2:
        return
    if not all(str(ref.get("control_type") or "").strip().lower() == "icon" for ref in refs):
        return
    bounds_list = [_coerce_ref_bounds(ref.get("bounds")) for ref in refs]
    if any(bounds is None for bounds in bounds_list):
        return
    left = min(bounds[0] for bounds in bounds_list if bounds is not None)
    top = min(bounds[1] for bounds in bounds_list if bounds is not None)
    right = max(bounds[2] for bounds in bounds_list if bounds is not None)
    bottom = max(bounds[3] for bounds in bounds_list if bounds is not None)
    padding = 16
    job["bounds"] = [left - padding, top - padding, right + padding, bottom + padding]
    job["crop_policy"] = "candidate_union"


def _crop_roi(screenshot: Image.Image, bounds: list[Any]) -> Image.Image:
    if len(bounds) != 4:
        return screenshot
    left, top, right, bottom = [int(v) for v in bounds[:4]]
    left = max(0, min(screenshot.width, left))
    right = max(0, min(screenshot.width, right))
    top = max(0, min(screenshot.height, top))
    bottom = max(0, min(screenshot.height, bottom))
    if right <= left or bottom <= top:
        return screenshot
    return screenshot.crop((left, top, right, bottom))


def _coerce_ref_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        left, top, right, bottom = [int(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    max_edge = int(max_edge)
    if max_edge <= 0:
        return image
    current_max = max(image.size)
    if current_max <= max_edge:
        return image
    scale = max_edge / current_max
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _candidate_refs_for_ids(job: dict[str, Any], candidate_ids: list[Any]) -> list[dict[str, Any]]:
    allowed = {str(item) for item in candidate_ids}
    refs = [
        dict(ref)
        for ref in list(job.get("candidate_refs") or [])
        if str(ref.get("candidate_id") or "") in allowed
    ]
    if refs:
        return refs
    return [
        {"marker": f"C{idx + 1}", "candidate_id": str(candidate_id)}
        for idx, candidate_id in enumerate(candidate_ids)
    ]


def _prompt_candidate_refs(job: dict[str, Any]) -> list[dict[str, Any]]:
    refs = list(job.get("candidate_refs") or [])
    if not refs:
        refs = _candidate_refs_for_ids(job, list(job.get("candidate_ids") or []))
    prompt_refs: list[dict[str, Any]] = []
    for ref in refs:
        marker = str(ref.get("marker") or "")
        candidate_id = str(ref.get("candidate_id") or "")
        if not marker or not candidate_id:
            continue
        item = {"marker": marker, "candidate_id": candidate_id}
        control_type = str(ref.get("control_type") or "").strip()
        if control_type:
            item["control_type"] = control_type
        text = str(ref.get("text") or "").strip()
        if text:
            item["text"] = text[:40]
        local_hint = str(ref.get("local_hint") or "").strip()
        if local_hint:
            item["local_hint"] = local_hint[:40]
        prompt_refs.append(item)
    return prompt_refs


def _draw_candidate_markers(image: Image.Image, job: dict[str, Any]) -> Image.Image:
    refs = list(job.get("candidate_refs") or [])
    if not refs:
        return image
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    roi_left, roi_top = _roi_origin(job.get("bounds") or [])
    for ref in refs:
        marker = str(ref.get("marker") or "")
        bounds = _marker_bounds(ref.get("bounds"), roi_left=roi_left, roi_top=roi_top, image=marked)
        if not marker or bounds is None:
            continue
        left, top, right, bottom = bounds
        draw.rectangle((left, top, right, bottom), outline=(255, 64, 64), width=2)
        label_w = max(18, 8 * len(marker) + 8)
        label_h = 16
        label_left = max(0, min(marked.width - label_w, left))
        label_top = max(0, top - label_h)
        draw.rectangle((label_left, label_top, label_left + label_w, label_top + label_h), fill=(255, 64, 64))
        draw.text((label_left + 3, label_top + 2), marker, fill=(255, 255, 255))
    return marked


def _roi_origin(bounds: list[Any]) -> tuple[int, int]:
    if len(bounds) != 4:
        return (0, 0)
    try:
        return (int(bounds[0]), int(bounds[1]))
    except (TypeError, ValueError):
        return (0, 0)


def _marker_bounds(
    value: Any,
    *,
    roi_left: int,
    roi_top: int,
    image: Image.Image,
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        left, top, right, bottom = [int(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    local = [left - roi_left, top - roi_top, right - roi_left, bottom - roi_top]
    if local[2] <= 0 or local[3] <= 0 or local[0] >= image.width or local[1] >= image.height:
        local = [left, top, right, bottom]
    left = max(0, min(image.width - 1, local[0]))
    top = max(0, min(image.height - 1, local[1]))
    right = max(0, min(image.width, local[2]))
    bottom = max(0, min(image.height, local[3]))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _resolve_marker_annotations(parsed: dict[str, Any], job: dict[str, Any]) -> None:
    marker_to_candidate = {
        str(ref.get("marker") or ""): str(ref.get("candidate_id") or "")
        for ref in list(job.get("candidate_refs") or [])
        if ref.get("marker") and ref.get("candidate_id")
    }
    if not marker_to_candidate:
        return
    for annotation in list(parsed.get("candidate_annotations") or []):
        if not isinstance(annotation, dict):
            continue
        marker = str(annotation.get("marker") or "")
        resolved = marker_to_candidate.get(marker)
        if resolved:
            annotation["candidate_id"] = resolved


def _write_debug_bundle(
    job: dict[str, Any],
    *,
    crop: Image.Image,
    prompt_text: str,
    raw_text: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    root_value = os.environ.get("OPENCLAW_ROI_VLM_DEBUG_DIR", "data/debug/roi_vlm")
    if str(root_value).strip().lower() in {"", "0", "false", "off"}:
        return {}
    try:
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        canvas_id = _safe_path_part(str(job.get("canvas_id") or "canvas"))
        roi_id = _safe_path_part(str(job.get("roi_id") or "roi"))
        debug_dir = root / f"{timestamp}_{canvas_id}_{roi_id}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        crop.save(debug_dir / "sent.png")
        (debug_dir / "prompt.json").write_text(_format_json_text(prompt_text), encoding="utf-8")
        (debug_dir / "raw.txt").write_text(raw_text or "", encoding="utf-8")
        (debug_dir / "parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        (debug_dir / "candidate_refs.json").write_text(
            json.dumps(list(job.get("candidate_refs") or []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "dir": str(debug_dir),
            "sent_image": str(debug_dir / "sent.png"),
            "prompt": str(debug_dir / "prompt.json"),
            "raw": str(debug_dir / "raw.txt"),
            "parsed": str(debug_dir / "parsed.json"),
            "candidate_refs": str(debug_dir / "candidate_refs.json"),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the worker.
        return {"error": str(exc)}


def _format_json_text(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text or ""


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:80] or "unknown"


def _parse_json(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise RuntimeError("roi_vlm_empty_response")
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("roi_vlm_json_parse_failed")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("roi_vlm_json_not_object")
    return parsed
