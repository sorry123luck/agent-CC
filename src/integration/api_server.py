"""
API 服务器

核心协议主干：observe → query → diff → remember → feedback
每个端点都接真实引擎，不做占位返回。
"""

from contextlib import asynccontextmanager
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Any

from src.common.logger import setup_logger
from src.storage.db import init_db
from src.indexer.catalog_service import CatalogService
from src.canvas.canvas_cache import get_canvas_cache
from src.integration.api_models import (
    ActRequest,
    ActResponse,
    CanvasDetail,
    CanvasProcessingResponse,
    CanvasSnapshotResponse,
    CanvasSummary,
    CandidateOverrideResponse,
    CandidateOverrideUpsertRequest,
    CapabilitiesResponse,
    CandidateResponse,
    ControlTransitionGraphResponse,
    ControlTransitionRecordRequest,
    ControlTransitionResponse,
    DiffRequest,
    DiffResponse,
    FeedbackRequest,
    FeedbackResponse,
    GeometricRegionResponse,
    JobStatusResponse,
    LaunchTargetResponse,
    LayoutAuditVlmRequest,
    LayoutAuditVlmResponse,
    ObserveRequest,
    ObserveResponse,
    PageModelResponse,
    PageModelTreeResponse,
    ProviderTraceResponse,
    QueryRequest,
    QueryResponse,
    ReadRegionRequest,
    ReadRegionResponse,
    ReadRegionTextBlock,
    RefineRequest,
    RefineResponse,
    RefineResultItem,
    RegionResponse,
    RememberRequest,
    RememberResponse,
    RoiVlmSupplementRequest,
    RoiVlmSupplementResponse,
    ScrollRegionRequest,
    ScrollRegionResponse,
    SemanticCompletionRequest,
    SemanticCompletionResponse,
    SemanticModelerModelInfo,
    SemanticModelerModelListRequest,
    SemanticModelerModelListResponse,
    SemanticModelerSettingsResponse,
    SemanticModelerSettingsUpdateRequest,
    SemanticModelerTestRequest,
    SemanticModelerTestResponse,
    StateTemplateResponse,
    TransitionEdgeResponse,
    TransitionGraphResponse,
    VirtualModelCandidate,
    VirtualModelDetail,
    VLMSemanticRequest,
    VLMSemanticResponse,
    WindowCandidateResponse,
    WindowListItem,
    WindowResolveRequest,
    WindowResolveResponse,
)

setup_logger()
logger = logging.getLogger(__name__)


def _warm_up_runtime_services() -> None:
    """Preload heavy local vision/OCR services after API startup."""
    if os.environ.get("OPENCLAW_DISABLE_RUNTIME_WARMUP") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        logger.info("Runtime warmup skipped for test environment")
        return

    def _worker() -> None:
        from src.perception.ocr_service import OCRService
        from src.perception.providers.remote_vision_provider import OmniParserRemoteVisionProvider

        try:
            vision_status = OmniParserRemoteVisionProvider().warm_up()
            logger.info("Runtime warmup vision: %s", vision_status)
        except Exception:
            logger.warning("Runtime warmup vision failed", exc_info=True)

        try:
            ocr_result = OCRService().warm_up()
            logger.info(
                "Runtime warmup OCR: success=%s mode=%s reused=%s startup=%s elapsed=%s error=%s",
                ocr_result.success,
                ocr_result.worker_mode,
                ocr_result.worker_reused,
                ocr_result.startup_seconds,
                ocr_result.elapsed_seconds,
                ocr_result.error,
            )
        except Exception:
            logger.warning("Runtime warmup OCR failed", exc_info=True)

    threading.Thread(target=_worker, name="openclaw-runtime-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db = init_db("data/openclaw.db")
    db.create_all()
    from src.integration.processing_state import registry
    registry.recover_incomplete_jobs()
    _warm_up_runtime_services()
    yield


app = FastAPI(
    title="OpenClaw Desktop Agent API",
    version="1.0.0",
    description="桌面智能代理平台的 HTTP API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 本地请求/响应模型 =====

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class LaunchRequest(BaseModel):
    name: str

class SearchResponse(BaseModel):
    success: bool
    results: list[dict]
    total: int

class LaunchResponse(BaseModel):
    success: bool
    app_name: str
    message: str
    hwnd: int | None = None


# ===== 辅助函数 =====

def _candidate_to_response(c) -> CandidateResponse:
    """Convert a Candidate dataclass to the API response model."""
    role = c.semantic_role.value if hasattr(c.semantic_role, "value") else str(c.semantic_role)
    return CandidateResponse(
        element_id=c.element_id,
        semantic_role=role,
        text=c.text or "",
        name=c.name,
        control_type=c.control_type or "",
        bounds=list(c.bounds) if c.bounds else None,
        click_point=list(c.click_point) if c.click_point else None,
        confidence=c.confidence,
        confidence_level=c.confidence_level.value if hasattr(c.confidence_level, "value") else str(c.confidence_level),
        risk_tags=c.risk_tags,
        risk_level=c.risk_level.value if hasattr(c.risk_level, "value") else str(c.risk_level),
        provider_sources=c.provider_sources,
        interactable=c.interactable,
        from_memory=c.from_memory,
        suggest_confirm=c.suggest_confirm,
        region_id=c.region_id,
        locator_ids=c.locator_ids,
        attributes=c.attributes if c.attributes else {},
        visual_type=c.visual_type or "",
        semantic_tags=c.semantic_tags or [],
        role_label=c.role_label,
        role_confidence=c.role_confidence,
        role_source=c.role_source or "",
        role_evidence=c.role_evidence or [],
        refine_status=c.refine_status or "unreviewed",
        stable_key_id=c.stable_key_id,
    )


def _effective_canvas(canvas):
    """Apply the effective candidate read model for API consumers."""
    from src.memory.effective_candidate_service import EffectiveCandidateService

    return EffectiveCandidateService().apply(canvas)


def _geometric_regions_to_response(geometric_regions: list | None) -> list[GeometricRegionResponse]:
    """Convert GeometricRegion dataclasses to API response models."""
    if not geometric_regions:
        return []
    result = []
    for gr in geometric_regions:
        d = gr.to_dict() if hasattr(gr, "to_dict") else gr
        if isinstance(d, dict):
            result.append(GeometricRegionResponse(
                region_id=d.get("region_id", ""),
                bounds=d.get("bounds", []),
                boundary_evidence=d.get("boundary_evidence", []),
                density_profile=list(d.get("density_profile", (0.0, 0.0))),
                geometry_confidence=d.get("geometry_confidence", 0.0),
                parent_region_id=d.get("parent_region_id"),
            ))
    return result


def _canvas_to_operability_row(canvas) -> dict[str, Any]:
    """Build a sample-matrix-like row from the current in-memory canvas."""
    artifacts = canvas.artifacts if isinstance(getattr(canvas, "artifacts", None), dict) else {}
    roi_plan = artifacts.get("roi_selection_plan") if isinstance(artifacts.get("roi_selection_plan"), dict) else {}
    visual_pattern = artifacts.get("visual_pattern") if isinstance(artifacts.get("visual_pattern"), dict) else {}
    purposes = _operability_roi_purposes(canvas, roi_plan)
    input_candidates = _operability_input_candidates(canvas)
    send_candidate = _operability_send_candidate(canvas)
    search_candidate = _operability_search_candidate(canvas)
    observe_timing = artifacts.get("observe_timing") if isinstance(artifacts.get("observe_timing"), dict) else {}
    total_seconds = observe_timing.get("total_seconds") or 0
    try:
        observe_elapsed_ms = int(float(total_seconds) * 1000)
    except (TypeError, ValueError):
        observe_elapsed_ms = 0
    process_name = str(getattr(getattr(canvas, "app", None), "process_name", "") or "")
    sample = str(getattr(canvas, "canvas_id", "") or "")
    row: dict[str, Any] = {
        "sample": sample,
        "canvas_id": sample,
        "process_name": process_name,
        "title": str(getattr(getattr(canvas, "window", None), "title", "") or ""),
        "mode": str(roi_plan.get("mode") or visual_pattern.get("mode") or getattr(canvas, "page_class", "") or ""),
        "chat_variant": str(artifacts.get("chat_variant") or ""),
        "observe_elapsed_ms": observe_elapsed_ms,
        "element_count": len(getattr(canvas, "elements", None) or []),
        "region_count": len(getattr(canvas, "regions", None) or []),
        "roi_purposes": purposes,
        "quality_warnings": _operability_quality_warnings(canvas),
        "composer_input_candidate_count": len(input_candidates),
        "composer_input_safe_count": len([item for item in input_candidates if item.get("safe_to_type") is True]),
        "composer_input_review_count": len([item for item in input_candidates if item.get("safe_to_type") is not True]),
        "composer_input_review_click_points": [item["click_point"] for item in input_candidates if item.get("click_point")],
        "composer_input_primary_click_point": input_candidates[0].get("click_point") if input_candidates else [],
        "read_region_contexts": _operability_read_region_contexts(canvas),
    }
    if send_candidate:
        bounds = send_candidate.get("bounds") or []
        row.update(
            {
                "composer_vlm_send_hint_count": 1,
                "composer_send_target_candidate_id": send_candidate.get("element_id", ""),
                "composer_send_target_bounds": bounds,
                "composer_send_target_width": _bounds_width(bounds),
                "composer_send_target_height": _bounds_height(bounds),
            }
        )
    if search_candidate:
        row.update(
            {
                "search_candidate_count": 1,
                "search_primary_click_point": search_candidate.get("click_point") or [],
                "search_bounds": search_candidate.get("bounds") or [],
                "search_source": search_candidate.get("source") or "candidate",
                "search_evidence": search_candidate.get("evidence") or "",
            }
        )
    return row


def _canvas_to_operability_acceptance(row: dict[str, Any], canvas) -> dict[str, Any]:
    warnings = list(row.get("quality_warnings") or [])
    failures: list[str] = []
    quality = canvas.artifacts.get("perception_quality") if isinstance(getattr(canvas, "artifacts", None), dict) else {}
    quality_status = str((quality or {}).get("status") or "").lower() if isinstance(quality, dict) else ""
    if quality_status in {"failed", "unreliable"}:
        failures.append("perception_quality_unreliable")
    status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "sample": row.get("sample", ""),
        "process_name": row.get("process_name", ""),
        "status": status,
        "failures": failures,
        "warnings": warnings,
    }


def _operability_roi_purposes(canvas, roi_plan: dict[str, Any]) -> list[str]:
    rois = [item for item in list(roi_plan.get("rois") or []) if isinstance(item, dict)]
    purposes = [str(item.get("purpose") or "").strip() for item in rois if str(item.get("purpose") or "").strip()]
    if purposes:
        return purposes
    return [
        str(getattr(region, "role", "") or "").strip()
        for region in list(getattr(canvas, "regions", None) or [])
        if str(getattr(region, "role", "") or "").strip()
    ]


def _operability_input_candidates(canvas) -> list[dict[str, Any]]:
    result = []
    for candidate in list(getattr(canvas, "elements", None) or []):
        role = _candidate_role(candidate)
        if role not in {"message_input", "text_input", "search_input"}:
            continue
        attrs = getattr(candidate, "attributes", None) or {}
        if role == "search_input":
            continue
        result.append(
            {
                "element_id": getattr(candidate, "element_id", ""),
                "bounds": list(getattr(candidate, "bounds", None) or []),
                "click_point": list(getattr(candidate, "click_point", None) or []),
                "safe_to_type": bool(attrs.get("safe_to_type")) if attrs.get("safe_to_type") is True else False,
            }
        )
    return result


def _operability_send_candidate(canvas) -> dict[str, Any]:
    candidates = []
    for candidate in list(getattr(canvas, "elements", None) or []):
        role = _candidate_role(candidate)
        risk_tags = {str(item) for item in list(getattr(candidate, "risk_tags", None) or [])}
        label = str(getattr(candidate, "role_label", "") or "").lower()
        if role == "send_button" or "send" in risk_tags or "发送" in label or "send" in label:
            candidates.append(candidate)
    if not candidates:
        return {}
    candidate = max(candidates, key=lambda item: _bounds_area(list(getattr(item, "bounds", None) or [])))
    attrs = getattr(candidate, "attributes", None) or {}
    return {
        "element_id": getattr(candidate, "element_id", ""),
        "bounds": list(getattr(candidate, "bounds", None) or []),
        "click_point": list(getattr(candidate, "click_point", None) or []),
        "actionability": str(attrs.get("actionability") or "controlled_probe"),
    }


def _operability_search_candidate(canvas) -> dict[str, Any]:
    for candidate in list(getattr(canvas, "elements", None) or []):
        role = _candidate_role(candidate)
        text = f"{getattr(candidate, 'text', '') or ''} {getattr(candidate, 'name', '') or ''} {getattr(candidate, 'role_label', '') or ''}".lower()
        if role == "search_input" or "search" in text or "搜索" in text:
            return {
                "element_id": getattr(candidate, "element_id", ""),
                "bounds": list(getattr(candidate, "bounds", None) or []),
                "click_point": list(getattr(candidate, "click_point", None) or []),
                "source": "candidate",
                "evidence": text.strip(),
            }
    return {}


def _operability_quality_warnings(canvas) -> list[str]:
    artifacts = canvas.artifacts if isinstance(getattr(canvas, "artifacts", None), dict) else {}
    quality = artifacts.get("perception_quality") if isinstance(artifacts.get("perception_quality"), dict) else {}
    warnings = [str(item) for item in list(quality.get("warnings") or []) if str(item)]
    if getattr(canvas, "partial", False):
        warnings.append("canvas_partial")
    if getattr(canvas, "loading", False):
        warnings.append("canvas_loading")
    return list(dict.fromkeys(warnings))


def _operability_read_region_contexts(canvas) -> dict[str, dict[str, Any]]:
    regions_by_id = {
        str(getattr(region, "region_id", "") or ""): region
        for region in list(getattr(canvas, "regions", None) or [])
    }
    result: dict[str, dict[str, Any]] = {}
    for context in list(getattr(canvas, "scroll_contexts", None) or []):
        region_id = str(getattr(context, "region_id", "") or "")
        region = regions_by_id.get(region_id)
        role = str(getattr(region, "role", "") or "")
        if not role:
            continue
        result[role] = {
            "region_id": region_id,
            "scroll_context_id": str(getattr(context, "scroll_context_id", "") or ""),
            "scroll_type": str(getattr(context, "scroll_type", "") or "vertical"),
            "is_virtual": bool(getattr(context, "is_virtual", False)),
        }
    return result


def _read_region_scroll_context(canvas, region) -> dict[str, Any] | None:
    region_id = str(getattr(region, "region_id", "") or "")
    for context in list(getattr(canvas, "scroll_contexts", None) or []):
        if str(getattr(context, "region_id", "") or "") != region_id:
            continue
        scroll_context_id = str(getattr(context, "scroll_context_id", "") or "")
        if not scroll_context_id:
            return None
        return {
            "scroll_context_id": scroll_context_id,
            "region_id": region_id,
            "scroll_type": str(getattr(context, "scroll_type", "") or "vertical"),
            "is_virtual": bool(getattr(context, "is_virtual", False)),
        }
    return None


def _current_viewport_long_content_strategy(scroll_context: dict[str, Any] | None) -> dict[str, str]:
    return {
        "status": "current_viewport_only",
        "reason": "controlled_scroll_available" if scroll_context else "no_scroll_context",
    }


def _scroll_region_action_plan(region, scroll_context: dict[str, Any], request: ScrollRegionRequest) -> dict[str, Any]:
    bounds = list(getattr(region, "bounds", None) or [])
    return {
        "action": "scroll",
        "direction": str(request.direction or "down"),
        "amount": str(request.amount or "page"),
        "region_bounds": [int(item) for item in bounds] if len(bounds) == 4 else [],
        "scroll_context_id": str(scroll_context.get("scroll_context_id") or ""),
        "follow_up": "observe_then_read_region",
    }


def _scroll_region_delta(request: ScrollRegionRequest) -> int:
    amount = str(request.amount or "page").lower()
    steps = 4 if amount == "page" else 1
    direction = str(request.direction or "down").lower()
    sign = -1 if direction in {"down", "right"} else 1
    return sign * 120 * steps


def _scroll_region_center(region) -> tuple[int, int]:
    bounds = list(getattr(region, "bounds", None) or [])
    if len(bounds) != 4:
        return 0, 0
    left, top, right, bottom = [int(item) for item in bounds]
    return (left + right) // 2, (top + bottom) // 2


def _read_region_suggested_next_actions(region, scroll_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if scroll_context is None:
        return []
    return [
        {
            "action": "scroll_region_dry_run",
            "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
            "request": {
                "region_role": str(getattr(region, "role", "") or ""),
                "direction": "down",
                "amount": "page",
                "dry_run": True,
                "execute_confirmed": False,
                "read_after": True,
            },
        }
    ]


def _model_dump(model) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _build_read_region_response(
    *,
    canvas_id: str,
    canvas,
    region,
    include_ocr: bool = True,
    include_elements: bool = True,
    allow_crop_ocr: bool = False,
    screenshot=None,
) -> ReadRegionResponse:
    method = _default_read_method_for_region(str(getattr(region, "role", "") or ""))
    text_blocks, warnings, crop_ocr_used = _harvest_region_text_blocks(
        canvas,
        region,
        include_ocr=include_ocr,
        include_elements=include_elements,
        allow_crop_ocr=allow_crop_ocr,
        screenshot=screenshot,
    )
    if not text_blocks:
        warnings.append("region_text_empty")
    scroll_context = _read_region_scroll_context(canvas, region)
    return ReadRegionResponse(
        canvas_id=canvas_id,
        region_id=str(getattr(region, "region_id", "") or ""),
        region_role=str(getattr(region, "role", "") or ""),
        method=method,
        read_scope="current_viewport",
        scroll_context=scroll_context,
        long_content_strategy=_current_viewport_long_content_strategy(scroll_context),
        status="warn" if warnings else "pass",
        text="\n".join(block.text for block in text_blocks if block.text),
        text_blocks=text_blocks,
        warnings=warnings,
        crop_ocr_used=crop_ocr_used,
        suggested_next_actions=_read_region_suggested_next_actions(region, scroll_context),
    )


def _stitch_read_region_text(before_read: ReadRegionResponse | None, after_read: ReadRegionResponse | None) -> tuple[str, dict[str, int]]:
    before_blocks = list(before_read.text_blocks if before_read else [])
    after_blocks = list(after_read.text_blocks if after_read else [])
    texts: list[str] = []
    seen: set[str] = set()
    deduped = 0
    for block in [*before_blocks, *after_blocks]:
        text = str(block.text or "").strip()
        if not text:
            continue
        if text in seen:
            deduped += 1
            continue
        seen.add(text)
        texts.append(text)
    return "\n".join(texts), {
        "before_block_count": len(before_blocks),
        "after_block_count": len(after_blocks),
        "stitched_block_count": len(texts),
        "deduped_count": deduped,
    }


def _candidate_bounds(candidate) -> list[int]:
    bounds = list(getattr(candidate, "bounds", None) or [])
    return [int(item) for item in bounds] if len(bounds) == 4 else []


def _candidate_click_point(candidate) -> list[int]:
    point = list(getattr(candidate, "click_point", None) or [])
    if len(point) == 2:
        return [int(point[0]), int(point[1])]
    bounds = _candidate_bounds(candidate)
    if len(bounds) == 4:
        left, top, right, bottom = bounds
        return [(left + right) // 2, (top + bottom) // 2]
    return []


def _act_expected_readback_text(request: ActRequest) -> tuple[str, str]:
    params = dict(request.params or {})
    if str(request.action or "") == "send":
        expected = str(params.get("expected_text") or params.get("text") or "").strip()
        if params.get("expected_text"):
            return expected, "params.expected_text"
        if params.get("text"):
            return expected, "params.text"
        return "", ""
    text = str(params.get("text") or "").strip()
    return text, "params.text" if text else ""


def _act_readback_region_role(canvas, candidate=None) -> str:
    if candidate is not None:
        candidate_region_id = str(getattr(candidate, "region_id", "") or "")
        if candidate_region_id:
            region = _find_read_region(canvas, region_id=candidate_region_id, region_role=None)
            if region is not None:
                role = str(getattr(region, "role", "") or "")
                if role:
                    return role
    preferred = ["message_stream", "chat_area", "content_area"]
    regions = list(getattr(canvas, "regions", None) or [])
    roles = [str(getattr(region, "role", "") or "") for region in regions]
    for role in preferred:
        if role in roles:
            return role
    return roles[0] if roles else ""


def _build_act_readback_plan(request: ActRequest, canvas, candidate=None) -> dict[str, Any] | None:
    action = str(request.action or "")
    if action not in {"type_text", "send", "click", "double_click", "scroll"}:
        return None
    expected_text, source = _act_expected_readback_text(request)
    plan = {
        "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
        "request": {
            "region_role": _act_readback_region_role(
                canvas,
                candidate if action in {"click", "double_click", "scroll"} else None,
            ),
            "include_elements": True,
            "include_ocr": True,
            "allow_crop_ocr": action in {"type_text", "send"},
        },
        "expected_text": expected_text,
        "source": source,
    }
    if action in {"click", "double_click", "scroll"}:
        plan["after_observe"] = True
        plan["failure_policy"] = "downgrade_verification_to_review"
    return plan


def _build_act_state_probe_plan(request: ActRequest, canvas) -> dict[str, Any] | None:
    action = str(request.action or "")
    if action not in {"type_text", "send"}:
        return None
    current = _act_current_state_probe(canvas)
    expected_text, expected_source = _act_expected_readback_text(request)
    before_requirements: list[dict[str, Any]] = []
    after_expectations: list[dict[str, Any]] = []
    if action == "type_text":
        after_expectations.append({"field": "input_state", "expected": "filled", "source": expected_source})
        if current.get("send_enabled") is not None:
            after_expectations.append(
                {"field": "send_enabled", "expected": True, "condition": "send_control_present"}
            )
    elif action == "send":
        before_requirements.append({"field": "send_enabled", "expected": True})
        if expected_text:
            after_expectations.append(
                {"field": "message_stream_contains", "expected": expected_text, "source": expected_source}
            )
    return {
        "current": current,
        "before_requirements": before_requirements,
        "after_expectations": after_expectations,
        "observe_after": True,
        "diff_after": True,
        "readback_after": action in {"type_text", "send"},
        "failure_policy": "keep_action_blocked_until_probe_passes",
    }


def _act_current_state_probe(canvas) -> dict[str, Any]:
    from src.memory.page_identity import build_layout_fingerprint

    fp = build_layout_fingerprint(canvas)
    flags = dict(fp.state_flags)
    return {
        "input_state": fp.input_state,
        "send_enabled": flags.get("send_enabled"),
    }


def _is_send_candidate(candidate, canvas=None) -> bool:
    role = _candidate_role(candidate)
    risk_tags = {str(item).lower() for item in list(getattr(candidate, "risk_tags", None) or [])}
    label = str(getattr(candidate, "role_label", "") or "").lower()
    text = str(getattr(candidate, "text", "") or "").lower()
    name = str(getattr(candidate, "name", "") or "").lower()
    return (
        role == "send_button"
        or "send" in risk_tags
        or "发送" in label
        or "send" in label
        or "发送" in text
        or text == "send"
        or name == "send"
        or _is_chat_composer_send_like_candidate(canvas, candidate)
    )


def _is_chat_composer_send_like_candidate(canvas, candidate) -> bool:
    """Conservatively accept unlabeled bottom-right chat send targets."""
    if canvas is None:
        return False
    element_id = str(getattr(candidate, "element_id", "") or "")
    if element_id.startswith("synthetic_"):
        return False
    if not _canvas_looks_like_chat_or_collaboration(canvas):
        return False
    bounds = _candidate_bounds(candidate)
    if len(bounds) != 4:
        return False
    left, top, right, bottom = [int(value) for value in bounds]
    width = right - left
    height = bottom - top
    if width < 44 or height < 20 or width > 180 or height > 60:
        return False
    canvas_width, canvas_height = _infer_canvas_size_for_act(canvas)
    if canvas_width <= 0 or canvas_height <= 0:
        return False
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return (
        top >= int(canvas_height * 0.72)
        and center_x >= canvas_width * 0.60
        and center_y >= canvas_height * 0.70
    )


def _canvas_looks_like_chat_or_collaboration(canvas) -> bool:
    page = getattr(canvas, "page", None)
    page_class = str(getattr(page, "page_class", "") or "").lower()
    if any(token in page_class for token in ("chat", "wechat", "qq", "feishu", "collaboration")):
        return True
    artifacts = getattr(canvas, "artifacts", None) or {}
    visual_pattern = artifacts.get("visual_pattern") if isinstance(artifacts, dict) else {}
    mode = str((visual_pattern or {}).get("mode") or "").lower() if isinstance(visual_pattern, dict) else ""
    return mode in {"chat_workspace", "collaboration_inbox", "chat_document"}


def _infer_canvas_size_for_act(canvas) -> tuple[int, int]:
    window = getattr(canvas, "window", None)
    rect_client = getattr(window, "rect_client", None) if window else None
    if isinstance(rect_client, (list, tuple)) and len(rect_client) == 4:
        width = int(rect_client[2]) - int(rect_client[0])
        height = int(rect_client[3]) - int(rect_client[1])
        if width > 0 and height > 0:
            return width, height
    artifacts = getattr(canvas, "artifacts", None) or {}
    size = artifacts.get("screenshot_size") if isinstance(artifacts, dict) else None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    max_right = 0
    max_bottom = 0
    for element in list(getattr(canvas, "elements", None) or []):
        bounds = _candidate_bounds(element)
        if len(bounds) == 4:
            max_right = max(max_right, int(bounds[2]))
            max_bottom = max(max_bottom, int(bounds[3]))
    return max_right, max_bottom


def _candidate_enabled(candidate) -> bool:
    attrs = getattr(candidate, "attributes", None) or {}
    for key in ("enabled", "is_enabled", "send_enabled"):
        if isinstance(attrs.get(key), bool):
            return bool(attrs[key])
    state = getattr(candidate, "state", None)
    if state is not None and hasattr(state, "enabled"):
        return bool(getattr(state, "enabled"))
    return True


def _build_act_preflight_response(request: ActRequest, canvas, candidate) -> ActResponse:
    attrs = getattr(candidate, "attributes", None) or {}
    actionability = str(attrs.get("actionability") or "")
    safe_to_type = attrs.get("safe_to_type") is True
    warnings: list[str] = []
    action = str(request.action or "")
    expected_text, _expected_source = _act_expected_readback_text(request)
    send_candidate = _is_send_candidate(candidate, canvas)
    candidate_enabled = _candidate_enabled(candidate)
    action_level = _act_action_level(
        action=action,
        actionability=actionability,
        safe_to_type=safe_to_type,
        send_candidate=send_candidate,
        candidate_enabled=candidate_enabled,
    )
    if action == "type_text" and not safe_to_type:
        warnings.append("type_text_requires_safe_to_type")
    if action == "type_text" and not expected_text:
        warnings.append("type_text_requires_text_for_readback")
    if action == "send" and not send_candidate:
        warnings.append("send_requires_send_candidate")
    if action == "send" and send_candidate and not candidate_enabled:
        warnings.append("send_button_disabled")
    if action == "send" and not expected_text:
        warnings.append("send_requires_expected_text_for_readback")
    if action in {"click", "double_click"} and actionability in {"review", "semantic_only"}:
        warnings.append("click_target_requires_review")
    if action_level != "read_only" and not request.dry_run and not request.execute_confirmed:
        warnings.append("act_execute_requires_confirmation")
    if action_level == "blocked" and not warnings:
        warnings.append("act_action_blocked_by_policy")
    if action_level == "controlled" and not warnings:
        warnings.append("act_execution_adapter_pending")
    policy = _act_execution_policy(
        action=action,
        action_level=action_level,
        request=request,
    )

    action_plan = {
        "canvas_id": request.canvas_id,
        "candidate_id": request.candidate_id,
        "action": action,
        "action_level": "blocked" if any(_act_blocking_warning(item) for item in warnings) else action_level,
        "params": dict(request.params or {}),
        "bounds": _candidate_bounds(candidate),
        "click_point": _candidate_click_point(candidate),
        "candidate": {
            "element_id": str(getattr(candidate, "element_id", "") or ""),
            "semantic_role": str(getattr(candidate, "semantic_role", "") or ""),
            "control_type": str(getattr(candidate, "control_type", "") or ""),
            "text": str(getattr(candidate, "text", "") or ""),
            "bounds": _candidate_bounds(candidate),
            "click_point": _candidate_click_point(candidate),
            "risk_tags": [str(item) for item in list(getattr(candidate, "risk_tags", None) or [])],
            "actionability": actionability,
            "safe_to_type": safe_to_type,
            "enabled": candidate_enabled,
            "semantic_role_hint": "send_button" if send_candidate and _candidate_role(candidate) != "send_button" else "",
        },
        "window": {
            "hwnd": int(getattr(getattr(canvas, "window", None), "hwnd", 0) or 0),
            "title": str(getattr(getattr(canvas, "window", None), "title", "") or ""),
        },
        "execution_policy": {
            "dry_run": bool(request.dry_run),
            "execute_confirmed": bool(request.execute_confirmed),
            **policy,
        },
        "verification_plan": {
            "observe_after": action_level == "controlled",
            "diff_after": action_level == "controlled",
            "readback_after": action in {"type_text", "send", "click", "double_click", "scroll"},
            "readback_plan": _build_act_readback_plan(request, canvas, candidate),
            "state_probe_plan": _build_act_state_probe_plan(request, canvas),
            "requires_human_or_policy_confirmation": action_level in {"review", "blocked"}
            or any(_act_blocking_warning(item) for item in warnings),
        },
    }
    if not warnings:
        return ActResponse(
            execution_result="planned",
            error_message=None,
            can_execute=action_level == "read_only",
            action_plan=action_plan,
            warnings=[],
        )
    if warnings == ["act_execution_adapter_pending"]:
        return ActResponse(
            execution_result="planned",
            error_message="Act preflight passed, but desktop execution adapter is not enabled in this endpoint yet.",
            can_execute=False,
            action_plan=action_plan,
            warnings=warnings,
        )
    return ActResponse(
        execution_result="blocked",
        error_message="Act preflight blocked by safety or confirmation policy.",
        can_execute=False,
        action_plan=action_plan,
        warnings=warnings,
    )


def _act_blocking_warning(warning: str) -> bool:
    return warning not in {"act_execution_adapter_pending"}


def _act_execution_policy(*, action: str, action_level: str, request: ActRequest) -> dict[str, Any]:
    confirmed_would_touch = action in {"click", "double_click", "scroll"} and action_level == "controlled"
    execution_enabled = action in {"read", "inspect", "click", "double_click", "scroll"} and action_level in {
        "read_only",
        "controlled",
    }
    return {
        "executes_desktop_input": bool(confirmed_would_touch and not request.dry_run and request.execute_confirmed),
        "confirmed_execution_would_touch_desktop": confirmed_would_touch,
        "requires_execute_confirmed": action_level == "controlled",
        "enabled_for_execution": execution_enabled,
    }


def _act_action_level(
    *,
    action: str,
    actionability: str,
    safe_to_type: bool,
    send_candidate: bool = False,
    candidate_enabled: bool = True,
) -> str:
    if action in {"read", "inspect"}:
        return "read_only"
    if action == "type_text":
        return "controlled" if safe_to_type else "blocked"
    if action == "send":
        if not send_candidate or not candidate_enabled:
            return "blocked"
        return "review" if actionability in {"review", "semantic_only"} else "controlled"
    if action in {"click", "double_click", "scroll"}:
        return "review" if actionability in {"review", "semantic_only"} else "controlled"
    if action in {"key_press", "hotkey"}:
        return "review"
    return "blocked"


def _act_scroll_delta(params: dict[str, Any]) -> int:
    amount = str(params.get("amount") or "page").lower()
    steps = 4 if amount == "page" else 1
    direction = str(params.get("direction") or "down").lower()
    sign = -1 if direction in {"down", "right"} else 1
    return sign * 120 * steps


def _build_diff_response_from_canvases(before, after) -> DiffResponse:
    from src.canvas.diff_engine import CanvasDiffEngine
    from src.memory.memory_service import MemoryService

    memory = MemoryService()
    before_effective = _effective_canvas(before)
    after_effective = _effective_canvas(after)
    engine = CanvasDiffEngine(
        key_store=memory.key_store,
        before_canvas=before_effective,
        after_canvas=after_effective,
    )
    changes = engine.diff(before_effective, after_effective, detail_level="changes")
    return DiffResponse(
        new_canvas_id=after.canvas_id,
        added=[_candidate_to_response(c) for c in changes.added],
        removed=[_candidate_to_response(c) for c in changes.removed],
        preserved=[_candidate_to_response(c) for c in changes.preserved],
        page_changed=changes.page_changed,
        page_class_changed=changes.page_class_changed,
        summary=changes.summary,
    )


def _build_act_verification_result(
    *,
    action: str,
    before_canvas,
    before_canvas_id: str,
    after_canvas,
    triggered_diff: DiffResponse | None,
    readback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diff_checked = triggered_diff is not None
    changed = False
    if triggered_diff is not None:
        changed = bool(
            triggered_diff.page_changed
            or triggered_diff.page_class_changed
            or triggered_diff.added
            or triggered_diff.removed
        )
    readback_status = str((readback or {}).get("status") or "")
    verification_status = "pass" if after_canvas is not None and diff_checked else "review"
    if readback is not None and readback_status != "pass":
        verification_status = "review"
    return {
        "status": verification_status,
        "action": action,
        "observed_after": after_canvas is not None,
        "diff_checked": diff_checked,
        "readback_checked": readback is not None,
        "readback": readback or {},
        "before_canvas_id": before_canvas_id,
        "after_canvas_id": str(getattr(after_canvas, "canvas_id", "") or "") if after_canvas is not None else "",
        "page_changed": bool(triggered_diff.page_changed) if triggered_diff else False,
        "page_class_changed": bool(triggered_diff.page_class_changed) if triggered_diff else False,
        "changed": changed,
        "summary": str(triggered_diff.summary or "") if triggered_diff else "",
        "transition": _build_act_transition_result(before_canvas=before_canvas, after_canvas=after_canvas),
        "next_recommended_checks": [
            "read-region"
        ] if action == "scroll" and readback is None else [],
    }


def _build_act_transition_result(*, before_canvas, after_canvas) -> dict[str, Any]:
    before_page = getattr(before_canvas, "page", None)
    after_page = getattr(after_canvas, "page", None) if after_canvas is not None else None
    before_page_class = str(getattr(before_page, "page_class", "") or "")
    after_page_class = str(getattr(after_page, "page_class", "") or "")
    before_state = str(getattr(before_canvas, "state_template_id", "") or "")
    after_state = str(getattr(after_canvas, "state_template_id", "") or "") if after_canvas is not None else ""
    before_model = str(getattr(before_canvas, "page_model_id", "") or "")
    after_model = str(getattr(after_canvas, "page_model_id", "") or "") if after_canvas is not None else ""
    return {
        "before_canvas_id": str(getattr(before_canvas, "canvas_id", "") or ""),
        "after_canvas_id": str(getattr(after_canvas, "canvas_id", "") or "") if after_canvas is not None else "",
        "before_page_class": before_page_class,
        "after_page_class": after_page_class,
        "before_page_model_id": before_model,
        "after_page_model_id": after_model,
        "before_state_template_id": before_state,
        "after_state_template_id": after_state,
        "page_changed": bool(before_page_class and after_page_class and before_page_class != after_page_class),
        "page_model_changed": bool(before_model and after_model and before_model != after_model),
        "state_changed": bool(before_state != after_state),
    }


def _act_candidate_key(candidate) -> tuple[str, str | None]:
    stable_key_id = str(getattr(candidate, "stable_key_id", "") or "")
    if stable_key_id:
        return stable_key_id, stable_key_id
    element_id = str(getattr(candidate, "element_id", "") or "")
    if element_id:
        return f"transient:{element_id}", None
    return "", None


def _record_act_control_transition(
    *,
    action: str,
    candidate,
    transition: dict[str, Any],
    success: bool,
) -> dict[str, Any]:
    from_class = str(transition.get("before_page_class") or "")
    to_class = str(transition.get("after_page_class") or "")
    candidate_key, stable_key_id = _act_candidate_key(candidate)
    if not from_class or not to_class or not candidate_key:
        return {
            "recorded": False,
            "candidate_key": candidate_key,
            "stable_key_id": stable_key_id or "",
            "reason": "missing_transition_identity",
        }
    try:
        from src.memory.transition_graph import TransitionGraphManager
        from src.storage.db import Session

        with Session() as session:
            manager = TransitionGraphManager(session)
            manager.record_control_transition(
                candidate_key=candidate_key,
                stable_key_id=stable_key_id,
                from_class=from_class,
                to_class=to_class,
                action_type=action,
                canvas_id_before=str(transition.get("before_canvas_id") or ""),
                canvas_id_after=str(transition.get("after_canvas_id") or ""),
                success=success,
                metadata={
                    "source": "api.act.confirmed_execution",
                    "page_changed": bool(transition.get("page_changed")),
                    "page_model_changed": bool(transition.get("page_model_changed")),
                    "state_changed": bool(transition.get("state_changed")),
                },
            )
    except Exception as exc:  # pragma: no cover - persistence failure must not mask action result
        logging.getLogger(__name__).warning("Act control transition persistence failed: %s", exc)
        return {
            "recorded": False,
            "candidate_key": candidate_key,
            "stable_key_id": stable_key_id or "",
            "reason": "persistence_failed",
        }
    return {
        "recorded": True,
        "candidate_key": candidate_key,
        "stable_key_id": stable_key_id or "",
        "reason": "",
    }


def _act_read_result(request: ActRequest, canvas, candidate) -> dict[str, Any]:
    params = dict(request.params or {})
    region = None
    region_id = str(params.get("region_id") or "")
    region_role = str(params.get("region_role") or "")
    if region_id or region_role:
        region = _find_read_region(canvas, region_id=region_id or None, region_role=region_role or None)
    if region is None:
        candidate_region_id = str(getattr(candidate, "region_id", "") or "")
        region = _find_read_region(canvas, region_id=candidate_region_id or None, region_role=None)
    if region is None:
        return {
            "status": "missing_region",
            "warnings": ["act_read_region_missing"],
        }
    read_response = _build_read_region_response(
        canvas_id=str(getattr(canvas, "canvas_id", "") or request.canvas_id or ""),
        canvas=canvas,
        region=region,
        include_ocr=True,
        include_elements=True,
        allow_crop_ocr=bool(params.get("allow_crop_ocr") is True),
    )
    return {
        "status": "pass" if read_response.status == "pass" else "warn",
        "action": str(request.action or ""),
        "region_id": str(getattr(region, "region_id", "") or ""),
        "region_role": str(getattr(region, "role", "") or ""),
        "after_read": _model_dump(read_response),
        "warnings": list(read_response.warnings or []),
    }


def _execute_act_read_only(request: ActRequest, canvas, candidate, preflight: ActResponse) -> ActResponse:
    readback = _act_read_result(request, canvas, candidate)
    verification_result = {
        "status": "pass" if readback.get("status") == "pass" else "review",
        "action": str(request.action or ""),
        "observed_after": False,
        "diff_checked": False,
        "readback_checked": True,
        "readback": readback,
        "before_canvas_id": str(getattr(canvas, "canvas_id", "") or request.canvas_id or ""),
        "after_canvas_id": str(getattr(canvas, "canvas_id", "") or request.canvas_id or ""),
        "page_changed": False,
        "page_class_changed": False,
        "changed": False,
        "summary": "Read-only action returned current canvas evidence.",
        "transition": _build_act_transition_result(before_canvas=canvas, after_canvas=canvas),
        "next_recommended_checks": [],
    }
    return ActResponse(
        execution_result="executed",
        error_message=None,
        can_execute=True,
        action_plan=dict(preflight.action_plan or {}),
        verification_result=verification_result,
        warnings=list(readback.get("warnings") or []),
    )


def _act_readback_after_result(request: ActRequest, before_canvas, after_canvas) -> dict[str, Any] | None:
    params = dict(request.params or {})
    if params.get("read_after") is False:
        return None
    region_role = str(params.get("region_role") or "")
    region_id = str(params.get("region_id") or "")
    region = None
    if region_id or region_role:
        region = _find_read_region(after_canvas, region_id=region_id or None, region_role=region_role or None)
    if region is None:
        candidate = next(
            (
                item
                for item in list(getattr(before_canvas, "elements", None) or [])
                if str(getattr(item, "element_id", "") or "") == request.candidate_id
            ),
            None,
        )
        candidate_region_id = str(getattr(candidate, "region_id", "") or "") if candidate is not None else ""
        before_region = _find_read_region(before_canvas, region_id=candidate_region_id or None, region_role=None)
        if before_region is not None:
            region = _find_read_region(
                after_canvas,
                region_id=None,
                region_role=str(getattr(before_region, "role", "") or "") or None,
            )
    if region is None:
        region = _find_read_region(after_canvas, region_id=None, region_role="message_stream")
    if region is None:
        return {
            "status": "missing_region",
            "warnings": ["act_scroll_readback_region_missing"],
        }
    after_read = _build_read_region_response(
        canvas_id=str(getattr(after_canvas, "canvas_id", "") or ""),
        canvas=after_canvas,
        region=region,
        include_ocr=True,
        include_elements=True,
        allow_crop_ocr=bool(params.get("allow_crop_ocr_after") is True),
    )
    return {
        "status": "pass" if after_read.status == "pass" else "warn",
        "action": str(request.action or ""),
        "region_id": str(getattr(region, "region_id", "") or ""),
        "region_role": str(getattr(region, "role", "") or ""),
        "after_read": _model_dump(after_read),
        "warnings": list(after_read.warnings or []),
    }


async def _execute_act_confirmed(request: ActRequest, canvas, candidate, preflight: ActResponse) -> ActResponse:
    action_plan = dict(preflight.action_plan or {})
    action = str(request.action or "")
    bounds = _candidate_bounds(candidate)
    click_point = _candidate_click_point(candidate)
    hwnd = int(getattr(getattr(canvas, "window", None), "hwnd", 0) or 0)
    if not hwnd:
        return ActResponse(
            execution_result="blocked",
            error_message="Act execution requires a target window handle.",
            can_execute=False,
            action_plan=action_plan,
            warnings=["missing_hwnd"],
        )
    if action not in {"click", "double_click", "scroll"}:
        return ActResponse(
            execution_result="blocked",
            error_message=f"Confirmed execution for action {action} is not enabled.",
            can_execute=False,
            action_plan=action_plan,
            warnings=["act_action_not_executable"],
        )

    from src.execution.action_executor import ActionExecutor

    executor = ActionExecutor()
    if action in {"click", "double_click"}:
        if len(bounds) != 4:
            return ActResponse(
                execution_result="blocked",
                error_message="Click execution requires candidate bounds.",
                can_execute=False,
                action_plan=action_plan,
                warnings=["missing_candidate_bounds"],
            )
        left, top, right, bottom = bounds
        execution = executor.click_element(
            hwnd,
            left,
            top,
            max(1, right - left),
            max(1, bottom - top),
            double=action == "double_click",
        )
    else:
        if len(click_point) != 2:
            return ActResponse(
                execution_result="blocked",
                error_message="Scroll execution requires a candidate click point or bounds.",
                can_execute=False,
                action_plan=action_plan,
                warnings=["missing_candidate_click_point"],
            )
        execution = executor.scroll_at(
            hwnd,
            client_x=click_point[0],
            client_y=click_point[1],
            delta=_act_scroll_delta(dict(request.params or {})),
        )

    execution_payload = {
        "success": bool(execution.success),
        "action": execution.action,
        "details": execution.details,
        "error": execution.error,
    }
    action_plan["execution_result"] = execution_payload
    if not execution.success:
        return ActResponse(
            execution_result="failed",
            error_message=execution.error or execution.details or "Act execution failed.",
            can_execute=True,
            action_plan=action_plan,
            warnings=["act_execution_failed"],
        )

    after, _model_match_status, _vlm_info = await run_in_threadpool(
        _do_observe,
        hwnd,
        allow_vlm=False,
        force_vlm=False,
        run_enhancement_phases=False,
    )
    triggered_diff = _build_diff_response_from_canvases(canvas, after)
    readback = _act_readback_after_result(request, canvas, after) if action in {"click", "double_click", "scroll"} else None
    verification_result = _build_act_verification_result(
        action=action,
        before_canvas=canvas,
        before_canvas_id=str(getattr(canvas, "canvas_id", "") or request.canvas_id or ""),
        after_canvas=after,
        triggered_diff=triggered_diff,
        readback=readback,
    )
    verification_result["control_transition"] = _record_act_control_transition(
        action=action,
        candidate=candidate,
        transition=dict(verification_result.get("transition") or {}),
        success=True,
    )
    warnings = list(readback.get("warnings") or []) if isinstance(readback, dict) else []
    return ActResponse(
        execution_result="executed",
        error_message=None,
        can_execute=True,
        action_plan=action_plan,
        triggered_diff=triggered_diff,
        verification_result=verification_result,
        warnings=warnings,
    )


def _candidate_role(candidate) -> str:
    role = getattr(candidate, "semantic_role", "")
    return role.value if hasattr(role, "value") else str(role)


def _bounds_width(bounds: list[Any]) -> int:
    if len(bounds) != 4:
        return 0
    try:
        return max(0, int(bounds[2]) - int(bounds[0]))
    except (TypeError, ValueError):
        return 0


def _bounds_height(bounds: list[Any]) -> int:
    if len(bounds) != 4:
        return 0
    try:
        return max(0, int(bounds[3]) - int(bounds[1]))
    except (TypeError, ValueError):
        return 0


def _bounds_area(bounds: list[Any]) -> int:
    return _bounds_width(bounds) * _bounds_height(bounds)


def _find_read_region(canvas, *, region_id: str | None, region_role: str | None):
    regions = list(getattr(canvas, "regions", None) or [])
    if region_id:
        for region in regions:
            if str(getattr(region, "region_id", "") or "") == region_id:
                return region
    if region_role:
        for region in regions:
            if str(getattr(region, "role", "") or "") == region_role:
                return region
    return None


def _default_read_method_for_region(region_role: str) -> str:
    if region_role in {"message_stream", "message_thread", "chat_area"}:
        return "chat_crop_ocr_readback"
    return "region_text_harvest"


def _harvest_region_text_blocks(
    canvas,
    region,
    *,
    include_ocr: bool,
    include_elements: bool,
    allow_crop_ocr: bool = False,
    screenshot=None,
) -> tuple[list[ReadRegionTextBlock], list[str], bool]:
    region_bounds = list(getattr(region, "bounds", None) or [])
    blocks: list[ReadRegionTextBlock] = []
    warnings: list[str] = []
    crop_ocr_used = False
    if include_elements:
        blocks.extend(_region_element_text_blocks(canvas, region, region_bounds=region_bounds))
    if include_ocr:
        blocks.extend(_region_ocr_text_blocks(canvas, region_bounds=region_bounds))
    if allow_crop_ocr:
        if screenshot is None:
            warnings.append("missing_screenshot_for_crop_ocr")
        elif len(region_bounds) != 4:
            warnings.append("invalid_region_bounds_for_crop_ocr")
        else:
            crop_blocks, crop_warnings = _region_crop_ocr_text_blocks(screenshot, region_bounds=region_bounds)
            warnings.extend(crop_warnings)
            if crop_blocks:
                crop_ocr_used = True
                blocks.extend(crop_blocks)
    return _dedupe_text_blocks(blocks), warnings, crop_ocr_used


def _region_element_text_blocks(canvas, region, *, region_bounds: list[Any]) -> list[ReadRegionTextBlock]:
    region_id = str(getattr(region, "region_id", "") or "")
    result = []
    for candidate in list(getattr(canvas, "elements", None) or []):
        text = str(getattr(candidate, "text", "") or getattr(candidate, "name", "") or getattr(candidate, "value", "") or "").strip()
        if not text:
            continue
        bounds = list(getattr(candidate, "bounds", None) or [])
        if str(getattr(candidate, "region_id", "") or "") != region_id and not _bounds_inside(bounds, region_bounds):
            continue
        result.append(
            ReadRegionTextBlock(
                text=text,
                bounds=[int(item) for item in bounds] if len(bounds) == 4 else [],
                confidence=float(getattr(candidate, "confidence", 0.0) or 0.0),
                source="candidate_text",
                element_id=str(getattr(candidate, "element_id", "") or ""),
            )
        )
    return result


def _region_ocr_text_blocks(canvas, *, region_bounds: list[Any]) -> list[ReadRegionTextBlock]:
    artifacts = canvas.artifacts if isinstance(getattr(canvas, "artifacts", None), dict) else {}
    result = []
    for block in list(artifacts.get("ocr_blocks") or []):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        bounds = list(block.get("bbox") or block.get("bounds") or [])
        if not text or not _bounds_inside(bounds, region_bounds):
            continue
        try:
            confidence = float(block.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        result.append(
            ReadRegionTextBlock(
                text=text,
                bounds=[int(item) for item in bounds] if len(bounds) == 4 else [],
                confidence=confidence,
                source="ocr_block",
            )
        )
    return result


def _region_crop_ocr_text_blocks(screenshot, *, region_bounds: list[Any]) -> tuple[list[ReadRegionTextBlock], list[str]]:
    try:
        left, top, right, bottom = [int(item) for item in region_bounds]
    except (TypeError, ValueError):
        return [], ["invalid_region_bounds_for_crop_ocr"]
    if right <= left or bottom <= top:
        return [], ["invalid_region_bounds_for_crop_ocr"]

    try:
        from src.perception.ocr_service import get_ocr_service

        result = get_ocr_service().extract_with_metadata(
            screenshot,
            min_text_length=1,
            filter_pure_digits=False,
            filter_pure_symbols=False,
            region=(left, top, right, bottom),
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for external OCR worker
        return [], [f"crop_ocr_failed:{exc}"]

    if not getattr(result, "success", False):
        error = str(getattr(result, "error", "") or "unknown")
        return [], [f"crop_ocr_failed:{error}"]

    blocks = []
    for block in list(getattr(result, "blocks", None) or []):
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        bounds = list(getattr(block, "bbox", None) or [])
        blocks.append(
            ReadRegionTextBlock(
                text=text,
                bounds=[int(item) for item in bounds] if len(bounds) == 4 else [],
                confidence=float(getattr(block, "confidence", 0.0) or 0.0),
                source="crop_ocr",
            )
        )
    if not blocks:
        return [], ["crop_ocr_empty"]
    return blocks, []


def _bounds_inside(bounds: list[Any], outer: list[Any]) -> bool:
    if len(bounds) != 4 or len(outer) != 4:
        return False
    try:
        left, top, right, bottom = [int(item) for item in bounds]
        outer_left, outer_top, outer_right, outer_bottom = [int(item) for item in outer]
    except (TypeError, ValueError):
        return False
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    return outer_left <= center_x <= outer_right and outer_top <= center_y <= outer_bottom


def _dedupe_text_blocks(blocks: list[ReadRegionTextBlock]) -> list[ReadRegionTextBlock]:
    result = []
    seen = set()
    for block in blocks:
        key = (block.text, tuple(block.bounds), block.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(block)
    return result


def _refresh_provider_status_from_trace(canvas) -> None:
    """Rebuild provider summary fields from the current provider_trace."""
    trace = getattr(canvas, "provider_trace", None)
    if trace is None:
        return

    providers_used: list[str] = []
    for flag, name in [
        ("uia_used", "uia"),
        ("ocr_used", "ocr"),
        ("vision_used", "vision"),
        ("vlm_used", "vlm"),
        ("dom_used", "dom"),
    ]:
        if getattr(trace, flag, False):
            providers_used.append(name)

    providers_failed: list[str] = []
    details = getattr(trace, "provider_details", None) or {}
    if isinstance(details, dict):
        for name, detail in details.items():
            if isinstance(detail, dict) and detail.get("success") is False:
                providers_failed.append(name)

    canvas.providers_used = providers_used
    canvas.providers_failed = providers_failed
    canvas.partial = bool(providers_failed)


def _build_vlm_prompt_input(zone_page, canvas, task_mode: str = "candidate_annotation"):
    """Build PromptInput from zone_page + canvas for VLM semantic analysis."""
    from src.vlm.prompt_builder import PromptInput

    # Extract candidates in omni format
    omni_candidates = []
    for elem in canvas.elements:
        c: dict = {"element_id": elem.element_id}
        if elem.bounds:
            c["bounds"] = list(elem.bounds)
        if elem.text:
            c["text"] = elem.text
        role = elem.semantic_role.value if hasattr(elem.semantic_role, "value") else str(elem.semantic_role)
        if role and role != "unknown":
            c["semantic_role"] = role
        if elem.control_type:
            c["control_type"] = elem.control_type
        omni_candidates.append(c)

    return PromptInput(
        screenshot=zone_page.screenshot,
        omni_candidates=omni_candidates,
        task_mode=task_mode,
    )


def _candidate_dict_from_element(elem) -> dict:
    c: dict = {"element_id": elem.element_id}
    if elem.bounds:
        c["bounds"] = list(elem.bounds)
    if getattr(elem, "text", None):
        c["text"] = elem.text
    role = elem.semantic_role.value if hasattr(elem.semantic_role, "value") else str(elem.semantic_role)
    if role and role != "unknown":
        c["semantic_role"] = role
    if getattr(elem, "control_type", None):
        c["control_type"] = elem.control_type
    if getattr(elem, "confidence", None) is not None:
        c["confidence"] = float(elem.confidence)
    return c


def _build_cached_vlm_prompt_input(canvas, screenshot, *, task_mode: str, candidate_ids: list[str] | None = None):
    """Build PromptInput for manual/background VLM tasks from cached canvas data."""
    from src.vlm.prompt_builder import PromptInput
    from src.vlm.prompt_modes import ICON_CROP_UNDERSTANDING

    candidate_ids = candidate_ids or []
    selected_ids = set(candidate_ids)

    if task_mode == ICON_CROP_UNDERSTANDING and selected_ids:
        from src.vlm.icon_crop import crop_candidate_icon

        for elem in canvas.elements:
            if elem.element_id not in selected_ids:
                continue
            crop = crop_candidate_icon(screenshot, elem)
            if crop is None:
                continue
            candidate = _candidate_dict_from_element(elem)
            candidate["bounds"] = list(crop.bounds)
            candidate["crop_dhash"] = crop.dhash
            return PromptInput(
                screenshot=crop.image,
                omni_candidates=[candidate],
                task_hint=(
                    "This is a single candidate crop for icon semantic classification. "
                    f"original_candidate_id={elem.element_id}; crop_dhash={crop.dhash}. "
                    "Annotate only this candidate and do not output bounds."
                ),
                task_mode=task_mode,
            )

    omni_candidates = []
    for elem in canvas.elements:
        if selected_ids and elem.element_id not in selected_ids:
            continue
        omni_candidates.append(_candidate_dict_from_element(elem))

    task_hint = ""
    if task_mode == "missing_audit":
        task_hint = (
            "Audit the current local candidates and semantic model for visible missing areas or controls. "
            "Return suggestions only; do not create safe clickable controls."
        )
    elif task_mode == ICON_CROP_UNDERSTANDING:
        task_hint = (
            "Classify the visible icon-like candidates. If this is not a crop, be conservative and use review."
        )

    return PromptInput(
        screenshot=screenshot,
        omni_candidates=omni_candidates,
        task_hint=task_hint,
        task_mode=task_mode,
    )


def _build_model_workbench_summary(
    *,
    app_id: str,
    display_name: str | None,
    surface_type: str | None,
    page_class: str,
    state_label: str | None,
    candidates: list[VirtualModelCandidate],
    regions: list[dict],
    vlm_model_dict: dict | None,
) -> dict[str, Any]:
    """Build derived AppShell/Region/Element workbench data without new tables."""
    from collections import Counter

    source_counts = Counter(c.source or "unknown" for c in candidates)
    state_counts = Counter(c.permanence_state or "new" for c in candidates)
    region_counts = Counter(c.canonical_region or "" for c in candidates)

    region_templates: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for idx, region in enumerate(regions or []):
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("region_id") or region.get("zone_id") or f"region_{idx}")
        if region_id in seen_regions:
            continue
        seen_regions.add(region_id)
        region_templates.append({
            "region_id": region_id,
            "role": region.get("role") or region.get("zone_type") or region.get("content_type") or "unknown",
            "purpose": region.get("purpose") or region.get("note") or "",
            "bounds": region.get("bounds") or [],
            "shared": bool(region.get("shared")),
            "candidate_count": int(region_counts.get(region_id, 0)),
            "source": "vlm_shared" if region.get("shared") else "vlm",
        })

    element_templates: list[dict[str, Any]] = []
    for candidate in candidates:
        kind = "fixed" if candidate.is_fixed_control or candidate.permanence_state == "fixed_anchor" else "dynamic"
        if candidate.source in {"manual", "agent"}:
            actionability = "safe"
        elif candidate.source == "vlm":
            actionability = "review"
        elif candidate.permanence_state in {"stable", "fixed_anchor"}:
            actionability = "safe"
        else:
            actionability = "review"
        element_templates.append({
            "key_id": candidate.key_id,
            "role": candidate.canonical_role,
            "label": candidate.role_label or candidate.canonical_text or candidate.canonical_role,
            "region_id": candidate.canonical_region,
            "kind": kind,
            "source": candidate.source,
            "permanence_state": candidate.permanence_state,
            "actionability": actionability,
            "confidence": candidate.confidence,
            "relative_bounds": candidate.relative_bounds,
            "provider_sources": candidate.provider_sources,
        })

    visible_items = []
    missing_suggestions = []
    if vlm_model_dict:
        visible_items = list(vlm_model_dict.get("visible_items") or [])
        for suggestion in list(vlm_model_dict.get("missing_suggestions") or []):
            if not isinstance(suggestion, dict):
                continue
            sanitized = dict(suggestion)
            if sanitized.get("actionability") == "safe":
                sanitized["actionability"] = "review"
            else:
                sanitized["actionability"] = sanitized.get("actionability") or "semantic_only"
            missing_suggestions.append(sanitized)

    app_shell = {
        "app_id": app_id,
        "display_name": display_name,
        "surface_type": surface_type,
        "page_class": page_class,
        "state_label": state_label,
        "region_roles": sorted({str(r.get("role") or "unknown") for r in region_templates}),
        "shared_region_count": sum(1 for r in region_templates if r.get("shared")),
        "fixed_anchor_count": state_counts.get("fixed_anchor", 0),
    }
    model_layers = {
        "source_counts": dict(source_counts),
        "permanence_counts": dict(state_counts),
        "candidate_count": len(candidates),
        "region_count": len(region_templates),
        "missing_suggestion_count": len(missing_suggestions),
        "visible_item_count": len(visible_items),
        "priority_order": ["manual", "agent", "vlm_semantic", "vlm", "persistent", "canvas_transient"],
    }
    return {
        "model_layers": model_layers,
        "app_shell": app_shell,
        "region_templates": region_templates,
        "element_templates": element_templates,
        "visible_items": visible_items,
        "missing_suggestions": missing_suggestions,
    }


def _canvas_dimensions(canvas) -> tuple[int, int]:
    """Return best-effort canvas dimensions in screenshot/window pixels."""
    width, height = 0, 0
    if getattr(canvas, "window", None):
        if canvas.window.rect_client:
            x1, y1, x2, y2 = canvas.window.rect_client
            width, height = max(width, x2 - x1), max(height, y2 - y1)
        if (width <= 0 or height <= 0) and canvas.window.rect_screen:
            x1, y1, x2, y2 = canvas.window.rect_screen
            width, height = max(width, x2 - x1), max(height, y2 - y1)
    if width <= 0 or height <= 0:
        for elem in getattr(canvas, "elements", []):
            if elem.bounds and len(elem.bounds) >= 4:
                width = max(width, int(elem.bounds[2]))
                height = max(height, int(elem.bounds[3]))
    return width, height


def _norm_vlm_model_bounds(model_dict: dict, canvas, vlm_image_size: tuple[int, int] | None = None) -> None:
    """Normalize VLM bounds in a model dict from VLM space → sent-image → original coordinates.

    Two correction layers:
      1. VLM image_size → sent image_size (model-agnostic corner alignment)
      2. sent image_size → original coordinates (resize correction)

    Also validates bounds sanity strictly. Out-of-range VLM bounds are rejected,
    not clamped into a clickable-looking success.
    """
    artifacts = getattr(canvas, "artifacts", None) or {}
    scale_info = artifacts.get("vlm_image_scale", None)
    if not scale_info:
        attrs = getattr(canvas, "attributes", None) or {}
        scale_info = attrs.get("vlm_image_scale", None)

    orig_w, orig_h = 0, 0
    sent_w, sent_h = 0, 0
    needs_scale = False

    if scale_info:
        orig = scale_info.get("original_size", {})
        sent = scale_info.get("sent_image_size", {})
        orig_w = orig.get("w", 0)
        orig_h = orig.get("h", 0)
        sent_w = sent.get("w", 0)
        sent_h = sent.get("h", 0)
        if orig_w and orig_h and sent_w and sent_h and (orig_w != sent_w or orig_h != sent_h):
            needs_scale = True

    # Use image_size from model_dict as reference for validation
    im = model_dict.get("image_size", [])
    ref_w = im[0] if im and len(im) >= 2 and im[0] > 0 else orig_w
    ref_h = im[1] if im and len(im) >= 2 and im[1] > 0 else orig_h

    # Layer 1: VLM self-reported image_size → sent image_size (corner alignment)
    vlm_sx, vlm_sy = 1.0, 1.0
    if vlm_image_size and sent_w > 0 and sent_h > 0:
        vlm_w, vlm_h = vlm_image_size
        if vlm_w > 0 and vlm_h > 0 and (vlm_w != sent_w or vlm_h != sent_h):
            vlm_sx = sent_w / vlm_w
            vlm_sy = sent_h / vlm_h

    # Layer 2: sent → orig (resize correction)
    resize_sx = orig_w / sent_w if needs_scale else 1.0
    resize_sy = orig_h / sent_h if needs_scale else 1.0

    sx = vlm_sx * resize_sx
    sy = vlm_sy * resize_sy

    needs_scaling = needs_scale or vlm_sx != 1.0 or vlm_sy != 1.0

    import logging
    _log = logging.getLogger(__name__)

    def _scale_and_validate(bounds_list, label="", container=None):
        import math

        if not bounds_list or len(bounds_list) < 4:
            return True  # keep entries without bounds
        if needs_scaling:
            bounds_list[0] = int(math.floor(bounds_list[0] * sx))
            bounds_list[1] = int(math.floor(bounds_list[1] * sy))
            bounds_list[2] = int(math.ceil(bounds_list[2] * sx))
            bounds_list[3] = int(math.ceil(bounds_list[3] * sy))
        l, t, r, b = bounds_list[0], bounds_list[1], bounds_list[2], bounds_list[3]
        # Validate
        if l >= r or t >= b:
            _log.warning("VLM persist: invalid bounds %s in %s — marking for removal",
                         bounds_list, label)
            return False  # signal caller to remove this entry
        if ref_w > 0 and ref_h > 0:
            if l < 0:
                _log.warning("VLM persist: left outside image %s in %s", bounds_list, label)
                return False
            if t < 0:
                _log.warning("VLM persist: top outside image %s in %s", bounds_list, label)
                return False
            if r > ref_w:
                _log.warning("VLM persist: right outside image %s in %s", bounds_list, label)
                return False
            if b > ref_h:
                _log.warning("VLM persist: bottom outside image %s in %s", bounds_list, label)
                return False
        # Recheck
        if bounds_list[0] >= bounds_list[2] or bounds_list[1] >= bounds_list[3]:
            return False
        return True

    # Filter out invalid entries
    for container_name, key in [("fixed_controls", None), ("regions", None), ("dynamic_zones", None)]:
        items = model_dict.get(container_name, []) or []
        if not items:
            continue
        valid = []
        for item in items:
            b = item.get("bounds")
            label = item.get("control_id") or item.get("region_id") or item.get("zone_id") or "?"
            if not b:
                valid.append(item)
                continue
            if _scale_and_validate(b, label=label):
                valid.append(item)
        model_dict[container_name] = valid


def _augment_vlm_model_dict(vlm_model, canvas, image_size: tuple[int, int] | None = None) -> dict:
    """Add durable rendering and identity aliases to a VLM semantic model dict.

    Also normalizes VLM coordinates from sent-image space to original window
    coordinates, so persisted model dict matches the original image_size.
    """
    model_dict = vlm_model.to_dict() if hasattr(vlm_model, "to_dict") else dict(vlm_model or {})

    # Capture VLM's self-reported image_size before overwriting (needed for
    # VLM→sent correction when the model guessed different dimensions).
    vlm_im = model_dict.get("image_size", [])
    vlm_image_size: tuple[int, int] | None = None
    if vlm_im and len(vlm_im) >= 2 and vlm_im[0] > 0 and vlm_im[1] > 0:
        vlm_image_size = (vlm_im[0], vlm_im[1])

    width, height = image_size or _canvas_dimensions(canvas)
    if width > 0 and height > 0:
        model_dict["image_size"] = [int(width), int(height)]

    # Normalize VLM bounds: VLM space → sent space → original window coordinates
    _norm_vlm_model_bounds(model_dict, canvas, vlm_image_size=vlm_image_size)

    control_aliases: dict[str, str] = {}
    for elem in getattr(canvas, "elements", []):
        attrs = getattr(elem, "attributes", None) or {}
        vlm_control_id = attrs.get("vlm_control_id")
        stable_key_id = getattr(elem, "stable_key_id", None)
        if vlm_control_id and stable_key_id:
            control_aliases[str(vlm_control_id)] = str(stable_key_id)

    if control_aliases:
        model_dict["control_aliases"] = control_aliases
        for ctrl in model_dict.get("fixed_controls", []) or []:
            ctrl_id = ctrl.get("control_id")
            stable_key_id = control_aliases.get(str(ctrl_id)) if ctrl_id else None
            if stable_key_id:
                ctrl["stable_key_id"] = stable_key_id
                aliases = list(ctrl.get("aliases") or [])
                for alias in (str(ctrl_id), stable_key_id):
                    if alias and alias not in aliases:
                        aliases.append(alias)
                ctrl["aliases"] = aliases
    return model_dict


def _update_latest_vlm_response_context(
    session,
    *,
    screenshot_hash: str | None,
    page_model_id: str,
    state_template_id: str,
    parsed_model: dict,
) -> None:
    """Attach page/state context and durable aliases to the latest VLM response."""
    if not screenshot_hash:
        return
    import json as _json
    import sqlalchemy

    session.execute(
        sqlalchemy.text(
            """UPDATE vlm_responses
               SET page_model_id = :pm_id,
                   state_template_id = :st_id,
                   parsed_model = :parsed_model
               WHERE response_id = (
                   SELECT response_id FROM vlm_responses
                   WHERE screenshot_hash = :screenshot_hash
                     AND status = 'success'
                     AND parsed_model != ''
                   ORDER BY created_at DESC
                   LIMIT 1
               )"""
        ),
        {
            "pm_id": page_model_id,
            "st_id": state_template_id,
            "parsed_model": _json.dumps(parsed_model, ensure_ascii=False),
            "screenshot_hash": screenshot_hash,
        },
    )


def _do_observe(
    hwnd: int,
    allow_vlm: bool = False,
    force_vlm: bool = False,
    run_enhancement_phases: bool = True,
    fast_perception: bool | None = None,
    lightweight_uia: bool = False,
    capture_mode: str = "window",
) -> tuple["InteractionCanvas", str, dict]:
    """Run the full perception pipeline and cache the result.

    VLM Semantic Modeler runs AFTER perception but BEFORE resolve_page_model,
    so VLM data (app_identity, page_state, regions, controls) flows into
    page model resolution naturally.

    Returns:
        (canvas, model_match_status, vlm_semantic_info)
    """
    import hashlib
    import time as _time

    _t_start = _time.perf_counter()
    _timing: dict[str, float] = {}
    from src.perception.perception_service import PerceptionService
    from src.memory.memory_service import MemoryService

    _t_phase = _time.perf_counter()
    perception = PerceptionService()
    zone_page = perception.analyze(
        hwnd,
        allow_vlm=allow_vlm,
        force_vlm=force_vlm,
        fast_mode=(not run_enhancement_phases if fast_perception is None else fast_perception),
        lightweight_uia=lightweight_uia,
        screenshot_capture_mode=capture_mode,
    )
    _timing["1_perception_analyze"] = _time.perf_counter() - _t_phase

    # Extract process_name from window_info for page_model classification
    _t_phase = _time.perf_counter()
    process_name = zone_page.window_info.process_name if zone_page.window_info else None
    canvas = perception.create_page_snapshot(zone_page, process_name=process_name)
    canvas.artifacts["source_hwnd"] = hwnd
    screenshot_hash = None
    if getattr(zone_page, "screenshot", None) is not None:
        from io import BytesIO

        buf = BytesIO()
        zone_page.screenshot.save(buf, format="PNG")
        screenshot_hash = hashlib.sha256(buf.getvalue()).hexdigest()
    _timing["2_create_canvas"] = _time.perf_counter() - _t_phase

    # --- VLM Semantic Modeler (BEFORE resolve_page_model) ---
    _t_vlm_sm = _time.perf_counter()
    vlm_semantic_info: dict = {"used": False, "status": "skipped", "provider": ""}
    _pending_vlm = None  # (bridge, vlm_model) for later persist

    try:
        from src.common.config_manager import load_config
        from src.vlm.modeler import get_modeler_singleton
        from src.perception.vlm_semantic_bridge import VLMSemanticBridge
        from src.storage.db import Session as DbSessionFactory

        config = load_config()
        sm = config.semantic_modeler
        if sm.enabled and (allow_vlm or force_vlm):
            modeler = get_modeler_singleton(
                enabled=sm.enabled,
                provider=sm.provider,
                api_key=sm.api_key,
                endpoint=sm.endpoint,
                model=sm.model,
                provider_variant=sm.provider_variant,
                free_model_only=sm.free_model_only,
                fallback_provider=sm.fallback_provider,
                fallback_model=sm.fallback_model,
                prompt_version=sm.prompt_version,
                timeout_seconds=sm.timeout_seconds,
                daily_call_limit=sm.daily_call_limit,
                monthly_budget_usd=sm.monthly_budget_usd,
                allow_free_models=sm.allow_free_models,
                save_raw_response=sm.save_raw_response,
                redact_dynamic_content=sm.redact_dynamic_content,
                thinking_mode=sm.thinking_mode,
                image_max_width=sm.image_max_width,
                db_session_factory=DbSessionFactory,
            )
            if modeler is not None:
                # Phase 6: Check if any element is fixed_anchor for VLM frequency reduction
                _perm_state = None
                _obs_count = 0
                try:
                    from src.storage.schema import StableCandidateKey as _SCK
                    from src.storage.db import Session as _Ses
                    _app_id = getattr(getattr(canvas, "app", None), "app_id", None)
                    _pc = getattr(canvas, "page_class", None)
                    if _app_id and _pc:
                        with _Ses() as _s:
                            _row = _s.query(_SCK).filter(
                                _SCK.app_id == _app_id,
                                _SCK.page_class == _pc,
                                _SCK.permanence_state == "fixed_anchor",
                            ).first()
                            if _row:
                                _perm_state = "fixed_anchor"
                                _obs_count = _row.verify_count or 0
                except Exception:
                    pass
                # force_vlm only means bypass cache, not gate entry
                _t_should = _time.perf_counter()
                should, reason = modeler.should_analyze(
                    force=force_vlm,
                    permanence_state=_perm_state,
                    observe_count=_obs_count,
                )
                _timing["3a_vlm_should_analyze"] = _time.perf_counter() - _t_should
                if should or force_vlm:
                    prompt_input = _build_vlm_prompt_input(zone_page, canvas)
                    _t_vlm_call = _time.perf_counter()
                    vlm_result = modeler.analyze(prompt_input, force=force_vlm)
                    _timing["3b_vlm_analyze_call"] = _time.perf_counter() - _t_vlm_call
                    if vlm_result.status in ("success", "cached") and vlm_result.model:
                        # Store VLM image scale for bounds normalization
                        # VLM receives a potentially resized image; its coordinates
                        # are in sent-image space and must be scaled back to
                        # original window coordinates.
                        _orig_w, _orig_h = prompt_input.screenshot.size
                        if _orig_w > sm.image_max_width:
                            _sent_w = sm.image_max_width
                            _sent_h = int(_orig_h * sm.image_max_width / _orig_w)
                        else:
                            _sent_w = _orig_w
                            _sent_h = _orig_h
                        canvas.artifacts["vlm_image_scale"] = {
                            "original_size": {"w": _orig_w, "h": _orig_h},
                            "sent_image_size": {"w": _sent_w, "h": _sent_h},
                        }
                        _t_bridge = _time.perf_counter()
                        bridge = VLMSemanticBridge()
                        canvas = bridge.apply_to_canvas(canvas, vlm_result.model, db_session_factory=DbSessionFactory)
                        _timing["3c_vlm_bridge_apply"] = _time.perf_counter() - _t_bridge
                        _pending_vlm = (bridge, vlm_result.model)
                    vlm_semantic_info = {
                        "used": vlm_result.status in ("success", "cached"),
                        "status": vlm_result.status,
                        "provider": vlm_result.provider_name,
                    }
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("VLM Semantic Modeler error: %s", exc)
        vlm_semantic_info["status"] = "failed"
    _timing["3_vlm_semantic_modeler"] = _time.perf_counter() - _t_vlm_sm

    # --- resolve_page_model (uses VLM-enriched canvas) ---
    _t_phase = _time.perf_counter()
    memory = MemoryService()
    has_screenshot = getattr(zone_page, 'screenshot', None) is not None

    # Extract VLM hints for page model resolution
    vlm_app_name = None
    vlm_display_name = None
    vlm_state_label = None
    vlm_state_flags = None
    if _pending_vlm is not None:
        _, vlm_model = _pending_vlm
        if vlm_model.app_identity and vlm_model.app_identity.app_name:
            vlm_app_name = vlm_model.app_identity.app_name
            vlm_display_name = vlm_model.app_identity.display_name
        if vlm_model.page_state:
            vlm_state_label = vlm_model.page_state.state_label
            vlm_state_flags = list(vlm_model.page_state.state_flags) if vlm_model.page_state.state_flags else None

    page_model_id, state_template_id, page_status, state_status = memory.resolve_page_model(
        canvas, has_screenshot=has_screenshot,
        vlm_app_name=vlm_app_name,
        vlm_display_name=vlm_display_name,
        vlm_state_label=vlm_state_label,
        vlm_state_flags=vlm_state_flags,
    )
    canvas.page_model_id = page_model_id
    canvas.state_template_id = state_template_id
    _timing["4_resolve_page_model"] = _time.perf_counter() - _t_phase

    # 组合模型匹配状态
    if page_status == "new":
        model_match_status = "new_page"
    elif state_status == "new":
        model_match_status = "new_state"
    else:
        model_match_status = "reused"

    # Resolve persistent identity for candidates（带页面上下文）
    _t_phase = _time.perf_counter()
    memory.resolve_candidate_keys(canvas, page_model_id, state_template_id)
    _timing["5_resolve_candidate_keys"] = _time.perf_counter() - _t_phase

    # --- Phase D.5: Evidence Collection (Phase 2 新增) ---
    _t_phase = _time.perf_counter()
    if run_enhancement_phases:
        try:
            from src.memory.evidence_collector import EvidenceCollector
            from src.storage.db import Session

            collector = EvidenceCollector()
            with Session() as ev_session:
                ev_count = collector.collect(
                    ev_session, canvas,
                    vlm_used=vlm_semantic_info.get("used", False),
                )
            if ev_count > 0:
                import logging
                logging.getLogger(__name__).debug("Evidence collected: %d records", ev_count)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Evidence collection failed: %s", exc)
    _timing["6_evidence_collect"] = _time.perf_counter() - _t_phase

    # --- Phase D.6: ConfidenceProfile Fusion (Phase 3 新增) ---
    _t_phase = _time.perf_counter()
    if run_enhancement_phases:
        try:
            from src.memory.evidence_fusion_engine import EvidenceFusionEngine
            from src.storage.db import Session

            fusion_engine = EvidenceFusionEngine()
            with Session() as fu_session:
                fused_count = fusion_engine.apply_to_canvas(fu_session, canvas)
            if fused_count > 0:
                import logging
                logging.getLogger(__name__).debug(
                    "ConfidenceProfile fused: %d elements", fused_count,
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("ConfidenceProfile fusion failed: %s", exc)
    _timing["7_confidence_fusion"] = _time.perf_counter() - _t_phase

    # --- Persist VLM corrections (AFTER resolve_candidate_keys for stable_key_id) ---
    _t_vlm_persist = _time.perf_counter()
    if run_enhancement_phases and _pending_vlm is not None:
        bridge, vlm_model = _pending_vlm
        app_id = canvas.app.app_id or "unknown"
        # Build element_id → stable_key_id mapping
        element_to_stable_key: dict[str, str] = {}
        for elem in canvas.elements:
            sid = getattr(elem, 'stable_key_id', None)
            if sid:
                element_to_stable_key[elem.element_id] = sid
        try:
            from src.storage.db import Session
            with Session() as session:
                augmented = _augment_vlm_model_dict(
                    vlm_model,
                    canvas,
                    image_size=(zone_page.screenshot.width, zone_page.screenshot.height)
                    if getattr(zone_page, "screenshot", None) is not None
                    else None,
                )
                _update_latest_vlm_response_context(
                    session,
                    screenshot_hash=screenshot_hash,
                    page_model_id=page_model_id,
                    state_template_id=state_template_id,
                    parsed_model=augmented,
                )
                session.commit()
            bridge.persist_corrections(
                vlm_model,
                app_id=app_id,
                page_model_id=page_model_id,
                state_template_id=state_template_id,
                element_to_stable_key=element_to_stable_key,
                db_session_factory=Session,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("VLM persist_corrections failed: %s", exc)

        # Record VLM transitions
        if vlm_model.transitions:
            try:
                from src.storage.db import Session
                from src.memory.transition_graph import TransitionGraphManager
                with Session() as session:
                    tgm = TransitionGraphManager(session)
                    tgm.record_vlm_transitions(
                        [t.to_dict() for t in vlm_model.transitions],
                        page_model_id,
                    )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("VLM record_transitions failed: %s", exc)
    _timing["8_vlm_persist"] = _time.perf_counter() - _t_vlm_persist

    # --- Phase D.7: Visual Anchor Comparison (Phase 5 新增) ---
    _t_phase = _time.perf_counter()
    if run_enhancement_phases:
        try:
            from src.memory.visual_anchor_engine import VisualAnchorEngine
            from src.storage.db import Session

            anchor_engine = VisualAnchorEngine()
            with Session() as va_session:
                anchor_count = anchor_engine.apply_to_canvas(va_session, canvas, zone_page.screenshot)
                va_session.commit()
            if anchor_count > 0:
                import logging
                logging.getLogger(__name__).debug(
                    "Visual anchor processed: %d elements", anchor_count,
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Visual anchor comparison failed: %s", exc)
    _timing["9_visual_anchor"] = _time.perf_counter() - _t_phase

    # --- Phase D.8: Permanence State Machine (Phase 6 新增) ---
    _t_phase = _time.perf_counter()
    if run_enhancement_phases:
        try:
            from src.memory.permanence_state_machine import PermanenceStateMachine
            from src.storage.db import Session

            sm = PermanenceStateMachine()
            with Session() as sm_session:
                for element in canvas.elements:
                    stable_key_id = getattr(element, "stable_key_id", None)
                    if not stable_key_id:
                        continue
                    attrs = getattr(element, "attributes", None) or {}
                    # Get verify_count from DB
                    from src.storage.schema import StableCandidateKey as SCKModel

                    row = sm_session.query(SCKModel).filter(SCKModel.key_id == stable_key_id).first()
                    verify_count = row.verify_count if row else 1
                    sm.evaluate_and_update(
                        sm_session,
                        stable_key_id,
                        attrs,
                        verify_count=verify_count,
                        seen_count=verify_count,
                    )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Permanence state machine failed: %s", exc)
    _timing["10_permanence_sm"] = _time.perf_counter() - _t_phase

    # Cache the full canvas and screenshot for query/diff/remember/feedback
    _t_phase = _time.perf_counter()
    cache = get_canvas_cache()
    _refresh_provider_status_from_trace(canvas)
    cache.put(canvas, screenshot=zone_page.screenshot, persist=True)
    _timing["11_cache_put"] = _time.perf_counter() - _t_phase

    _timing["TOTAL"] = _time.perf_counter() - _t_start
    from src.integration.observe_timing import attach_observe_timing
    attach_observe_timing(canvas, _timing)
    import logging
    _timing_msg = {k: f"{v:.3f}s" for k, v in _timing.items()}
    logging.getLogger(__name__).info("OBSERVE TIMING BREAKDOWN: %s", _timing_msg)
    import sys
    print(f"OBSERVE TIMING BREAKDOWN: {_timing_msg}", file=sys.stderr, flush=True)

    return canvas, model_match_status, vlm_semantic_info


# ===== 路由：基础 =====

@app.get("/")
async def root():
    return {
        "name": "OpenClaw Desktop Agent API",
        "version": "1.0.0",
        "status": "running",
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/v1/providers/vision/health")
async def vision_provider_health():
    from src.perception.providers.remote_vision_provider import OmniParserRemoteVisionProvider

    provider = OmniParserRemoteVisionProvider()
    return provider.health_status()


# ===== 路由：软件管理 =====

@app.get("/api/v1/apps/search", response_model=SearchResponse)
async def search_apps(query: str, limit: int = 10):
    service = CatalogService()
    results = service.search(query)
    return SearchResponse(success=True, results=results[:limit], total=len(results))

@app.post("/api/v1/apps/launch", response_model=LaunchResponse)
async def launch_app(request: LaunchRequest):
    service = CatalogService()
    result = service.launch(request.name)
    return LaunchResponse(**result)

@app.get("/api/v1/apps", response_model=dict)
async def list_apps(limit: int = 100):
    from src.storage.db import Session
    from src.storage.repositories import AppRepository
    with Session() as sess:
        repo = AppRepository(sess)
        all_apps = repo.list_all()
        apps_data = [
            {
                "canonical_name": a.canonical_name,
                "display_name": a.display_name,
                "publisher": a.publisher,
                "install_location": a.install_location,
                "launch_path": a.launch_path,
                "version": a.version,
            }
            for a in all_apps[:limit]
        ]
        return {"success": True, "total": len(all_apps), "apps": apps_data}


# ===== 窗口解析 =====

@app.post("/api/v1/windows/resolve", response_model=WindowResolveResponse)
async def windows_resolve(request: WindowResolveRequest):
    """查找目标软件的窗口。只查找，不启动。

    返回 found_window、候选窗口列表、可选启动路径、建议下一步。
    Agent 根据返回决定是否调用 apps/launch 启动软件。
    """
    from src.windows.window_resolver import get_window_resolver

    resolver = get_window_resolver()
    result = resolver.resolve(request.app_name, prefer_session=request.prefer_session)

    return WindowResolveResponse(
        app_name=result.app_name,
        found_window=result.found_window,
        candidates=[
            WindowCandidateResponse(
                hwnd=c.hwnd,
                title=c.title,
                process_name=c.process_name,
                process_id=c.process_id,
                is_foreground=c.is_foreground,
                source=c.source,
                confidence=c.confidence,
            )
            for c in result.candidates
        ],
        launch_targets=[
            LaunchTargetResponse(
                app_name=t.app_name,
                executable=t.executable,
                args=t.args,
                description=t.description,
            )
            for t in result.launch_targets
        ],
        suggested_next_step=result.suggested_next_step,
    )


# ===== 核心协议：observe =====

@app.post("/api/v1/observe", response_model=ObserveResponse)
async def observe(request: ObserveRequest):
    """生成交互画布并缓存。后续 query/diff/remember 都通过 canvas_id 访问。"""
    from src.windows.window_enum import WindowEnumService
    from src.windows.screenshot_service import ScreenshotService

    started = time.perf_counter()
    hwnd = request.hwnd
    if hwnd is None:
        svc = WindowEnumService()
        fg = svc.get_foreground_window()
        if fg is None:
            raise HTTPException(status_code=404, detail="No foreground window found")
        hwnd = fg.hwnd

    # Normalize to top-level window
    hwnd = ScreenshotService.normalize_hwnd(hwnd)
    if request.capture_mode not in {"window", "screen_region"}:
        raise HTTPException(status_code=422, detail="invalid capture_mode: use window or screen_region")

    # Reject minimized windows
    if ScreenshotService.is_minimized(hwnd):
        raise HTTPException(
            status_code=422,
            detail="window_minimized: 窗口已最小化，请先恢复窗口再 observe",
        )

    enhancement_job_id = None
    response_processing_state: str | None = None
    try:
        if request.async_enhance:
            canvas, model_match_status, vlm_info = await run_in_threadpool(
                _do_observe, hwnd, allow_vlm=False, force_vlm=False, run_enhancement_phases=False,
                fast_perception=True, capture_mode=request.capture_mode,
            )
            from src.integration.processing_state import registry
            registry.set_canvas_state(canvas.canvas_id, "local_ready")
            response_processing_state = "local_ready"
            from src.integration.background_jobs import submit_canvas_job
            vlm_modes = request.vlm_modes or ["candidate_annotation", "region_understanding"]
            requested_vlm_modes = vlm_modes if (request.allow_vlm or request.force_vlm) else []
            enhancement_job_id, _created = submit_canvas_job(
                canvas_id=canvas.canvas_id,
                job_type="canvas_enhancement",
                queued_state="vlm_queued" if requested_vlm_modes else "fusion_running",
                running_state="vlm_running" if requested_vlm_modes else "fusion_running",
                success_state="enhanced_ready",
                fn=lambda cid=canvas.canvas_id, force=request.force_vlm, modes=requested_vlm_modes: _run_canvas_enhancement_sync(
                    cid, force, modes,
                ),
                use_vlm_semaphore=bool(requested_vlm_modes),
                max_attempts=2,
            )
        else:
            canvas, model_match_status, vlm_info = await run_in_threadpool(
                _do_observe, hwnd, allow_vlm=request.allow_vlm, force_vlm=request.force_vlm,
                capture_mode=request.capture_mode,
            )
            from src.integration.processing_state import registry
            registry.set_canvas_state(canvas.canvas_id, "enhanced_ready" if vlm_info.get("used") else "local_ready")
    except ValueError as e:
        # capture_invalid, window_minimized, etc.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from src.integration.processing_state import registry
    processing = registry.get_canvas_state(canvas.canvas_id)
    if response_processing_state is None:
        response_processing_state = processing.processing_state
    geo_regions = canvas.artifacts.get("geometric_regions") if canvas.artifacts else None
    geo_count = len(geo_regions) if geo_regions else 0
    fusion_diag = canvas.artifacts.get("fusion_diagnostics") if canvas.artifacts else {}
    perception_quality = canvas.artifacts.get("perception_quality") if canvas.artifacts else {}
    visual_pattern = canvas.artifacts.get("visual_pattern") if canvas.artifacts else {}
    roi_selection_plan = canvas.artifacts.get("roi_selection_plan") if canvas.artifacts else {}

    return ObserveResponse(
        canvas_id=canvas.canvas_id,
        elapsed_ms=_elapsed_ms(started),
        app_id=canvas.app.app_id,
        page_class=canvas.page_class,
        surface_type=canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type),
        stable=canvas.stable,
        loading=canvas.loading,
        partial=canvas.partial,
        confidence=canvas.page.class_confidence,
        element_count=len(canvas.elements),
        region_count=len(canvas.regions),
        providers_used=canvas.providers_used,
        providers_failed=canvas.providers_failed,
        canvas_schema_version=canvas.canvas_schema_version,
        page_model_id=canvas.page_model_id,
        state_template_id=canvas.state_template_id,
        model_match_status=model_match_status,
        vlm_semantic_used=vlm_info.get("used", False),
        vlm_semantic_status=vlm_info.get("status", ""),
        vlm_semantic_provider=vlm_info.get("provider", ""),
        processing_state=response_processing_state,
        enhancement_job_id=enhancement_job_id,
        geometric_region_count=geo_count,
        fusion_diagnostics=fusion_diag if isinstance(fusion_diag, dict) else {},
        perception_quality=perception_quality if isinstance(perception_quality, dict) else {},
        visual_pattern=visual_pattern if isinstance(visual_pattern, dict) else {},
        roi_selection_plan=roi_selection_plan if isinstance(roi_selection_plan, dict) else {},
    )


# ===== 核心协议：query =====

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """基于 canvas_id 查询真实候选点位。"""
    from src.canvas.query_engine import CanvasQueryEngine, QueryTarget

    cache = get_canvas_cache()
    canvas = cache.get(request.canvas_id)
    if canvas is None:
        raise HTTPException(
            status_code=404,
            detail=f"Canvas {request.canvas_id} not found. Call observe first.",
        )

    target = QueryTarget(
        text=request.target.text,
        semantic_role=request.target.semantic_role,
        region=request.target.region,
        natural_language=request.target.natural_language,
        composite=request.target.composite,
    )

    engine = CanvasQueryEngine()
    effective_canvas = _effective_canvas(canvas)
    result = engine.query(
        canvas=effective_canvas,
        target=target,
        max_results=request.max_results,
        min_confidence=request.min_confidence,
    )

    return QueryResponse(
        candidates=[_candidate_to_response(c) for c in result.candidates],
        suggestions=result.suggestions,
        total_matched=result.total_matched,
    )


# ===== 核心协议：diff =====

@app.post("/api/v1/diff", response_model=DiffResponse)
async def diff(request: DiffRequest):
    """对比前后两个真实画布。新画布自动缓存。"""
    from src.canvas.diff_engine import CanvasDiffEngine

    cache = get_canvas_cache()
    before = cache.get(request.previous_canvas_id)
    if before is None:
        raise HTTPException(
            status_code=404,
            detail=f"Previous canvas {request.previous_canvas_id} not found.",
        )

    try:
        after, _model_match_status, _vlm_info = await run_in_threadpool(_do_observe, request.hwnd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from src.memory.memory_service import MemoryService
    memory = MemoryService()
    before_effective = _effective_canvas(before)
    after_effective = _effective_canvas(after)
    engine = CanvasDiffEngine(
        key_store=memory.key_store,
        before_canvas=before_effective,
        after_canvas=after_effective,
    )
    changes = engine.diff(before_effective, after_effective, detail_level=request.detail_level)

    return DiffResponse(
        new_canvas_id=after.canvas_id,
        added=[_candidate_to_response(c) for c in changes.added],
        removed=[_candidate_to_response(c) for c in changes.removed],
        preserved=[_candidate_to_response(c) for c in changes.preserved],
        page_changed=changes.page_changed,
        page_class_changed=changes.page_class_changed,
        summary=changes.summary,
    )


# ===== 核心协议：remember =====

@app.post("/api/v1/remember", response_model=RememberResponse)
async def remember(request: RememberRequest):
    """将画布信息写入记忆系统（PageTemplate 表）。"""
    import hashlib
    import json as json_mod
    from src.storage.db import Session
    from src.storage.schema import PageTemplate

    cache = get_canvas_cache()
    canvas = cache.get(request.canvas_id)
    if canvas is None:
        raise HTTPException(
            status_code=404,
            detail=f"Canvas {request.canvas_id} not found. Call observe first.",
        )

    if not request.confirm:
        return RememberResponse(template_id="", status="rejected")

    app_id = canvas.app.app_id or "unknown"
    page_type = canvas.page_class

    # Build structure JSON from InteractionCanvas elements
    elements_data = []
    for elem in canvas.elements:
        elements_data.append({
            "element_id": elem.element_id,
            "semantic_role": elem.semantic_role.value if hasattr(elem.semantic_role, "value") else str(elem.semantic_role),
            "control_type": elem.control_type,
            "text": elem.text,
            "name": elem.name,
            "bounds": list(elem.bounds) if elem.bounds else None,
            "region_id": elem.region_id,
        })

    structure_json = json_mod.dumps(elements_data, sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(structure_json.encode()).hexdigest()[:16]
    template_id = f"{app_id}/{page_type}"

    try:
        with Session() as session:
            existing = (
                session.query(PageTemplate)
                .filter_by(app_id=app_id, page_type=page_type)
                .first()
            )
            if existing:
                existing.structure_json = elements_data
                existing.fingerprint_hash = fingerprint
                existing.confidence = canvas.page.class_confidence
                session.commit()
                status = "updated"
            else:
                template = PageTemplate(
                    app_id=app_id,
                    page_type=page_type,
                    fingerprint_hash=fingerprint,
                    structure_json=elements_data,
                    confidence=canvas.page.class_confidence,
                )
                session.add(template)
                session.commit()
                status = "created"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory write failed: {e}")

    return RememberResponse(
        template_id=template_id,
        status=status,
    )


# ===== 核心协议：feedback =====

@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """记录 Agent 反馈并返回置信度更新。"""
    from src.memory.feedback_manager import FeedbackManager
    from src.storage.db import Session

    cache = get_canvas_cache()
    canvas = cache.get(request.canvas_id)

    try:
        with Session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(
                canvas_id=request.canvas_id,
                candidate_key=request.candidate_key,
                feedback_type=request.feedback_type,
                detail=request.detail,
            )
            stats = manager.get_stats(request.candidate_key)
            confidence_update = manager.update_confidence(request.candidate_key, request.feedback_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback write failed: {e}")

    # Phase 8: recompute ConfidenceProfile after feedback
    try:
        from src.memory.evidence_fusion_engine import EvidenceFusionEngine

        if canvas is not None:
            fusion_engine = EvidenceFusionEngine()
            with Session() as fu_session:
                fusion_engine.apply_to_canvas(fu_session, canvas)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Post-feedback fusion recompute failed: %s", exc)

    return FeedbackResponse(
        recorded=True,
        confidence_update=round(confidence_update, 4),
    )


# ===== 核心协议：act（可选执行适配层） =====

@app.post("/api/v1/act", response_model=ActResponse)
async def act(request: ActRequest):
    """执行动作 — 可选执行适配层。

    act 不是核心决策模块。核心闭环是 observe→query→diff→remember→feedback。
    act 只执行 Agent 明确指定的候选动作，不做决策、不自动规划。
    当前阶段实现执行前 preflight/dry-run，不执行真实桌面动作。
    """
    if request.canvas_id:
        cache = get_canvas_cache()
        canvas = cache.get(request.canvas_id)
        if canvas is None:
            return ActResponse(
                execution_result="blocked",
                error_message=f"Canvas {request.canvas_id} not found.",
                warnings=["canvas_not_found"],
            )
        candidate = next(
            (item for item in list(getattr(canvas, "elements", None) or []) if str(getattr(item, "element_id", "") or "") == request.candidate_id),
            None,
        )
        if candidate is None:
            return ActResponse(
                execution_result="blocked",
                error_message=f"Candidate {request.candidate_id} not found in canvas {request.canvas_id}.",
                warnings=["candidate_not_found"],
            )
        preflight = _build_act_preflight_response(request, canvas, candidate)
        if str(request.action or "") in {"read", "inspect"} and not request.dry_run and preflight.action_plan.get("action_level") == "read_only":
            return _execute_act_read_only(request, canvas, candidate, preflight)
        if request.dry_run or preflight.execution_result == "blocked":
            return preflight
        if not request.execute_confirmed:
            return preflight
        return await _execute_act_confirmed(request, canvas, candidate, preflight)

    return ActResponse(
        execution_result="blocked",
        error_message=(
            "Act is an optional execution adapter, not a core decision module. "
            "The core protocol loop is observe→query→diff→remember→feedback. "
            "Action execution requires executor integration (UIA/DOM/Input) which is pending."
        ),
    )


# ===== 核心协议：capabilities =====

@app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    cache = get_canvas_cache()
    return CapabilitiesResponse(
        version="1.0",
        supported_actions=["read", "inspect", "click", "double_click", "type_text", "send", "key_press", "hotkey", "scroll"],
        action_policy=_capabilities_action_policy(),
        supported_query_types=["text", "semantic_role", "region", "natural_language", "composite"],
        supported_canvas_region_apis=[
            "GET /api/v1/canvases/{canvas_id}/operability",
            "POST /api/v1/canvases/{canvas_id}/read-region",
            "POST /api/v1/canvases/{canvas_id}/scroll-region",
        ],
        supported_memory_apis=[
            "GET /api/v1/transitions",
            "GET /api/v1/control-transitions",
            "POST /api/v1/control-transitions",
        ],
        providers=["uia", "ocr", "vision", "memory"],
        surface_types=["native_uia", "browser", "electron_webview", "canvas_self_drawn"],
    )


def _capabilities_action_policy() -> dict[str, dict[str, Any]]:
    """Expose machine-readable action safety boundaries for external agents."""
    controlled_verification = ["observe_after", "diff_after", "readback_after"]
    return {
        "read": {
            "action_level": "read_only",
            "enabled_for_execution": True,
            "executes_desktop_input": False,
            "requires_execute_confirmed": False,
            "verification": ["readback_current_canvas"],
        },
        "inspect": {
            "action_level": "read_only",
            "enabled_for_execution": True,
            "executes_desktop_input": False,
            "requires_execute_confirmed": False,
            "verification": ["readback_current_canvas"],
        },
        "click": {
            "action_level": "controlled",
            "enabled_for_execution": True,
            "executes_desktop_input": True,
            "requires_execute_confirmed": True,
            "verification": controlled_verification,
        },
        "double_click": {
            "action_level": "controlled",
            "enabled_for_execution": True,
            "executes_desktop_input": True,
            "requires_execute_confirmed": True,
            "verification": controlled_verification,
        },
        "scroll": {
            "action_level": "controlled",
            "enabled_for_execution": True,
            "executes_desktop_input": True,
            "requires_execute_confirmed": True,
            "verification": controlled_verification,
        },
        "type_text": {
            "action_level": "blocked_by_default",
            "enabled_for_execution": False,
            "executes_desktop_input": False,
            "requires_execute_confirmed": True,
            "requires_safe_to_type": True,
            "requires_expected_text": True,
            "verification": ["state_probe", "readback_expected_text"],
        },
        "send": {
            "action_level": "blocked_by_default",
            "enabled_for_execution": False,
            "executes_desktop_input": False,
            "requires_execute_confirmed": True,
            "requires_send_candidate": True,
            "requires_expected_text": True,
            "verification": ["state_probe", "readback_expected_text"],
        },
        "key_press": {
            "action_level": "review",
            "enabled_for_execution": False,
            "executes_desktop_input": False,
            "requires_execute_confirmed": True,
            "verification": ["manual_or_policy_review_required"],
        },
        "hotkey": {
            "action_level": "review",
            "enabled_for_execution": False,
            "executes_desktop_input": False,
            "requires_execute_confirmed": True,
            "verification": ["manual_or_policy_review_required"],
        },
    }


# ===== Console: window & canvas browsing =====

@app.get("/api/v1/windows", response_model=list[WindowListItem])
async def list_windows():
    """List all visible windows."""
    from src.windows.window_enum import WindowEnumService, WindowState

    svc = WindowEnumService()
    windows = svc.enumerate_all(refresh=True)
    fg = svc.get_foreground_window()
    fg_hwnd = fg.hwnd if fg else 0
    return [
        WindowListItem(
            hwnd=w.hwnd,
            title=w.title,
            class_name=w.class_name,
            process_name=w.process_name,
            process_id=w.process_id,
            is_foreground=(w.hwnd == fg_hwnd),
            is_minimized=(w.state == WindowState.MINIMIZED),
        )
        for w in windows
        if w.title  # filter out untitled windows
    ]


@app.get("/api/v1/canvases", response_model=list[CanvasSummary])
async def list_canvases():
    """List all cached canvas summaries."""
    from src.memory.page_identity import get_app_display_name
    cache = get_canvas_cache()
    items = cache.list_all()
    from src.storage.db import Session
    from src.storage.schema import CanvasSnapshotRecord

    with Session() as session:
        recent_snapshot_ids = [
            row.canvas_id for row in session.query(CanvasSnapshotRecord.canvas_id)
            .filter(CanvasSnapshotRecord.canvas_json.isnot(None))
            .order_by(CanvasSnapshotRecord.captured_at.desc())
            .limit(200)
            .all()
        ]
    cached_ids = {canvas_id for canvas_id, _canvas in items}
    for canvas_id in recent_snapshot_ids:
        if canvas_id in cached_ids:
            continue
        canvas = cache.get(canvas_id)
        if canvas is not None:
            items.append((canvas_id, canvas))
            cached_ids.add(canvas_id)

    result = []
    cache_keys = {canvas_id for canvas_id, _canvas in items}
    seen_canvas_ids: set[str] = set()
    for cache_key, canvas in items:
        object_canvas_id = getattr(canvas, "canvas_id", cache_key)
        if object_canvas_id != cache_key and object_canvas_id in cache_keys:
            import logging

            logging.getLogger(__name__).warning(
                "Skipping mismatched canvas cache alias: cache_key=%s object_canvas_id=%s",
                cache_key,
                object_canvas_id,
            )
            continue
        visible_canvas_id = object_canvas_id if object_canvas_id == cache_key else cache_key
        if visible_canvas_id in seen_canvas_ids:
            continue
        seen_canvas_ids.add(visible_canvas_id)
        from src.integration.processing_state import registry
        processing = registry.get_canvas_state(visible_canvas_id)
        has_screenshot = cache.get_screenshot(visible_canvas_id) is not None
        surface_val = canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type)
        app_id = canvas.app.app_id
        process_name = canvas.app.process_name if canvas.app else None
        display_name = get_app_display_name(app_id or "unknown", process_name)
        result.append(CanvasSummary(
            canvas_id=visible_canvas_id,
            app_id=app_id,
            display_name=display_name,
            window_title=canvas.window.title if canvas.window else "",
            process_name=process_name,
            page_class=canvas.page_class,
            surface_type=surface_val,
            element_count=len(canvas.elements),
            region_count=len(canvas.regions),
            captured_at=canvas.captured_at.isoformat() if canvas.captured_at else "",
            providers_used=canvas.providers_used,
            stable=canvas.stable,
            loading=canvas.loading,
            partial=canvas.partial,
            has_screenshot=has_screenshot,
            page_model_id=getattr(canvas, 'page_model_id', None),
            state_template_id=getattr(canvas, 'state_template_id', None),
            processing_state=processing.processing_state,
            processing_error=processing.last_error,
        ))
    result.sort(key=lambda item: item.captured_at or "", reverse=True)
    return result


def _delete_canvas_db_records(session, canvas_id: str, *, delete_snapshot: bool = True) -> dict[str, int]:
    """Delete persistent records tied to one canvas id."""
    from sqlalchemy import or_
    from src.storage.schema import (
        CandidateConfidenceProfileRecord,
        CandidateEvidenceRecord,
        CandidateOverrideRecord,
        CandidateStateRecord,
        CanvasProcessingRecord,
        CanvasSnapshotRecord,
        PageModelRecord,
        ProcessingJobRecord,
        StateTemplateRecord,
        VisualAssetRecord,
        VisualObservationRecord,
        VLMResponseRecord,
    )

    counts: dict[str, int] = {}
    snapshot_state_id = None
    if delete_snapshot:
        snapshot = session.query(CanvasSnapshotRecord).filter_by(canvas_id=canvas_id).first()
        if snapshot is not None:
            snapshot_state_id = snapshot.state_template_id
            session.delete(snapshot)
            counts["canvas_snapshot"] = 1
        else:
            counts["canvas_snapshot"] = 0

    counts["processing_state"] = session.query(CanvasProcessingRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["processing_jobs"] = session.query(ProcessingJobRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["candidate_evidence"] = session.query(CandidateEvidenceRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["candidate_overrides"] = session.query(CandidateOverrideRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["confidence_profiles"] = session.query(CandidateConfidenceProfileRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["visual_assets"] = session.query(VisualAssetRecord).filter_by(
        canvas_id=canvas_id,
    ).delete(synchronize_session=False)
    counts["visual_observations"] = session.query(VisualObservationRecord).filter(
        or_(
            VisualObservationRecord.canvas_id_a == canvas_id,
            VisualObservationRecord.canvas_id_b == canvas_id,
            VisualObservationRecord.observed_canvas_id == canvas_id,
        )
    ).delete(synchronize_session=False)

    if snapshot_state_id:
        session.flush()
        st = session.query(StateTemplateRecord).filter_by(
            state_template_id=snapshot_state_id,
        ).first()
        if st is not None:
            st.snapshot_count = session.query(CanvasSnapshotRecord).filter_by(
                state_template_id=snapshot_state_id,
            ).count()
            session.flush()

    return counts


def _cancel_canvas_jobs(canvas_id: str) -> int:
    """Cooperatively cancel queued/running jobs before deleting a canvas."""
    from src.integration.processing_state import TERMINAL_STATES, registry

    cancelled = 0
    for job in registry.list_jobs_for_canvas(canvas_id):
        if job.state not in TERMINAL_STATES:
            registry.cancel_job(job.job_id, reason="canvas_deleted")
            cancelled += 1
    return cancelled


@app.delete("/api/v1/canvases/{canvas_id}")
async def delete_canvas(canvas_id: str):
    """Delete a canvas cache entry, its warm screenshot, and persistent canvas records."""
    from src.storage.db import Session

    cancelled_jobs = _cancel_canvas_jobs(canvas_id)
    cache_result = get_canvas_cache().remove_persistent(canvas_id)
    with Session() as session:
        counts = _delete_canvas_db_records(session, canvas_id, delete_snapshot=True)
        session.commit()

    deleted_count = sum(counts.values()) + int(cache_result.get("memory", False)) + int(cache_result.get("warm", False))
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    return {
        "deleted": True,
        "canvas_id": canvas_id,
        "cache": cache_result,
        "records": counts,
        "cancelled_jobs": cancelled_jobs,
    }


@app.get("/api/v1/canvases/{canvas_id}", response_model=CanvasDetail)
async def get_canvas_detail(canvas_id: str):
    """Get full canvas detail including elements and regions."""
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")
    canvas = _effective_canvas(canvas)

    surface_val = canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type)

    # Build region responses
    regions = []
    for r in canvas.regions:
        region_elem_count = len(r.element_ids) if r.element_ids else 0
        subtype_val = r.subtype.value if hasattr(r.subtype, "value") else str(r.subtype)
        regions.append(RegionResponse(
            region_id=r.region_id,
            role=r.role,
            subtype=subtype_val,
            bounds=list(r.bounds) if r.bounds else None,
            element_count=region_elem_count,
        ))

    # Build provider trace
    pt = canvas.provider_trace
    provider_trace = ProviderTraceResponse(
        uia_used=pt.uia_used,
        ocr_used=pt.ocr_used,
        vision_used=pt.vision_used,
        vlm_used=pt.vlm_used,
        dom_used=pt.dom_used,
        provider_details=pt.provider_details if pt.provider_details else {},
    ) if pt else None

    # Window dimensions
    win_w, win_h = 0, 0
    if canvas.window and canvas.window.rect_client:
        rc = canvas.window.rect_client
        win_w = rc[2] - rc[0]
        win_h = rc[3] - rc[1]

    has_screenshot = cache.get_screenshot(canvas_id) is not None
    screenshot = cache.get_screenshot(canvas_id)
    screenshot_w, screenshot_h = screenshot.size if screenshot else (0, 0)
    from src.integration.processing_state import registry
    processing = registry.get_canvas_state(canvas_id)

    # Lookup page_model_name and state_label from DB
    page_model_name = None
    state_label = None
    pm_id = getattr(canvas, 'page_model_id', None)
    st_id = getattr(canvas, 'state_template_id', None)
    if pm_id or st_id:
        from src.storage.db import Session
        from src.storage.schema import PageModelRecord, StateTemplateRecord
        with Session() as db_session:
            if pm_id:
                pm = db_session.query(PageModelRecord).filter_by(page_model_id=pm_id).first()
                if pm:
                    page_model_name = pm.display_name
            if st_id:
                st = db_session.query(StateTemplateRecord).filter_by(state_template_id=st_id).first()
                if st:
                    state_label = st.state_label

    geo_regions = _geometric_regions_to_response(
        canvas.artifacts.get("geometric_regions") if canvas.artifacts else None
    )
    fusion_diag = canvas.artifacts.get("fusion_diagnostics") if canvas.artifacts else {}
    perception_quality = canvas.artifacts.get("perception_quality") if canvas.artifacts else {}
    visual_pattern = canvas.artifacts.get("visual_pattern") if canvas.artifacts else {}
    roi_selection_plan = canvas.artifacts.get("roi_selection_plan") if canvas.artifacts else {}
    ocr_blocks = canvas.artifacts.get("ocr_blocks") if canvas.artifacts else []
    vision_candidates = canvas.artifacts.get("vision_candidates") if canvas.artifacts else []
    roi_vlm_semantic_supplements = (
        canvas.artifacts.get("roi_vlm_semantic_supplements") if canvas.artifacts else []
    )
    roi_vlm_timeouts = canvas.artifacts.get("roi_vlm_timeouts") if canvas.artifacts else []
    roi_vlm_late_failures = canvas.artifacts.get("roi_vlm_late_failures") if canvas.artifacts else []
    roi_vlm_rejected_responses = canvas.artifacts.get("roi_vlm_rejected_responses") if canvas.artifacts else []
    observe_timing = canvas.artifacts.get("observe_timing") if canvas.artifacts else {}
    _refresh_provider_status_from_trace(canvas)

    return CanvasDetail(
        canvas_id=canvas.canvas_id,
        app_id=canvas.app.app_id,
        window_title=canvas.window.title if canvas.window else "",
        page_class=canvas.page_class,
        surface_type=surface_val,
        stable=canvas.stable,
        loading=canvas.loading,
        partial=canvas.partial,
        captured_at=canvas.captured_at.isoformat() if canvas.captured_at else "",
        window_width=win_w,
        window_height=win_h,
        screenshot_width=screenshot_w,
        screenshot_height=screenshot_h,
        elements=[_candidate_to_response(c) for c in canvas.elements],
        regions=regions,
        geometric_regions=geo_regions,
        provider_trace=provider_trace,
        perception_quality=perception_quality if isinstance(perception_quality, dict) else {},
        visual_pattern=visual_pattern if isinstance(visual_pattern, dict) else {},
        roi_selection_plan=roi_selection_plan if isinstance(roi_selection_plan, dict) else {},
        ocr_blocks=ocr_blocks if isinstance(ocr_blocks, list) else [],
        vision_candidates=vision_candidates if isinstance(vision_candidates, list) else [],
        roi_vlm_semantic_supplements=(
            roi_vlm_semantic_supplements if isinstance(roi_vlm_semantic_supplements, list) else []
        ),
        roi_vlm_timeouts=roi_vlm_timeouts if isinstance(roi_vlm_timeouts, list) else [],
        roi_vlm_late_failures=(
            roi_vlm_late_failures if isinstance(roi_vlm_late_failures, list) else []
        ),
        roi_vlm_rejected_responses=(
            roi_vlm_rejected_responses if isinstance(roi_vlm_rejected_responses, list) else []
        ),
        observe_timing=observe_timing if isinstance(observe_timing, dict) else {},
        providers_used=canvas.providers_used,
        providers_failed=canvas.providers_failed,
        has_screenshot=has_screenshot,
        page_model_id=pm_id,
        state_template_id=st_id,
        page_model_name=page_model_name,
        state_label=state_label,
        processing_state=processing.processing_state,
        processing_error=processing.last_error,
        processing_updated_at=processing.updated_at,
        fusion_diagnostics=fusion_diag if isinstance(fusion_diag, dict) else {},
    )


@app.get("/api/v1/canvases/{canvas_id}/operability")
async def get_canvas_operability(canvas_id: str):
    """Return a read-only agent-facing operability contract for one canvas."""
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")
    canvas = _effective_canvas(canvas)

    from scripts.analyze_page_operability import build_page_operability_report

    row = _canvas_to_operability_row(canvas)
    acceptance = _canvas_to_operability_acceptance(row, canvas)
    return build_page_operability_report(
        sample_matrix={"rows": [row]},
        recognition_acceptance={"rows": [acceptance]},
    )


@app.post("/api/v1/canvases/{canvas_id}/read-region", response_model=ReadRegionResponse)
async def read_canvas_region(canvas_id: str, request: ReadRegionRequest):
    """Read current text evidence from one canvas region without acting."""
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")
    canvas = _effective_canvas(canvas)
    region = _find_read_region(canvas, region_id=request.region_id, region_role=request.region_role)
    if region is None:
        target = request.region_id or request.region_role or ""
        raise HTTPException(status_code=404, detail=f"Region {target} not found")
    response = _build_read_region_response(
        canvas_id=canvas_id,
        canvas=canvas,
        region=region,
        include_ocr=request.include_ocr,
        include_elements=request.include_elements,
        allow_crop_ocr=request.allow_crop_ocr,
        screenshot=cache.get_screenshot(canvas_id) if request.allow_crop_ocr else None,
    )
    if request.method:
        response.method = request.method
    return response


@app.post("/api/v1/canvases/{canvas_id}/scroll-region", response_model=ScrollRegionResponse)
async def scroll_canvas_region(canvas_id: str, request: ScrollRegionRequest):
    """Return a scroll-region plan without executing desktop input."""
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")
    canvas = _effective_canvas(canvas)
    region = _find_read_region(canvas, region_id=request.region_id, region_role=request.region_role)
    if region is None:
        target = request.region_id or request.region_role or ""
        raise HTTPException(status_code=404, detail=f"Region {target} not found")

    scroll_context = _read_region_scroll_context(canvas, region)
    if scroll_context is None:
        return ScrollRegionResponse(
            canvas_id=canvas_id,
            region_id=str(getattr(region, "region_id", "") or ""),
            region_role=str(getattr(region, "role", "") or ""),
            dry_run=request.dry_run,
            status="blocked",
            can_execute=False,
            scroll_context=None,
            action_plan={},
            warnings=["no_scroll_context"],
        )

    action_plan = _scroll_region_action_plan(region, scroll_context, request)
    if not request.dry_run and not request.execute_confirmed:
        return ScrollRegionResponse(
            canvas_id=canvas_id,
            region_id=str(getattr(region, "region_id", "") or ""),
            region_role=str(getattr(region, "role", "") or ""),
            dry_run=request.dry_run,
            status="blocked",
            can_execute=False,
            scroll_context=scroll_context,
            action_plan=action_plan,
            warnings=["scroll_region_execute_requires_confirmation"],
        )
    if request.dry_run:
        return ScrollRegionResponse(
            canvas_id=canvas_id,
            region_id=str(getattr(region, "region_id", "") or ""),
            region_role=str(getattr(region, "role", "") or ""),
            dry_run=request.dry_run,
            status="planned",
            can_execute=False,
            scroll_context=scroll_context,
            action_plan=action_plan,
            warnings=["scroll_region_dry_run_only"],
        )

    hwnd = getattr(getattr(canvas, "window", None), "hwnd", 0) or 0
    if not hwnd:
        return ScrollRegionResponse(
            canvas_id=canvas_id,
            region_id=str(getattr(region, "region_id", "") or ""),
            region_role=str(getattr(region, "role", "") or ""),
            dry_run=request.dry_run,
            status="blocked",
            can_execute=False,
            scroll_context=scroll_context,
            action_plan=action_plan,
            warnings=["missing_hwnd"],
        )

    from src.execution.action_executor import ActionExecutor

    center_x, center_y = _scroll_region_center(region)
    execution = ActionExecutor().scroll_at(
        hwnd,
        client_x=center_x,
        client_y=center_y,
        delta=_scroll_region_delta(request),
    )
    execution_payload = {
        "success": bool(execution.success),
        "action": execution.action,
        "details": execution.details,
        "error": execution.error,
    }
    if not execution.success:
        return ScrollRegionResponse(
            canvas_id=canvas_id,
            region_id=str(getattr(region, "region_id", "") or ""),
            region_role=str(getattr(region, "role", "") or ""),
            dry_run=request.dry_run,
            status="failed",
            can_execute=True,
            scroll_context=scroll_context,
            action_plan=action_plan,
            execution_result=execution_payload,
            warnings=["scroll_region_execution_failed"],
        )

    before_read = _build_read_region_response(canvas_id=canvas_id, canvas=canvas, region=region)
    after_canvas = None
    if request.observe_after:
        after_canvas, _model_match_status, _vlm_info = await run_in_threadpool(
            _do_observe,
            hwnd,
            allow_vlm=False,
            force_vlm=False,
            run_enhancement_phases=False,
        )

    after_read = None
    if request.read_after and after_canvas is not None:
        after_region = _find_read_region(after_canvas, region_id=None, region_role=str(getattr(region, "role", "") or ""))
        if after_region is not None:
            after_read = _build_read_region_response(
                canvas_id=after_canvas.canvas_id,
                canvas=after_canvas,
                region=after_region,
                allow_crop_ocr=request.allow_crop_ocr_after,
                screenshot=cache.get_screenshot(after_canvas.canvas_id) if request.allow_crop_ocr_after else None,
            )
    stitched_text, stitch_report = _stitch_read_region_text(before_read, after_read)
    return ScrollRegionResponse(
        canvas_id=canvas_id,
        region_id=str(getattr(region, "region_id", "") or ""),
        region_role=str(getattr(region, "role", "") or ""),
        dry_run=request.dry_run,
        status="executed",
        can_execute=True,
        scroll_context=scroll_context,
        action_plan=action_plan,
        execution_result=execution_payload,
        after_canvas_id=after_canvas.canvas_id if after_canvas is not None else None,
        before_read=_model_dump(before_read),
        after_read=_model_dump(after_read) if after_read is not None else None,
        stitched_text=stitched_text,
        stitch_report=stitch_report,
        warnings=[] if after_read is not None else ["scroll_region_after_read_missing"],
    )


@app.get("/api/v1/canvases/{canvas_id}/processing", response_model=CanvasProcessingResponse)
async def get_canvas_processing(canvas_id: str):
    """Return current background processing state for a canvas."""
    from src.integration.processing_state import registry

    state = registry.get_canvas_state(canvas_id)
    jobs = registry.list_jobs_for_canvas(canvas_id)
    terminal_states = {"enhanced_ready", "failed", "cancelled"}
    active_jobs = [job for job in jobs if job.state not in terminal_states]
    current_phase = ""
    if state.processing_state not in {"local_ready", *terminal_states}:
        current_phase = state.processing_state
    return CanvasProcessingResponse(
        **state.to_dict(),
        current_phase=current_phase,
        is_processing=bool(current_phase),
        latest_job=JobStatusResponse(**jobs[0].to_dict()) if jobs else None,
        active_jobs=[JobStatusResponse(**job.to_dict()) for job in active_jobs],
    )


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: str):
    """Cooperatively cancel a queued/running background job."""
    from src.integration.processing_state import registry

    job = registry.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(**job.to_dict())


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Return current status for a background job."""
    from src.integration.processing_state import registry

    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(**job.to_dict())


@app.get("/api/v1/jobs", response_model=list[JobStatusResponse])
async def list_jobs(canvas_id: str | None = None, limit: int = 100):
    """Return recent background jobs, optionally filtered by canvas."""
    from src.integration.processing_state import registry

    limit = max(1, min(500, limit))
    jobs = registry.list_jobs_for_canvas(canvas_id) if canvas_id else registry.list_jobs(limit=limit)
    return [JobStatusResponse(**job.to_dict()) for job in jobs[:limit]]


@app.get("/api/v1/canvases/{canvas_id}/candidates/{element_id}/crop")
async def get_candidate_crop(
    canvas_id: str,
    element_id: str,
    padding: int = 4,
    describe: bool = False,
):
    """Crop a candidate element's region from the canvas screenshot.

    Returns a PNG image of the element bounds + padding pixels.
    In-memory only — no disk writes.

    If describe=true, returns JSON with a text description of the element
    (calls mimo-v2.5 vision API internally).
    """
    from io import BytesIO
    from fastapi.responses import Response

    cache = get_canvas_cache()
    screenshot = cache.get_screenshot(canvas_id)
    if screenshot is None:
        raise HTTPException(status_code=404, detail=f"Screenshot for canvas {canvas_id} not found")

    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")
    canvas = _effective_canvas(canvas)

    # Find the element
    target_el = None
    for el in canvas.elements:
        if el.element_id == element_id:
            target_el = el
            break
    if target_el is None:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found in canvas {canvas_id}")
    if not target_el.bounds or len(target_el.bounds) < 4:
        raise HTTPException(status_code=422, detail=f"Element {element_id} has no bounds")

    l, t, r, b = target_el.bounds
    img_w, img_h = screenshot.size

    # Apply padding and clamp to image bounds
    x1 = max(0, int(l) - padding)
    y1 = max(0, int(t) - padding)
    x2 = min(img_w, int(r) + padding)
    y2 = min(img_h, int(b) + padding)

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="Invalid crop region")

    cropped = screenshot.crop((x1, y1, x2, y2))

    if describe:
        from src.perception.image_describer import describe_image

        description = describe_image(cropped)
        return JSONResponse(
            content={
                "element_id": element_id,
                "description": description,
                "bounds": [l, t, r, b],
                "model": "mimo-v2.5",
            }
        )

    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/v1/canvases/{canvas_id}/screenshot")
async def get_canvas_screenshot(
    canvas_id: str,
    overlay: bool = False,
    layers: str | None = None,
):
    """Return canvas screenshot as PNG.

    - overlay=false: raw screenshot (default, main mode for frontend bbox overlay)
    - overlay=true: backend debug overlay preview (frontend should NOT draw its own bbox)
    - layers: comma-separated layer names, only used when overlay=true
    """
    from io import BytesIO
    from fastapi.responses import Response

    cache = get_canvas_cache()
    screenshot = cache.get_screenshot(canvas_id)
    if screenshot is None:
        raise HTTPException(status_code=404, detail=f"Screenshot for canvas {canvas_id} not found")

    if not overlay:
        # Raw screenshot — main mode
        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Overlay mode — backend debug preview
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    from src.perception.debug_tools import overlay_page_snapshot

    layer_set: set[str] | None = None
    if layers:
        layer_set = {l.strip() for l in layers.split(",") if l.strip()}

    result_img = overlay_page_snapshot(
        canvas, screenshot, layers=layer_set, return_image=True,
    )

    buf = BytesIO()
    result_img.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ===== Refine: candidate semantic review =====

@app.post("/api/v1/canvases/{canvas_id}/refine", response_model=RefineResponse)
async def refine_canvas(canvas_id: str, request: RefineRequest):
    """Run second-pass semantic review on canvas candidates.

    Refine never overwrites original fields — it only writes to
    visual_type, semantic_tags, role_label, role_confidence,
    role_source, role_evidence, refine_status on the Candidate objects.

    mode="agent": External Agent provides results[] directly — no inference.
    mode="heuristic": Engine infers from existing fields.
    """
    from src.canvas.refine_engine import RefineEngine, RefineResult

    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    engine = RefineEngine()

    if request.mode == "agent" and not request.results:
        raise HTTPException(
            status_code=400,
            detail={"errors": ["mode=agent requires non-empty results"]},
        )

    # Agent mode: convert RefineResultItem -> RefineResult
    agent_results = None
    if request.mode == "agent" and request.results:
        agent_results = [
            RefineResult(
                element_id=r.element_id,
                visual_type=r.visual_type,
                semantic_tags=r.semantic_tags,
                role_label=r.role_label,
                role_confidence=r.role_confidence,
                role_source=r.role_source,
                role_evidence=r.role_evidence,
                refine_status=r.refine_status,
            )
            for r in request.results
        ]

    output = engine.refine(
        canvas_id=canvas_id,
        candidates=canvas.elements,
        candidate_ids=request.candidate_ids,
        mode=request.mode,
        agent_results=agent_results,
    )

    # Return 400 if agent mode had invalid element_ids
    if output.errors:
        raise HTTPException(status_code=400, detail={"errors": output.errors})

    # Write refine results back to Candidate objects in the cached canvas
    result_map = {r.element_id: r for r in output.results}
    for el in canvas.elements:
        r = result_map.get(el.element_id)
        if r:
            el.visual_type = r.visual_type
            el.semantic_tags = r.semantic_tags
            el.role_label = r.role_label
            el.role_confidence = r.role_confidence
            el.role_source = r.role_source
            el.role_evidence = r.role_evidence
            el.refine_status = r.refine_status

    return RefineResponse(
        canvas_id=canvas_id,
        mode=output.mode,
        results=[
            RefineResultItem(
                element_id=r.element_id,
                visual_type=r.visual_type,
                semantic_tags=r.semantic_tags,
                role_label=r.role_label,
                role_confidence=r.role_confidence,
                role_source=r.role_source,
                role_evidence=r.role_evidence,
                refine_status=r.refine_status,
            )
            for r in output.results
        ],
        total_refined=output.total_refined,
        total_uncertain=output.total_uncertain,
        total_unreviewed=output.total_unreviewed,
    )


# ===== VLM Semantic Settings =====

def _request_updates(model: BaseModel) -> dict:
    """Return explicitly supplied non-null fields from a Pydantic model."""
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _semantic_settings_response(config) -> SemanticModelerSettingsResponse:
    from src.common.config_manager import semantic_modeler_to_public_dict

    return SemanticModelerSettingsResponse(**semantic_modeler_to_public_dict(config))


@app.get("/api/v1/settings/semantic-modeler", response_model=SemanticModelerSettingsResponse)
async def get_semantic_modeler_settings():
    """读取 VLM Semantic Modeler 运行时配置。不会返回明文 API key。"""
    from src.common.config_manager import load_config

    return _semantic_settings_response(load_config().semantic_modeler)


@app.put("/api/v1/settings/semantic-modeler", response_model=SemanticModelerSettingsResponse)
async def update_semantic_modeler_settings(request: SemanticModelerSettingsUpdateRequest):
    """保存 VLM Semantic Modeler 运行时配置到 data/local。"""
    from src.common.config_manager import save_local_semantic_modeler_config

    config = save_local_semantic_modeler_config(_request_updates(request))
    return _semantic_settings_response(config)


@app.post("/api/v1/settings/semantic-modeler/test", response_model=SemanticModelerTestResponse)
async def test_semantic_modeler_settings(request: SemanticModelerTestRequest):
    """检查或真实调用一次 VLM Semantic Provider。"""
    from dataclasses import asdict

    from PIL import Image

    from src.common.config_manager import SemanticModelerConfig, load_config
    from src.vlm.provider import VLMSemanticRequest as ProviderRequest, create_modeler_provider

    base = asdict(load_config().semantic_modeler)
    updates = _request_updates(request)
    run_call = bool(updates.pop("run_call", False))
    base.update(updates)
    config = SemanticModelerConfig(**{k: v for k, v in base.items() if k in SemanticModelerConfig.__dataclass_fields__})

    provider = create_modeler_provider(
        provider=config.provider,
        api_key=config.api_key,
        endpoint=config.endpoint,
        model=config.model,
        provider_variant=config.provider_variant,
        free_model_only=config.free_model_only,
        thinking_mode=config.thinking_mode,
        image_max_width=config.image_max_width,
        proxy_url=config.proxy_url,
        proxy_port=config.proxy_port,
    )
    if provider is None:
        return SemanticModelerTestResponse(
            success=False,
            available=False,
            provider=config.provider,
            model=config.model,
            endpoint=config.endpoint,
            message="Provider 未配置或类型不支持",
            error="provider_unavailable",
        )

    endpoint = getattr(provider, "_endpoint", config.endpoint)
    available = provider.is_available()
    if not run_call:
        return SemanticModelerTestResponse(
            success=available,
            available=available,
            provider=provider.name,
            model=getattr(provider, "_model", config.model),
            endpoint=endpoint,
            message="配置可用" if available else "配置不可用：检查 API key、模型名或 free-only 限制",
        )

    if not available:
        return SemanticModelerTestResponse(
            success=False,
            available=False,
            provider=provider.name,
            model=getattr(provider, "_model", config.model),
            endpoint=endpoint,
            message="配置不可用，未发送请求",
            error="provider_unavailable",
        )

    screenshot = Image.new("RGB", (160, 90), color=(18, 24, 38))
    provider_request = ProviderRequest(
        screenshot=screenshot,
        system_prompt='Return strict JSON only. The JSON schema is {"ok": boolean, "provider": string}.',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Return {"ok": true, "provider": "vlm-test"} as JSON.'},
                ],
            }
        ],
        max_tokens=128,
        provider_options={"timeout": config.timeout_seconds},
    )
    raw = provider.analyze_page(provider_request)
    success = bool(raw.raw_text) and raw.finish_reason != "error"
    return SemanticModelerTestResponse(
        success=success,
        available=True,
        provider=raw.provider_name or provider.name,
        model=raw.model_name or getattr(provider, "_model", config.model),
        endpoint=endpoint,
        message="测试调用成功" if success else "测试调用失败",
        error=None if success else (raw.error or "provider_call_failed"),
        latency_ms=raw.latency_ms,
        token_input=raw.token_input,
        token_output=raw.token_output,
        finish_reason=raw.finish_reason,
        raw_preview=(raw.raw_text or raw.error or "")[:500],
    )


def _semantic_config_from_updates(updates: dict):
    from dataclasses import asdict

    from src.common.config_manager import SemanticModelerConfig, load_config

    base = asdict(load_config().semantic_modeler)
    base.update(updates)
    return SemanticModelerConfig(**{k: v for k, v in base.items() if k in SemanticModelerConfig.__dataclass_fields__})


def _model_list_endpoint(provider: str, endpoint: str) -> str:
    value = (endpoint or "").strip().rstrip("/")
    provider = (provider or "").lower()
    if provider == "openrouter" or "openrouter.ai" in value:
        if not value:
            return "https://openrouter.ai/api/v1/models"
        if value.endswith("/chat/completions"):
            return value[: -len("/chat/completions")] + "/models"
        if value.endswith("/models"):
            return value
        if value.endswith("/v1") or value.endswith("/api/v1"):
            return f"{value}/models"
        return f"{value}/api/v1/models" if value == "https://openrouter.ai" else f"{value}/models"
    if provider in {"openai", "minimax", "mimo", "moonshot", "qwen", "doubao", "openai_compatible"}:
        if not value:
            value = {
                "moonshot": "https://api.moonshot.cn/v1",
                "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "doubao": "https://ark.cn-beijing.volces.com/api/v3",
                "minimax": "https://api.minimax.chat/v1",
                "mimo": "https://api.minimax.chat/v1",
            }.get(provider, "https://api.openai.com/v1")
        if value.endswith("/chat/completions"):
            return value[: -len("/chat/completions")] + "/models"
        if value.endswith("/models"):
            return value
        if value == "https://api.openai.com":
            return "https://api.openai.com/v1/models"
        if value.endswith("/v1") or value.endswith("/api/v1"):
            return f"{value}/models"
        return f"{value}/models"
    if provider == "anthropic":
        if not value:
            return "https://api.anthropic.com/v1/models"
        if value.endswith("/messages"):
            return value[: -len("/messages")] + "/models"
        if value.endswith("/models"):
            return value
        if value.endswith("/v1"):
            return f"{value}/models"
        return f"{value}/models"
    return value


def _is_vision_model(model: dict, model_id: str) -> bool:
    return _classify_model(model, model_id)[0] in {"image", "multimodal"}


def _classify_model(model: dict, model_id: str) -> tuple[str, str]:
    """Best-effort model capability classification.

    Provider model list formats are inconsistent. Prefer explicit modality
    metadata, then fall back to conservative name/description markers.
    """
    architecture = model.get("architecture") or {}
    input_modalities = model.get("input_modalities") or architecture.get("input_modalities") or []
    output_modalities = model.get("output_modalities") or architecture.get("output_modalities") or []
    input_set = {str(m).lower() for m in input_modalities}
    output_set = {str(m).lower() for m in output_modalities}
    has_image = bool(input_set & {"image", "vision", "visual"})
    has_text = not input_set or bool(input_set & {"text", "language"})
    if has_image and has_text:
        return "multimodal", "多模态（文本+图片）"
    if has_image:
        return "image", "图片模型"
    text = " ".join(
        str(v).lower()
        for v in [
            model_id,
            model.get("name", ""),
            model.get("description", ""),
            architecture.get("modality", ""),
        ]
    )
    image_markers = [
        "vision", "visual", "image", "ocr", "omni", "multimodal",
        "qwen2.5-vl", "qwen-vl", "pixtral", "llava",
        "gpt-4o", "gemini", "claude-3",
    ]
    if any(marker in text for marker in image_markers):
        return "multimodal", "可能支持图片"
    if input_set == {"text"} or output_set or model_id:
        return "text", "文本模型"
    return "unknown", "未知"


def _model_info_from_provider_item(item: dict, provider: str) -> SemanticModelerModelInfo:
    model_id = str(item.get("id") or item.get("model") or "")
    architecture = item.get("architecture") or {}
    pricing = item.get("pricing") or {}
    input_modalities = item.get("input_modalities") or architecture.get("input_modalities") or []
    output_modalities = item.get("output_modalities") or architecture.get("output_modalities") or []
    model_type, capability_label = _classify_model(item, model_id)
    pricing_values = [
        str(pricing.get(key) or "")
        for key in ("prompt", "input", "completion", "output")
        if pricing.get(key) is not None
    ]
    is_free = model_id.endswith(":free") or (
        bool(pricing_values) and all(v in {"", "0", "0.0", "0.000000"} for v in pricing_values)
    )
    return SemanticModelerModelInfo(
        id=model_id,
        name=str(item.get("name") or model_id),
        provider=provider,
        description=str(item.get("description") or ""),
        vision_capable=model_type in {"image", "multimodal"},
        model_type=model_type,
        capability_label=capability_label,
        is_free=is_free,
        input_modalities=[str(m) for m in input_modalities],
        output_modalities=[str(m) for m in output_modalities],
        pricing_prompt=str(pricing.get("prompt") or pricing.get("input") or ""),
        pricing_completion=str(pricing.get("completion") or pricing.get("output") or ""),
        context_length=item.get("context_length"),
    )


@app.post("/api/v1/settings/semantic-modeler/models", response_model=SemanticModelerModelListResponse)
async def list_semantic_modeler_models(request: SemanticModelerModelListRequest):
    """Fetch provider model list and mark likely vision-capable models."""
    from src.vlm.transport import create_vlm_session

    updates = _request_updates(request)
    vision_only = bool(updates.pop("vision_only", False))
    config = _semantic_config_from_updates(updates)
    provider = (config.provider or "").lower()
    endpoint = _model_list_endpoint(provider, config.endpoint)
    if not endpoint:
        return SemanticModelerModelListResponse(
            success=False,
            provider=config.provider,
            endpoint="",
            error="provider_does_not_support_model_listing",
        )

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if provider == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if provider == "openrouter":
        headers["HTTP-Referer"] = "http://127.0.0.1/openclaw"
        headers["X-Title"] = "OpenClaw Desktop Agent"

    try:
        session = create_vlm_session(proxy_url=config.proxy_url, proxy_port=config.proxy_port)
        response = session.get(endpoint, headers=headers, timeout=max(5, config.timeout_seconds))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return SemanticModelerModelListResponse(
            success=False,
            provider=config.provider,
            endpoint=endpoint,
            error=str(exc),
        )

    raw_models = payload.get("data", payload if isinstance(payload, list) else [])
    if not isinstance(raw_models, list):
        raw_models = []
    all_models = [_model_info_from_provider_item(m, config.provider) for m in raw_models if isinstance(m, dict)]
    models = list(all_models)
    if vision_only:
        models = [m for m in models if m.vision_capable]
    if config.free_model_only:
        models = [m for m in models if m.is_free]
    models.sort(key=lambda m: (not m.is_free, m.model_type != "multimodal", m.model_type != "image", m.id))
    return SemanticModelerModelListResponse(
        success=True,
        provider=config.provider,
        endpoint=endpoint,
        models=models,
        total_count=len(all_models),
        filtered_count=len(models),
    )


# ===== VLM Semantic =====

def _run_vlm_semantic_sync(request: VLMSemanticRequest) -> VLMSemanticResponse:
    """手动触发 VLM 语义分析（不经过完整 observe 流程）。

    用于调试和手动增强：对已缓存的 canvas 运行 VLM 分析并合并结果。
    """
    cache = get_canvas_cache()
    canvas = cache.get(request.canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {request.canvas_id} not found")

    from src.common.config_manager import load_config
    from src.vlm.modeler import get_modeler_singleton
    from src.perception.vlm_semantic_bridge import VLMSemanticBridge
    from src.storage.db import Session as DbSessionFactory

    config = load_config()
    sm = config.semantic_modeler
    modeler = get_modeler_singleton(
        enabled=sm.enabled,
        provider=sm.provider,
        api_key=sm.api_key,
        endpoint=sm.endpoint,
        model=sm.model,
        provider_variant=sm.provider_variant,
        free_model_only=sm.free_model_only,
        fallback_provider=sm.fallback_provider,
        fallback_model=sm.fallback_model,
        prompt_version=sm.prompt_version,
        timeout_seconds=sm.timeout_seconds,
        daily_call_limit=sm.daily_call_limit,
        monthly_budget_usd=sm.monthly_budget_usd,
        allow_free_models=sm.allow_free_models,
        save_raw_response=sm.save_raw_response,
        redact_dynamic_content=sm.redact_dynamic_content,
        thinking_mode=sm.thinking_mode,
        image_max_width=sm.image_max_width,
        db_session_factory=DbSessionFactory,
    )
    if modeler is None:
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status="skipped",
            error="VLM Semantic Modeler not configured or unavailable",
            error_code="provider_unavailable",
        )

    should, reason = modeler.should_analyze(force=request.force)
    if not should and not request.force:
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status="skipped",
            error=f"Analysis not needed: {reason}",
            message=f"跳过：{reason}",
        )

    # Build PromptInput from cached canvas
    screenshot = cache.get_screenshot(request.canvas_id)
    if screenshot is None:
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status="failed",
            error="No screenshot available for this canvas",
            error_code="no_screenshot",
            message="当前画布没有可用截图，无法调用 VLM",
        )
    import hashlib as _hashlib
    from io import BytesIO as _BytesIO

    _buf = _BytesIO()
    screenshot.save(_buf, format="PNG")
    screenshot_hash = _hashlib.sha256(_buf.getvalue()).hexdigest()

    prompt_input = _build_cached_vlm_prompt_input(
        canvas,
        screenshot,
        task_mode=request.task_mode or "candidate_annotation",
        candidate_ids=request.candidate_ids,
    )

    try:
        vlm_result = modeler.analyze(prompt_input, force=request.force)
    except Exception as exc:
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status="failed",
            error=str(exc),
            error_code="exception",
        )

    if vlm_result.status not in ("success", "cached") or vlm_result.model is None:
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status=vlm_result.status,
            provider=vlm_result.provider_name,
            model=vlm_result.model_name,
            error=vlm_result.error,
            error_code=vlm_result.error_code,
            message=vlm_result.error or "VLM 分析失败",
            latency_ms=vlm_result.latency_ms,
            token_input=vlm_result.token_input,
            token_output=vlm_result.token_output,
            from_cache=vlm_result.from_cache,
            warning_count=len(vlm_result.warnings),
        )

    # Merge into canvas (pass db_session_factory for shared_regions)
    bridge = VLMSemanticBridge()
    new_canvas = bridge.apply_to_canvas(canvas, vlm_result.model, db_session_factory=DbSessionFactory)

    # Ensure page_model_id / state_template_id are resolved
    page_model_id = new_canvas.page_model_id
    state_template_id = new_canvas.state_template_id
    if not page_model_id or not state_template_id:
        from src.memory.memory_service import MemoryService
        memory = MemoryService()
        has_screenshot = screenshot is not None
        page_model_id, state_template_id, _, _ = memory.resolve_page_model(
            new_canvas, has_screenshot=has_screenshot,
            vlm_app_name=vlm_result.model.app_identity.app_name or None,
            vlm_display_name=vlm_result.model.app_identity.display_name or None,
            vlm_state_label=vlm_result.model.page_state.state_label or None,
            vlm_state_flags=list(vlm_result.model.page_state.state_flags) if vlm_result.model.page_state.state_flags else None,
        )
        new_canvas.page_model_id = page_model_id
        new_canvas.state_template_id = state_template_id

    # Resolve candidate keys (needed for persist_corrections stable_key mapping)
    from src.memory.memory_service import MemoryService
    memory = MemoryService()
    memory.resolve_candidate_keys(new_canvas, page_model_id, state_template_id)

    # Persist VLM corrections (same path as observe)
    element_to_stable_key: dict[str, str] = {}
    for elem in new_canvas.elements:
        sid = getattr(elem, 'stable_key_id', None)
        if sid:
            element_to_stable_key[elem.element_id] = sid
    try:
        with DbSessionFactory() as session:
            augmented = _augment_vlm_model_dict(
                vlm_result.model,
                new_canvas,
                image_size=(screenshot.width, screenshot.height),
            )
            _update_latest_vlm_response_context(
                session,
                screenshot_hash=screenshot_hash,
                page_model_id=page_model_id,
                state_template_id=state_template_id,
                parsed_model=augmented,
            )
            session.commit()
        bridge.persist_corrections(
            vlm_result.model,
            app_id=new_canvas.app.app_id or "unknown",
            page_model_id=page_model_id,
            state_template_id=state_template_id,
            element_to_stable_key=element_to_stable_key,
            db_session_factory=DbSessionFactory,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("VLM persist_corrections failed: %s", exc)

    # Record VLM transitions
    if vlm_result.model.transitions:
        try:
            from src.memory.transition_graph import TransitionGraphManager
            with DbSessionFactory() as session:
                tgm = TransitionGraphManager(session)
                tgm.record_vlm_transitions(
                    [t.to_dict() for t in vlm_result.model.transitions],
                    page_model_id,
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("VLM record_transitions failed: %s", exc)

    # Update cache
    cache.put(new_canvas, screenshot=screenshot)

    return VLMSemanticResponse(
        canvas_id=new_canvas.canvas_id,
        status=vlm_result.status,
        provider=vlm_result.provider_name,
        model=vlm_result.model_name,
        message="VLM 分析成功" if vlm_result.status == "success" else "使用 VLM 缓存结果",
        latency_ms=vlm_result.latency_ms,
        token_input=vlm_result.token_input,
        token_output=vlm_result.token_output,
        from_cache=vlm_result.from_cache,
        region_count=len(vlm_result.model.regions),
        control_count=len(vlm_result.model.fixed_controls),
        correction_count=len(vlm_result.model.candidate_corrections),
        dynamic_zone_count=len(vlm_result.model.dynamic_zones),
        warning_count=len(vlm_result.warnings),
    )


def _run_canvas_enhancement_sync(canvas_id: str, force: bool, modes: list[str]) -> VLMSemanticResponse:
    """Run asynchronous enhancement phases for an already local-ready canvas."""
    from src.canvas.canvas_cache import get_canvas_cache
    from src.integration.processing_state import registry
    from src.storage.db import Session

    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise RuntimeError(f"Canvas {canvas_id} not found for enhancement")

    source_hwnd = (getattr(canvas, "artifacts", None) or {}).get("source_hwnd")
    if source_hwnd:
        from src.integration.enhancement_policy import full_local_enhancement_strategy

        strategy = full_local_enhancement_strategy(canvas)
        registry.set_canvas_state(canvas_id, "fusion_running")
        full_canvas, _model_match_status, _vlm_info = _do_observe(
            int(source_hwnd),
            allow_vlm=False,
            force_vlm=False,
            run_enhancement_phases=False,
            fast_perception=False,
            lightweight_uia=(strategy == "lightweight_uia_vision"),
        )
        full_canvas.artifacts["local_enhancement_strategy"] = strategy
        full_canvas_id = full_canvas.canvas_id
        full_screenshot = cache.get_screenshot(full_canvas_id)
        if full_canvas_id != canvas_id:
            try:
                cache.remove_persistent(full_canvas_id)
                with Session() as cleanup_session:
                    _delete_canvas_db_records(cleanup_session, full_canvas_id, delete_snapshot=True)
                    cleanup_session.commit()
            except Exception as exc:  # noqa: BLE001 - duplicate cleanup is best-effort.
                logger.warning(
                    "Background enhancement duplicate canvas cleanup failed for %s: %s",
                    full_canvas_id,
                    exc,
                    exc_info=True,
                )
        full_canvas.canvas_id = canvas_id
        _preserve_async_semantic_artifacts(full_canvas, canvas, cache.get(canvas_id))
        full_canvas.artifacts["source_hwnd"] = source_hwnd
        cache.put(full_canvas, screenshot=full_screenshot or cache.get_screenshot(canvas_id), persist=True)
        canvas = full_canvas

    last = _run_vlm_modes_sync(canvas_id, force, modes) if modes else VLMSemanticResponse(
        canvas_id=canvas_id,
        status="skipped",
        message="VLM not requested; running local enhancement phases",
    )

    canvas = cache.get(canvas_id)
    screenshot = cache.get_screenshot(canvas_id)
    if canvas is None:
        raise RuntimeError(f"Canvas {canvas_id} not found after local enhancement")

    try:
        registry.set_canvas_state(canvas_id, "fusion_running")
        from src.memory.evidence_collector import EvidenceCollector
        from src.memory.evidence_fusion_engine import EvidenceFusionEngine

        with Session() as ev_session:
            EvidenceCollector().collect(
                ev_session,
                canvas,
                vlm_used=bool(modes and last.status in ("success", "cached")),
            )

        with Session() as fu_session:
            EvidenceFusionEngine().apply_to_canvas(fu_session, canvas)

        registry.set_canvas_state(canvas_id, "anchor_running")
        if screenshot is not None:
            from src.memory.visual_anchor_engine import VisualAnchorEngine

            with Session() as va_session:
                VisualAnchorEngine().apply_to_canvas(va_session, canvas, screenshot)
                va_session.commit()

        from src.memory.permanence_state_machine import PermanenceStateMachine
        from src.storage.schema import StableCandidateKey as SCKModel

        sm = PermanenceStateMachine()
        with Session() as sm_session:
            for element in canvas.elements:
                stable_key_id = getattr(element, "stable_key_id", None)
                if not stable_key_id:
                    continue
                row = sm_session.query(SCKModel).filter(SCKModel.key_id == stable_key_id).first()
                verify_count = row.verify_count if row else 1
                sm.evaluate_and_update(
                    sm_session,
                    stable_key_id,
                    getattr(element, "attributes", None) or {},
                    verify_count=verify_count,
                    seen_count=verify_count,
                )

        _refresh_provider_status_from_trace(canvas)
        cache.put(canvas, screenshot=screenshot, persist=True)
    except Exception as exc:
        raise RuntimeError(f"post_vlm_enhancement_failed: {exc}") from exc

    if modes and last.status == "failed":
        raise RuntimeError(last.error or last.message or "vlm_enhancement_failed")
    return last


_ASYNC_SEMANTIC_ARTIFACT_KEYS = (
    "roi_vlm_semantic_supplements",
    "roi_vlm_timeouts",
    "roi_vlm_late_failures",
    "roi_vlm_rejected_responses",
)


def _preserve_async_semantic_artifacts(target_canvas, *source_canvases) -> None:
    """Carry concurrent ROI VLM diagnostics across full-canvas replacement."""
    target_artifacts = getattr(target_canvas, "artifacts", None)
    if target_artifacts is None:
        target_canvas.artifacts = {}
        target_artifacts = target_canvas.artifacts

    for key in _ASYNC_SEMANTIC_ARTIFACT_KEYS:
        merged = list(target_artifacts.get(key) or [])
        seen = {repr(item) for item in merged}
        for source_canvas in source_canvases:
            source_artifacts = getattr(source_canvas, "artifacts", None) or {}
            for item in list(source_artifacts.get(key) or []):
                marker = repr(item)
                if marker in seen:
                    continue
                seen.add(marker)
                merged.append(item)
        if merged:
            target_artifacts[key] = merged


def _run_vlm_modes_sync(canvas_id: str, force: bool, modes: list[str]) -> VLMSemanticResponse:
    """Run one or more VLM enhancement modes against the same canvas."""
    normalized_modes = [m for m in modes if m] or ["candidate_annotation"]
    last: VLMSemanticResponse | None = None
    for mode in normalized_modes:
        last = _run_vlm_semantic_sync(
            VLMSemanticRequest(
                canvas_id=canvas_id,
                force=force,
                sync=True,
                task_mode=mode,
                candidate_ids=[],
            )
        )
        if last.status == "failed":
            return last
    return last or VLMSemanticResponse(canvas_id=canvas_id, status="skipped")


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _result_status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _roi_vlm_worker_timeout_seconds(deadline_ms: int) -> float:
    """Reserve HTTP/server overhead while preserving tiny test/debug budgets."""
    deadline_seconds = max(0.001, int(deadline_ms) / 1000.0)
    if deadline_ms <= 250:
        return round(deadline_seconds, 3)
    return round(max(0.1, deadline_seconds - 0.15), 3)


def _vlm_provider_timeout_seconds(interactive_timeout_seconds: float) -> float:
    """Keep provider calls alive after the interactive response budget expires."""
    try:
        from src.common.config_manager import load_config

        configured = float(getattr(load_config().semantic_modeler, "timeout_seconds", 0) or 0)
    except Exception:
        configured = 0.0
    ceiling = configured if configured > 0 else 15.0
    target = max(float(interactive_timeout_seconds) + 5.0, 15.0)
    return round(min(target, ceiling), 3)


def _merge_late_layout_audit_future(canvas_id: str, canvas, screenshot, future) -> None:
    try:
        from src.canvas.canvas_cache import get_canvas_cache
        from src.vlm.layout_audit_contract import merge_layout_audit_response

        response = future.result()
        merge_layout_audit_response(canvas, response, status="timeout_late_success")
        _refresh_roi_selection_plan(canvas)
        get_canvas_cache().put(canvas, screenshot=screenshot, persist=True)
    except Exception as exc:  # noqa: BLE001 - late diagnostics only.
        try:
            canvas.artifacts.setdefault("vlm_layout_audit_late_failures", []).append({
                "status": "timeout_late_failed",
                "error": str(exc),
                "received_at": time.time(),
            })
            get_canvas_cache().put(canvas, screenshot=screenshot, persist=True)
        except Exception:
            pass
        logger.warning("Late layout audit VLM failed for canvas=%s: %s", canvas_id, exc, exc_info=True)


def _refresh_roi_selection_plan(canvas) -> None:
    from src.perception.roi_selection import RoiSelectionPlanner

    plan = RoiSelectionPlanner().plan(canvas)
    canvas.artifacts["roi_selection_plan"] = plan.to_dict()


def _response_dict(model: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.post("/api/v1/canvases/{canvas_id}/roi-vlm", response_model=RoiVlmSupplementResponse)
async def roi_vlm_supplement(canvas_id: str, request: RoiVlmSupplementRequest):
    """Prepare ROI-scoped VLM supplement jobs.

    Dry-run returns compact local ROI/candidate payloads without calling VLM.
    Non-dry-run calls the configured ROI provider worker under a deadline and
    keeps late responses mergeable through roi_contract.
    """
    started = time.perf_counter()
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    from src.vlm.roi_contract import build_roi_vlm_jobs

    jobs = build_roi_vlm_jobs(canvas)
    requested = {str(roi_id) for roi_id in request.roi_ids if roi_id}
    if requested:
        jobs = [job for job in jobs if job.get("roi_id") in requested]

    deadline_ms = max(1, int(request.deadline_ms))
    if request.dry_run:
        return RoiVlmSupplementResponse(
            canvas_id=canvas_id,
            status="dry_run",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            jobs=jobs,
            result_status_counts={},
            message="ROI VLM dry-run payloads prepared; no provider call made",
        )

    if not jobs:
        return RoiVlmSupplementResponse(
            canvas_id=canvas_id,
            status="skipped",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            jobs=[],
            result_status_counts={},
            message="No ROI VLM jobs selected",
        )

    screenshot = cache.get_screenshot(canvas_id)
    if screenshot is None:
        raise HTTPException(status_code=422, detail="No screenshot available for ROI VLM")

    from src.vlm.roi_async import RoiVlmAsyncRunner
    from src.vlm.roi_profile import roi_vlm_profile_for_job
    from src.vlm.roi_provider_worker import run_roi_vlm_provider_job
    from src.integration.processing_state import registry

    def persist_late_roi_result(late_canvas, _late_job, _late_result):
        late_status = str((_late_result or {}).get("status") or "")
        if late_status == "timeout_late_success":
            registry.set_canvas_state(canvas_id, "semantic_late_merged")
        elif late_status:
            registry.set_canvas_state(canvas_id, "semantic_late_failed", error=late_status)
        cache.put(late_canvas, screenshot=screenshot, persist=True)

    registry.set_canvas_state(canvas_id, "semantic_pending")
    worker_timeout_seconds = _roi_vlm_worker_timeout_seconds(deadline_ms)
    provider_timeout_seconds = _vlm_provider_timeout_seconds(worker_timeout_seconds)
    runner = RoiVlmAsyncRunner(
        timeout_seconds=worker_timeout_seconds,
        on_late_result=persist_late_roi_result,
    )
    def run_provider_for_roi(roi_job):
        return run_roi_vlm_provider_job(
            roi_job,
            screenshot=screenshot,
            timeout_seconds=provider_timeout_seconds,
            **roi_vlm_profile_for_job(roi_job),
        )

    results = []
    try:
        results = runner.submit_many(canvas, jobs, run_provider_for_roi)
    finally:
        runner.shutdown(wait=False)

    cache.put(canvas, screenshot=screenshot, persist=True)
    if any(result.get("status") == "timeout" for result in results):
        status = "timeout"
        registry.set_canvas_state(canvas_id, "semantic_timeout")
    elif any(result.get("status") == "failed" for result in results):
        status = "failed"
        first_error = next((str(result.get("error") or "") for result in results if result.get("status") == "failed"), "")
        registry.set_canvas_state(canvas_id, "semantic_failed", error=first_error)
    elif any(result.get("status") == "rejected" for result in results):
        status = "partial"
        registry.set_canvas_state(canvas_id, "semantic_partial")
    else:
        status = "success"
        registry.set_canvas_state(canvas_id, "semantic_ready")

    return RoiVlmSupplementResponse(
        canvas_id=canvas_id,
        status=status,
        deadline_ms=deadline_ms,
        elapsed_ms=_elapsed_ms(started),
        accept_late=bool(request.accept_late),
        jobs=jobs,
        results=results,
        result_status_counts=_result_status_counts(results),
        message="ROI VLM provider jobs processed",
    )


@app.post("/api/v1/canvases/{canvas_id}/layout-audit-vlm", response_model=LayoutAuditVlmResponse)
async def layout_audit_vlm(canvas_id: str, request: LayoutAuditVlmRequest):
    """Run low-frequency full-screenshot layout audit when local layout failed."""
    started = time.perf_counter()
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    from src.vlm.layout_audit_contract import build_layout_audit_job, merge_layout_audit_response

    deadline_ms = max(1, int(request.deadline_ms))
    job = build_layout_audit_job(canvas)
    if job is None:
        return LayoutAuditVlmResponse(
            canvas_id=canvas_id,
            status="skipped",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            job=None,
            message="layout audit not needed by current perception quality",
        )
    if request.dry_run:
        return LayoutAuditVlmResponse(
            canvas_id=canvas_id,
            status="dry_run",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            job=job,
            message="VLM layout audit dry-run payload prepared; no provider call made",
        )

    screenshot = cache.get_screenshot(canvas_id)
    if screenshot is None:
        raise HTTPException(status_code=422, detail="No screenshot available for VLM layout audit")

    from src.vlm.layout_audit_provider_worker import run_layout_audit_provider_job

    worker_timeout_seconds = _roi_vlm_worker_timeout_seconds(deadline_ms)
    provider_timeout_seconds = _vlm_provider_timeout_seconds(worker_timeout_seconds)
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openclaw-layout-audit-vlm")
    future = executor.submit(
        run_layout_audit_provider_job,
        job,
        screenshot=screenshot,
        timeout_seconds=provider_timeout_seconds,
        image_max_edge=512,
        max_tokens=384,
    )
    try:
        response = future.result(timeout=worker_timeout_seconds)
    except TimeoutError:
        canvas.artifacts.setdefault("vlm_layout_audit_timeouts", []).append({
            "timeout_seconds": worker_timeout_seconds,
            "timed_out_at": time.time(),
            "discard_late_response": False,
        })
        future.add_done_callback(
            lambda done: _merge_late_layout_audit_future(
                canvas_id,
                canvas,
                screenshot,
                done,
            )
        )
        executor.shutdown(wait=False)
        cache.put(canvas, screenshot=screenshot, persist=True)
        return LayoutAuditVlmResponse(
            canvas_id=canvas_id,
            status="timeout",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            job=job,
            result={
                "status": "timeout",
                "late_response_policy": "accept_if_review_only",
            },
            message="VLM layout audit timed out; late response remains mergeable",
        )
    except Exception as exc:  # noqa: BLE001 - provider diagnostics.
        executor.shutdown(wait=False)
        return LayoutAuditVlmResponse(
            canvas_id=canvas_id,
            status="failed",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            job=job,
            result={"status": "failed", "error": str(exc)},
            message="VLM layout audit provider job failed",
        )
    else:
        executor.shutdown(wait=False)
        try:
            validation = merge_layout_audit_response(canvas, response, status="success")
            if validation.accepted:
                _refresh_roi_selection_plan(canvas)
            cache.put(canvas, screenshot=screenshot, persist=True)
            result = {
                "status": "success" if validation.accepted else "rejected",
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            }
            return LayoutAuditVlmResponse(
                canvas_id=canvas_id,
                status=result["status"],
                deadline_ms=deadline_ms,
                elapsed_ms=_elapsed_ms(started),
                accept_late=bool(request.accept_late),
                job=job,
                result=result,
                message="VLM layout audit provider job processed",
            )
        except Exception as exc:  # noqa: BLE001 - validation diagnostics.
            return LayoutAuditVlmResponse(
                canvas_id=canvas_id,
                status="failed",
                deadline_ms=deadline_ms,
                elapsed_ms=_elapsed_ms(started),
                accept_late=bool(request.accept_late),
                job=job,
                result={"status": "failed", "error": str(exc)},
                message="VLM layout audit merge failed",
            )


@app.post("/api/v1/canvases/{canvas_id}/semantic-completion", response_model=SemanticCompletionResponse)
async def semantic_completion(canvas_id: str, request: SemanticCompletionRequest):
    """Quality-driven semantic supplement orchestration.

    The caller gets one decision surface instead of manually choosing between
    full-screenshot layout audit and ROI-scoped semantic supplement.
    """
    started = time.perf_counter()
    cache = get_canvas_cache()
    canvas = cache.get(canvas_id)
    if canvas is None:
        raise HTTPException(status_code=404, detail=f"Canvas {canvas_id} not found")

    deadline_ms = max(1, int(request.deadline_ms))
    artifacts = getattr(canvas, "artifacts", None) or {}
    quality = artifacts.get("perception_quality") or {}
    warnings = set(quality.get("warnings") or [])
    screenshot = cache.get_screenshot(canvas_id)
    layout_needed = "layout_audit_needed" in warnings
    layout_blocked = (
        "layout_audit_blocked_no_screenshot" in warnings
        or (layout_needed and screenshot is None and not request.dry_run)
    )
    stages: list[dict[str, Any]] = []

    if layout_blocked:
        stages.append({
            "name": "layout_audit",
            "status": "blocked",
            "reason": "no_screenshot",
        })
        return SemanticCompletionResponse(
            canvas_id=canvas_id,
            status="blocked",
            deadline_ms=deadline_ms,
            elapsed_ms=_elapsed_ms(started),
            accept_late=bool(request.accept_late),
            stages=stages,
            next_action="recapture_with_screenshot",
            message="Semantic completion needs screenshot evidence before VLM layout audit or ROI supplement",
        )

    layout_result: dict[str, Any] | None = None
    if layout_needed:
        layout_response = await layout_audit_vlm(
            canvas_id,
            LayoutAuditVlmRequest(
                deadline_ms=deadline_ms,
                accept_late=bool(request.accept_late),
                dry_run=bool(request.dry_run),
            ),
        )
        layout_result = _response_dict(layout_response)
        stages.append({
            "name": "layout_audit",
            "status": layout_result.get("status", ""),
            "elapsed_ms": layout_result.get("elapsed_ms", 0),
        })
        if layout_result.get("status") in {"failed", "rejected"}:
            return SemanticCompletionResponse(
                canvas_id=canvas_id,
                status=str(layout_result.get("status") or "failed"),
                deadline_ms=deadline_ms,
                elapsed_ms=_elapsed_ms(started),
                accept_late=bool(request.accept_late),
                stages=stages,
                layout_audit=layout_result,
                next_action="inspect_layout_audit_result",
                message="Layout audit did not produce accepted review-only regions",
            )
    else:
        stages.append({
            "name": "layout_audit",
            "status": "skipped",
            "reason": "not_needed",
        })

    roi_response = await roi_vlm_supplement(
        canvas_id,
        RoiVlmSupplementRequest(
            roi_ids=list(request.roi_ids or []),
            deadline_ms=deadline_ms,
            accept_late=bool(request.accept_late),
            dry_run=bool(request.dry_run),
        ),
        warnings=["act_requires_canvas_id_for_preflight"],
    )
    roi_result = _response_dict(roi_response)
    stages.append({
        "name": "roi_vlm",
        "status": roi_result.get("status", ""),
        "elapsed_ms": roi_result.get("elapsed_ms", 0),
        "job_count": len(roi_result.get("jobs") or []),
    })

    if request.dry_run:
        status = "dry_run"
        next_action = "run_semantic_completion"
    elif roi_result.get("status") == "skipped":
        status = "skipped"
        next_action = "inspect_roi_selection_plan"
    elif roi_result.get("status") == "timeout":
        status = "timeout"
        next_action = "accept_late_roi_results"
    elif roi_result.get("status") in {"failed", "partial"}:
        status = str(roi_result.get("status"))
        next_action = "inspect_roi_vlm_result"
    else:
        status = "success"
        next_action = "query_or_review_semantic_supplements"

    return SemanticCompletionResponse(
        canvas_id=canvas_id,
        status=status,
        deadline_ms=deadline_ms,
        elapsed_ms=_elapsed_ms(started),
        accept_late=bool(request.accept_late),
        stages=stages,
        layout_audit=layout_result,
        roi_vlm=roi_result,
        next_action=next_action,
        message="Semantic completion orchestration processed",
    )


@app.post("/api/v1/vlm-semantic", response_model=VLMSemanticResponse)
async def vlm_semantic(request: VLMSemanticRequest):
    """Trigger VLM semantic analysis. Defaults to background execution."""
    if not request.sync:
        cache = get_canvas_cache()
        if cache.get(request.canvas_id) is None:
            raise HTTPException(status_code=404, detail=f"Canvas {request.canvas_id} not found")
        from src.integration.background_jobs import submit_canvas_job
        from src.integration.processing_state import registry

        job_id, created = submit_canvas_job(
            canvas_id=request.canvas_id,
            job_type="vlm_semantic",
            queued_state="vlm_queued",
            running_state="vlm_running",
            success_state="enhanced_ready",
            fn=lambda: _run_vlm_semantic_sync(
                VLMSemanticRequest(
                    canvas_id=request.canvas_id,
                    force=request.force,
                    sync=True,
                    task_mode=request.task_mode,
                    candidate_ids=list(request.candidate_ids),
                ),
            ),
            use_vlm_semaphore=True,
            max_attempts=2,
        )
        state = registry.get_canvas_state(request.canvas_id)
        return VLMSemanticResponse(
            canvas_id=request.canvas_id,
            status="queued" if created else "running",
            message="VLM 语义增强任务已启动" if created else "VLM 语义增强任务已在运行",
            job_id=job_id,
            processing_state=state.processing_state,
        )

    return await run_in_threadpool(_run_vlm_semantic_sync, request)


# ===== E Phase 2: Page Model API =====

@app.get("/api/v1/page-models", response_model=list[PageModelResponse])
async def list_page_models(app_id: str | None = None):
    """获取所有 PageModel 列表。"""
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session

    memory = MemoryService()
    with Session() as session:
        models = memory.page_model_store.get_page_models(session, app_id)

    return [
        PageModelResponse(
            page_model_id=m.page_model_id,
            app_id=m.app_id,
            page_class_prefix=m.page_class_prefix,
            display_name=m.display_name,
            surface_type=m.surface_type,
            observe_count=m.observe_count,
            state_count=m.state_count,
            last_seen_at=m.last_seen_at,
        )
        for m in models
    ]


@app.delete("/api/v1/page-models/{page_model_id}")
async def delete_page_model(page_model_id: str):
    """删除一个页面模型及其状态、截图和关联证据。"""
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session
    from src.storage.schema import (
        CandidateConfidenceProfileRecord,
        CandidateEvidenceRecord,
        CandidateOverrideRecord,
        CanvasSnapshotRecord,
        VLMResponseRecord,
        VisualAssetRecord,
    )

    cache = get_canvas_cache()
    with Session() as session:
        canvas_ids = [
            row.canvas_id
            for row in session.query(CanvasSnapshotRecord.canvas_id).filter_by(
                page_model_id=page_model_id,
            ).all()
        ]
        if not canvas_ids:
            existing_models = MemoryService().page_model_store.get_page_models(session)
            if not any(m.page_model_id == page_model_id for m in existing_models):
                raise HTTPException(status_code=404, detail=f"Page model {page_model_id} not found")

        record_counts: dict[str, int] = {}
        cancelled_jobs = 0
        cache_counts = {"memory": 0, "warm": 0}
        for canvas_id in canvas_ids:
            cancelled_jobs += _cancel_canvas_jobs(canvas_id)
            cache_result = cache.remove_persistent(canvas_id)
            cache_counts["memory"] += int(cache_result.get("memory", False))
            cache_counts["warm"] += int(cache_result.get("warm", False))
            for key, value in _delete_canvas_db_records(session, canvas_id, delete_snapshot=False).items():
                record_counts[key] = record_counts.get(key, 0) + value

        record_counts["candidate_overrides_page"] = session.query(CandidateOverrideRecord).filter_by(
            page_model_id=page_model_id,
        ).delete(synchronize_session=False)
        record_counts["candidate_evidence_page"] = session.query(CandidateEvidenceRecord).filter_by(
            page_model_id=page_model_id,
        ).delete(synchronize_session=False)
        record_counts["confidence_profiles_page"] = session.query(CandidateConfidenceProfileRecord).filter_by(
            page_model_id=page_model_id,
        ).delete(synchronize_session=False)
        record_counts["visual_assets_page"] = session.query(VisualAssetRecord).filter_by(
            page_model_id=page_model_id,
        ).delete(synchronize_session=False)
        record_counts["vlm_responses_page"] = session.query(VLMResponseRecord).filter_by(
            page_model_id=page_model_id,
        ).delete(synchronize_session=False)

        memory = MemoryService()
        deleted = memory.page_model_store.delete_page_model(session, page_model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Page model {page_model_id} not found")
        session.commit()

    return {
        "deleted": True,
        "page_model_id": page_model_id,
        "canvas_count": len(canvas_ids),
        "cache": cache_counts,
        "records": record_counts,
        "cancelled_jobs": cancelled_jobs,
    }


@app.get("/api/v1/page-models/{page_model_id}/state-templates", response_model=list[StateTemplateResponse])
async def list_state_templates(page_model_id: str):
    """获取某个 PageModel 的所有 StateTemplate。"""
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session

    memory = MemoryService()
    with Session() as session:
        templates = memory.page_model_store.get_state_templates(session, page_model_id)

    return [
        StateTemplateResponse(
            state_template_id=t.state_template_id,
            page_model_id=t.page_model_id,
            page_class=t.page_class,
            state_label=t.state_label,
            fixed_element_count=t.fixed_element_count,
            total_element_count=t.total_element_count,
            verify_count=t.verify_count,
            snapshot_count=t.snapshot_count,
            last_seen_at=t.last_seen_at,
        )
        for t in templates
    ]


@app.delete("/api/v1/state-templates/{state_template_id}")
async def delete_state_template(state_template_id: str):
    """删除一个页面状态模板及其截图和关联证据。"""
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session
    from src.storage.schema import (
        CandidateConfidenceProfileRecord,
        CandidateEvidenceRecord,
        CandidateOverrideRecord,
        VLMResponseRecord,
        VisualAssetRecord,
    )

    cache = get_canvas_cache()
    with Session() as session:
        memory = MemoryService()
        template = memory.page_model_store.get_state_template_by_id(session, state_template_id)
        if template is None:
            raise HTTPException(status_code=404, detail=f"State template {state_template_id} not found")
        snapshots = memory.page_model_store.get_canvas_snapshots(
            session,
            state_template_id=state_template_id,
        )
        canvas_ids = [s.canvas_id for s in snapshots]

        record_counts: dict[str, int] = {}
        cancelled_jobs = 0
        cache_counts = {"memory": 0, "warm": 0}
        for canvas_id in canvas_ids:
            cancelled_jobs += _cancel_canvas_jobs(canvas_id)
            cache_result = cache.remove_persistent(canvas_id)
            cache_counts["memory"] += int(cache_result.get("memory", False))
            cache_counts["warm"] += int(cache_result.get("warm", False))
            for key, value in _delete_canvas_db_records(session, canvas_id, delete_snapshot=False).items():
                record_counts[key] = record_counts.get(key, 0) + value

        record_counts["candidate_overrides_state"] = session.query(CandidateOverrideRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        record_counts["candidate_evidence_state"] = session.query(CandidateEvidenceRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        record_counts["confidence_profiles_state"] = session.query(CandidateConfidenceProfileRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        record_counts["visual_assets_state"] = session.query(VisualAssetRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)
        record_counts["vlm_responses_state"] = session.query(VLMResponseRecord).filter_by(
            state_template_id=state_template_id,
        ).delete(synchronize_session=False)

        deleted, page_model_id, _ = memory.page_model_store.delete_state_template(
            session,
            state_template_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail=f"State template {state_template_id} not found")
        session.commit()

    return {
        "deleted": True,
        "state_template_id": state_template_id,
        "page_model_id": page_model_id,
        "canvas_count": len(canvas_ids),
        "cache": cache_counts,
        "records": record_counts,
        "cancelled_jobs": cancelled_jobs,
    }


@app.get("/api/v1/page-models/tree", response_model=PageModelTreeResponse)
async def get_full_tree():
    """获取完整的软件模型树：PageModel → StateTemplate → CanvasSnapshot。"""
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session

    memory = MemoryService()
    cache = get_canvas_cache()
    with Session() as session:
        models = memory.page_model_store.get_page_models(session)
        visible_model_ids: set[str] = set()
        all_templates = []
        all_snapshots = []
        for m in models:
            templates = memory.page_model_store.get_state_templates(session, m.page_model_id)
            all_templates.extend(templates)
            if templates:
                visible_model_ids.add(m.page_model_id)
            for t in templates:
                snapshots = memory.page_model_store.get_canvas_snapshots(
                    session, state_template_id=t.state_template_id,
                )
                all_snapshots.extend(snapshots)
                if snapshots:
                    visible_model_ids.add(m.page_model_id)
        models = [m for m in models if m.page_model_id in visible_model_ids]

    return PageModelTreeResponse(
        page_models=[
            PageModelResponse(
                page_model_id=m.page_model_id,
                app_id=m.app_id,
                page_class_prefix=m.page_class_prefix,
                display_name=m.display_name,
                surface_type=m.surface_type,
                observe_count=m.observe_count,
                state_count=m.state_count,
                last_seen_at=m.last_seen_at,
            )
            for m in models
        ],
        state_templates=[
            StateTemplateResponse(
                state_template_id=t.state_template_id,
                page_model_id=t.page_model_id,
                page_class=t.page_class,
                state_label=t.state_label,
                fixed_element_count=t.fixed_element_count,
                total_element_count=t.total_element_count,
                verify_count=t.verify_count,
                snapshot_count=t.snapshot_count,
                last_seen_at=t.last_seen_at,
            )
            for t in all_templates
        ],
        canvas_snapshots=[
            CanvasSnapshotResponse(
                snapshot_id=s.snapshot_id,
                canvas_id=s.canvas_id,
                page_model_id=s.page_model_id,
                state_template_id=s.state_template_id,
                captured_at=s.captured_at,
                element_count=s.element_count,
                has_screenshot=s.has_screenshot,
                available=cache.get(s.canvas_id) is not None,
            )
            for s in all_snapshots
        ],
    )


def _is_root_shell_virtual_candidate(
    canonical_role: str,
    relative_bounds: list[float] | tuple[float, ...],
    provider_sources: list[str] | tuple[str, ...],
) -> bool:
    """Return True for UIA root-window shells that should not render as model controls."""
    sources = {str(source).lower() for source in (provider_sources or [])}
    if "uia" not in sources:
        return False

    role = str(canonical_role or "").strip().lower()
    if role not in {"title_bar", "sidebar", "container", "layout"}:
        return False

    if len(relative_bounds or []) < 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in relative_bounds[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height >= 0.85 and width >= 0.90 and height >= 0.90


@app.get("/api/v1/state-templates/{state_template_id}/virtual-model", response_model=VirtualModelDetail)
async def get_virtual_model(state_template_id: str):
    """获取虚拟模型 — 按优先级合并：manual override > VLM > persistent > canvas transient。

    数据来源：
    1. CandidateOverride（source=manual 最高优先级）
    2. VLM semantic model（vlm_responses 最新 parsed_model）
    3. Persistent stable candidates（stable_candidate_keys）
    4. Current canvas transient elements
    """
    import json as _json
    from src.memory.memory_service import MemoryService
    from src.memory.candidate_key_store import CandidateKeyStore
    from src.memory.candidate_override_store import CandidateOverrideStore
    from src.storage.db import Session

    memory = MemoryService()
    key_store = CandidateKeyStore()
    override_store = CandidateOverrideStore()

    with Session() as session:
        st = memory.page_model_store.get_state_template_by_id(session, state_template_id)
        if st is None:
            raise HTTPException(status_code=404, detail=f"State template {state_template_id} not found")

        models = memory.page_model_store.get_page_models(session)
        pm = next((m for m in models if m.page_model_id == st.page_model_id), None)

        persistent_candidates = key_store.get_candidates_for_state(session, state_template_id)

        # Phase 9: Look up confidence profiles for persistent candidates
        from src.memory.confidence_profile_store import ConfidenceProfileStore
        profile_store = ConfidenceProfileStore()
        confidence_profiles: dict[str, dict] = {}
        for pc in persistent_candidates:
            prof = profile_store.get_latest(session, pc.key_id)
            if prof:
                # Only keep the dimension fields needed by frontend
                confidence_profiles[pc.key_id] = {
                    "spatial_confidence": prof.get("spatial_confidence", 0.0),
                    "semantic_confidence": prof.get("semantic_confidence", 0.0),
                    "text_confidence": prof.get("text_confidence", 0.0),
                    "structure_confidence": prof.get("structure_confidence", 0.0),
                    "visual_anchor_confidence": prof.get("visual_anchor_confidence", 0.0),
                    "memory_confidence": prof.get("memory_confidence", 0.0),
                    "action_confidence": prof.get("action_confidence", 0.0),
                    "conflict_penalty": prof.get("conflict_penalty", 0.0),
                    "fused_confidence": prof.get("fused_confidence", 0.0),
                    "source_count": prof.get("source_count", 0),
                    "source_diversity": prof.get("source_diversity", 0),
                }

        # Get candidate overrides for this state (includes VLM + manual)
        overrides = override_store.list(
            session,
            state_template_id=state_template_id,
            page_model_id=st.page_model_id,
        )

        # Get latest VLM semantic model for this exact state.
        # Same-app history is only used for shared/global regions, never for
        # fixed_controls. Otherwise controls from another page/state can leak
        # into the current virtual model.
        vlm_model_dict = None
        shared_vlm_regions: list[dict] = []
        try:
            result = session.execute(
                __import__("sqlalchemy").text(
                    """SELECT v.parsed_model FROM vlm_responses v
                       WHERE v.status = 'success' AND v.parsed_model != ''
                       AND v.state_template_id = :st_id
                       ORDER BY v.created_at DESC LIMIT 1"""
                ),
                {"st_id": state_template_id},
            )
            row = result.fetchone()
            if row and row[0]:
                vlm_model_dict = _json.loads(row[0])

            shared_result = session.execute(
                __import__("sqlalchemy").text(
                    """SELECT v.parsed_model FROM vlm_responses v
                       JOIN page_models p ON v.page_model_id = p.page_model_id
                       WHERE v.status = 'success' AND v.parsed_model != ''
                       AND p.app_id = :app_id
                       AND (v.state_template_id IS NULL OR v.state_template_id != :st_id)
                       ORDER BY v.created_at DESC LIMIT 50"""
                ),
                {"st_id": state_template_id, "app_id": st.app_id},
            )
            seen_shared_region_ids: set[str] = set()
            for shared_row in shared_result.fetchall():
                try:
                    shared_model = _json.loads(shared_row[0])
                except Exception:
                    continue
                for region in shared_model.get("shared_regions") or []:
                    rid = region.get("region_id") or _json.dumps(region, sort_keys=True)
                    if rid in seen_shared_region_ids:
                        continue
                    seen_shared_region_ids.add(rid)
                    shared_region = dict(region)
                    shared_region["shared"] = True
                    shared_vlm_regions.append(shared_region)
        except Exception:
            pass

        snapshots = memory.page_model_store.get_canvas_snapshots(
            session, state_template_id=state_template_id,
        )

    # Parse layout_fingerprint
    fp = {}
    try:
        fp = _json.loads(st.layout_fingerprint) if st.layout_fingerprint else {}
    except (ValueError, TypeError):
        pass

    # --- Build candidates with priority merging ---
    seen_keys: set[str] = set()
    result_candidates: list[VirtualModelCandidate] = []

    # Priority 1: CandidateOverrides (manual > vlm_semantic)
    # Sort overrides so manual source comes first
    _SOURCE_SORT = {"manual": 0, "agent": 1, "vlm_semantic": 2}
    sorted_overrides = sorted(overrides, key=lambda o: _SOURCE_SORT.get(o.source or "manual", 99))

    for override in sorted_overrides:
        if override.status != "active":
            continue
        # Determine key_id for this override
        key_id = override.stable_key_id or override.element_id or override.scope_key
        if key_id in seen_keys:
            continue
        seen_keys.add(key_id)
        result_candidates.append(VirtualModelCandidate(
            key_id=key_id,
            canonical_text="",
            canonical_role=override.semantic_role or "",
            canonical_region=override.region_id or "",
            role_label=override.label,
            visual_type=override.visual_type or "",
            relative_bounds=override.relative_bounds or [],
            confidence=1.0,
            provider_sources=[override.source or "manual"],
            is_fixed_control=False,
            permanence_state="new",
            verify_count=0,
            seen_count=0,
            source=override.source or "manual",
        ))

    # --- Resolve canvas dimensions for pixel→relative conversion ---
    canvas_w, canvas_h = 0, 0
    has_available_canvas = False
    cache = get_canvas_cache()
    latest_canvas = None
    for snap in snapshots:
        c = cache.get(snap.canvas_id)
        if c is None:
            continue
        has_available_canvas = True
        latest_canvas = c
        if c.window:
            if c.window.rect_client:
                x1, y1, x2, y2 = c.window.rect_client
                canvas_w, canvas_h = x2 - x1, y2 - y1
            if (canvas_w <= 0 or canvas_h <= 0) and c.window.rect_screen:
                x1, y1, x2, y2 = c.window.rect_screen
                canvas_w, canvas_h = x2 - x1, y2 - y1
        if canvas_w <= 0 or canvas_h <= 0:
            for elem in c.elements:
                if elem.bounds and len(elem.bounds) >= 4:
                    canvas_w = max(canvas_w, elem.bounds[2])
                    canvas_h = max(canvas_h, elem.bounds[3])
        break  # only use latest available canvas

    if (canvas_w <= 0 or canvas_h <= 0) and vlm_model_dict:
        image_size = vlm_model_dict.get("image_size") or []
        if len(image_size) >= 2:
            canvas_w, canvas_h = int(image_size[0]), int(image_size[1])

    def _pixel_to_relative(bounds: list[int | float]) -> list[float]:
        """Convert pixel bounds [l, t, r, b] to 0-1 relative bounds."""
        if len(bounds) < 4 or canvas_w <= 0 or canvas_h <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(bounds[0]) / canvas_w, float(bounds[1]) / canvas_h,
                float(bounds[2]) / canvas_w, float(bounds[3]) / canvas_h]

    # Build vlm_control_id → stable_key_id mapping from canvas elements
    vlm_ctrl_to_stable_key: dict[str, str] = {}
    if latest_canvas is not None:
        for elem in latest_canvas.elements:
            attrs = getattr(elem, 'attributes', None) or {}
            vlm_cid = attrs.get("vlm_control_id")
            sid = getattr(elem, 'stable_key_id', None)
            if vlm_cid and sid:
                vlm_ctrl_to_stable_key[vlm_cid] = sid

    # Priority 2: VLM fixed_controls from parsed_model (not already covered by overrides)
    vlm_regions: list[dict] = []
    if vlm_model_dict:
        for r in vlm_model_dict.get("regions", []):
            vlm_regions.append(r)
        for r in shared_vlm_regions:
            vlm_regions.append(r)
        control_aliases = vlm_model_dict.get("control_aliases") or {}
        for ctrl in vlm_model_dict.get("fixed_controls", []):
            ctrl_id = ctrl.get("control_id", "")
            # Map VLM control_id to stable_key_id if available,
            # so overrides targeting the same element via stable_key_id match.
            mapped_key = (
                ctrl.get("stable_key_id")
                or control_aliases.get(ctrl_id)
                or vlm_ctrl_to_stable_key.get(ctrl_id)
                or ctrl_id
            )
            aliases = {
                str(a)
                for a in (ctrl.get("aliases") or [])
                if a
            }
            if ctrl_id:
                aliases.add(str(ctrl_id))
            if mapped_key:
                aliases.add(str(mapped_key))
            if aliases & seen_keys:
                continue
            seen_keys.update(aliases)
            bounds = ctrl.get("bounds", [])
            rel_bounds = _pixel_to_relative(bounds)
            result_candidates.append(VirtualModelCandidate(
                key_id=mapped_key,
                canonical_text=ctrl.get("text", ""),
                canonical_role=ctrl.get("semantic_role", ""),
                canonical_region=ctrl.get("region_id", ""),
                visual_type=ctrl.get("visual_type", ""),
                relative_bounds=rel_bounds,
                confidence=ctrl.get("confidence", 0.8),
                provider_sources=["vlm"],
                is_fixed_control=ctrl.get("interactable", False),
                verify_count=0,
                seen_count=0,
                permanence_state="new",
                source="vlm",
            ))

    # Priority 3: Persistent stable candidates (not already covered)
    for c in persistent_candidates:
        if c.key_id in seen_keys:
            continue
        if _is_root_shell_virtual_candidate(c.canonical_role, c.relative_bounds, c.provider_sources):
            seen_keys.add(c.key_id)
            continue
        seen_keys.add(c.key_id)
        result_candidates.append(VirtualModelCandidate(
            key_id=c.key_id,
            canonical_text=c.canonical_text,
            canonical_role=c.canonical_role,
            canonical_region=c.canonical_region,
            role_label=c.role_label,
            semantic_tags=c.semantic_tags,
            visual_type=c.visual_type,
            relative_bounds=c.relative_bounds,
            confidence=c.confidence,
            provider_sources=c.provider_sources,
            is_fixed_control=c.is_fixed_control,
            verify_count=c.verify_count,
            seen_count=c.seen_count,
            permanence_state=c.permanence_state,
            confidence_profile=confidence_profiles.get(c.key_id),
            source="persistent",
        ))

    # Priority 4: Canvas transient elements (not already covered)
    if latest_canvas is not None:
        for elem in latest_canvas.elements:
            elem_key = getattr(elem, 'stable_key_id', None) or f"transient_{elem.element_id}"
            if elem_key in seen_keys:
                continue
            seen_keys.add(elem_key)
            role_str = elem.semantic_role.value if hasattr(elem.semantic_role, "value") else str(elem.semantic_role)
            bounds = elem.bounds or (0, 0, 0, 0)
            rel_bounds = _pixel_to_relative(bounds)
            tags = sorted(elem.semantic_tags) if elem.semantic_tags else []
            sources = sorted(elem.provider_sources) if elem.provider_sources else []
            if _is_root_shell_virtual_candidate(role_str, rel_bounds, sources):
                continue
            result_candidates.append(VirtualModelCandidate(
                key_id=elem_key,
                canonical_text=elem.text or "",
                canonical_role=role_str,
                canonical_region=getattr(elem, 'region_id', '') or "",
                role_label=getattr(elem, 'role_label', None),
                semantic_tags=tags,
                visual_type=getattr(elem, 'visual_type', '') or "",
                relative_bounds=rel_bounds,
                confidence=elem.confidence if hasattr(elem, 'confidence') else 0.0,
                provider_sources=sources,
                is_fixed_control=False,
                verify_count=0,
                seen_count=0,
                permanence_state="new",
                source="canvas_transient",
            ))

    workbench = _build_model_workbench_summary(
        app_id=st.app_id,
        display_name=pm.display_name if pm else None,
        surface_type=pm.surface_type if pm else None,
        page_class=st.page_class,
        state_label=st.state_label,
        candidates=result_candidates,
        regions=vlm_regions,
        vlm_model_dict=vlm_model_dict,
    )
    try:
        from src.memory.model_template_store import ModelTemplateStore
        ModelTemplateStore().upsert_workbench_summary(
            session,
            app_id=st.app_id,
            page_model_id=st.page_model_id,
            state_template_id=st.state_template_id,
            display_name=pm.display_name if pm else None,
            surface_type=pm.surface_type if pm else None,
            page_class=st.page_class,
            state_label=st.state_label,
            workbench=workbench,
        )
        session.commit()
    except Exception:
        # Workbench persistence is best-effort; the virtual model response itself
        # should remain available even if template snapshot storage fails.
        pass

    return VirtualModelDetail(
        state_template_id=st.state_template_id,
        page_model_id=st.page_model_id,
        app_id=st.app_id,
        display_name=pm.display_name if pm else None,
        page_class=st.page_class,
        state_label=st.state_label,
        surface_type=pm.surface_type if pm else None,
        layout_fingerprint=fp,
        fixed_element_count=st.fixed_element_count,
        total_element_count=st.total_element_count,
        verify_count=st.verify_count,
        snapshot_count=st.snapshot_count,
        last_seen_at=st.last_seen_at,
        has_available_canvas=has_available_canvas,
        candidates=result_candidates,
        regions=vlm_regions,
        vlm_semantic_model=vlm_model_dict,
        model_layers=workbench["model_layers"],
        app_shell=workbench["app_shell"],
        region_templates=workbench["region_templates"],
        element_templates=workbench["element_templates"],
        visible_items=workbench["visible_items"],
        missing_suggestions=workbench["missing_suggestions"],
    )


def _override_to_response(data) -> CandidateOverrideResponse:
    return CandidateOverrideResponse(
        override_id=data.override_id,
        scope_key=data.scope_key,
        app_id=data.app_id,
        page_model_id=data.page_model_id,
        state_template_id=data.state_template_id,
        canvas_id=data.canvas_id,
        element_id=data.element_id,
        stable_key_id=data.stable_key_id,
        label=data.label,
        semantic_role=data.semantic_role,
        visual_type=data.visual_type,
        region_id=data.region_id,
        kind=data.kind,
        relative_bounds=data.relative_bounds,
        absolute_bounds=data.absolute_bounds,
        source=data.source,
        status=data.status,
        created_at=data.created_at,
        updated_at=data.updated_at,
    )


def _transition_to_response(info) -> TransitionEdgeResponse:
    last = info.last_observed_at
    if hasattr(last, "isoformat"):
        last_value = last.isoformat()
    elif last is None:
        last_value = None
    else:
        last_value = str(last)
    return TransitionEdgeResponse(
        from_page_class=info.from_page_class,
        to_page_class=info.to_page_class,
        trigger_action=info.trigger_action,
        trigger_candidate_key=info.trigger_candidate_key,
        observe_count=info.observe_count,
        success_count=info.success_count,
        success_rate=info.success_rate,
        last_observed_at=last_value,
        drifted=info.drifted,
    )


def _control_transition_to_response(info) -> ControlTransitionResponse:
    last = info.last_observed_at
    if hasattr(last, "isoformat"):
        last_value = last.isoformat()
    elif last is None:
        last_value = None
    else:
        last_value = str(last)
    return ControlTransitionResponse(
        candidate_key=info.candidate_key,
        stable_key_id=info.stable_key_id,
        from_page_class=info.from_page_class,
        to_page_class=info.to_page_class,
        action_type=info.action_type,
        observe_count=info.observe_count,
        success_count=info.success_count,
        failure_count=info.failure_count,
        success_rate=info.success_rate,
        last_observed_at=last_value,
        canvas_id_before=info.canvas_id_before,
        canvas_id_after=info.canvas_id_after,
    )


@app.get("/api/v1/transitions", response_model=TransitionGraphResponse)
async def list_transitions(
    page_class: str | None = None,
    candidate_key: str | None = None,
):
    """Read historical transition graph edges without running any recognition work."""
    if not page_class and not candidate_key:
        raise HTTPException(status_code=400, detail="page_class or candidate_key is required")
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    with Session() as session:
        manager = TransitionGraphManager(session)
        if candidate_key:
            transitions = manager.get_candidate_experience(candidate_key)
        else:
            transitions = manager.get_transitions(page_class or "")
    rows = [_transition_to_response(t) for t in transitions]
    return TransitionGraphResponse(
        page_class=page_class,
        candidate_key=candidate_key,
        transitions=rows,
        total=len(rows),
    )


@app.get("/api/v1/state-templates/{state_template_id}/transitions", response_model=TransitionGraphResponse)
async def get_state_template_transitions(state_template_id: str):
    """Read transition graph edges for a state template's page_class."""
    from src.memory.memory_service import MemoryService
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    memory = MemoryService()
    with Session() as session:
        st = memory.page_model_store.get_state_template_by_id(session, state_template_id)
        if st is None:
            raise HTTPException(status_code=404, detail=f"State template {state_template_id} not found")
        transitions = TransitionGraphManager(session).get_transitions(st.page_class)
    rows = [_transition_to_response(t) for t in transitions]
    return TransitionGraphResponse(
        page_class=st.page_class,
        transitions=rows,
        total=len(rows),
    )


@app.get("/api/v1/control-transitions", response_model=ControlTransitionGraphResponse)
async def list_control_transitions(
    page_class: str | None = None,
    candidate_key: str | None = None,
):
    """Read control→state transition associations."""
    if not page_class and not candidate_key:
        raise HTTPException(status_code=400, detail="page_class or candidate_key is required")
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    with Session() as session:
        manager = TransitionGraphManager(session)
        if candidate_key:
            transitions = manager.get_control_transitions(candidate_key)
        else:
            transitions = manager.get_control_transitions_from_state(page_class or "")
    rows = [_control_transition_to_response(item) for item in transitions]
    return ControlTransitionGraphResponse(
        page_class=page_class,
        candidate_key=candidate_key,
        transitions=rows,
        total=len(rows),
    )


@app.post("/api/v1/control-transitions", response_model=ControlTransitionResponse)
async def record_control_transition(request: ControlTransitionRecordRequest):
    """Record a confirmed control→state transition association."""
    from src.memory.transition_graph import TransitionGraphManager
    from src.storage.db import Session

    with Session() as session:
        manager = TransitionGraphManager(session)
        manager.record_control_transition(
            candidate_key=request.candidate_key,
            stable_key_id=request.stable_key_id,
            from_class=request.from_page_class,
            to_class=request.to_page_class,
            action_type=request.action_type,
            canvas_id_before=request.canvas_id_before,
            canvas_id_after=request.canvas_id_after,
            success=request.success,
            metadata=request.metadata,
        )
        transitions = manager.get_control_transitions(request.candidate_key)
    for item in transitions:
        if (
            item.from_page_class == request.from_page_class
            and item.to_page_class == request.to_page_class
            and item.action_type == request.action_type
        ):
            return _control_transition_to_response(item)
    raise HTTPException(status_code=500, detail="Control transition was not persisted")


@app.get("/api/v1/state-templates/{state_template_id}/model-templates")
async def get_state_template_model_templates(state_template_id: str):
    """Return persisted AppShell/Region/Element template snapshots."""
    from src.memory.model_template_store import ModelTemplateStore
    from src.storage.db import Session
    from src.storage.schema import StateTemplateRecord

    store = ModelTemplateStore()
    with Session() as session:
        st = session.get(StateTemplateRecord, state_template_id)
        if st is None:
            raise HTTPException(status_code=404, detail=f"State template {state_template_id} not found")
        return {
            "state_template_id": state_template_id,
            "app_shell": store.get_app_shell(session, state_template_id) or {},
            "region_templates": store.list_region_templates(session, state_template_id),
            "element_templates": store.list_element_templates(session, state_template_id),
        }


@app.get("/api/v1/candidate-overrides", response_model=list[CandidateOverrideResponse])
async def list_candidate_overrides(
    state_template_id: str | None = None,
    canvas_id: str | None = None,
    page_model_id: str | None = None,
):
    """列出候选修正覆盖层。

    返回的是人工/Agent/VLM 修正，不修改原始 observe 结果。前端按 scope_key
    合并为有效候选。
    """
    from src.memory.candidate_override_store import CandidateOverrideStore
    from src.storage.db import Session

    store = CandidateOverrideStore()
    with Session() as session:
        rows = store.list(
            session,
            state_template_id=state_template_id,
            canvas_id=canvas_id,
            page_model_id=page_model_id,
        )
    return [_override_to_response(row) for row in rows]


@app.put("/api/v1/candidate-overrides", response_model=CandidateOverrideResponse)
async def upsert_candidate_override(request: CandidateOverrideUpsertRequest):
    """创建或更新候选修正覆盖层。"""
    from src.memory.candidate_override_store import CandidateOverrideStore
    from src.memory.evidence_store import EvidenceStore
    from src.storage.db import Session
    import json as _json
    import logging as _logging

    store = CandidateOverrideStore()
    with Session() as session:
        row = store.upsert(
            session,
            scope_key=request.scope_key,
            app_id=request.app_id,
            page_model_id=request.page_model_id,
            state_template_id=request.state_template_id,
            canvas_id=request.canvas_id,
            element_id=request.element_id,
            stable_key_id=request.stable_key_id,
            label=request.label,
            semantic_role=request.semantic_role,
            visual_type=request.visual_type,
            region_id=request.region_id,
            kind=request.kind,
            relative_bounds=request.relative_bounds,
            absolute_bounds=request.absolute_bounds,
            source=request.source,
            status=request.status,
        )
        if (request.source or "manual") in {"manual", "agent"} and request.canvas_id and request.element_id:
            try:
                EvidenceStore().create_evidence(
                    session,
                    provider=request.source or "manual",
                    evidence_scope="candidate",
                    evidence_event="override",
                    canvas_id=request.canvas_id,
                    element_id=request.element_id,
                    stable_key_id=request.stable_key_id,
                    page_model_id=request.page_model_id,
                    state_template_id=request.state_template_id,
                    region_id=request.region_id,
                    semantic_role=request.semantic_role,
                    control_type=request.visual_type,
                    bounds_json=_json.dumps(request.absolute_bounds) if request.absolute_bounds else None,
                    relative_bounds_json=_json.dumps(request.relative_bounds) if request.relative_bounds else None,
                    raw_confidence=1.0,
                    semantic_score=1.0 if request.semantic_role else None,
                    visual_score=1.0 if request.visual_type else None,
                    metadata=_json.dumps({
                        "scope_key": request.scope_key,
                        "kind": request.kind,
                        "label": request.label,
                        "status": request.status,
                    }, ensure_ascii=False),
                )
            except Exception as exc:
                _logging.getLogger(__name__).warning("Candidate override evidence write failed: %s", exc)
        session.commit()
    return _override_to_response(row)


@app.delete("/api/v1/candidate-overrides/{scope_key}")
async def delete_candidate_override(scope_key: str):
    """删除一个候选修正覆盖层。"""
    from src.memory.candidate_override_store import CandidateOverrideStore
    from src.storage.db import Session

    store = CandidateOverrideStore()
    with Session() as session:
        deleted = store.delete(session, scope_key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Candidate override {scope_key} not found")
    return {"deleted": True, "scope_key": scope_key}


@app.get("/api/v1/state-templates/{state_template_id}/detail", response_model=CanvasDetail)
async def get_state_template_detail(state_template_id: str):
    """获取 StateTemplate 的最新快照画布详情（含元素和区域）。

    从 canvas_snapshots 表取最新 canvas_id，再从 CanvasCache 加载。
    如果画布不在缓存中（已过期），返回 404。
    """
    from src.memory.memory_service import MemoryService
    from src.storage.db import Session

    memory = MemoryService()
    cache = get_canvas_cache()

    with Session() as session:
        snapshots = memory.page_model_store.get_canvas_snapshots(
            session, state_template_id=state_template_id,
        )

    if not snapshots:
        raise HTTPException(status_code=404, detail=f"No snapshots for state template {state_template_id}")

    # Try snapshots from newest to oldest
    snapshots.sort(key=lambda s: s.captured_at, reverse=True)
    for snap in snapshots:
        canvas = cache.get(snap.canvas_id)
        if canvas is not None:
            canvas = _effective_canvas(canvas)
            # Reuse the same logic as get_canvas_detail
            surface_val = canvas.surface_type.value if hasattr(canvas.surface_type, "value") else str(canvas.surface_type)
            regions = []
            for r in canvas.regions:
                region_elem_count = len(r.element_ids) if r.element_ids else 0
                subtype_val = r.subtype.value if hasattr(r.subtype, "value") else str(r.subtype)
                regions.append(RegionResponse(
                    region_id=r.region_id,
                    role=r.role,
                    subtype=subtype_val,
                    bounds=list(r.bounds) if r.bounds else None,
                    element_count=region_elem_count,
                ))

            pt = canvas.provider_trace
            provider_trace = ProviderTraceResponse(
                uia_used=pt.uia_used,
                ocr_used=pt.ocr_used,
                vision_used=pt.vision_used,
                vlm_used=pt.vlm_used,
                dom_used=pt.dom_used,
                provider_details=pt.provider_details if pt.provider_details else {},
            ) if pt else None

            win_w, win_h = 0, 0
            if canvas.window and canvas.window.rect_client:
                rc = canvas.window.rect_client
                win_w = rc[2] - rc[0]
                win_h = rc[3] - rc[1]

            has_screenshot = cache.get_screenshot(snap.canvas_id) is not None
            screenshot = cache.get_screenshot(snap.canvas_id)
            screenshot_w, screenshot_h = screenshot.size if screenshot else (0, 0)

            geo_regions = _geometric_regions_to_response(
                canvas.artifacts.get("geometric_regions") if canvas.artifacts else None
            )
            fusion_diag = canvas.artifacts.get("fusion_diagnostics") if canvas.artifacts else {}
            _refresh_provider_status_from_trace(canvas)

            return CanvasDetail(
                canvas_id=canvas.canvas_id,
                app_id=canvas.app.app_id,
                window_title=canvas.window.title if canvas.window else "",
                page_class=canvas.page_class,
                surface_type=surface_val,
                elements=[_candidate_to_response(el) for el in canvas.elements],
                regions=regions,
                geometric_regions=geo_regions,
                fusion_diagnostics=fusion_diag if isinstance(fusion_diag, dict) else {},
                window_width=win_w,
                window_height=win_h,
                captured_at=canvas.captured_at.isoformat() if canvas.captured_at else "",
                providers_used=canvas.providers_used,
                providers_failed=canvas.providers_failed,
                stable=canvas.stable,
                loading=canvas.loading,
                partial=canvas.partial,
                provider_trace=provider_trace,
                has_screenshot=has_screenshot,
                screenshot_width=screenshot_w,
                screenshot_height=screenshot_h,
                page_model_id=getattr(canvas, "page_model_id", None),
                state_template_id=getattr(canvas, "state_template_id", None),
            )

    raise HTTPException(status_code=404, detail=f"Canvas snapshots for state template {state_template_id} are expired")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
