"""InteractionCanvas <-> JSON serialization for persistence.

Handles all nested dataclasses, enums, datetimes, and tuples.
Produces/consumes JSON-compatible dicts suitable for storage.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.perception.page_compiler_models import (
    Anchor,
    AnchorKind,
    AppInfo,
    Candidate,
    CandidateKind,
    CaptureReason,
    ConfidenceLevel,
    ContentAreaSubtype,
    CoordinateSpace,
    DecisionAction,
    DecisionAlternative,
    DecisionRecord,
    ElementRelation,
    ElementState,
    InteractionCanvas,
    Locator,
    LocatorKind,
    LocatorStats,
    LocatorStatus,
    PageInfo,
    ProviderTrace,
    Region,
    RelationType,
    RiskLevel,
    RiskTag,
    ScrollContext,
    SemanticRole,
    SurfaceEvidence,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def _enum_val(e: Any) -> str:
    if hasattr(e, "value"):
        return e.value
    return str(e)


def _datetime_str(d: datetime) -> str:
    return d.isoformat()


def _tuple_list(t: tuple | None) -> list | None:
    return list(t) if t is not None else None


def _app_to_dict(a: AppInfo) -> dict:
    return {"app_id": a.app_id, "app_profile_id": a.app_profile_id,
            "process_name": a.process_name, "process_id": a.process_id,
            "exe_path": a.exe_path}


def _app_from_dict(d: dict) -> AppInfo:
    return AppInfo(app_id=d.get("app_id"), app_profile_id=d.get("app_profile_id"),
                   process_name=d.get("process_name"), process_id=d.get("process_id"),
                   exe_path=d.get("exe_path"))


def _window_to_dict(w: WindowInfoSnapshot) -> dict:
    return {"hwnd": w.hwnd, "title": w.title, "class_name": w.class_name,
            "rect_screen": _tuple_list(w.rect_screen),
            "rect_client": _tuple_list(w.rect_client),
            "dpi_scale": w.dpi_scale, "is_foreground": w.is_foreground,
            "is_minimized": w.is_minimized, "occluded": w.occluded}


def _window_from_dict(d: dict) -> WindowInfoSnapshot:
    return WindowInfoSnapshot(
        hwnd=d["hwnd"], title=d.get("title", ""), class_name=d.get("class_name"),
        rect_screen=tuple(d["rect_screen"]) if d.get("rect_screen") else None,
        rect_client=tuple(d["rect_client"]) if d.get("rect_client") else None,
        dpi_scale=d.get("dpi_scale", 1.0), is_foreground=d.get("is_foreground", False),
        is_minimized=d.get("is_minimized", False), occluded=d.get("occluded", False))


def _surface_evidence_to_dict(se: SurfaceEvidence) -> dict:
    return {"layer": se.layer, "feature": se.feature, "weight": se.weight,
            "value": se.value, "description": se.description}


def _surface_evidence_from_dict(d: dict) -> SurfaceEvidence:
    return SurfaceEvidence(layer=d["layer"], feature=d["feature"], weight=d.get("weight", 0.0),
                           value=d.get("value", False), description=d.get("description", ""))


def _surface_to_dict(s: SurfaceInfo) -> dict:
    return {"surface_type": _enum_val(s.surface_type), "confidence": s.confidence,
            "evidence": [_surface_evidence_to_dict(e) for e in s.evidence]}


def _surface_from_dict(d: dict) -> SurfaceInfo:
    return SurfaceInfo(
        surface_type=SurfaceType(d.get("surface_type", "unknown")),
        confidence=d.get("confidence", 0.0),
        evidence=[_surface_evidence_from_dict(e) for e in d.get("evidence", [])])


def _page_to_dict(p: PageInfo) -> dict:
    return {"page_class": p.page_class, "class_confidence": p.class_confidence,
            "page_variant": p.page_variant, "fingerprint": p.fingerprint,
            "state_flags": p.state_flags}


def _page_from_dict(d: dict) -> PageInfo:
    return PageInfo(
        page_class=d.get("page_class", "unknown/unknown/unknown"),
        class_confidence=d.get("class_confidence", 0.0),
        page_variant=d.get("page_variant", "default"),
        fingerprint=d.get("fingerprint", {}),
        state_flags=d.get("state_flags", {}))


def _state_to_dict(s: ElementState) -> dict:
    return {"visible": s.visible, "enabled": s.enabled, "focused": s.focused,
            "selected": s.selected, "checked": s.checked, "expanded": s.expanded}


def _state_from_dict(d: dict) -> ElementState:
    return ElementState(visible=d.get("visible", True), enabled=d.get("enabled", True),
                        focused=d.get("focused", False), selected=d.get("selected", False),
                        checked=d.get("checked", False), expanded=d.get("expanded", False))


def _scroll_to_dict(sc: ScrollContext) -> dict:
    return {"scroll_context_id": sc.scroll_context_id, "region_id": sc.region_id,
            "scroll_type": sc.scroll_type, "viewport_height": sc.viewport_height,
            "viewport_width": sc.viewport_width, "scroll_offset": sc.scroll_offset,
            "total_content_height": sc.total_content_height, "is_virtual": sc.is_virtual}


def _scroll_from_dict(d: dict) -> ScrollContext:
    return ScrollContext(
        scroll_context_id=d["scroll_context_id"], region_id=d["region_id"],
        scroll_type=d.get("scroll_type", "vertical"),
        viewport_height=d.get("viewport_height", 0),
        viewport_width=d.get("viewport_width", 0),
        scroll_offset=d.get("scroll_offset", 0),
        total_content_height=d.get("total_content_height"),
        is_virtual=d.get("is_virtual", False))


def _locator_stats_to_dict(ls: LocatorStats) -> dict:
    return {"use_count": ls.use_count, "success_count": ls.success_count,
            "drift_count": ls.drift_count,
            "last_used_at": _datetime_str(ls.last_used_at) if ls.last_used_at else None,
            "last_success_at": _datetime_str(ls.last_success_at) if ls.last_success_at else None}


def _locator_stats_from_dict(d: dict) -> LocatorStats:
    return LocatorStats(
        use_count=d.get("use_count", 0), success_count=d.get("success_count", 0),
        drift_count=d.get("drift_count", 0),
        last_used_at=datetime.fromisoformat(d["last_used_at"]) if d.get("last_used_at") else None,
        last_success_at=datetime.fromisoformat(d["last_success_at"]) if d.get("last_success_at") else None)


def _locator_to_dict(loc: Locator) -> dict:
    return {"locator_id": loc.locator_id, "element_ref": loc.element_ref,
            "kind": _enum_val(loc.kind), "priority": loc.priority,
            "status": _enum_val(loc.status),
            "coordinate_space": _enum_val(loc.coordinate_space),
            "scope_region_id": loc.scope_region_id,
            "selector": loc.selector, "expected": loc.expected,
            "anchor_refs": loc.anchor_refs, "preconditions": loc.preconditions,
            "verification_hints": loc.verification_hints,
            "confidence": loc.confidence, "durability_score": loc.durability_score,
            "cost_score": loc.cost_score,
            "stats": _locator_stats_to_dict(loc.stats),
            "lifecycle": loc.lifecycle, "notes": loc.notes}


def _locator_from_dict(d: dict) -> Locator:
    return Locator(
        locator_id=d["locator_id"], element_ref=d["element_ref"],
        kind=LocatorKind(d.get("kind", "uia")),
        priority=d.get("priority", 0),
        status=LocatorStatus(d.get("status", "candidate")),
        coordinate_space=CoordinateSpace(d.get("coordinate_space", "window")),
        scope_region_id=d.get("scope_region_id"),
        selector=d.get("selector", {}), expected=d.get("expected", {}),
        anchor_refs=d.get("anchor_refs", []),
        preconditions=d.get("preconditions", {}),
        verification_hints=d.get("verification_hints", {}),
        confidence=d.get("confidence", 0.0),
        durability_score=d.get("durability_score", 0.0),
        cost_score=d.get("cost_score", 0.0),
        stats=_locator_stats_from_dict(d.get("stats", {})),
        lifecycle=d.get("lifecycle", "candidate"),
        notes=d.get("notes", ""))


def _anchor_to_dict(a: Anchor) -> dict:
    return {"anchor_id": a.anchor_id, "kind": _enum_val(a.kind),
            "element_refs": a.element_refs, "signature": a.signature,
            "stability_score": a.stability_score, "drift_threshold": a.drift_threshold,
            "region_id": a.region_id, "notes": a.notes}


def _anchor_from_dict(d: dict) -> Anchor:
    return Anchor(
        anchor_id=d["anchor_id"], kind=AnchorKind(d.get("kind", "single")),
        element_refs=d.get("element_refs", []),
        signature=d.get("signature", {}),
        stability_score=d.get("stability_score", 0.0),
        drift_threshold=d.get("drift_threshold", 0.15),
        region_id=d.get("region_id"), notes=d.get("notes", ""))


def _candidate_to_dict(c: Candidate) -> dict:
    return {"element_id": c.element_id, "region_id": c.region_id,
            "semantic_role": _enum_val(c.semantic_role),
            "control_type": c.control_type,
            "bounds": _tuple_list(c.bounds),
            "text": c.text, "name": c.name, "value": c.value,
            "placeholder": c.placeholder,
            "interactable": c.interactable,
            "state": _state_to_dict(c.state),
            "provider_sources": c.provider_sources,
            "locator_ids": c.locator_ids, "anchor_ids": c.anchor_ids,
            "content_group_id": c.content_group_id,
            "tab_group_id": c.tab_group_id,
            "attributes": c.attributes, "confidence": c.confidence,
            "confidence_level": _enum_val(c.confidence_level),
            "risk_tags": c.risk_tags,
            "risk_level": _enum_val(c.risk_level),
            "from_memory": c.from_memory,
            "suggest_confirm": c.suggest_confirm,
            "click_point": _tuple_list(c.click_point),
            "scroll_context": _scroll_to_dict(c.scroll_context) if c.scroll_context else None,
            "visual_type": c.visual_type,
            "semantic_tags": c.semantic_tags,
            "role_label": c.role_label,
            "role_confidence": c.role_confidence,
            "role_source": c.role_source,
            "role_evidence": c.role_evidence,
            "refine_status": c.refine_status,
            "stable_key_id": c.stable_key_id}


def _candidate_from_dict(d: dict) -> Candidate:
    sc_data = d.get("scroll_context")
    return Candidate(
        element_id=d["element_id"], region_id=d.get("region_id"),
        semantic_role=SemanticRole(d.get("semantic_role", "unknown")),
        control_type=d.get("control_type", ""),
        bounds=tuple(d["bounds"]) if d.get("bounds") else None,
        text=d.get("text", ""), name=d.get("name"),
        value=d.get("value"), placeholder=d.get("placeholder"),
        interactable=d.get("interactable", True),
        state=_state_from_dict(d.get("state", {})),
        provider_sources=d.get("provider_sources", []),
        locator_ids=d.get("locator_ids", []),
        anchor_ids=d.get("anchor_ids", []),
        content_group_id=d.get("content_group_id"),
        tab_group_id=d.get("tab_group_id"),
        attributes=d.get("attributes", {}),
        confidence=d.get("confidence", 0.0),
        confidence_level=ConfidenceLevel(d.get("confidence_level", "medium")),
        risk_tags=d.get("risk_tags", []),
        risk_level=RiskLevel(d.get("risk_level", "L0")),
        from_memory=d.get("from_memory", False),
        suggest_confirm=d.get("suggest_confirm", False),
        click_point=tuple(d["click_point"]) if d.get("click_point") else None,
        scroll_context=_scroll_from_dict(sc_data) if sc_data else None,
        visual_type=d.get("visual_type", ""),
        semantic_tags=d.get("semantic_tags", []),
        role_label=d.get("role_label"),
        role_confidence=d.get("role_confidence", 0.0),
        role_source=d.get("role_source", ""),
        role_evidence=d.get("role_evidence", []),
        refine_status=d.get("refine_status", "unreviewed"),
        stable_key_id=d.get("stable_key_id"))


def _region_to_dict(r: Region) -> dict:
    return {"region_id": r.region_id, "role": r.role,
            "subtype": _enum_val(r.subtype),
            "parent_region_id": r.parent_region_id,
            "bounds": _tuple_list(r.bounds),
            "scroll_context_id": r.scroll_context_id,
            "element_ids": r.element_ids, "anchor_ids": r.anchor_ids,
            "child_region_ids": r.child_region_ids,
            "attributes": r.attributes}


def _region_from_dict(d: dict) -> Region:
    return Region(
        region_id=d["region_id"], role=d.get("role", "unknown"),
        subtype=ContentAreaSubtype(d.get("subtype", "unknown")),
        parent_region_id=d.get("parent_region_id"),
        bounds=tuple(d["bounds"]) if d.get("bounds") else None,
        scroll_context_id=d.get("scroll_context_id"),
        element_ids=d.get("element_ids", []),
        anchor_ids=d.get("anchor_ids", []),
        child_region_ids=d.get("child_region_ids", []),
        attributes=d.get("attributes", {}))


def _relation_to_dict(r: ElementRelation) -> dict:
    return {"relation_id": r.relation_id, "from_id": r.from_id, "to_id": r.to_id,
            "type": _enum_val(r.type), "strength": r.strength,
            "offset": _tuple_list(r.offset)}


def _relation_from_dict(d: dict) -> ElementRelation:
    return ElementRelation(
        relation_id=d["relation_id"], from_id=d["from_id"], to_id=d["to_id"],
        type=RelationType(d.get("type", "right_of")),
        strength=d.get("strength", 1.0),
        offset=tuple(d["offset"]) if d.get("offset") else None)


def _provider_to_dict(pt: ProviderTrace) -> dict:
    return {"uia_used": pt.uia_used, "ocr_used": pt.ocr_used,
            "vision_used": pt.vision_used, "vlm_used": pt.vlm_used,
            "dom_used": pt.dom_used, "opencv_used": pt.opencv_used,
            "template_used": pt.template_used,
            "provider_details": pt.provider_details}


def _provider_from_dict(d: dict) -> ProviderTrace:
    return ProviderTrace(
        uia_used=d.get("uia_used", False), ocr_used=d.get("ocr_used", False),
        vision_used=d.get("vision_used", False), vlm_used=d.get("vlm_used", False),
        dom_used=d.get("dom_used", False), opencv_used=d.get("opencv_used", False),
        template_used=d.get("template_used", False),
        provider_details=d.get("provider_details", {}))


def _safe_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Filter artifacts to JSON-serializable values only."""
    result: dict[str, Any] = {}
    for k, v in artifacts.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = repr(v)
    return result


def canvas_to_dict(canvas: InteractionCanvas) -> dict[str, Any]:
    """Serialize InteractionCanvas to a JSON-compatible dict."""
    return {
        "schema_version": canvas.schema_version,
        "canvas_id": canvas.canvas_id,
        "captured_at": _datetime_str(canvas.captured_at),
        "capture_reason": _enum_val(canvas.capture_reason),
        "app": _app_to_dict(canvas.app),
        "session_id": canvas.session_id,
        "window": _window_to_dict(canvas.window) if canvas.window else None,
        "surface": _surface_to_dict(canvas.surface),
        "page": _page_to_dict(canvas.page),
        "scroll_contexts": [_scroll_to_dict(sc) for sc in canvas.scroll_contexts],
        "regions": [_region_to_dict(r) for r in canvas.regions],
        "elements": [_candidate_to_dict(e) for e in canvas.elements],
        "locators": [_locator_to_dict(loc) for loc in canvas.locators],
        "anchors": [_anchor_to_dict(a) for a in canvas.anchors],
        "relations": [_relation_to_dict(r) for r in canvas.relations],
        "provider_trace": _provider_to_dict(canvas.provider_trace),
        "artifacts": _safe_artifacts(canvas.artifacts),
        "monitor_id": canvas.monitor_id,
        "stable": canvas.stable,
        "loading": canvas.loading,
        "animation_detected": canvas.animation_detected,
        "partial": canvas.partial,
        "providers_used": canvas.providers_used,
        "providers_failed": canvas.providers_failed,
        "canvas_schema_version": canvas.canvas_schema_version,
        "page_model_id": canvas.page_model_id,
        "state_template_id": canvas.state_template_id,
    }


def canvas_from_dict(data: dict[str, Any]) -> InteractionCanvas:
    """Deserialize InteractionCanvas from a JSON-compatible dict."""
    return InteractionCanvas(
        schema_version=data.get("schema_version", "2.0"),
        canvas_id=data.get("canvas_id", ""),
        captured_at=datetime.fromisoformat(data["captured_at"]) if data.get("captured_at") else datetime.now(),
        capture_reason=CaptureReason(data.get("capture_reason", "scheduled")),
        app=_app_from_dict(data.get("app", {})),
        session_id=data.get("session_id"),
        window=_window_from_dict(data["window"]) if data.get("window") else None,
        surface=_surface_from_dict(data.get("surface", {})),
        page=_page_from_dict(data.get("page", {})),
        scroll_contexts=[_scroll_from_dict(sc) for sc in data.get("scroll_contexts", [])],
        regions=[_region_from_dict(r) for r in data.get("regions", [])],
        elements=[_candidate_from_dict(e) for e in data.get("elements", [])],
        locators=[_locator_from_dict(loc) for loc in data.get("locators", [])],
        anchors=[_anchor_from_dict(a) for a in data.get("anchors", [])],
        relations=[_relation_from_dict(r) for r in data.get("relations", [])],
        provider_trace=_provider_from_dict(data.get("provider_trace", {})),
        artifacts=data.get("artifacts", {}),
        monitor_id=data.get("monitor_id", 0),
        stable=data.get("stable", True),
        loading=data.get("loading", False),
        animation_detected=data.get("animation_detected", False),
        partial=data.get("partial", False),
        providers_used=data.get("providers_used", []),
        providers_failed=data.get("providers_failed", []),
        canvas_schema_version=data.get("canvas_schema_version", "1.0"),
        page_model_id=data.get("page_model_id"),
        state_template_id=data.get("state_template_id"),
    )


def dumps(canvas: InteractionCanvas) -> str:
    """Serialize InteractionCanvas to JSON string."""
    return json.dumps(canvas_to_dict(canvas), ensure_ascii=False)


def loads(json_str: str) -> InteractionCanvas:
    """Deserialize InteractionCanvas from JSON string."""
    return canvas_from_dict(json.loads(json_str))
