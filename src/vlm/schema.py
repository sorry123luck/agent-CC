"""VLM Semantic Modeler JSON Schema — 数据类 + 校验。

bounds 坐标体系：窗口局部坐标 [left, top, right, bottom]，
原点=窗口左上角，单位=原始截图像素。
与 InteractionCanvas 内部坐标系一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CURRENT_SCHEMA_VERSION = "1.0"
SCHEMA_VERSION_V2 = "2.0"

# 动态内容区域 ID（fixed_controls 在这些区域内出现时应 warning）
_DYNAMIC_CONTENT_ZONE_IDS: frozenset[str] = frozenset({
    "message_area",
    "content_stream",
    "document_body",
    "chat_message_area",
    "chat_messages",
    "article_body",
    "code_block",
    "list_dynamic_content",
})

# 允许包含固定控件的区域（toolbar / input_area / table_action_column 等）
_FUNCTIONAL_CONTROL_ZONE_IDS: frozenset[str] = frozenset({
    "input_area",
    "toolbar",
    "table_action_column",
    "top_bar",
    "status_bar",
    "navigation",
    "menu_bar",
    "action_bar",
})


class ValidationError(Exception):
    """JSON Schema 校验失败。"""


# ---------------------------------------------------------------------------
# 子结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppIdentity:
    app_name: str = ""
    app_id: str | None = None
    surface_type: str = "native_uia"
    display_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"app_name": self.app_name, "app_id": self.app_id, "surface_type": self.surface_type, "display_name": self.display_name}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AppIdentity:
        return cls(app_name=d.get("app_name", ""), app_id=d.get("app_id"), surface_type=d.get("surface_type", "native_uia"), display_name=d.get("display_name", ""))


@dataclass(frozen=True)
class PageState:
    page_class: str = ""
    state_label: str = ""
    state_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"page_class": self.page_class, "state_label": self.state_label, "state_flags": list(self.state_flags)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PageState:
        return cls(page_class=d.get("page_class", ""), state_label=d.get("state_label", ""), state_flags=list(d.get("state_flags", [])))


@dataclass(frozen=True)
class Region:
    region_id: str = ""
    role: str = ""
    bounds: list[int] = field(default_factory=list)
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "role": self.role, "bounds": list(self.bounds), "purpose": self.purpose}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Region:
        return cls(region_id=d.get("region_id", ""), role=d.get("role", ""), bounds=list(d.get("bounds", [])), purpose=d.get("purpose", ""))


@dataclass(frozen=True)
class Control:
    control_id: str = ""
    region_id: str = ""
    control_type: str = ""
    text: str = ""
    bounds: list[int] = field(default_factory=list)
    semantic_role: str = ""
    visual_type: str = ""
    interactable: bool = True
    confidence: float = 0.0
    source_candidate_ids: list[str] = field(default_factory=list)
    matches_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "region_id": self.region_id,
            "control_type": self.control_type,
            "text": self.text,
            "bounds": list(self.bounds),
            "semantic_role": self.semantic_role,
            "visual_type": self.visual_type,
            "interactable": self.interactable,
            "confidence": self.confidence,
            "source_candidate_ids": list(self.source_candidate_ids),
            "matches_candidate_id": self.matches_candidate_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Control:
        return cls(
            control_id=d.get("control_id", ""),
            region_id=d.get("region_id", ""),
            control_type=d.get("control_type", ""),
            text=d.get("text", ""),
            bounds=list(d.get("bounds", [])),
            semantic_role=d.get("semantic_role", ""),
            visual_type=d.get("visual_type", ""),
            interactable=d.get("interactable", True),
            confidence=float(d.get("confidence", 0.0)),
            source_candidate_ids=list(d.get("source_candidate_ids", [])),
            matches_candidate_id=d.get("matches_candidate_id"),
        )


@dataclass(frozen=True)
class DynamicZone:
    zone_id: str = ""
    region_id: str = ""
    content_type: str = ""
    bounds: list[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"zone_id": self.zone_id, "region_id": self.region_id, "content_type": self.content_type, "bounds": list(self.bounds), "note": self.note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DynamicZone:
        return cls(zone_id=d.get("zone_id", ""), region_id=d.get("region_id", ""), content_type=d.get("content_type", ""), bounds=list(d.get("bounds", [])), note=d.get("note", ""))


@dataclass(frozen=True)
class CandidateCorrection:
    candidate_id: str = ""
    corrected_type: str | None = None
    corrected_role: str | None = None
    corrected_confidence: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "corrected_type": self.corrected_type,
            "corrected_role": self.corrected_role,
            "corrected_confidence": self.corrected_confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateCorrection:
        return cls(
            candidate_id=d.get("candidate_id", ""),
            corrected_type=d.get("corrected_type"),
            corrected_role=d.get("corrected_role"),
            corrected_confidence=d.get("corrected_confidence"),
            reason=d.get("reason", ""),
        )


@dataclass(frozen=True)
class Transition:
    from_state: str = ""
    trigger: str = ""
    to_state: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"from_state": self.from_state, "trigger": self.trigger, "to_state": self.to_state, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transition:
        return cls(from_state=d.get("from_state", ""), trigger=d.get("trigger", ""), to_state=d.get("to_state", ""), confidence=float(d.get("confidence", 0.0)))


# ---------------------------------------------------------------------------
# 顶层模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageSemanticModel:
    schema_version: str = CURRENT_SCHEMA_VERSION
    app_identity: AppIdentity = field(default_factory=AppIdentity)
    page_state: PageState = field(default_factory=PageState)
    image_size: list[int] = field(default_factory=list)  # [width, height] in screenshot pixels
    regions: list[Region] = field(default_factory=list)
    fixed_controls: list[Control] = field(default_factory=list)
    dynamic_zones: list[DynamicZone] = field(default_factory=list)
    candidate_corrections: list[CandidateCorrection] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    shared_regions: list[dict[str, Any]] | None = None
    global_controls: list[dict[str, Any]] | None = None
    task_mode: str = ""
    coordinate_space: str = ""
    visible_items: list[dict[str, Any]] = field(default_factory=list)
    missing_suggestions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "app_identity": self.app_identity.to_dict(),
            "page_state": self.page_state.to_dict(),
            "image_size": list(self.image_size),
            "regions": [r.to_dict() for r in self.regions],
            "fixed_controls": [c.to_dict() for c in self.fixed_controls],
            "dynamic_zones": [z.to_dict() for z in self.dynamic_zones],
            "candidate_corrections": [c.to_dict() for c in self.candidate_corrections],
            "transitions": [t.to_dict() for t in self.transitions],
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "shared_regions": self.shared_regions,
            "global_controls": self.global_controls,
            "task_mode": self.task_mode,
            "coordinate_space": self.coordinate_space,
            "visible_items": list(self.visible_items),
            "missing_suggestions": list(self.missing_suggestions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PageSemanticModel:
        return cls(
            schema_version=d.get("schema_version", CURRENT_SCHEMA_VERSION),
            app_identity=AppIdentity.from_dict(d.get("app_identity", {})),
            page_state=PageState.from_dict(d.get("page_state", {})),
            image_size=list(d.get("image_size", [])),
            regions=[Region.from_dict(r) for r in d.get("regions", [])],
            fixed_controls=[Control.from_dict(c) for c in d.get("fixed_controls", [])],
            dynamic_zones=[DynamicZone.from_dict(z) for z in d.get("dynamic_zones", [])],
            candidate_corrections=[CandidateCorrection.from_dict(c) for c in d.get("candidate_corrections", [])],
            transitions=[Transition.from_dict(t) for t in d.get("transitions", [])],
            confidence=float(d.get("confidence", 0.0)),
            needs_review=d.get("needs_review", True),
            shared_regions=d.get("shared_regions"),
            global_controls=d.get("global_controls"),
            task_mode=d.get("task_mode", ""),
            coordinate_space=d.get("coordinate_space", ""),
            visible_items=list(d.get("visible_items", []) or []),
            missing_suggestions=list(d.get("missing_suggestions", []) or []),
            metadata=dict(d.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class VLMSemanticModelerResult:
    """Modeler 返回结果封装。"""
    status: str = "failed"
    model: PageSemanticModel | None = None
    error: str | None = None
    error_code: str | None = None
    from_cache: bool = False
    provider_name: str = ""
    model_name: str = ""
    token_input: int = 0
    token_output: int = 0
    latency_ms: int = 0
    raw_response_redacted: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model.to_dict() if self.model else None,
            "error": self.error,
            "error_code": self.error_code,
            "from_cache": self.from_cache,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "token_input": self.token_input,
            "token_output": self.token_output,
            "latency_ms": self.latency_ms,
            "raw_response_redacted": self.raw_response_redacted,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VLMSemanticModelerResult:
        model = PageSemanticModel.from_dict(d["model"]) if d.get("model") else None
        return cls(
            status=d.get("status", "failed"),
            model=model,
            error=d.get("error"),
            error_code=d.get("error_code"),
            from_cache=d.get("from_cache", False),
            provider_name=d.get("provider_name", ""),
            model_name=d.get("model_name", ""),
            token_input=d.get("token_input", 0),
            token_output=d.get("token_output", 0),
            latency_ms=d.get("latency_ms", 0),
            raw_response_redacted=d.get("raw_response_redacted"),
            warnings=list(d.get("warnings", [])),
        )


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def _validate_bounds(bounds: list[int], label: str, image_size: tuple[int, int] | None = None) -> str | None:
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValidationError(f"{label}: bounds must be [left, top, right, bottom], got {bounds}")
    left, top, right, bottom = bounds
    if not all(isinstance(v, int) for v in bounds):
        raise ValidationError(f"{label}: bounds values must be integers, got {bounds}")
    if left >= right or top >= bottom:
        raise ValidationError(f"{label}: must satisfy left < right and top < bottom, got {bounds}")
    if any(v < 0 for v in bounds):
        raise ValidationError(f"{label}: bounds must be non-negative, got {bounds}")
    if image_size is not None:
        img_w, img_h = image_size
        if right > img_w or bottom > img_h:
            raise ValidationError(
                f"{label}: bounds {[left, top, right, bottom]} exceeded image size {image_size}"
            )
    return None


def _check_dynamic_overlap(
    fixed_controls: list[Control],
    dynamic_zones: list[DynamicZone],
) -> list[str]:
    """检查 fixed_controls 与 dynamic_zones 的重叠，返回 warnings。"""
    warnings: list[str] = []
    for zone in dynamic_zones:
        zone_id = zone.zone_id or zone.region_id
        # 功能性区域（toolbar/input_area 等）允许包含固定控件
        if zone_id in _FUNCTIONAL_CONTROL_ZONE_IDS:
            continue
        # 动态内容区域（chat_messages/article_body 等）内的固定控件应 warning
        if zone_id in _DYNAMIC_CONTENT_ZONE_IDS:
            for ctrl in fixed_controls:
                if ctrl.region_id == zone_id:
                    warnings.append(
                        f"fixed_control '{ctrl.control_id}' is inside dynamic content zone '{zone_id}'"
                    )
    return warnings


def validate_page_semantic_model(
    raw_json: dict[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    known_candidate_ids: set[str] | None = None,
) -> tuple[PageSemanticModel, list[str]]:
    """严格校验 VLM 返回的 JSON，返回 (模型实例, warnings)。

    Args:
        raw_json: VLM 返回的原始 dict。
        image_size: (width, height) — 如果提供，bounds 不得超过图片尺寸。
        known_candidate_ids: 已知候选 ID 集合 — 引用不存在的 ID 产生 warning。

    Returns:
        (PageSemanticModel, warnings)

    Raises:
        ValidationError: 必填字段缺失、类型错误、bounds 不合法等。
    """
    warnings: list[str] = []

    # --- schema_version ---
    sv = raw_json.get("schema_version", "")
    if sv == SCHEMA_VERSION_V2:
        return _validate_page_semantic_model_v2(
            raw_json,
            image_size=image_size,
            known_candidate_ids=known_candidate_ids,
        )
    if sv != CURRENT_SCHEMA_VERSION:
        raise ValidationError(f"schema_version mismatch: expected '{CURRENT_SCHEMA_VERSION}', got '{sv}'")

    # --- app_identity（必须存在且 app_name 非空） ---
    raw_ai = raw_json.get("app_identity")
    if not isinstance(raw_ai, dict):
        raise ValidationError("app_identity must be a dict")
    if not raw_ai.get("app_name") or not isinstance(raw_ai["app_name"], str):
        raise ValidationError("app_identity.app_name must be a non-empty string")

    # --- page_state（必须存在且 page_class/state_label 非空） ---
    raw_ps = raw_json.get("page_state")
    if not isinstance(raw_ps, dict):
        raise ValidationError("page_state must be a dict")
    if not raw_ps.get("page_class") or not isinstance(raw_ps["page_class"], str):
        raise ValidationError("page_state.page_class must be a non-empty string")
    if not raw_ps.get("state_label") or not isinstance(raw_ps["state_label"], str):
        raise ValidationError("page_state.state_label must be a non-empty string")

    # --- regions（必须非空） ---
    raw_regions = raw_json.get("regions")
    if not raw_regions or not isinstance(raw_regions, list):
        raise ValidationError("regions must be a non-empty list")

    # --- 逐项校验 ---
    for i, r in enumerate(raw_regions):
        w = _validate_bounds(r.get("bounds", []), f"regions[{i}]", image_size)
        if w:
            warnings.append(w)

    raw_controls = raw_json.get("fixed_controls", [])
    if not isinstance(raw_controls, list):
        raise ValidationError("fixed_controls must be a list")
    for i, c in enumerate(raw_controls):
        w = _validate_bounds(c.get("bounds", []), f"fixed_controls[{i}]", image_size)
        if w:
            warnings.append(w)
        conf = c.get("confidence", 0.0)
        if not (0.0 <= conf <= 1.0):
            raise ValidationError(f"fixed_controls[{i}].confidence must be in [0.0, 1.0], got {conf}")
        # source_candidate_ids 必须为字符串列表
        scids = c.get("source_candidate_ids", [])
        if not isinstance(scids, list) or not all(isinstance(s, str) for s in scids):
            raise ValidationError(f"fixed_controls[{i}].source_candidate_ids must be list[str]")
        # 引用不存在的候选 ID → warning
        if known_candidate_ids is not None:
            for sid in scids:
                if sid not in known_candidate_ids:
                    warnings.append(f"fixed_controls[{i}].source_candidate_ids references unknown candidate '{sid}'")
            mcid = c.get("matches_candidate_id")
            if mcid is not None and mcid not in known_candidate_ids:
                warnings.append(f"fixed_controls[{i}].matches_candidate_id references unknown candidate '{mcid}'")

    raw_dzones = raw_json.get("dynamic_zones", [])
    if not isinstance(raw_dzones, list):
        raise ValidationError("dynamic_zones must be a list")
    for i, z in enumerate(raw_dzones):
        w = _validate_bounds(z.get("bounds", []), f"dynamic_zones[{i}]", image_size)
        if w:
            warnings.append(w)

    raw_transitions = raw_json.get("transitions", [])
    if not isinstance(raw_transitions, list):
        raise ValidationError("transitions must be a list")
    valid_transitions = []
    for i, t in enumerate(raw_transitions):
        if not t.get("to_state"):
            warnings.append(f"transitions[{i}] missing to_state — skipped")
            continue
        valid_transitions.append(t)
    raw_json["transitions"] = valid_transitions

    raw_corrections = raw_json.get("candidate_corrections", [])
    if not isinstance(raw_corrections, list):
        raise ValidationError("candidate_corrections must be a list")

    confidence = raw_json.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValidationError(f"confidence must be in [0.0, 1.0], got {confidence}")

    # --- 构建模型 ---
    model_json = dict(raw_json)
    if image_size is not None and not model_json.get("image_size"):
        model_json["image_size"] = [int(image_size[0]), int(image_size[1])]
    model = PageSemanticModel.from_dict(model_json)

    # --- dynamic/fixed 重叠检查 ---
    overlap_warnings = _check_dynamic_overlap(model.fixed_controls, model.dynamic_zones)
    warnings.extend(overlap_warnings)

    return model, warnings


def _validate_optional_bounds(
    item: dict[str, Any],
    key: str,
    label: str,
    image_size: tuple[int, int] | None,
) -> list[int]:
    bounds = item.get(key, [])
    if bounds:
        _validate_bounds(bounds, label, image_size)
        return list(bounds)
    return []


def _validate_page_semantic_model_v2(
    raw_json: dict[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    known_candidate_ids: set[str] | None = None,
) -> tuple[PageSemanticModel, list[str]]:
    """Validate schema v2 and adapt it to the existing PageSemanticModel.

    v2 candidate annotations do not own bounds. They become candidate_corrections
    so the bridge can apply semantics to local candidates without treating VLM as
    the primary coordinate detector.
    """
    warnings: list[str] = []
    task_mode = raw_json.get("task_mode", "")
    if task_mode not in {
        "candidate_annotation",
        "region_understanding",
        "full_page_recognition",
        "missing_audit",
        "icon_crop_understanding",
    }:
        raise ValidationError(f"task_mode unsupported for schema v2: {task_mode}")

    coordinate_space = raw_json.get("coordinate_space", "")
    if coordinate_space and coordinate_space != "vlm_image":
        raise ValidationError(f"schema v2 coordinate_space must be 'vlm_image', got '{coordinate_space}'")

    raw_ai = raw_json.get("app_identity") or {}
    if not isinstance(raw_ai, dict):
        raise ValidationError("app_identity must be a dict")
    if not raw_ai.get("app_name"):
        raw_ai = {"app_name": "unknown", **raw_ai}

    raw_ps = raw_json.get("page_state") or {}
    if not isinstance(raw_ps, dict):
        raise ValidationError("page_state must be a dict")
    if not raw_ps.get("page_class"):
        raw_ps = {"page_class": "unknown", **raw_ps}
    if not raw_ps.get("state_label"):
        raw_ps = {"state_label": "unknown", **raw_ps}

    regions: list[dict[str, Any]] = []
    for i, region in enumerate(raw_json.get("layout_regions", []) or []):
        if not isinstance(region, dict):
            raise ValidationError(f"layout_regions[{i}] must be a dict")
        bounds = _validate_optional_bounds(region, "bounds", f"layout_regions[{i}]", image_size)
        if not bounds:
            warnings.append(f"layout_regions[{i}] missing bounds — skipped")
            continue
        regions.append({
            "region_id": str(region.get("region_id") or f"r{i}"),
            "role": str(region.get("role") or "unknown"),
            "bounds": bounds,
            "purpose": str(region.get("purpose") or ""),
        })

    for i, zone in enumerate(raw_json.get("inferred_zones", []) or []):
        if not isinstance(zone, dict):
            raise ValidationError(f"inferred_zones[{i}] must be a dict")
        bounds = _validate_optional_bounds(zone, "bounds", f"inferred_zones[{i}]", image_size)
        if not bounds:
            warnings.append(f"inferred_zones[{i}] missing bounds — skipped")
            continue
        regions.append({
            "region_id": str(zone.get("zone_id") or f"inferred_{i}"),
            "role": str(zone.get("zone_type") or "inferred_zone"),
            "bounds": bounds,
            "purpose": str(zone.get("purpose") or zone.get("note") or ""),
        })

    dynamic_zones: list[dict[str, Any]] = []
    for i, zone in enumerate(raw_json.get("dynamic_zones", []) or []):
        if not isinstance(zone, dict):
            raise ValidationError(f"dynamic_zones[{i}] must be a dict")
        bounds = _validate_optional_bounds(zone, "bounds", f"dynamic_zones[{i}]", image_size)
        if not bounds:
            warnings.append(f"dynamic_zones[{i}] missing bounds — skipped")
            continue
        dynamic_zones.append({
            "zone_id": str(zone.get("zone_id") or f"dz{i}"),
            "region_id": str(zone.get("region_id") or ""),
            "content_type": str(zone.get("content_type") or "unknown"),
            "bounds": bounds,
            "note": str(zone.get("note") or ""),
        })

    visible_items = raw_json.get("visible_items", []) or []
    if not isinstance(visible_items, list):
        raise ValidationError("visible_items must be a list")
    sanitized_visible_items: list[dict[str, Any]] = []
    for i, item in enumerate(visible_items):
        if not isinstance(item, dict):
            warnings.append(f"visible_items[{i}] must be a dict — skipped")
            continue
        matched = item.get("matched_candidate_ids", []) or []
        if matched and (not isinstance(matched, list) or not all(isinstance(x, str) for x in matched)):
            raise ValidationError(f"visible_items[{i}].matched_candidate_ids must be list[str]")
        if known_candidate_ids is not None:
            matched = [x for x in matched if x in known_candidate_ids]
        sanitized = dict(item)
        sanitized["matched_candidate_ids"] = matched
        sanitized_visible_items.append(sanitized)

    missing_suggestions = raw_json.get("missing_suggestions", []) or []
    if not isinstance(missing_suggestions, list):
        raise ValidationError("missing_suggestions must be a list")
    sanitized_missing_suggestions: list[dict[str, Any]] = []
    for i, suggestion in enumerate(missing_suggestions):
        if not isinstance(suggestion, dict):
            warnings.append(f"missing_suggestions[{i}] must be a dict — skipped")
            continue
        sanitized = dict(suggestion)
        rough_area = sanitized.get("rough_area")
        if rough_area:
            sanitized["rough_area"] = _validate_optional_bounds(
                sanitized,
                "rough_area",
                f"missing_suggestions[{i}].rough_area",
                image_size,
            )
        actionability = str(sanitized.get("actionability") or "semantic_only")
        if actionability == "safe":
            warnings.append(f"missing_suggestions[{i}] requested safe actionability; downgraded to review")
            actionability = "review"
        sanitized["actionability"] = actionability
        sanitized_missing_suggestions.append(sanitized)

    corrections: list[dict[str, Any]] = []
    annotations = raw_json.get("candidate_annotations", []) or []
    if not isinstance(annotations, list):
        raise ValidationError("candidate_annotations must be a list")
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise ValidationError(f"candidate_annotations[{i}] must be a dict")
        candidate_id = str(ann.get("candidate_id") or "")
        if not candidate_id:
            warnings.append(f"candidate_annotations[{i}] missing candidate_id — skipped")
            continue
        if known_candidate_ids is not None and candidate_id not in known_candidate_ids:
            warnings.append(f"candidate_annotations[{i}] references unknown candidate '{candidate_id}' — skipped")
            continue
        if "bounds" in ann:
            warnings.append(f"candidate_annotations[{i}] includes bounds; ignored because coordinates come from candidate")
        confidence = ann.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"candidate_annotations[{i}].confidence must be in [0.0, 1.0], got {confidence}")
        actionability = str(ann.get("actionability") or "review")
        if actionability == "safe" and known_candidate_ids is not None and candidate_id not in known_candidate_ids:
            actionability = "review"
        corrections.append({
            "candidate_id": candidate_id,
            "corrected_type": ann.get("control_type"),
            "corrected_role": ann.get("semantic_role"),
            "corrected_confidence": float(confidence),
            "reason": str(ann.get("reason") or f"v2:{actionability}"),
        })

    confidence = raw_json.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValidationError(f"confidence must be in [0.0, 1.0], got {confidence}")

    model_json = {
        "schema_version": SCHEMA_VERSION_V2,
        "task_mode": task_mode,
        "coordinate_space": coordinate_space or "vlm_image",
        "app_identity": raw_ai,
        "page_state": raw_ps,
        "image_size": list(raw_json.get("image_size") or list(image_size or (0, 0))),
        "regions": regions,
        "fixed_controls": [],
        "dynamic_zones": dynamic_zones,
        "candidate_corrections": corrections,
        "transitions": [],
        "confidence": float(confidence),
        "needs_review": bool(raw_json.get("needs_review", False)),
        "visible_items": sanitized_visible_items,
        "missing_suggestions": sanitized_missing_suggestions,
        "metadata": {
            "schema_v2_raw_task_mode": task_mode,
        },
    }
    model = PageSemanticModel.from_dict(model_json)
    return model, warnings
