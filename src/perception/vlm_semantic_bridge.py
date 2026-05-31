"""VLM Semantic Bridge — 将 PageSemanticModel 合并到 InteractionCanvas。

职责：
1. VLM app_identity → canvas.app
2. VLM page_state → canvas.page + canvas.attributes
3. VLM regions → 合并到 canvas.regions
4. VLM fixed_controls → 转换为 Candidate 加入 canvas.elements
5. VLM candidate_corrections → 写入 attributes.vlm_correction
6. VLM dynamic_zones → 标记 region.attributes.dynamic_zone
7. persist_corrections → 写入 CandidateOverrideStore（持久化）
8. merge_shared_regions → 跨页面区域归并
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from src.vlm.schema import (
    AppIdentity,
    CandidateCorrection,
    Control,
    DynamicZone,
    PageSemanticModel,
    PageState,
    Region as VLMRegion,
)

logger = logging.getLogger(__name__)


def _get_vlm_scale(
    canvas: Any,
    semantic_model: Any | None = None,
) -> tuple[float, float] | None:
    """从 canvas.artifacts 读取 VLM 图片综合缩放因子。

    两层矫正：
    1. resize 缩放：sent_image_size → original_size（图片被压缩时）
    2. VLM 偏差矫正：VLM 自报 image_size → sent_image_size（VLM 猜错尺寸时）

    返回 (scale_x, scale_y) 或 None（不需要缩放时）。
    """
    artifacts = getattr(canvas, "artifacts", None) or {}
    scale_info = artifacts.get("vlm_image_scale", None)
    if not scale_info:
        # Fallback: check attributes (legacy locations)
        attrs = getattr(canvas, "attributes", None) or {}
        scale_info = attrs.get("vlm_image_scale", None)
    if not scale_info:
        return None
    orig = scale_info.get("original_size", {})
    sent = scale_info.get("sent_image_size", {})
    orig_w = orig.get("w", 0)
    orig_h = orig.get("h", 0)
    sent_w = sent.get("w", 0)
    sent_h = sent.get("h", 0)
    if not orig_w or not orig_h or not sent_w or not sent_h:
        return None

    # Layer 1: resize correction (sent → orig)
    resize_sx = orig_w / sent_w if sent_w > 0 else 1.0
    resize_sy = orig_h / sent_h if sent_h > 0 else 1.0

    # Layer 2: VLM image_size correction (VLM space → sent space)
    vlm_sx, vlm_sy = 1.0, 1.0
    if semantic_model is not None:
        im = getattr(semantic_model, "image_size", None)
        if im and len(im) >= 2 and im[0] > 0 and im[1] > 0:
            vlm_w, vlm_h = im[0], im[1]
            if vlm_w > 0 and vlm_h > 0 and (vlm_w != sent_w or vlm_h != sent_h):
                vlm_sx = sent_w / vlm_w
                vlm_sy = sent_h / vlm_h
                logger.debug(
                    "VLM image_size mismatch: VLM reports %dx%d, sent %dx%d — "
                    "applying correction x%.4f, x%.4f",
                    vlm_w, vlm_h, sent_w, sent_h, vlm_sx, vlm_sy,
                )

    combined_sx = resize_sx * vlm_sx
    combined_sy = resize_sy * vlm_sy

    if combined_sx == 1.0 and combined_sy == 1.0:
        return None
    return (combined_sx, combined_sy)


def _normalize_vlm_bounds(
    bounds: tuple[int, int, int, int],
    scale: tuple[float, float],
) -> tuple[int, int, int, int]:
    """将 VLM bounds 从 sent-image 坐标归一化到原始窗口坐标。"""
    sx, sy = scale
    return (
        int(round(bounds[0] * sx)),
        int(round(bounds[1] * sy)),
        int(round(bounds[2] * sx)),
        int(round(bounds[3] * sy)),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算两个 bounds (left, top, right, bottom) 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _vlm_bounds_to_tuple(bounds: list[int]) -> tuple[int, int, int, int] | None:
    if len(bounds) < 4:
        return None
    return (bounds[0], bounds[1], bounds[2], bounds[3])


_VLM_ROLE_TO_SEMANTIC_ROLE: dict[str, str] = {
    "send_button": "send_button",
    "search_input": "search_input",
    "message_input": "message_input",
    "text_input": "text_input",
    "button": "button",
    "icon_button": "icon_button",
    "tab": "menu_item",
    "menu_item": "menu_item",
    "toggle": "toggle_button",
    "cancel_button": "cancel_button",
    "submit_button": "submit_button",
    "input": "text_input",
}


def _validate_and_clamp_vlm_bounds(
    bounds: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    *,
    elem_id: str = "",
    tolerance: int = 20,
) -> tuple[tuple[int, int, int, int] | None, bool, str]:
    """Strictly validate VLM bounds against image dimensions.

    VLM coordinates should already be normalized/validated by the modeler.
    Do not clamp here; clamping hides coordinate-space bugs and can turn an
    invalid VLM guess into a clickable candidate.
    """
    l, t, r, b = bounds

    # Fundamental validity: must have positive area
    if l >= r or t >= b:
        return None, False, f"invalid geometry: l={l} r={r} t={t} b={b}"
    if r <= 0 or b <= 0 or l >= image_w or t >= image_h:
        return None, False, f"completely outside image {image_w}x{image_h}: [{l},{t},{r},{b}]"

    if l < 0 or t < 0 or r > image_w or b > image_h:
        return None, False, f"outside image {image_w}x{image_h}: [{l},{t},{r},{b}]"
    return bounds, True, "ok"


class VLMSemanticBridge:
    """将 PageSemanticModel 合并到 InteractionCanvas。返回新 canvas（不可变）。"""

    def apply_to_canvas(
        self,
        canvas: Any,
        semantic_model: PageSemanticModel,
        *,
        db_session_factory: Any = None,
    ) -> Any:
        """应用 VLM 语义到 canvas。返回 deepcopy 后的新 canvas。"""
        new_canvas = deepcopy(canvas)
        scale = _get_vlm_scale(new_canvas, semantic_model)
        # Determine image dimensions for bounds validation
        im_w, im_h = self._get_image_dims(new_canvas, semantic_model)
        new_canvas = self._apply_app_identity(new_canvas, semantic_model.app_identity)
        new_canvas = self._apply_page_state(new_canvas, semantic_model.page_state)
        new_canvas = self._merge_regions(new_canvas, semantic_model.regions, scale=scale, image_w=im_w, image_h=im_h)
        new_canvas = self._add_vlm_candidates(new_canvas, semantic_model.fixed_controls, scale=scale, image_w=im_w, image_h=im_h)
        new_canvas = self._apply_corrections(new_canvas, semantic_model.candidate_corrections)
        new_canvas = self._mark_dynamic_zones(new_canvas, semantic_model.dynamic_zones, scale=scale, image_w=im_w, image_h=im_h)
        new_canvas = self._attach_vlm_semantic_artifacts(new_canvas, semantic_model)
        # Mark shared regions (cross-page stable regions)
        if db_session_factory is not None:
            dynamic_zone_ids = {
                r.attributes.get("vlm_zone_id", "")
                for r in new_canvas.regions
                if (r.attributes or {}).get("dynamic_zone")
            }
            app_id = new_canvas.app.app_id or "unknown"
            self.merge_shared_regions(
                app_id,
                new_canvas.regions,
                dynamic_zone_ids,
                db_session_factory=db_session_factory,
            )
        return new_canvas

    @staticmethod
    def _get_image_dims(canvas: Any, semantic_model: PageSemanticModel) -> tuple[int, int]:
        """Get image dimensions for bounds validation.

        Modeler returns normalized original-screenshot coordinates. Prefer the
        normalized model image_size, then the original screenshot size recorded
        in artifacts. Do not validate normalized bounds against the smaller VLM
        sent-image dimensions.
        """
        im = getattr(semantic_model, "image_size", None)
        if im and len(im) >= 2 and im[0] > 0 and im[1] > 0:
            return int(im[0]), int(im[1])
        scale_info = (canvas.artifacts or {}).get("vlm_image_scale", {})
        orig = scale_info.get("original_size", {})
        if orig.get("w", 0) > 0 and orig.get("h", 0) > 0:
            return orig["w"], orig["h"]
        window = getattr(canvas, "window", None)
        if window and window.rect_client:
            r = window.rect_client
            return r[2] - r[0], r[3] - r[1]
        return 896, 648  # fallback

    def _apply_app_identity(self, canvas: Any, identity: AppIdentity) -> Any:
        """VLM app_identity → canvas.app。仅当当前 app_id 为 unknown 时覆盖。"""
        if identity.app_name and (not canvas.app.app_id or canvas.app.app_id == "unknown"):
            canvas.app.app_id = identity.app_name.lower().replace(" ", "_")
        if identity.app_name:
            canvas.app.attributes = getattr(canvas.app, "attributes", {})
            if not canvas.app.attributes:
                canvas.app.attributes = {}
            canvas.app.attributes["vlm_app_name"] = identity.app_name
        return canvas

    def _apply_page_state(self, canvas: Any, page_state: PageState) -> Any:
        """VLM page_state → canvas.page + canvas.attributes。"""
        if page_state.page_class and page_state.page_class != "unknown":
            canvas.page.page_class = page_state.page_class
        if page_state.state_label and page_state.state_label != "unknown":
            canvas.page.state_label = page_state.state_label
        if page_state.state_flags:
            canvas.attributes = getattr(canvas, "attributes", {})
            if not canvas.attributes:
                canvas.attributes = {}
            canvas.attributes["vlm_state_flags"] = list(page_state.state_flags)
        return canvas

    def _attach_vlm_semantic_artifacts(self, canvas: Any, semantic_model: PageSemanticModel) -> Any:
        """Attach non-actionable VLM audit artifacts without creating safe controls."""
        artifacts = dict(getattr(canvas, "artifacts", None) or {})
        if getattr(semantic_model, "visible_items", None):
            artifacts["vlm_visible_items"] = list(semantic_model.visible_items)
        if getattr(semantic_model, "missing_suggestions", None):
            artifacts["vlm_missing_suggestions"] = list(semantic_model.missing_suggestions)
        if getattr(semantic_model, "task_mode", ""):
            artifacts["vlm_last_task_mode"] = semantic_model.task_mode
        canvas.artifacts = artifacts
        return canvas

    def _merge_regions(
        self,
        canvas: Any,
        vlm_regions: list[VLMRegion],
        *,
        scale: tuple[float, float] | None = None,
        image_w: int = 0,
        image_h: int = 0,
    ) -> Any:
        """合并 VLM 区域到 canvas.regions。IoU > 0.5 → 增强；不重叠 → 新增。"""
        from src.perception.page_compiler_models import Region as CanvasRegion

        existing = canvas.regions
        matched_indices: set[int] = set()
        rejected = 0

        for vlm_r in vlm_regions:
            raw_bounds = _vlm_bounds_to_tuple(vlm_r.bounds)
            if raw_bounds is None:
                continue
            vlm_bounds = _normalize_vlm_bounds(raw_bounds, scale) if scale else raw_bounds

            # Validate bounds
            if image_w > 0 and image_h > 0:
                clamped, keep, reason = _validate_and_clamp_vlm_bounds(
                    vlm_bounds, image_w, image_h, elem_id=vlm_r.region_id,
                )
                if not keep:
                    rejected += 1
                    logger.warning("VLM region %s rejected: %s", vlm_r.region_id, reason)
                    continue
                if reason not in ("ok",):
                    logger.info("VLM region %s %s", vlm_r.region_id, reason)
                vlm_bounds = clamped

            best_idx = -1
            best_iou = 0.0
            for i, er in enumerate(existing):
                if i in matched_indices:
                    continue
                if er.bounds is None:
                    continue
                iou = _iou(vlm_bounds, er.bounds)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > 0.5 and best_idx >= 0:
                matched_indices.add(best_idx)
                old = existing[best_idx]
                attrs = dict(old.attributes) if old.attributes else {}
                attrs["vlm_purpose"] = vlm_r.purpose
                attrs["vlm_region_id"] = vlm_r.region_id
                attrs["vlm_role"] = vlm_r.role
                existing[best_idx] = CanvasRegion(
                    region_id=old.region_id,
                    role=old.role if old.role != "unknown" else vlm_r.role,
                    subtype=old.subtype,
                    parent_region_id=old.parent_region_id,
                    bounds=old.bounds,
                    scroll_context_id=old.scroll_context_id,
                    element_ids=list(old.element_ids),
                    anchor_ids=list(old.anchor_ids),
                    child_region_ids=list(old.child_region_ids),
                    attributes=attrs,
                )
            else:
                new_region = CanvasRegion(
                    region_id=f"vlm_{vlm_r.region_id}",
                    role=vlm_r.role or "unknown",
                    bounds=vlm_bounds,
                    attributes={"vlm_purpose": vlm_r.purpose, "source": "vlm"},
                )
                existing.append(new_region)

        canvas.regions = existing
        if rejected:
            logger.warning("VLM regions: %d accepted, %d rejected (invalid bounds)", len(vlm_regions) - rejected, rejected)
        return canvas

    def _add_vlm_candidates(
        self,
        canvas: Any,
        controls: list[Control],
        *,
        scale: tuple[float, float] | None = None,
        image_w: int = 0,
        image_h: int = 0,
    ) -> Any:
        """VLM fixed_controls → Candidate。按 IoU 匹配现有候选，未匹配的新增。"""
        from src.perception.page_compiler_models import Candidate

        existing = canvas.elements
        matched_element_ids: set[str] = set()
        rejected = 0

        for ctrl in controls:
            raw_bounds = _vlm_bounds_to_tuple(ctrl.bounds)
            if raw_bounds is None:
                continue
            ctrl_bounds = _normalize_vlm_bounds(raw_bounds, scale) if scale else raw_bounds

            # Validate bounds
            if image_w > 0 and image_h > 0:
                clamped, keep, reason = _validate_and_clamp_vlm_bounds(
                    ctrl_bounds, image_w, image_h, elem_id=ctrl.control_id,
                )
                if not keep:
                    rejected += 1
                    logger.warning("VLM control %s rejected: %s", ctrl.control_id, reason)
                    continue
                if reason not in ("ok",):
                    logger.info("VLM control %s %s", ctrl.control_id, reason)
                ctrl_bounds = clamped

            best_idx = -1
            best_iou = 0.0
            for i, elem in enumerate(existing):
                if elem.element_id in matched_element_ids:
                    continue
                if elem.bounds is None:
                    continue
                iou = _iou(ctrl_bounds, elem.bounds)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > 0.5 and best_idx >= 0:
                matched_element_ids.add(existing[best_idx].element_id)
                old = existing[best_idx]
                attrs = dict(old.attributes) if old.attributes else {}
                attrs["vlm_control_id"] = ctrl.control_id
                attrs["vlm_semantic_role"] = ctrl.semantic_role
                attrs["vlm_visual_type"] = ctrl.visual_type
                old.attributes = attrs
                if not old.provider_sources or "vlm" not in old.provider_sources:
                    old.provider_sources.append("vlm")
            else:
                semantic_role_str = _VLM_ROLE_TO_SEMANTIC_ROLE.get(
                    ctrl.semantic_role, "unknown"
                )
                new_elem = Candidate(
                    element_id=f"vlm_{ctrl.control_id}",
                    region_id=f"vlm_{ctrl.region_id}" if ctrl.region_id else None,
                    semantic_role=semantic_role_str,
                    control_type=ctrl.control_type,
                    bounds=ctrl_bounds,
                    text=ctrl.text,
                    interactable=ctrl.interactable,
                    confidence=ctrl.confidence,
                    provider_sources=["vlm"],
                    attributes={
                        "vlm_control_id": ctrl.control_id,
                        "source": "vlm",
                    },
                )
                existing.append(new_elem)

        canvas.elements = existing
        if rejected:
            logger.warning("VLM controls: %d accepted, %d rejected (invalid bounds)", len(controls) - rejected, rejected)
        return canvas

    def _apply_corrections(
        self,
        canvas: Any,
        corrections: list[CandidateCorrection],
    ) -> Any:
        """VLM candidate_corrections → attributes.vlm_correction。"""
        correction_map = {c.candidate_id: c for c in corrections}
        for elem in canvas.elements:
            # 匹配方式：element_id 或 vlm_control_id
            corr = correction_map.get(elem.element_id)
            if corr is None:
                vlm_id = (elem.attributes or {}).get("vlm_control_id")
                if vlm_id:
                    corr = correction_map.get(vlm_id)
            if corr is not None:
                attrs = dict(elem.attributes) if elem.attributes else {}
                attrs["vlm_correction"] = {
                    "corrected_type": corr.corrected_type,
                    "corrected_role": corr.corrected_role,
                    "corrected_confidence": corr.corrected_confidence,
                    "reason": corr.reason,
                }
                elem.attributes = attrs
                if corr.corrected_role:
                    elem.semantic_role = corr.corrected_role
                if corr.corrected_type:
                    if not getattr(elem, "control_type", "") or elem.control_type == "unknown":
                        elem.control_type = corr.corrected_type
                    if hasattr(elem, "visual_type"):
                        elem.visual_type = corr.corrected_type
                if corr.corrected_confidence is not None:
                    elem.confidence = max(float(getattr(elem, "confidence", 0.0) or 0.0), float(corr.corrected_confidence))
        return canvas

    def _mark_dynamic_zones(
        self,
        canvas: Any,
        dynamic_zones: list[DynamicZone],
        *,
        scale: tuple[float, float] | None = None,
        image_w: int = 0,
        image_h: int = 0,
    ) -> Any:
        """标记与 VLM dynamic_zone 重叠的区域为动态内容区。"""
        for zone in dynamic_zones:
            raw_bounds = _vlm_bounds_to_tuple(zone.bounds)
            if raw_bounds is None:
                continue
            zone_bounds = _normalize_vlm_bounds(raw_bounds, scale) if scale else raw_bounds

            # Validate bounds
            if image_w > 0 and image_h > 0:
                clamped, keep, reason = _validate_and_clamp_vlm_bounds(
                    zone_bounds, image_w, image_h, elem_id=zone.zone_id,
                )
                if not keep:
                    logger.warning("VLM dynamic_zone %s rejected: %s", zone.zone_id, reason)
                    continue
                if reason not in ("ok",):
                    logger.info("VLM dynamic_zone %s %s", zone.zone_id, reason)
                zone_bounds = clamped
            for region in canvas.regions:
                if region.bounds is None:
                    continue
                iou = _iou(zone_bounds, region.bounds)
                if iou > 0.3:
                    attrs = dict(region.attributes) if region.attributes else {}
                    attrs["dynamic_zone"] = True
                    attrs["vlm_content_type"] = zone.content_type
                    attrs["vlm_zone_id"] = zone.zone_id
                    region.attributes = attrs
        return canvas

    def persist_corrections(
        self,
        semantic_model: PageSemanticModel,
        *,
        app_id: str,
        page_model_id: str | None = None,
        state_template_id: str | None = None,
        element_to_stable_key: dict[str, str] | None = None,
        db_session_factory: Any = None,
    ) -> int:
        """将 VLM 修正写入 CandidateOverrideStore。返回写入数。

        应在 resolve_candidate_keys 之后调用，此时 element_to_stable_key
        已包含 element_id → stable_key_id 的映射。
        """
        if not semantic_model.candidate_corrections:
            return 0
        if db_session_factory is None:
            return 0
        if element_to_stable_key is None:
            element_to_stable_key = {}

        try:
            from src.memory.candidate_override_store import CandidateOverrideStore
            from src.vlm.modeler import _open_session

            store = CandidateOverrideStore()
            count = 0
            with _open_session(db_session_factory) as session:
                if session is None:
                    return 0
                for corr in semantic_model.candidate_corrections:
                    stable_key_id = element_to_stable_key.get(corr.candidate_id)
                    # Generate scope_key that candidate_override_keys() can match
                    if stable_key_id and page_model_id:
                        scope_key = f"stable:{page_model_id}:{stable_key_id}"
                    elif stable_key_id:
                        scope_key = f"stable:{stable_key_id}"
                    elif state_template_id:
                        scope_key = f"state:{state_template_id}:transient_{corr.candidate_id}"
                    else:
                        scope_key = f"vlm:{app_id}:{corr.candidate_id}"
                    kwargs: dict[str, Any] = {
                        "scope_key": scope_key,
                        "app_id": app_id,
                        "page_model_id": page_model_id,
                        "state_template_id": state_template_id,
                        "element_id": corr.candidate_id,
                        "source": "vlm_semantic",
                        "status": "active",
                    }
                    if stable_key_id:
                        kwargs["stable_key_id"] = stable_key_id
                    if corr.corrected_role:
                        kwargs["semantic_role"] = corr.corrected_role
                    if corr.corrected_type:
                        kwargs["visual_type"] = corr.corrected_type
                    store.upsert(session, **kwargs)
                    count += 1
                session.commit()
            return count
        except Exception as exc:
            logger.warning("persist_corrections failed: %s", exc)
            return 0

    def merge_shared_regions(
        self,
        app_id: str,
        current_regions: list[Any],
        current_dynamic_zone_ids: set[str],
        db_session_factory: Any = None,
    ) -> list[dict[str, Any]]:
        """跨页面 shared_regions 归并。

        同一 app 下，不同 vlm_responses 中 bounds/role 相似且不在 dynamic_zone
        中的区域 → 归并为 shared_region。

        Returns:
            归并后的 shared_regions 列表（dict 格式）
        """
        if db_session_factory is None:
            return []

        try:
            from src.vlm.modeler import _open_session

            with _open_session(db_session_factory) as session:
                if session is None:
                    return []
                result = session.execute(
                    __import__("sqlalchemy").text(
                        """SELECT v.parsed_model FROM vlm_responses v
                           JOIN page_models p ON v.page_model_id = p.page_model_id
                           WHERE v.status = 'success' AND v.parsed_model != ''
                           AND p.app_id = :app_id
                           ORDER BY v.created_at DESC LIMIT 50"""
                    ),
                    {"app_id": app_id},
                )
                rows = result.fetchall()

            # 收集所有 regions（排除 dynamic_zone）
            region_signatures: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                try:
                    model_dict = __import__("json").loads(row[0])
                    for r in model_dict.get("regions", []):
                        role = r.get("role", "")
                        bounds = r.get("bounds", [])
                        if len(bounds) < 4:
                            continue
                        # 跳过 dynamic_zone
                        zone_id = r.get("zone_id", "") or r.get("region_id", "")
                        if zone_id in current_dynamic_zone_ids:
                            continue
                        # 签名：role + 近似 bounds（四舍五入到 10px）
                        approx = tuple(round(b / 10) * 10 for b in bounds[:4])
                        sig = f"{role}:{approx}"
                        if sig not in region_signatures:
                            region_signatures[sig] = []
                        region_signatures[sig].append(r)
                except Exception:
                    continue

            # 出现 >= 3 次 → shared
            shared: list[dict[str, Any]] = []
            for sig, occurrences in region_signatures.items():
                if len(occurrences) >= 3:
                    ref = occurrences[0]
                    shared.append({
                        "region_id": ref.get("region_id", ""),
                        "role": ref.get("role", ""),
                        "bounds": ref.get("bounds", []),
                        "purpose": ref.get("purpose", ""),
                        "occurrence_count": len(occurrences),
                        "shared": True,
                    })

            # 标记当前 canvas 中匹配的区域
            for sr in shared:
                sr_bounds = _vlm_bounds_to_tuple(sr.get("bounds", []))
                if sr_bounds is None:
                    continue
                for region in current_regions:
                    if region.bounds is None:
                        continue
                    if _iou(sr_bounds, region.bounds) > 0.6:
                        if hasattr(region, "attributes"):
                            attrs = dict(region.attributes or {})
                            attrs["shared"] = True
                            region.attributes = attrs

            return shared
        except Exception as exc:
            logger.warning("merge_shared_regions failed: %s", exc)
            return []
