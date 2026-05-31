"""
Page Compiler — P3 页面结构编译器

整合所有感知结果，输出标准 InteractionCanvas。

使用流程：
1. 接收原始感知数据（UIA 元素、进程信息、DOM 等）
2. 可选：接收 ZonePartitioner 结果（WindowZoneStructure）用于真实区域划分
3. 调用 surface_classifier 判定 surface_type
4. 调用 content_area_classifier 判定 content_area subtype
5. 调用 page_class_classifier 判定 page_class（消费 surface_type + content_area + 实际 regions）
6. 生成 elements + 关联 region_id
7. 生成 locators（按 surface_type 优先级）
8. 生成 anchors（关键元素锚点）
9. 生成 relations（元素布局关系）
10. 组装 provider_trace
11. 输出完整 InteractionCanvas

设计原则：
- 每次感知调用一次 compile()，输出完整快照
- 不做增量更新（增量在 P5 记忆层处理）
- zone_structure 可选；有则用真实区域，无则降级为单虚拟 content_area
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.page_compiler_models import (
    InteractionCanvas,
    SurfaceInfo,
    PageInfo,
    SurfaceType,
    ContentAreaSubtype,
    Region,
    Candidate,
    SemanticRole,
    Locator,
    LocatorKind,
    Anchor,
    ElementRelation,
    AppInfo,
    WindowInfoSnapshot,
    ProviderTrace,
    ScrollContext,
    VisualRegion,
)
from src.perception.surface_classifier import SurfaceClassifier, SurfaceClassification
from src.perception.content_area_classifier import (
    ContentAreaClassifier,
    ContentAreaClassification,
)
from src.perception.page_class_classifier import (
    PageClassClassifier,
    PageClassResult,
)
from src.perception.page_compiler_builders import InteractionCanvasEngineSupportBuilder
from src.perception.page_compiler_candidates import populate_boundary_candidate_artifacts
from src.perception.page_compiler_locators import InteractionCanvasEngineLocatorBuilder
from src.perception.page_compiler_regions import InteractionCanvasEngineRegionBuilder
from src.perception.page_compiler_subtypes import InteractionCanvasEngineSubtypeBuilder
from src.perception.zone_partitioner import (
    ZonePartitioner,
    WindowZoneStructure,
    ZoneType,
)


# Locator kind priority by surface_type
_LOCATOR_PRIORITY_BY_SURFACE: dict[SurfaceType, list[LocatorKind]] = {
    SurfaceType.NATIVE_UIA: [
        LocatorKind.UIA,
        LocatorKind.RELATIVE,
        LocatorKind.OCR,
        LocatorKind.TEMPLATE_ICON,
        LocatorKind.VISION_BBOX,
        LocatorKind.EPHEMERAL_COORD,
    ],
    SurfaceType.BROWSER: [
        LocatorKind.DOM,
        LocatorKind.RELATIVE,
        LocatorKind.VISION_BBOX,
        LocatorKind.OCR,
        LocatorKind.TEMPLATE_ICON,
        LocatorKind.EPHEMERAL_COORD,
    ],
    SurfaceType.ELECTRON_WEBVIEW: [
        LocatorKind.UIA,
        LocatorKind.DOM,
        LocatorKind.VISION_BBOX,
        LocatorKind.RELATIVE,
        LocatorKind.OCR,
        LocatorKind.TEMPLATE_ICON,
        LocatorKind.EPHEMERAL_COORD,
    ],
    SurfaceType.CANVAS_SELF_DRAWN: [
        LocatorKind.VISION_BBOX,
        LocatorKind.OCR,
        LocatorKind.TEMPLATE_ICON,
        LocatorKind.RELATIVE,
        LocatorKind.EPHEMERAL_COORD,
    ],
    SurfaceType.UNKNOWN: [
        LocatorKind.UIA,
        LocatorKind.RELATIVE,
        LocatorKind.OCR,
        LocatorKind.VISION_BBOX,
        LocatorKind.EPHEMERAL_COORD,
    ],
}


def _vertically_overlaps(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return True when two boxes overlap enough on the Y axis."""
    top = max(first[1], second[1])
    bottom = min(first[3], second[3])
    return bottom > top


class InteractionCanvasEngine:
    """
    页面结构编译器

    将各种感知输入编译为标准 InteractionCanvas。
    """

    def __init__(self) -> None:
        self._surface_classifier = SurfaceClassifier()
        self._content_area_classifier = ContentAreaClassifier()
        self._page_classifier = PageClassClassifier()
        self._zone_partitioner = ZonePartitioner()
        self._support_builder = InteractionCanvasEngineSupportBuilder()
        self._locator_builder = InteractionCanvasEngineLocatorBuilder(
            _LOCATOR_PRIORITY_BY_SURFACE
        )
        self._subtype_builder = InteractionCanvasEngineSubtypeBuilder(
            self._support_builder.infer_semantic_role
        )
        self._region_builder = InteractionCanvasEngineRegionBuilder(
            infer_semantic_role=self._support_builder.infer_semantic_role,
            get_element_bounds=self._get_element_bounds,
            coerce_bbox=self._coerce_bbox,
            subtype_builder=self._subtype_builder,
        )

    def compile(
        self,
        process_name: str | None = None,
        process_path: str | None = None,
        window_hwnd: int = 0,
        window_title: str = "",
        uia_element_count: int = 0,
        uia_control_types: list[str] | None = None,
        has_dom_bridge: bool = False,
        dom_ready: bool = False,
        visual_score: float | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        vision_control_groups: list[dict[str, Any]] | None = None,
        vision_interaction_hints: list[dict[str, Any]] | None = None,
        structure_evidence_score: float | None = None,
        zone_structure: WindowZoneStructure | None = None,
        allow_legacy_zone_reconstruction: bool = True,
        visual_regions: list[VisualRegion] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> InteractionCanvas:
        """
        编译入口

        Args:
            process_name: 进程名（e.g. "WeChat.exe"）
            process_path: 进程路径
            window_hwnd: 窗口句柄
            window_title: 窗口标题
            uia_element_count: UIA 元素数量
            uia_control_types: UIA 控件类型列表
            has_dom_bridge: 是否有 DOM bridge
            dom_ready: DOM 是否就绪
            visual_score: 视觉分数
            raw_elements: 原始 UIA 元素列表
            zone_structure: ZonePartitioner 结果（可选，有则用真实区域）

        Returns:
            InteractionCanvas 完整页面快照
        """
        raw_elements = raw_elements or []
        ocr_blocks = ocr_blocks or []
        vision_candidates = vision_candidates or []
        vision_layout_regions = vision_layout_regions or []
        vision_control_groups = vision_control_groups or []
        vision_interaction_hints = vision_interaction_hints or []

        # === 1. 分类：surface_type ===
        surface_result = self._surface_classifier.classify(
            process_name=process_name,
            process_path=process_path,
            uia_element_count=uia_element_count,
            uia_control_types=uia_control_types,
            has_dom_bridge=has_dom_bridge,
            dom_ready=dom_ready,
            visual_score=visual_score,
        )
        surface_info = self._build_surface_info(surface_result)
        surface_type = surface_info.surface_type
        raw_elements = self._augment_raw_elements(raw_elements, ocr_blocks, surface_type)

        # === 2. 区域划分（可选，使用 ZonePartitioner 结果）===
        if zone_structure is None and raw_elements and allow_legacy_zone_reconstruction:
            # 用 ZonePartitioner 从 raw_elements 重建区域
            window_width = self._estimate_window_width(raw_elements)
            window_height = self._estimate_window_height(raw_elements)
            zone_structure = self._zone_partitioner.partition_from_dicts(
                raw_elements, window_width, window_height
            )

        # === 3. 分类：content_area subtype ===
        content_area_elements = self._extract_content_area_elements(raw_elements, zone_structure)
        content_subtype, content_subtype_confidence = self._classify_content_area(
            content_area_elements
        )

        fallback_regions: list[Region] | None = None
        fallback_region_map: dict[str, str] | None = None
        if zone_structure is None:
            fallback_regions, fallback_region_map = self._infer_layout_regions(
                raw_elements,
                content_subtype,
                vision_layout_regions,
                geometric_regions,
                fusion_diagnostics,
            )

        # === 4. 构建 elements（带 region_id 关联 + 来源归因）===
        elements, element_region_map, element_source_stats = self._build_elements(
            raw_elements,
            zone_structure,
            surface_type,
            fallback_region_map,
            ocr_blocks,
            vision_candidates,
            vision_control_groups,
        )

        # === 5. 构建 regions（使用真实 ZonePartitioner 结果）===
        regions = self._build_regions(
            raw_elements,
            zone_structure,
            surface_type,
            content_subtype,
            element_region_map,
            fallback_regions,
            vision_layout_regions,
            geometric_regions,
            fusion_diagnostics,
        )

        # === 6. 生成 locators（DOM locator 仅在 has_dom_bridge=True 时生成）===
        elements = self._synthesize_boundary_elements(elements, regions)
        self._assign_content_groups(elements, regions)
        locators = self._generate_locators(elements, surface_type, has_dom_bridge)

        # === 7. 生成 anchors ===
        anchors = self._generate_anchors(elements, regions)
        self._attach_anchor_refs_to_locators(elements, locators)

        # === 8. 生成 relations ===
        relations = self._generate_relations(elements, regions)
        scroll_contexts = self._generate_scroll_contexts(regions, elements)

        # === 9. 分类：page_class（消费 surface_type + content_area + 实际 regions）===
        snapshot_for_page_class = self._build_snapshot_for_page_class(
            process_name, window_hwnd, window_title,
            surface_info, elements, regions
        )
        page_class_result = self._page_classifier.classify(snapshot_for_page_class)
        page_info = self._build_page_info(page_class_result)

        # === 10. 构建 ProviderTrace ===
        provider_trace = self._build_provider_trace(
            uia_element_count=uia_element_count,
            has_dom_bridge=has_dom_bridge,
            visual_score=visual_score,
            zone_structure=zone_structure,
            element_source_stats=element_source_stats,
            locators=locators,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            vision_layout_regions=vision_layout_regions,
            vision_control_groups=vision_control_groups,
            vision_interaction_hints=vision_interaction_hints,
            structure_evidence_score=structure_evidence_score,
        )

        # === 11. 组装 InteractionCanvas ===
        canvas_id = f"snap_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        artifacts: dict[str, Any] = {}
        if ocr_blocks:
            artifacts["ocr_blocks"] = [
                {
                    "text": str(block.get("text") or ""),
                    "bbox": list(self._coerce_bbox(block.get("bbox")) or ()),
                    "confidence": block.get("confidence"),
                }
                for block in ocr_blocks
                if self._coerce_bbox(block.get("bbox")) is not None
            ]
        if vision_candidates:
            artifacts["vision_candidates"] = [
                dict(candidate)
                for candidate in vision_candidates
                if self._coerce_bbox(candidate.get("bbox")) is not None
            ]
        if vision_layout_regions:
            artifacts["vision_layout_regions"] = [dict(region) for region in vision_layout_regions]
        if vision_control_groups:
            artifacts["vision_control_groups"] = [dict(group) for group in vision_control_groups]
        if vision_interaction_hints:
            artifacts["vision_interaction_hints"] = [
                dict(hint) for hint in vision_interaction_hints
            ]
        if structure_evidence_score is not None:
            artifacts["structure_evidence_score"] = structure_evidence_score

        # Phase 1: geometric_regions (neutral, parallel with visual_regions)
        if geometric_regions:
            artifacts["geometric_regions"] = [gr.to_dict() for gr in geometric_regions]
        if fusion_diagnostics:
            artifacts["fusion_diagnostics"] = dict(fusion_diagnostics)

        # Derive providers_used / providers_failed from provider_trace
        providers_used = []
        providers_failed = []
        for flag, name in [
            ("uia_used", "uia"), ("ocr_used", "ocr"),
            ("vision_used", "vision"), ("vlm_used", "vlm"),
            ("dom_used", "dom"),
        ]:
            if getattr(provider_trace, flag, False):
                providers_used.append(name)
        # Check for failed providers in details
        for name, detail in provider_trace.provider_details.items():
            if isinstance(detail, dict) and detail.get("success") is False:
                if name not in providers_used:
                    providers_failed.append(name)

        snapshot = InteractionCanvas(
            canvas_id=canvas_id,
            captured_at=now,
            app=AppInfo(
                process_name=process_name,
                exe_path=process_path,
            ),
            window=WindowInfoSnapshot(
                hwnd=window_hwnd,
                title=window_title,
            ),
            surface=surface_info,
            page=page_info,
            regions=regions,
            elements=elements,
            locators=locators,
            anchors=anchors,
            relations=relations,
            scroll_contexts=scroll_contexts,
            provider_trace=provider_trace,
            artifacts=artifacts,
            # New Phase 1 fields
            providers_used=providers_used,
            providers_failed=providers_failed,
            partial=len(providers_failed) > 0,
            stable=not (getattr(page_info, "state_flags", {}).get("loading", False)),
        )
        populate_boundary_candidate_artifacts(snapshot)
        return snapshot

    # -------------------------------------------------------------------------
    # Surface
    # -------------------------------------------------------------------------

    def _build_surface_info(self, result: SurfaceClassification) -> SurfaceInfo:
        """从 surface 分类结果构建 SurfaceInfo"""
        from src.perception.page_compiler_models import SurfaceEvidence

        evidence = [
            SurfaceEvidence(
                layer=e.layer,
                feature=e.feature,
                weight=e.weight,
                value=e.value,
                description=e.description,
            )
            for e in result.evidence
        ]

        return SurfaceInfo(
            surface_type=result.surface_type,
            confidence=result.confidence,
            evidence=evidence,
        )

    # -------------------------------------------------------------------------
    # Zone / Region
    # -------------------------------------------------------------------------

    def _get_element_bounds(self, elem: dict[str, Any]) -> tuple[int, int, int, int] | None:
        """
        从 dict 提取 bounds，兼容以下格式：
        - {"bounding_rect": (l, t, r, b)}  ← 当前主格式（来自 perception_service）
        - {"left": l, "top": t, "right": r, "bottom": b}  ← 旧格式（直接字段）
        - UIAElementInfo 对象（有 left/top/right/bottom 属性）
        """
        # UIAElementInfo 对象
        if hasattr(elem, "bounding_rect") and elem.bounding_rect:
            return elem.bounding_rect
        # dict with bounding_rect tuple
        if "bounding_rect" in elem:
            br = elem["bounding_rect"]
            if isinstance(br, tuple) and len(br) == 4:
                return br
            if isinstance(br, dict):
                l = br.get("left", 0)
                t = br.get("top", 0)
                r = br.get("right", 0)
                b = br.get("bottom", 0)
                return (l, t, r, b)
        # dict with individual keys
        l = elem.get("left")
        t = elem.get("top")
        r = elem.get("right")
        b = elem.get("bottom")
        if all(v is not None for v in (l, t, r, b)):
            return (l, t, r, b)
        return None

    def _estimate_window_width(self, raw_elements: list[dict[str, Any]]) -> int:
        if not raw_elements:
            return 1920
        max_right = 0
        for e in raw_elements:
            bounds = self._get_element_bounds(e)
            if bounds:
                max_right = max(max_right, bounds[2])
        return max(max_right + 50, 100)

    def _estimate_window_height(self, raw_elements: list[dict[str, Any]]) -> int:
        if not raw_elements:
            return 1080
        max_bottom = 0
        for e in raw_elements:
            bounds = self._get_element_bounds(e)
            if bounds:
                max_bottom = max(max_bottom, bounds[3])
        return max(max_bottom + 50, 100)

    def _augment_raw_elements(
        self,
        raw_elements: list[dict[str, Any]],
        ocr_blocks: list[dict[str, Any]],
        surface_type: SurfaceType,
    ) -> list[dict[str, Any]]:
        """Augment sparse UIA trees with OCR-backed synthetic elements."""
        if not ocr_blocks:
            return raw_elements
        if surface_type not in {
            SurfaceType.ELECTRON_WEBVIEW,
            SurfaceType.CANVAS_SELF_DRAWN,
            SurfaceType.BROWSER,
        } and len(raw_elements) > 4:
            return raw_elements
        if len(raw_elements) > 12:
            return raw_elements

        augmented = list(raw_elements)
        for index, block in enumerate(ocr_blocks):
            bbox = self._coerce_bbox(block.get("bbox"))
            text = str(block.get("text") or "").strip()
            if bbox is None or not text:
                continue
            if self._max_iou_with_elements(bbox, raw_elements) >= 0.65:
                continue
            control_type = self._infer_synthetic_control_type(text, bbox, raw_elements)
            augmented.append(
                {
                    "element_id": f"ocr_synth_{index}",
                    "control_type": control_type,
                    "name": text,
                    "text": text,
                    "bounding_rect": bbox,
                    "synthetic_source": "ocr",
                    "ocr_confidence": block.get("confidence"),
                }
            )
        return augmented

    def _max_iou_with_elements(
        self,
        bbox: tuple[int, int, int, int],
        raw_elements: list[dict[str, Any]],
    ) -> float:
        best_score = 0.0
        for element in raw_elements:
            bounds = self._get_element_bounds(element)
            if bounds is None:
                continue
            best_score = max(best_score, self._compute_iou(bounds, bbox))
        return best_score

    def _infer_synthetic_control_type(
        self,
        text: str,
        bbox: tuple[int, int, int, int],
        raw_elements: list[dict[str, Any]],
    ) -> str:
        left, top, right, bottom = bbox
        width = max(right - left, 1)
        center_x = (left + right) // 2
        window_left = min((self._get_element_bounds(element) or bbox)[0] for element in (raw_elements or [{}]))
        window_right = max((self._get_element_bounds(element) or bbox)[2] for element in (raw_elements or [{}]))
        window_width = max(window_right - window_left, 1)
        lower_text = text.lower()

        if any(token in text for token in ("搜索", "查找", "筛选")) or any(
            token in lower_text for token in ("search", "find", "filter")
        ):
            return "EditControl"
        if any(token in text for token in ("发送", "提交", "确定", "确认", "取消", "测试")) or any(
            token in lower_text for token in ("send", "submit", "confirm", "cancel", "test")
        ):
            return "ButtonControl"
        if center_x <= window_left + int(window_width * 0.38):
            return "ListItemControl"
        if width >= int(window_width * 0.45):
            return "PaneControl"
        return "TextControl"

    def _synthesize_boundary_elements(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> list[Candidate]:
        """Promote strong region geometry into boundary candidates."""
        synthesized = list(elements)
        existing_ids = {element.element_id for element in synthesized}
        region_elements: dict[str, list[Candidate]] = {}
        for element in synthesized:
            if element.region_id:
                region_elements.setdefault(element.region_id, []).append(element)

        for region in regions:
            if region.role != "composer_area":
                continue
            synthetic_input = self._synthesize_message_input_candidate(
                region,
                region_elements.get(region.region_id, []),
                elements,
                existing_ids,
            )
            if synthetic_input is not None:
                synthesized.append(synthetic_input)
                existing_ids.add(synthetic_input.element_id)
        return synthesized

    def _synthesize_message_input_candidate(
        self,
        region: Region,
        region_items: list[Candidate],
        all_items: list[Candidate],
        existing_ids: set[str],
    ) -> Candidate | None:
        """Create a message-input boundary candidate inside composer area."""
        if region.bounds is None:
            return None
        if any(
            item.semantic_role in {SemanticRole.MESSAGE_INPUT, SemanticRole.TEXT_INPUT}
            for item in region_items
        ):
            return None

        left, top, right, bottom = region.bounds
        send_buttons = [
            item for item in all_items
            if item.bounds is not None and item.semantic_role in {
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.BUTTON,
            }
            and _vertically_overlaps(item.bounds, region.bounds)
            and item.bounds[0] >= left
        ]
        candidate_right = right - 16
        if send_buttons:
            candidate_right = min(item.bounds[0] for item in send_buttons if item.bounds) - 8
        candidate_left = left + 12
        candidate_top = top + max(8, int((bottom - top) * 0.12))
        candidate_bottom = bottom - max(8, int((bottom - top) * 0.12))
        if candidate_right <= candidate_left + 80:
            if send_buttons:
                primary_send = min(
                    (item for item in send_buttons if item.bounds is not None),
                    key=lambda item: item.bounds[0],
                )
                send_left, send_top, _, send_bottom = primary_send.bounds
                candidate_right = send_left - 8
                candidate_left = max(12, candidate_right - 320)
                candidate_top = min(candidate_top, send_top)
                candidate_bottom = max(candidate_bottom, send_bottom)
            else:
                candidate_right = max(candidate_left + 160, right - 20)

        candidate_id = f"{region.region_id}_message_input"
        if candidate_id in existing_ids:
            return None

        return Candidate(
            element_id=candidate_id,
            region_id=region.region_id,
            semantic_role=SemanticRole.MESSAGE_INPUT,
            control_type="SyntheticInputControl",
            bounds=(candidate_left, candidate_top, candidate_right, candidate_bottom),
            text="",
            name="message_input",
            interactable=True,
            provider_sources=["boundary_candidate"],
            attributes={
                "synthetic_source": "boundary_candidate",
                "candidate_kind": "message_input_boundary",
                "needs_manual_label": False,
            },
        )

    def _extract_content_area_elements(
        self,
        raw_elements: list[dict[str, Any]],
        zone_structure: WindowZoneStructure | None,
    ) -> list[dict[str, Any]]:
        """从原始元素或 ZonePartitioner 结果中提取 content_area 元素"""
        if zone_structure is not None:
            content_zone = zone_structure.get_zone(ZoneType.CONTENT_AREA)
            if content_zone and content_zone.elements:
                # 将 MergedElement 转回 dict 格式供 ContentAreaClassifier 使用
                return [
                    {
                        "element_id": e.element_id or f"zone_elem_{i}",
                        "control_type": e.control_type,
                        "name": e.name,
                        "text": e.name,  # MergedElement 用 name 作为文本
                        "bounding_rect": e.bounding_rect,
                    }
                    for i, e in enumerate(content_zone.elements)
                ]

        # 降级策略：取中间 Y 区域的元素
        if not raw_elements:
            return []

        all_rects = []
        for e in raw_elements:
            bounds = self._get_element_bounds(e)
            if bounds:
                all_rects.append(bounds)
        if not all_rects:
            return raw_elements

        min_y = min(r[1] for r in all_rects)
        max_y = max(r[3] for r in all_rects)
        window_height = max_y - min_y if max_y > min_y else 1

        threshold_top = min_y + int(window_height * 0.2)
        threshold_bottom = max_y - int(window_height * 0.2)

        content_elements = []
        for e in raw_elements:
            bounds = self._get_element_bounds(e)
            if bounds:
                if bounds[1] >= threshold_top and bounds[3] <= threshold_bottom:
                    content_elements.append(e)

        return content_elements if content_elements else raw_elements[:10]

    def _classify_content_area(
        self, elements: list[dict[str, Any]]
    ) -> tuple[ContentAreaSubtype, float]:
        """对 content_area 元素进行 subtype 分类"""
        result = self._content_area_classifier.classify(elements)
        return result.subtype, result.confidence

    def _build_elements(
        self,
        raw_elements: list[dict[str, Any]],
        zone_structure: WindowZoneStructure | None,
        surface_type: SurfaceType,
        fallback_region_map: dict[str, str] | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        vision_control_groups: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Candidate], dict[str, str], dict[str, int]]:
        """
        从原始元素构建 Candidate 列表，并建立 element → region 映射

        Returns:
            (elements, element_region_map, element_source_stats)
            element_source_stats: {"zone_partition": N, "uia": N} 来源统计
        """
        elements: list[Candidate] = []
        element_region_map: dict[str, str] = {}
        element_source_stats: dict[str, int] = {"zone_partition": 0, "uia": 0, "ocr": 0, "vision": 0}
        fallback_region_map = fallback_region_map or {}
        ocr_blocks = ocr_blocks or []
        vision_candidates = vision_candidates or []
        vision_control_groups = vision_control_groups or []

        # 如果有 zone_structure，建立 element -> zone 的映射
        element_to_zone: dict[str, str] = {}
        if zone_structure is not None:
            for zone in zone_structure.zones:
                for merged_elem in zone.elements:
                    # MergedElement 有 original_element_ids 列表
                    for orig_id in merged_elem.original_element_ids:
                        element_to_zone[orig_id] = zone.zone_type.value
                    # 也用 MergedElement 自己的 element_id
                    element_to_zone[merged_elem.element_id] = zone.zone_type.value

        for i, elem_dict in enumerate(raw_elements):
            element_id = elem_dict.get("element_id", f"elem_{i}")
            bounds = self._get_element_bounds(elem_dict)  # 统一支持 bounding_rect / left/top/right/bottom / UIAElementInfo
            control_type = elem_dict.get("control_type", "")
            text = elem_dict.get("text", "") or elem_dict.get("name", "")
            name = elem_dict.get("name")
            value = elem_dict.get("value")
            automation_id = elem_dict.get("automation_id")
            ocr_match = self._match_ocr_block(bounds, ocr_blocks)
            if (not text or not text.strip()) and ocr_match:
                text = ocr_match.get("text", text)
                name = name or ocr_match.get("text")
            vision_match = self._match_vision_candidate(bounds, vision_candidates)

            # 推断 region_id
            region_id = element_region_map.get(element_id)
            if region_id is None:
                # 尝试从 element_to_zone 映射
                zone_type = element_to_zone.get(element_id)
                if zone_type:
                    region_id = self._zone_type_to_region_id(zone_type)
                elif element_id in fallback_region_map:
                    region_id = fallback_region_map[element_id]
                elif bounds:
                    region_id = "region_content_0"  # 降级默认

            # 推断 semantic_role
            semantic_role = self._resolve_semantic_role(
                control_type=control_type,
                text=text,
                name=name,
                vision_match=vision_match,
            )
            if elem_dict.get("synthetic_source") == "ocr":
                semantic_role = self._refine_synthetic_semantic_role(
                    semantic_role,
                    control_type,
                    bounds,
                    raw_elements,
                )

            # 推断 interactable
            interactable = self._is_interactable(control_type)

            # 元素来源归因：来自 zone_partition 还是直接来自 UIA/fallback
            source = "zone_partition" if element_id in element_to_zone else elem_dict.get("synthetic_source", "uia")
            element_source_stats[source] += 1
            provider_sources = [source]
            attributes: dict[str, Any] = {"automation_id": automation_id} if automation_id else {}
            if ocr_match:
                provider_sources.append("ocr")
                if source != "ocr":
                    element_source_stats["ocr"] += 1
                attributes["ocr_text"] = ocr_match.get("text")
                attributes["ocr_confidence"] = ocr_match.get("confidence")
                attributes["ocr_bbox"] = ocr_match.get("bbox")
            if vision_match:
                provider_sources.append("vision")
                element_source_stats["vision"] += 1
                attributes["vision_candidate"] = vision_match
                if vision_match.get("group_id"):
                    attributes["vision_group_id"] = vision_match.get("group_id")
                if vision_match.get("region_role"):
                    attributes["vision_region_role"] = vision_match.get("region_role")
                if vision_match.get("interaction_hints"):
                    attributes["interaction_hints"] = vision_match.get("interaction_hints")
                if vision_match.get("structure_evidence_score") is not None:
                    attributes["structure_evidence_score"] = vision_match.get("structure_evidence_score")

            # Compute numeric confidence from provider sources
            confidence = 0.8  # UIA base confidence
            if vision_match:
                confidence = max(confidence, float(vision_match.get("confidence", 0.5)))
            if ocr_match:
                confidence = max(confidence, float(ocr_match.get("confidence", 0.5)))

            elem = Candidate(
                element_id=element_id,
                region_id=region_id,
                bounds=bounds,
                control_type=control_type,
                text=text,
                name=name,
                value=value,
                semantic_role=semantic_role,
                interactable=interactable,
                confidence=confidence,
                provider_sources=list(dict.fromkeys(provider_sources)),
                attributes=attributes,
            )
            self._apply_vision_group_hint(elem, vision_control_groups)
            elements.append(elem)
            if region_id:
                element_region_map[element_id] = region_id

        # Vision-only candidates: vision candidates that didn't match any UIA element.
        # These must become standalone Candidate objects so Agent query can find them.
        matched_ids: set[str] = set()
        for elem in elements:
            vc = elem.attributes.get("vision_candidate")
            if vc:
                matched_ids.add(vc.get("candidate_id", ""))

        for vc in vision_candidates:
            cid = vc.get("candidate_id", "")
            if cid in matched_ids:
                continue
            bbox = self._coerce_bbox(vc.get("bbox"))
            if bbox is None:
                continue
            source = vc.get("source", "vision")
            confidence = float(vc.get("confidence", 0.5))
            elements.append(Candidate(
                element_id=cid or f"vision_only_{len(elements)}",
                semantic_role=SemanticRole.UNKNOWN,
                control_type=vc.get("kind", ""),
                bounds=bbox,
                text=vc.get("text", ""),
                provider_sources=[source],
                confidence=confidence,
                interactable=True,
                attributes={"vision_only": True, "source": source},
            ))
            element_source_stats["vision"] += 1

        return elements, element_region_map, element_source_stats

    def _apply_vision_group_hint(
        self,
        element: Candidate,
        vision_control_groups: list[dict[str, Any]],
    ) -> None:
        """Attach vision group hints to elements before region-based grouping."""
        vision_group_id = element.attributes.get("vision_group_id")
        if vision_group_id:
            element.content_group_id = vision_group_id
            return
        for group in vision_control_groups:
            member_ids = group.get("member_ids") or group.get("elements") or []
            if element.element_id not in member_ids:
                continue
            group_id = group.get("group_id") or group.get("id")
            if group_id:
                element.content_group_id = str(group_id)
            break

    def _match_ocr_block(
        self,
        bounds: tuple[int, int, int, int] | None,
        ocr_blocks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find the best overlapping OCR block for one element."""
        if bounds is None:
            return None
        best_match: dict[str, Any] | None = None
        best_score = 0.0
        for block in ocr_blocks:
            bbox = self._coerce_bbox(block.get("bbox"))
            if bbox is None:
                continue
            iou = self._compute_iou(bounds, bbox)
            if iou > best_score and iou >= 0.20:
                best_score = iou
                best_match = {
                    "text": block.get("text", ""),
                    "bbox": bbox,
                    "confidence": block.get("confidence", 0.0),
                }
        return best_match

    def _refine_synthetic_semantic_role(
        self,
        semantic_role: SemanticRole,
        control_type: str,
        bounds: tuple[int, int, int, int] | None,
        raw_elements: list[dict[str, Any]],
    ) -> SemanticRole:
        if bounds is None:
            return semantic_role
        control_type_lower = str(control_type or "").lower()
        if "listitem" not in control_type_lower and semantic_role != SemanticRole.LIST_ITEM:
            return semantic_role
        window_left = min((self._get_element_bounds(element) or bounds)[0] for element in (raw_elements or [{}]))
        window_right = max((self._get_element_bounds(element) or bounds)[2] for element in (raw_elements or [{}]))
        window_width = max(window_right - window_left, 1)
        center_x = (bounds[0] + bounds[2]) // 2
        if center_x <= window_left + int(window_width * 0.38):
            return SemanticRole.CHAT_ITEM
        return semantic_role

    def _match_vision_candidate(
        self,
        bounds: tuple[int, int, int, int] | None,
        vision_candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find the best overlapping vision candidate for one element."""
        if bounds is None:
            return None
        best_match: dict[str, Any] | None = None
        best_score = 0.0
        for candidate in vision_candidates:
            bbox = self._coerce_bbox(candidate.get("bbox"))
            if bbox is None:
                continue
            iou = self._compute_iou(bounds, bbox)
            if iou > best_score and iou >= 0.15:
                best_score = iou
                best_match = candidate
        return best_match

    def _coerce_bbox(self, value: Any) -> tuple[int, int, int, int] | None:
        """Coerce a list/tuple bbox into xyxy tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])

    def _compute_iou(
        self,
        rect_a: tuple[int, int, int, int],
        rect_b: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two xyxy boxes."""
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _zone_type_to_region_id(self, zone_type: str) -> str:
        return self._region_builder.zone_type_to_region_id(zone_type)

    def _infer_layout_regions(
        self,
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[Region], dict[str, str]]:
        return self._region_builder.infer_layout_regions(
            raw_elements,
            content_subtype,
            vision_layout_regions,
            geometric_regions,
            fusion_diagnostics,
        )

    def _add_content_subregions(
        self,
        regions: list[Region],
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        inferred_region_map: dict[str, str],
        vision_layout_regions: list[dict[str, Any]],
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        return self._region_builder.add_content_subregions(
            regions,
            raw_elements,
            content_subtype,
            inferred_region_map,
            vision_layout_regions,
            geometric_regions,
            fusion_diagnostics,
        )

    def _build_standard_content_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        existing_regions: list[Region],
    ) -> list[Region]:
        return self._region_builder.build_standard_content_regions(
            content_region,
            content_elements,
            existing_regions,
        )

    def _build_dialog_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        existing_regions: list[Region],
    ) -> list[Region]:
        return self._region_builder.build_dialog_subregions(
            content_region,
            content_elements,
            existing_regions,
        )

    def _is_dialog_layout_dominant(
        self,
        dialog_regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> bool:
        return self._region_builder.is_dialog_layout_dominant(
            dialog_regions,
            content_elements,
        )

    def _order_child_regions(self, child_regions: list[Region]) -> list[Region]:
        return self._region_builder.order_child_regions(child_regions)

    def _refresh_child_region_assignments(
        self,
        content_region: Region,
        child_regions: list[Region],
        inferred_region_map: dict[str, str],
    ) -> None:
        self._region_builder.refresh_child_region_assignments(
            content_region,
            child_regions,
            inferred_region_map,
        )

    def _build_vision_guided_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        vision_layout_regions: list[dict[str, Any]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._region_builder.build_vision_guided_regions(
            content_region,
            content_elements,
            vision_layout_regions,
            inferred_region_map,
        )

    def _merge_candidate_regions(
        self,
        existing_regions: list[Region],
        preferred_regions: list[Region],
    ) -> list[Region]:
        return self._region_builder.merge_candidate_regions(
            existing_regions,
            preferred_regions,
        )

    def _region_id_for_role(self, role: str, index: int) -> str:
        return self._region_builder.region_id_for_role(role, index)

    def _normalize_region_role(self, value: Any) -> str | None:
        return self._region_builder.normalize_region_role(value)

    def _clip_bbox_to_region(
        self,
        bbox: tuple[int, int, int, int],
        region_bounds: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        return self._region_builder.clip_bbox_to_region(bbox, region_bounds)

    def _select_region_members_from_bounds(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        region_bounds: tuple[int, int, int, int],
    ) -> list[str]:
        return self._region_builder.select_region_members_from_bounds(
            content_elements,
            region_bounds,
        )

    def _resolve_structural_content_subtype(
        self,
        content_subtype: ContentAreaSubtype,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> ContentAreaSubtype:
        return self._region_builder.resolve_structural_content_subtype(
            content_subtype,
            content_elements,
            content_bounds,
        )

    def _has_editor_shell(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_editor_shell(content_elements, content_bounds)

    def _has_grid_table_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_grid_table_layout(content_elements, content_bounds)

    def _has_canvas_doc_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_canvas_doc_layout(content_elements, content_bounds)

    def _has_dashboard_widget_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_dashboard_widget_layout(content_elements, content_bounds)

    def _has_chat_stream_area(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_chat_stream_area(content_elements, content_bounds)

    def _has_bottom_action_band(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_bottom_action_band(content_elements, content_bounds)

    def _has_list_detail_split(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        return self._region_builder.has_list_detail_split(content_elements, content_bounds)

    def _build_chat_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_chat_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_form_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_form_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_list_detail_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_list_detail_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_editor_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_editor_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_dashboard_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_dashboard_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_grid_table_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_grid_table_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _build_canvas_doc_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        return self._subtype_builder.build_canvas_doc_subregions(
            content_region,
            content_elements,
            inferred_region_map,
        )

    def _partition_workspace_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        promote_filters: bool = False,
        grid_priority: bool = False,
        doc_viewer_priority: bool = False,
    ) -> tuple[
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
    ]:
        return self._subtype_builder.partition_workspace_regions(
            content_region,
            content_elements,
            promote_filters=promote_filters,
            grid_priority=grid_priority,
            doc_viewer_priority=doc_viewer_priority,
        )

    def _extract_filter_items(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        return self._subtype_builder.extract_filter_items(
            content_region,
            content_elements,
        )

    def _materialize_partitioned_regions(
        self,
        content_region: Region,
        inferred_region_map: dict[str, str],
        toolbar_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        filter_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        side_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        viewport_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        action_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    ) -> list[Region]:
        return self._subtype_builder.materialize_partitioned_regions(
            content_region,
            inferred_region_map,
            toolbar_items=toolbar_items,
            filter_items=filter_items,
            side_items=side_items,
            viewport_items=viewport_items,
            action_items=action_items,
        )

    def _annotate_dialog_region(
        self,
        dialog_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        self._region_builder.annotate_dialog_region(dialog_region, content_elements)

    def _annotate_dashboard_regions(
        self,
        regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        return self._subtype_builder.annotate_dashboard_regions(
            regions,
            content_elements,
        )

    def _annotate_grid_table_regions(
        self,
        regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        return self._subtype_builder.annotate_grid_table_regions(
            regions,
            content_elements,
        )

    def _estimate_row_group_count(
        self,
        row_ids: list[str],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> int:
        return self._subtype_builder.estimate_row_group_count(
            row_ids,
            content_elements,
        )

    def _make_child_region(
        self,
        region_id: str,
        role: str,
        parent_region: Region,
        items: list[tuple[str, tuple[int, int, int, int]]],
    ) -> Region:
        return self._subtype_builder.make_child_region(
            region_id,
            role,
            parent_region,
            items,
        )

    def _assign_content_groups(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> None:
        """Assign content groups from region roles."""
        region_by_id = {region.region_id: region for region in regions}
        element_by_id = {element.element_id: element for element in elements}
        for element in elements:
            region = region_by_id.get(element.region_id or "")
            if region is None:
                continue
            if element.content_group_id:
                continue
            role = region.role
            if role == "composer_area":
                element.content_group_id = "cg_composer"
            elif role == "message_stream":
                element.content_group_id = "cg_message_stream"
            elif role == "form_fields":
                element.content_group_id = "cg_form_fields"
            elif role in {"form_actions", "action_bar"}:
                element.content_group_id = "cg_action_bar"
            elif role in {"list_panel", "side_panel"}:
                element.content_group_id = "cg_list_panel"
            elif role in {"detail_panel", "viewport", "dialog_body"}:
                element.content_group_id = "cg_viewport"
            elif role == "toolbar":
                element.content_group_id = "cg_toolbar"
            elif role == "filter_bar":
                element.content_group_id = "cg_filter_bar"
        for region in regions:
            if region.role == "dialog_body":
                internal_roles = region.attributes.get("internal_roles") or {}
                for slot, element_ids in internal_roles.items():
                    if not isinstance(element_ids, list):
                        continue
                    for element_id in element_ids:
                        target = element_by_id.get(element_id)
                        if target is not None:
                            target.attributes["dialog_slot"] = slot
            if region.role == "viewport":
                widget_groups = region.attributes.get("widget_groups") or {}
                for group_name, element_ids in widget_groups.items():
                    if not isinstance(element_ids, list):
                        continue
                    for element_id in element_ids:
                        target = element_by_id.get(element_id)
                        if target is not None:
                            target.attributes["widget_group_id"] = f"widget_{group_name}"
                for element_id in region.attributes.get("grid_header_ids") or []:
                    target = element_by_id.get(element_id)
                    if target is not None:
                        target.attributes["grid_role"] = "header"
                for element_id in region.attributes.get("grid_row_ids") or []:
                    target = element_by_id.get(element_id)
                    if target is not None:
                        target.attributes["grid_role"] = "row"
                for element_id in region.attributes.get("grid_pagination_ids") or []:
                    target = element_by_id.get(element_id)
                    if target is not None:
                        target.attributes["grid_role"] = "pagination"

    def _generate_scroll_contexts(
        self,
        regions: list[Region],
        elements: list[Candidate],
    ) -> list[ScrollContext]:
        """Generate scroll contexts from viewport-like regions."""
        scroll_contexts: list[ScrollContext] = []
        element_map = {element.element_id: element for element in elements}
        for region in regions:
            if region.bounds is None:
                continue
            if region.role not in {
                "message_stream",
                "list_panel",
                "detail_panel",
                "viewport",
                "dialog_body",
                "content_area",
            }:
                continue
            left, top, right, bottom = region.bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            if height < 120:
                continue
            context_id = f"scroll_{region.region_id}"
            region.scroll_context_id = context_id
            scroll_contexts.append(
                ScrollContext(
                    scroll_context_id=context_id,
                    region_id=region.region_id,
                    scroll_type="vertical",
                    viewport_height=height,
                    viewport_width=width,
                    scroll_offset=0,
                    total_content_height=self._estimate_total_content_height(region, element_map),
                    is_virtual=region.role in {"list_panel", "message_stream"},
                )
            )
        return scroll_contexts

    def _estimate_total_content_height(
        self,
        region: Region,
        element_map: dict[str, Candidate],
    ) -> int | None:
        """Estimate content height from child element bounds."""
        rects = [
            element_map[element_id].bounds
            for element_id in region.element_ids
            if element_id in element_map and element_map[element_id].bounds is not None
        ]
        if not rects or region.bounds is None:
            return None
        return max(max(rect[3] for rect in rects) - region.bounds[1], region.bounds[3] - region.bounds[1])

    def _build_regions(
        self,
        raw_elements: list[dict[str, Any]],
        zone_structure: WindowZoneStructure | None,
        surface_type: SurfaceType,
        content_subtype: ContentAreaSubtype,
        element_region_map: dict[str, str],
        fallback_regions: list[Region] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        return self._region_builder.build_regions(
            raw_elements,
            zone_structure,
            surface_type,
            content_subtype,
            element_region_map,
            fallback_regions,
            vision_layout_regions,
            geometric_regions,
            fusion_diagnostics,
        )

    def _build_regions_from_zone_structure(
        self,
        zone_structure: WindowZoneStructure,
        content_subtype: ContentAreaSubtype,
        element_region_map: dict[str, str],
    ) -> list[Region]:
        del element_region_map
        return self._region_builder.build_regions_from_zone_structure(
            zone_structure,
            content_subtype,
        )

    def _build_fallback_region(
        self,
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        element_region_map: dict[str, str],
        fallback_regions: list[Region] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        return self._region_builder.build_fallback_region(
            raw_elements,
            content_subtype,
            element_region_map,
            fallback_regions,
            vision_layout_regions,
            geometric_regions,
            fusion_diagnostics,
        )

    def _generate_locators(
        self,
        elements: list[Candidate],
        surface_type: SurfaceType,
        has_dom_bridge: bool = False,
    ) -> list[Locator]:
        return self._locator_builder.generate_locators(
            elements,
            surface_type,
            has_dom_bridge,
        )

    def _generate_element_locators(
        self,
        element: Candidate,
        surface_type: SurfaceType,
        priority_by_kind: list[LocatorKind],
        has_dom_bridge: bool = False,
    ) -> list[Locator]:
        return self._locator_builder.generate_element_locators(
            element,
            surface_type,
            priority_by_kind,
            has_dom_bridge,
        )

    def _generate_anchors(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> list[Anchor]:
        return self._locator_builder.generate_anchors(elements, regions)

    def _attach_anchor_refs_to_locators(
        self,
        elements: list[Candidate],
        locators: list[Locator],
    ) -> None:
        self._locator_builder.attach_anchor_refs_to_locators(elements, locators)

    def _compute_anchor_stability(
        self,
        element: Candidate,
        name_counts: dict[str, int],
        area: int,
        all_areas: list[int],
    ) -> float:
        return self._locator_builder.compute_anchor_stability(
            element,
            name_counts,
            area,
            all_areas,
        )

    def _generate_relations(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> list[ElementRelation]:
        return self._locator_builder.generate_relations(elements, regions)

    def _build_snapshot_for_page_class(
        self,
        process_name: str | None,
        window_hwnd: int,
        window_title: str,
        surface_info: SurfaceInfo,
        elements: list[Candidate],
        regions: list[Region],
    ) -> InteractionCanvas:
        """构建用于 page_class 分类的 InteractionCanvas（消费实际 regions）"""
        return InteractionCanvas(
            canvas_id="temp",
            app=AppInfo(process_name=process_name),
            window=WindowInfoSnapshot(hwnd=window_hwnd, title=window_title),
            surface=surface_info,
            page=PageInfo(),
            elements=elements,
            regions=regions,
        )

    def _build_page_info(self, result: PageClassResult) -> PageInfo:
        """构建 PageInfo"""
        return PageInfo(
            page_class=result.page_class,
            class_confidence=result.class_confidence,
            fingerprint=result.fingerprint,
        )

    # -------------------------------------------------------------------------
    # Provider trace
    # -------------------------------------------------------------------------

    def _build_provider_trace(
        self,
        uia_element_count: int,
        has_dom_bridge: bool,
        visual_score: float | None,
        zone_structure: WindowZoneStructure | None,
        element_source_stats: dict[str, int] | None = None,
        locators: list | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        vision_control_groups: list[dict[str, Any]] | None = None,
        vision_interaction_hints: list[dict[str, Any]] | None = None,
        structure_evidence_score: float | None = None,
    ) -> ProviderTrace:
        return self._support_builder.build_provider_trace(
            uia_element_count=uia_element_count,
            has_dom_bridge=has_dom_bridge,
            visual_score=visual_score,
            zone_structure=zone_structure,
            element_source_stats=element_source_stats,
            locators=locators,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            vision_layout_regions=vision_layout_regions,
            vision_control_groups=vision_control_groups,
            vision_interaction_hints=vision_interaction_hints,
            structure_evidence_score=structure_evidence_score,
        )

    def _resolve_semantic_role(
        self,
        control_type: str,
        text: str,
        name: str | None,
        vision_match: dict[str, Any] | None = None,
    ) -> SemanticRole:
        return self._support_builder.resolve_semantic_role(
            control_type,
            text,
            name,
            vision_match,
        )

    def _semantic_role_from_vision_label(self, value: Any) -> SemanticRole:
        return self._support_builder.semantic_role_from_vision_label(value)

    def _infer_semantic_role(
        self, control_type: str, text: str, name: str | None
    ) -> SemanticRole:
        return self._support_builder.infer_semantic_role(
            control_type,
            text,
            name,
        )

    def _is_interactable(self, control_type: str) -> bool:
        return self._support_builder.is_interactable(control_type)
