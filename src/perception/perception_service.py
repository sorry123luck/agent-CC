"""
统一感知服务

整合元素归并、区域划分、OCR 辅助为统一的感知层。

设计原则（2026-04-05 明确）：
- GUI 文本：优先使用 UIA 的 Name/Value 属性
- OCR：仅作为渲染内容（Canvas/WebView/Game）的辅助兜底，不用于 GUI chrome
- 页面结构理解：以 UIA 元素树 + 归并 + 区域划分为核心
- VLM：内部增强 Provider，按 allow_vlm/force_vlm 条件调用，只输出候选不输出决策
"""
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from src.windows.window_enum import WindowEnumService, WindowInfoExt
from src.perception.ocr_service import OCRTextBlock, get_ocr_service
from src.perception.page_compiler_candidates import populate_boundary_candidate_artifacts
from src.perception.providers.remote_vision_provider import OmniParserRemoteVisionProvider
from src.perception.uia_client import UIAClient, UIAElementInfo
from src.perception.element_merger import ElementMerger, MergedElement
from src.perception.zone_partitioner import ZonePartitioner, ZoneInfo, WindowZoneStructure, ZoneType
from src.perception.visual_layout import VisualLayoutSegmenter
from src.perception.geometric_partitioner import (
    GeometricPartitioner,
    GeometricRegion,
    PartitionerConfig,
    PartitionDiagnostics,
)
from src.perception.semantic_fusion import (
    SemanticFusion,
    FusionConfig,
    FusionDiagnostics,
)
from src.common.config_manager import load_config

logger = logging.getLogger(__name__)


@contextmanager
def _uia_com_context() -> Iterator[None]:
    """Initialize COM for UIA on the current thread when pywin32 is available."""
    pythoncom = None
    initialized = False
    try:
        import pythoncom as _pythoncom

        pythoncom = _pythoncom
        pythoncom.CoInitialize()
        initialized = True
    except ImportError:
        pythoncom = None
    except Exception:
        pythoncom = None

    try:
        yield
    finally:
        if initialized and pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass


@dataclass
class ZonePageStructure:
    """
    统一页面结构（Phase 3 增强版）

    替代原有的 PageStructure，作为 Phase 3+ 的标准输出格式。
    """
    window_info: WindowInfoExt | None = None
    zones: list[ZoneInfo] = field(default_factory=list)
    all_elements_merged: list[MergedElement] = field(default_factory=list)
    screenshot: Image.Image | None = None
    screenshot_size: tuple[int, int] | None = None
    screenshot_provider_details: dict[str, Any] = field(default_factory=dict)
    # OCR 仅用于渲染内容区域兜底
    ocr_auxiliary_texts: list[str] = field(default_factory=list)  # OCR 辅助文本列表
    ocr_blocks: list[OCRTextBlock] = field(default_factory=list)
    vision_candidates: list[dict[str, Any]] = field(default_factory=list)
    ocr_provider_details: dict[str, Any] = field(default_factory=dict)
    visual_score: float | None = None
    vision_provider_details: dict[str, Any] = field(default_factory=dict)
    vision_layout_regions: list[dict[str, Any]] = field(default_factory=list)
    vision_control_groups: list[dict[str, Any]] = field(default_factory=list)
    vision_interaction_hints: list[dict[str, Any]] = field(default_factory=list)
    structure_evidence_score: float | None = None
    uia_provider_details: dict[str, Any] = field(default_factory=dict)
    fast_mode: bool = False


class PerceptionService:
    """
    统一感知服务

    使用流程：
    1. 获取窗口信息 + UIA 元素
    2. 归并细碎元素
    3. 划分功能区域
    4. OCR 仅作为内容区兜底
    """

    # Minimum number of local vision candidates to skip VLM call
    _VLM_MIN_LOCAL_CANDIDATES = 3
    _VLM_LOW_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._element_merger = ElementMerger()
        self._zone_partitioner = ZonePartitioner(merger=self._element_merger)
        self._vision_provider = OmniParserRemoteVisionProvider()
        from src.runtime.fallback_policy import FallbackPolicy
        self._fallback_policy = FallbackPolicy()
        # VLM provider — lazy init from config
        from src.perception.providers.vlm_provider import VLMProvider
        self._vlm_provider = VLMProvider()

    def analyze(
        self,
        hwnd: int,
        allow_vlm: bool = False,
        force_vlm: bool = False,
        fast_mode: bool = False,
        lightweight_uia: bool = False,
        screenshot_capture_mode: str = "window",
    ) -> ZonePageStructure:
        """
        分析窗口并生成统一的页面结构

        坐标系统：
        - 内部统一坐标系：窗口局部原始坐标（窗口左上角 = (0, 0)，单位 = 原始像素）
        - 截图：整窗原始截图（不再自动缩放）
        - UIA bounding_rect：屏幕绝对坐标，进入 analyze 后立即归一化为窗口局部坐标

        Args:
            hwnd: 窗口句柄

        Returns:
            ZonePageStructure 统一页面结构
        """
        import time as _time
        _t0 = _time.perf_counter()
        _sub_timing: dict[str, float] = {}

        from src.runtime.timeout_budget import TimeoutBudget, DEFAULT_OBSERVE_BUDGET_SECONDS
        budget = TimeoutBudget(total_seconds=DEFAULT_OBSERVE_BUDGET_SECONDS)

        # 1. 获取窗口信息
        enum_svc = WindowEnumService()
        all_windows = enum_svc.enumerate_all(refresh=True)
        window_info = next((w for w in all_windows if w.hwnd == hwnd), None)

        if window_info is None:
            fg = enum_svc.get_foreground_window()
            if fg and fg.hwnd == hwnd:
                window_info = fg

        # 2. 获取 UIA 元素（屏幕绝对坐标）
        _t_uia = _time.perf_counter()
        uia_provider_details: dict[str, Any] = {"provider": "uia", "success": False}
        try:
            with _uia_com_context():
                uia_client = UIAClient(hwnd)
                if fast_mode or lightweight_uia:
                    # Fast path must return quickly even for apps with slow/deep UIA
                    # trees. Full local vision/semantic enhancement runs separately.
                    uia_elements = [uia_client.get_root_element()]
                    uia_provider_details.update(
                        {
                            "success": True,
                            "mode": "root_only",
                            "reason": "fast_mode" if fast_mode else "lightweight_uia",
                            "element_count": len(uia_elements),
                        }
                    )
                elif self._should_use_bounded_uia(window_info):
                    uia_elements = uia_client.find_all_bounded(max_elements=900, timeout_seconds=8.0)
                    uia_provider_details.update(getattr(uia_client, "find_all_bounded_diagnostics", {}) or {})
                    uia_provider_details.update(
                        {
                            "success": True,
                            "mode": uia_provider_details.get("mode") or "bounded",
                            "element_count": len(uia_elements),
                        }
                    )
                else:
                    uia_elements = uia_client.find_all()
                    uia_provider_details.update(
                        {
                            "success": True,
                            "mode": "full",
                            "element_count": len(uia_elements),
                        }
                    )
        except Exception as exc:
            uia_elements = []
            uia_provider_details.update({"success": False, "error": str(exc), "element_count": 0})
        _sub_timing["uia_enum"] = _time.perf_counter() - _t_uia
        uia_provider_details["elapsed_seconds"] = _sub_timing["uia_enum"]
        budget.record("uia", _sub_timing["uia_enum"])
        from src.runtime.fallback_policy import ProviderResult as _PR
        self._fallback_policy.record_result(_PR(
            provider_id="uia", success=bool(uia_provider_details.get("success")),
            elapsed_seconds=_sub_timing["uia_enum"],
            error=str(uia_provider_details.get("error") or "") or None,
        ))

        # 2.1 过滤无效坐标元素（仍在屏幕绝对坐标下进行）
        uia_elements = self._filter_invalid_elements(uia_elements, window_info)

        # 2.2 获取整窗尺寸（用于后续映射）
        window_width, window_height = self._get_content_size(uia_elements, window_info)

        # 2.3 坐标归一化：屏幕绝对坐标 → 窗口局部原始坐标
        # 此后所有 UIA/merged/zone 坐标均为窗口局部原始坐标
        uia_elements = self._normalize_elements_to_window_coords(uia_elements, window_info)

        # 3. 归并元素（使用窗口局部坐标）
        _t_merge = _time.perf_counter()
        merged_elements = self._element_merger.merge(uia_elements)
        _sub_timing["merge"] = _time.perf_counter() - _t_merge

        # 4. 划分区域（使用窗口局部坐标）
        _t_zone = _time.perf_counter()
        zone_structure = self._zone_partitioner.partition(
            uia_elements, window_width, window_height
        )
        _sub_timing["zone_partition"] = _time.perf_counter() - _t_zone

        # 5. 获取截图（原始尺寸，不缩放）
        _t_shot = _time.perf_counter()
        screenshot = None
        screenshot_size = None
        screenshot_provider_details: dict[str, Any] = {
            "provider": "screen_region_capture" if screenshot_capture_mode == "screen_region" else "window_capture",
            "success": False,
        }
        try:
            from src.windows.screenshot_service import ScreenshotService
            screenshot_svc = ScreenshotService()
            if screenshot_capture_mode == "screen_region":
                from src.windows.window_bounds import get_capture_frame_bounds

                capture_rect = get_capture_frame_bounds(hwnd)
                screenshot = screenshot_svc.capture(mode="region", rect=capture_rect)
                screenshot_provider_details["screen_region"] = list(capture_rect)
            else:
                screenshot = screenshot_svc.capture(mode="window", target=hwnd)
            screenshot_size = screenshot.size
            screenshot_provider_details.update(
                {
                    "success": True,
                    "size": list(screenshot_size),
                }
            )
        except Exception as exc:
            screenshot_provider_details["error"] = str(exc)
        _sub_timing["screenshot"] = _time.perf_counter() - _t_shot
        screenshot_provider_details["elapsed_seconds"] = _sub_timing["screenshot"]

        # 6. OCR 兜底：对内容区截图运行 OCR（仅作为辅助，不用于 GUI chrome）
        # 截图是原图，zone bounds 是窗口局部坐标，直接用于 crop（无需缩放映射）
        ocr_auxiliary_texts: list[str] = []
        ocr_blocks: list[OCRTextBlock] = []
        ocr_provider_details: dict[str, Any] = {}
        visual_score: float | None = None
        vision_provider_details: dict[str, Any] = {}
        vision_layout_regions: list[dict[str, Any]] = []
        vision_control_groups: list[dict[str, Any]] = []
        vision_interaction_hints: list[dict[str, Any]] = []
        structure_evidence_score: float | None = None
        _t_ocr = _time.perf_counter()
        budget.allocate("ocr", 30.0)
        if budget.remaining() <= 0:
            budget.skip("ocr", "budget_exhausted")
        elif fast_mode or lightweight_uia:
            ocr_provider_details = {
                "provider": "paddleocr_bridge",
                "success": False,
                "error": "local_fast_path_deferred" if fast_mode else "lightweight_uia_vision_deferred",
            }
        elif screenshot and zone_structure.zones:
            try:
                content_zone = zone_structure.get_zone(ZoneType.CONTENT_AREA)
                if content_zone and content_zone.bounding_rect and screenshot_size:
                    l, t, r, b = content_zone.bounding_rect
                    # 直接使用窗口局部坐标作为截图 crop 坐标
                    x1 = max(0, int(l))
                    y1 = max(0, int(t))
                    x2 = min(screenshot_size[0], int(r))
                    y2 = min(screenshot_size[1], int(b))
                    if x2 > x1 and y2 > y1:
                        ocr_result = get_ocr_service().extract_with_metadata(
                            screenshot,
                            region=(x1, y1, x2, y2),
                        )
                        ocr_blocks = ocr_result.blocks
                        ocr_auxiliary_texts = [block.text for block in ocr_blocks]
                        ocr_provider_details = {
                            "provider": ocr_result.provider,
                            "success": ocr_result.success,
                            "error": ocr_result.error,
                            "elapsed_seconds": ocr_result.elapsed_seconds,
                            "used_region": ocr_result.used_region,
                            "worker_command": ocr_result.worker_command,
                            "worker_mode": ocr_result.worker_mode,
                            "worker_reused": ocr_result.worker_reused,
                            "startup_seconds": ocr_result.startup_seconds,
                            "fallback_reason": ocr_result.fallback_reason,
                            "block_count": len(ocr_result.blocks),
                        }
            except Exception:
                pass
        if screenshot and screenshot_size and not ocr_blocks and self._should_run_sparse_full_window_ocr(
            uia_elements,
            window_info,
        ):
            ocr_region, sparse_ocr_reason = self._sparse_ocr_region(screenshot_size, window_info)
            skip_reason = self._sparse_full_window_ocr_skip_reason((ocr_region[2] - ocr_region[0], ocr_region[3] - ocr_region[1]))
            if skip_reason:
                ocr_provider_details = {
                    "provider": "paddleocr_bridge",
                    "success": False,
                    "error": skip_reason,
                    "elapsed_seconds": 0.0,
                    "used_region": ocr_region,
                    "fallback_reason": sparse_ocr_reason,
                    "block_count": 0,
                }
            else:
                try:
                    ocr_result = get_ocr_service().extract_with_metadata(
                        screenshot,
                        region=ocr_region,
                    )
                    ocr_blocks = ocr_result.blocks
                    ocr_auxiliary_texts = [block.text for block in ocr_blocks]
                    ocr_provider_details = {
                        "provider": ocr_result.provider,
                        "success": ocr_result.success,
                        "error": ocr_result.error,
                        "elapsed_seconds": ocr_result.elapsed_seconds,
                        "used_region": ocr_result.used_region,
                        "worker_command": ocr_result.worker_command,
                        "worker_mode": ocr_result.worker_mode,
                        "worker_reused": ocr_result.worker_reused,
                        "startup_seconds": ocr_result.startup_seconds,
                        "fallback_reason": sparse_ocr_reason,
                        "block_count": len(ocr_result.blocks),
                    }
                except Exception:
                    pass
        _sub_timing["ocr"] = _time.perf_counter() - _t_ocr
        budget.record("ocr", _sub_timing["ocr"])

        vision_candidates: list[dict[str, Any]] = []
        _t_vision = _time.perf_counter()
        budget.allocate("omniparser", 90.0)
        _omniparser_called = False
        _op_skip, _op_reason = self._fallback_policy.should_skip("omniparser")
        if _op_skip:
            budget.skip("omniparser", _op_reason)
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": _op_reason,
                "candidate_count": 0,
            }
        elif budget.remaining() <= 0:
            budget.skip("omniparser", "budget_exhausted")
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": "budget_exhausted",
                "candidate_count": 0,
            }
        elif fast_mode:
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": "local_fast_path_deferred",
                "candidate_count": 0,
            }
        elif screenshot is not None:
            try:
                vision_result = self._vision_provider.parse_screenshot(screenshot)
                _omniparser_called = True
                vision_provider_details = {
                    "provider": vision_result.provider,
                    "success": vision_result.success,
                    "error": vision_result.error,
                    "candidate_count": len(vision_result.candidates),
                    "layout_region_count": len(vision_result.layout_regions or []),
                    "control_group_count": len(vision_result.control_groups or []),
                    "interaction_hint_count": len(vision_result.interaction_hints or []),
                }
                if vision_result.visual_score is not None:
                    visual_score = vision_result.visual_score
                if vision_result.structure_evidence_score is not None:
                    structure_evidence_score = vision_result.structure_evidence_score
                vision_candidates = [
                    {
                        "candidate_id": candidate.element_id,
                        "bbox": list(candidate.bounding_box),
                        "confidence": candidate.confidence,
                        "kind": candidate.semantic_label,
                        "text": candidate.text,
                        "icon_type": candidate.icon_type,
                        "region_role": candidate.region_role,
                        "group_id": candidate.group_id,
                        "interaction_hints": candidate.interaction_hints or {},
                        "structure_evidence_score": candidate.structure_evidence_score,
                        "attributes": candidate.attributes or {},
                        "source": vision_result.provider,
                    }
                    for candidate in vision_result.candidates
                ]
                if vision_result.layout_regions:
                    vision_layout_regions = vision_result.layout_regions
                    vision_provider_details["layout_regions"] = vision_result.layout_regions
                if vision_result.control_groups:
                    vision_control_groups = vision_result.control_groups
                    vision_provider_details["control_groups"] = vision_result.control_groups
                if vision_result.interaction_hints:
                    vision_interaction_hints = vision_result.interaction_hints
                    vision_provider_details["interaction_hints"] = vision_result.interaction_hints
                if structure_evidence_score is not None:
                    vision_provider_details["structure_evidence_score"] = structure_evidence_score
            except Exception as exc:
                _omniparser_called = True
                vision_provider_details = {
                    "provider": "omniparser",
                    "success": False,
                    "error": str(exc),
                    "candidate_count": 0,
                }
            if not vision_provider_details.get("success"):
                vision_candidates = self._collect_vision_candidates(
                    screenshot=screenshot,
                    ocr_blocks=ocr_blocks,
                )
                if vision_candidates:
                    vision_provider_details["fallback"] = "ocr_sidecar"
                    vision_provider_details["candidate_count"] = len(vision_candidates)
        _sub_timing["omniparser"] = _time.perf_counter() - _t_vision
        budget.record("omniparser", _sub_timing["omniparser"])
        if _omniparser_called:
            self._fallback_policy.record_result(_PR(
                provider_id="omniparser",
                success=bool(vision_provider_details.get("success")),
                elapsed_seconds=_sub_timing["omniparser"],
                error=str(vision_provider_details.get("error") or "") or None,
            ))
        if ocr_provider_details:
            ocr_provider_details.setdefault("wall_elapsed_seconds", _sub_timing["ocr"])
        if vision_provider_details:
            vision_provider_details["elapsed_seconds"] = _sub_timing["omniparser"]

        # Track whether local vision succeeded (for _should_call_vlm)
        vision_success = vision_provider_details.get("success", False)

        # VLM internal enhancement: call after OmniParser if conditions met
        _t_vlm_local = _time.perf_counter()
        vlm_called = False
        budget.allocate("vlm", 60.0)
        _vlm_skip, _vlm_reason = self._fallback_policy.should_skip("vlm")
        _vlm_skipped = False
        if _vlm_skip:
            budget.skip("vlm", _vlm_reason)
            _vlm_skipped = True
        elif budget.remaining() <= 0:
            budget.skip("vlm", "budget_exhausted")
            _vlm_skipped = True
        elif screenshot is not None and self._should_call_vlm(
            allow_vlm, force_vlm, vision_candidates, vision_success
        ):
            try:
                from src.perception.providers.vlm_provider import VLMCandidate
                vlm_raw = self._vlm_provider.analyze_screenshot(
                    image=screenshot,
                    window_title=window_info.title if window_info else "",
                )
                if vlm_raw:
                    vlm_called = True
                    for vc in vlm_raw:
                        vision_candidates.append({
                            "candidate_id": f"vlm_{len(vision_candidates)}",
                            "bbox": list(vc.bbox),
                            "confidence": vc.confidence,
                            "kind": vc.control_type,
                            "text": vc.text,
                            "source": "vlm",
                        })
                    vision_provider_details["vlm_candidate_count"] = len(vlm_raw)
                    logger.info("VLM added %d candidates", len(vlm_raw))
                self._fallback_policy.record_result(_PR(
                    provider_id="vlm", success=True,
                    elapsed_seconds=0.0,
                ))
            except Exception as exc:
                logger.warning("VLM call failed: %s", exc)
                vision_provider_details["vlm_error"] = str(exc)
                self._fallback_policy.record_result(_PR(
                    provider_id="vlm", success=False,
                    elapsed_seconds=0.0, error=str(exc),
                ))
        _sub_timing["vlm_local"] = _time.perf_counter() - _t_vlm_local
        budget.record("vlm", _sub_timing["vlm_local"])

        _sub_timing["total"] = _time.perf_counter() - _t0
        _sub_timing["budget"] = budget.summary()
        _sub_timing["fallback"] = self._fallback_policy.summary()
        if ocr_provider_details:
            ocr_provider_details["analysis_timing"] = dict(_sub_timing)
        if vision_provider_details:
            vision_provider_details["analysis_timing"] = dict(_sub_timing)
        import sys
        _timing_str = {k: (f"{v:.3f}s" if isinstance(v, (int, float)) else str(v)[:80]) for k, v in _sub_timing.items()}
        logger.info("PerceptionService.analyze timing: %s", _timing_str)
        print(f"PerceptionService.analyze timing: {_timing_str}", file=sys.stderr, flush=True)

        return ZonePageStructure(
            window_info=window_info,
            zones=zone_structure.zones,
            all_elements_merged=merged_elements,
            screenshot=screenshot,
            screenshot_size=screenshot_size,
            screenshot_provider_details=screenshot_provider_details,
            ocr_auxiliary_texts=ocr_auxiliary_texts,
            ocr_blocks=ocr_blocks,
            vision_candidates=vision_candidates,
            ocr_provider_details=ocr_provider_details,
            visual_score=visual_score,
            vision_provider_details=vision_provider_details,
            vision_layout_regions=vision_layout_regions,
            vision_control_groups=vision_control_groups,
            vision_interaction_hints=vision_interaction_hints,
            structure_evidence_score=structure_evidence_score,
            uia_provider_details=uia_provider_details,
            fast_mode=fast_mode,
        )

    def create_page_snapshot(
        self,
        zone_page: ZonePageStructure,
        process_name: str | None = None,
        process_path: str | None = None,
        has_dom_bridge: bool = False,
        dom_ready: bool = False,
        visual_score: float | None = None,
    ):
        """
        从 ZonePageStructure 生成完整 InteractionCanvas

        整合 ZonePartitioner 结果 + InteractionCanvasEngine 生成完整 InteractionCanvas。

        Args:
            zone_page: ZonePageStructure from analyze()
            process_name: 进程名
            process_path: 进程路径
            has_dom_bridge: 是否有 DOM bridge
            dom_ready: DOM 是否就绪
            visual_score: 视觉分数

        Returns:
            InteractionCanvas 完整页面快照
        """
        from src.perception.page_compiler import InteractionCanvasEngine
        from src.perception.zone_partitioner import WindowZoneStructure

        # 将 MergedElement 转回 dict 格式（用于 InteractionCanvasEngine），保留关键定位属性
        raw_elements: list[dict[str, Any]] = []
        for merged in zone_page.all_elements_merged:
            # 从原始 UIAElementInfo 提取 automation_id（第一个元素有则取）
            automation_id: str | None = None
            if merged.elements:
                first = merged.elements[0]
                if hasattr(first, "automation_id") and first.automation_id:
                    automation_id = first.automation_id

            raw_elements.append({
                "element_id": merged.element_id or f"elem_{len(raw_elements)}",
                "control_type": merged.control_type,
                "name": merged.name,
                "text": merged.name,
                "bounding_rect": merged.bounding_rect,
                "automation_id": automation_id,
            })

        # 从 zone_page.zones 构建 legacy zone hint。
        # Pro 主链不再把 ZonePartitioner 结果作为 InteractionCanvasEngine 的主结构真相源。
        real_zone_structure: WindowZoneStructure | None = None
        if zone_page.zones:
            # 优先使用 window_info.rect 计算整窗尺寸
            if zone_page.window_info and zone_page.window_info.rect:
                win_rect = zone_page.window_info.rect
                win_w = win_rect[2] - win_rect[0]
                win_h = win_rect[3] - win_rect[1]
            elif zone_page.screenshot_size:
                win_w, win_h = zone_page.screenshot_size
            else:
                win_w, win_h = 1920, 1080

            real_zone_structure = WindowZoneStructure(
                zones=zone_page.zones,
                window_width=win_w,
                window_height=win_h,
            )

        compiler = InteractionCanvasEngine()

        # 获取 window 信息
        window_hwnd = zone_page.window_info.hwnd if zone_page.window_info else 0
        window_title = zone_page.window_info.title if zone_page.window_info else ""
        app_cfg = load_config()
        snapshot_stage_timing: dict[str, float] = {}
        snapshot_stage_details: dict[str, Any] = {}

        # Legacy VisualLayoutSegmenter is now diagnostic-only. It is disabled by
        # default because geometric partitioning has become the active region path.
        visual_regions = None
        if zone_page.screenshot is not None and app_cfg.geometric_partitioner.run_legacy_visual_layout:
            t_legacy = time.perf_counter()
            try:
                from src.perception.page_compiler_models import VisualRegion
                raw_regions = VisualLayoutSegmenter().segment(zone_page.screenshot)
                snapshot_stage_timing["legacy_visual_layout"] = time.perf_counter() - t_legacy
                visual_regions = [
                    VisualRegion(
                        region_type=vc.role_hint,
                        bounds=vc.bounds,
                        confidence=vc.geometry_confidence,
                        actionability=vc.actionability,
                        source=vc.source,
                    )
                    for vc in raw_regions
                ]
                snapshot_stage_details["legacy_visual_layout"] = {
                    "success": True,
                    "count": len(visual_regions),
                    "elapsed_seconds": snapshot_stage_timing["legacy_visual_layout"],
                }
            except Exception:
                snapshot_stage_timing["legacy_visual_layout"] = time.perf_counter() - t_legacy
                snapshot_stage_details["legacy_visual_layout"] = {
                    "success": False,
                    "error": "legacy_visual_layout_failed",
                    "elapsed_seconds": snapshot_stage_timing["legacy_visual_layout"],
                }
                visual_regions = None
        elif zone_page.screenshot is not None:
            snapshot_stage_details["legacy_visual_layout"] = {
                "success": None,
                "skipped": True,
                "reason": "disabled_by_config",
            }

        # Layer 1: GeometricPartitioner (neutral geometry, no semantic labels)
        geometric_regions: list[GeometricRegion] | None = None
        partition_diagnostics: PartitionDiagnostics | None = None
        fusion_diagnostics: FusionDiagnostics | None = None
        if getattr(zone_page, "fast_mode", False) and zone_page.screenshot is not None:
            t_geometric = time.perf_counter()
            try:
                geometric_regions, partition_diagnostics = self._run_fast_geometric_partition(
                    zone_page.screenshot,
                )
                snapshot_stage_timing["geometric_partitioner"] = time.perf_counter() - t_geometric
                snapshot_stage_details["geometric_partitioner"] = {
                    "success": True,
                    "mode": "fast_light",
                    "count": len(geometric_regions) if geometric_regions else 0,
                    "elapsed_seconds": snapshot_stage_timing["geometric_partitioner"],
                }
            except Exception:
                snapshot_stage_timing["geometric_partitioner"] = time.perf_counter() - t_geometric
                snapshot_stage_details["geometric_partitioner"] = {
                    "success": False,
                    "mode": "fast_light",
                    "error": "fast_geometric_partitioner_failed",
                    "elapsed_seconds": snapshot_stage_timing["geometric_partitioner"],
                }
                logger.warning("Fast GeometricPartitioner failed, continuing without geometric regions", exc_info=True)
        elif getattr(zone_page, "fast_mode", False):
            snapshot_stage_details["geometric_partitioner"] = {
                "success": None,
                "skipped": True,
                "reason": "no_screenshot_in_fast_mode",
            }
        elif zone_page.screenshot is not None:
            t_geometric = time.perf_counter()
            try:
                geometric_mode = "full"
                if self._should_bound_full_geometric_partition(zone_page.screenshot):
                    geometric_regions, partition_diagnostics = self._run_bounded_full_geometric_partition(
                        zone_page.screenshot,
                    )
                    geometric_mode = "bounded_full"
                else:
                    gp_config = PartitionerConfig(
                        separator_threshold=app_cfg.geometric_partitioner.separator_threshold,
                        min_span_ratio=app_cfg.geometric_partitioner.min_span_ratio,
                        density_contrast_threshold=app_cfg.geometric_partitioner.density_contrast_threshold,
                        min_region_area_ratio=app_cfg.geometric_partitioner.min_region_area_ratio,
                        border_edge_threshold=app_cfg.geometric_partitioner.border_edge_threshold,
                        floating_min_confidence=app_cfg.geometric_partitioner.floating_min_confidence,
                        grid_min_card_count=app_cfg.geometric_partitioner.grid_min_card_count,
                        content_std_min=app_cfg.geometric_partitioner.content_std_min,
                        content_std_max=app_cfg.geometric_partitioner.content_std_max,
                        low_texture_std_max=app_cfg.geometric_partitioner.low_texture_std_max,
                    )
                    partitioner = GeometricPartitioner(config=gp_config)
                    geometric_regions = partitioner.partition(zone_page.screenshot)
                    partition_diagnostics = partitioner.diagnostics
                snapshot_stage_timing["geometric_partitioner"] = time.perf_counter() - t_geometric
                logger.info(
                    "GeometricPartitioner: %d regions, %d separators (accepted/rejected=%d/%d)",
                    len(geometric_regions) if geometric_regions else 0,
                    partition_diagnostics.accepted_separators if partition_diagnostics else 0,
                    len(partition_diagnostics.rejected_separators) if partition_diagnostics else 0,
                )
                snapshot_stage_details["geometric_partitioner"] = {
                    "success": True,
                    "mode": geometric_mode,
                    "count": len(geometric_regions) if geometric_regions else 0,
                    "elapsed_seconds": snapshot_stage_timing["geometric_partitioner"],
                }

                # Layer 2: SemanticFusion (optional, non-blocking)
                t_fusion = time.perf_counter()
                try:
                    fusion_cfg = FusionConfig(
                        uia_input_confidence=app_cfg.semantic_fusion.uia_input_confidence,
                        action_bar_min_buttons=app_cfg.semantic_fusion.action_bar_min_buttons,
                        action_bar_band_ratio=app_cfg.semantic_fusion.action_bar_band_ratio,
                        navigation_min_items=app_cfg.semantic_fusion.navigation_min_items,
                        navigation_side_band_ratio=app_cfg.semantic_fusion.navigation_side_band_ratio,
                        content_min_area_ratio=app_cfg.semantic_fusion.content_min_area_ratio,
                        unknown_threshold=app_cfg.semantic_fusion.unknown_threshold,
                    )
                    fusion = SemanticFusion(config=fusion_cfg)
                    img_w, img_h = zone_page.screenshot.size
                    fusion_evidence = fusion.collect_evidence(
                        geometric_regions=geometric_regions,
                        raw_elements=raw_elements,
                        ocr_blocks=[
                            {"text": block.text, "bbox": block.bbox, "confidence": block.confidence}
                            for block in zone_page.ocr_blocks
                        ],
                        vision_candidates=zone_page.vision_candidates,
                        image_width=img_w,
                        image_height=img_h,
                    )
                    fusion_results = fusion.fuse(fusion_evidence)
                    fusion_diagnostics = fusion.build_diagnostics(fusion_results)
                    snapshot_stage_timing["semantic_fusion"] = time.perf_counter() - t_fusion
                    logger.info(
                        "SemanticFusion: %d labeled / %d unknown",
                        fusion_diagnostics.labeled_count,
                        fusion_diagnostics.unknown_count,
                    )
                    snapshot_stage_details["semantic_fusion"] = {
                        "success": True,
                        "labeled_count": fusion_diagnostics.labeled_count,
                        "unknown_count": fusion_diagnostics.unknown_count,
                        "elapsed_seconds": snapshot_stage_timing["semantic_fusion"],
                    }
                except Exception:
                    snapshot_stage_timing["semantic_fusion"] = time.perf_counter() - t_fusion
                    snapshot_stage_details["semantic_fusion"] = {
                        "success": False,
                        "error": "semantic_fusion_failed",
                        "elapsed_seconds": snapshot_stage_timing["semantic_fusion"],
                    }
                    logger.warning("SemanticFusion failed, continuing without fusion", exc_info=True)
            except Exception:
                snapshot_stage_timing["geometric_partitioner"] = time.perf_counter() - t_geometric
                snapshot_stage_details["geometric_partitioner"] = {
                    "success": False,
                    "error": "geometric_partitioner_failed",
                    "elapsed_seconds": snapshot_stage_timing["geometric_partitioner"],
                }
                logger.warning("GeometricPartitioner failed, continuing without geometric regions", exc_info=True)

        t_compile = time.perf_counter()
        snapshot = compiler.compile(
            process_name=process_name,
            process_path=process_path,
            window_hwnd=window_hwnd,
            window_title=window_title,
            uia_element_count=len(raw_elements),
            uia_control_types=self._count_control_types(raw_elements),
            has_dom_bridge=has_dom_bridge,
            dom_ready=dom_ready,
            visual_score=visual_score if visual_score is not None else zone_page.visual_score,
            raw_elements=raw_elements,
            ocr_blocks=[
                {
                    "text": block.text,
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                }
                for block in zone_page.ocr_blocks
            ],
            vision_candidates=zone_page.vision_candidates,
            vision_layout_regions=zone_page.vision_layout_regions,
            vision_control_groups=zone_page.vision_control_groups,
            vision_interaction_hints=zone_page.vision_interaction_hints,
            structure_evidence_score=zone_page.structure_evidence_score,
            zone_structure=None,
            allow_legacy_zone_reconstruction=False,
            visual_regions=visual_regions,
            geometric_regions=geometric_regions,
            fusion_diagnostics=fusion_diagnostics.to_dict() if fusion_diagnostics is not None else None,
        )
        snapshot_stage_timing["page_compile"] = time.perf_counter() - t_compile
        snapshot_stage_details["page_compile"] = {
            "success": True,
            "elapsed_seconds": snapshot_stage_timing["page_compile"],
        }
        snapshot.artifacts["snapshot_build_timing"] = snapshot_stage_timing
        snapshot.provider_trace.provider_details["snapshot_build_timing"] = snapshot_stage_details

        if real_zone_structure is not None:
            legacy_zone_hints = {
                "used_as_hint": True,
                "zone_count": len(real_zone_structure.zones),
                "zones": [
                    {
                        "zone_type": zone.zone_type.value,
                        "bounds": zone.bounding_rect,
                        "element_count": len(zone.elements),
                    }
                    for zone in real_zone_structure.zones
                ],
            }
            snapshot.artifacts["legacy_zone_hints"] = legacy_zone_hints
            snapshot.provider_trace.provider_details["legacy_zone_hints"] = legacy_zone_hints

        if zone_page.ocr_provider_details:
            snapshot.artifacts["ocr_provider"] = zone_page.ocr_provider_details
            snapshot.provider_trace.provider_details["ocr_bridge"] = zone_page.ocr_provider_details
        if zone_page.uia_provider_details:
            snapshot.artifacts["uia_provider"] = zone_page.uia_provider_details
            snapshot.provider_trace.provider_details["uia_provider"] = zone_page.uia_provider_details
        if zone_page.screenshot_provider_details:
            snapshot.artifacts["screenshot_provider"] = zone_page.screenshot_provider_details
            snapshot.provider_trace.provider_details["screenshot_provider"] = zone_page.screenshot_provider_details
        if zone_page.ocr_blocks:
            snapshot.artifacts["ocr_blocks"] = [
                {
                    "text": block.text,
                    "bbox": list(block.bbox),
                    "confidence": block.confidence,
                }
                for block in zone_page.ocr_blocks
            ]
        if zone_page.vision_candidates:
            vision_source = str(
                zone_page.vision_provider_details.get("provider")
                or "vision"
            )
            snapshot.artifacts["vision_candidates"] = [
                {
                    **candidate,
                    "source": candidate.get("source") or vision_source,
                }
                for candidate in zone_page.vision_candidates
            ]
            snapshot.provider_trace.provider_details["vision_candidates"] = {
                "count": len(zone_page.vision_candidates),
            }
        if zone_page.vision_layout_regions:
            snapshot.artifacts["vision_layout_regions"] = zone_page.vision_layout_regions
            snapshot.provider_trace.provider_details["vision_layout_regions"] = {
                "count": len(zone_page.vision_layout_regions),
            }
        if zone_page.vision_control_groups:
            snapshot.artifacts["vision_control_groups"] = zone_page.vision_control_groups
            snapshot.provider_trace.provider_details["vision_control_groups"] = {
                "count": len(zone_page.vision_control_groups),
            }
        if zone_page.vision_interaction_hints:
            snapshot.artifacts["vision_interaction_hints"] = zone_page.vision_interaction_hints
            snapshot.provider_trace.provider_details["vision_interaction_hints"] = {
                "count": len(zone_page.vision_interaction_hints),
            }
        if zone_page.structure_evidence_score is not None:
            snapshot.artifacts["structure_evidence_score"] = zone_page.structure_evidence_score
            snapshot.provider_trace.provider_details["structure_evidence_score"] = (
                zone_page.structure_evidence_score
            )
        if zone_page.vision_provider_details:
            snapshot.artifacts["vision_provider"] = zone_page.vision_provider_details
            snapshot.provider_trace.provider_details["vision_provider"] = zone_page.vision_provider_details

        # Track VLM provider status for providers_used / providers_failed
        vlm_details = zone_page.vision_provider_details or {}
        if vlm_details.get("vlm_candidate_count"):
            snapshot.provider_trace.vlm_used = True
            snapshot.provider_trace.provider_details["vlm"] = {
                "success": True,
                "candidate_count": vlm_details["vlm_candidate_count"],
            }
        elif vlm_details.get("vlm_error"):
            snapshot.provider_trace.provider_details["vlm"] = {
                "success": False,
                "error": vlm_details["vlm_error"],
            }

        # Re-derive providers_used / providers_failed after VLM status is set
        # (compiler.compile() derives them before VLM runs, so we must update)
        providers_used = []
        providers_failed = []
        for flag, name in [
            ("uia_used", "uia"), ("ocr_used", "ocr"),
            ("vision_used", "vision"), ("vlm_used", "vlm"),
            ("dom_used", "dom"),
        ]:
            if getattr(snapshot.provider_trace, flag, False):
                providers_used.append(name)
        for name, detail in snapshot.provider_trace.provider_details.items():
            if isinstance(detail, dict) and detail.get("success") is False:
                if name not in providers_used:
                    providers_failed.append(name)
        snapshot.providers_used = providers_used
        snapshot.providers_failed = providers_failed

        # Phase 1: write geometric_partitioner + semantic_fusion artifacts (parallel with visual_regions)
        if geometric_regions is not None:
            geometric_region_dicts = [gr.to_dict() for gr in geometric_regions]
            snapshot.artifacts["geometric_regions"] = geometric_region_dicts
            snapshot.provider_trace.provider_details["geometric_partitioner"] = {
                "success": True,
                "mode": snapshot_stage_details.get("geometric_partitioner", {}).get(
                    "mode",
                    "fast_light" if getattr(zone_page, "fast_mode", False) else "full",
                ),
                "count": len(geometric_region_dicts),
                "elapsed_seconds": snapshot_stage_timing.get("geometric_partitioner"),
            }
            if partition_diagnostics is not None:
                snapshot.artifacts["partition_diagnostics"] = partition_diagnostics.to_dict()
        if fusion_diagnostics is not None:
            snapshot.artifacts["fusion_diagnostics"] = fusion_diagnostics.to_dict()

        # Phase R2: StructuredRegion overlay (read-only, diagnostics only)
        try:
            from src.perception.structured_region_builder import StructuredRegionOverlayBuilder
            overlay_builder = StructuredRegionOverlayBuilder()
            fusion_diag_dict = fusion_diagnostics.to_dict() if fusion_diagnostics is not None else None
            win_w, win_h = 0, 0
            if snapshot.window and snapshot.window.rect_client:
                rc = snapshot.window.rect_client
                win_w = rc[2] - rc[0]
                win_h = rc[3] - rc[1]
            if win_w <= 0 and zone_page.screenshot_size:
                win_w, win_h = zone_page.screenshot_size
            structured_overlay = overlay_builder.build(
                geometric_regions=geometric_regions,
                fusion_diagnostics=fusion_diag_dict,
                raw_elements=raw_elements if raw_elements else None,
                ocr_blocks=[{"text": b.text, "bbox": b.bbox, "confidence": b.confidence} for b in zone_page.ocr_blocks] if zone_page.ocr_blocks else None,
                vision_candidates=zone_page.vision_candidates if zone_page.vision_candidates else None,
                window_width=win_w,
                window_height=win_h,
                screenshot=zone_page.screenshot,
            )
            snapshot.artifacts["structured_region_overlay"] = structured_overlay.to_dict()
        except Exception:
            logger.debug("StructuredRegion overlay failed, continuing without it", exc_info=True)

        if zone_page.screenshot_size is not None:
            snapshot.artifacts["screenshot_size"] = list(zone_page.screenshot_size)

        try:
            from src.perception.perception_quality import PerceptionQualityEvaluator
            from src.perception.roi_selection import RoiSelectionPlanner
            from src.perception.visual_pattern import VisualPatternClassifier

            visual_pattern = VisualPatternClassifier().classify(snapshot)
            self._synthesize_chat_input_candidate(snapshot, mode=visual_pattern.mode)
            perception_quality = PerceptionQualityEvaluator().evaluate(snapshot)
            roi_selection_plan = RoiSelectionPlanner().plan(snapshot)

            snapshot.artifacts["visual_pattern"] = visual_pattern.to_dict()
            snapshot.artifacts["perception_quality"] = perception_quality.to_dict()
            snapshot.artifacts["roi_selection_plan"] = roi_selection_plan.to_dict()
            snapshot.provider_trace.provider_details["visual_pattern"] = visual_pattern.to_dict()
            snapshot.provider_trace.provider_details["perception_quality"] = perception_quality.to_dict()
            snapshot.provider_trace.provider_details["roi_selection_plan"] = {
                "mode": roi_selection_plan.mode,
                "roi_count": len(roi_selection_plan.rois),
                "skipped_reason": roi_selection_plan.skipped_reason,
            }
        except Exception:
            logger.warning("Perception quality/ROI planning failed", exc_info=True)

        populate_boundary_candidate_artifacts(snapshot)

        return snapshot

    def _synthesize_chat_input_candidate(self, snapshot: Any, *, mode: str) -> None:
        """Add a low-trust chat input candidate when local providers only expose toolbar icons."""
        if mode not in {"collaboration_inbox", "chat_workspace", "chat_document"}:
            return
        from src.perception.page_compiler_models import Candidate, SemanticRole

        artifacts = getattr(snapshot, "artifacts", None) or {}
        width, height = self._infer_snapshot_size(snapshot)
        if width <= 0 or height <= 0:
            return
        if any(
            self._is_review_chat_input_candidate(element)
            for element in list(getattr(snapshot, "elements", []) or [])
        ):
            return
        if mode == "collaboration_inbox":
            bottom_icons = [
                element for element in list(getattr(snapshot, "elements", []) or [])
                if str(getattr(element, "control_type", "") or "").strip().lower() == "icon"
                and getattr(element, "bounds", None)
                and int(element.bounds[1]) >= int(height * 0.70)
                and int(element.bounds[0]) >= int(width * 0.45)
            ]
            if len(bottom_icons) < 2:
                return
            left = int(width * 0.50)
            top = max(0, height - 110)
            right = max(left + 160, width - 28)
            bottom = max(top + 48, height - 30)
            element_id = "synthetic_collaboration_composer_input"
            evidence = "collaboration detail pane has bottom toolbar icons but no input candidate"
        else:
            bottom_controls = self._chat_bottom_control_bounds(snapshot, width=width, height=height)
            if len(bottom_controls) < 3:
                return
            right_control_left = min(
                (bounds[0] for bounds in bottom_controls if bounds[0] >= int(width * 0.68)),
                default=0,
            )
            if right_control_left <= 0:
                return
            left_toolbar_controls = [
                bounds
                for bounds in bottom_controls
                if ((bounds[0] + bounds[2]) // 2) <= int(width * 0.58)
            ]
            toolbar_left = min((bounds[0] for bounds in left_toolbar_controls), default=int(width * 0.30))
            icon_row_top = min((bounds[1] for bounds in bottom_controls), default=height - 32)
            left = max(int(width * 0.30), toolbar_left - 8)
            top = max(0, height - max(110, int(height * 0.16)))
            right = min(width - 20, right_control_left - 8)
            bottom = min(height - 18, icon_row_top - 6)
            if bottom <= top + 32:
                bottom = max(top + 48, height - 18)
            if right <= left + 120:
                return
            element_id = "synthetic_chat_composer_input"
            evidence = "chat composer has bottom toolbar/send controls but no input candidate"
        if any(getattr(element, "element_id", "") == element_id for element in getattr(snapshot, "elements", []) or []):
            return
        candidate = Candidate(
            element_id=element_id,
            region_id=None,
            semantic_role=SemanticRole.MESSAGE_INPUT,
            control_type="SyntheticInputControl",
            bounds=(left, top, min(width, right), min(height, bottom)),
            text="",
            name=None,
            interactable=True,
            confidence=0.35,
            provider_sources=[],
            suggest_confirm=True,
            visual_type="input",
            refine_status="uncertain",
            role_label="消息输入区候选",
            role_confidence=0.35,
            role_source="geometry_fallback",
            role_evidence=[evidence],
            attributes={
                "synthetic_source": "geometry_fallback",
                "candidate_kind": "message_input_candidate",
                "actionability": "review",
                "safe_to_type": False,
                "needs_manual_label": True,
            },
        )
        snapshot.elements.append(candidate)

    def _is_review_chat_input_candidate(self, element: Any) -> bool:
        attrs = getattr(element, "attributes", None) or {}
        return (
            getattr(element, "element_id", "") == "synthetic_chat_composer_input"
            or getattr(element, "element_id", "") == "synthetic_collaboration_composer_input"
            or (
                attrs.get("candidate_kind") == "message_input_candidate"
                and attrs.get("actionability") == "review"
            )
        )

    def _bounds_in_chat_composer_band(self, bounds: Any, *, width: int, height: int) -> bool:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return False
        left, top, right, bottom = (int(value) for value in bounds)
        if right <= left or bottom <= top:
            return False
        return top >= int(height * 0.60) and left >= int(width * 0.20)

    def _infer_snapshot_size(self, snapshot: Any) -> tuple[int, int]:
        """Infer canvas pixel size when capture metadata is missing from live detail."""
        artifacts = getattr(snapshot, "artifacts", None) or {}
        size = artifacts.get("screenshot_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])

        window = getattr(snapshot, "window", None)
        rect_client = getattr(window, "rect_client", None) if window else None
        if isinstance(rect_client, (list, tuple)) and len(rect_client) == 4:
            width = int(rect_client[2]) - int(rect_client[0])
            height = int(rect_client[3]) - int(rect_client[1])
            if width > 0 and height > 0:
                return width, height

        max_right = 0
        max_bottom = 0

        def collect(bounds: Any) -> None:
            nonlocal max_right, max_bottom
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                return
            left, top, right, bottom = (int(value) for value in bounds)
            if right <= left or bottom <= top:
                return
            max_right = max(max_right, right)
            max_bottom = max(max_bottom, bottom)

        for element in list(getattr(snapshot, "elements", []) or []):
            collect(getattr(element, "bounds", None))
        for candidate in list(artifacts.get("vision_candidates") or []):
            collect(candidate.get("bbox") or candidate.get("bounds"))

        if max_right > 0 and max_bottom > 0:
            return max_right + 24, max_bottom + 18
        return 0, 0

    def _chat_bottom_control_bounds(self, snapshot: Any, *, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Collect bottom composer controls from elements and vision diagnostics."""
        controls: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()

        def add(bounds: Any) -> None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                return
            rect = tuple(int(value) for value in bounds)
            left, top, right, bottom = rect
            if right <= left or bottom <= top:
                return
            if top < int(height * 0.66) or left < int(width * 0.20):
                return
            if rect in seen:
                return
            seen.add(rect)
            controls.append(rect)

        for element in list(getattr(snapshot, "elements", []) or []):
            add(getattr(element, "bounds", None))
        artifacts = getattr(snapshot, "artifacts", None) or {}
        for candidate in list(artifacts.get("vision_candidates") or []):
            add(candidate.get("bbox") or candidate.get("bounds"))
        controls.sort(key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))
        return controls

    def _run_fast_geometric_partition(
        self,
        screenshot: Image.Image,
        *,
        max_edge: int = 384,
    ) -> tuple[list[GeometricRegion], PartitionDiagnostics]:
        """Run a bounded-cost geometric pass for fast observe diagnostics."""
        original_w, original_h = screenshot.size
        current_max = max(original_w, original_h)
        scale = 1.0
        image = screenshot
        if current_max > max_edge:
            scale = max_edge / current_max
            image = screenshot.resize(
                (
                    max(1, int(round(original_w * scale))),
                    max(1, int(round(original_h * scale))),
                ),
                Image.Resampling.BILINEAR,
            )

        partitioner = GeometricPartitioner(config=PartitionerConfig(
            separator_threshold=8.0,
            min_span_ratio=0.45,
            density_contrast_threshold=0.08,
            min_region_area_ratio=0.01,
            border_edge_threshold=10,
            floating_min_confidence=0.70,
            grid_min_card_count=4,
        ))
        regions = partitioner.partition(image)
        diagnostics = partitioner.diagnostics
        setattr(diagnostics, "fast_mode", True)
        setattr(diagnostics, "scale", round(scale, 4))
        if scale == 1.0:
            return regions, diagnostics

        scaled_regions = [
            GeometricRegion(
                region_id=region.region_id,
                bounds=(
                    max(0, min(original_w, int(round(region.bounds[0] / scale)))),
                    max(0, min(original_h, int(round(region.bounds[1] / scale)))),
                    max(0, min(original_w, int(round(region.bounds[2] / scale)))),
                    max(0, min(original_h, int(round(region.bounds[3] / scale)))),
                ),
                boundary_evidence=list(region.boundary_evidence),
                density_profile=region.density_profile,
                geometry_confidence=region.geometry_confidence,
                parent_region_id=region.parent_region_id,
            )
            for region in regions
        ]
        return scaled_regions, diagnostics

    def _should_bound_full_geometric_partition(self, screenshot: Image.Image) -> bool:
        width, height = screenshot.size
        return max(width, height) > 1024 or (width * height) > 900_000

    def _run_bounded_full_geometric_partition(
        self,
        screenshot: Image.Image,
    ) -> tuple[list[GeometricRegion], PartitionDiagnostics]:
        regions, diagnostics = self._run_fast_geometric_partition(screenshot, max_edge=640)
        setattr(diagnostics, "bounded_full", True)
        setattr(diagnostics, "fast_mode", False)
        return regions, diagnostics

    def _should_call_vlm(
        self,
        allow_vlm: bool,
        force_vlm: bool,
        vision_candidates: list[dict[str, Any]],
        vision_success: bool,
    ) -> bool:
        """Determine whether to call VLM provider. Pure function, independently testable.

        Conditions:
        1. VLM provider must be available (configured + api_key present)
        2. force_vlm=True → always call
        3. allow_vlm=False → never call
        4. allow_vlm=True + local provider failed → call
        5. allow_vlm=True + few local candidates (< _VLM_MIN_LOCAL_CANDIDATES) → call
        6. allow_vlm=True + local max confidence < threshold → call
        """
        if not self._vlm_provider.available:
            return False
        if force_vlm:
            return True
        if not allow_vlm:
            return False
        if not vision_success:
            return True
        if len(vision_candidates) < self._VLM_MIN_LOCAL_CANDIDATES:
            return True
        max_conf = max((c.get("confidence", 0.0) for c in vision_candidates), default=0.0)
        if max_conf < self._VLM_LOW_CONFIDENCE_THRESHOLD:
            return True
        return False

    def _should_use_bounded_uia(self, window_info: WindowInfoExt | None) -> bool:
        """Use bounded UIA traversal for apps with known deep accessibility trees."""
        process_name = str(getattr(window_info, "process_name", "") or "").lower()
        title = str(getattr(window_info, "title", "") or "").lower()
        return (
            process_name in {"chrome.exe", "code.exe", "msedge.exe"}
            or "visual studio code" in title
            or "chrome" in title
        )

    def _should_run_sparse_full_window_ocr(
        self,
        uia_elements: list[UIAElementInfo],
        window_info: WindowInfoExt | None = None,
    ) -> bool:
        """Use OCR over the full screenshot when sparse UIA produced no text blocks."""
        if len(uia_elements) <= 3:
            return True
        process_name = str(getattr(window_info, "process_name", "") or "").lower()
        title = str(getattr(window_info, "title", "") or "").lower()
        uia_blob = " ".join(
            [
                str(getattr(element, "name", "") or "")
                for element in uia_elements[:12]
            ]
        ).lower()
        self_drawn_sparse = (
            process_name == "qq.exe"
            or process_name.startswith("voicemeeter")
            or title == "qq"
            or "voicemeeter" in title
            or "qq" in uia_blob
            or "voicemeeter" in uia_blob
        )
        return self_drawn_sparse and len(uia_elements) <= 12

    def _sparse_full_window_ocr_skip_reason(self, screenshot_size: tuple[int, int] | None) -> str | None:
        if not screenshot_size:
            return None
        width, height = int(screenshot_size[0]), int(screenshot_size[1])
        if width <= 0 or height <= 0:
            return "sparse_full_window_ocr_skipped_invalid_screenshot_size"
        if width > 2400 and (width / height) >= 6.0:
            return "sparse_full_window_ocr_skipped_ultrawide_budget"
        return None

    def _sparse_ocr_region(
        self,
        screenshot_size: tuple[int, int],
        window_info: WindowInfoExt | None = None,
    ) -> tuple[tuple[int, int, int, int], str]:
        width, height = int(screenshot_size[0]), int(screenshot_size[1])
        process_name = str(getattr(window_info, "process_name", "") or "").lower()
        title = str(getattr(window_info, "title", "") or "").lower()
        if process_name.startswith("voicemeeter") or "voicemeeter" in title:
            return (0, 0, width, min(height, int(height * 0.16))), "sparse_uia_top_band"
        return (0, 0, width, height), "sparse_uia_full_window"

    def _collect_vision_candidates(
        self,
        screenshot: Image.Image | None,
        ocr_blocks: list[OCRTextBlock],
    ) -> list[dict[str, Any]]:
        """Build minimal OCR-backed vision candidates until OmniParser provider is connected."""
        if screenshot is None:
            return []
        candidates: list[dict[str, Any]] = []
        for index, block in enumerate(ocr_blocks):
            candidates.append(
                {
                    "candidate_id": f"vision_ocr_{index}",
                    "kind": "ocr_text_candidate",
                    "bbox": list(block.bbox),
                    "text": block.text,
                    "confidence": block.confidence,
                    "source": "paddleocr_bridge",
                }
            )
        return candidates

    def _count_control_types(self, elements: list[dict[str, Any]]) -> list[str]:
        """统计控件类型"""
        from collections import Counter
        types = [e.get("control_type", "Unknown") for e in elements]
        return [ct for ct, _ in Counter(types).most_common(20)]

    def _normalize_elements_to_window_coords(
        self,
        elements: list[UIAElementInfo],
        window_info: WindowInfoExt | None,
    ) -> list[UIAElementInfo]:
        """
        将 UIA 元素的屏幕绝对坐标转换为窗口局部原始坐标

        输入：UIA bounding_rect（屏幕绝对坐标）
        输出：窗口局部原始坐标（窗口左上角 = (0, 0)）

        转换公式：
        local_rect = (
            left - window_rect.left,
            top - window_rect.top,
            right - window_rect.left,
            bottom - window_rect.top,
        )

        归一化后校验：
        - right > left, bottom > top（正面积）
        - 不允许大面积负坐标
        - 不允许超出窗口尺寸太多
        """
        if not window_info or not window_info.rect:
            return elements  # 无法归一化，原样返回

        win_l, win_t, win_r, win_b = window_info.rect
        win_w = win_r - win_l
        win_h = win_b - win_t
        # 允许元素超出窗口一定范围（边缘元素、阴影等）
        margin = max(win_w, win_h) * 0.5

        result: list[UIAElementInfo] = []
        for elem in elements:
            if not elem.bounding_rect:
                result.append(elem)
                continue

            l, t, r, b = elem.bounding_rect
            local = (l - win_l, t - win_t, r - win_l, b - win_t)
            ll, lt, lr, lb = local

            # 校验：正面积
            if lr <= ll or lb <= lt:
                continue  # 丢弃零面积或反转元素

            # 校验：不允许大面积负坐标
            if ll < -margin or lt < -margin:
                continue

            # 校验：不允许超出窗口太多
            if lr > win_w + margin or lb > win_h + margin:
                continue

            # Elements with no visible intersection with the screenshot window
            # should not enter the final canvas. Partially out-of-bounds elements
            # are clipped to keep query/action coordinates aligned to screenshot pixels.
            if lr <= 0 or lb <= 0 or ll >= win_w or lt >= win_h:
                continue
            clipped = (
                max(0, int(ll)),
                max(0, int(lt)),
                min(win_w, int(lr)),
                min(win_h, int(lb)),
            )
            cl, ct, cr, cb = clipped
            if cr <= cl or cb <= ct:
                continue

            # 校验：宽高合理（至少 1px，不超过窗口 2 倍）
            w = cr - cl
            h = cb - ct
            if w < 1 or h < 1:
                continue
            if w > win_w * 2 or h > win_h * 2:
                continue

            normalized = UIAElementInfo(
                name=elem.name,
                automation_id=elem.automation_id,
                control_type=elem.control_type,
                bounding_rect=clipped,
                is_enabled=elem.is_enabled,
                handle=elem.handle,
                element_id=elem.element_id,
            )
            result.append(normalized)

        from src.common.logger import get_logger
        logger = get_logger(__name__)
        if len(result) < len(elements):
            logger.debug(
                f"坐标归一化：过滤 {len(elements) - len(result)} 个异常坐标元素，"
                f"保留 {len(result)}/{len(elements)} 个"
            )

        return result

    def _get_content_size(
        self,
        uia_elements: list[UIAElementInfo],
        window_info: WindowInfoExt | None,
    ) -> tuple[int, int]:
        """
        获取整窗截图映射尺寸（用于 ZonePartitioner）。

        规则：优先使用 window_info.rect 的整窗尺寸。
        不再猜测 PaneControl 内容区，因为那是"语义尺寸"，不是"截图映射尺寸"。
        """
        if window_info and window_info.rect:
            rect = window_info.rect
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            if w > 100 and h > 100:
                return w, h

        # 回退：取归一化后根元素尺寸
        if uia_elements:
            for elem in uia_elements[1:]:
                if elem.bounding_rect:
                    l, t, r, b = elem.bounding_rect
                    if l == 0 and t == 0:
                        if 50 < (r - l) < 10000 and 50 < (b - t) < 10000:
                            return (r - l), (b - t)

        return 1920, 1080

    def get_zone_by_type(
        self, zone_structure: WindowZoneStructure, zone_type: ZoneType
    ) -> ZoneInfo | None:
        """获取指定类型的区域"""
        return zone_structure.get_zone(zone_type)

    def get_content_zone_elements(
        self, zone_structure: WindowZoneStructure
    ) -> list[MergedElement]:
        """获取内容区域的元素（用于 OCR 兜底）"""
        content_zone = zone_structure.get_zone(ZoneType.CONTENT_AREA)
        if content_zone:
            return content_zone.elements
        return []

    def get_merge_stats(
        self, original: list[UIAElementInfo], merged: list[MergedElement]
    ) -> dict:
        """获取归并统计"""
        return self._element_merger.get_merge_stats(original, merged)

    def _filter_invalid_elements(
        self,
        elements: list[UIAElementInfo],
        window_info: WindowInfoExt | None,
    ) -> list[UIAElementInfo]:
        """
        过滤掉坐标严重越界的无效 UIA 元素

        WebView / iframe 等子元素可能使用不同的坐标原点，导致 bounding_rect
        报告为类似 (-4958, -10390) 或 (16777592, 16772258) 的严重越界值。

        过滤逻辑：
        - 元素 bounding_rect 的任意一边超出窗口范围 ±TOLERANCE 则过滤
        - TOLERANCE = 窗口尺寸 * 1.5，超出此范围视为坐标系统错误
        """
        if not elements:
            return elements

        if window_info and window_info.rect:
            wx0, wy0, wx1, wy1 = window_info.rect
            ww = wx1 - wx0
            wh = wy1 - wy0
            # 扩展窗口范围作为容差（防止边缘元素被误删）
            margin = int(max(ww, wh) * 1.5)
            valid_x0 = wx0 - margin
            valid_x1 = wx1 + margin
            valid_y0 = wy0 - margin
            valid_y1 = wy1 + margin
        else:
            # 无窗口信息时，使用默认容差
            valid_x0, valid_y0 = -2000, -2000
            valid_x1, valid_y1 = 3000, 3000

        def is_valid(elem: UIAElementInfo) -> bool:
            if not elem.bounding_rect:
                return True  # 无坐标的元素保留（后续处理）
            l, t, r, b = elem.bounding_rect
            # 过滤：坐标完全超出容差范围
            if r < valid_x0 or l > valid_x1 or b < valid_y0 or t > valid_y1:
                return False
            # 过滤：坐标值明显错误（如接近 int32 上限）
            if any(abs(v) > 100000 for v in (l, t, r, b)):
                return False
            return True

        filtered = [e for e in elements if is_valid(e)]
        if len(filtered) < len(elements):
            from src.common.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(
                f"过滤了 {len(elements) - len(filtered)} 个无效坐标元素，"
                f"保留 {len(filtered)}/{len(elements)} 个"
            )
        return filtered
