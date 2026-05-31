"""
感知融合服务

融合 UIA 元素树和 OCR 识别结果，输出统一页面结构

重要说明（2026-04-05）：
当前融合链路已通，但 OCR 质量在深色 GUI 场景下严重不合格。
GUI 界面文本应优先使用 UIA 的 Name/Value 属性获取，而非 OCR。
OCR 仅作为渲染内容（WebView/Canvas/Game）的辅助识别兜底。
GUI 结构解析（元素归并、区域划分）需要单独规划，不在本模块范围内。
详细评估见 docs/perception_evaluation.md
"""
from dataclasses import dataclass, field
from typing import Any

from src.windows.window_enum import WindowInfoExt
from src.perception.uia_client import UIAElementInfo, UIAClient
from src.perception.ocr_service import OCRTextBlock, OCRService


@dataclass
class FusedElement:
    """融合后的元素"""
    uia_element: UIAElementInfo | None  # 对应的 UIA 元素（如果没有则为 None）
    ocr_text: str | None  # OCR 识别的文本（如果没有则为 None）
    ocr_bbox: tuple[int, int, int, int] | None  # OCR 文本的边界框
    ocr_confidence: float | None  # OCR 置信度
    fused_bbox: tuple[int, int, int, int]  # 融合后的边界框
    sources: list[str] = field(default_factory=list)  # 来源列表 ["uia", "ocr"]


@dataclass
class PageStructure:
    """统一页面结构"""
    window_info: WindowInfoExt | None  # 窗口基本信息
    uia_elements: list[UIAElementInfo] = field(default_factory=list)  # UIA 元素列表
    ocr_blocks: list[OCRTextBlock] = field(default_factory=list)  # OCR 文本块列表
    fused_elements: list[FusedElement] = field(default_factory=list)  # 融合后的元素列表
    screenshot_size: tuple[int, int] | None = None  # 截图尺寸 (width, height)


class PerceptionFusion:
    """感知融合服务"""

    def __init__(self) -> None:
        self._ocr_service: OCRService | None = None

    def analyze(self, hwnd: int) -> PageStructure:
        """
        分析窗口并生成统一页面结构

        Args:
            hwnd: 窗口句柄

        Returns:
            PageStructure 统一页面结构
        """
        from src.windows.window_enum import WindowEnumService

        # 获取窗口信息
        enum_svc = WindowEnumService()
        all_windows = enum_svc.enumerate_all(refresh=True)
        window_info = next((w for w in all_windows if w.hwnd == hwnd), None)

        if window_info is None:
            # 尝试使用前台窗口信息
            fg = enum_svc.get_foreground_window()
            if fg and fg.hwnd == hwnd:
                window_info = fg

        # 获取 UIA 元素
        try:
            uia_client = UIAClient(hwnd)
            uia_elements = uia_client.find_all()
        except Exception:
            uia_elements = []

        # 获取 OCR 结果
        screenshot_size: tuple[int, int] | None = None
        window_content_size: tuple[int, int] | None = None
        try:
            from src.windows.screenshot_service import ScreenshotService
            screenshot_svc = ScreenshotService()
            img = screenshot_svc.capture(mode="window", target=hwnd)
            screenshot_size = img.size

            # 计算窗口内容区尺寸（从 UIA 第一个内容子元素获取，而非根窗口）
            # 根窗口的 bounding_rect 包含装饰边框，不是真正的内容区
            if uia_elements:
                # 找第一个 left=0, top=0 的子元素作为内容区
                content_size = None
                for elem in uia_elements[1:]:  # 跳过根元素
                    if elem.bounding_rect:
                        l, t, r, b = elem.bounding_rect
                        if l == 0 and t == 0 and elem.width and elem.height:
                            content_size = (elem.width, elem.height)
                            break
                if content_size:
                    window_content_size = content_size

            if self._ocr_service is None:
                self._ocr_service = OCRService()
            ocr_blocks = self._ocr_service.extract(img)
        except Exception:
            ocr_blocks = []
            screenshot_size = None

        # 执行融合（传入截图尺寸和窗口内容区尺寸用于坐标转换）
        fused_elements = self._fuse(
            uia_elements, ocr_blocks, window_info,
            screenshot_size=screenshot_size,
            window_content_size=window_content_size,
        )

        return PageStructure(
            window_info=window_info,
            uia_elements=uia_elements,
            ocr_blocks=ocr_blocks,
            fused_elements=fused_elements,
            screenshot_size=screenshot_size,
        )

    def _fuse(
        self,
        uia_elements: list[UIAElementInfo],
        ocr_blocks: list[OCRTextBlock],
        window_info: WindowInfoExt | None,
        screenshot_size: tuple[int, int] | None = None,
        window_content_size: tuple[int, int] | None = None,
    ) -> list[FusedElement]:
        """
        融合 UIA 元素和 OCR 文本块

        融合逻辑：
        - 计算每个 UIA 元素与每个 OCR 文本块的 IOU
        - IOU > 0.5 时认为匹配，融合为一个元素
        - 未匹配到 OCR 的 UIA 元素保留 UIA 信息
        - 未匹配到 UIA 的 OCR 文本块单独作为一个文本元素
        - 世界坐标系：所有坐标以窗口内容区左上角为原点（去除了窗口装饰边框）

        Args:
            uia_elements: UIA 元素列表
            ocr_blocks: OCR 文本块列表
            window_info: 窗口信息
            screenshot_size: 缩放后截图尺寸 (width, height)
            window_content_size: 窗口内容区原始尺寸 (width, height)

        Returns:
            融合后的元素列表
        """
        fused: list[FusedElement] = []
        ocr_matched = [False] * len(ocr_blocks)
        uia_matched = [False] * len(uia_elements)

        # 注意：OCR 坐标现在基于原始截图（不再缩放）
        # 因此 OCR bbox 和 UIA bbox 都在同一坐标系（窗口局部原始坐标）下
        # 不再需要 scale_x/scale_y 转换

        def ocr_to_window_coords(
            ocr_bbox: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            """OCR bbox 已经是窗口局部原始坐标，直接返回"""
            return ocr_bbox

        def is_valid_rect(
            bbox: tuple[int, int, int, int],
            max_width: int = 10000,
            max_height: int = 10000,
            bounds_max_x: int | None = None,
            bounds_max_y: int | None = None,
        ) -> bool:
            """检查边界框是否有效

            Args:
                bbox: 边界框 (left, top, right, bottom)
                max_width: 最大宽度
                max_height: 最大高度
                bounds_max_x: 窗口内容区最大 X 坐标（用于过滤超出边界的 rect）
                bounds_max_y: 窗口内容区最大 Y 坐标
            """
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            # 宽高必须为正
            if width <= 0 or height <= 0:
                return False
            # 宽高必须在合理范围内
            if width > max_width or height > max_height:
                return False
            # 坐标不能是极端负值
            if left < -5000 or top < -5000:
                return False
            # 对于 UIA 元素，如果 left 或 top < 0，说明它包含了窗口装饰边框
            # 这类元素不属于窗口内容区，应该过滤掉
            # （OCR rect 不可能有负坐标，因为截图从 0,0 开始）
            if left < 0 or top < 0:
                return False
            # 坐标超出窗口内容区边界（如果有提供 bounds）
            if bounds_max_x is not None and (left > bounds_max_x or right > bounds_max_x):
                return False
            if bounds_max_y is not None and (top > bounds_max_y or bottom > bounds_max_y):
                return False
            return True

        def to_window_coords(
            bbox: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            """将边界框转换为窗口内容区相对坐标

            注意：经过 perception_service.analyze() 的 _normalize_elements_to_window_coords()
            处理后，UIA bounding_rect 已经是窗口局部原始坐标，无需再转换。
            """
            return bbox

        # UIA 元素与 OCR 文本块匹配（只处理有效 rect）
        bounds_x = window_content_size[0] if window_content_size else None
        bounds_y = window_content_size[1] if window_content_size else None
        for i, uia_elem in enumerate(uia_elements):
            if not uia_elem.bounding_rect:
                continue
            if not is_valid_rect(uia_elem.bounding_rect, bounds_max_x=bounds_x, bounds_max_y=bounds_y):
                continue

            best_iou = 0.0
            best_ocr_idx = -1

            # UIA rect 是窗口内容区相对坐标
            uia_rect = uia_elem.bounding_rect

            for j, ocr_block in enumerate(ocr_blocks):
                # 将 OCR 坐标转换为窗口内容区相对坐标
                ocr_rect = ocr_to_window_coords(ocr_block.bbox)
                iou = self._compute_iou(uia_rect, ocr_rect)
                if iou > best_iou:
                    best_iou = iou
                    best_ocr_idx = j

            if best_iou > 0.5:
                # 匹配成功
                ocr_block = ocr_blocks[best_ocr_idx]
                ocr_matched[best_ocr_idx] = True
                uia_matched[i] = True

                # OCR 转换后的坐标
                ocr_window_rect = ocr_to_window_coords(ocr_block.bbox)

                # 融合边界框（取并集，在窗口内容区坐标系中）
                fused_bbox = self._union_rect(uia_rect, ocr_window_rect)

                # OCR 的原始图像坐标保存下来供参考
                fused.append(FusedElement(
                    uia_element=uia_elem,
                    ocr_text=ocr_block.text,
                    ocr_bbox=ocr_block.bbox,  # 原始截图坐标
                    ocr_confidence=ocr_block.confidence,
                    fused_bbox=fused_bbox,  # 窗口内容区相对坐标
                    sources=["uia", "ocr"],
                ))

        # 未匹配的 UIA 元素（只包含有效 rect）
        for i, uia_elem in enumerate(uia_elements):
            if not uia_matched[i] and uia_elem.bounding_rect and is_valid_rect(
                uia_elem.bounding_rect, bounds_max_x=bounds_x, bounds_max_y=bounds_y
            ):
                fused.append(FusedElement(
                    uia_element=uia_elem,
                    ocr_text=None,
                    ocr_bbox=None,
                    ocr_confidence=None,
                    fused_bbox=uia_elem.bounding_rect,  # 已经是窗口内容区相对坐标
                    sources=["uia"],
                ))

        # 未匹配的 OCR 文本块
        for j, ocr_block in enumerate(ocr_blocks):
            if not ocr_matched[j]:
                ocr_window_rect = ocr_to_window_coords(ocr_block.bbox)
                fused.append(FusedElement(
                    uia_element=None,
                    ocr_text=ocr_block.text,
                    ocr_bbox=ocr_block.bbox,
                    ocr_confidence=ocr_block.confidence,
                    fused_bbox=ocr_window_rect,  # 窗口内容区相对坐标
                    sources=["ocr"],
                ))

        return fused

    def _compute_iou(
        self,
        rect1: tuple[int, int, int, int],
        rect2: tuple[int, int, int, int],
    ) -> float:
        """
        计算两个矩形的 IOU (Intersection over Union)

        Args:
            rect1: (left, top, right, bottom)
            rect2: (left, top, right, bottom)

        Returns:
            IOU 值，范围 [0, 1]
        """
        # 计算交集
        left = max(rect1[0], rect2[0])
        top = max(rect1[1], rect2[1])
        right = min(rect1[2], rect2[2])
        bottom = min(rect1[3], rect2[3])

        if left >= right or top >= bottom:
            return 0.0

        intersection = (right - left) * (bottom - top)

        # 计算各自面积
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        # 计算并集
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _union_rect(
        self,
        rect1: tuple[int, int, int, int],
        rect2: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """
        计算两个矩形的并集边界框

        Args:
            rect1: (left, top, right, bottom)
            rect2: (left, top, right, bottom)

        Returns:
            并集边界框 (left, top, right, bottom)
        """
        left = min(rect1[0], rect2[0])
        top = min(rect1[1], rect2[1])
        right = max(rect1[2], rect2[2])
        bottom = max(rect1[3], rect2[3])
        return (left, top, right, bottom)
