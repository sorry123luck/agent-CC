"""
OCR Provider 接口

提供 OCR 文本提取能力。
职责边界（2026-04-05 确认）：
- 只负责文本提取，不负责页面主划区
- 只作为渲染内容（Canvas/WebView/Game）的辅助兜底
- 不用于 GUI chrome 的主要识别
"""

from dataclasses import dataclass
from typing import Any

from src.perception.page_compiler_models import (
    SurfaceType,
    WindowInfoSnapshot,
    InteractionCanvas,
)
from src.perception.providers.base import PerceptionProviderBase


@dataclass
class OCRBlock:
    """
    OCR 识别结果块

    对应 PaddleOCR 的通用输出结构。
    """
    text: str                    # 识别出的文本
    confidence: float            # 置信度（0-1）
    bounding_box: tuple[int, int, int, int]  # 左上右下坐标
    # 扩展字段
    angle: float | None = None  # 文本角度
    language: str | None = None  # 语言


class IOCRProvider(PerceptionProviderBase):
    """
    OCR Provider 接口

    职责：
    - 从图像区域提取文本
    - 适用于 Canvas 自绘、WebView、Game 等无法用 UIA 提取文本的场景
    - 仅作为补充手段，不替代 UIA 的 Name/Value 属性

    支持的 surface_type：
    - CANVAS_SELF_DRAWN（主要）
    - BROWSER（WebView 内容补充）
    - ELECTRON_WEBVIEW（内核内容补充）
    """

    @property
    def name(self) -> str:
        return "OCRProvider"

    def supports(self, surface_type: SurfaceType) -> bool:
        """OCR 支持 Canvas、WebView 和 Electron 内核"""
        return surface_type in (
            SurfaceType.CANVAS_SELF_DRAWN,
            SurfaceType.BROWSER,
            SurfaceType.ELECTRON_WEBVIEW,
        )

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """
        从窗口提取 OCR 文本

        通常不直接对整个窗口做 OCR，而是对指定区域（content_area）做裁剪后 OCR。

        Returns:
            InteractionCanvas 包含 elements（含 OCR 提取的文本）
        """
        raise NotImplementedError(f"{self.name}.extract() must be implemented")

    def extract_from_image(
        self,
        image: Any,  # PIL.Image.Image
        region: tuple[int, int, int, int] | None = None,
    ) -> list[OCRBlock]:
        """
        从图像（或图像区域）提取文本

        Args:
            image: PIL Image 对象
            region: 可选的裁剪区域 (left, top, right, bottom)

        Returns:
            list[OCRBlock] OCR 识别结果列表
        """
        raise NotImplementedError(f"{self.name}.extract_from_image() must be implemented")

    def extract_text_only(
        self,
        image: Any,  # PIL.Image.Image
        region: tuple[int, int, int, int] | None = None,
    ) -> list[str]:
        """
        简易接口：只返回文本列表

        Args:
            image: PIL Image 对象
            region: 可选的裁剪区域

        Returns:
            list[str] 识别出的文本列表
        """
        blocks = self.extract_from_image(image, region)
        return [block.text for block in blocks if block.confidence > 0.5]
