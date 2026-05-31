"""
Template Provider 接口

提供图标模板匹配能力（OpenCV 图像识别）。
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
class TemplateMatch:
    """
    模板匹配结果
    """
    template_name: str                           # 模板名称
    confidence: float                           # 匹配置信度（0-1）
    bounding_box: tuple[int, int, int, int]    # 匹配位置（左上右下）
    center: tuple[int, int]                     # 匹配中心点


class ITemplateProvider(PerceptionProviderBase):
    """
    Template Provider 接口

    职责：
    - 在截图或窗口中匹配已知图标模板
    - 适用于标准 GUI 应用的工具栏按钮、菜单图标等
    - 作为定位器的兜底手段

    支持的 surface_type：
    - NATIVE_UIA（主要）
    - CANVAS_SELF_DRAWN（兜底）
    - ELECTRON_WEBVIEW（外壳部分）
    """

    @property
    def name(self) -> str:
        return "TemplateProvider"

    def supports(self, surface_type: SurfaceType) -> bool:
        """模板 Provider 支持原生 GUI、Canvas 和 Electron 外壳"""
        return surface_type in (
            SurfaceType.NATIVE_UIA,
            SurfaceType.CANVAS_SELF_DRAWN,
            SurfaceType.ELECTRON_WEBVIEW,
        )

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """
        在窗口中匹配已注册图标模板

        Returns:
            InteractionCanvas 包含匹配到的元素
        """
        raise NotImplementedError(f"{self.name}.extract() must be implemented")

    def register_template(
        self,
        name: str,
        image: Any,  # PIL.Image.Image
        description: str | None = None,
    ) -> bool:
        """
        注册新的图标模板

        Args:
            name: 模板名称（唯一标识）
            image: 模板图像
            description: 可选描述

        Returns:
            是否注册成功
        """
        raise NotImplementedError(f"{self.name}.register_template() must be implemented")

    def find_template(
        self,
        template_name: str,
        search_image: Any,  # PIL.Image.Image
        threshold: float = 0.80,
    ) -> list[TemplateMatch]:
        """
        在图像中查找指定模板

        Args:
            template_name: 已注册的模板名称
            search_image: 要搜索的图像
            threshold: 匹配阈值（0-1）

        Returns:
            list[TemplateMatch] 匹配结果列表
        """
        raise NotImplementedError(f"{self.name}.find_template() must be implemented")

    def find_all_templates(
        self,
        search_image: Any,  # PIL.Image.Image
        threshold: float = 0.80,
    ) -> list[TemplateMatch]:
        """
        在图像中匹配所有已注册模板

        Args:
            search_image: 要搜索的图像
            threshold: 匹配阈值

        Returns:
            list[TemplateMatch] 所有匹配结果
        """
        raise NotImplementedError(f"{self.name}.find_all_templates() must be implemented")

    def list_templates(self) -> list[str]:
        """
        列出所有已注册的模板名称

        Returns:
            list[str] 模板名称列表
        """
        raise NotImplementedError(f"{self.name}.list_templates() must be implemented")
