"""
DOM Provider 接口

提供浏览器 DOM 提取能力（仅适用于 browser 和 electron_webview）。
"""

from typing import Any

from src.perception.page_compiler_models import (
    SurfaceType,
    WindowInfoSnapshot,
    InteractionCanvas,
)
from src.perception.providers.base import PerceptionProviderBase


class IDOMProvider(PerceptionProviderBase):
    """
    DOM Provider 接口

    职责：
    - 从浏览器 tab 提取 DOM 树
    - 获取元素的 accessibility 信息
    - 支持 Playwright CDP 协议

    支持的 surface_type：
    - BROWSER（主要）
    - ELECTRON_WEBVIEW（内核部分）
    """

    @property
    def name(self) -> str:
        return "DOMProvider"

    def supports(self, surface_type: SurfaceType) -> bool:
        """DOM Provider 只支持 browser 和 electron_webview"""
        return surface_type in (
            SurfaceType.BROWSER,
            SurfaceType.ELECTRON_WEBVIEW,
        )

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """
        从浏览器 tab 提取 DOM 结构

        Returns:
            InteractionCanvas 包含 elements（DOM 元素）
        """
        raise NotImplementedError(f"{self.name}.extract() must be implemented")

    def connect(self, browser_type: str, url: str | None = None) -> bool:
        """
        连接到浏览器实例或 tab

        Args:
            browser_type: 浏览器类型 ("chrome", "msedge", "firefox")
            url: 可选的目标 URL（不传则取当前 active tab）

        Returns:
            是否连接成功
        """
        raise NotImplementedError(f"{self.name}.connect() must be implemented")

    def disconnect(self) -> None:
        """断开与浏览器的连接"""
        raise NotImplementedError(f"{self.name}.disconnect() must be implemented")

    def get_dom_tree(self) -> dict[str, Any]:
        """
        获取完整 DOM 树

        Returns:
            DOM 树字典，包含：
            - tag_name: str
            - attributes: dict
            - children: list
            - text: str | None
            - rect: tuple | None
        """
        raise NotImplementedError(f"{self.name}.get_dom_tree() must be implemented")

    def get_element_by_locator(
        self,
        selector: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        根据 selector 查找 DOM 元素

        Args:
            selector: locator selector dict，包含 kind (css/xpath/accessibility_id) 和 value

        Returns:
            元素 dict 或 None
        """
        raise NotImplementedError(f"{self.name}.get_element_by_locator() must be implemented")

    def perform_action(
        self,
        selector: dict[str, Any],
        action: str,
    ) -> dict[str, Any]:
        """
        Execute a DOM-native action against a locator.

        Returns:
            dict:
            - success: bool
            - details: str
            - rect: tuple | None
        """
        raise NotImplementedError(f"{self.name}.perform_action() must be implemented")

    def is_dom_ready(self) -> bool:
        """
        检查 DOM 是否就绪（页面加载完成）

        Returns:
            DOM 是否可访问
        """
        raise NotImplementedError(f"{self.name}.is_dom_ready() must be implemented")

    def take_screenshot(self, full_page: bool = False) -> bytes:
        """
        获取页面截图

        Args:
            full_page: 是否截取整个可滚动页面

        Returns:
            PNG 格式的图像数据
        """
        raise NotImplementedError(f"{self.name}.take_screenshot() must be implemented")
