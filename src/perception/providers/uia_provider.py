"""
UIA Provider 接口

提供 Windows UIA (UI Automation) 元素提取能力。
"""

from typing import Any

from src.perception.page_compiler_models import (
    SurfaceType,
    WindowInfoSnapshot,
    InteractionCanvas,
)
from src.perception.providers.base import PerceptionProviderBase


class IUIAProvider(PerceptionProviderBase):
    """
    UIA Provider 接口

    职责：
    - 从窗口提取 UIA 元素树
    - 过滤无效坐标元素
    - 支持原生 GUI 和 Electron 外壳的 UIA 提取

    支持的 surface_type：
    - NATIVE_UIA（主要）
    - ELECTRON_WEBVIEW（外壳部分）
    - CANVAS_SELF_DRAWN（兜底）
    """

    @property
    def name(self) -> str:
        return "UIAProvider"

    def supports(self, surface_type: SurfaceType) -> bool:
        """UIA 支持原生 GUI、Electron 外壳和 Canvas 自绘"""
        return surface_type in (
            SurfaceType.NATIVE_UIA,
            SurfaceType.ELECTRON_WEBVIEW,
            SurfaceType.CANVAS_SELF_DRAWN,
        )

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """
        从窗口提取 UIA 元素

        Returns:
            InteractionCanvas 包含 elements（UIA 元素列表）
        """
        raise NotImplementedError(f"{self.name}.extract() must be implemented")

    def find_all_elements(self, hwnd: int) -> list[dict[str, Any]]:
        """
        查找窗口所有 UIA 元素

        Returns:
            list[dict] 元素列表，每个元素包含：
            - control_type: str
            - name: str | None
            - value: str | None
            - bounding_rect: tuple[int, int, int, int] | None
            - left, top, right, bottom: int
            - width, height: int
            - class_name: str | None
            - automation_id: str | None
        """
        raise NotImplementedError(f"{self.name}.find_all_elements() must be implemented")

    def get_element_at_point(self, hwnd: int, x: int, y: int) -> dict[str, Any] | None:
        """
        获取指定坐标点的 UIA 元素

        Returns:
            元素 dict 或 None
        """
        raise NotImplementedError(f"{self.name}.get_element_at_point() must be implemented")

    def invoke_element(self, hwnd: int, automation_id: str) -> bool:
        """
        调用 UIA 元素的默认操作（如点击按钮）

        Returns:
            是否成功
        """
        raise NotImplementedError(f"{self.name}.invoke_element() must be implemented")

    def set_element_value(self, hwnd: int, automation_id: str, value: str) -> bool:
        """
        设置 UIA 元素的值（如设置输入框文本）

        Returns:
            是否成功
        """
        raise NotImplementedError(f"{self.name}.set_element_value() must be implemented")
