"""
Provider 抽象基类

为各类 Provider 提供公共基类和通用工具方法。
"""

from abc import ABC, abstractmethod
from typing import Any

from src.perception.page_compiler_models import (
    SurfaceType,
    WindowInfoSnapshot,
    InteractionCanvas,
)


class PerceptionProviderBase(ABC):
    """
    Provider 抽象基类

    所有 Provider 必须：
    1. 实现 name() 方法返回 Provider 名称
    2. 实现 supports(surface_type) 方法声明支持的 surface_type
    3. 实现 extract(window_info, existing_structure) 方法执行提取
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""
        ...

    @abstractmethod
    def supports(self, surface_type: SurfaceType) -> bool:
        """是否支持给定 surface_type"""
        ...

    @abstractmethod
    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """
        从窗口提取页面结构

        Args:
            window_info: 窗口信息快照
            existing_structure: 已有的页面结构（供增量更新）

        Returns:
            InteractionCanvas 更新后的页面结构
        """
        ...

    def _validate_window_info(self, window_info: WindowInfoSnapshot) -> None:
        """验证 window_info 是否有效"""
        if not window_info.hwnd:
            raise ValueError("window_info.hwnd is required")
        if window_info.hwnd <= 0:
            raise ValueError(f"Invalid hwnd: {window_info.hwnd}")

    def _validate_surface_type(self, surface_type: SurfaceType) -> None:
        """验证 surface_type 是否被该 Provider 支持"""
        if not self.supports(surface_type):
            raise ValueError(
                f"{self.name} does not support surface_type: {surface_type}"
            )

    def get_provider_metadata(self) -> dict[str, Any]:
        """返回 Provider 元信息（可被子类覆盖）"""
        return {
            "name": self.name,
            "supports": [st.value for st in SurfaceType if self.supports(st)],
        }
