"""
UIA 客户端

提供 UIA (UI Automation) 元素遍历和发现功能
"""
from dataclasses import dataclass
from typing import Callable

import uiautomation as uia

from src.common.errors import ElementNotFoundError


@dataclass
class UIAElementInfo:
    """UIA 元素信息"""
    name: str | None
    automation_id: str | None
    control_type: str | None
    bounding_rect: tuple[int, int, int, int] | None  # left, top, right, bottom
    is_enabled: bool
    handle: int | None = None
    element_id: str | None = None  # 全局唯一标识，从 UIAClient 自动生成

    @property
    def rect(self) -> tuple[int, int, int, int] | None:
        """返回边界矩形 (left, top, right, bottom)"""
        return self.bounding_rect

    @property
    def left(self) -> int | None:
        return self.bounding_rect[0] if self.bounding_rect else None

    @property
    def top(self) -> int | None:
        return self.bounding_rect[1] if self.bounding_rect else None

    @property
    def right(self) -> int | None:
        return self.bounding_rect[2] if self.bounding_rect else None

    @property
    def bottom(self) -> int | None:
        return self.bounding_rect[3] if self.bounding_rect else None

    @property
    def width(self) -> int | None:
        if self.bounding_rect:
            return self.bounding_rect[2] - self.bounding_rect[0]
        return None

    @property
    def height(self) -> int | None:
        if self.bounding_rect:
            return self.bounding_rect[3] - self.bounding_rect[1]
        return None


class UIAClient:
    """UIA 客户端"""

    def __init__(self, hwnd: int) -> None:
        """
        初始化 UIA 客户端

        Args:
            hwnd: 窗口句柄
        """
        self._hwnd = hwnd
        self._root = uia.ControlFromHandle(hwnd)
        self._element_id_counter = 0  # 为每个 UIAElementInfo 生成唯一 element_id

        if not self._root:
            raise ElementNotFoundError(f"无法获取窗口句柄 {hwnd} 的 UIA 根元素")

    def _next_element_id(self) -> str:
        """生成下一个唯一 element_id"""
        self._element_id_counter += 1
        return f"uia_{self._hwnd}_{self._element_id_counter}"

    def get_root_element(self) -> UIAElementInfo:
        """
        获取根元素

        Returns:
            根元素信息
        """
        return self._to_element_info(self._root, self._next_element_id())

    def find_by_id(self, automation_id: str) -> list[UIAElementInfo]:
        """
        按 AutomationId 查找元素

        Args:
            automation_id: AutomationId

        Returns:
            匹配的元素列表
        """
        return self._find_with_filter(lambda c: self._match_automation_id(c, automation_id))

    def find_by_name(self, name: str, exact: bool = False) -> list[UIAElementInfo]:
        """
        按 Name 属性查找元素

        Args:
            name: 元素名称
            exact: 是否精确匹配

        Returns:
            匹配的元素列表
        """
        return self._find_with_filter(lambda c: self._match_name(c, name, exact))

    def find_by_control_type(
        self, control_type: str
    ) -> list[UIAElementInfo]:
        """
        按 ControlType 查找元素

        Args:
            control_type: ControlType 名称（如 "Button", "Edit", "Window"）

        Returns:
            匹配的元素列表
        """
        return self._find_with_filter(lambda c: self._match_control_type(c, control_type))

    def find_by_class(
        self, class_name: str
    ) -> list[UIAElementInfo]:
        """
        按 ClassName 查找元素

        Args:
            class_name: 类名

        Returns:
            匹配的元素列表
        """
        return self._find_with_filter(lambda c: self._match_class_name(c, class_name))

    def find_all(self) -> list[UIAElementInfo]:
        """
        获取所有子元素

        Returns:
            所有子元素列表
        """
        return self._find_with_filter(lambda _: True)

    def get_element_tree(
        self, max_depth: int = 10
    ) -> list[tuple[int, UIAElementInfo]]:
        """
        获取元素树（深度优先遍历）

        Args:
            max_depth: 最大遍历深度

        Returns:
            (depth, element_info) 元组列表
        """
        result: list[tuple[int, UIAElementInfo]] = []

        def traverse(control: uia.Control, depth: int) -> None:
            if depth > max_depth:
                return
            result.append((depth, self._to_element_info(control, self._next_element_id())))
            try:
                children = control.GetChildren()
                for child in children:
                    traverse(child, depth + 1)
            except Exception:
                pass

        traverse(self._root, 0)
        return result

    def _find_with_filter(
        self, predicate: Callable[[uia.Control], bool]
    ) -> list[UIAElementInfo]:
        """
        使用过滤器查找所有匹配的元素（深度优先遍历）

        Args:
            predicate: 过滤函数

        Returns:
            匹配的元素列表
        """
        result: list[UIAElementInfo] = []

        def traverse(control: uia.Control) -> None:
            try:
                if predicate(control):
                    result.append(self._to_element_info(control, self._next_element_id()))
                children = control.GetChildren()
                for child in children:
                    traverse(child)
            except Exception:
                pass

        traverse(self._root)
        return result

    def _to_element_info(self, control: uia.Control, element_id: str | None = None) -> UIAElementInfo:
        """将 UIA Control 转换为 UIAElementInfo"""
        try:
            rect = control.BoundingRectangle
            if rect:
                bounding_rect = (rect.left, rect.top, rect.right, rect.bottom)
            else:
                bounding_rect = None
        except Exception:
            bounding_rect = None

        try:
            is_enabled = control.IsEnabled
        except Exception:
            is_enabled = True

        try:
            handle = control.Handle
        except Exception:
            handle = None

        return UIAElementInfo(
            name=control.Name if hasattr(control, "Name") else None,
            automation_id=control.AutomationId if hasattr(control, "AutomationId") else None,
            control_type=control.ControlTypeName if hasattr(control, "ControlTypeName") else None,
            bounding_rect=bounding_rect,
            is_enabled=is_enabled,
            handle=handle,
            element_id=element_id,
        )

    def _match_automation_id(self, control: uia.Control, automation_id: str) -> bool:
        """检查控件的 AutomationId 是否匹配"""
        try:
            return hasattr(control, "AutomationId") and control.AutomationId == automation_id
        except Exception:
            return False

    def _match_name(self, control: uia.Control, name: str, exact: bool) -> bool:
        """检查控件的 Name 是否匹配"""
        try:
            if not hasattr(control, "Name"):
                return False
            if exact:
                return control.Name == name
            else:
                return name.lower() in control.Name.lower()
        except Exception:
            return False

    def _match_control_type(self, control: uia.Control, control_type: str) -> bool:
        """检查控件的 ControlType 是否匹配"""
        try:
            if not hasattr(control, "ControlTypeName"):
                return False
            # ControlTypeName 格式如 "ButtonControl"
            return control_type.lower() in control.ControlTypeName.lower()
        except Exception:
            return False

    def _match_class_name(self, control: uia.Control, class_name: str) -> bool:
        """检查控件的 ClassName 是否匹配"""
        try:
            if not hasattr(control, "ClassName"):
                return False
            return control.ClassName.lower() == class_name.lower()
        except Exception:
            return False
