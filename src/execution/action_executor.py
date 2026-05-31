"""
动作执行器

提供鼠标点击、键盘输入、UIA 调用等动作执行能力。

设计原则：
- 所有动作执行前必须激活目标窗口
- 高风险动作（发消息、付款等）必须先本地校验
- 动作执行后应验证是否成功
"""
from ctypes import windll, Structure, c_long
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import win32api
import win32con
import win32gui
import win32clipboard

import uiautomation as uia

from src.common.errors import ActionExecutionError


class POINT(Structure):
    """点坐标结构"""
    _fields_ = [("x", c_long), ("y", c_long)]


class MouseButton(Enum):
    """鼠标按钮"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class KeyModifier(Enum):
    """键盘修饰键"""
    ALT = "alt"
    CTRL = "ctrl"
    SHIFT = "shift"
    WIN = "win"


@dataclass
class ActionResult:
    """动作执行结果"""
    success: bool
    action: str
    details: str = ""
    error: str | None = None


class ActionExecutor:
    """
    动作执行器

    提供鼠标点击、键盘输入、UIA 调用等动作。
    """

    def __init__(self) -> None:
        self._activator = self._load_activator()

    def _load_activator(self):
        """延迟导入 WindowActivator"""
        from src.windows.window_activator import WindowActivator
        return WindowActivator()

    # -------------------------------------------------------------------------
    # 鼠标操作
    # -------------------------------------------------------------------------

    def click(
        self,
        hwnd: int,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        double: bool = False,
    ) -> ActionResult:
        """
        在指定窗口的绝对坐标处点击

        Args:
            hwnd: 窗口句柄
            x: 绝对屏幕 X 坐标
            y: 绝对屏幕 Y 坐标
            button: 鼠标按钮
            double: 是否双击

        Returns:
            ActionResult 执行结果
        """
        try:
            # 1. 激活窗口
            if not self._activator.activate(hwnd):
                return ActionResult(False, "click", f"窗口 {hwnd} 激活失败", "activation_failed")

            # 2. 设置鼠标位置并点击
            self._set_cursor_pos(x, y)
            self._sleep(0.05)

            down_flag = self._get_mouse_button_down_flag(button)
            up_flag = self._get_mouse_button_up_flag(button)

            if double:
                # 双击：两次完整的点击事件
                win32api.mouse_event(down_flag, 0, 0, 0, 0)
                win32api.mouse_event(up_flag, 0, 0, 0, 0)
                self._sleep(0.05)
                win32api.mouse_event(down_flag, 0, 0, 0, 0)
                win32api.mouse_event(up_flag, 0, 0, 0, 0)
            else:
                win32api.mouse_event(down_flag, 0, 0, 0, 0)
                self._sleep(0.03)
                win32api.mouse_event(up_flag, 0, 0, 0, 0)

            return ActionResult(True, "click", f"点击 ({x}, {y})")

        except Exception as e:
            return ActionResult(False, "click", "", str(e))

    def click_element(
        self,
        hwnd: int,
        element_x: int,
        element_y: int,
        element_width: int,
        element_height: int,
        button: MouseButton = MouseButton.LEFT,
        double: bool = False,
    ) -> ActionResult:
        """
        点击元素中心点

        Args:
            hwnd: 窗口句柄
            element_x: 元素左边界（窗口内容区坐标）
            element_y: 元素上边界（窗口内容区坐标）
            element_width: 元素宽度
            element_height: 元素高度
            button: 鼠标按钮
            double: 是否双击

        Returns:
            ActionResult 执行结果
        """
        # 计算元素中心点
        center_x = element_x + element_width // 2
        center_y = element_y + element_height // 2

        # 转换为屏幕坐标
        screen_x, screen_y = self._client_to_screen(hwnd, center_x, center_y)

        return self.click(hwnd, screen_x, screen_y, button, double)

    def scroll(
        self,
        hwnd: int,
        delta: int = 120,
    ) -> ActionResult:
        """
        滚动鼠标滚轮

        Args:
            hwnd: 窗口句柄
            delta: 滚动量（正数向上，负数向下，默认 120）

        Returns:
            ActionResult 执行结果
        """
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "scroll", f"窗口 {hwnd} 激活失败", "activation_failed")

            # 获取当前鼠标位置
            cur_x, cur_y = win32api.GetCursorPos()

            # 发送滚轮事件
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, delta, 0)

            return ActionResult(True, "scroll", f"滚动 {delta}")

        except Exception as e:
            return ActionResult(False, "scroll", "", str(e))

    def scroll_at(
        self,
        hwnd: int,
        client_x: int,
        client_y: int,
        delta: int = 120,
    ) -> ActionResult:
        """Scroll at a specific client-coordinate point inside the target window."""
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "scroll", f"窗口 {hwnd} 激活失败", "activation_failed")
            screen_x, screen_y = self._client_to_screen(hwnd, client_x, client_y)
            self._set_cursor_pos(screen_x, screen_y)
            self._sleep(0.05)
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, delta, 0)
            return ActionResult(True, "scroll", f"滚动 {delta} at ({client_x}, {client_y})")
        except Exception as e:
            return ActionResult(False, "scroll", "", str(e))

    # -------------------------------------------------------------------------
    # 键盘操作
    # -------------------------------------------------------------------------

    def send_text(
        self,
        hwnd: int,
        text: str,
    ) -> ActionResult:
        """
        发送文本（到当前焦点窗口）

        Args:
            hwnd: 窗口句柄（用于激活）
            text: 要发送的文本

        Returns:
            ActionResult 执行结果
        """
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "send_text", f"窗口 {hwnd} 激活失败", "activation_failed")

            # 打开剪贴板
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text)
            win32clipboard.CloseClipboard()

            # Ctrl+V 粘贴
            self._sleep(0.05)
            self._send_keys("^v")

            return ActionResult(True, "send_text", f"发送文本: {text[:20]}...")

        except Exception as e:
            return ActionResult(False, "send_text", "", str(e))

    def send_keys(
        self,
        hwnd: int,
        keys: str,
    ) -> ActionResult:
        """
        发送快捷键（如 "^c" 代表 Ctrl+C）

        Args:
            hwnd: 窗口句柄
            keys: 按键字符串（pywinauto 格式）

        Returns:
            ActionResult 执行结果
        """
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "send_keys", f"窗口 {hwnd} 激活失败", "activation_failed")

            self._sleep(0.05)
            self._send_keys(keys)

            return ActionResult(True, "send_keys", f"发送按键: {keys}")

        except Exception as e:
            return ActionResult(False, "send_keys", "", str(e))

    # -------------------------------------------------------------------------
    # UIA 调用
    # -------------------------------------------------------------------------

    def invoke_uia_element(
        self,
        hwnd: int,
        automation_id: str | None = None,
        name: str | None = None,
        control_type: str | None = None,
    ) -> ActionResult:
        """
        通过 UIA 找到元素并调用（InvokePattern）

        Args:
            hwnd: 窗口句柄
            automation_id: UIA 自动化 ID
            name: UIA 元素名称
            control_type: UIA 控制类型

        Returns:
            ActionResult 执行结果
        """
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "invoke_uia", f"窗口 {hwnd} 激活失败", "activation_failed")

            # 查找元素
            elem = self._find_uia_element(hwnd, automation_id, name, control_type)
            if elem is None:
                return ActionResult(False, "invoke_uia", f"未找到元素: id={automation_id}, name={name}", "element_not_found")

            # 尝试调用 Invoke
            try:
                invoke_pattern = elem.GetInvokePattern()
                invoke_pattern.Invoke()
                return ActionResult(True, "invoke_uia", f"调用元素: {automation_id or name}")
            except Exception:
                # 尝试其他模式
                try:
                    # 尝试 Click
                    elem.Click()
                    return ActionResult(True, "invoke_uia", f"点击元素: {automation_id or name}")
                except Exception as e2:
                    return ActionResult(False, "invoke_uia", f"元素不支持 Invoke: {e2}", "invoke_failed")

        except Exception as e:
            return ActionResult(False, "invoke_uia", "", str(e))

    def select_uia_item(
        self,
        hwnd: int,
        item_name: str,
        automation_id: str | None = None,
    ) -> ActionResult:
        """
        在列表中选择项

        Args:
            hwnd: 窗口句柄
            item_name: 要选择的项名称
            automation_id: 列表的自动化 ID

        Returns:
            ActionResult 执行结果
        """
        try:
            if not self._activator.activate(hwnd):
                return ActionResult(False, "select_uia_item", f"窗口 {hwnd} 激活失败", "activation_failed")

            # 找到列表项
            try:
                list_elem = uia.WindowControl(hwnd=hwnd, foundIndex=0)
                if automation_id:
                    list_elem = uia.PaneControl(hwnd=hwnd, automationId=automation_id, foundIndex=0)

                item = list_elem.ListItemControl(name=item_name, foundIndex=0)
                item.Select()
                return ActionResult(True, "select_uia_item", f"选择项: {item_name}")
            except Exception:
                # 尝试直接点击
                elem = self._find_uia_element(hwnd, automation_id, item_name, "ListItem")
                if elem:
                    elem.Click()
                    return ActionResult(True, "select_uia_item", f"点击选择: {item_name}")
                return ActionResult(False, "select_uia_item", f"未找到项: {item_name}", "item_not_found")

        except Exception as e:
            return ActionResult(False, "select_uia_item", "", str(e))

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _find_uia_element(
        self,
        hwnd: int,
        automation_id: str | None,
        name: str | None,
        control_type: str | None,
    ):
        """在窗口中查找 UIA 元素"""
        try:
            window = uia.WindowControl(hwnd=hwnd, foundIndex=0)

            if automation_id:
                return window.EditControl(automationId=automation_id, foundIndex=0)
            if name:
                return window.ButtonControl(name=name, foundIndex=0)
            if control_type:
                return window.CustomControl(controlType=control_type, foundIndex=0)

            return None
        except Exception:
            return None

    def _client_to_screen(self, hwnd: int, client_x: int, client_y: int) -> tuple[int, int]:
        """客户端坐标转屏幕坐标"""
        try:
            pt = POINT(client_x, client_y)
            windll.user32.ClientToScreen(hwnd, pt)
            return (pt.x, pt.y)
        except Exception:
            # 回退：使用 GetWindowRect
            rect = win32gui.GetWindowRect(hwnd)
            return (rect[0] + client_x, rect[1] + client_y)

    def _set_cursor_pos(self, x: int, y: int) -> None:
        """设置鼠标位置"""
        win32api.SetCursorPos((x, y))

    def _get_mouse_button_down_flag(self, button: MouseButton) -> int:
        """获取鼠标按钮按下标志"""
        if button == MouseButton.LEFT:
            return win32con.MOUSEEVENTF_LEFTDOWN
        elif button == MouseButton.RIGHT:
            return win32con.MOUSEEVENTF_RIGHTDOWN
        elif button == MouseButton.MIDDLE:
            return win32con.MOUSEEVENTF_MIDDLEDOWN
        return win32con.MOUSEEVENTF_LEFTDOWN

    def _get_mouse_button_up_flag(self, button: MouseButton) -> int:
        """获取鼠标按钮释放标志"""
        if button == MouseButton.LEFT:
            return win32con.MOUSEEVENTF_LEFTUP
        elif button == MouseButton.RIGHT:
            return win32con.MOUSEEVENTF_RIGHTUP
        elif button == MouseButton.MIDDLE:
            return win32con.MOUSEEVENTF_MIDDLEUP
        return win32con.MOUSEEVENTF_LEFTUP

    def _send_keys(self, keys: str) -> None:
        """发送按键（支持修饰键和少量特殊键）"""
        modifier_stack: list[int] = []

        for token in self._tokenize_keys(keys):
            if token == "^":
                modifier_stack.append(win32con.VK_CONTROL)
                continue
            if token == "+":
                modifier_stack.append(win32con.VK_SHIFT)
                continue
            if token == "%":
                modifier_stack.append(win32con.VK_MENU)
                continue

            for modifier in modifier_stack:
                win32api.keybd_event(modifier, 0, 0, 0)

            self._send_key_token(token)

            for modifier in reversed(modifier_stack):
                win32api.keybd_event(modifier, 0, win32con.KEYEVENTF_KEYUP, 0)
            modifier_stack.clear()

        for modifier in reversed(modifier_stack):
            win32api.keybd_event(modifier, 0, win32con.KEYEVENTF_KEYUP, 0)

    def _tokenize_keys(self, keys: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(keys):
            if keys[index] == "{":
                end = keys.find("}", index)
                if end != -1:
                    tokens.append(keys[index : end + 1])
                    index = end + 1
                    continue
            tokens.append(keys[index])
            index += 1
        return tokens

    def _send_key_token(self, token: str) -> None:
        special_keys = {
            "{ENTER}": win32con.VK_RETURN,
            "{TAB}": win32con.VK_TAB,
            "{ESC}": win32con.VK_ESCAPE,
            "{SPACE}": win32con.VK_SPACE,
        }
        if token in special_keys:
            vk_code = special_keys[token]
            win32api.keybd_event(vk_code, 0, 0, 0)
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            return

        if token in {"\r", "\n"}:
            win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
            win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
            return

        vk = win32api.VkKeyScan(token)
        vk_code = vk & 0xFF
        modifiers = vk >> 8
        if modifiers & 1:
            win32api.keybd_event(win32con.VK_SHIFT, 0, 0, 0)
        if modifiers & 2:
            win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        if modifiers & 4:
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)

        win32api.keybd_event(vk_code, 0, 0, 0)
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)

        if modifiers & 4:
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        if modifiers & 2:
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        if modifiers & 1:
            win32api.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)

    def _sleep(self, seconds: float) -> None:
        """等待（避免系统过快操作）"""
        import time
        time.sleep(seconds)
