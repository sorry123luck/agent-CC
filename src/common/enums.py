"""
枚举类型定义
"""

from enum import Enum


class AppSource(str, Enum):
    """软件来源"""
    REGISTRY = "registry"
    STARTMENU = "startmenu"
    DESKTOP = "desktop"
    FILESYSTEM = "filesystem"
    UWP = "uwp"
    USER_ADDED = "user_added"


class LaunchTargetType(str, Enum):
    """启动入口类型"""
    EXE = "exe"
    UWP = "uwp"
    SHORTCUT = "shortcut"
    URL = "url"


class PageType(str, Enum):
    """页面类型"""
    UNKNOWN = "unknown"
    MAIN = "main"
    SETTINGS = "settings"
    DIALOG = "dialog"
    LIST = "list"
    FORM = "form"
    CHAT = "chat"
    BROWSER = "browser"


class ElementRole(str, Enum):
    """界面元素角色"""
    WINDOW = "window"
    TOOLBAR = "toolbar"
    NAVIGATION = "navigation"
    SIDEBAR = "sidebar"
    LIST = "list"
    TABLE = "table"
    CARD = "card"
    INPUT = "input"
    SEARCH_BOX = "search_box"
    BUTTON = "button"
    ICON_BUTTON = "icon_button"
    MENU = "menu"
    DIALOG = "dialog"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SWITCH = "switch"
    TAB = "tab"
    TEXT = "text"
    EMPTY_STATE = "empty_state"


class ActionType(str, Enum):
    """动作类型"""
    FOCUS_WINDOW = "focus_window"
    WAIT_WINDOW = "wait_window"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    SCROLL = "scroll"
    TYPE_TEXT = "type_text"
    HOTKEY = "hotkey"
    PRESS_KEY = "press_key"
    SELECT_LIST_ITEM = "select_list_item"
    SET_VALUE_VIA_UIA = "set_value_via_uia"
    INVOKE_BUTTON_VIA_UIA = "invoke_button_via_uia"
    DRAG = "drag"
    CLIPBOARD_PASTE = "clipboard_paste"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerifyResult(str, Enum):
    """验证结果"""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
