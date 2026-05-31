from src.windows.window_action_context import (
    chat_search_capture_mode,
    _find_tray_control,
    post_selection_escape_count,
    pre_probe_escape_count,
)


def test_chat_search_capture_mode_uses_screen_region_for_wechat_popups():
    assert chat_search_capture_mode("weixin.exe") == "screen_region"
    assert chat_search_capture_mode("wechat.exe") == "screen_region"
    assert chat_search_capture_mode("qq.exe") == "window"
    assert chat_search_capture_mode("feishu.exe") == "window"


def test_escape_policy_keeps_wechat_visible():
    assert pre_probe_escape_count("weixin.exe") == 0
    assert post_selection_escape_count("weixin.exe") == 0
    assert pre_probe_escape_count("qq.exe") == 1
    assert pre_probe_escape_count("feishu.exe") == 2


class _FakeControl:
    def __init__(self, name="", class_name="", children=None):
        self.Name = name
        self.ClassName = class_name
        self._children = children or []

    def GetChildren(self):
        return self._children


class _FakeAutomation:
    def __init__(self, root):
        self._root = root

    def GetRootControl(self):
        return self._root


def test_find_tray_control_requires_tray_class_and_process_keyword():
    matching = _FakeControl("微信", "SystemTray.NormalButton")
    non_tray = _FakeControl("微信", "Button")
    root = _FakeControl(children=[non_tray, matching])

    assert _find_tray_control(_FakeAutomation(root), "weixin.exe") is matching
