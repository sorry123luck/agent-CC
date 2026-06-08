"""Tests for page compiler support semantic inference."""

from src.perception.page_compiler_builders import InteractionCanvasEngineSupportBuilder
from src.perception.page_compiler_models import SemanticRole


def test_window_control_is_container_not_title_bar():
    builder = InteractionCanvasEngineSupportBuilder()

    role = builder.infer_semantic_role("WindowControl", "", "WeChat")

    assert role == SemanticRole.CONTAINER


def test_plain_pane_control_is_layout_not_sidebar():
    builder = InteractionCanvasEngineSupportBuilder()

    role = builder.infer_semantic_role("PaneControl", "", "MMUIRenderSubWindowHW")

    assert role == SemanticRole.LAYOUT


def test_named_sidebar_pane_still_maps_to_sidebar():
    builder = InteractionCanvasEngineSupportBuilder()

    role = builder.infer_semantic_role("PaneControl", "", "Sidebar")

    assert role == SemanticRole.SIDEBAR


# --- R1: has_cjk removal tests ---


class TestListItemCjkNotChatItem:
    """ListItemControl + CJK text should NOT be CHAT_ITEM unless chat token matches."""

    def setup_method(self):
        self.builder = InteractionCanvasEngineSupportBuilder()

    def test_paint_file_menu_not_chat(self):
        """Paint 菜单项 '文件' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "文件", None)
        assert role == SemanticRole.LIST_ITEM

    def test_paint_edit_menu_not_chat(self):
        """Paint 菜单项 '编辑' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "编辑", None)
        assert role == SemanticRole.LIST_ITEM

    def test_paint_view_menu_not_chat(self):
        """Paint 菜单项 '查看' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "查看", None)
        assert role == SemanticRole.LIST_ITEM

    def test_generic_cjk_list_item_not_chat(self):
        """普通中文列表项 '苹果' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "苹果", None)
        assert role == SemanticRole.LIST_ITEM

    def test_settings_list_item_not_chat(self):
        """设置项 '设置' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "设置", None)
        assert role == SemanticRole.LIST_ITEM

    def test_music_list_item_not_chat(self):
        """音乐列表项 '音乐' 不应是 CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "音乐", None)
        assert role == SemanticRole.LIST_ITEM


class TestChatItemOnlyFromChatTokens:
    """CHAT_ITEM should only come from explicit chat tokens."""

    def setup_method(self):
        self.builder = InteractionCanvasEngineSupportBuilder()

    def test_chat_token_huihua(self):
        """'会话' 是 chat token → CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "会话列表", None)
        assert role == SemanticRole.CHAT_ITEM

    def test_chat_token_liaotian(self):
        """'聊天' 是 chat token → CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "聊天", None)
        assert role == SemanticRole.CHAT_ITEM

    def test_chat_token_lianxiren(self):
        """'联系人' 是 chat token → CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "联系人", None)
        assert role == SemanticRole.CHAT_ITEM

    def test_chat_token_qunliao(self):
        """'群聊' 是 chat token → CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "群聊", None)
        assert role == SemanticRole.CHAT_ITEM

    def test_chat_token_wenjianchuanshuzhushou(self):
        """'文件传输助手' 是 chat token → CHAT_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "文件传输助手", None)
        assert role == SemanticRole.CHAT_ITEM

    def test_non_chat_cjk_text_not_chat(self):
        """普通中文文本 '你好世界' 不是 chat token → LIST_ITEM."""
        role = self.builder.infer_semantic_role("ListItemControl", "你好世界", None)
        assert role == SemanticRole.LIST_ITEM


class TestTextControlCjkIsText:
    """TextControl + CJK should be TEXT, not CHAT_ITEM."""

    def setup_method(self):
        self.builder = InteractionCanvasEngineSupportBuilder()

    def test_paint_file_as_text_control(self):
        """TextControl + '文件' → TEXT."""
        role = self.builder.infer_semantic_role("TextControl", "文件", None)
        assert role == SemanticRole.TEXT

    def test_paint_100_percent(self):
        """TextControl + '100%' → TEXT."""
        role = self.builder.infer_semantic_role("TextControl", "100%", None)
        assert role == SemanticRole.TEXT

    def test_chrome_title_text(self):
        """TextControl + Chrome 标题 → TEXT."""
        role = self.builder.infer_semantic_role("TextControl", "首次调用 API", None)
        assert role == SemanticRole.TEXT


class TestSendSearchPreserved:
    """发送/搜索 should still map correctly."""

    def setup_method(self):
        self.builder = InteractionCanvasEngineSupportBuilder()

    def test_send_button_control(self):
        """ButtonControl + '发送' → SEND_BUTTON."""
        role = self.builder.infer_semantic_role("ButtonControl", "发送", None)
        assert role == SemanticRole.SEND_BUTTON

    def test_search_edit_control(self):
        """EditControl + '搜索' → SEARCH_INPUT."""
        role = self.builder.infer_semantic_role("EditControl", "搜索", None)
        assert role == SemanticRole.SEARCH_INPUT

    def test_send_text_control(self):
        """TextControl + '发送' → TEXT (TextControl matches 'text' pattern first)."""
        role = self.builder.infer_semantic_role("TextControl", "发送", None)
        assert role == SemanticRole.TEXT

    def test_search_text_control(self):
        """TextControl + '搜索' → SEARCH_INPUT (text-level fallback)."""
        role = self.builder.infer_semantic_role("TextControl", "搜索", None)
        assert role == SemanticRole.SEARCH_INPUT
