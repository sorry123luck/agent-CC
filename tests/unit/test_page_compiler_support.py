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
