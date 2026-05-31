"""Runtime validation matrix definitions.

The real app runner is gated by OPENCLAW_RUN_REAL_APP_TESTS. These tests keep the
scenario/query/metric contract executable even when real desktop automation is
disabled on CI or a developer machine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeScenario:
    app: str
    label: str
    required_queries: tuple[str, ...]


RUNTIME_SCENARIOS: tuple[RuntimeScenario, ...] = (
    RuntimeScenario("wechat", "WeChat empty input", ("role:search_input", "role:message_input", "text:发送", "natural:左侧导航入口")),
    RuntimeScenario("wechat", "WeChat input with text", ("role:message_input", "text:发送", "natural:顶部更多按钮")),
    RuntimeScenario("chrome", "Chrome active tab/address bar", ("role:search_input", "natural:顶部导航入口")),
    RuntimeScenario("explorer", "File Explorer list/detail", ("natural:主内容区域", "natural:文件列表")),
    RuntimeScenario("notepad", "Notepad editor/menu", ("natural:编辑区域", "text:文件")),
    RuntimeScenario("taskmgr", "Task Manager native UI", ("natural:进程列表", "natural:顶部标签")),
    RuntimeScenario("openclaw", "DeskCanvas console self-observe", ("natural:窗口列表", "natural:画布缓存")),
)


RUNTIME_METRICS: tuple[str, ...] = (
    "observe_latency_ms",
    "ocr_latency_ms",
    "vision_latency_ms",
    "vlm_latency_ms",
    "element_count",
    "region_count",
    "visual_region_count",
    "text_coverage",
    "actionable_count",
    "safe_query_success_rate",
    "manual_anchor_conflicts",
    "model_split_count",
)


def test_runtime_matrix_covers_required_scenarios():
    labels = {scenario.label for scenario in RUNTIME_SCENARIOS}

    assert "WeChat empty input" in labels
    assert "WeChat input with text" in labels
    assert "Chrome active tab/address bar" in labels
    assert "File Explorer list/detail" in labels
    assert "Notepad editor/menu" in labels
    assert "Task Manager native UI" in labels
    assert "DeskCanvas console self-observe" in labels


def test_runtime_matrix_queries_cover_core_agent_intents():
    all_queries = {query for scenario in RUNTIME_SCENARIOS for query in scenario.required_queries}

    assert "role:search_input" in all_queries
    assert "role:message_input" in all_queries
    assert "text:发送" in all_queries
    assert "natural:顶部更多按钮" in all_queries
    assert "natural:左侧导航入口" in all_queries
    assert "natural:主内容区域" in all_queries


def test_runtime_matrix_metrics_match_quality_report_contract():
    required = {
        "observe_latency_ms",
        "ocr_latency_ms",
        "vision_latency_ms",
        "vlm_latency_ms",
        "element_count",
        "region_count",
        "visual_region_count",
        "text_coverage",
        "actionable_count",
        "safe_query_success_rate",
        "manual_anchor_conflicts",
        "model_split_count",
    }

    assert required.issubset(set(RUNTIME_METRICS))
