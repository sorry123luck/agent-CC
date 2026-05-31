"""Mock VLM Semantic Provider — 测试用。"""

from __future__ import annotations

import json
from typing import Any

from src.vlm.provider import (
    ProviderCapabilities,
    VLMSemanticRawResponse,
    VLMSemanticRequest,
)


def _default_page_json() -> dict[str, Any]:
    """返回一份合法的页面语义理解结果。"""
    return {
        "schema_version": "1.0",
        "app_identity": {"app_name": "测试应用", "app_id": "test_app", "surface_type": "native_uia"},
        "page_state": {"page_class": "home", "state_label": "主页", "state_flags": []},
        "regions": [
            {"region_id": "top_bar", "role": "navigation", "bounds": [0, 0, 800, 40], "purpose": "顶部导航栏"},
            {"region_id": "main_area", "role": "content", "bounds": [0, 40, 800, 600], "purpose": "主内容区"},
        ],
        "fixed_controls": [
            {
                "control_id": "ctrl_1",
                "region_id": "top_bar",
                "control_type": "button",
                "text": "设置",
                "bounds": [700, 5, 790, 35],
                "semantic_role": "settings_button",
                "visual_type": "button",
                "interactable": True,
                "confidence": 0.9,
                "source_candidate_ids": [],
                "matches_candidate_id": None,
            },
        ],
        "dynamic_zones": [],
        "candidate_corrections": [],
        "transitions": [],
        "confidence": 0.8,
        "needs_review": False,
    }


def _new_page_json() -> dict[str, Any]:
    """模拟新页面（需要 review）。"""
    result = _default_page_json()
    result["needs_review"] = True
    result["confidence"] = 0.5
    return result


def _drift_page_json() -> dict[str, Any]:
    """模拟漂移页面（有候选修正）。"""
    result = _default_page_json()
    result["candidate_corrections"] = [
        {"candidate_id": "uia_3", "corrected_type": "icon", "corrected_role": "close_button", "corrected_confidence": 0.7, "reason": "VLM 识别为关闭图标"},
    ]
    result["needs_review"] = True
    return result


class MockProvider:
    """Mock provider — 返回预定义 JSON，支持注入场景。"""

    def __init__(
        self,
        scenario: str = "default",
        fail: bool = False,
        truncated_json: bool = False,
        markdown_wrapped: bool = False,
    ) -> None:
        self._scenario = scenario
        self._fail = fail
        self._truncated_json = truncated_json
        self._markdown_wrapped = markdown_wrapped
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_id(self) -> str:
        return "mock/mock-model"

    @property
    def cost_tier(self) -> int:
        return 1

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_json_object=True,
            supports_json_schema=True,
            supports_multi_image=True,
            max_images_per_request=3,
            is_free=True,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )

    def is_available(self) -> bool:
        return not self._fail

    def analyze_page(self, request: VLMSemanticRequest) -> VLMSemanticRawResponse:
        self._call_count += 1

        if self._fail:
            return VLMSemanticRawResponse(
                raw_text="",
                provider_name="mock",
                model_name="mock-model",
                finish_reason="error",
            )

        scenarios: dict[str, dict[str, Any]] = {
            "default": _default_page_json(),
            "new_page": _new_page_json(),
            "drift": _drift_page_json(),
        }
        data = scenarios.get(self._scenario, _default_page_json())

        raw_text = json.dumps(data, ensure_ascii=False)

        if self._truncated_json:
            raw_text = raw_text[:len(raw_text) // 2]

        if self._markdown_wrapped:
            raw_text = f"```json\n{raw_text}\n```"

        return VLMSemanticRawResponse(
            raw_text=raw_text,
            provider_name="mock",
            model_name="mock-model",
            token_input=500,
            token_output=300,
            latency_ms=100,
            finish_reason="stop",
        )
