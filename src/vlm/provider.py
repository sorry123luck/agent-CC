"""VLM Semantic Provider abstraction — Protocol + 请求/响应数据类 + 工厂。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 请求 / 响应
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VLMSemanticRequest:
    """统一承载所有输入，不只传 prompt 字符串。"""
    screenshot: Image.Image
    messages: list[dict[str, Any]]
    system_prompt: str
    max_tokens: int = 8192
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMSemanticRawResponse:
    """统一承载原始响应。"""
    raw_text: str
    provider_name: str
    model_name: str
    token_input: int = 0
    token_output: int = 0
    latency_ms: int = 0
    finish_reason: str = ""
    capabilities_used: frozenset[str] = frozenset()
    error: str = ""


# ---------------------------------------------------------------------------
# 能力声明
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider 能力声明，不假设所有 provider 都支持相同功能。"""
    supports_json_object: bool = False
    supports_json_schema: bool = False
    supports_image_url: bool = True
    supports_base64_image: bool = True
    supports_system_prompt: bool = True
    supports_multi_image: bool = False
    max_images_per_request: int = 1
    max_tokens_limit: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    is_free: bool = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VLMSemanticProvider(Protocol):
    """所有 VLM Semantic provider 必须实现的接口。"""

    @property
    def name(self) -> str: ...

    @property
    def model_id(self) -> str:
        """provider + model 的唯一标识，如 'openai/gpt-4o', 'minimax/MiniMax-VL-01'。"""
        ...

    @property
    def cost_tier(self) -> int:
        """1=最便宜（本地），2=中等（MiniMax/MiMo），3=贵（GPT-4o/Claude）"""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def is_available(self) -> bool: ...

    def analyze_page(self, request: VLMSemanticRequest) -> VLMSemanticRawResponse:
        """发送请求，返回原始响应。不做 JSON 解析。"""
        ...


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def create_modeler_provider(
    provider: str,
    api_key: str = "",
    endpoint: str = "",
    model: str = "",
    provider_variant: str = "",
    free_model_only: bool = False,
    thinking_mode: str = "auto",
    image_max_width: int = 1280,
    proxy_url: str = "",
    proxy_port: int = 0,
) -> VLMSemanticProvider | None:
    """根据配置创建 provider。

    Args:
        provider: "openai" / "openrouter" / "anthropic" / "minimax" / "mimo" / "mock" / "disabled"
        api_key: API key
        endpoint: 自定义 endpoint
        model: 模型名
        provider_variant: OpenRouter 等兼容 API 的 variant 标识
        free_model_only: 只使用免费模型
        thinking_mode: "auto" / "on" / "off"
        image_max_width: 图片最大宽度（像素），超过则压缩
        proxy_url: VLM API proxy URL; empty means direct transport
        proxy_port: Optional localhost proxy port, e.g. 7890

    Returns:
        VLMSemanticProvider 实例，或 None（disabled / 未知 provider）
    """
    if provider == "disabled" or not provider:
        return None

    if provider in ("openai", "openrouter", "moonshot", "qwen", "doubao", "openai_compatible"):
        from src.vlm.providers.openai_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            provider_variant=provider_variant or provider,
            free_model_only=free_model_only,
            thinking_mode=thinking_mode,
            image_max_width=image_max_width,
            proxy_url=proxy_url,
            proxy_port=proxy_port,
        )

    if provider == "anthropic":
        from src.vlm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            thinking_mode=thinking_mode,
            image_max_width=image_max_width,
            proxy_url=proxy_url,
            proxy_port=proxy_port,
        )

    if provider in ("minimax", "mimo"):
        from src.vlm.providers.minimax_provider import MiniMaxProvider
        return MiniMaxProvider(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            image_max_width=image_max_width,
            proxy_url=proxy_url,
            proxy_port=proxy_port,
        )

    if provider == "mock":
        from src.vlm.providers.mock_provider import MockProvider
        return MockProvider()

    logger.warning("Unknown VLM Semantic provider: %s", provider)
    return None
