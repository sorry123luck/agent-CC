"""MiniMax / MiMo VLM Semantic Provider — OpenAI-compatible 格式。"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any

import requests
from PIL import Image

from src.vlm.image_utils import image_to_base64_url
from src.vlm.provider import (
    ProviderCapabilities,
    VLMSemanticRawResponse,
    VLMSemanticRequest,
)
from src.vlm.transport import create_vlm_session, normalize_proxy_url

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.minimax.chat/v1/text/chatcompletion_v2"


class MiniMaxProvider:
    """MiniMax / MiMo API — OpenAI-compatible，可配置 auth header。"""

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "MiniMax-VL-01",
        image_max_width: int = 1280,
        proxy_url: str = "",
        proxy_port: int = 0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint or _DEFAULT_ENDPOINT
        self._model = model or "MiniMax-VL-01"
        self._image_max_width = image_max_width
        self._proxy_url = normalize_proxy_url(proxy_url=proxy_url, proxy_port=proxy_port)

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def model_id(self) -> str:
        return f"minimax/{self._model}"

    @property
    def cost_tier(self) -> int:
        return 2

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_json_object=True,
            supports_json_schema=False,
            supports_image_url=True,
            supports_base64_image=True,
            supports_system_prompt=True,
            supports_multi_image=True,
            max_images_per_request=5,
            max_tokens_limit=4096,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )

    def is_available(self) -> bool:
        return bool(self._api_key)

    def analyze_page(self, request: VLMSemanticRequest) -> VLMSemanticRawResponse:
        if not self.is_available():
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error",
            )

        # The modeler/prompt layer prepares the actual VLM image size and
        # matching coordinates. Do not resize again in the provider.
        image_url = image_to_base64_url(
            request.screenshot, max_width=request.screenshot.width,
        )

        # 构建 messages（system + user）— OpenAI-compatible 格式
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        user_content: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.get("role") == "user":
                for part in msg.get("content", []):
                    if isinstance(part, dict):
                        user_content.append(part)
                    elif isinstance(part, str):
                        user_content.append({"type": "text", "text": part})

        # 如果没有图片，添加截图
        has_image = any(p.get("type") == "image_url" for p in user_content)
        if not has_image:
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": min(request.max_tokens, self.capabilities.max_tokens_limit),
        }

        # response_format
        if self.capabilities.supports_json_object:
            payload.setdefault("response_format", {"type": "json_object"})

        # provider_options 覆盖（过滤 transport 参数）
        _TRANSPORT_KEYS = frozenset({"timeout"})
        for k, v in request.provider_options.items():
            if k not in _TRANSPORT_KEYS:
                payload[k] = v

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        timeout = request.provider_options.get("timeout", 60)

        t0 = time.monotonic()
        try:
            session = create_vlm_session(proxy_url=self._proxy_url)
            resp = session.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("MiniMax VLM call failed: %s", exc)
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error",
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return VLMSemanticRawResponse(
                raw_text=text,
                provider_name=self.name,
                model_name=self._model,
                token_input=usage.get("prompt_tokens", 0),
                token_output=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning("MiniMax VLM response parse error: %s", exc)
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error", latency_ms=latency_ms,
            )
