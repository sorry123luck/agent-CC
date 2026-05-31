"""Anthropic-compatible VLM Semantic Provider."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any

import requests
from PIL import Image

from src.vlm.image_utils import compress_screenshot
from src.vlm.provider import (
    ProviderCapabilities,
    VLMSemanticRawResponse,
    VLMSemanticRequest,
)
from src.vlm.transport import create_vlm_session, normalize_proxy_url

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude API — 不支持 response_format=json_object，靠 prompt 约束。"""

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "claude-sonnet-4-20250514",
        thinking_mode: str = "auto",
        image_max_width: int = 1280,
        proxy_url: str = "",
        proxy_port: int = 0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint or "https://api.anthropic.com/v1/messages"
        self._model = model or "claude-sonnet-4-20250514"
        self._thinking_mode = thinking_mode
        self._image_max_width = image_max_width
        self._proxy_url = normalize_proxy_url(proxy_url=proxy_url, proxy_port=proxy_port)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return f"anthropic/{self._model}"

    @property
    def cost_tier(self) -> int:
        return 3

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_json_object=False,
            supports_json_schema=False,
            supports_image_url=False,
            supports_base64_image=True,
            supports_system_prompt=True,
            supports_multi_image=True,
            max_images_per_request=5,
            max_tokens_limit=8192,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
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
        compressed_img, img_fmt, _ = compress_screenshot(
            request.screenshot, max_width=request.screenshot.width,
        )
        buf = io.BytesIO()
        compressed_img.save(buf, format=img_fmt)
        b64_image = base64.b64encode(buf.getvalue()).decode("ascii")
        media_type = "image/png" if img_fmt == "PNG" else "image/jpeg"

        # Anthropic 用 system 顶层参数 + messages 数组
        system_text = request.system_prompt or ""
        messages: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            elif msg.get("role") == "user":
                parts = msg.get("content", [])
                content_blocks: list[dict[str, Any]] = []
                for part in parts:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            # 转换为 Anthropic base64 格式
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # data:image/png;base64,xxx
                                media_type = url.split(";")[0].split(":")[1]
                                data = url.split(",", 1)[1]
                            else:
                                media_type = "image/png"
                                data = url
                            content_blocks.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": data},
                            })
                        elif part.get("type") == "text":
                            content_blocks.append({"type": "text", "text": part.get("text", "")})
                        else:
                            content_blocks.append(part)
                    elif isinstance(part, str):
                        content_blocks.append({"type": "text", "text": part})

                # 如果没有图片，添加截图
                has_image = any(p.get("type") == "image" for p in content_blocks)
                if not has_image:
                    content_blocks.insert(0, {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64_image},
                    })

                messages.append({"role": "user", "content": content_blocks})

        if not messages:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_image}},
                    {"type": "text", "text": "请分析此页面。"},
                ],
            })

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": min(request.max_tokens, self.capabilities.max_tokens_limit),
            "messages": messages,
        }
        if system_text:
            payload["system"] = system_text

        # Thinking mode: Claude 支持 extended thinking
        if self._thinking_mode == "off":
            # Claude API 不支持显式禁用 thinking，但可以通过 system prompt 引导
            pass  # 不做额外处理
        elif self._thinking_mode == "on":
            payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}

        # provider_options 覆盖
        for k, v in request.provider_options.items():
            if k not in ("timeout",):
                payload[k] = v

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        t0 = time.monotonic()
        try:
            session = create_vlm_session(proxy_url=self._proxy_url)
            resp = session.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=request.provider_options.get("timeout", 60),
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Anthropic VLM call failed: %s", exc)
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error",
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            data = resp.json()
            # Anthropic 响应格式：content[0].text
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            usage = data.get("usage", {})
            return VLMSemanticRawResponse(
                raw_text=text,
                provider_name=self.name,
                model_name=self._model,
                token_input=usage.get("input_tokens", 0),
                token_output=usage.get("output_tokens", 0),
                latency_ms=latency_ms,
                finish_reason=data.get("stop_reason", "stop"),
            )
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning("Anthropic VLM response parse error: %s", exc)
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error", latency_ms=latency_ms,
            )
