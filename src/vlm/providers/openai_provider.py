"""OpenAI-compatible VLM Semantic Provider — 含 OpenRouter。"""

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
from src.vlm.trace_artifacts import (
    sanitize_payload_for_trace,
    save_data_url_image,
    save_full_payload_enabled,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)


def _normalize_chat_completions_endpoint(endpoint: str, provider_variant: str = "") -> str:
    """Accept either a provider base URL or a full chat completions URL."""
    value = (endpoint or "").strip().rstrip("/")
    if not value:
        defaults = {
            "moonshot": "https://api.moonshot.cn/v1/chat/completions",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        }
        return defaults.get(provider_variant, "https://api.openai.com/v1/chat/completions")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1") or value.endswith("/api/v1"):
        return f"{value}/chat/completions"
    if value in {"https://api.openai.com", "https://openrouter.ai"}:
        suffix = "api/v1" if "openrouter.ai" in value else "v1"
        return f"{value}/{suffix}/chat/completions"
    if provider_variant in {"openai", "openrouter", "moonshot", "qwen", "doubao", "openai_compatible"} and "chat/completions" not in value:
        return f"{value}/chat/completions"
    return value


class OpenAICompatibleProvider:
    """OpenAI / OpenRouter / 兼容 API。"""

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "gpt-4o",
        provider_variant: str = "",
        free_model_only: bool = False,
        thinking_mode: str = "auto",
        image_max_width: int = 1280,
        proxy_url: str = "",
        proxy_port: int = 0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = _normalize_chat_completions_endpoint(endpoint, provider_variant)
        self._model = model or "gpt-4o"
        self._provider_variant = provider_variant
        self._free_model_only = free_model_only
        self._thinking_mode = thinking_mode
        self._image_max_width = image_max_width
        self._proxy_url = normalize_proxy_url(proxy_url=proxy_url, proxy_port=proxy_port)

    @property
    def name(self) -> str:
        if self._provider_variant:
            return self._provider_variant
        return "openai"

    @property
    def model_id(self) -> str:
        return f"{self.name}/{self._model}"

    @property
    def cost_tier(self) -> int:
        return 1 if self._is_free_model() else 3

    def _is_free_model(self) -> bool:
        """模型本身是否免费：仅看 :free 后缀，不考虑 free_model_only 策略。"""
        return self._model.endswith(":free")

    def _is_moonshot_kimi_k2(self) -> bool:
        model = self._model.lower()
        return self.name == "moonshot" and (
            model.startswith("kimi-k2")
            or model.startswith("kimi-k2.")
        )

    def _supports_json_object_payload(self) -> bool:
        """Whether it is safe to send OpenAI's response_format parameter.

        Many "OpenAI-compatible" providers accept chat/completions but do not
        support every OpenAI parameter. Moonshot/Kimi is better constrained by
        prompt-only JSON in this pipeline; forcing response_format can turn a
        valid multimodal call into a provider-specific compatibility problem.
        """
        return self.name not in {"moonshot", "qwen"}

    @property
    def capabilities(self) -> ProviderCapabilities:
        is_free = self._is_free_model()
        return ProviderCapabilities(
            supports_json_object=self._supports_json_object_payload(),
            supports_json_schema=self._supports_json_object_payload(),
            supports_image_url=True,
            supports_base64_image=True,
            supports_system_prompt=True,
            supports_multi_image=True,
            max_images_per_request=10,
            max_tokens_limit=16384,
            cost_per_1k_input=0.0 if is_free else 0.005,
            cost_per_1k_output=0.0 if is_free else 0.015,
            is_free=is_free,
        )

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        # free_model_only 策略：非免费模型直接不可用
        if self._free_model_only and not self._is_free_model():
            return False
        return True

    @staticmethod
    def _request_stats(payload: dict[str, Any]) -> dict[str, int | bool]:
        """Collect request diagnostics for UI/debugging without leaking data."""
        image_count = 0
        image_bytes = 0
        text_chars = 0
        for msg in payload.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                text_chars += len(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_chars += len(str(part.get("text", "")))
                if part.get("type") == "image_url":
                    image_count += 1
                    url = str((part.get("image_url") or {}).get("url", ""))
                    if "," in url:
                        b64 = url.split(",", 1)[1]
                        image_bytes += int(len(b64) * 3 / 4)
                    else:
                        image_bytes += len(url)
        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except TypeError:
            payload_bytes = 0
        return {
            "image_count": image_count,
            "image_bytes": image_bytes,
            "text_chars": text_chars,
            "payload_bytes": payload_bytes,
            "has_response_format": "response_format" in payload,
            "has_thinking": "thinking" in payload,
        }

    def analyze_page(self, request: VLMSemanticRequest) -> VLMSemanticRawResponse:
        if not self.is_available():
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model, finish_reason="error",
                error="provider_unavailable_or_free_model_only_blocked",
            )

        # The modeler/prompt layer is responsible for choosing the actual VLM
        # image size and matching all prompt coordinates to it. Do not resize
        # again here; only encode the provided image.
        image_url = image_to_base64_url(
            request.screenshot, max_width=request.screenshot.width, jpeg_quality=85,
        )

        trace_dir = request.provider_options.get("_trace_dir")
        sent_image_meta = None

        if trace_dir:
            sent_image_meta = save_data_url_image(
                trace_dir,
                "03_vlm_sent_image",
                image_url,
            )

        # 构建 messages（system + user）
        messages: list[dict[str, Any]] = []
        if self.capabilities.supports_system_prompt and request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # user content: 文本 + 图片
        user_content: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.get("role") == "user":
                for part in msg.get("content", []):
                    if isinstance(part, dict):
                        user_content.append(part)
                    elif isinstance(part, str):
                        user_content.append({"type": "text", "text": part})

        # 如果 messages 中没有 image，手动添加截图
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

        # Thinking mode: 通用配置，不针对特定模型做 hack。
        # thinking_mode="off" → 所有模型禁用思考（快速模式）
        # thinking_mode="on"  → 启用（如果模型支持）
        # thinking_mode="auto" → 语义理解不需要深度推理，默认禁用
        if self._thinking_mode == "off" or self._thinking_mode == "auto":
            payload["thinking"] = {"type": "disabled"}
            # 禁用 thinking 时限制 max_tokens，避免模型生成过多思考内容
            if self._thinking_mode == "auto":
                payload["max_tokens"] = min(payload["max_tokens"], 8192)

        # response_format（如果 provider 支持）
        if self.capabilities.supports_json_object:
            payload.setdefault("response_format", {"type": "json_object"})

        # provider_options 覆盖（过滤 transport / OpenClaw-internal 参数）
        _TRANSPORT_KEYS = frozenset({"timeout", "_trace_dir"})
        _INTERNAL_KEYS = frozenset({"response_contract", "roi_profile"})
        for k, v in request.provider_options.items():
            if k not in _TRANSPORT_KEYS and k not in _INTERNAL_KEYS:
                payload[k] = v

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "http://127.0.0.1/openclaw"
            headers["X-Title"] = "OpenClaw Desktop Agent"

        timeout = request.provider_options.get("timeout", 60)
        request_stats = self._request_stats(payload)

        if trace_dir:
            sanitized_payload = sanitize_payload_for_trace(payload, sent_image_meta)
            write_json(trace_dir, "04_provider_payload_sanitized.json", sanitized_payload)

            if save_full_payload_enabled():
                write_json(trace_dir, "04_provider_payload_full.json", payload)

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

            if trace_dir:
                response_record = {
                    "status_code": resp.status_code,
                    "headers": dict(resp.headers),
                    "text": resp.text,
                }
                try:
                    response_record["json"] = resp.json()
                except Exception:
                    response_record["json"] = None

                write_json(trace_dir, "05_http_response_full.json", response_record)
        except requests.RequestException as exc:
            status = ""
            body = ""
            if getattr(exc, "response", None) is not None:
                status = f"HTTP {exc.response.status_code}"
                body = (exc.response.text or "")[:1000]
            detail = f"{status} {body}".strip() or str(exc)
            stats_text = ", ".join(f"{k}={v}" for k, v in request_stats.items())
            detail = f"{detail}; request_stats: {stats_text}"
            logger.warning("OpenAI-compatible VLM call failed: %s", detail)

            if trace_dir:
                error_record = {
                    "error": str(exc),
                    "detail": detail,
                    "request_stats": request_stats,
                }

                if getattr(exc, "response", None) is not None:
                    error_record["status_code"] = exc.response.status_code
                    error_record["response_text"] = exc.response.text

                write_json(trace_dir, "05_http_response_full.json", error_record)

            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error",
                latency_ms=int((time.monotonic() - t0) * 1000),
                error=detail,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]

            if trace_dir:
                write_text(trace_dir, "06_assistant_raw_text.txt", text)

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
            preview = ""
            try:
                preview = resp.text[:1000]
            except Exception:
                pass
            detail = f"{exc}; response={preview}".strip()
            logger.warning("OpenAI-compatible VLM response parse error: %s", detail)
            return VLMSemanticRawResponse(
                raw_text="", provider_name=self.name, model_name=self._model,
                finish_reason="error", latency_ms=latency_ms, error=detail,
            )
