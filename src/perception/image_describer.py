"""Image describer using mimo-v2.5 API for UI element classification.

Calls a vision-capable model to describe cropped UI element images.
Used as a refinement step when OCR/OmniParser can't identify an element.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from src.common.config_manager import load_config

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages"
_DEFAULT_MODEL = "mimo-v2.5"
_DEFAULT_PROMPT = (
    "这是一个Windows桌面UI元素的截图。请简短描述这是什么类型的UI元素"
    "（图标/按钮/文字/输入框），以及在该软件中可能代表什么功能。"
    "用中文回答，50字以内。"
)


def _get_config() -> dict[str, Any]:
    """Load image_describe config from models.yaml."""
    cfg = load_config()
    # Access the raw dict from YAML — config_manager doesn't parse image_describe yet
    # Fall back to defaults if not configured
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "models.yaml",
        )
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        section = raw.get("models", {}).get("image_describe", {})
        return {
            "enabled": section.get("enabled", True),
            "endpoint": section.get("endpoint", _DEFAULT_ENDPOINT),
            "api_key": os.environ.get(
                section.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"), ""
            ),
            "model": section.get("model", _DEFAULT_MODEL),
            "max_tokens": section.get("max_tokens", 500),
            "timeout": section.get("timeout", 30),
            "prompt": section.get("prompt", _DEFAULT_PROMPT),
        }
    except Exception:
        return {
            "enabled": True,
            "endpoint": _DEFAULT_ENDPOINT,
            "api_key": os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
            "model": _DEFAULT_MODEL,
            "max_tokens": 500,
            "timeout": 30,
            "prompt": _DEFAULT_PROMPT,
        }


def describe_image(image: Image.Image, prompt: str | None = None) -> str:
    """Describe a PIL Image using mimo-v2.5 API.

    Args:
        image: PIL Image to describe.
        prompt: Optional custom prompt override.

    Returns:
        Text description of the image, or empty string on failure.
    """
    cfg = _get_config()
    if not cfg["enabled"] or not cfg["api_key"]:
        logger.warning("Image describer disabled or no API key")
        return ""

    # Encode image to base64 PNG
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt or cfg["prompt"],
                    },
                ],
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
    }

    try:
        with httpx.Client(timeout=cfg["timeout"]) as client:
            resp = client.post(cfg["endpoint"], json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from response content
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()

        # If thinking consumed all tokens, no text block
        logger.warning("Image describer returned no text content")
        return ""

    except Exception as e:
        logger.error("Image describer API call failed: %s", e)
        return ""
