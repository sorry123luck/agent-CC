"""VLM Semantic Modeler — 响应脱敏。"""

from __future__ import annotations

import json
import re
from typing import Any

from src.vlm.schema import PageSemanticModel


# ---------------------------------------------------------------------------
# Raw response 脱敏
# ---------------------------------------------------------------------------

# 敏感文本模式（电话、邮箱、身份证、银行卡等）
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{11}\b"),                                    # 手机号
    re.compile(r"\b\d{15,19}\b"),                                  # 身份证/银行卡
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # 邮箱
]


def _redact_string(text: str) -> str:
    """对单个字符串进行脱敏。"""
    result = text
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def _redact_value(value: Any) -> Any:
    """递归脱敏 JSON 值。"""
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    return value


def redact_raw_response(
    raw_response: str,
    *,
    keep_structure: bool = True,
    redact_dynamic_content: bool = True,
) -> str:
    """脱敏原始响应。

    Args:
        raw_response: VLM 返回的原始 JSON 字符串。
        keep_structure: 保留 JSON 结构，将敏感文本值替换为 [REDACTED]。
        redact_dynamic_content: dynamic_zones 中的 note 不保留原文。

    Returns:
        脱敏后的字符串。
    """
    if not keep_structure:
        return "[REDACTED]"

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # JSON 解析失败，对整个字符串脱敏
        return _redact_string(raw_response)

    if redact_dynamic_content:
        # 清空 dynamic_zones 中的 note
        for zone in data.get("dynamic_zones", []):
            if "note" in zone:
                zone["note"] = "[DYNAMIC_CONTENT_REDACTED]"

    # 对所有字符串值进行敏感信息脱敏
    redacted = _redact_value(data)
    return json.dumps(redacted, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PageSemanticModel 脱敏
# ---------------------------------------------------------------------------


def sanitize_for_storage(model: PageSemanticModel) -> dict[str, Any]:
    """返回适合长期存储的 dict。

    - dynamic_zones: 只保留 zone_id、region_id、content_type、bounds，清空 note
    - fixed_controls: text 字段保留（UI 控件文字不是隐私）
    - transitions: 保留
    - 其他字段原样保留
    """
    d = model.to_dict()

    # 清空 dynamic_zones 的 note
    for zone in d.get("dynamic_zones", []):
        zone["note"] = "[REDACTED_FOR_STORAGE]"

    return d
