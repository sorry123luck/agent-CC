"""VLM 图片压缩工具 — 减小 payload、降低延迟。"""

from __future__ import annotations

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_MAX_WIDTH = 1280
DEFAULT_JPEG_QUALITY = 85


def compress_screenshot(
    image: Image.Image,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[Image.Image, str, int]:
    """压缩截图用于 VLM 请求。

    策略：
    1. 宽度超过 max_width 时等比缩放（LANCZOS 保文字清晰）
    2. 转为 JPEG 减小体积（UI 截图用 JPEG 足够，VLM 不需要像素级精度）
    3. 返回 (压缩后图片, 格式, base64字节数)

    Args:
        image: 原始截图
        max_width: 最大宽度（像素），默认 1280。超过则缩放。
        jpeg_quality: JPEG 质量 1-100，默认 85。

    Returns:
        (compressed_image, format_name, estimated_base64_bytes)
    """
    img = image
    orig_w, orig_h = img.size

    # Step 1: 缩放
    if orig_w > max_width:
        ratio = max_width / orig_w
        new_h = int(orig_h * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)
        logger.debug("Image resized: %dx%d -> %dx%d", orig_w, orig_h, max_width, new_h)

    # Step 2: 确定输出格式
    # 有 alpha 通道 → PNG（JPEG 不支持透明）
    # 无 alpha → JPEG（更小）
    if img.mode in ("RGBA", "LA", "PA"):
        fmt = "PNG"
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    else:
        fmt = "JPEG"
        # 统一转 RGB（JPEG 不支持 P/LA 等模式）
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)

    data_bytes = buf.getvalue()
    # base64 编码后约 4/3 倍
    b64_bytes = int(len(data_bytes) * 4 / 3) + 4

    logger.debug(
        "Image compressed: %s q=%d, %d bytes (b64 ~%d bytes)",
        fmt, jpeg_quality, len(data_bytes), b64_bytes,
    )

    return img, fmt, b64_bytes


def image_to_base64_url(
    image: Image.Image,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """压缩截图并返回 data:image/...;base64,... URL。

    直接用于 OpenAI-compatible API 的 image_url 字段。
    """
    import base64

    _, fmt, _ = compress_screenshot(image, max_width=max_width, jpeg_quality=jpeg_quality)

    # 重新编码（compress_screenshot 返回的是 PIL Image）
    img = image
    if img.size[0] > max_width:
        ratio = max_width / img.size[0]
        img = img.resize((max_width, int(img.size[1] * ratio)), Image.LANCZOS)

    if fmt == "PNG":
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        media_type = "image/jpeg"

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{media_type};base64,{b64}"
