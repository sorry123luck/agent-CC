"""VLM Refine POC v2: B+ vs C-lite with cleaned prompt and metadata.

Key changes from v1:
- OmniParser captions marked as untrusted; VLM told to ignore them
- Metadata splits ocr_text (trusted) vs omni_caption_untrusted
- role_label in Chinese (人类可读)
- semantic_tags in dot notation (机器稳定标签)
- Confidence threshold: < 0.7 → uncertain, allow unknown_icon/unknown_button
- B+: annotated screenshot + cleaned metadata
- C-lite: B+ + crop atlas only for low-confidence/icon candidates

Usage: python scripts/vlm_refine_poc_v2.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO

import httpx
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

API_BASE = "http://127.0.0.1:8000/api/v1"
VLM_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages"
VLM_MODEL = "mimo-v2.5"
VLM_API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

CANVAS_ID = ""

# 25 key elements from previous POC
TARGET_IDS = [
    "vision_0", "vision_2", "vision_8", "vision_19", "vision_24",
    "vision_25", "vision_36", "vision_38", "vision_44", "vision_45",
    "vision_46", "vision_47", "vision_48", "vision_51", "vision_52",
    "vision_55", "vision_62", "vision_63", "vision_64", "vision_65",
    "vision_69", "vision_73", "vision_77", "vision_81", "vision_83",
]

# Confidence threshold for uncertain marking
CONFIDENCE_THRESHOLD = 0.7

# Elements that are likely icons or low-confidence → include crop atlas for these
ICON_OR_LOW_CONF_IDS = {
    "vision_8",   # ↓ arrow
    "vision_19",  # W shape
    "vision_24",  # ● dot
    "vision_44",  # +
    "vision_45",  # folder icon
    "vision_46",  # smiley
    "vision_47",  # ×
    "vision_48",  # 章 (OmniParser misread)
    "vision_52",  # garbled text
    "vision_62",  # microphone
    "vision_63",  # minimize
    "vision_64",  # green bubble
    "vision_65",  # magnifier
    "vision_69",  # Shuffle (OmniParser misread)
    "vision_73",  # pencil
    "vision_77",  # 3D modeling (OmniParser misread)
    "vision_81",  # PowerPoint (OmniParser misread)
    "vision_83",  # Hide
}


@dataclass
class VLMRefineResult:
    element_id: str
    visual_type: str = "unknown"
    role_label: str = ""
    semantic_tags: list[str] = field(default_factory=list)
    role_confidence: float = 0.0
    role_evidence: str = ""
    refine_status: str = "unreviewed"


@dataclass
class ExperimentResult:
    approach: str
    results: list[VLMRefineResult] = field(default_factory=list)
    raw_response: str = ""
    parse_ok: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    element_count: int = 0
    error: str = ""


# ── Helpers ──────────────────────────────────────────────────────────


def get_canvas_data() -> dict:
    resp = httpx.get(f"{API_BASE}/canvases/{CANVAS_ID}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_screenshot() -> Image.Image:
    resp = httpx.get(f"{API_BASE}/canvases/{CANVAS_ID}/screenshot", timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def get_crop(element_id: str) -> Image.Image:
    resp = httpx.get(
        f"{API_BASE}/canvases/{CANVAS_ID}/candidates/{element_id}/crop",
        timeout=30,
    )
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_font(size: int = 14):
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Metadata Builder (v2: split trusted/untrusted) ──────────────────


def _extract_ocr_text(el: dict) -> str:
    """Extract real OCR/UIA text (trusted)."""
    parts = []
    # UIA text fields
    for key in ("text", "name", "value", "placeholder"):
        val = el.get(key, "")
        if val and val.strip():
            parts.append(val.strip())
    # Also check attributes for additional UIA info
    attrs = el.get("attributes", {})
    for key in ("help_text", "accelerator_key", "access_key"):
        val = attrs.get(key, "")
        if val and val.strip():
            parts.append(val.strip())
    return " | ".join(parts) if parts else ""


def _extract_omni_caption(el: dict) -> str:
    """Extract OmniParser/Firenze caption (untrusted)."""
    # OmniParser captions typically end up in 'text' when provider is vision
    # or in attributes.caption
    sources = el.get("provider_sources", [])
    attrs = el.get("attributes", {})
    caption = attrs.get("caption", "")
    if caption:
        return caption
    # If the text came from vision provider only, it's likely an OmniParser caption
    if sources == ["vision"] and el.get("text"):
        return el["text"]
    return ""


def _determine_trusted_text(el: dict) -> tuple[str, str]:
    """Return (trusted_text, untrusted_caption) for an element."""
    sources = el.get("provider_sources", [])
    ocr_text = ""
    omni_caption = ""

    # If UIA is a source, text/name are trustworthy
    if "uia" in sources:
        ocr_text = _extract_ocr_text(el)
        # If vision also contributed, its caption is untrusted
        if "vision" in sources:
            omni_caption = el.get("attributes", {}).get("caption", "")
    elif "ocr" in sources:
        # OCR text is trusted (real text extraction)
        ocr_text = el.get("text", "")
    elif "vision" in sources:
        # Vision-only: caption is untrusted
        omni_caption = _extract_omni_caption(el) or el.get("text", "")

    return ocr_text, omni_caption


def build_metadata_text_v2(canvas_data: dict, target_ids: list[str]) -> str:
    """Build candidate metadata with ocr_text vs omni_caption split."""
    elements = {e["element_id"]: e for e in canvas_data.get("elements", [])}
    lines = []
    for eid in target_ids:
        el = elements.get(eid)
        if not el:
            continue

        bounds = el.get("bounds", [])
        w = bounds[2] - bounds[0] if len(bounds) >= 4 else 0
        h = bounds[3] - bounds[1] if len(bounds) >= 4 else 0
        conf = el.get("confidence", 0)
        sources = el.get("provider_sources", [])

        ocr_text, omni_caption = _determine_trusted_text(el)

        # Region context
        region_id = el.get("region_id", "")
        semantic_role = el.get("semantic_role", "")

        line = f"- {eid}: {w}x{h} src={sources} conf={conf:.2f}"
        if semantic_role and semantic_role != "unknown":
            line +=" role={semantic_role}"
        if region_id:
            line += f" region={region_id}"
        if ocr_text:
            line += f' ocr_text="{ocr_text}"'
        if omni_caption:
            line += f' omni_caption_untrusted="{omni_caption}"'

        lines.append(line)

    return "\n".join(lines)


# ── Prompt Builder (v2: explicit OmniParser distrust) ────────────────


def build_vlm_prompt_v2(approach: str, metadata: str) -> str:
    """Build VLM prompt that explicitly distrusts OmniParser captions."""

    system_context = """你是一个 Windows 桌面 UI 元素语义分析专家。

## 重要规则

1. **OmniParser/Firenze 的文字标签不可信**。这些标签经常把图标误识别为无关文字（如把设置齿轮识别为"章"，把电话图标识别为"PowerPoint"）。不要根据这些标签判断元素功能。

2. **只根据以下信息判断**：
   - 截图中元素的**视觉外观**（形状、颜色、图标样式）
   - 元素在界面中的**位置**（顶部栏、侧边栏、底部输入区等）
   - **上下文**（相邻元素是什么、所在区域的功能）
   - **真实 OCR/UIA 文本**（标记为 ocr_text 的字段，这是从界面实际提取的文字）

3. **omni_caption_untrusted 字段仅供参考**，不要作为判断依据。如果它与视觉不符，以视觉为准。

4. **低可信度处理**：
   - 如果你对某个元素的功能不确定，role_confidence 应 < 0.7
   - 不确定的元素用 refine_status: "uncertain"
   - 完全无法识别的小图标可以用 "未知图标" / "未知按钮" 作为 role_label

## 返回格式

返回 JSON 数组，每个元素包含：
- element_id: 元素ID（必须使用完整的 vision_XX 格式）
- visual_type: icon / icon_button / button / text / input / indicator / unknown
- role_label: **中文**语义角色标签（如 "搜索框"、"发送按钮"、"添加按钮"）
- semantic_tags: **机器标签**列表，用点分隔命名空间（如 ["input.search", "action.send"]）
- role_confidence: 置信度 0-1（低于 0.7 标记为 uncertain）
- role_evidence: 判断依据（中文简短描述，说明你根据什么视觉特征判断的）
- refine_status: "refined"（确认）/ "uncertain"（不确定）

只返回 JSON 数组，不要其他文字。"""

    if approach == "B+":
        return f"""{system_context}

## 任务

这是一个微信桌面版的截图，每个 UI 元素的边界框已标注 element_id（如 v46、v52）。
请根据截图中看到的**视觉内容**，为以下候选元素返回语义分析结果。

**不要参考 omni_caption_untrusted 字段**，只根据视觉外观、位置、上下文和 ocr_text 判断。

候选元素元数据：
{metadata}"""

    else:  # C-lite
        return f"""{system_context}

## 任务

我提供了两张图：
1. 带编号边界框的微信全截图（标注了每个候选元素的位置和 ID）
2. 部分候选元素的裁剪图集（仅包含难以识别的图标和低可信元素）

**裁剪图集的用途**：让你更清楚地看到小图标的细节。对于已经有明确文字的元素，不需要看裁剪图。

**不要参考 omni_caption_untrusted 字段**，只根据视觉外观、位置、上下文和 ocr_text 判断。

候选元素元数据：
{metadata}"""


# ── Image Generation ─────────────────────────────────────────────────


def generate_annotated_screenshot(
    screenshot: Image.Image,
    canvas_data: dict,
    target_ids: list[str],
) -> Image.Image:
    img = screenshot.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(12)
    elements = {e["element_id"]: e for e in canvas_data.get("elements", [])}
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#008000", "#FF69B4", "#4169E1",
    ]
    for i, eid in enumerate(target_ids):
        el = elements.get(eid)
        if not el or not el.get("bounds") or len(el["bounds"]) < 4:
            continue
        l, t, r, b = el["bounds"]
        color = colors[i % len(colors)]
        draw.rectangle([l, t, r, b], outline=color, width=2)
        label = eid.replace("vision_", "v")
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, t - th - 4)
        draw.rectangle([l, label_y, l + tw + 6, label_y + th + 4], fill=color)
        draw.text((l + 3, label_y + 2), label, fill="white", font=font)
    return img


def generate_crop_atlas(
    crops: dict[str, Image.Image],
    cols: int = 8,
    cell_size: int = 80,
    padding: int = 4,
) -> Image.Image:
    n = len(crops)
    rows = (n + cols - 1) // cols
    atlas_w = cols * (cell_size + padding) + padding
    atlas_h = rows * (cell_size + padding + 18) + padding
    atlas = Image.new("RGB", (atlas_w, atlas_h), "#F0F0F0")
    draw = ImageDraw.Draw(atlas)
    font = get_font(10)
    for i, (eid, crop) in enumerate(crops.items()):
        row, col = divmod(i, cols)
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding + 18)
        crop_resized = crop.copy()
        crop_resized.thumbnail((cell_size, cell_size - 18), Image.Resampling.LANCZOS)
        cx = x + (cell_size - crop_resized.width) // 2
        cy = y + (cell_size - 18 - crop_resized.height) // 2
        atlas.paste(crop_resized, (cx, cy))
        draw.rectangle([x, y, x + cell_size, y + cell_size - 18], outline="#CCCCCC")
        label = eid.replace("vision_", "v")
        draw.text((x + 2, y + cell_size - 16), label, fill="#333333", font=font)
    return atlas


# ── VLM Call ─────────────────────────────────────────────────────────


def call_vlm(
    images: list[Image.Image],
    prompt: str,
) -> tuple[str, int, int, int]:
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_to_b64(img),
            },
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": VLM_MODEL,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": VLM_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    t0 = time.time()
    with httpx.Client(timeout=120) as client:
        resp = client.post(VLM_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
    latency = int((time.time() - t0) * 1000)

    data = resp.json()
    usage = data.get("usage", {})
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text = block["text"]
    return text, latency, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ── Parser ───────────────────────────────────────────────────────────


def parse_vlm_response(text: str) -> list[VLMRefineResult]:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    if not text.startswith("["):
        start = text.find("[")
        if start >= 0:
            text = text[start:]

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        last_complete = text.rfind("},")
        if last_complete > 0:
            fix = text[: last_complete + 1] + "]"
            try:
                items = json.loads(fix)
            except json.JSONDecodeError:
                return []
        else:
            open_count = text.count("{") - text.count("}")
            close_count = text.count("]") - text.count("[")
            fix = text.rstrip().rstrip(",")
            if open_count > 0:
                fix += "}" * open_count
            if close_count < 0:
                fix += "]" * abs(close_count)
            try:
                items = json.loads(fix)
            except json.JSONDecodeError:
                return []

    results = []
    for item in items:
        if not isinstance(item, dict) or "element_id" not in item:
            continue

        conf = item.get("role_confidence", 0.0)
        status = item.get("refine_status", "unreviewed")

        # Enforce confidence threshold
        if conf < CONFIDENCE_THRESHOLD and status == "refined":
            status = "uncertain"

        results.append(VLMRefineResult(
            element_id=item["element_id"],
            visual_type=item.get("visual_type", "unknown"),
            role_label=item.get("role_label", ""),
            semantic_tags=item.get("semantic_tags", []),
            role_confidence=conf,
            role_evidence=item.get("role_evidence", ""),
            refine_status=status,
        ))
    return results


# ── Experiments ──────────────────────────────────────────────────────


def run_experiment_bp(
    annotated: Image.Image,
    metadata: str,
) -> ExperimentResult:
    """B+: Annotated screenshot + cleaned metadata."""
    exp = ExperimentResult(approach="B+")
    prompt = build_vlm_prompt_v2("B+", metadata)
    try:
        text, latency, inp, out = call_vlm([annotated], prompt)
        exp.raw_response = text
        exp.latency_ms = latency
        exp.input_tokens = inp
        exp.output_tokens = out
        exp.results = parse_vlm_response(text)
        exp.parse_ok = len(exp.results) > 0
        exp.element_count = len(exp.results)
    except Exception as e:
        exp.error = str(e)
    return exp


def run_experiment_clite(
    annotated: Image.Image,
    atlas: Image.Image,
    metadata: str,
) -> ExperimentResult:
    """C-lite: B+ + crop atlas for icon/low-confidence elements only."""
    exp = ExperimentResult(approach="C-lite")
    prompt = build_vlm_prompt_v2("C-lite", metadata)
    try:
        text, latency, inp, out = call_vlm([annotated, atlas], prompt)
        exp.raw_response = text
        exp.latency_ms = latency
        exp.input_tokens = inp
        exp.output_tokens = out
        exp.results = parse_vlm_response(text)
        exp.parse_ok = len(exp.results) > 0
        exp.element_count = len(exp.results)
    except Exception as e:
        exp.error = str(e)
    return exp


# ── Refine Write-back & Query ────────────────────────────────────────


def write_refine(results: list[VLMRefineResult]) -> dict:
    payload = {
        "mode": "agent",
        "results": [
            {
                "element_id": r.element_id,
                "visual_type": r.visual_type,
                "semantic_tags": r.semantic_tags,
                "role_label": r.role_label,
                "role_confidence": r.role_confidence,
                "role_source": "vlm_refine_poc_v2",
                "role_evidence": [r.role_evidence] if isinstance(r.role_evidence, str) else r.role_evidence,
                "refine_status": r.refine_status,
            }
            for r in results
        ],
    }
    resp = httpx.post(f"{API_BASE}/canvases/{CANVAS_ID}/refine", json=payload)
    resp.raise_for_status()
    return resp.json()


def query_verify(queries: dict[str, str]) -> dict[str, dict]:
    """Verify with multiple query strategies.

    Args:
        queries: {label: role_label_value} e.g. {"发送按钮": "发送按钮"}

    Returns: {label: {"composite_hit": N, "nl_hit": N}}
    """
    results = {}
    for label, role_val in queries.items():
        # Strategy 1: composite query with role_label
        payload_composite = {
            "canvas_id": CANVAS_ID,
            "target": {"composite": {"role_label": role_val}},
        }
        resp1 = httpx.post(f"{API_BASE}/query", json=payload_composite)
        composite_hit = resp1.json().get("total_matched", 0)

        # Strategy 2: natural_language query
        payload_nl = {
            "canvas_id": CANVAS_ID,
            "target": {"natural_language": role_val},
        }
        resp2 = httpx.post(f"{API_BASE}/query", json=payload_nl)
        nl_hit = resp2.json().get("total_matched", 0)

        results[label] = {
            "composite_hit": composite_hit,
            "nl_hit": nl_hit,
        }
    return results


# ── Analysis ─────────────────────────────────────────────────────────


# Ground truth mapping: element_id → expected Chinese role_label
GROUND_TRUTH = {
    "vision_0": "搜索框",
    "vision_2": "群聊名称",
    "vision_8": "展开按钮",
    "vision_19": "应用图标",
    "vision_24": "未读标记",
    "vision_25": "消息内容",
    "vision_36": "时间标签",
    "vision_38": "消息预览",
    "vision_44": "添加按钮",
    "vision_45": "文件按钮",
    "vision_46": "关闭按钮",  # or emoji button — need visual check
    "vision_47": "清除按钮",
    "vision_48": "设置按钮",
    "vision_51": "聊天标题",
    "vision_52": "新消息提示",
    "vision_55": "发送按钮",
    "vision_62": "语音按钮",
    "vision_63": "最小化按钮",
    "vision_64": "未读消息",
    "vision_65": "搜索按钮",
    "vision_69": "通知按钮",
    "vision_73": "截图标注",
    "vision_77": "收藏按钮",
    "vision_81": "语音通话",
    "vision_83": "侧边栏折叠",
}

# Semantic tag ground truth
TAG_GROUND_TRUTH = {
    "vision_0": ["input.search"],
    "vision_44": ["action.add"],
    "vision_48": ["action.settings"],
    "vision_55": ["action.send"],
    "vision_62": ["media.voice"],
    "vision_64": ["notification.unread"],
}


def analyze_accuracy(exp: ExperimentResult) -> dict:
    """Compute accuracy metrics for an experiment."""
    results_by_id = {r.element_id: r for r in exp.results}

    role_correct = 0
    role_partial = 0
    role_wrong = 0
    role_missing = 0
    tag_correct = 0
    uncertain_count = 0
    uncertain_justified = 0  # uncertain + actually wrong ground truth match

    details = []

    for eid, expected_role in GROUND_TRUTH.items():
        r = results_by_id.get(eid)
        if not r:
            role_missing += 1
            details.append({
                "element_id": eid,
                "expected": expected_role,
                "got": "(missing)",
                "verdict": "missing",
            })
            continue

        if r.refine_status == "uncertain":
            uncertain_count += 1

        got_role = r.role_label
        # Exact match
        if got_role == expected_role:
            role_correct += 1
            verdict = "exact"
        # Partial match: check if key words overlap
        elif any(kw in got_role for kw in expected_role.split("/")) or \
             any(kw in expected_role for kw in got_role.split("/")):
            role_partial += 1
            verdict = "partial"
        else:
            role_wrong += 1
            verdict = "wrong"

        # Check semantic tags
        if eid in TAG_GROUND_TRUTH:
            expected_tags = set(TAG_GROUND_TRUTH[eid])
            got_tags = set(r.semantic_tags)
            if expected_tags & got_tags:
                tag_correct += 1

        details.append({
            "element_id": eid,
            "expected": expected_role,
            "got": got_role,
            "confidence": r.role_confidence,
            "status": r.refine_status,
            "verdict": verdict,
        })

    total = len(GROUND_TRUTH)
    return {
        "total": total,
        "exact": role_correct,
        "partial": role_partial,
        "wrong": role_wrong,
        "missing": role_missing,
        "accuracy": (role_correct + role_partial) / total if total else 0,
        "uncertain": uncertain_count,
        "tag_correct": tag_correct,
        "tag_total": len(TAG_GROUND_TRUTH),
        "details": details,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    global CANVAS_ID

    print("=" * 70)
    print("VLM Refine POC v2: B+ vs C-lite")
    print("=" * 70)

    # Step 0: Observe
    print("\n[0] Observing WeChat window...")
    resp = httpx.post(f"{API_BASE}/observe", json={"hwnd": 23006328}, timeout=180)
    resp.raise_for_status()
    CANVAS_ID = resp.json()["canvas_id"]
    print(f"    Canvas: {CANVAS_ID}")

    # Step 1: Get data
    print("\n[1] Fetching canvas data and screenshot...")
    canvas_data = get_canvas_data()
    screenshot = get_screenshot()
    all_elements = canvas_data.get("elements", [])
    valid_targets = [
        eid for eid in TARGET_IDS
        if any(e["element_id"] == eid for e in all_elements)
    ]
    print(f"    Elements: {len(all_elements)}, Targets: {len(valid_targets)}")

    # Step 2: Generate images
    print("\n[2] Generating images...")
    annotated = generate_annotated_screenshot(screenshot, canvas_data, valid_targets)
    annotated.save(os.path.join(PROJECT_ROOT, "output_annotated_v2.png"))
    print("    Annotated screenshot: output_annotated_v2.png")

    # Collect crops for C-lite (only icon/low-confidence elements)
    crops = {}
    for eid in valid_targets:
        if eid in ICON_OR_LOW_CONF_IDS:
            try:
                crops[eid] = get_crop(eid)
            except Exception:
                pass
    print(f"    Crops for C-lite: {len(crops)} elements")

    atlas = generate_crop_atlas(crops)
    atlas.save(os.path.join(PROJECT_ROOT, "output_atlas_v2.png"))
    print("    Crop atlas: output_atlas_v2.png")

    # Build metadata (v2: split trusted/untrusted)
    metadata = build_metadata_text_v2(canvas_data, valid_targets)
    print(f"\n    Metadata preview (first 5 lines):")
    for line in metadata.split("\n")[:5]:
        print(f"      {line}")

    # Step 3: Run experiments
    print("\n[3] Running experiments...")

    print("\n  --- B+: Annotated + Cleaned Metadata ---")
    exp_bp = run_experiment_bp(annotated, metadata)
    print(f"    Latency: {exp_bp.latency_ms}ms")
    print(f"    Tokens: {exp_bp.input_tokens} in / {exp_bp.output_tokens} out")
    print(f"    Parse OK: {exp_bp.parse_ok}, Elements: {exp_bp.element_count}")
    if exp_bp.error:
        print(f"    ERROR: {exp_bp.error}")

    print("\n  --- C-lite: B+ + Selective Crop Atlas ---")
    exp_cl = run_experiment_clite(annotated, atlas, metadata)
    print(f"    Latency: {exp_cl.latency_ms}ms")
    print(f"    Tokens: {exp_cl.input_tokens} in / {exp_cl.output_tokens} out")
    print(f"    Parse OK: {exp_cl.parse_ok}, Elements: {exp_cl.element_count}")
    if exp_cl.error:
        print(f"    ERROR: {exp_cl.error}")

    # Step 4: Accuracy analysis
    print("\n" + "=" * 70)
    print("ACCURACY ANALYSIS")
    print("=" * 70)

    for exp in [exp_bp, exp_cl]:
        acc = analyze_accuracy(exp)
        print(f"\n  Approach {exp.approach}:")
        print(f"    Role accuracy: {acc['exact']} exact + {acc['partial']} partial / {acc['total']} total = {acc['accuracy']:.1%}")
        print(f"    Wrong: {acc['wrong']}, Missing: {acc['missing']}")
        print(f"    Uncertain: {acc['uncertain']}")
        print(f"    Tag accuracy: {acc['tag_correct']}/{acc['tag_total']}")

        # Show key element details
        print(f"\n    Key elements:")
        key_ids = ["vision_0", "vision_44", "vision_46", "vision_48",
                    "vision_52", "vision_55", "vision_62", "vision_64"]
        for d in acc["details"]:
            if d["element_id"] in key_ids:
                marker = "✅" if d["verdict"] == "exact" else \
                         "⚠️" if d["verdict"] == "partial" else \
                         "❌" if d["verdict"] == "wrong" else "❓"
                conf_str = f" conf={d.get('confidence', 0):.2f}" if "confidence" in d else ""
                status_str = f" [{d.get('status', '')}]" if "status" in d else ""
                print(f"      {marker} {d['element_id']}: expected={d['expected']} → got={d['got']}{conf_str}{status_str}")

    # Step 5: ID alignment check
    print("\n" + "=" * 70)
    print("ID ALIGNMENT")
    print("=" * 70)
    target_set = set(valid_targets)
    for exp in [exp_bp, exp_cl]:
        found_ids = {r.element_id for r in exp.results}
        aligned = found_ids & target_set
        extra = found_ids - target_set
        missing = target_set - found_ids
        print(f"\n  {exp.approach}: {len(aligned)}/{len(target_set)} aligned, "
              f"{len(extra)} extra, {len(missing)} missing")
        if missing:
            print(f"    Missing: {sorted(missing)}")

    # Step 6: Write back best result and query verify
    best = exp_cl if exp_cl.parse_ok and exp_cl.element_count >= exp_bp.element_count else exp_bp
    print(f"\n[6] Writing {best.approach} results back to canvas...")
    refine_resp = write_refine(best.results)
    print(f"    Refined: {refine_resp.get('total_refined', 0)}")

    print("\n[7] Query verification...")
    queries = {
        "搜索框": "搜索框",
        "添加按钮": "添加按钮",
        "设置按钮": "设置按钮",
        "发送按钮": "发送按钮",
        "语音按钮": "语音按钮",
        "未读消息": "未读消息",
        "关闭按钮": "关闭按钮",
        "新消息提示": "新消息提示",
    }
    qresults = query_verify(queries)
    for label, hits in qresults.items():
        c_hit = "✅" if hits["composite_hit"] > 0 else "❌"
        n_hit = "✅" if hits["nl_hit"] > 0 else "❌"
        print(f"    {c_hit} composite/{label}: {hits['composite_hit']}  "
              f"{n_hit} nl/{label}: {hits['nl_hit']}")

    # Step 7: Save results
    print("\n[8] Saving results...")
    for exp in [exp_bp, exp_cl]:
        safe_name = exp.approach.replace("+", "plus").replace("-", "_")
        path = os.path.join(PROJECT_ROOT, f"vlm_response_{safe_name}.json")
        acc = analyze_accuracy(exp)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "approach": exp.approach,
                "latency_ms": exp.latency_ms,
                "input_tokens": exp.input_tokens,
                "output_tokens": exp.output_tokens,
                "parse_ok": exp.parse_ok,
                "element_count": exp.element_count,
                "error": exp.error,
                "accuracy": {
                    "exact": acc["exact"],
                    "partial": acc["partial"],
                    "wrong": acc["wrong"],
                    "missing": acc["missing"],
                    "total": acc["total"],
                    "role_accuracy": acc["accuracy"],
                    "uncertain": acc["uncertain"],
                    "tag_correct": acc["tag_correct"],
                    "tag_total": acc["tag_total"],
                },
                "results": [
                    {
                        "element_id": r.element_id,
                        "visual_type": r.visual_type,
                        "role_label": r.role_label,
                        "semantic_tags": r.semantic_tags,
                        "role_confidence": r.role_confidence,
                        "role_evidence": r.role_evidence,
                        "refine_status": r.refine_status,
                    }
                    for r in exp.results
                ],
                "raw_response": exp.raw_response,
                "accuracy_details": acc["details"],
            }, f, ensure_ascii=False, indent=2)
        print(f"    Saved: vlm_response_{safe_name}.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
