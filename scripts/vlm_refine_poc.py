"""VLM Refine POC: Compare three approaches for VLM-based candidate semantic refinement.

Approach A: Raw full screenshot → V2.5
Approach B: Annotated screenshot (bbox + element_id) → V2.5
Approach C: Annotated screenshot + crop atlas → V2.5

Usage: python scripts/vlm_refine_poc.py
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

API_BASE = "http://127.0.0.1:8000/api/v1"
VLM_ENDPOINT = "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages"
VLM_MODEL = "mimo-v2.5"
VLM_API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

# Canvas to test (set after observe)
CANVAS_ID = ""

# Candidates to include in the experiment (25 key elements)
TARGET_IDS = [
    "vision_0",   # 搜索
    "vision_2",   # 发货群
    "vision_8",   # ↓
    "vision_19",  # W
    "vision_24",  # 2
    "vision_25",  # ●
    "vision_36",  # 上
    "vision_38",  # 西
    "vision_44",  # +
    "vision_45",  # Ⅱ
    "vision_46",  # ×
    "vision_47",  # 章
    "vision_48",  # □
    "vision_51",  # 空
    "vision_52",  # 发送
    "vision_55",  # -
    "vision_62",  # Account
    "vision_63",  # Minimize
    "vision_64",  # Search function
    "vision_65",  # Animals
    "vision_69",  # Shuffle
    "vision_73",  # User profile
    "vision_77",  # Drawing
    "vision_81",  # Towel
    "vision_83",  # Hide
]


@dataclass
class VLMRefineResult:
    element_id: str
    visual_type: str = "unknown"
    role_label: str = ""
    semantic_tags: list[str] = field(default_factory=list)
    role_confidence: float = 0.0
    role_evidence: list[str] = field(default_factory=list)
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
    """Fetch canvas data from API."""
    resp = httpx.get(f"{API_BASE}/canvases/{CANVAS_ID}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_screenshot() -> Image.Image:
    """Fetch screenshot as PIL Image."""
    resp = httpx.get(f"{API_BASE}/canvases/{CANVAS_ID}/screenshot", timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def get_crop(element_id: str) -> Image.Image:
    """Fetch candidate crop as PIL Image."""
    resp = httpx.get(f"{API_BASE}/canvases/{CANVAS_ID}/candidates/{element_id}/crop", timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def img_to_b64(img: Image.Image) -> str:
    """Encode PIL Image to base64 PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font for drawing text."""
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Image Generation ─────────────────────────────────────────────────


def generate_annotated_screenshot(
    screenshot: Image.Image,
    canvas_data: dict,
    target_ids: list[str],
) -> Image.Image:
    """Draw bounding boxes with element_id labels on screenshot."""
    img = screenshot.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(12)

    elements = {e["element_id"]: e for e in canvas_data.get("elements", [])}

    # Color cycle for visual distinction
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

        # Draw bbox
        draw.rectangle([l, t, r, b], outline=color, width=2)

        # Draw label background
        label = eid.replace("vision_", "v")
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, t - th - 4)
        draw.rectangle([l, label_y, l + tw + 6, label_y + th + 4], fill=color)
        draw.text((l + 3, label_y + 2), label, fill="white", font=font)

    return img


def generate_crop_atlas(
    crops: dict[str, Image.Image],
    cols: int = 10,
    cell_size: int = 80,
    padding: int = 4,
) -> Image.Image:
    """Create a grid atlas of candidate crops with element_id labels."""
    n = len(crops)
    rows = (n + cols - 1) // cols

    atlas_w = cols * (cell_size + padding) + padding
    atlas_h = rows * (cell_size + padding + 18) + padding  # 18 for label

    atlas = Image.new("RGB", (atlas_w, atlas_h), "#F0F0F0")
    draw = ImageDraw.Draw(atlas)
    font = get_font(10)

    for i, (eid, crop) in enumerate(crops.items()):
        row, col = divmod(i, cols)
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding + 18)

        # Resize crop to fit cell
        crop_resized = crop.copy()
        crop_resized.thumbnail((cell_size, cell_size - 18), Image.Resampling.LANCZOS)

        # Center in cell
        cx = x + (cell_size - crop_resized.width) // 2
        cy = y + (cell_size - 18 - crop_resized.height) // 2
        atlas.paste(crop_resized, (cx, cy))

        # Draw border
        draw.rectangle([x, y, x + cell_size, y + cell_size - 18], outline="#CCCCCC")

        # Draw label
        label = eid.replace("vision_", "v")
        draw.text((x + 2, y + cell_size - 16), label, fill="#333333", font=font)

    return atlas


# ── VLM Calls ────────────────────────────────────────────────────────


def build_metadata_text(canvas_data: dict, target_ids: list[str]) -> str:
    """Build candidate metadata text for VLM context."""
    elements = {e["element_id"]: e for e in canvas_data.get("elements", [])}
    lines = []
    for eid in target_ids:
        el = elements.get(eid)
        if not el:
            continue
        text = el.get("text", "").strip()
        bounds = el.get("bounds", [])
        w = bounds[2] - bounds[0] if len(bounds) >= 4 else 0
        h = bounds[3] - bounds[1] if len(bounds) >= 4 else 0
        sources = el.get("provider_sources", [])
        conf = el.get("confidence", 0)
        lines.append(f"- {eid}: text=\"{text}\" {w}x{h} src={sources} conf={conf}")
    return "\n".join(lines)


def build_vlm_prompt(
    approach: str,
    metadata: str,
) -> str:
    """Build the VLM prompt for each approach."""
    base = """你是一个 Windows 桌面 UI 元素语义分析专家。
请分析截图中的 UI 元素，返回 JSON 数组，每个元素包含：
- element_id: 元素ID
- visual_type: icon / icon_button / button / text / input / indicator / unknown
- role_label: 语义角色标签（英文，如 send_button, add_button, search_bar）
- semantic_tags: 语义标签列表（英文，如 ["send", "submit"]）
- role_confidence: 置信度 0-1
- role_evidence: 判断依据（中文简短描述）
- refine_status: refined（已确认）/ uncertain（不确定）

只返回 JSON 数组，不要其他文字。"""

    if approach == "A":
        return f"""{base}

这是一个微信桌面版的截图。请识别所有可见的 UI 元素（按钮、图标、输入框、文字标签等），返回 JSON 数组。"""

    elif approach == "B":
        return f"""{base}

这是一个微信桌面版的截图，每个 UI 元素的边界框已标注 element_id（如 v46、v52）。
请根据截图中看到的内容，为以下候选元素返回语义分析结果：

{metadata}

返回的 element_id 必须使用完整的 vision_XX 格式。"""

    else:  # C
        return f"""{base}

我提供了两张图：
1. 带编号边界框的微信全截图（标注了每个候选元素的位置和 ID）
2. 候选元素裁剪图集（每个 crop 标注了 element_id）

请结合全截图上下文和裁剪细节，为以下候选元素返回语义分析结果：

{metadata}

返回的 element_id 必须使用完整的 vision_XX 格式。
重点关注：小图标的真实功能、OCR 无法识别的元素、英文标注可能错误的元素。"""


def call_vlm(
    images: list[Image.Image],
    prompt: str,
) -> tuple[str, int, int, int]:
    """Call VLM API with images and prompt.

    Returns: (response_text, latency_ms, input_tokens, output_tokens)
    """
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


def parse_vlm_response(text: str) -> list[VLMRefineResult]:
    """Parse VLM JSON response into VLMRefineResult list."""
    text = text.strip()

    # Handle markdown code blocks
    if "```" in text:
        # Extract content between first ``` and last ```
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

    # Try to fix truncated JSON by closing open brackets
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover truncated JSON
        # Count open vs close brackets
        open_count = text.count("{") - text.count("}")
        close_count = text.count("]") - text.count("[")
        fix = text
        # Remove last incomplete object if needed
        last_complete = fix.rfind("},")
        if last_complete > 0:
            fix = fix[: last_complete + 1] + "]"
        else:
            # Just close brackets
            if open_count > 0:
                fix = fix.rstrip().rstrip(",")
                fix += "}" * max(0, open_count)
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
        results.append(VLMRefineResult(
            element_id=item["element_id"],
            visual_type=item.get("visual_type", "unknown"),
            role_label=item.get("role_label", ""),
            semantic_tags=item.get("semantic_tags", []),
            role_confidence=item.get("role_confidence", 0.0),
            role_evidence=item.get("role_evidence", []),
            refine_status=item.get("refine_status", "unreviewed"),
        ))
    return results


# ── Experiments ───────────────────────────────────────────────────────


def run_experiment_a(screenshot: Image.Image, metadata: str) -> ExperimentResult:
    """A: Raw screenshot → V2.5."""
    exp = ExperimentResult(approach="A")
    prompt = build_vlm_prompt("A", metadata)
    try:
        text, latency, inp, out = call_vlm([screenshot], prompt)
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


def run_experiment_b(annotated: Image.Image, metadata: str) -> ExperimentResult:
    """B: Annotated screenshot → V2.5."""
    exp = ExperimentResult(approach="B")
    prompt = build_vlm_prompt("B", metadata)
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


def run_experiment_c(
    annotated: Image.Image,
    atlas: Image.Image,
    metadata: str,
) -> ExperimentResult:
    """C: Annotated screenshot + crop atlas → V2.5."""
    exp = ExperimentResult(approach="C")
    prompt = build_vlm_prompt("C", metadata)
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


# ── Refine Write-back ────────────────────────────────────────────────


def write_refine(results: list[VLMRefineResult]) -> dict:
    """Write refine results back to canvas via API."""
    payload = {
        "mode": "agent",
        "results": [
            {
                "element_id": r.element_id,
                "visual_type": r.visual_type,
                "semantic_tags": r.semantic_tags,
                "role_label": r.role_label,
                "role_confidence": r.role_confidence,
                "role_source": "vlm_refine_poc",
                "role_evidence": r.role_evidence if isinstance(r.role_evidence, list) else [str(r.role_evidence)],
                "refine_status": r.refine_status,
            }
            for r in results
        ],
    }
    resp = httpx.post(f"{API_BASE}/canvases/{CANVAS_ID}/refine", json=payload)
    resp.raise_for_status()
    return resp.json()


def query_verify(queries: list[str]) -> dict[str, int]:
    """Verify refine results via queries. Returns {query: match_count}."""
    results = {}
    for q in queries:
        # Try composite query with role_label
        payload = {
            "canvas_id": CANVAS_ID,
            "target": {"composite": {"role_label": q}},
        }
        resp = httpx.post(f"{API_BASE}/query", json=payload)
        data = resp.json()
        results[q] = data.get("total_matched", 0)
    return results


# ── Main ─────────────────────────────────────────────────────────────


def main():
    global CANVAS_ID

    print("=" * 60)
    print("VLM Refine POC: Three Approaches Comparison")
    print("=" * 60)

    # Step 0: Observe WeChat
    print("\n[0] Observing WeChat window...")
    resp = httpx.post(f"{API_BASE}/observe", json={"hwnd": 23006328}, timeout=180)
    resp.raise_for_status()
    CANVAS_ID = resp.json()["canvas_id"]
    print(f"    Canvas: {CANVAS_ID}")

    # Step 1: Get data
    print("\n[1] Fetching canvas data and screenshot...")
    canvas_data = get_canvas_data()
    screenshot = get_screenshot()
    print(f"    Screenshot: {screenshot.size}")
    print(f"    Elements: {len(canvas_data.get('elements', []))}")

    # Filter to target candidates
    all_elements = canvas_data.get("elements", [])
    valid_targets = [
        eid for eid in TARGET_IDS
        if any(e["element_id"] == eid for e in all_elements)
    ]
    print(f"    Target candidates: {len(valid_targets)}")

    # Step 2: Generate images
    print("\n[2] Generating annotated screenshot and crop atlas...")
    annotated = generate_annotated_screenshot(screenshot, canvas_data, valid_targets)
    annotated.save(os.path.join(PROJECT_ROOT, "output_annotated.png"))
    print(f"    Annotated screenshot saved: output_annotated.png")

    # Collect crops
    crops = {}
    for eid in valid_targets:
        try:
            crops[eid] = get_crop(eid)
        except Exception:
            pass
    print(f"    Crops collected: {len(crops)}")

    atlas = generate_crop_atlas(crops)
    atlas.save(os.path.join(PROJECT_ROOT, "output_atlas.png"))
    print(f"    Crop atlas saved: output_atlas.png")

    # Build metadata
    metadata = build_metadata_text(canvas_data, valid_targets)

    # Step 3: Run experiments
    print("\n[3] Running experiments...")

    print("\n  --- Approach A: Raw screenshot ---")
    exp_a = run_experiment_a(screenshot, metadata)
    print(f"    Latency: {exp_a.latency_ms}ms")
    print(f"    Tokens: {exp_a.input_tokens} in / {exp_a.output_tokens} out")
    print(f"    Parse OK: {exp_a.parse_ok}")
    print(f"    Elements found: {exp_a.element_count}")
    if exp_a.error:
        print(f"    ERROR: {exp_a.error}")

    print("\n  --- Approach B: Annotated screenshot ---")
    exp_b = run_experiment_b(annotated, metadata)
    print(f"    Latency: {exp_b.latency_ms}ms")
    print(f"    Tokens: {exp_b.input_tokens} in / {exp_b.output_tokens} out")
    print(f"    Parse OK: {exp_b.parse_ok}")
    print(f"    Elements found: {exp_b.element_count}")
    if exp_b.error:
        print(f"    ERROR: {exp_b.error}")

    print("\n  --- Approach C: Annotated + Atlas ---")
    exp_c = run_experiment_c(annotated, atlas, metadata)
    print(f"    Latency: {exp_c.latency_ms}ms")
    print(f"    Tokens: {exp_c.input_tokens} in / {exp_c.output_tokens} out")
    print(f"    Parse OK: {exp_c.parse_ok}")
    print(f"    Elements found: {exp_c.element_count}")
    if exp_c.error:
        print(f"    ERROR: {exp_c.error}")

    # Step 4: Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    target_set = set(valid_targets)

    for exp in [exp_a, exp_b, exp_c]:
        found_ids = {r.element_id for r in exp.results}
        aligned = found_ids & target_set
        extra = found_ids - target_set
        missing = target_set - found_ids

        print(f"\n  Approach {exp.approach}:")
        print(f"    ID alignment: {len(aligned)}/{len(target_set)} matched")
        print(f"    Extra IDs (not in target): {len(extra)}")
        print(f"    Missing IDs: {len(missing)}")

        # Check specific elements
        for label, check_ids in [
            ("search", ["vision_0"]),
            ("send", ["vision_52", "vision_55"]),
            ("add", ["vision_44", "vision_46"]),
            ("settings", ["vision_51"]),
        ]:
            found = [r for r in exp.results if r.element_id in check_ids]
            if found:
                r = found[0]
                print(f"    {label}: {r.element_id} → role={r.role_label}, conf={r.role_confidence:.2f}")

    # Step 5: Use best result (C) for refine write-back
    if exp_c.parse_ok:
        print("\n[5] Writing Approach C results back to canvas...")
        refine_resp = write_refine(exp_c.results)
        print(f"    Refined: {refine_resp.get('total_refined', 0)}")

        # Step 6: Query verification
        print("\n[6] Query verification...")
        queries = [
            "send_button",
            "add_button",
            "close_button",
            "settings_button",
            "search_bar",
            "notification",
            "new_messages",
        ]
        qresults = query_verify(queries)
        for q, count in qresults.items():
            hit = "✅" if count > 0 else "❌"
            print(f"    {hit} {q}: {count} match(es)")

    # Step 7: Save raw responses
    print("\n[7] Saving raw responses...")
    for exp in [exp_a, exp_b, exp_c]:
        path = os.path.join(PROJECT_ROOT, f"vlm_response_{exp.approach}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "approach": exp.approach,
                "latency_ms": exp.latency_ms,
                "input_tokens": exp.input_tokens,
                "output_tokens": exp.output_tokens,
                "parse_ok": exp.parse_ok,
                "element_count": exp.element_count,
                "error": exp.error,
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
            }, f, ensure_ascii=False, indent=2)
        print(f"    Saved: vlm_response_{exp.approach}.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
