"""
P0 感知调试工具集

服务于 P3 页面结构编译器的最小调试工具：
- inspect_current_window: 检查当前窗口信息
- dump_page_snapshot: 导出页面快照（JSON/YAML）
- overlay_page_snapshot: 可视化覆盖层（regions/elements/anchors/locators）
- save_debug_bundle: 保存完整调试包（snapshot + evidence + overlay）

这些工具仅用于开发/调试，不进入生产路径。
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.execution.execution_record import build_execution_record
from src.perception.page_compiler_models import (
    InteractionCanvas,
    SurfaceType,
    ContentAreaSubtype,
    SurfaceInfo,
    SurfaceEvidence,
    Candidate,
    Locator,
    LocatorKind,
)
from src.perception.page_compiler_candidates import build_boundary_candidates
from src.perception.openclaw_protocol import build_openclaw_payload, normalize_decision_record


# =============================================================================
# 窗口检查
# =============================================================================

def inspect_current_window(hwnd: int) -> dict[str, Any]:
    """
    检查指定窗口的基本信息

    Args:
        hwnd: 窗口句柄

    Returns:
        dict 包含窗口基本信息
    """
    from src.windows.window_enum import WindowEnumService

    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)
    window_info = next((w for w in windows if w.hwnd == hwnd), None)

    if not window_info:
        return {"error": f"Window with hwnd={hwnd} not found"}

    return {
        "hwnd": window_info.hwnd,
        "title": window_info.title,
        "class_name": window_info.class_name,
        "rect": window_info.rect,
        "process_id": window_info.process_id,
        "process_name": window_info.process_name,
        "is_foreground": window_info.is_foreground,
        "is_minimized": window_info.is_minimized,
        "dpi_scale": window_info.dpi_scale,
    }


# =============================================================================
# Snapshot 导出
# =============================================================================

def dump_page_snapshot(
    snapshot: InteractionCanvas,
    format: str = "json",
    include_evidence: bool = True,
) -> str:
    """
    导出 InteractionCanvas 为可读格式

    Args:
        snapshot: InteractionCanvas 对象
        format: 输出格式 ("json" | "yaml")
        include_evidence: 是否包含 evidence 详情

    Returns:
        str 格式化的快照内容
    """
    def _serialize(obj: Any) -> Any:
        # dataclass → dict recursively (check before enum since dataclasses are not enums)
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for name in obj.__dataclass_fields__:
                result[name] = _serialize(getattr(obj, name))
            return result
        # Enum → value
        if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (str, int, float, bool, type(None))):
            return obj.value
        # datetime → ISO string
        if isinstance(obj, datetime):
            return obj.isoformat()
        # list / dict → recursively serialize contents
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        # primitive types pass through
        return obj

    data = _serialize(snapshot)

    if not include_evidence:
        # 精简 evidence
        if "surface" in data and "evidence" in data["surface"]:
            data["surface"]["evidence"] = [
                {"layer": e["layer"], "feature": e["feature"], "description": e["description"]}
                for e in data["surface"]["evidence"]
            ]

    if format == "yaml":
        try:
            import yaml
            return yaml.dump(data, allow_unicode=True, sort_keys=False)
        except ImportError:
            format = "json"

    if format == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)

    return str(data)


# =============================================================================
# 快速分类接口
# =============================================================================

def classify_window(
    hwnd: int,
    uia_element_count: int = 0,
    uia_control_types: list[str] | None = None,
    has_dom_bridge: bool = False,
    dom_ready: bool = False,
    visual_score: float | None = None,
) -> dict[str, Any]:
    """
    对窗口进行 surface_type 分类（调试用快速接口）
    """
    from src.windows.window_enum import WindowEnumService
    from src.perception.surface_classifier import SurfaceClassifier

    enum_svc = WindowEnumService()
    windows = enum_svc.enumerate_all(refresh=True)
    window_info = next((w for w in windows if w.hwnd == hwnd), None)
    process_name = window_info.process_name if window_info else None

    classifier = SurfaceClassifier()
    result = classifier.classify(
        process_name=process_name,
        uia_element_count=uia_element_count,
        uia_control_types=uia_control_types,
        has_dom_bridge=has_dom_bridge,
        dom_ready=dom_ready,
        visual_score=visual_score,
    )

    return {
        "surface_type": result.surface_type.value,
        "confidence": result.confidence,
        "locator_priority": result.locator_priority,
        "evidence": [
            {
                "layer": e.layer,
                "feature": e.feature,
                "weight": e.weight,
                "value": e.value,
                "description": e.description,
            }
            for e in result.evidence
        ],
    }


def classify_content_area(
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    对 content_area 元素进行 subtype 分类（调试用快速接口）
    """
    from src.perception.content_area_classifier import ContentAreaClassifier

    classifier = ContentAreaClassifier()
    result = classifier.classify(elements)

    return {
        "subtype": result.subtype.value,
        "confidence": result.confidence,
        "evidence": [
            {
                "feature": e.feature,
                "weight": e.weight,
                "value": e.value,
                "description": e.description,
            }
            for e in result.evidence
        ],
    }


# =============================================================================
# 可视化覆盖层
# =============================================================================

def overlay_page_snapshot(
    snapshot: InteractionCanvas,
    screenshot: Image.Image | None = None,
    output_path: str | Path | None = None,
    scale: float = 1.0,
    *,
    layers: set[str] | None = None,
    return_image: bool = False,
) -> Path | Image.Image:
    """
    在截图上绘制 InteractionCanvas 可视化覆盖层（原图 1:1 绘制）

    Args:
        snapshot: InteractionCanvas 对象
        screenshot: 原始截图（PIL Image），可选
        output_path: 输出路径，默认保存到 data/debug/
        scale: [已废弃] 缩放比例，当前主链路统一使用原图 1:1，此参数不再生效
        layers: 要绘制的层集合。None = 全部绘制。
                可选值: regions, elements, anchors, locators, ocr, vision, scroll, info
        return_image: True 时返回 PIL.Image，不写磁盘

    Returns:
        Path（写磁盘模式）或 Image.Image（return_image 模式）
    """
    # 如果没有截图，创建空白图像
    if screenshot is None:
        window = snapshot.window
        if window and window.rect_client:
            w, h = window.rect_client[2] - window.rect_client[0], window.rect_client[3] - window.rect_client[1]
        elif snapshot.regions:
            max_right = max((r.bounds[2] for r in snapshot.regions if r.bounds), default=1920)
            max_bottom = max((r.bounds[3] for r in snapshot.regions if r.bounds), default=1080)
            w, h = max_right, max_bottom
        else:
            w, h = 1920, 1080
        screenshot = Image.new("RGB", (max(w, 100), max(h, 100)), color=(40, 40, 40))

    # 复制图像以避免修改原图
    img = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # 层绘制映射
    all_layers = layers is None
    _draw_map: list[tuple[str, Any]] = [
        ("regions", lambda: _draw_regions(draw, snapshot.regions)),
        ("elements", lambda: _draw_elements(draw, snapshot.elements)),
        ("anchors", lambda: _draw_anchors(draw, snapshot.anchors, snapshot.elements)),
        ("locators", lambda: _draw_locators(draw, snapshot.locators, snapshot.elements)),
        ("ocr", lambda: _draw_ocr_blocks(draw, snapshot.artifacts.get("ocr_blocks", []))),
        ("vision", lambda: _draw_vision_candidates(draw, snapshot.artifacts.get("vision_candidates", []))),
        ("scroll", lambda: _draw_scroll_contexts(draw, snapshot.regions, snapshot.scroll_contexts)),
        ("info", lambda: _draw_info_label(draw, snapshot)),
    ]

    for layer_name, draw_fn in _draw_map:
        if all_layers or (layers and layer_name in layers):
            draw_fn()

    # 内存返回模式：不写磁盘
    if return_image:
        return img

    # 写磁盘模式（向后兼容）
    if output_path is None:
        debug_dir = Path("data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = debug_dir / f"overlay_{snapshot.canvas_id}_{ts}.png"

    output_path = Path(output_path)
    img.save(output_path)
    return output_path


def overlay_boundary_candidates(
    snapshot: InteractionCanvas,
    screenshot: Image.Image | None = None,
    output_path: str | Path | None = None,
    draw_labels: bool = False,
) -> Path:
    """
    Draw boundary candidates on a dedicated overlay so candidate retention quality
    can be checked without region/element noise.
    """
    if output_path is None:
        debug_dir = Path("data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = debug_dir / f"candidate_overlay_{snapshot.canvas_id}_{ts}.png"

    output_path = Path(output_path)

    if screenshot is None:
        window = snapshot.window
        if window and window.rect_client:
            w, h = window.rect_client[2] - window.rect_client[0], window.rect_client[3] - window.rect_client[1]
        elif snapshot.regions:
            max_right = max((r.bounds[2] for r in snapshot.regions if r.bounds), default=1920)
            max_bottom = max((r.bounds[3] for r in snapshot.regions if r.bounds), default=1080)
            w, h = max_right, max_bottom
        else:
            w, h = 1920, 1080
        screenshot = Image.new("RGB", (max(w, 100), max(h, 100)), color=(24, 24, 24))

    img = screenshot.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    _draw_boundary_candidates(
        draw,
        build_boundary_candidates(snapshot),
        draw_labels=draw_labels,
    )
    img.save(output_path)
    return output_path


def _draw_regions(draw: ImageDraw.Draw, regions: list) -> None:
    """绘制 regions 边框和标签"""
    role_colors = {
        "title_bar": (255, 200, 0),    # 黄色
        "menu_bar": (255, 165, 0),      # 橙色
        "tool_bar": (255, 140, 0),      # 深橙
        "side_bar": (100, 149, 237),    # 矢车菊蓝
        "content_area": (60, 179, 113), # 中绿
        "status_bar": (147, 112, 219),  # 紫色
        "unknown": (128, 128, 128),     # 灰色
    }

    for region in regions:
        if not region.bounds:
            continue
        l, t, r, b = region.bounds
        if l >= r or t >= b:
            continue
        color = role_colors.get(region.role, (128, 128, 128))

        # 绘制边框
        for i in range(2):  # 画两像素边框，更明显
            draw.rectangle([l - i, t - i, r + i, b + i], outline=color, width=2)

        # 添加标签背景
        label = f"{region.role}"
        try:
            draw.text((l + 4, t + 4), label, fill=color)
        except Exception:
            pass


def _draw_elements(draw: ImageDraw.Draw, elements: list[Candidate]) -> None:
    """绘制 elements 色块标注"""
    for elem in elements:
        if not elem.bounds:
            continue
        l, t, r, b = elem.bounds
        if l >= r or t >= b:
            continue

        # 可交互元素用绿色边框，不可交互用灰色
        if elem.interactable:
            color = (0, 200, 0)
        else:
            color = (180, 180, 180)

        # 画小边框标注
        draw.rectangle([l, t, r, b], outline=color, width=1)


def _draw_anchors(draw: ImageDraw.Draw, anchors: list, elements: list[Candidate]) -> None:
    """绘制 anchors 红色 X 标记"""
    elem_by_id = {e.element_id: e for e in elements}

    for anchor in anchors:
        if not anchor.element_refs:
            continue
        for elem_id in anchor.element_refs:
            elem = elem_by_id.get(elem_id)
            if not elem or not elem.bounds:
                continue
            l, t, r, b = elem.bounds
            cx, cy = (l + r) // 2, (t + b) // 2
            size = 8

            # 画红色 X
            draw.line([(cx - size, cy - size), (cx + size, cy + size)], fill=(255, 0, 0), width=2)
            draw.line([(cx + size, cy - size), (cx - size, cy + size)], fill=(255, 0, 0), width=2)


def _draw_locators(draw: ImageDraw.Draw, locators: list[Locator], elements: list[Candidate]) -> None:
    """绘制 locator 命中点（UIA locator 用蓝色点）"""
    elem_by_id = {e.element_id: e for e in elements}

    for locator in locators:
        if locator.kind != LocatorKind.UIA:
            continue
        elem = elem_by_id.get(locator.element_ref)
        if not elem or not elem.bounds:
            continue
        l, t, r, b = elem.bounds
        cx, cy = (l + r) // 2, (t + b) // 2

        # 蓝色圆点
        radius = 5
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=(0, 112, 255),
            outline=(0, 60, 200),
            width=1,
        )


def _draw_ocr_blocks(draw: ImageDraw.Draw, ocr_blocks: list[dict[str, Any]]) -> None:
    """Draw OCR text blocks as yellow dashed-like rectangles."""
    for block in ocr_blocks:
        bbox = block.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        l, t, r, b = [int(value) for value in bbox]
        draw.rectangle([l, t, r, b], outline=(255, 215, 0), width=1)
        text = str(block.get("text") or "")[:18]
        if text:
            draw.text((l + 2, max(0, t - 12)), text, fill=(255, 215, 0))


def _draw_vision_candidates(draw: ImageDraw.Draw, vision_candidates: list[dict[str, Any]]) -> None:
    """Draw vision candidates and their semantic labels."""
    for candidate in vision_candidates:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        l, t, r, b = [int(value) for value in bbox]
        draw.rectangle([l, t, r, b], outline=(255, 0, 255), width=1)
        label = str(candidate.get("kind") or candidate.get("region_role") or "vision")[:24]
        draw.text((l + 2, b + 2), label, fill=(255, 0, 255))


def _draw_boundary_candidates(
    draw: ImageDraw.Draw,
    candidates: list[dict[str, Any]],
    draw_labels: bool = False,
) -> None:
    """Draw weak-semantic boundary candidates with minimal clutter."""
    source_colors = {
        "ocr": (255, 215, 0),
        "omniparser": (255, 0, 255),
        "vision": (255, 0, 255),
        "uia": (0, 160, 255),
        "dom": (0, 160, 255),
        "merged": (60, 220, 140),
        "unknown": (180, 180, 180),
    }
    for index, candidate in enumerate(candidates):
        bbox = candidate.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        l, t, r, b = [int(value) for value in bbox]
        source = str(candidate.get("source") or "unknown").lower()
        color = source_colors.get(source, (180, 180, 180))
        draw.rectangle([l, t, r, b], outline=color, width=2)
        if draw_labels:
            candidate_id = str(candidate.get("candidate_id") or f"cand_{index}")
            label = f"C{index}:{candidate_id[:18]}"
            draw.text((l + 2, max(0, t - 12)), label, fill=color)


def _draw_scroll_contexts(draw: ImageDraw.Draw, regions: list, scroll_contexts: list) -> None:
    """Draw viewport-like regions that own scroll contexts."""
    region_by_id = {region.region_id: region for region in regions}
    for context in scroll_contexts:
        region = region_by_id.get(context.region_id)
        if region is None or not region.bounds:
            continue
        l, t, r, b = region.bounds
        draw.rectangle([l + 3, t + 3, r - 3, b - 3], outline=(0, 255, 255), width=1)
        draw.text((l + 4, t + 18), f"scroll:{context.scroll_context_id}", fill=(0, 255, 255))


def _draw_info_label(draw: ImageDraw.Draw, snapshot: InteractionCanvas) -> None:
    """在图像角落添加信息标签"""
    surface = snapshot.surface
    page = snapshot.page

    lines = [
        f"surface: {surface.surface_type.value} ({surface.confidence:.2f})",
        f"page_class: {page.page_class}",
        f"regions: {len(snapshot.regions)}",
        f"elements: {len(snapshot.elements)}",
        f"locators: {len(snapshot.locators)}",
        f"anchors: {len(snapshot.anchors)}",
        f"ocr_blocks: {len(snapshot.artifacts.get('ocr_blocks', []))}",
        f"vision_candidates: {len(snapshot.artifacts.get('vision_candidates', []))}",
        f"scroll_contexts: {len(snapshot.scroll_contexts)}",
    ]

    # 在左上角画黑色背景
    label_h = len(lines) * 14 + 8
    draw.rectangle([4, 4, 280, 6 + label_h], fill=(0, 0, 0, 180))

    y = 8
    for line in lines:
        try:
            draw.text((8, y), line, fill=(255, 255, 255))
            y += 14
        except Exception:
            pass


# =============================================================================
# 完整调试包
# =============================================================================

def save_debug_bundle(
    snapshot: InteractionCanvas,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    screenshot: Image.Image | None = None,
) -> dict[str, Any]:
    """
    保存完整调试包到 data/debug/

    生成文件：
    - {bundle_name}_snapshot.json — InteractionCanvas 完整 JSON
    - {bundle_name}_evidence.json — surface 证据摘要
    - {bundle_name}_regions.json — regions 结构摘要
    - {bundle_name}_elements.json — elements 结构摘要
    - {bundle_name}_locators.json — locators 摘要
    - {bundle_name}_overlay.png — 可视化覆盖层图像

    Args:
        snapshot: InteractionCanvas 对象
        output_dir: 输出目录，默认 data/debug/
        bundle_name: 包名称（不含扩展名），默认用 canvas_id
        screenshot: 原始截图（PIL Image），用于生成 overlay

    Returns:
        dict 包含保存的文件路径和摘要
    """
    if output_dir is None:
        output_dir = Path("data/debug")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_name = bundle_name or snapshot.canvas_id or f"debug_{uuid.uuid4().hex[:8]}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_prefix = f"{bundle_name}_{ts}"

    files: dict[str, str] = {}

    # 1. InteractionCanvas JSON
    snapshot_path = output_dir / f"{bundle_prefix}_snapshot.json"
    snapshot_path.write_text(dump_page_snapshot(snapshot, format="json"), encoding="utf-8")
    files["snapshot"] = str(snapshot_path)

    if screenshot is not None:
        raw_path = output_dir / f"{bundle_prefix}_raw.png"
        screenshot.save(raw_path)
        files["raw"] = str(raw_path)

    # 2. surface 证据摘要
    surface = snapshot.surface
    evidence_summary = {
        "surface_type": surface.surface_type.value,
        "confidence": surface.confidence,
        "evidence_count": len(surface.evidence),
        "evidence": [
            {"layer": e.layer, "feature": e.feature, "weight": e.weight,
             "value": e.value, "description": e.description}
            for e in surface.evidence
        ],
    }
    evidence_path = output_dir / f"{bundle_prefix}_evidence.json"
    evidence_path.write_text(json.dumps(evidence_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["evidence"] = str(evidence_path)

    # 3. regions 结构摘要
    regions_summary = {
        "region_count": len(snapshot.regions),
        "regions": [
            {
                "region_id": r.region_id,
                "role": r.role,
                "subtype": r.subtype.value if r.subtype else None,
                "bounds": r.bounds,
                "element_count": len(r.element_ids),
                "child_region_count": len(r.child_region_ids),
            }
            for r in snapshot.regions
        ],
    }
    regions_path = output_dir / f"{bundle_prefix}_regions.json"
    regions_path.write_text(json.dumps(regions_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["regions"] = str(regions_path)

    # 4. elements 结构摘要
    elements_summary = {
        "element_count": len(snapshot.elements),
        "elements": [
            {
                "element_id": e.element_id,
                "region_id": e.region_id,
                "semantic_role": e.semantic_role.value if e.semantic_role else None,
                "control_type": e.control_type,
                "text": e.text[:50] if e.text else None,
                "bounds": e.bounds,
                "interactable": e.interactable,
                "locator_count": len(e.locator_ids),
                "anchor_count": len(e.anchor_ids),
            }
            for e in snapshot.elements[:100]
        ],
    }
    elements_path = output_dir / f"{bundle_prefix}_elements.json"
    elements_path.write_text(json.dumps(elements_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["elements"] = str(elements_path)

    # 5. locators 摘要
    locators_summary = {
        "locator_count": len(snapshot.locators),
        "locators": [
            {
                "locator_id": l.locator_id,
                "element_ref": l.element_ref,
                "kind": l.kind.value,
                "priority": l.priority,
                "status": l.status.value,
                "confidence": l.confidence,
                "durability_score": l.durability_score,
            }
            for l in snapshot.locators[:50]
        ],
    }
    locators_path = output_dir / f"{bundle_prefix}_locators.json"
    locators_path.write_text(json.dumps(locators_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["locators"] = str(locators_path)

    # 6. overlay 图像
    overlay_path = output_dir / f"{bundle_prefix}_overlay.png"
    try:
        overlay_output = overlay_page_snapshot(snapshot, screenshot=screenshot, output_path=overlay_path)
        files["overlay"] = str(overlay_output)
    except Exception as e:
        files["overlay_error"] = str(e)

    candidate_overlay_path = output_dir / f"{bundle_prefix}_candidate_overlay.png"
    try:
        candidate_overlay_output = overlay_boundary_candidates(
            snapshot,
            screenshot=screenshot,
            output_path=candidate_overlay_path,
            draw_labels=False,
        )
        files["candidate_overlay"] = str(candidate_overlay_output)
    except Exception as e:
        files["candidate_overlay_error"] = str(e)

    # 7. provider_trace
    provider_summary = {
        "uia_used": snapshot.provider_trace.uia_used,
        "ocr_used": snapshot.provider_trace.ocr_used,
        "vision_used": snapshot.provider_trace.vision_used,
        "dom_used": snapshot.provider_trace.dom_used,
        "details": snapshot.provider_trace.provider_details,
    }
    provider_path = output_dir / f"{bundle_prefix}_provider.json"
    provider_path.write_text(json.dumps(provider_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["provider"] = str(provider_path)

    scroll_contexts_summary = {
        "scroll_context_count": len(snapshot.scroll_contexts),
        "scroll_contexts": [
            {
                "scroll_context_id": ctx.scroll_context_id,
                "region_id": ctx.region_id,
                "scroll_type": ctx.scroll_type,
                "viewport_height": ctx.viewport_height,
                "viewport_width": ctx.viewport_width,
                "total_content_height": ctx.total_content_height,
                "is_virtual": ctx.is_virtual,
            }
            for ctx in snapshot.scroll_contexts
        ],
    }
    scroll_contexts_path = output_dir / f"{bundle_prefix}_scroll_contexts.json"
    scroll_contexts_path.write_text(
        json.dumps(scroll_contexts_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files["scroll_contexts"] = str(scroll_contexts_path)

    anchors_summary = {
        "anchor_count": len(snapshot.anchors),
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "kind": anchor.kind.value,
                "region_id": anchor.region_id,
                "element_refs": anchor.element_refs,
                "stability_score": anchor.stability_score,
            }
            for anchor in snapshot.anchors
        ],
    }
    anchors_path = output_dir / f"{bundle_prefix}_anchors.json"
    anchors_path.write_text(json.dumps(anchors_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["anchors"] = str(anchors_path)

    vision_summary = {
        "vision_candidate_count": len(snapshot.artifacts.get("vision_candidates", [])),
        "vision_layout_region_count": len(snapshot.artifacts.get("vision_layout_regions", [])),
        "vision_control_group_count": len(snapshot.artifacts.get("vision_control_groups", [])),
        "structure_evidence_score": snapshot.artifacts.get("structure_evidence_score"),
        "vision_candidates": snapshot.artifacts.get("vision_candidates", []),
        "vision_layout_regions": snapshot.artifacts.get("vision_layout_regions", []),
        "vision_control_groups": snapshot.artifacts.get("vision_control_groups", []),
        "vision_interaction_hints": snapshot.artifacts.get("vision_interaction_hints", []),
    }
    vision_path = output_dir / f"{bundle_prefix}_vision.json"
    vision_path.write_text(json.dumps(vision_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["vision"] = str(vision_path)

    ocr_summary = {
        "ocr_block_count": len(snapshot.artifacts.get("ocr_blocks", [])),
        "ocr_blocks": snapshot.artifacts.get("ocr_blocks", []),
        "ocr_provider": snapshot.artifacts.get("ocr_provider", {}),
    }
    ocr_path = output_dir / f"{bundle_prefix}_ocr.json"
    ocr_path.write_text(json.dumps(ocr_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    files["ocr"] = str(ocr_path)

    candidates_summary = {
        "candidate_count": len(build_boundary_candidates(snapshot)),
        "candidate_stats": snapshot.artifacts.get("boundary_candidate_stats", {}),
        "candidates": build_boundary_candidates(snapshot),
    }
    candidates_path = output_dir / f"{bundle_prefix}_candidates.json"
    candidates_path.write_text(
        json.dumps(candidates_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files["candidates"] = str(candidates_path)

    return {
        "bundle_name": bundle_prefix,
        "output_dir": str(output_dir),
        "files": files,
        "summary": {
            "surface_type": surface.surface_type.value,
            "confidence": surface.confidence,
            "page_class": snapshot.page.page_class,
            "region_count": len(snapshot.regions),
            "element_count": len(snapshot.elements),
            "locator_count": len(snapshot.locators),
            "anchor_count": len(snapshot.anchors),
            "relation_count": len(snapshot.relations),
            "scroll_context_count": len(snapshot.scroll_contexts),
            "ocr_block_count": len(snapshot.artifacts.get("ocr_blocks", [])),
            "vision_candidate_count": len(snapshot.artifacts.get("vision_candidates", [])),
            "boundary_candidate_count": len(build_boundary_candidates(snapshot)),
        },
    }


def build_action_target_debug(
    snapshot: InteractionCanvas | dict[str, Any],
    element_id: str,
) -> dict[str, Any]:
    """Build a compact evidence trace for one action target."""
    if isinstance(snapshot, dict):
        return _build_action_target_debug_from_dict(snapshot, element_id)

    element = snapshot.get_element(element_id)
    if element is None:
        return {"element_id": element_id, "error": "element_not_found"}

    region = snapshot.get_region(element.region_id) if element.region_id else None
    locators = [snapshot.get_locator(locator_id) for locator_id in element.locator_ids]
    anchors = [anchor for anchor in snapshot.anchors if anchor.anchor_id in element.anchor_ids]
    scroll_context = None
    if region and region.scroll_context_id:
        scroll_context = next(
            (ctx for ctx in snapshot.scroll_contexts if ctx.scroll_context_id == region.scroll_context_id),
            None,
        )

    return {
        "element_id": element.element_id,
        "semantic_role": element.semantic_role.value,
        "region": {
            "region_id": region.region_id if region else None,
            "role": region.role if region else None,
            "bounds": region.bounds if region else None,
            "parent_region_id": region.parent_region_id if region else None,
            "child_region_ids": region.child_region_ids if region else [],
        },
        "content_group_id": element.content_group_id,
        "bounds": element.bounds,
        "interaction_hints": dict(element.attributes.get("interaction_hints") or {}),
        "structure_evidence_score": element.attributes.get(
            "structure_evidence_score",
            snapshot.artifacts.get("structure_evidence_score"),
        ),
        "provider_sources": list(element.provider_sources),
        "locators": [
            {
                "locator_id": locator.locator_id,
                "kind": locator.kind.value,
                "priority": locator.priority,
                "confidence": locator.confidence,
                "durability_score": locator.durability_score,
                "cost_score": locator.cost_score,
                "anchor_refs": locator.anchor_refs,
                "verification_hints": locator.verification_hints,
            }
            for locator in locators
            if locator is not None
        ],
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "kind": anchor.kind.value,
                "stability_score": anchor.stability_score,
                "signature": anchor.signature,
            }
            for anchor in anchors
        ],
        "scroll_context": (
            {
                "scroll_context_id": scroll_context.scroll_context_id,
                "scroll_type": scroll_context.scroll_type,
                "viewport_height": scroll_context.viewport_height,
                "viewport_width": scroll_context.viewport_width,
                "total_content_height": scroll_context.total_content_height,
                "is_virtual": scroll_context.is_virtual,
            }
            if scroll_context
            else None
        ),
        "vision_candidate": element.attributes.get("vision_candidate"),
        "ocr_text": element.attributes.get("ocr_text"),
    }


def build_snapshot_evidence_trace(snapshot: InteractionCanvas | dict[str, Any]) -> dict[str, Any]:
    """Summarize evidence flow for before/after comparison and failure analysis."""
    if isinstance(snapshot, dict):
        regions = snapshot.get("regions", [])
        artifacts = snapshot.get("artifacts", {})
        provider_trace = snapshot.get("provider_trace", {}).get("provider_details", {})
        return {
            "canvas_id": snapshot.get("canvas_id"),
            "surface_type": snapshot.get("surface", {}).get("surface_type"),
            "page_class": snapshot.get("page", {}).get("page_class"),
            "region_roles": [region.get("role") for region in regions],
            "child_region_count": sum(len(region.get("child_region_ids", [])) for region in regions),
            "scroll_context_count": len(snapshot.get("scroll_contexts", [])),
            "anchor_count": len(snapshot.get("anchors", [])),
            "locator_count": len(snapshot.get("locators", [])),
            "ocr_block_count": len(artifacts.get("ocr_blocks", [])),
            "vision_candidate_count": len(artifacts.get("vision_candidates", [])),
            "vision_layout_region_count": len(artifacts.get("vision_layout_regions", [])),
            "vision_control_group_count": len(artifacts.get("vision_control_groups", [])),
            "structure_evidence_score": artifacts.get("structure_evidence_score"),
            "provider_trace": provider_trace,
        }

    return {
        "canvas_id": snapshot.canvas_id,
        "surface_type": snapshot.surface.surface_type.value,
        "page_class": snapshot.page.page_class,
        "region_roles": [region.role for region in snapshot.regions],
        "child_region_count": sum(len(region.child_region_ids) for region in snapshot.regions),
        "scroll_context_count": len(snapshot.scroll_contexts),
        "anchor_count": len(snapshot.anchors),
        "locator_count": len(snapshot.locators),
        "ocr_block_count": len(snapshot.artifacts.get("ocr_blocks", [])),
        "vision_candidate_count": len(snapshot.artifacts.get("vision_candidates", [])),
        "vision_layout_region_count": len(snapshot.artifacts.get("vision_layout_regions", [])),
        "vision_control_group_count": len(snapshot.artifacts.get("vision_control_groups", [])),
        "structure_evidence_score": snapshot.artifacts.get("structure_evidence_score"),
        "provider_trace": snapshot.provider_trace.provider_details,
    }


def build_action_outcome_trace(outcome: Any) -> dict[str, Any]:
    """Summarize an ActionOutcome-like payload for inspector/debug consumption."""
    return build_execution_record(outcome)


def _build_action_target_debug_from_dict(
    snapshot: dict[str, Any],
    element_id: str,
) -> dict[str, Any]:
    elements = snapshot.get("elements", [])
    regions = snapshot.get("regions", [])
    locators = snapshot.get("locators", [])
    anchors = snapshot.get("anchors", [])
    scroll_contexts = snapshot.get("scroll_contexts", [])
    artifacts = snapshot.get("artifacts", {})

    element = next((item for item in elements if item.get("element_id") == element_id), None)
    if element is None:
        return {"element_id": element_id, "error": "element_not_found"}

    region_id = element.get("region_id")
    region = next((item for item in regions if item.get("region_id") == region_id), None)
    locator_items = [
        item for item in locators if item.get("locator_id") in set(element.get("locator_ids", []))
    ]
    anchor_items = [
        item for item in anchors if item.get("anchor_id") in set(element.get("anchor_ids", []))
    ]
    scroll_context = None
    scroll_context_id = region.get("scroll_context_id") if region else None
    if scroll_context_id:
        scroll_context = next(
            (item for item in scroll_contexts if item.get("scroll_context_id") == scroll_context_id),
            None,
        )

    attributes = element.get("attributes", {})
    return {
        "element_id": element.get("element_id"),
        "semantic_role": element.get("semantic_role"),
        "region": {
            "region_id": region.get("region_id") if region else None,
            "role": region.get("role") if region else None,
            "bounds": region.get("bounds") if region else None,
            "parent_region_id": region.get("parent_region_id") if region else None,
            "child_region_ids": region.get("child_region_ids", []) if region else [],
        },
        "content_group_id": element.get("content_group_id"),
        "bounds": element.get("bounds"),
        "interaction_hints": dict(attributes.get("interaction_hints") or {}),
        "structure_evidence_score": attributes.get(
            "structure_evidence_score",
            artifacts.get("structure_evidence_score"),
        ),
        "provider_sources": list(element.get("provider_sources", [])),
        "locators": [
            {
                "locator_id": locator.get("locator_id"),
                "kind": locator.get("kind"),
                "priority": locator.get("priority"),
                "confidence": locator.get("confidence"),
                "durability_score": locator.get("durability_score"),
                "cost_score": locator.get("cost_score"),
                "anchor_refs": locator.get("anchor_refs", []),
                "verification_hints": locator.get("verification_hints", {}),
            }
            for locator in locator_items
        ],
        "anchors": [
            {
                "anchor_id": anchor.get("anchor_id"),
                "kind": anchor.get("kind"),
                "stability_score": anchor.get("stability_score"),
                "signature": anchor.get("signature", {}),
            }
            for anchor in anchor_items
        ],
        "scroll_context": scroll_context,
        "vision_candidate": attributes.get("vision_candidate"),
        "ocr_text": attributes.get("ocr_text"),
    }
