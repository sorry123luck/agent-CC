"""
Phase 2 可视化验证脚本

对选定窗口生成三类标注图：
1. OCR 标注图
2. UIA 元素标注图
3. 融合结果标注图

导出到 data/debug/
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 确保 data/debug 目录存在
DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 颜色定义
COLOR_OCR = "#FF6B6B"      # 红色 - OCR 文本框
COLOR_UIA = "#4ECDC4"      # 青色 - UIA 元素
COLOR_FUSED_UIA_ONLY = "#95E1D3"    # 浅青 - 仅 UIA
COLOR_FUSED_OCR_ONLY = "#F38181"    # 浅红 - 仅 OCR
COLOR_FUSED_BOTH = "#FFE66D"        # 黄色 - UIA + OCR 融合
COLOR_TEXT_LABEL = "#FFFFFF"


def get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """获取字体，优先使用系统中文字体"""
    # 尝试常见中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/simsun.ttc",   # 宋体
        "C:/Windows/Fonts/arial.ttf",    # Arial
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    # 回退到默认字体
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_ocr_annotation(img: Image.Image, ocr_blocks, scale: float = 1.0) -> Image.Image:
    """在截图上绘制 OCR 文本框标注"""
    draw = ImageDraw.Draw(img.copy())
    font = get_font(14)
    font_small = get_font(12)

    for block in ocr_blocks:
        bbox = block.bbox
        # OCR bbox 已经是截图坐标（如果有缩放需要还原）
        x1, y1, x2, y2 = bbox

        # 绘制红色边框
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_OCR, width=2)

        # 绘制背景色块用于显示文字
        text = block.text[:20] + "..." if len(block.text) > 20 else block.text
        label = f"{text} ({block.confidence:.2f})"

        # 文字背景
        text_bbox = draw.textbbox((x1, y1), label, font=font_small)
        bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
        draw.text((x1, y1), label, fill=COLOR_OCR, font=font_small)

    return img


def draw_uia_annotation(img: Image.Image, uia_elements, scale: float = 1.0,
                        screenshot_size=None, window_content_size=None) -> Image.Image:
    """在截图上绘制 UIA 元素框标注

    UIA bounding_rect 是窗口内容区坐标，需要缩放映射到截图坐标
    """
    draw = ImageDraw.Draw(img.copy())
    font = get_font(12)

    for elem in uia_elements:
        if not elem.bounding_rect:
            continue

        bbox = elem.bounding_rect  # 窗口内容区坐标 (left, top, right, bottom)

        # 计算缩放比例
        if screenshot_size and window_content_size:
            scale_x = screenshot_size[0] / window_content_size[0]
            scale_y = screenshot_size[1] / window_content_size[1]
            # 缩放到截图坐标
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
        else:
            x1, y1, x2, y2 = bbox

        # 过滤无效坐标
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
        if x2 > img.width + 100 or y2 > img.height + 100:
            continue

        # 绘制青色边框
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_UIA, width=2)

        # 标签
        label_parts = []
        if elem.control_type:
            ct = elem.control_type.replace("Control", "")
            label_parts.append(ct)
        if elem.name and len(elem.name) < 15:
            label_parts.append(elem.name)

        if label_parts:
            label = " | ".join(label_parts)
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 160))
            draw.text((x1, y1), label, fill=COLOR_UIA, font=font)

    return img


def draw_fusion_annotation(img: Image.Image, fused_elements,
                            screenshot_size=None, window_content_size=None) -> Image.Image:
    """绘制融合结果标注

    融合元素有三种来源组合：
    - sources=["uia"]: 仅 UIA，绘制青色
    - sources=["ocr"]: 仅 OCR，绘制红色
    - sources=["uia", "ocr"]: 融合，绘制黄色
    """
    draw = ImageDraw.Draw(img.copy())
    font = get_font(12)

    uia_only_count = 0
    ocr_only_count = 0
    fused_count = 0

    for elem in fused_elements:
        bbox = elem.fused_bbox  # 窗口内容区坐标

        # 计算缩放比例
        if screenshot_size and window_content_size:
            scale_x = screenshot_size[0] / window_content_size[0]
            scale_y = screenshot_size[1] / window_content_size[1]
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
        else:
            x1, y1, x2, y2 = bbox

        # 过滤无效坐标
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < -100 or y1 < -100 or x2 > img.width + 100 or y2 > img.height + 100:
            continue

        sources = elem.sources

        if "uia" in sources and "ocr" in sources:
            # 融合元素 - 黄色
            color = COLOR_FUSED_BOTH
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            fused_count += 1
        elif "ocr" in sources and elem.ocr_text:
            # 仅 OCR - 红色
            color = COLOR_FUSED_OCR_ONLY
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            ocr_only_count += 1
        elif "uia" in sources:
            # 仅 UIA - 青色
            color = COLOR_FUSED_UIA_ONLY
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            uia_only_count += 1

        # 标签
        if elem.ocr_text and len(elem.ocr_text) < 20:
            label = elem.ocr_text[:18]
        elif elem.uia_element and elem.uia_element.control_type:
            ct = elem.uia_element.control_type.replace("Control", "")[:10]
            label = ct
        else:
            continue

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        if text_bbox[2] < img.width and text_bbox[3] < img.height:
            bg_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 140))
            draw.text((x1, y1), label, fill=color, font=font)

    # 在图片下方添加图例
    legend_y = img.height - 30
    legend_items = [
        (COLOR_FUSED_BOTH, f"融合 (UIA+OCR): {fused_count}"),
        (COLOR_FUSED_OCR_ONLY, f"仅 OCR: {ocr_only_count}"),
        (COLOR_FUSED_UIA_ONLY, f"仅 UIA: {uia_only_count}"),
    ]

    legend_x = 10
    for color, text in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 20], outline=color, width=2)
        draw.text((legend_x + 25, legend_y), text, fill=COLOR_TEXT_LABEL, font=font)
        legend_x += 200

    return img


def run_visual_verification():
    """运行可视化验证"""
    from src.windows.window_enum import WindowEnumService
    from src.perception.perception_fusion import PerceptionFusion
    from src.perception.uia_client import UIAClient
    from src.perception.ocr_service import OCRService
    from src.windows.screenshot_service import ScreenshotService

    print("=" * 60)
    print("Phase 2 可视化验证")
    print("=" * 60)

    # 1. 选择窗口 - 找一个文本丰富的窗口
    enum_svc = WindowEnumService()
    all_windows = enum_svc.enumerate_all(refresh=True)
    print(f"\n当前窗口数: {len(all_windows)}")

    # 优先选择 VS Code 或其他编辑器窗口
    target_window = None
    for w in all_windows:
        title = w.title or ""
        if any(kw in title.lower() for kw in ["visual studio", "code", "notepad", "pycharm", "idea"]):
            if "explorer" not in title.lower() and "task" not in title.lower():
                target_window = w
                break

    if not target_window:
        # 找任意有内容的窗口
        for w in all_windows:
            w_rect = w.rect or (0, 0, 0, 0)
            w_w = w_rect[2] - w_rect[0]
            w_h = w_rect[3] - w_rect[1]
            if w_w > 400 and w_h > 300:
                if "start" not in w.title.lower() and "task" not in w.title.lower():
                    target_window = w
                    break

    if not target_window:
        print("未找到可用窗口")
        return

    win_rect = target_window.rect or (0, 0, 0, 0)
    win_w = win_rect[2] - win_rect[0]
    win_h = win_rect[3] - win_rect[1]
    print(f"\n选择窗口: {target_window.title}")
    print(f"  句柄: {target_window.hwnd}")
    print(f"  窗口区域: {win_rect[0]},{win_rect[1]} → {win_rect[2]},{win_rect[3]} ({win_w}x{win_h})")
    print(f"  进程: {target_window.process_name} (PID: {target_window.process_id})")

    # 2. 获取截图
    screenshot_svc = ScreenshotService()
    img = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    screenshot_size = img.size
    print(f"\n截图尺寸: {screenshot_size[0]}x{screenshot_size[1]}")

    # 3. 获取 UIA 元素
    uia_client = UIAClient(target_window.hwnd)
    uia_elements = uia_client.find_all()
    print(f"UIA 元素总数: {len(uia_elements)}")

    # 获取窗口内容区尺寸（从 UIA 第一个内容子元素获取，而非根窗口）
    # 根窗口 bounding_rect 包含装饰边框，不是真正的内容区
    window_content_size = None
    if uia_elements:
        # 找第一个 left=0, top=0 的子元素作为内容区
        for elem in uia_elements[1:]:
            if elem.bounding_rect:
                l, t, r, b = elem.bounding_rect
                if l == 0 and t == 0 and elem.width and elem.height:
                    window_content_size = (elem.width, elem.height)
                    break
    if not window_content_size:
        window_content_size = (win_w - 16, win_h - 16)
    print(f"窗口内容区尺寸: {window_content_size[0]}x{window_content_size[1]}")

    # 4. 获取 OCR 结果
    print("\n正在运行 OCR（首次加载模型需要几秒）...")
    ocr_service = OCRService()
    ocr_blocks = ocr_service.extract(img)
    print(f"OCR 文本块数: {len(ocr_blocks)}")

    # 5. 融合分析
    print("\n正在融合分析...")
    fusion = PerceptionFusion()
    page = fusion.analyze(target_window.hwnd)
    fused_elements = page.fused_elements
    print(f"融合元素总数: {len(fused_elements)}")

    uia_only = [e for e in fused_elements if e.sources == ["uia"]]
    ocr_only = [e for e in fused_elements if e.sources == ["ocr"]]
    both = [e for e in fused_elements if "uia" in e.sources and "ocr" in e.sources]
    print(f"  仅 UIA: {len(uia_only)}")
    print(f"  仅 OCR: {len(ocr_only)}")
    print(f"  融合: {len(both)}")

    # 6. 生成标注图
    print("\n生成标注图...")

    # 图 1: OCR 标注
    ocr_img = draw_ocr_annotation(img.copy(), ocr_blocks)
    ocr_path = DEBUG_DIR / "viz_ocr_annotation.png"
    ocr_img.save(ocr_path)
    print(f"  OCR 标注图: {ocr_path}")

    # 图 2: UIA 标注
    uia_img = draw_uia_annotation(
        img.copy(), uia_elements,
        screenshot_size=screenshot_size,
        window_content_size=window_content_size
    )
    uia_path = DEBUG_DIR / "viz_uia_annotation.png"
    uia_img.save(uia_path)
    print(f"  UIA 标注图: {uia_path}")

    # 图 3: 融合结果
    fused_img = draw_fusion_annotation(
        img.copy(), fused_elements,
        screenshot_size=screenshot_size,
        window_content_size=window_content_size
    )
    fused_path = DEBUG_DIR / "viz_fusion_annotation.png"
    fused_img.save(fused_path)
    print(f"  融合结果图: {fused_path}")

    # 7. 同时保存一张缩放后的原图用于对照
    raw_path = DEBUG_DIR / "viz_raw_screenshot.png"
    img.save(raw_path)
    print(f"  原图: {raw_path}")

    # 8. 生成分析报告
    report = []
    report.append("=" * 60)
    report.append("Phase 2 可视化验证报告")
    report.append("=" * 60)
    report.append(f"\n窗口: {target_window.title}")
    report.append(f"句柄: {target_window.hwnd}")
    report.append(f"窗口内容区: {window_content_size[0]}x{window_content_size[1]}")
    report.append(f"截图尺寸: {screenshot_size[0]}x{screenshot_size[1]}")
    report.append(f"\nUIA 元素: {len(uia_elements)} 个")
    report.append(f"OCR 文本块: {len(ocr_blocks)} 个")
    report.append(f"融合元素: {len(fused_elements)} 个")
    report.append(f"  - 仅 UIA: {len(uia_only)}")
    report.append(f"  - 仅 OCR: {len(ocr_only)}")
    report.append(f"  - UIA + OCR 融合: {len(both)}")

    report.append("\n--- OCR 识别样本（前 10 个）---")
    for i, block in enumerate(ocr_blocks[:10]):
        report.append(f"  [{i+1}] \"{block.text}\" @ {block.bbox} (conf={block.confidence:.2f})")

    if fused_elements:
        report.append("\n--- 融合元素样本（前 10 个）---")
        for i, elem in enumerate(fused_elements[:10]):
            sources_str = "+".join(elem.sources)
            if elem.ocr_text:
                report.append(f"  [{i+1}] [{sources_str}] \"{elem.ocr_text}\" @ {elem.fused_bbox}")
            elif elem.uia_element and elem.uia_element.control_type:
                ct = elem.uia_element.control_type.replace("Control", "")
                report.append(f"  [{i+1}] [{sources_str}] <{ct}> @ {elem.fused_bbox}")

    report.append("\n--- 坐标对齐检查 ---")
    # 检查 UIA 元素是否在截图范围内
    out_of_bounds = 0
    if screenshot_size and window_content_size:
        scale_x = screenshot_size[0] / window_content_size[0]
        scale_y = screenshot_size[1] / window_content_size[1]
        for elem in uia_elements:
            if not elem.bounding_rect:
                continue
            x2 = int(elem.bounding_rect[2] * scale_x)
            y2 = int(elem.bounding_rect[3] * scale_y)
            if x2 > screenshot_size[0] or y2 > screenshot_size[1]:
                out_of_bounds += 1

    report.append(f"UIA 元素超出截图边界: {out_of_bounds} 个")

    # OCR 负坐标检查
    ocr_negative = sum(1 for b in ocr_blocks if b.bbox[0] < 0 or b.bbox[1] < 0)
    report.append(f"OCR 文本块有负坐标: {ocr_negative} 个")

    report_text = "\n".join(report)
    print("\n" + report_text)

    # 保存报告
    report_path = DEBUG_DIR / "viz_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {report_path}")

    print("\n" + "=" * 60)
    print("导出文件列表:")
    print(f"  1. {ocr_path}")
    print(f"  2. {uia_path}")
    print(f"  3. {fused_path}")
    print(f"  4. {raw_path}")
    print(f"  5. {report_path}")
    print("=" * 60)

    return {
        "window": target_window,
        "screenshot_size": screenshot_size,
        "window_content_size": window_content_size,
        "uia_count": len(uia_elements),
        "ocr_count": len(ocr_blocks),
        "fused_count": len(fused_elements),
        "uia_only": len(uia_only),
        "ocr_only": len(ocr_only),
        "both": len(both),
        "ocr_blocks": ocr_blocks,
        "fused_elements": fused_elements,
        "out_of_bounds": out_of_bounds,
        "ocr_negative": ocr_negative,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    result = run_visual_verification()

    if result:
        print("\n✅ 可视化验证完成")
    else:
        print("\n❌ 可视化验证失败")
