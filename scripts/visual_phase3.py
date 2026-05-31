"""
Phase 3 可视化验证脚本

生成三类可视化产物：
1. 归并效果对比图（原始 UIA 元素 vs 归并后元素）
2. 区域划分效果图（标注各功能区）
3. ZonePageStructure 综合输出图

导出到 data/debug/
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 颜色定义
COLOR_ORIGINAL = "#FF6B6B"      # 红色 - 原始元素
COLOR_MERGED = "#4ECDC4"         # 青色 - 归并后元素
COLOR_ZONE_TITLE = "#FF9F43"     # 橙色 - 标题栏
COLOR_ZONE_MENU = "#EE5A24"      # 深橙 - 菜单栏
COLOR_ZONE_TOOL = "#0ABDE3"      # 蓝色 - 工具栏
COLOR_ZONE_SIDE = "#10AC84"       # 绿色 - 侧边栏
COLOR_ZONE_CONTENT = "#5F27CD"   # 紫色 - 内容区
COLOR_ZONE_STATUS = "#F0A500"    # 黄色 - 状态栏
COLOR_ZONE_UNKNOWN = "#576574"    # 灰色 - 未知区域

ZONE_COLORS = {
    "title_bar": COLOR_ZONE_TITLE,
    "menu_bar": COLOR_ZONE_MENU,
    "tool_bar": COLOR_ZONE_TOOL,
    "side_bar": COLOR_ZONE_SIDE,
    "content_area": COLOR_ZONE_CONTENT,
    "status_bar": COLOR_ZONE_STATUS,
    "unknown": COLOR_ZONE_UNKNOWN,
}


def get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """获取字体"""
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/seguisb.ttf"]:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_original_elements(img: Image.Image, uia_elements, scale_x: float, scale_y: float, origin_x: int = 0, origin_y: int = 0) -> Image.Image:
    """在截图上绘制原始 UIA 元素（红色细框）

    Args:
        origin_x, origin_y: 窗口在屏幕上的左上角坐标（截图坐标系的偏移）
    """
    draw = ImageDraw.Draw(img.copy())
    font = get_font(10)
    count = 0

    for elem in uia_elements:
        if not elem.bounding_rect:
            continue
        l, t, r, b = elem.bounding_rect
        # 转换为截图坐标：减去窗口在屏幕上的偏移
        x1, y1, x2, y2 = int((l - origin_x) * scale_x), int((t - origin_y) * scale_y), \
                          int((r - origin_x) * scale_x), int((b - origin_y) * scale_y)

        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
        if x2 > img.width + 50 or y2 > img.height + 50:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=COLOR_ORIGINAL, width=1)
        count += 1

    draw.text((5, 5), f"Original UIA: {count} elements", fill=COLOR_ORIGINAL, font=font)
    return img


def draw_merged_elements(img: Image.Image, merged_elements, scale_x: float, scale_y: float, origin_x: int = 0, origin_y: int = 0) -> Image.Image:
    """在截图上绘制归并后元素（青色框）

    Args:
        origin_x, origin_y: 窗口在屏幕上的左上角坐标（截图坐标系的偏移）
    """
    draw = ImageDraw.Draw(img.copy())
    font = get_font(14)
    merged_count = 0

    for elem in merged_elements:
        bbox = elem.bounding_rect
        # 转换为截图坐标：减去窗口在屏幕上的偏移
        x1, y1, x2, y2 = int((bbox[0] - origin_x) * scale_x), int((bbox[1] - origin_y) * scale_y), \
                          int((bbox[2] - origin_x) * scale_x), int((bbox[3] - origin_y) * scale_y)

        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < -50 or y1 < -50 or x2 > img.width + 50 or y2 > img.height + 50:
            continue

        color = COLOR_MERGED if elem.is_merged else "#888888"
        width = 3 if elem.is_merged else 1
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        merged_count += 1

    draw.text((5, 5), f"Merged: {merged_count} elements", fill=COLOR_MERGED, font=font)
    return img


def draw_zone_overlay(img: Image.Image, zones, scale_x: float, scale_y: float, origin_x: int = 0, origin_y: int = 0) -> Image.Image:
    """在截图上绘制区域划分覆盖层

    Args:
        origin_x, origin_y: 窗口在屏幕上的左上角坐标（截图坐标系的偏移）
    """
    draw = ImageDraw.Draw(img.copy())
    font = get_font(16)

    # 先画区域背景色块（半透明）
    for zone in zones:
        if not zone.bounding_rect:
            continue
        l, t, r, b = zone.bounding_rect
        # 转换为截图坐标：减去窗口在屏幕上的偏移
        x1, y1, x2, y2 = int((l - origin_x) * scale_x), int((t - origin_y) * scale_y), \
                          int((r - origin_x) * scale_x), int((b - origin_y) * scale_y)

        color_hex = ZONE_COLORS.get(zone.zone_type.value, COLOR_ZONE_UNKNOWN)
        # 解析颜色
        r_c = int(color_hex[1:3], 16)
        g_c = int(color_hex[3:5], 16)
        b_c = int(color_hex[5:7], 16)

        # 画半透明背景
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(r_c, g_c, b_c, 40))
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")

    # 再画区域边框和标签
    for zone in zones:
        if not zone.bounding_rect:
            continue
        l, t, r, b = zone.bounding_rect
        # 转换为截图坐标：减去窗口在屏幕上的偏移
        x1, y1, x2, y2 = int((l - origin_x) * scale_x), int((t - origin_y) * scale_y), \
                          int((r - origin_x) * scale_x), int((b - origin_y) * scale_y)

        color_hex = ZONE_COLORS.get(zone.zone_type.value, COLOR_ZONE_UNKNOWN)

        # 边框
        draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=3)

        # 标签
        label = f"{zone.name} ({zone.element_count})"
        text_bbox = draw.textbbox((x1 + 5, y1 + 5), label, font=font)
        bg_bbox = (text_bbox[0] - 3, text_bbox[1] - 3, text_bbox[2] + 3, text_bbox[3] + 3)
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 160))
        draw.text((x1 + 5, y1 + 5), label, fill=color_hex, font=font)

    # 绘制图例
    legend_y = img.height - 30
    legend_x = 10
    for zone_name, color in ZONE_COLORS.items():
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], outline=color, width=2)
        draw.text((legend_x + 20, legend_y), zone_name, fill=color, font=font)
        legend_x += 130

    return img


def run_phase3_verification():
    """运行 Phase 3 可视化验证"""
    from src.windows.window_enum import WindowEnumService
    from src.perception.perception_service import PerceptionService
    from src.perception.element_merger import ElementMerger
    from src.perception.uia_client import UIAClient

    print("=" * 60)
    print("Phase 3 可视化验证")


    # 1. 选择窗口
    enum_svc = WindowEnumService()
    all_windows = enum_svc.enumerate_all(refresh=True)
    print(f"\n当前窗口数: {len(all_windows)}")

    target_window = None
    for w in all_windows:
        title = w.title or ""
        # 优先选择 VS Code（窗口标题包含 Visual Studio Code）
        if "visual studio code" in title.lower():
            target_window = w
            break

    # 如果没找到，用 hwnd 直接指定（VS Code）
    if not target_window:
        vscode_hwnd = 2558580
        for w in all_windows:
            if w.hwnd == vscode_hwnd:
                target_window = w
                break

    if not target_window:
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
    content_w = win_w - 16
    content_h = win_h - 16

    print(f"\n选择窗口: {target_window.title}")
    print(f"  句柄: {target_window.hwnd}")
    print(f"  窗口: {win_rect[0]},{win_rect[1]} → {win_rect[2]},{win_rect[3]} ({win_w}x{win_h})")

    # 2. 获取 UIA 元素
    uia_client = UIAClient(target_window.hwnd)
    uia_elements = uia_client.find_all()
    print(f"\nUIA 元素总数: {len(uia_elements)}")

    # 3. 获取截图
    from src.windows.screenshot_service import ScreenshotService
    screenshot_svc = ScreenshotService()
    img = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    screenshot_size = img.size
    print(f"截图尺寸: {screenshot_size[0]}x{screenshot_size[1]}")

    # 4. 获取窗口内容区尺寸
    content_w, content_h = content_w or 1920, content_h or 1032

    # 5. 计算缩放比例
    scale_x = screenshot_size[0] / content_w
    scale_y = screenshot_size[1] / content_h
    print(f"缩放比例: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # 6. 归并元素
    merger = ElementMerger()
    merged_elements = merger.merge(uia_elements)
    stats = merger.get_merge_stats(uia_elements, merged_elements)
    print(f"\n归并统计:")
    print(f"  原始元素: {stats['original_count']}")
    print(f"  归并后: {stats['merged_count']}")
    print(f"  归并率: {stats['reduction_ratio']:.1%}")
    print(f"  归并组数: {stats['merged_elements']}")

    # 7. 区域划分
    from src.perception.zone_partitioner import ZonePartitioner
    partitioner = ZonePartitioner(merger=merger)
    zone_structure = partitioner.partition(uia_elements, content_w, content_h)
    print(f"\n区域划分结果:")
    for zone in zone_structure.zones:
        print(f"  {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")

    # 8. 生成可视化图片
    print("\n生成可视化图片...")

    # 窗口在屏幕上的左上角坐标（用于坐标转换）
    origin_x = win_rect[0]
    origin_y = win_rect[1]

    # 图1: 原始 UIA 元素
    img_orig = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    orig_img = draw_original_elements(img_orig, uia_elements, scale_x, scale_y, origin_x, origin_y)
    orig_path = DEBUG_DIR / "viz3_original_uia.png"
    orig_img.save(orig_path)
    print(f"  原始 UIA 元素图: {orig_path}")

    # 图2: 归并后元素
    img_merged = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    merged_img = draw_merged_elements(img_merged, merged_elements, scale_x, scale_y, origin_x, origin_y)
    merged_path = DEBUG_DIR / "viz3_merged_elements.png"
    merged_img.save(merged_path)
    print(f"  归并效果图: {merged_path}")

    # 图3: 区域划分
    img_zone = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    zone_img = draw_zone_overlay(img_zone, zone_structure.zones, scale_x, scale_y, origin_x, origin_y)
    zone_path = DEBUG_DIR / "viz3_zone_partition.png"
    zone_img.save(zone_path)
    print(f"  区域划分图: {zone_path}")

    # 图4: 综合图（归并元素 + 区域标注）
    img_combo = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    combo_img = draw_merged_elements(img_combo, merged_elements, scale_x, scale_y, origin_x, origin_y)
    combo_img = draw_zone_overlay(combo_img, zone_structure.zones, scale_x, scale_y, origin_x, origin_y)
    combo_path = DEBUG_DIR / "viz3_combined.png"
    combo_img.save(combo_path)
    print(f"  综合图: {combo_path}")

    # 9. 生成报告
    report = []
    report.append("=" * 60)
    report.append("Phase 3 可视化验证报告")
    report.append("=" * 60)
    report.append(f"\n窗口: {target_window.title}")
    report.append(f"句柄: {target_window.hwnd}")
    report.append(f"内容区尺寸: {content_w}x{content_h}")
    report.append(f"截图尺寸: {screenshot_size[0]}x{screenshot_size[1]}")
    report.append(f"\n归并效果:")
    report.append(f"  原始元素: {stats['original_count']}")
    report.append(f"  归并后: {stats['merged_count']}")
    report.append(f"  归并率: {stats['reduction_ratio']:.1%}")
    report.append(f"  归并组数: {stats['merged_elements']}")
    report.append(f"\n区域划分:")
    for zone in zone_structure.zones:
        report.append(f"  {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")
    report.append(f"\n导出文件:")
    report.append(f"  1. {orig_path}")
    report.append(f"  2. {merged_path}")
    report.append(f"  3. {zone_path}")
    report.append(f"  4. {combo_path}")

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = DEBUG_DIR / "viz3_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {report_path}")

    print("\n" + "=" * 60)
    print("导出文件列表:")
    print(f"  1. {orig_path}")
    print(f"  2. {merged_path}")
    print(f"  3. {zone_path}")
    print(f"  4. {combo_path}")
    print("=" * 60)

    return {
        "window": target_window,
        "stats": stats,
        "zone_count": len(zone_structure.zones),
        "merged_count": len(merged_elements),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    result = run_phase3_verification()
    if result:
        print("\n[OK] Phase 3 可视化验证完成")
    else:
        print("\n[FAIL] Phase 3 可视化验证失败")
