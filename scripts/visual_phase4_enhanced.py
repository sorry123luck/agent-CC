"""
Phase 4 增强版可视化验证脚本

生成清晰可复核的 Phase 4 动作执行可视化产物：

每张动作目标图包含：
1. 全局图：带坐标轴原点、刻度参考线、目标框、中心十字、点击点标记
2. 局部放大图：目标区域 4x 放大，显示元素细节
3. 综合图：多目标用不同颜色区分，编号标注

坐标说明：
- UIA 元素 bounding_rect 目前为屏幕绝对坐标（待修复）
- 可视化直接使用截图坐标（PrintWindow 捕获的窗口截图）
- 展示原始坐标值和截图映射后的坐标值
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 颜色定义
COLOR_TARGETS = [
    "#FF4757",  # 红色
    "#2ED573",  # 绿色
    "#1E90FF",  # 蓝色
    "#FFA502",  # 橙色
    "#A55EEA",  # 紫色
]
COLOR_CLICK_POINT = "#00FF00"   # 绿色 - 实际点击点
COLOR_CROSSHAIR = "#FFFF00"    # 黄色 - 十字准心
COLOR_AXIS = "#888888"         # 灰色 - 坐标轴
COLOR_GRID = "#444444"         # 深灰 - 网格线
COLOR_TEXT_BG = "#000000CC"    # 半透明黑底
COLOR_ZONE_ORIGIN = "#00FFFF"  # 青色 - 坐标系原点
COLOR_INVALID = "#FF0000"      # 红色警告


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


def draw_coordinate_grid(img: Image.Image, spacing: int = 100) -> ImageDraw:
    """在图像上绘制坐标网格和原点标记"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = get_font(10)

    # 绘制垂直参考线
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=COLOR_GRID, width=1)

    # 绘制水平参考线
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=COLOR_GRID, width=1)

    # 绘制坐标轴
    draw.line([(0, 0), (0, h)], fill=COLOR_AXIS, width=2)  # Y轴
    draw.line([(0, 0), (w, 0)], fill=COLOR_AXIS, width=2)  # X轴

    # 原点标记
    draw.ellipse([-5, -5, 5, 5], fill=COLOR_ZONE_ORIGIN)
    draw.text((5, 5), "Origin(0,0)", fill=COLOR_ZONE_ORIGIN, font=font)

    return draw


def clamp_to_image(val: int, max_val: int) -> int:
    """将值限制在图像范围内"""
    return max(0, min(val, max_val - 1))


def draw_enhanced_action_target(
    img: Image.Image,
    target_x: int, target_y: int,
    target_w: int, target_h: int,
    action_name: str,
    target_color: str,
    elem_type: str = "unknown",
    zone_name: str = "unknown",
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    在截图上标注动作目标（增强版）

    返回: (标注后的图像, 截图上的实际框坐标)
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(16)
    font_small = get_font(12)

    # 原始坐标映射到截图坐标（计算边界后再 clamp）
    raw_screen_left = int(target_x * scale_x)
    raw_screen_top = int(target_y * scale_y)
    raw_screen_right = raw_screen_left + int(target_w * scale_x)
    raw_screen_bottom = raw_screen_top + int(target_h * scale_y)

    # clamp 到图像边界
    left = clamp_to_image(raw_screen_left, annotated.width)
    top = clamp_to_image(raw_screen_top, annotated.height)
    right = clamp_to_image(raw_screen_right, annotated.width)
    bottom = clamp_to_image(raw_screen_bottom, annotated.height)

    # 确保 right > left, bottom > top（clamp 后可能反转为 0 宽/高）
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1

    # 检查原始坐标是否有效（非负，在合理范围内）
    raw_valid = target_x >= 0 and target_y >= 0 and target_w > 0 and target_h > 0

    # 检查映射后坐标是否在图像内
    clipped = (raw_screen_left != left or raw_screen_top != top or
               raw_screen_right != right or raw_screen_bottom != bottom)
    is_valid = raw_valid and not clipped

    # 计算元素中心点
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    # 1. 画目标矩形框（带宽度边框）
    if is_valid:
        draw.rectangle(
            [left - 2, top - 2, right + 2, bottom + 2],
            outline=target_color,
            width=4
        )
    else:
        # 坐标无效时，画警告框
        draw.rectangle(
            [left - 2, top - 2, right + 2, bottom + 2],
            outline=COLOR_INVALID,
            width=4
        )
        # 画X线表示无效
        draw.line([left, top, right, bottom], fill=COLOR_INVALID, width=3)
        draw.line([right, top, left, bottom], fill=COLOR_INVALID, width=3)

    # 2. 画中心十字准心（大号醒目）
    cross_size = 25
    draw.line([cx - cross_size, cy, cx + cross_size, cy], fill=COLOR_CROSSHAIR, width=3)
    draw.line([cx, cy - cross_size, cx, cy + cross_size], fill=COLOR_CROSSHAIR, width=3)
    # 中心圆圈
    draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=COLOR_CROSSHAIR, width=2)

    # 3. 画点击点（大号绿色叉）
    click_x, click_y = cx, cy
    click_size = 12
    draw.line([click_x - click_size, click_y - click_size, click_x + click_size, click_y + click_size],
              fill=COLOR_CLICK_POINT, width=4)
    draw.line([click_x + click_size, click_y - click_size, click_x - click_size, click_y + click_size],
              fill=COLOR_CLICK_POINT, width=4)

    # 4. 标签背景和文字
    # 显示原始坐标和截图映射后的坐标
    screen_l = int(target_x * scale_x)
    screen_t = int(target_y * scale_y)
    screen_r = screen_l + int(target_w * scale_x)
    screen_b = screen_t + int(target_h * scale_y)
    label_lines = [
        f"{action_name.upper()}",
        f"#{action_name.split('_')[-1] if '_' in action_name else '0'}",
        f"Type: {elem_type}",
        f"Zone: {zone_name}",
        f"Raw: ({target_x},{target_y})",
        f"Screen: ({left},{top},{right},{bottom})",
    ]
    if clipped:
        label_lines.append(f"(clipped from {screen_r},{screen_b})")
    if not is_valid:
        label_lines.insert(2, "!! INVALID COORDS !!")

    # 计算标签位置（放在框上方，或图像顶部如果框太靠上）
    label_x = left
    label_y = max(0, top - 25 * len(label_lines) - 5)

    # 画标签背景
    max_text_w = max(draw.textbbox((0, 0), line, font=font)[2] for line in label_lines)
    label_h = len(label_lines) * 25 + 5
    draw.rectangle(
        [label_x - 3, label_y - 3, label_x + max_text_w + 6, label_y + label_h],
        fill=COLOR_TEXT_BG
    )

    # 画标签文字
    for i, line in enumerate(label_lines):
        text_color = COLOR_INVALID if "INVALID" in line else target_color
        draw.text((label_x, label_y + i * 25), line, fill=text_color, font=font_small)

    # 5. 标注元素宽高
    elem_w = right - left
    elem_h = bottom - top
    dim_text = f"{elem_w}x{elem_h}"
    dim_bbox = draw.textbbox((right + 5, cy - 10), dim_text, font=font_small)
    draw.rectangle([dim_bbox[0] - 2, dim_bbox[1] - 2, dim_bbox[2] + 2, dim_bbox[3] + 2], fill=COLOR_TEXT_BG)
    draw.text((right + 5, cy - 10), dim_text, fill="#FFFFFF", font=font_small)

    return annotated, (left, top, right, bottom)


def create_zoomed_crop(
    img: Image.Image,
    bbox: tuple[int, int, int, int],
    crop_size: tuple[int, int] = (300, 200),
    padding: int = 50,
) -> Image.Image:
    """
    创建目标区域的局部放大图

    Args:
        img: 原始截图
        bbox: 截图上的目标框坐标 (left, top, right, bottom)
        crop_size: 放大图尺寸 (宽, 高)
        padding: 框周围额外 padding

    Returns:
        放大后的裁剪图
    """
    left, top, right, bottom = bbox
    w, h = img.size

    # 扩展裁剪区域
    crop_left = max(0, left - padding)
    crop_top = max(0, top - padding)
    crop_right = min(w, right + padding)
    crop_bottom = min(h, bottom + padding)

    # 裁剪
    crop = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 调整大小
    crop_resized = crop.resize(crop_size, Image.LANCZOS)

    # 添加边框表示这是放大图
    bordered = Image.new("RGB", (crop_size[0] + 10, crop_size[1] + 10), (50, 50, 50))
    bordered.paste(crop_resized, (5, 5))

    draw = ImageDraw.Draw(bordered)
    # 画边框
    draw.rectangle([0, 0, crop_size[0] + 9, crop_size[1] + 9], outline=COLOR_ZONE_ORIGIN, width=2)

    # 添加"ZOOM"标签
    font = get_font(14)
    draw.text((10, crop_size[1] - 25), "ZOOM", fill=COLOR_ZONE_ORIGIN, font=font)

    return bordered


def create_composite_image(
    full_images: list[Image.Image],
    zoom_images: list[Image.Image],
    targets_info: list[dict],
    window_title: str,
    screenshot_size: tuple[int, int],
) -> Image.Image:
    """
    创建综合图：上方多目标并排，下方放大图并排

    Returns:
        综合大图
    """
    n = len(full_images)

    # 布局：上面板（多目标图横排），下面板（放大图横排）
    thumb_w = 400
    thumb_h = int(thumb_w * 9 / 16)  # 16:9
    zoom_w = 250
    zoom_h = 180

    panel_h = thumb_h + 80  # 标题区域
    composite_w = max(thumb_w * n + 50, 900)
    composite_h = panel_h * 2 + 100

    composite = Image.new("RGB", (composite_w, composite_h), (30, 30, 40))
    draw = ImageDraw.Draw(composite)
    font_title = get_font(20)
    font_body = get_font(14)

    # 标题
    draw.text((20, 15), f"Phase 4 Action Targets: {window_title}", fill=(255, 255, 255), font=font_title)
    draw.text((20, 45), f"Screenshot: {screenshot_size[0]}x{screenshot_size[1]} | Scale: real-size", fill=(200, 200, 200), font=font_body)

    # 上排：多目标全局图
    for i, (img, info) in enumerate(zip(full_images, targets_info)):
        thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        x_offset = 20 + i * (thumb_w + 10)
        y_offset = panel_h - thumb_h
        composite.paste(thumb, (x_offset, y_offset))

        # 编号标签
        color = COLOR_TARGETS[i % len(COLOR_TARGETS)]
        draw.rectangle([x_offset - 3, y_offset - 3, x_offset + 50, y_offset + 25], fill=color)
        draw.text((x_offset, y_offset - 3), f"#{i}", fill=(0, 0, 0), font=font_body)

    # 下排：放大图
    for i, (zoom_img, info) in enumerate(zip(zoom_images, targets_info)):
        zm = zoom_img.resize((zoom_w, zoom_h), Image.LANCZOS)
        x_offset = 20 + i * (zoom_w + 10)
        y_offset = panel_h * 2 + 20
        composite.paste(zm, (x_offset, y_offset))

        # 编号标签
        color = COLOR_TARGETS[i % len(COLOR_TARGETS)]
        draw.rectangle([x_offset - 3, y_offset - 3, x_offset + 50, y_offset + 25], fill=color)
        draw.text((x_offset, y_offset - 3), f"#{i} ZOOM", fill=(0, 0, 0), font=font_body)

    # 图例
    legend_y = composite_h - 40
    draw.text((20, legend_y), "Colors: ", fill=(200, 200, 200), font=font_body)
    for i, info in enumerate(targets_info):
        color = COLOR_TARGETS[i % len(COLOR_TARGETS)]
        x = 100 + i * 80
        draw.rectangle([x, legend_y, x + 15, legend_y + 15], fill=color)
        draw.text((x + 20, legend_y), f"T{i}", fill=(200, 200, 200), font=font_body)

    # 十字和点击点说明
    draw.text((composite_w - 300, legend_y),
              "X=ClickPt  + =Crosshair  |=TargetBox",
              fill=(200, 200, 200), font=font_body)

    return composite


def run_enhanced_verification():
    """运行增强版 Phase 4 可视化验证"""
    print("=" * 60)
    print("Phase 4 增强版可视化验证")
    print("=" * 60)

    # 1. 选择窗口
    from src.windows.window_enum import WindowEnumService
    from src.perception.perception_service import PerceptionService

    enum_svc = WindowEnumService()
    all_windows = enum_svc.enumerate_all(refresh=True)
    print(f"\n当前窗口数: {len(all_windows)}")

    # 优先选择 VS Code
    target_window = None
    for w in all_windows:
        if w.title and "visual studio code" in w.title.lower():
            target_window = w
            break

    # 备用：找任意大窗口
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
        print("[FAIL] 未找到可用窗口")
        return None

    print(f"\n选择窗口: {target_window.title}")
    print(f"  句柄: {target_window.hwnd}")
    print(f"  窗口坐标: {target_window.rect}")

    # 2. 获取感知结果
    print("\n[1] 运行 PerceptionService.analyze()...")
    svc = PerceptionService()
    result = svc.analyze(target_window.hwnd)

    # 3. 截图
    from src.windows.screenshot_service import ScreenshotService
    screenshot_svc = ScreenshotService()
    img = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    screenshot_size = img.size
    print(f"截图尺寸: {screenshot_size}")

    # 窗口信息用于坐标映射
    win_rect = target_window.rect or (0, 0, 0, 0)
    win_x, win_y = win_rect[0], win_rect[1]
    win_w, win_h = win_rect[2] - win_rect[0], win_rect[3] - win_rect[1]

    # 由于 UIA bounding_rect 可能不是内容区坐标，
    # 我们直接使用截图像素坐标（截图就是窗口内容区）
    # 截图坐标系：左上角为原点，向右向下增大
    # scale = 1.0 表示元素坐标直接等于截图像素坐标
    scale_x = 1.0
    scale_y = 1.0

    # 4. 收集动作目标
    print("\n[2] 收集动作目标...")

    all_targets = []

    # 从各个 zone 收集元素（不限 content_area）
    for zone in result.zones:
        zone_name = zone.name
        for elem in zone.elements[:3]:  # 每个 zone 最多取 3 个
            bbox = elem.bounding_rect
            if not bbox:
                continue
            l, t, r, b = bbox
            if l >= r or t >= b:
                continue
            # 检查坐标是否合理（正值，在窗口范围内）
            if t >= 0 and l >= 0 and r <= screenshot_size[0] * 2 and b <= screenshot_size[1] * 2:
                all_targets.append({
                    "bbox": (l, t, r, b),
                    "zone_name": zone_name,
                    "zone_type": zone.zone_type.value,
                    "elem_type": elem.control_type or "unknown",
                    "name": elem.name,
                    "width": r - l,
                    "height": b - t,
                })

    # 限制目标数量
    targets = all_targets[:5]
    print(f"  有效目标数量: {len(targets)}")

    if not targets:
        print("[WARN] 没有找到有效目标，尝试使用 UIA 元素原始坐标...")
        # 备选：直接取 UIA 元素的 bounding_rect，不管坐标是否合理
        content_zone = None
        for zone in result.zones:
            if zone.zone_type.value == "content_area":
                content_zone = zone
                break
        if content_zone:
            for elem in content_zone.elements[:5]:
                bbox = elem.bounding_rect
                if bbox:
                    l, t, r, b = bbox
                    if l < r and t < b:
                        targets.append({
                            "bbox": bbox,
                            "zone_name": content_zone.name,
                            "zone_type": "content_area",
                            "elem_type": elem.control_type or "unknown",
                            "name": elem.name,
                            "width": r - l,
                            "height": b - t,
                        })

    # 5. 生成增强可视化
    full_images = []
    zoom_images = []
    targets_info = []

    print("\n[3] 生成增强可视化...")

    for i, target in enumerate(targets):
        l, t, r, b = target["bbox"]
        elem_type = target["elem_type"]
        zone_name = target["zone_name"]
        target_color = COLOR_TARGETS[i % len(COLOR_TARGETS)]

        # 绘制全局图（带坐标网格）
        draw_coordinate_grid(img)
        full_img, screen_bbox = draw_enhanced_action_target(
            img.copy(),
            l, t, r - l, b - t,
            action_name=f"target_{i}",
            target_color=target_color,
            elem_type=elem_type,
            zone_name=zone_name,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        full_images.append(full_img)

        # 生成放大图
        zoom_img = create_zoomed_crop(img, screen_bbox, crop_size=(320, 220), padding=40)
        zoom_images.append(zoom_img)

        # 坐标信息
        is_valid = (t >= 0 and l >= 0)
        targets_info.append({
            "index": i,
            "raw_bbox": (l, t, r, b),
            "screen_bbox": screen_bbox,
            "zone": zone_name,
            "type": elem_type,
            "valid": is_valid,
        })

        status = "[OK]" if is_valid else "[WARN]"
        print(f"  目标 {i}: zone={zone_name}, type={elem_type}, raw=({l},{t},{r},{b}), screen={screen_bbox}, valid={is_valid}")

        # 保存单独目标图
        path = DEBUG_DIR / f"viz4_enhanced_target_{i}.png"
        full_img.save(path)
        print(f"    保存: {path}")

        # 保存放大图
        zoom_path = DEBUG_DIR / f"viz4_enhanced_zoom_{i}.png"
        zoom_img.save(zoom_path)
        print(f"    保存: {zoom_path}")

    # 6. 生成综合图
    if full_images and zoom_images:
        composite = create_composite_image(
            full_images, zoom_images, targets_info,
            target_window.title, screenshot_size
        )
        composite_path = DEBUG_DIR / "viz4_enhanced_composite.png"
        composite.save(composite_path)
        print(f"\n  综合图: {composite_path}")

    # 7. 报告
    print("\n" + "=" * 60)
    print("Phase 4 增强可视化验证报告")
    print("=" * 60)
    print(f"\n窗口: {target_window.title}")
    print(f"句柄: {target_window.hwnd}")
    print(f"窗口坐标: {win_rect}")
    print(f"截图尺寸: {screenshot_size}")
    print(f"坐标系: 截图像素坐标 (左上角原点)")
    print(f"缩放比例: scale_x={scale_x}, scale_y={scale_y} (1:1映射)")
    print(f"\n目标数量: {len(targets)}")
    for info in targets_info:
        valid_str = "有效" if info["valid"] else "无效坐标"
        print(f"  目标 {info['index']}: {info['zone']}/{info['type']} | raw={info['raw_bbox']} | screen={info['screen_bbox']} | {valid_str}")

    print(f"\n坐标说明:")
    print(f"  - UIA bounding_rect: 可能是屏幕绝对坐标或内容区相对坐标")
    print(f"  - 截图坐标系: PrintWindow捕获，左上角(0,0)，向右向下增大")
    print(f"  - scale=1.0: 假设元素坐标直接等于截图像素坐标")
    print(f"  - 若raw坐标为负值或极大值，说明UIA坐标系有问题")

    print(f"\n导出文件:")
    for i in range(len(targets)):
        print(f"  {i+1}. {DEBUG_DIR / f'viz4_enhanced_target_{i}.png'}")
        print(f"     {DEBUG_DIR / f'viz4_enhanced_zoom_{i}.png'}")
    if full_images and zoom_images:
        print(f"  综合图: {DEBUG_DIR / 'viz4_enhanced_composite.png'}")

    # 写入报告文件
    report_lines = [
        "=" * 60,
        "Phase 4 增强可视化验证报告",
        "=" * 60,
        f"窗口: {target_window.title}",
        f"句柄: {target_window.hwnd}",
        f"窗口坐标: {win_rect}",
        f"截图尺寸: {screenshot_size}",
        f"坐标系: 截图像素坐标 (左上角原点, scale=1.0)",
        "",
        f"目标数量: {len(targets)}",
    ]
    for info in targets_info:
        valid_str = "有效" if info["valid"] else "无效坐标"
        report_lines.append(
            f"  目标 {info['index']}: zone={info['zone']}, type={info['type']}, "
            f"raw={info['raw_bbox']}, screen={info['screen_bbox']}, {valid_str}"
        )

    report_lines.extend([
        "",
        "坐标说明:",
        "- UIA bounding_rect 可能为屏幕绝对坐标或内容区相对坐标（待修复）",
        "- 截图坐标系：PrintWindow捕获，左上角(0,0)，向右向下增大",
        "- scale=1.0: 假设元素坐标直接映射到截图像素",
        "- 若raw坐标为负值或极大值，说明UIA坐标系需要修复",
        "",
        "注意: 当前动作执行为坐标级点击（直接点击指定坐标），",
        "尚未实现元素级点击（通过UIA自动化ID定位元素后点击）。",
        "后续计划: 元素级点击优先，坐标点击兜底。",
    ])

    report_text = "\n".join(report_lines)
    report_path = DEBUG_DIR / "viz4_enhanced_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {report_path}")

    print("\n" + "=" * 60)
    print("[OK] Phase 4 增强可视化验证完成")
    print("=" * 60)

    return {
        "window": target_window,
        "target_count": len(targets),
        "targets_info": targets_info,
        "screenshot_size": screenshot_size,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    result = run_enhanced_verification()
    if result:
        print("\n[OK] Phase 4 增强可视化验证完成")
    else:
        print("\n[FAIL] Phase 4 增强可视化验证失败")
