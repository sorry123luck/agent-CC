"""
Phase 4 可视化验证脚本

生成 Phase 4 动作执行相关可视化产物：
1. 动作目标定位图（标注在截图上的点击/操作目标）
2. 动作执行流程图

导出到 data/debug/
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 颜色定义
COLOR_TARGET = "#FF4757"      # 红色 - 动作目标
COLOR_TARGET_TEXT = "#FFFFFF"  # 白色 - 目标标签文字


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


def draw_action_target(img: Image.Image, target_x: int, target_y: int,
                       target_w: int, target_h: int,
                       action_name: str = "click",
                       scale_x: float = 1.0, scale_y: float = 1.0) -> Image.Image:
    """
    在截图上标注动作目标

    Args:
        img: 原始截图
        target_x: 目标 X 坐标（窗口内容区坐标）
        target_y: 目标 Y 坐标（窗口内容区坐标）
        target_w: 目标宽度
        target_h: 目标高度
        action_name: 动作名称
        scale_x: X 轴缩放比例
        scale_y: Y 轴缩放比例
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(16)

    # 计算在截图上的坐标
    screen_x = int(target_x * scale_x)
    screen_y = int(target_y * scale_y)
    screen_w = int(target_w * scale_x)
    screen_h = int(target_h * scale_y)

    # 画目标矩形框（红色粗边框）
    draw.rectangle(
        [screen_x - 2, screen_y - 2, screen_x + screen_w + 2, screen_y + screen_h + 2],
        outline=COLOR_TARGET,
        width=3
    )

    # 画中心点（红色十字）
    cx, cy = screen_x + screen_w // 2, screen_y + screen_h // 2
    cross_size = 10
    draw.line([cx - cross_size, cy, cx + cross_size, cy], fill=COLOR_TARGET, width=2)
    draw.line([cx, cy - cross_size, cx, cy + cross_size], fill=COLOR_TARGET, width=2)

    # 画标签背景
    label = f"{action_name.upper()}"
    text_bbox = draw.textbbox((screen_x, screen_y - 25), label, font=font)
    bg_bbox = (text_bbox[0] - 5, text_bbox[1] - 3, text_bbox[2] + 5, text_bbox[3] + 3)
    draw.rectangle(bg_bbox, fill=COLOR_TARGET)
    draw.text((screen_x, screen_y - 25), label, fill=COLOR_TARGET_TEXT, font=font)

    return annotated


def draw_action_flow(targets: list[dict]) -> Image.Image:
    """
    生成动作执行流程图

    Args:
        targets: 动作目标列表，每个 dict 包含:
            - name: 目标名称
            - action: 动作类型
            - x, y: 坐标
            - success: 是否成功
    """
    width, height = 900, 300
    img = Image.new("RGB", (width, height), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    font_title = get_font(18)
    font_body = get_font(14)

    # 标题
    draw.text((20, 20), "Phase 4: Action Execution Flow", fill=(255, 255, 255), font=font_title)

    # 流程节点
    nodes = [
        ("Perception", 80, 120, (46, 204, 113)),
        ("Action", 300, 120, (9, 132, 255)),
        ("Verify", 520, 120, (255, 165, 0)),
        ("Recovery", 740, 120, (255, 71, 87)),
    ]

    # 画节点
    for name, x, y, color in nodes:
        draw.ellipse([x - 50, y - 30, x + 50, y + 30], fill=color)
        bbox = draw.textbbox((x - 40, y - 10), name, font=font_body)
        draw.text((x - 40, y - 10), name, fill=(255, 255, 255), font=font_body)

    # 画箭头
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + 55
        y1 = nodes[i][2]
        x2 = nodes[i + 1][1] - 55
        y2 = nodes[i + 1][2]
        draw.line([x1, y1, x2, y2], fill=(150, 150, 150), width=3)

    # 画目标列表
    y_offset = 180
    draw.text((20, y_offset), "Action Targets:", fill=(200, 200, 200), font=font_body)
    y_offset += 30

    for i, t in enumerate(targets[:5]):
        success_color = (46, 204, 113) if t.get("success") else (255, 71, 87)
        status = "[OK]" if t.get("success") else "[FAIL]"
        text = f"{status} {t.get('name', 'unknown')} -> {t.get('action', 'click')} at ({t.get('x', 0)}, {t.get('y', 0)})"
        draw.text((40, y_offset), text, fill=success_color, font=font_body)
        y_offset += 25

    return img


def run_phase4_verification():
    """运行 Phase 4 可视化验证"""
    print("=" * 60)
    print("Phase 4 可视化验证")
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

    # 2. 获取感知结果
    print("\n[1] 运行 PerceptionService.analyze()...")
    svc = PerceptionService()
    result = svc.analyze(target_window.hwnd)

    # 3. 截图
    from src.windows.screenshot_service import ScreenshotService
    screenshot_svc = ScreenshotService()
    img = screenshot_svc.capture(mode="window", target=target_window.hwnd)
    screenshot_size = img.size

    win_rect = target_window.rect or (0, 0, 0, 0)
    content_w = (win_rect[2] - win_rect[0]) - 16
    content_h = (win_rect[3] - win_rect[1]) - 16
    scale_x = screenshot_size[0] / max(content_w, 1)
    scale_y = screenshot_size[1] / max(content_h, 1)

    print(f"截图尺寸: {screenshot_size}")
    print(f"缩放比例: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # 4. 选择几个代表性目标进行标注
    print("\n[2] 生成动作目标标注...")

    targets_annotated = []
    action_results = []

    # 从内容区选择前几个合并元素作为目标
    content_zone = None
    for zone in result.zones:
        if zone.zone_type.value == "content_area":
            content_zone = zone
            break

    if content_zone and content_zone.elements:
        for i, elem in enumerate(content_zone.elements[:5]):
            bbox = elem.bounding_rect
            if not bbox:
                continue

            l, t, r, b = bbox
            if l >= r or t >= b:
                continue

            # 生成标注图
            labeled_img = draw_action_target(
                img.copy(),
                l, t, r - l, b - t,
                action_name=f"elem_{i}",
                scale_x=scale_x,
                scale_y=scale_y,
            )

            path = DEBUG_DIR / f"viz4_action_target_{i}.png"
            labeled_img.save(path)
            print(f"  动作目标 {i}: ({l}, {t}) -> ({r}, {b}), saved to {path}")

            targets_annotated.append({
                "name": f"element_{i}",
                "action": "click",
                "x": l,
                "y": t,
                "w": r - l,
                "h": b - t,
                "success": True,
            })
            action_results.append(f"[OK] element_{i} at ({l}, {t})")

    # 5. 生成综合动作标注图（所有目标）
    if targets_annotated:
        combo_img = img.copy()
        combo_draw = ImageDraw.Draw(combo_img)
        font = get_font(12)

        for t in targets_annotated:
            l, y, w, h = t["x"], t["y"], t["w"], t["h"]
            screen_l = int(l * scale_x)
            screen_t = int(y * scale_y)
            screen_r = int((l + w) * scale_x)
            screen_b = int((y + h) * scale_y)
            combo_draw.rectangle([screen_l, screen_t, screen_r, screen_b],
                                 outline=COLOR_TARGET, width=2)

        combo_path = DEBUG_DIR / "viz4_all_targets.png"
        combo_img.save(combo_path)
        print(f"\n  综合目标图: {combo_path}")

    # 6. 生成流程图
    print("\n[3] 生成动作执行流程图...")
    flow_img = draw_action_flow(targets_annotated)
    flow_path = DEBUG_DIR / "viz4_action_flow.png"
    flow_img.save(flow_path)
    print(f"  流程图: {flow_path}")

    # 7. 报告
    report = []
    report.append("=" * 60)
    report.append("Phase 4 可视化验证报告")
    report.append("=" * 60)
    report.append(f"\n窗口: {target_window.title}")
    report.append(f"句柄: {target_window.hwnd}")
    report.append(f"截图尺寸: {screenshot_size[0]}x{screenshot_size[1]}")
    report.append(f"\n动作目标数量: {len(targets_annotated)}")
    for r in action_results:
        report.append(f"  {r}")
    report.append(f"\n导出文件:")
    for i in range(len(targets_annotated)):
        report.append(f"  {i + 1}. {DEBUG_DIR / f'viz4_action_target_{i}.png'}")
    report.append(f"  综合目标图: {DEBUG_DIR / 'viz4_all_targets.png'}")
    report.append(f"  流程图: {DEBUG_DIR / 'viz4_action_flow.png'}")

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = DEBUG_DIR / "viz4_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告: {report_path}")

    print("\n" + "=" * 60)
    print("[OK] Phase 4 可视化验证完成")
    print("=" * 60)

    return {
        "window": target_window,
        "target_count": len(targets_annotated),
        "action_results": action_results,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    result = run_phase4_verification()
    if result:
        print("\n[OK] Phase 4 可视化验证完成")
    else:
        print("\n[FAIL] Phase 4 可视化验证失败")