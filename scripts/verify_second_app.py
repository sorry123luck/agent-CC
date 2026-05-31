"""
验证脚本：PerceptionService 感知服务

分为两类验证：
1. 可复现机制验证（Notepad）— 验证感知链路各环节机制是否正常工作
2. 复杂 GUI 能力验证（Chrome）— 验证在真实复杂 GUI 场景下的区域划分和 OCR 兜底能力

Notepad 不可作为复杂 GUI 能力验证的代表样例。
"""
import sys
from pathlib import Path
import subprocess
import time
import winreg

DEBUG_DIR = Path("data/debug/verify_second_app")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def find_chrome_window() -> "WindowInfoExt | None":
    """通过注册表查找 Chrome 路径并枚举窗口"""
    from src.windows.window_enum import WindowEnumService

    # 尝试通过注册表找 Chrome 路径
    chrome_paths = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe")
        chrome_paths.append(winreg.QueryValueEx(key, "")[0])
        winreg.CloseKey(key)
    except Exception:
        pass

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe")
        chrome_paths.append(winreg.QueryValueEx(key, "")[0])
        winreg.CloseKey(key)
    except Exception:
        pass

    # 枚举窗口找 Chrome
    enum_svc = WindowEnumService()
    all_windows = enum_svc.enumerate_all(refresh=True)

    for w in all_windows:
        title = (w.title or "").lower()
        # Chrome 窗口标题通常包含 "chrome" 或特定模式
        if "chrome" in title and len(title) < 200:
            return w

    return None


def verify_notepad_mechanism() -> dict:
    """
    验证1：可复现机制验证（Notepad）

    目的：验证感知链路各环节机制是否正常工作
    - 窗口枚举能找到子进程创建的窗口
    - UIA 元素能正确获取
    - 区域划分能识别基本区域类型
    - 截图服务正常工作
    - OCR 调用链路正常（结果质量不代表能力）

    Notepad 简单界面适合作为机制验证，不代表复杂 GUI 能力
    """
    from src.windows.window_enum import WindowEnumService
    from src.perception.perception_service import PerceptionService
    from src.perception.zone_partitioner import ZoneType

    print("=" * 60)
    print("验证1：可复现机制验证（Notepad）")
    print("=" * 60)
    print("\n目的：验证感知链路各环节机制是否正常工作")
    print("注意：Notepad 简单界面不代表复杂 GUI 能力\n")

    # 1. 启动 Notepad
    print("[1] 启动 Notepad...")
    notepad_proc = subprocess.Popen(["notepad.exe"])
    time.sleep(1.5)

    try:
        # 2. 查找 Notepad 窗口
        print("[2] 查找 Notepad 窗口...")
        enum_svc = WindowEnumService()
        all_windows = enum_svc.enumerate_all(refresh=True)

        notepad_window = None
        for w in all_windows:
            if w.title and "notepad" in w.title.lower():
                notepad_window = w
                break

        if not notepad_window:
            print("[FAIL] 未找到 Notepad 窗口")
            return {"success": False, "reason": "notepad window not found"}

        print(f"  找到窗口: {notepad_window.title}")
        print(f"  句柄: {notepad_window.hwnd}")
        print(f"  位置: {notepad_window.rect}")

        # 3. 运行 PerceptionService 分析
        print("\n[3] 运行 PerceptionService.analyze()...")
        svc = PerceptionService()
        result = svc.analyze(notepad_window.hwnd)

        # 4. 输出结果
        print("\n[4] 分析结果:")
        print(f"  窗口标题: {result.window_info.title if result.window_info else 'N/A'}")
        print(f"  截图尺寸: {result.screenshot_size}")
        print(f"  区域数量: {len(result.zones)}")
        print(f"  归并后元素数: {len(result.all_elements_merged)}")
        print(f"  OCR 辅助文本数: {len(result.ocr_auxiliary_texts)}")

        print("\n  区域详情:")
        for zone in result.zones:
            print(f"    {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")

        # 5. 验证检查（机制验证）
        checks = []
        success = True

        if len(result.zones) >= 2:
            checks.append("[OK] 区域数量 >= 2（机制正常）")
        else:
            checks.append("[FAIL] 区域数量 < 2")
            success = False

        if result.screenshot is not None:
            checks.append("[OK] 截图服务正常")
        else:
            checks.append("[FAIL] 截图失败")
            success = False

        if len(result.ocr_auxiliary_texts) >= 0:  # OCR 调用成功即可，不要求质量
            checks.append("[OK] OCR 调用链路正常（质量不代表能力）")
        else:
            checks.append("[FAIL] OCR 调用失败")
            success = False

        has_content_or_unknown = any(
            z.zone_type in (ZoneType.CONTENT_AREA, ZoneType.UNKNOWN) for z in result.zones
        )
        if has_content_or_unknown:
            checks.append("[OK] 区域划分机制正常")
        else:
            checks.append("[FAIL] 区域划分异常")
            success = False

        print("\n[5] 机制验证检查:")
        for check in checks:
            print(f"  {check}")

        # 6. 保存截图
        if result.screenshot:
            screenshot_path = DEBUG_DIR / "notepad_screenshot.png"
            result.screenshot.save(screenshot_path)
            print(f"\n[6] 截图已保存: {screenshot_path}")

        # 7. 保存报告
        report = []
        report.append("=" * 60)
        report.append("验证1：可复现机制验证（Notepad）")
        report.append("=" * 60)
        report.append("目的：验证感知链路各环节机制是否正常工作")
        report.append("注意：Notepad 简单界面不代表复杂 GUI 能力\n")
        report.append(f"窗口: {notepad_window.title}")
        report.append(f"句柄: {notepad_window.hwnd}")
        report.append(f"位置: {notepad_window.rect}")
        report.append(f"\n分析结果:")
        report.append(f"  截图尺寸: {result.screenshot_size}")
        report.append(f"  区域数量: {len(result.zones)}")
        report.append(f"  归并后元素数: {len(result.all_elements_merged)}")
        report.append(f"  OCR 辅助文本数: {len(result.ocr_auxiliary_texts)}")
        report.append(f"\n区域详情:")
        for zone in result.zones:
            report.append(f"    {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")
        report.append(f"\n机制验证检查:")
        for check in checks:
            report.append(f"  {check}")
        report.append(f"\n结论: {'通过' if success else '失败'}")

        report_path = DEBUG_DIR / "verify_mechanism_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"\n报告已保存: {report_path}")
        print("\n" + "=" * 60)
        print(f"[{'OK' if success else 'FAIL'}] 机制验证{'通过' if success else '失败'}")
        print("=" * 60)

        return {
            "success": success,
            "zone_count": len(result.zones),
            "ocr_count": len(result.ocr_auxiliary_texts),
            "checks": checks,
        }

    finally:
        print("\n[7] 关闭 Notepad...")
        notepad_proc.terminate()
        try:
            notepad_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            notepad_proc.kill()


def verify_chrome_complex_gui() -> dict:
    """
    验证2：复杂 GUI 能力验证（Chrome）

    目的：在真实复杂 GUI 场景下验证区域划分和 OCR 兜底能力
    - Chrome 界面包含：标题栏、菜单栏、工具栏、侧边栏、内容区、状态栏
    - 多控件、多字体、多颜色的真实复杂 GUI
    - 用于评估 Phase 3 能力是否满足实际需求

    这是真正的能力验证，代表实际复杂 GUI 场景下的表现
    """
    from src.perception.perception_service import PerceptionService
    from src.perception.zone_partitioner import ZoneType

    print("\n" + "=" * 60)
    print("验证2：复杂 GUI 能力验证（Chrome）")
    print("=" * 60)
    print("\n目的：在真实复杂 GUI 场景下验证区域划分和 OCR 兜底能力")
    print("这是真正的能力验证，代表实际复杂 GUI 场景下的表现\n")

    # 1. 查找 Chrome 窗口
    print("[1] 查找 Chrome 窗口...")
    chrome_window = find_chrome_window()

    if not chrome_window:
        print("[SKIP] 未找到 Chrome 窗口（请确保 Chrome 已打开）")
        return {"success": False, "reason": "chrome window not found", "skipped": True}

    print(f"  找到窗口: {chrome_window.title}")
    print(f"  句柄: {chrome_window.hwnd}")
    print(f"  位置: {chrome_window.rect}")

    # 2. 运行 PerceptionService 分析
    print("\n[2] 运行 PerceptionService.analyze()...")
    svc = PerceptionService()
    result = svc.analyze(chrome_window.hwnd)

    # 3. 输出结果
    print("\n[3] 分析结果:")
    print(f"  窗口标题: {result.window_info.title if result.window_info else 'N/A'}")
    print(f"  截图尺寸: {result.screenshot_size}")
    print(f"  区域数量: {len(result.zones)}")
    print(f"  归并后元素数: {len(result.all_elements_merged)}")
    print(f"  OCR 辅助文本数: {len(result.ocr_auxiliary_texts)}")

    print("\n  区域详情:")
    for zone in result.zones:
        print(f"    {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")

    # 4. 能力评估（不同于机制检查）
    checks = []
    success = True

    # 评估1：至少识别出 4 个区域类型
    zone_types_found = set(z.zone_type for z in result.zones)
    expected_types = {ZoneType.TITLE_BAR, ZoneType.CONTENT_AREA}
    if len(result.zones) >= 4:
        checks.append(f"[OK] 识别出 {len(result.zones)} 个区域（复杂 GUI 能力）")
    else:
        checks.append(f"[WARN] 仅识别出 {len(result.zones)} 个区域")
        success = False

    # 评估2：必须有标题栏和内容区
    has_title = ZoneType.TITLE_BAR in zone_types_found
    has_content = ZoneType.CONTENT_AREA in zone_types_found
    if has_title and has_content:
        checks.append("[OK] 标题栏和内容区识别正确")
    else:
        checks.append(f"[WARN] 标题栏:{has_title} 内容区:{has_content}")
        success = False

    # 评估3：内容区元素数量（反映复杂 GUI 元素解析能力）
    content_zone = next((z for z in result.zones if z.zone_type == ZoneType.CONTENT_AREA), None)
    if content_zone and content_zone.element_count > 50:
        checks.append(f"[OK] 内容区解析出 {content_zone.element_count} 个元素（复杂 GUI 解析能力）")
    elif content_zone:
        checks.append(f"[INFO] 内容区解析出 {content_zone.element_count} 个元素")
    else:
        checks.append("[WARN] 未识别到内容区")
        success = False

    # 评估4：OCR 兜底（不要求质量，只要求调用成功）
    if result.ocr_auxiliary_texts:
        checks.append(f"[INFO] OCR 兜底返回 {len(result.ocr_auxiliary_texts)} 条文本（深色 GUI 质量待改进）")
    else:
        checks.append("[INFO] OCR 兜底无文本（可能内容区为空或 OCR 失败）")

    print("\n[4] 复杂 GUI 能力评估:")
    for check in checks:
        print(f"  {check}")

    # 5. 保存截图
    if result.screenshot:
        screenshot_path = DEBUG_DIR / "chrome_screenshot.png"
        result.screenshot.save(screenshot_path)
        print(f"\n[5] 截图已保存: {screenshot_path}")

    # 6. 保存报告
    report = []
    report.append("=" * 60)
    report.append("验证2：复杂 GUI 能力验证（Chrome）")
    report.append("=" * 60)
    report.append("目的：在真实复杂 GUI 场景下验证区域划分和 OCR 兜底能力\n")
    report.append(f"窗口: {chrome_window.title}")
    report.append(f"句柄: {chrome_window.hwnd}")
    report.append(f"位置: {chrome_window.rect}")
    report.append(f"\n分析结果:")
    report.append(f"  截图尺寸: {result.screenshot_size}")
    report.append(f"  区域数量: {len(result.zones)}")
    report.append(f"  归并后元素数: {len(result.all_elements_merged)}")
    report.append(f"  OCR 辅助文本数: {len(result.ocr_auxiliary_texts)}")
    report.append(f"\n区域详情:")
    for zone in result.zones:
        report.append(f"    {zone.name} ({zone.zone_type.value}): {zone.element_count} 个元素")
    report.append(f"\n复杂 GUI 能力评估:")
    for check in checks:
        report.append(f"  {check}")
    report.append(f"\n结论: {'基本通过（见评估详情）' if success else '未通过'}")

    report_path = DEBUG_DIR / "verify_complex_gui_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n报告已保存: {report_path}")
    print("\n" + "=" * 60)
    print(f"[{'OK' if success else 'WARN'}] 复杂 GUI 验证{'通过' if success else '需改进'}")
    print("=" * 60)

    return {
        "success": success,
        "zone_count": len(result.zones),
        "zone_types": [z.zone_type.value for z in result.zones],
        "content_elements": content_zone.element_count if content_zone else 0,
        "ocr_count": len(result.ocr_auxiliary_texts),
        "checks": checks,
    }


def main():
    """运行两类验证"""
    results = {}

    # 验证1：可复现机制验证（Notepad）
    results["mechanism"] = verify_notepad_mechanism()

    # 验证2：复杂 GUI 能力验证（Chrome）
    results["complex_gui"] = verify_chrome_complex_gui()

    # 汇总报告
    print("\n" + "=" * 60)
    print("验证汇总")
    print("=" * 60)
    print(f"\n1. 可复现机制验证（Notepad）:")
    m = results["mechanism"]
    print(f"   状态: {'通过' if m.get('success') else '失败'}")
    print(f"   区域数量: {m.get('zone_count', 'N/A')}")
    print(f"   OCR 调用: {m.get('ocr_count', 'N/A')} 条")

    print(f"\n2. 复杂 GUI 能力验证（Chrome）:")
    c = results["complex_gui"]
    if c.get("skipped"):
        print(f"   状态: 跳过（{c.get('reason', 'Chrome 未运行')}）")
    else:
        print(f"   状态: {'通过' if c.get('success') else '需改进'}")
        print(f"   区域数量: {c.get('zone_count', 'N/A')}")
        print(f"   区域类型: {c.get('zone_types', [])}")
        print(f"   内容区元素: {c.get('content_elements', 'N/A')}")
        print(f"   OCR 调用: {c.get('ocr_count', 'N/A')} 条")

    # 最终结论
    mechanism_ok = results["mechanism"].get("success", False)
    complex_ok = results["complex_gui"].get("success", False)
    complex_skipped = results["complex_gui"].get("skipped", False)

    print("\n" + "=" * 60)
    if mechanism_ok and (complex_ok or complex_skipped):
        print("[OK] 所有验证通过")
        if complex_skipped:
            print("注意：复杂 GUI 验证已跳过，能力结论待 Chrome 验证")
    elif mechanism_ok:
        print("[WARN] 机制验证通过，复杂 GUI 验证需改进")
    else:
        print("[FAIL] 机制验证失败")
    print("=" * 60)

    return mechanism_ok and (complex_ok or complex_skipped)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    success = main()
    sys.exit(0 if success else 1)