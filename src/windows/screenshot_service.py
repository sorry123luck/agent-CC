"""
截图服务

提供全屏、窗口、区域截图功能
默认返回原始尺寸截图，不再自动缩放
如需预览缩略图，请使用 create_preview() 方法
"""
from typing import Literal
import logging

from PIL import Image
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll, byref, c_int, c_uint

from src.windows.window_bounds import (
    crop_offsets_from_window_rect,
    get_capture_frame_bounds,
    get_window_rect,
)

logger = logging.getLogger(__name__)

# DPI awareness — call once at module load so GetWindowRect returns physical pixels.
# Without this, the process runs in DPI-virtualized mode and GetWindowRect returns
# logical (scaled) coordinates, causing PrintWindow to render into an undersized bitmap.
try:
    windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Minimum screenshot height (physical pixels) — anything shorter is likely just a title bar.
_MIN_VALID_SCREENSHOT_HEIGHT = 100


class ScreenshotService:
    """截图服务"""

    # 预览缩放长边像素值（仅用于 create_preview，不用于主 capture 链路）
    PREVIEW_LONG_EDGE: int = 1280

    def capture(
        self,
        mode: Literal["fullscreen", "window", "region"] = "window",
        target: int | None = None,
        rect: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        """
        截取屏幕截图（原始尺寸，不自动缩放）

        Args:
            mode: 截图模式
                - "fullscreen": 全屏截图
                - "window": 指定窗口截图
                - "region": 指定区域截图
            target: 窗口句柄（mode="window" 时必需）
            rect: 区域坐标元组 (left, top, right, bottom)（mode="region" 时必需）

        Returns:
            PIL.Image 对象（原始尺寸）

        Raises:
            ValueError: 参数无效
            Exception: 截图失败
        """
        if mode == "fullscreen":
            return self._capture_fullscreen()
        elif mode == "window":
            if not target:
                raise ValueError("window 模式需要提供 target (窗口句柄)")
            return self._capture_window(target)
        elif mode == "region":
            if not rect:
                raise ValueError("region 模式需要提供 rect (left, top, right, bottom)")
            return self._capture_region(rect)
        else:
            raise ValueError(f"不支持的截图模式: {mode}")

    def _capture_fullscreen(self) -> Image.Image:
        """截取全屏"""
        # 获取屏幕 DC
        screen_dc = win32gui.GetDC(0)
        mem_dc = win32ui.CreateDCFromHandle(screen_dc)

        # 创建兼容 DC
        save_dc = mem_dc.CreateCompatibleDC()

        # 获取屏幕尺寸
        width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        # 创建位图
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # 复制屏幕内容
        save_dc.BitBlt((0, 0), (width, height), mem_dc, (0, 0), win32con.SRCCOPY)

        # 转换为 PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        # 清理资源
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mem_dc.DeleteDC()
        win32gui.ReleaseDC(0, screen_dc)

        # 返回原始尺寸（不再自动缩放）
        return img

    @staticmethod
    def normalize_hwnd(hwnd: int) -> int:
        """Normalize hwnd to the top-level owner window (GA_ROOT).

        This avoids capturing a child control (e.g. an edit pane inside Notepad)
        instead of the full application window.
        """
        GA_ROOT = 2
        try:
            root = win32gui.GetAncestor(hwnd, GA_ROOT)
            if root:
                return root
        except Exception:
            pass
        return hwnd

    @staticmethod
    def is_minimized(hwnd: int) -> bool:
        """Check if a window is minimized (iconic)."""
        try:
            return win32gui.IsIconic(hwnd)
        except Exception:
            return False

    def _capture_window(self, hwnd: int) -> Image.Image:
        """截取指定窗口（自动归一到顶层窗口）"""
        try:
            # 归一到顶层窗口，避免截到子控件
            hwnd = self.normalize_hwnd(hwnd)

            # 最小化窗口无法截图
            if self.is_minimized(hwnd):
                raise ValueError("window_minimized: 窗口已最小化，请先恢复窗口")

            # PrintWindow renders into a GetWindowRect-sized bitmap. On modern
            # Windows apps that rect often includes invisible resize borders,
            # so crop to the DWM visible frame after capture.
            window_rect = get_window_rect(hwnd)
            frame_rect = get_capture_frame_bounds(hwnd)
            left, top, right, bottom = window_rect
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                raise ValueError(f"无效的窗口尺寸: {width}x{height}")

            # 获取窗口 DC
            hwnd_dc = win32gui.GetDC(hwnd)
            mem_dc = win32ui.CreateDCFromHandle(hwnd_dc)

            # 创建兼容 DC
            save_dc = mem_dc.CreateCompatibleDC()

            # 创建位图
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mem_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # 尝试使用 PrintWindow (PW_RENDERFULLCONTENT = 2)
            success = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

            if not success:
                # 如果 PrintWindow 失败，使用 BitBlt
                save_dc.BitBlt((0, 0), (width, height), mem_dc, (0, 0), win32con.SRCCOPY)

            # 转换为 PIL Image
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)
            img = Image.frombuffer(
                "RGB",
                (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                bmpstr,
                "raw",
                "BGRX",
                0,
                1,
            )

            # 清理资源
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mem_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            # 尺寸校验：截图高度不能只是标题栏
            img_w, img_h = img.size
            if img_h < _MIN_VALID_SCREENSHOT_HEIGHT:
                raise ValueError(
                    f"capture_invalid: 截图高度 {img_h}px < {_MIN_VALID_SCREENSHOT_HEIGHT}px，"
                    f"可能只截到了标题栏 (hwnd={hwnd})"
                )

            if frame_rect != window_rect:
                crop_box = crop_offsets_from_window_rect(window_rect, frame_rect)
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    img = img.crop(crop_box)

            # 返回原始尺寸（不再自动缩放）
            return img

        except Exception as e:
            raise Exception(f"窗口截图失败: {e}")

    def _capture_region(
        self, rect: tuple[int, int, int, int]
    ) -> Image.Image:
        """截取指定区域"""
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            raise ValueError(f"无效的区域尺寸: {width}x{height}")

        # 获取屏幕 DC
        screen_dc = win32gui.GetDC(0)
        mem_dc = win32ui.CreateDCFromHandle(screen_dc)

        # 创建兼容 DC
        save_dc = mem_dc.CreateCompatibleDC()

        # 创建位图
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # 复制区域内容
        save_dc.BitBlt((0, 0), (width, height), mem_dc, (left, top), win32con.SRCCOPY)

        # 转换为 PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        # 清理资源
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mem_dc.DeleteDC()
        win32gui.ReleaseDC(0, screen_dc)

        # 返回原始尺寸（不再自动缩放）
        return img

    def create_preview(self, img: Image.Image, max_long_edge: int | None = None) -> Image.Image:
        """
        创建预览缩略图（仅用于 debug/UI 展示，不影响主链路）

        Args:
            img: 原始 PIL Image
            max_long_edge: 缩放后长边最大值（默认 1280）

        Returns:
            缩放后的预览图
        """
        if max_long_edge is None:
            max_long_edge = self.PREVIEW_LONG_EDGE

        width, height = img.size
        long_edge = max(width, height)

        if long_edge <= max_long_edge:
            return img

        scale = max_long_edge / long_edge
        new_width = int(width * scale)
        new_height = int(height * scale)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _scale_image(self, img: Image.Image) -> Image.Image:
        """
        [已废弃] 请使用 create_preview() 代替

        将图片缩放到固定长边 1280px，短边按比例缩放
        此方法仅保留用于向后兼容，主链路不再调用
        """
        return self.create_preview(img)

    def capture_to_file(
        self,
        filepath: str,
        mode: Literal["fullscreen", "window", "region"] = "window",
        target: int | None = None,
        rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        """
        截取屏幕并保存到文件

        Args:
            filepath: 保存路径
            mode: 截图模式
            target: 窗口句柄
            rect: 区域坐标
        """
        img = self.capture(mode=mode, target=target, rect=rect)
        img.save(filepath)
