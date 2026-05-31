"""
ScreenshotService 单元测试

测试截图服务的缩放逻辑
"""
import pytest
import inspect
from unittest.mock import MagicMock, patch
from PIL import Image

from src.windows.screenshot_service import ScreenshotService
from src.windows import window_enum


class TestScreenshotServiceScale:
    """截图缩放逻辑测试"""

    def setup_method(self):
        self.svc = ScreenshotService()

    def test_scale_image_no_scale_needed(self):
        """图片小于目标尺寸时不需要缩放"""
        img = Image.new("RGB", (640, 480))
        result = self.svc._scale_image(img)
        assert result.size == (640, 480)

    def test_scale_image_landscape(self):
        """横向图片缩放（宽 > 高）"""
        # 2560x1440 -> 长边1280
        img = Image.new("RGB", (2560, 1440))
        result = self.svc._scale_image(img)
        assert result.size[0] == 1280  # 宽缩放到1280
        assert result.size[1] == 720   # 高按比例缩放

    def test_scale_image_portrait(self):
        """纵向图片缩放（高 > 宽）"""
        # 1440x2560 -> 长边1280
        img = Image.new("RGB", (1440, 2560))
        result = self.svc._scale_image(img)
        assert result.size[0] == 720   # 宽按比例缩放
        assert result.size[1] == 1280  # 高缩放到1280

    def test_scale_image_square(self):
        """正方形图片缩放"""
        # 2000x2000 -> 长边1280
        img = Image.new("RGB", (2000, 2000))
        result = self.svc._scale_image(img)
        assert result.size == (1280, 1280)

    def test_scale_image_exact_target(self):
        """图片正好是目标尺寸"""
        # 1280x720 -> 保持不变
        img = Image.new("RGB", (1280, 720))
        result = self.svc._scale_image(img)
        assert result.size == (1280, 720)

    def test_scale_image_slightly_larger(self):
        """图片略大于目标尺寸时会被缩放"""
        # 1300x730 -> 长边1300 > 1280，会被缩放
        # 1300 > 1280，scale = 1280/1300, result = (1280, 718)
        img = Image.new("RGB", (1300, 730))
        result = self.svc._scale_image(img)
        assert result.size[0] == 1280  # 长边被缩放到1280

    def test_scale_long_edge_1280(self):
        """验证长边固定为1280"""
        # 4K 横向
        img = Image.new("RGB", (3840, 2160))
        result = self.svc._scale_image(img)
        assert max(result.size) == 1280
        # 比例 3840:2160 = 16:9
        # 1280:720 = 16:9 ✓
        assert result.size == (1280, 720)


class TestScreenshotServiceCapture:
    """截图捕获参数验证测试"""

    def setup_method(self):
        self.svc = ScreenshotService()

    def test_capture_requires_target_for_window_mode(self):
        """window 模式需要 target 参数"""
        with pytest.raises(ValueError, match="window 模式需要提供 target"):
            self.svc.capture(mode="window")

    def test_capture_requires_rect_for_region_mode(self):
        """region 模式需要 rect 参数"""
        with pytest.raises(ValueError, match="region 模式需要提供 rect"):
            self.svc.capture(mode="region")

    def test_capture_invalid_mode(self):
        """无效的截图模式"""
        with pytest.raises(ValueError, match="不支持的截图模式"):
            self.svc.capture(mode="invalid")

    def test_window_capture_crops_to_dwm_visible_frame(self):
        """窗口截图必须裁掉 Win32 GetWindowRect 中的不可见 resize 边框。"""
        source = inspect.getsource(ScreenshotService._capture_window)
        assert "get_capture_frame_bounds" in source
        assert "crop_offsets_from_window_rect" in source
        assert ".crop(" in source

    def test_window_enumeration_uses_same_visible_frame_bounds(self):
        """窗口列表 rect 必须和截图使用同一套 DWM 可见边界。"""
        enum_source = inspect.getsource(window_enum._enum_callback)
        foreground_source = inspect.getsource(window_enum.WindowEnumService.get_foreground_window)
        assert "get_capture_frame_bounds" in enum_source
        assert "get_capture_frame_bounds" in foreground_source
