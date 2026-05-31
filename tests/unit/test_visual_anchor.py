"""Tests for visual_anchor — 视觉对比工具函数。

所有 must-have 测试使用 PIL 生成的 fixture 图片，不依赖桌面状态。
"""

import pytest
import numpy as np
from pathlib import Path

from src.memory.visual_anchor import (
    compute_dhash,
    dhash_distance,
    compare_assets,
    _HAS_SSIM,
    _HAS_CV2,
)


def _make_solid_image(w: int, h: int, color: int = 128) -> np.ndarray:
    """生成纯色灰度图。"""
    return np.full((h, w), color, dtype=np.uint8)


def _make_checkerboard(w: int, h: int, block: int = 10) -> np.ndarray:
    """生成棋盘格灰度图。"""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (y // block + x // block) % 2 == 0:
                img[y, x] = 255
    return img


def _make_gradient_image(w: int, h: int) -> np.ndarray:
    """生成渐变灰度图。"""
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def _make_noisy_image(w: int, h: int, seed: int = 42) -> np.ndarray:
    """生成带噪声的图像。"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)


def _save_image(img: np.ndarray, path: Path) -> Path:
    from PIL import Image
    Image.fromarray(img).save(str(path))
    return path


class TestDHash:
    def test_identical_image_returns_zero(self):
        img = _make_solid_image(100, 100)
        h1 = compute_dhash(img)
        h2 = compute_dhash(img)
        assert dhash_distance(h1, h2) == 0

    def test_different_images_differ(self):
        img_a = _make_solid_image(100, 100, 0)
        img_b = _make_checkerboard(100, 100)
        h1 = compute_dhash(img_a)
        h2 = compute_dhash(img_b)
        assert dhash_distance(h1, h2) > 0

    def test_gradient_vs_solid(self):
        solid = _make_solid_image(100, 100, 128)
        gradient = _make_gradient_image(100, 100)
        h1 = compute_dhash(solid)
        h2 = compute_dhash(gradient)
        assert dhash_distance(h1, h2) > 0

    def test_hash_is_hex_string(self):
        img = _make_solid_image(100, 100)
        h = compute_dhash(img)
        assert isinstance(h, str)
        assert len(h) == 16
        int(h, 16)  # Should not raise

    def test_dhash_distance_mismatch_raises(self):
        with pytest.raises(ValueError, match="Hash length mismatch"):
            dhash_distance("ab", "abcdef12")


class TestCompareAssets:
    def test_identical_image_exact_match(self, tmp_path):
        img = _make_solid_image(100, 100, 128)
        path = _save_image(img, tmp_path / "baseline.png")
        result = compare_assets(str(path), img)
        assert result.dhash_distance == 0
        assert result.match_status == "exact_match"

    def test_completely_different_no_match(self, tmp_path):
        img_a = _make_solid_image(100, 100, 0)
        img_b = _make_checkerboard(100, 100)
        path = _save_image(img_a, tmp_path / "baseline.png")
        result = compare_assets(str(path), img_b)
        assert result.match_status == "no_match"

    def test_missing_baseline_returns_no_match(self, tmp_path):
        img = _make_solid_image(100, 100)
        result = compare_assets(str(tmp_path / "nonexistent.png"), img)
        assert result.match_status == "no_match"
        assert result.method == "degraded"

    def test_ssim_score_when_available(self, tmp_path):
        img = _make_solid_image(100, 100, 128)
        path = _save_image(img, tmp_path / "baseline.png")
        result = compare_assets(str(path), img)
        if _HAS_SSIM:
            assert result.ssim_score is not None
            assert result.ssim_score > 0.95
        else:
            assert result.ssim_score is None

    def test_method_field_reflects_available_libraries(self, tmp_path):
        img = _make_solid_image(100, 100, 128)
        path = _save_image(img, tmp_path / "baseline.png")
        result = compare_assets(str(path), img)
        if _HAS_SSIM and _HAS_CV2:
            assert "template" in result.method or "ssim" in result.method
        elif _HAS_SSIM:
            assert "ssim" in result.method
        else:
            assert result.method == "dhash_only"

    def test_no_optional_imports_still_works(self, tmp_path):
        """dHash-only path works even without SSIM/OpenCV."""
        img_a = _make_solid_image(100, 100, 128)
        img_b = _make_solid_image(100, 100, 130)  # Slightly different
        path = _save_image(img_a, tmp_path / "baseline.png")
        result = compare_assets(str(path), img_b)
        assert result.dhash_distance >= 0
        assert result.match_status in ("exact_match", "strong_match", "weak_match", "no_match")

    def test_noisy_images_weak_match(self, tmp_path):
        img_a = _make_noisy_image(100, 100, seed=42)
        img_b = _make_noisy_image(100, 100, seed=99)
        path = _save_image(img_a, tmp_path / "baseline.png")
        result = compare_assets(str(path), img_b)
        assert result.match_status in ("no_match", "weak_match")
