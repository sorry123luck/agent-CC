"""
Visual Anchor — 跨截图视觉对比工具函数。

三级对比流水线：dHash 快筛 → SSIM 结构验证 → OpenCV 模板匹配。
SSIM 和 OpenCV 均为 optional dependency，缺失时返回 degraded 结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Optional: scikit-image (SSIM)
try:
    from skimage.metrics import structural_similarity as ssim_func
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False

# Optional: OpenCV (template matching)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass(frozen=True)
class VisualObservationResult:
    dhash_distance: int
    ssim_score: float | None
    template_score: float | None
    coordinate_drift: float | None
    match_status: str  # exact_match|strong_match|weak_match|no_match|drift_detected
    method: str  # dhash_only|dhash_ssim|dhash_ssim_template|degraded


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """计算 dHash（64-bit 差值哈希），返回 16 字符 hex 字符串。"""
    gray = _to_grayscale(image)
    resized = _resize_nearest(gray, hash_size + 1, hash_size)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size * hash_size // 4}x")


def dhash_distance(hash_a: str, hash_b: str) -> int:
    """计算两个 dHash 的汉明距离。"""
    if len(hash_a) != len(hash_b):
        raise ValueError(f"Hash length mismatch: {len(hash_a)} vs {len(hash_b)}")
    val_a = int(hash_a, 16)
    val_b = int(hash_b, 16)
    xor = val_a ^ val_b
    return bin(xor).count("1")


def compare_assets(
    baseline_path: str,
    current_image: np.ndarray,
    expected_bounds: tuple[int, int, int, int] | None = None,
    asset_root: str | None = None,
) -> VisualObservationResult:
    """三级对比流水线。

    Args:
        baseline_path: 基准帧文件路径
        current_image: 当前截图（numpy array, BGR 或 grayscale）
        expected_bounds: 可选的期望位置 (left, top, right, bottom)
        asset_root: 可选的资产根目录，传入时校验 baseline_path 不逃逸

    Returns:
        VisualObservationResult
    """
    baseline = _load_image(baseline_path, asset_root=asset_root)
    if baseline is None:
        return VisualObservationResult(
            dhash_distance=-1, ssim_score=None, template_score=None,
            coordinate_drift=None, match_status="no_match", method="degraded",
        )

    # Step 1: dHash 快筛
    hash_a = compute_dhash(baseline)
    hash_b = compute_dhash(current_image)
    dist = dhash_distance(hash_a, hash_b)

    if dist > 12:
        return VisualObservationResult(
            dhash_distance=dist, ssim_score=None, template_score=None,
            coordinate_drift=None, match_status="no_match", method="dhash_only",
        )

    # Step 2: SSIM 结构验证（optional）
    ssim_score: float | None = None
    if _HAS_SSIM:
        gray_a = _to_grayscale(baseline)
        gray_b = _to_grayscale(current_image)
        h = min(gray_a.shape[0], gray_b.shape[0])
        w = min(gray_a.shape[1], gray_b.shape[1])
        if h > 0 and w > 0:
            gray_a = gray_a[:h, :w]
            gray_b = gray_b[:h, :w]
            ssim_score = float(ssim_func(gray_a, gray_b))
            if ssim_score < 0.70:
                return VisualObservationResult(
                    dhash_distance=dist, ssim_score=ssim_score,
                    template_score=None, coordinate_drift=None,
                    match_status="no_match", method="dhash_ssim",
                )

    # Step 3: OpenCV 模板匹配（optional）
    template_score: float | None = None
    drift: float | None = None
    if _HAS_CV2 and expected_bounds is not None:
        template_score, drift = _template_match_in_roi(
            current_image, baseline, expected_bounds
        )

    # 判定 match_status
    if template_score is not None:
        if template_score < 0.80:
            status = "weak_match"
        elif drift is not None and drift > 5.0:
            status = "drift_detected"
        else:
            status = "strong_match"
        method = "dhash_ssim_template" if ssim_score is not None else "dhash_template"
    elif ssim_score is not None:
        status = "strong_match" if ssim_score >= 0.85 else "weak_match"
        method = "dhash_ssim"
    else:
        status = "strong_match" if dist <= 4 else "weak_match"
        method = "dhash_only"

    if dist == 0 and (ssim_score is None or ssim_score >= 0.95):
        status = "exact_match"

    return VisualObservationResult(
        dhash_distance=dist, ssim_score=ssim_score,
        template_score=template_score, coordinate_drift=drift,
        match_status=status, method=method,
    )


def _template_match_in_roi(
    full_image: np.ndarray,
    template: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    expansion: int = 50,
) -> tuple[float | None, float | None]:
    """在 ROI 附近做模板匹配，返回 (score, drift_pixels)。"""
    if not _HAS_CV2:
        return None, None

    gray_full = _to_grayscale(full_image)
    gray_tmpl = _to_grayscale(template)
    h, w = gray_full.shape[:2]
    l, t, r, b = roi_bounds
    el = max(0, l - expansion)
    et = max(0, t - expansion)
    er = min(w, r + expansion)
    eb = min(h, b + expansion)
    roi = gray_full[et:eb, el:er]

    if roi.shape[0] < gray_tmpl.shape[0] or roi.shape[1] < gray_tmpl.shape[1]:
        return None, None

    result = cv2.matchTemplate(roi, gray_tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    center_x = max_loc[0] + gray_tmpl.shape[1] / 2 + el
    center_y = max_loc[1] + gray_tmpl.shape[0] / 2 + et
    expected_cx = (l + r) / 2
    expected_cy = (t + b) / 2
    drift = ((center_x - expected_cx) ** 2 + (center_y - expected_cy) ** 2) ** 0.5
    return float(max_val), float(drift)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        # BGR order (OpenCV): weights = [B, G, R]
        return np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 4:
        # RGBA from PIL: channel order is RGB, so use RGB weights
        return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return img


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize((w, h), Image.NEAREST)
    return np.array(pil)


def _load_image(path: str, asset_root: str | None = None) -> np.ndarray | None:
    if asset_root is not None:
        import os
        base = os.path.realpath(asset_root)
        resolved = os.path.realpath(os.path.join(base, path) if not os.path.isabs(path) else path)
        if not resolved.startswith(base + os.sep) and resolved != base:
            return None
        path = resolved
    if _HAS_CV2:
        img = cv2.imread(path)
        if img is not None:
            return img
    try:
        from PIL import Image
        pil = Image.open(path)
        return np.array(pil.convert("RGB"))
    except Exception:
        return None
