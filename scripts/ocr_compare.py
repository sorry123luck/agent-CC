"""
OCR 方案对比实验脚本

对比多种 OCR 方案在不同场景下的效果：
1. EasyOCR 原始
2. EasyOCR + 预处理（灰度/对比度/反色/二值化）
3. PaddleOCR
4. Tesseract（对照组）

测试场景：
- VS Code 深色主题截图（GUI 场景）
- 小字号文本
- 中英文混合内容
"""
import sys
import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEBUG_DIR = Path("data/debug")
TEST_IMGS = {
    "vscode_dark": DEBUG_DIR / "viz_raw_screenshot.png",
    # 备用：用同一张图但裁剪不同区域代表不同场景
}


@dataclass
class OCRResult:
    """OCR 测试结果"""
    method: str
    blocks: int
    time_ms: float
    texts: list[str] = field(default_factory=list)
    sample_texts: list[str] = field(default_factory=list)
    avg_confidence: float = 0.0
    legible_chars: int = 0  # 可辨认字符数（人工评估）
    chinese_chars: int = 0
    english_chars: int = 0


def preprocess_image(img: Image.Image, method: str) -> Image.Image:
    """图像预处理"""
    if method == "original":
        return img
    elif method == "grayscale":
        return img.convert("L").convert("RGB")
    elif method == "contrast_1.5":
        return ImageEnhance.Contrast(img).enhance(1.5)
    elif method == "contrast_2.0":
        return ImageEnhance.Contrast(img).enhance(2.0)
    elif method == "brightness_1.5":
        return ImageEnhance.Brightness(img).enhance(1.5)
    elif method == "brightness_2.0":
        return ImageEnhance.Brightness(img).enhance(2.0)
    elif method == "invert":
        gray = img.convert("L")
        return ImageOps.invert(gray).convert("RGB")
    elif method == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    elif method == "edge_enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    elif method == "grayscale_invert":
        gray = img.convert("L")
        inv = ImageOps.invert(gray)
        return inv.convert("RGB")
    elif method == "high_contrast_bw":
        # 高对比度灰度 + 反转（模拟白底黑字）
        gray = img.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(3.0)
        return ImageOps.invert(high_contrast).convert("RGB")
    elif method == "adaptive_threshold":
        gray = img.convert("L")
        # 自适应阈值二值化
        arr = np.array(gray)
        mean_val = arr.mean()
        binary = (arr > mean_val * 0.8).astype(np.uint8) * 255
        return Image.fromarray(binary).convert("RGB")
    else:
        return img


def eval_easyocr(img: Image.Image, preprocess: str = "original") -> OCRResult:
    """EasyOCR 评估"""
    import easyocr

    t0 = time.time()
    processed = preprocess_image(img, preprocess)
    arr = np.array(processed)

    reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
    raw = reader.readtext(arr)
    elapsed = (time.time() - t0) * 1000

    texts = [item[1].strip() for item in raw]
    confidences = [item[2] for item in raw]

    result = OCRResult(
        method=f"EasyOCR_{preprocess}" if preprocess != "original" else "EasyOCR_original",
        blocks=len(texts),
        time_ms=elapsed,
        texts=texts,
        sample_texts=texts[:10],
        avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
    )

    # 字符统计
    for t in texts:
        for c in t:
            if '\u4e00' <= c <= '\u9fff':
                result.chinese_chars += 1
            elif c.isalpha() and ord(c) < 128:
                result.english_chars += 1

    return result


def eval_paddleocr(img: Image.Image) -> OCRResult:
    """PaddleOCR 评估"""
    try:
        from paddleocr import PaddleOCR

        t0 = time.time()
        arr = np.array(img)

        ocr = PaddleOCR(lang="en", show_log=False, enable_mkldnn=True)
        raw = ocr.ocr(arr)
        elapsed = (time.time() - t0) * 1000

        texts = []
        confidences = []
        if raw and raw[0]:
            for line in raw[0]:
                if line and len(line) >= 2:
                    text = line[1][0].strip() if isinstance(line[1], (list, tuple)) else str(line[1]).strip()
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                    texts.append(text)
                    confidences.append(conf)

        result = OCRResult(
            method="PaddleOCR",
            blocks=len(texts),
            time_ms=elapsed,
            texts=texts,
            sample_texts=texts[:10],
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
        )

        for t in texts:
            for c in t:
                if '\u4e00' <= c <= '\u9fff':
                    result.chinese_chars += 1
                elif c.isalpha() and ord(c) < 128:
                    result.english_chars += 1

        return result
    except Exception as e:
        return OCRResult(
            method="PaddleOCR",
            blocks=0,
            time_ms=0,
            texts=[],
            sample_texts=[f"ERROR: {type(e).__name__}: {str(e)[:100]}"],
        )


def eval_tesseract(img: Image.Image, preprocess: str = "original") -> OCRResult:
    """Tesseract OCR 评估"""
    try:
        import pytesseract

        t0 = time.time()
        processed = preprocess_image(img, preprocess)

        # Tesseract 配置
        config = "--oem 3 --psm 6 -l chi_sim+eng"
        raw = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        elapsed = (time.time() - t0) * 1000

        texts = []
        confidences = []
        for i, text in enumerate(raw["text"]):
            if text.strip():
                texts.append(text.strip())
                conf = int(raw["conf"][i])
                confidences.append(conf / 100.0)

        result = OCRResult(
            method=f"Tesseract_{preprocess}" if preprocess != "original" else "Tesseract_original",
            blocks=len(texts),
            time_ms=elapsed,
            texts=texts,
            sample_texts=texts[:10],
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
        )

        for t in texts:
            for c in t:
                if '\u4e00' <= c <= '\u9fff':
                    result.chinese_chars += 1
                elif c.isalpha() and ord(c) < 128:
                    result.english_chars += 1

        return result
    except Exception as e:
        return OCRResult(
            method=f"Tesseract_{preprocess}" if preprocess != "original" else "Tesseract_original",
            blocks=0,
            time_ms=0,
            texts=[],
            sample_texts=[f"ERROR: {type(e).__name__}: {str(e)[:100]}"],
        )


def run_comparison():
    """运行完整对比实验"""
    print("=" * 80)
    print("OCR 方案对比实验")
    print("=" * 80)

    # 加载测试图片
    img_path = TEST_IMGS["vscode_dark"]
    if not img_path.exists():
        print(f"测试图片不存在: {img_path}")
        return

    img = Image.open(img_path)
    print(f"\n测试图片: {img_path.name}")
    print(f"图片尺寸: {img.size}, 模式: {img.mode}")
    print(f"内容区: 1920x1032 (VS Code 深色主题)")

    results: list[OCRResult] = []

    # ============ EasyOCR 原始 + 预处理 ============
    print("\n--- EasyOCR 系列 ---")

    easyocr_methods = [
        "original",
        "grayscale",
        "contrast_1.5",
        "contrast_2.0",
        "brightness_1.5",
        "brightness_2.0",
        "invert",
        "sharpen",
        "edge_enhance",
        "grayscale_invert",
        "high_contrast_bw",
        "adaptive_threshold",
    ]

    for method in easyocr_methods:
        print(f"  Testing EasyOCR_{method}...", end=" ", flush=True)
        try:
            result = eval_easyocr(img, method)
            results.append(result)
            print(f"[{result.blocks} blocks, {result.time_ms:.0f}ms, avg_conf={result.avg_confidence:.3f}]")
        except Exception as e:
            print(f"[ERROR: {e}]")
            results.append(OCRResult(method=f"EasyOCR_{method}", blocks=0, time_ms=0,
                                     sample_texts=[f"ERROR: {e}"]))

    # ============ PaddleOCR ============
    print("\n--- PaddleOCR ---")
    print("  Testing PaddleOCR...", end=" ", flush=True)
    result = eval_paddleocr(img)
    results.append(result)
    print(f"[{result.blocks} blocks, {result.time_ms:.0f}ms, avg_conf={result.avg_confidence:.3f}]")

    # ============ Tesseract ============
    print("\n--- Tesseract 系列 ---")
    tesseract_methods = ["original", "grayscale", "high_contrast_bw"]
    for method in tesseract_methods:
        print(f"  Testing Tesseract_{method}...", end=" ", flush=True)
        try:
            result = eval_tesseract(img, method)
            results.append(result)
            print(f"[{result.blocks} blocks, {result.time_ms:.0f}ms, avg_conf={result.avg_confidence:.3f}]")
        except Exception as e:
            print(f"[ERROR: {e}]")
            results.append(OCRResult(method=f"Tesseract_{method}", blocks=0, time_ms=0,
                                     sample_texts=[f"ERROR: {e}"]))

    # ============ 输出汇总表格 ============
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)
    print(f"{'方法':<30} {'块数':>6} {'耗时(ms)':>10} {'均置信度':>8} {'中文字符':>10} {'英文字符':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r.method:<30} {r.blocks:>6} {r.time_ms:>10.0f} {r.avg_confidence:>8.3f} {r.chinese_chars:>10} {r.english_chars:>10}")

    # ============ 样本输出 ============
    print("\n" + "=" * 80)
    print("样本输出（前 10 个文本块）")
    print("=" * 80)
    for r in results:
        print(f"\n### {r.method}")
        if r.sample_texts:
            for i, t in enumerate(r.sample_texts[:10]):
                print(f"  [{i+1}] \"{t}\"")
        else:
            print("  (无输出)")

    # ============ 场景评估 ============
    print("\n" + "=" * 80)
    print("场景评估（基于 VS Code 深色主题截图）")
    print("=" * 80)

    # 找出各指标最佳方案
    best_blocks = max(results, key=lambda r: r.blocks)
    best_time = min([r for r in results if r.blocks > 0], key=lambda r: r.time_ms)
    best_conf = max([r for r in results if r.blocks > 0], key=lambda r: r.avg_confidence)
    best_chinese = max(results, key=lambda r: r.chinese_chars)
    best_english = max(results, key=lambda r: r.english_chars)

    print(f"输出块数最多: {best_blocks.method} ({best_blocks.blocks} blocks)")
    print(f"速度最快: {best_time.method} ({best_time.time_ms:.0f}ms)")
    print(f"平均置信度最高: {best_conf.method} ({best_conf.avg_confidence:.3f})")
    print(f"中文字符最多: {best_chinese.method} ({best_chinese.chinese_chars} chars)")
    print(f"英文字符最多: {best_english.method} ({best_english.english_chars} chars)")

    # 评估每个方法的可读性（人工判断样本输出中正常字符占比）
    print("\n--- 可读性评估（样本分析）---")
    for r in results:
        legible = 0
        garbage = 0
        for t in r.texts:
            # 简单启发式：包含至少一个正常可读字符（字母或中文）
            has_normal = any('\u4e00' <= c <= '\u9fff' or c.isalpha() for c in t)
            if has_normal:
                legible += 1
            else:
                garbage += 1
        total = legible + garbage
        ratio = legible / total if total > 0 else 0
        print(f"  {r.method:<35}: 可读 {legible}/{total} ({ratio:.0%})")

    return results


def test_synthetic_scene():
    """在合成清晰图片上验证 OCR 基本功能是否正常"""
    print("\n" + "=" * 80)
    print("合成场景验证（白底黑字，验证 OCR 基础功能）")
    print("=" * 80)

    from PIL import Image, ImageDraw, ImageFont

    test_cases = [
        ("chinese_mixed", "Hello 世界 123 ABC 你好"),
        ("english_only", "Hello World 123"),
        ("code_like", "def main(): print('hello')"),
        ("ui_text", "File  Edit  View  Help"),
    ]

    img_w, img_h = 600, 200
    font_size = 24
    font = None
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                pass

    if not font:
        print("  无法加载字体，跳过合成测试")
        return

    results = []
    for name, text in test_cases:
        test_img = Image.new("RGB", (img_w, img_h), color="white")
        draw = ImageDraw.Draw(test_img)
        draw.text((20, 80), text, fill="black", font=font)
        test_path = DEBUG_DIR / f"test_{name}.png"
        test_img.save(test_path)
        print(f"\n### {name}: \"{text}\"")
        print(f"  保存到: {test_path}")

        # EasyOCR
        try:
            r = eval_easyocr(test_img, "original")
            print(f"  EasyOCR: {r.blocks} blocks, \"{r.sample_texts}\"")
            results.append((name, "EasyOCR", r))
        except Exception as e:
            print(f"  EasyOCR ERROR: {e}")

        # Tesseract
        try:
            r = eval_tesseract(test_img, "original")
            print(f"  Tesseract: {r.blocks} blocks, \"{r.sample_texts}\"")
            results.append((name, "Tesseract", r))
        except Exception as e:
            print(f"  Tesseract ERROR: {e}")

    return results


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)

    # 1. 合成场景验证（验证 OCR 基础功能）
    synthetic_results = test_synthetic_scene()

    # 2. 真实场景对比
    real_results = run_comparison()

    print("\n" + "=" * 80)
    print("实验完成")
    print("=" * 80)
