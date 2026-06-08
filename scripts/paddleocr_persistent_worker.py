# -*- coding: utf-8 -*-
"""Persistent PaddleOCR worker.

Line protocol:
stdin  -> {"image_path": "...", "lang": "ch"}
stdout -> {"ok": true, "texts": [...], "bboxes_xyxy": [...], "time": 0.123, ...}

The process stays alive and keeps PaddleOCR model instances cached by language.
"""

from __future__ import annotations

import ctypes
import io
import json
import logging
import os
import sys
import time
from typing import Any

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")


def _preload_nvidia_dlls() -> None:
    """Pre-load NVIDIA DLLs so PaddleOCR can use GPU.

    Resolution order:
    1. OPENCLAW_NVIDIA_SITE_PACKAGES env var (explicit override)
    2. Auto-detect from current Python's site-packages
    """
    site = os.environ.get("OPENCLAW_NVIDIA_SITE_PACKAGES", "")
    if not site or not os.path.exists(site):
        import site as site_mod

        for sp in site_mod.getsitepackages():
            if os.path.exists(os.path.join(sp, "nvidia")):
                site = sp
                break
    nvidia_bins = [
        os.path.join(site, "nvidia", d, "bin")
        for d in ["cublas", "cudnn", "cuda_runtime", "curand", "cufft", "cusparse", "cusolver"]
    ]
    for directory in nvidia_bins:
        if not os.path.exists(directory):
            continue
        try:
            os.add_dll_directory(directory)
        except Exception:
            pass
        for dll in os.listdir(directory):
            if not dll.endswith(".dll"):
                continue
            try:
                ctypes.CDLL(os.path.join(directory, dll))
            except Exception:
                pass


_preload_nvidia_dlls()

import numpy as np
from PIL import Image


_OCR_BY_LANG: dict[str, Any] = {}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=_stdout, flush=True)


def _poly_to_xyxy(poly: Any) -> list[float]:
    pts = poly.tolist() if hasattr(poly, "tolist") else poly
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _get_ocr(lang: str) -> Any:
    if lang in _OCR_BY_LANG:
        return _OCR_BY_LANG[lang]

    from paddleocr import PaddleOCR

    t0 = time.time()
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    _OCR_BY_LANG[lang] = ocr
    _emit({"event": "model_loaded", "lang": lang, "load_time": round(time.time() - t0, 3)})
    return ocr


def _run_ocr(image_path: str, lang: str) -> dict[str, Any]:
    if not os.path.exists(image_path):
        return {"ok": False, "error": "file_not_found"}

    ocr = _get_ocr(lang)
    img = np.array(Image.open(image_path).convert("RGB"))

    t0 = time.time()
    result = ocr.predict(img)
    elapsed = time.time() - t0

    texts: list[str] = []
    bboxes_xyxy: list[list[float]] = []
    confidences: list[float] = []

    if result:
        r = result[0]
        res = r.res if hasattr(r, "res") else r
        rec_texts = res.get("rec_texts", [])
        rec_polys = res.get("rec_polys", [])
        rec_scores = res.get("rec_scores", []) or res.get("rec_confidences", [])

        for index, (poly, text) in enumerate(zip(rec_polys, rec_texts)):
            bboxes_xyxy.append(_poly_to_xyxy(poly))
            texts.append(text)
            if index < len(rec_scores):
                try:
                    confidences.append(float(rec_scores[index]))
                except Exception:
                    confidences.append(0.8)
            else:
                confidences.append(0.8)

    return {
        "ok": True,
        "texts": texts,
        "bboxes_xyxy": bboxes_xyxy,
        "confidences": confidences,
        "time": round(elapsed, 3),
        "engine": "PaddleOCR",
        "model": "PP-OCRv5",
    }


def main() -> int:
    _emit({"event": "ready", "worker": "paddleocr_persistent"})
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
            if request.get("op") == "shutdown":
                _emit({"ok": True, "event": "shutdown"})
                return 0
            image_path = str(request.get("image_path") or "")
            lang = str(request.get("lang") or "ch")
            _emit(_run_ocr(image_path, lang))
        except Exception as exc:
            _emit({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
