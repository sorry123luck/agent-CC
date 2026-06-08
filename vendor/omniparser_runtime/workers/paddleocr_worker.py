# -*- coding: utf-8 -*-
"""
PaddleOCR Worker - isolated GPU environment (D:\\ocr-paddle-env)
Input: image path via sys.argv[1]
Output: JSON to stdout
"""
import sys as _sys
import io
_stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)

import json
import time
import ctypes

# === Pre-load NVIDIA DLLs before any torch/paddle import ===
site = os.environ.get('OPENCLAW_NVIDIA_SITE_PACKAGES', '')
if not site or not os.path.exists(site):
    import site as _site_mod

    for _sp in _site_mod.getsitepackages():
        if os.path.exists(os.path.join(_sp, 'nvidia')):
            site = _sp
            break
nvidia_bins = [
    os.path.join(site, 'nvidia', d, 'bin')
    for d in ['cublas', 'cudnn', 'cuda_runtime', 'curand', 'cufft', 'cusparse', 'cusolver']
]
for d in nvidia_bins:
    if os.path.exists(d):
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        for dll in os.listdir(d):
            if dll.endswith('.dll'):
                try:
                    ctypes.CDLL(os.path.join(d, dll))
                except Exception:
                    pass
# ============================================================

import numpy as np
from PIL import Image


def poly_to_xyxy(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def run_ocr(image_path, lang='ch'):
    from paddleocr import PaddleOCR

    img = np.array(Image.open(image_path).convert('RGB'))

    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    t0 = time.time()
    result = ocr.predict(img)
    elapsed = time.time() - t0

    texts = []
    bboxes_xyxy = []

    if result:
        r = result[0]
        res = r.res if hasattr(r, 'res') else r
        rec_texts = res.get('rec_texts', [])
        rec_polys = res.get('rec_polys', [])

        for poly, txt in zip(rec_polys, rec_texts):
            bboxes_xyxy.append(poly_to_xyxy(poly.tolist() if hasattr(poly, 'tolist') else poly))
            texts.append(txt)

    return {
        "texts": texts,
        "bboxes_xyxy": bboxes_xyxy,
        "time": round(elapsed, 3),
        "engine": "PaddleOCR",
        "model": "PP-OCRv5"
    }


if __name__ == '__main__':
    if len(_sys.argv) < 2:
        print('{"error": "Usage: python paddleocr_worker.py <image_path>"}', file=_stdout)
        _sys.exit(1)

    image_path = _sys.argv[1]
    lang = _sys.argv[2] if len(_sys.argv) >= 3 else 'ch'
    if not os.path.exists(image_path):
        print('{"error": "File not found"}', file=_stdout)
        _sys.exit(1)

    try:
        result = run_ocr(image_path, lang=lang)
        print(json.dumps(result, ensure_ascii=False), file=_stdout)
    except Exception as e:
        print('{"error": "%s"}' % str(e), file=_stdout)
        _sys.exit(1)
