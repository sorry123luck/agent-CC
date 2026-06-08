# -*- coding: utf-8 -*-
"""
OCR Bridge - subprocess-based external OCR runner.
Currently supports PaddleOCR GPU via isolated venv.
Returns (texts, bboxes_xyxy) tuple compatible with OmniParser check_ocr_box format.
"""
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path


GPU_VENV_PY = os.environ.get("OPENCLAW_PADDLE_PY", r"D:\ocr-paddle-env\Scripts\python.exe")
WORKER_SCRIPT = os.environ.get(
    "OPENCLAW_OMNI_OCR_WORKER",
    str(Path(__file__).resolve().parents[1] / "workers" / "paddleocr_worker.py"),
)


def run_paddle_ocr(image_source, output_bb_format='xyxy'):
    """
    Run PaddleOCR GPU via isolated subprocess.

    Args:
        image_source: PIL Image or file path
        output_bb_format: 'xyxy' (default) - returns [x1,y1,x2,y2] normalized

    Returns:
        ((texts, bboxes_xywh), False) - compatible with check_ocr_box return format
    """
    # Save image to temp file if PIL Image
    if hasattr(image_source, 'convert'):
        # PIL Image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        try:
            image_source.save(temp_path)
            return _run_worker(temp_path, output_bb_format)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # File path
        return _run_worker(str(image_source), output_bb_format)


def _run_worker(image_path, output_bb_format):
    env = os.environ.copy()
    # Ensure subprocess inherits clean env (no torch DLL pollution)
    env.pop('PATH', None)
    env['PATH'] = os.environ.get('SYSTEMROOT', '') + r'\System32;' + \
                   os.environ.get('SYSTEMROOT', '') + r'\System32\wbem'

    result = subprocess.run(
        [GPU_VENV_PY, WORKER_SCRIPT, image_path],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=120,
        env=env
    )

    if result.returncode != 0:
        raise RuntimeError(f"PaddleOCR worker failed: {result.stderr[-500:]}")

    data = json.loads(result.stdout)

    if 'error' in data:
        raise RuntimeError(f"PaddleOCR worker error: {data['error']}")

    texts = data['texts']
    bboxes_xyxy = data['bboxes_xyxy']  # already absolute xyxy from worker

    if output_bb_format == 'xywh':
        # Convert xyxy to xywh
        bboxes_xywh = []
        for bb in bboxes_xyxy:
            x1, y1, x2, y2 = bb
            bboxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        return (texts, bboxes_xywh), False
    else:
        # xyxy - already in correct format
        return (texts, bboxes_xyxy), False
