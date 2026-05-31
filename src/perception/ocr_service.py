"""PaddleOCR bridge service for Pro baseline OCR intake."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import tempfile
import threading
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.common.utils import read_yaml


_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class OCRTextBlock:
    """Structured OCR text block in absolute image coordinates."""

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class OCRExtractionResult:
    """OCR extraction result plus provider metadata."""

    blocks: list[OCRTextBlock] = field(default_factory=list)
    provider: str = "paddleocr_bridge"
    worker_command: list[str] = field(default_factory=list)
    elapsed_seconds: float | None = None
    raw_format: str = "xyxy"
    used_region: tuple[int, int, int, int] | None = None
    success: bool = False
    error: str | None = None
    worker_mode: str = "subprocess"
    worker_reused: bool = False
    startup_seconds: float | None = None
    fallback_reason: str | None = None


@dataclass
class OCRServiceConfig:
    """Service-side OCR bridge configuration."""

    provider: str = "paddleocr_bridge"
    lang: str = "en"
    output_bb_format: str = "xyxy"
    worker_python: str = ""
    worker_script: str = str(_ROOT_DIR / "vendor" / "omniparser_runtime" / "workers" / "paddleocr_worker.py")
    persistent_worker_script: str = str(_ROOT_DIR / "scripts" / "paddleocr_persistent_worker.py")
    worker_mode: str = "persistent"
    worker_timeout_seconds: int = 120
    bridge_enabled: bool = True


class _PersistentOCRWorker:
    """JSON-lines PaddleOCR worker wrapper with one cached process."""

    def __init__(self, python_path: str, script_path: str, timeout_seconds: int) -> None:
        self.python_path = python_path
        self.script_path = script_path
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen | None = None
        self._stdout_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._lock = threading.Lock()
        self.last_startup_seconds: float | None = None
        self.last_reused: bool = False

    def run(self, image_path: str, lang: str) -> dict[str, Any]:
        import time

        with self._lock:
            reused = self._process is not None and self._process.poll() is None
            if not reused:
                start = time.perf_counter()
                self._start()
                self.last_startup_seconds = time.perf_counter() - start
            else:
                self.last_startup_seconds = 0.0
            self.last_reused = reused

            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.write(json.dumps({"image_path": image_path, "lang": lang}, ensure_ascii=False) + "\n")
            self._process.stdin.flush()

            deadline = time.perf_counter() + self.timeout_seconds
            while True:
                if self._process.poll() is not None:
                    raise RuntimeError(self._format_process_exit_error())
                remaining = max(0.0, deadline - time.perf_counter())
                if remaining <= 0:
                    raise TimeoutError("persistent_ocr_worker_timeout")
                try:
                    payload = self._stdout_queue.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue
                if payload.get("event") in {"ready", "model_loaded"}:
                    continue
                return payload

    def _start(self) -> None:
        import time

        script_path = self._script_path_for_subprocess()
        self._process = subprocess.Popen(
            [self.python_path, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True, name="paddleocr-stdout").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="paddleocr-stderr").start()

        # The worker emits "ready" before loading PaddleOCR. If the script path
        # cannot be opened (notably non-ASCII project paths under some venv
        # launchers), fail quickly instead of waiting for the full OCR timeout.
        deadline = time.perf_counter() + min(8.0, float(self.timeout_seconds))
        while time.perf_counter() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(self._format_process_exit_error())
            try:
                payload = self._stdout_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload.get("event") == "ready":
                return
            if payload.get("ok") is False:
                raise RuntimeError(str(payload.get("error") or "persistent_ocr_worker_start_failed"))
        raise TimeoutError("persistent_ocr_worker_start_timeout")

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                self._stdout_queue.put(json.loads(raw))
            except json.JSONDecodeError:
                self._stdout_queue.put({"ok": False, "error": raw})

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            raw = line.strip()
            if raw:
                self._stderr_lines.append(raw)

    def _script_path_for_subprocess(self) -> Path:
        source = Path(self.script_path)
        if not source.exists():
            raise FileNotFoundError(str(source))
        if self._is_ascii_path(source):
            return source

        target_dir = Path(tempfile.gettempdir()) / "openclaw_ocr_worker"
        target_dir.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha1(str(source.resolve()).encode("utf-8", errors="replace")).hexdigest()[:10]
        target = target_dir / f"{source.stem}_{source_key}{source.suffix}"
        if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(source, target)
        return target

    def _is_ascii_path(self, path: Path) -> bool:
        try:
            str(path).encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def _format_process_exit_error(self) -> str:
        code = self._process.returncode if self._process is not None else "unknown"
        stderr = "\n".join(self._stderr_lines[-8:]).strip()
        if stderr:
            return f"persistent_ocr_worker_exited:{code}: {stderr}"
        return f"persistent_ocr_worker_exited:{code}"


class OCRService:
    """Singleton OCR bridge service."""

    _instance: "OCRService | None" = None

    def __new__(cls) -> "OCRService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str | Path = "config/models.yaml") -> None:
        if getattr(self, "_initialized", False):
            return
        self._config_path = Path(config_path)
        self._config = self._load_config(self._config_path)
        self._persistent_worker: _PersistentOCRWorker | None = None
        self._initialized = True

    def warm_up(self) -> OCRExtractionResult:
        """Load the persistent OCR worker before the first real screenshot."""
        image = Image.new("RGB", (1, 1), color="white")
        return self.extract_with_metadata(
            image=image,
            min_text_length=1,
            filter_pure_digits=False,
            filter_pure_symbols=False,
        )

    def extract(
        self,
        image: Image.Image,
        min_text_length: int = 2,
        filter_pure_digits: bool = True,
        filter_pure_symbols: bool = True,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[OCRTextBlock]:
        """Backward-compatible OCR extraction API."""
        result = self.extract_with_metadata(
            image=image,
            min_text_length=min_text_length,
            filter_pure_digits=filter_pure_digits,
            filter_pure_symbols=filter_pure_symbols,
            region=region,
        )
        return result.blocks

    def extract_with_metadata(
        self,
        image: Image.Image,
        min_text_length: int = 2,
        filter_pure_digits: bool = True,
        filter_pure_symbols: bool = True,
        region: tuple[int, int, int, int] | None = None,
    ) -> OCRExtractionResult:
        """Run OCR via external PaddleOCR bridge worker."""
        if not self._config.bridge_enabled:
            return OCRExtractionResult(
                success=False,
                error="ocr_bridge_disabled",
                used_region=region,
            )

        working_image = image.crop(region) if region else image
        region_offset_x = region[0] if region else 0
        region_offset_y = region[1] if region else 0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            working_image.save(temp_path)
            fallback_reason = None
            worker_mode = self._config.worker_mode
            worker_reused = False
            startup_seconds = None
            worker_command = self._subprocess_command(temp_path)
            try:
                if self._config.worker_mode == "persistent":
                    payload = self._run_persistent_worker(temp_path)
                    worker_command = [
                        self._config.worker_python,
                        self._config.persistent_worker_script,
                    ]
                    worker_reused = bool(self._persistent_worker and self._persistent_worker.last_reused)
                    startup_seconds = self._persistent_worker.last_startup_seconds if self._persistent_worker else None
                else:
                    payload = self._run_subprocess_worker(temp_path)
            except Exception as exc:
                if self._config.worker_mode != "persistent":
                    raise
                fallback_reason = str(exc)
                worker_mode = "subprocess"
                worker_reused = False
                payload = self._run_subprocess_worker(temp_path)
                worker_command = self._subprocess_command(temp_path)

            if payload.get("ok") is False:
                return OCRExtractionResult(
                    provider=self._config.provider,
                    worker_command=worker_command,
                    used_region=region,
                    success=False,
                    error=str(payload.get("error") or "ocr_worker_failed"),
                    worker_mode=worker_mode,
                    worker_reused=worker_reused,
                    startup_seconds=startup_seconds,
                    fallback_reason=fallback_reason,
                )
            blocks = self._parse_blocks(
                payload=payload,
                region_offset=(region_offset_x, region_offset_y),
                min_text_length=min_text_length,
                filter_pure_digits=filter_pure_digits,
                filter_pure_symbols=filter_pure_symbols,
            )
            return OCRExtractionResult(
                blocks=blocks,
                provider=self._config.provider,
                worker_command=worker_command,
                elapsed_seconds=self._coerce_float(payload.get("time")),
                raw_format=self._config.output_bb_format,
                used_region=region,
                success=True,
                worker_mode=worker_mode,
                worker_reused=worker_reused,
                startup_seconds=startup_seconds,
                fallback_reason=fallback_reason,
            )
        except subprocess.TimeoutExpired:
            return OCRExtractionResult(
                provider=self._config.provider,
                worker_command=[
                    self._config.worker_python,
                    self._config.worker_script,
                    self._config.lang,
                ],
                used_region=region,
                success=False,
                error="ocr_worker_timeout",
                worker_mode=self._config.worker_mode,
            )
        except (OSError, json.JSONDecodeError) as exc:
            return OCRExtractionResult(
                provider=self._config.provider,
                worker_command=[
                    self._config.worker_python,
                    self._config.worker_script,
                    self._config.lang,
                ],
                used_region=region,
                success=False,
                error=str(exc),
                worker_mode=self._config.worker_mode,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _subprocess_command(self, temp_path: Path) -> list[str]:
        return [
            self._config.worker_python,
            self._config.worker_script,
            str(temp_path),
            self._config.lang,
        ]

    def _run_subprocess_worker(self, temp_path: Path) -> dict[str, Any]:
        worker_command = self._subprocess_command(temp_path)
        completed = subprocess.run(
            worker_command,
            capture_output=True,
            timeout=self._config.worker_timeout_seconds,
            check=False,
        )
        stdout_text = self._decode_stream(completed.stdout)
        stderr_text = self._decode_stream(completed.stderr)
        if completed.returncode != 0:
            error = stderr_text.strip() or stdout_text.strip() or "ocr_worker_failed"
            return {"ok": False, "error": error}
        return self._load_worker_payload(stdout_text)

    def _run_persistent_worker(self, temp_path: Path) -> dict[str, Any]:
        if self._persistent_worker is None:
            self._persistent_worker = _PersistentOCRWorker(
                self._config.worker_python,
                self._config.persistent_worker_script,
                self._config.worker_timeout_seconds,
            )
        return self._persistent_worker.run(str(temp_path), self._config.lang)

    def _parse_blocks(
        self,
        payload: dict[str, Any],
        region_offset: tuple[int, int],
        min_text_length: int,
        filter_pure_digits: bool,
        filter_pure_symbols: bool,
    ) -> list[OCRTextBlock]:
        """Normalize worker JSON into OCRTextBlock list."""
        texts = payload.get("texts") or []
        boxes = payload.get("bboxes_xyxy") or payload.get("bboxes") or []
        confidences = payload.get("confidences") or []
        offset_x, offset_y = region_offset
        blocks: list[OCRTextBlock] = []

        for index, text in enumerate(texts):
            if index >= len(boxes):
                break
            normalized_text = (text or "").strip()
            if len(normalized_text) < min_text_length:
                continue
            if filter_pure_digits and normalized_text.isdigit():
                continue
            if filter_pure_symbols and self._is_pure_symbols(normalized_text):
                continue

            bbox = self._normalize_bbox(boxes[index], offset_x=offset_x, offset_y=offset_y)
            if bbox is None:
                continue

            confidence = self._coerce_float(confidences[index] if index < len(confidences) else None)
            blocks.append(
                OCRTextBlock(
                    text=normalized_text,
                    bbox=bbox,
                    confidence=confidence if confidence is not None else 0.8,
                )
            )
        return blocks

    def _normalize_bbox(
        self,
        bbox: Any,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> tuple[int, int, int, int] | None:
        """Normalize xyxy/xywh arrays into absolute xyxy."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        if self._config.output_bb_format == "xywh":
            x2 = x1 + x2
            y2 = y1 + y2
        return (
            int(x1) + offset_x,
            int(y1) + offset_y,
            int(x2) + offset_x,
            int(y2) + offset_y,
        )

    def _load_config(self, config_path: Path) -> OCRServiceConfig:
        """Load OCR bridge config from models.yaml."""
        if not config_path.exists():
            return OCRServiceConfig()
        raw = read_yaml(str(config_path)) or {}
        ocr_cfg = (raw.get("models") or {}).get("ocr") or {}
        return OCRServiceConfig(
            provider=ocr_cfg.get("provider", "paddleocr_bridge"),
            lang=ocr_cfg.get("lang", "en"),
            output_bb_format=ocr_cfg.get("output_bb_format", "xyxy"),
            worker_python=ocr_cfg.get("worker_python", OCRServiceConfig.worker_python),
            worker_script=ocr_cfg.get("worker_script", OCRServiceConfig.worker_script),
            persistent_worker_script=ocr_cfg.get(
                "persistent_worker_script",
                OCRServiceConfig.persistent_worker_script,
            ),
            worker_mode=ocr_cfg.get("worker_mode", "persistent"),
            worker_timeout_seconds=int(ocr_cfg.get("worker_timeout_seconds", 120)),
            bridge_enabled=bool(ocr_cfg.get("bridge_enabled", True)),
        )

    def _is_pure_symbols(self, text: str) -> bool:
        """Check whether text contains only symbols."""
        import re

        return not bool(re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", text))

    def _coerce_float(self, value: Any) -> float | None:
        """Best-effort float conversion."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _decode_stream(self, value: bytes | str | None) -> str:
        """Best-effort decode for worker stdout/stderr on Windows."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value

        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")

    def _load_worker_payload(self, stdout_text: str) -> dict[str, Any]:
        """Parse worker stdout into JSON payload, tolerating noisy wrappers."""
        normalized = (stdout_text or "").strip()
        if not normalized:
            return {}
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            start = normalized.find("{")
            end = normalized.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(normalized[start : end + 1])


_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    """Get the global OCR service singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
