"""OCRService unit tests."""

from unittest.mock import MagicMock, patch

from PIL import Image

from src.perception.ocr_service import OCRTextBlock, OCRService, _PersistentOCRWorker


def _force_subprocess(service: OCRService) -> OCRService:
    """Keep legacy unit tests on the deterministic subprocess path."""
    service._config.worker_mode = "subprocess"  # noqa: SLF001
    return service


class TestOCRTextBlock:
    """OCRTextBlock 数据模型测试"""

    def test_text_block_creation(self):
        block = OCRTextBlock(
            text="测试文本",
            bbox=(10, 20, 100, 40),
            confidence=0.95,
        )
        assert block.text == "测试文本"
        assert block.bbox == (10, 20, 100, 40)
        assert block.confidence == 0.95


class TestOCRTextBlockBbox:
    """OCRTextBlock bbox 属性测试"""

    def test_bbox_dimensions(self):
        """bbox 解析"""
        block = OCRTextBlock(
            text="Hello",
            bbox=(50, 100, 150, 130),
            confidence=0.9,
        )
        # bbox = (left, top, right, bottom)
        left, top, right, bottom = block.bbox
        width = right - left
        height = bottom - top
        assert width == 100
        assert height == 30


class TestOCRService:
    """PaddleOCR bridge adapter tests."""

    def test_extract_with_metadata_parses_worker_json(self):
        service = _force_subprocess(OCRService())
        image = Image.new("RGB", (100, 80), color="white")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = (
            '{"texts":["Send","123"],"bboxes_xyxy":[[10,10,60,30],[70,10,90,30]],'
            '"confidences":[0.93,0.88],"time":0.42}'
        )
        completed.stderr = ""

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image)

        assert result.success is True
        assert result.elapsed_seconds == 0.42
        assert len(result.blocks) == 1
        assert result.blocks[0].text == "Send"
        assert result.blocks[0].bbox == (10, 10, 60, 30)
        assert result.worker_command[-1] == service._config.lang

    def test_extract_with_region_offsets_bboxes(self):
        service = _force_subprocess(OCRService())
        image = Image.new("RGB", (200, 120), color="white")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = (
            '{"texts":["Search"],"bboxes_xyxy":[[5,8,45,22]],'
            '"confidences":[0.91],"time":0.11}'
        )
        completed.stderr = ""

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image, region=(20, 30, 120, 90))

        assert result.success is True
        assert result.blocks[0].bbox == (25, 38, 65, 52)

    def test_extract_with_metadata_handles_worker_error(self):
        service = _force_subprocess(OCRService())
        image = Image.new("RGB", (100, 80), color="white")
        completed = MagicMock()
        completed.returncode = 1
        completed.stdout = ""
        completed.stderr = "worker failed"

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image)

        assert result.success is False
        assert result.error == "worker failed"

    def test_extract_with_metadata_decodes_bytes_stdout(self):
        service = _force_subprocess(OCRService())
        image = Image.new("RGB", (100, 80), color="white")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = (
            '{"texts":["搜索"],"bboxes_xyxy":[[10,10,80,30]],"confidences":[0.95],"time":0.33}'
        ).encode("utf-8")
        completed.stderr = b""

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image)

        assert result.success is True
        assert result.blocks[0].text == "搜索"
        assert result.blocks[0].bbox == (10, 10, 80, 30)

    def test_extract_with_metadata_handles_empty_stdout_without_crash(self):
        service = _force_subprocess(OCRService())
        image = Image.new("RGB", (100, 80), color="white")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = None
        completed.stderr = None

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image)

        assert result.success is True
        assert result.blocks == []

    def test_is_pure_symbols_treats_chinese_as_text(self):
        service = OCRService()
        assert service._is_pure_symbols("搜索") is False
        assert service._is_pure_symbols("发送") is False
        assert service._is_pure_symbols("...") is True

    def test_config_defaults_to_persistent_worker_mode(self):
        service = OCRService()
        service._config = service._load_config(service._config_path)  # noqa: SLF001
        assert service._config.worker_mode == "persistent"
        assert service._config.persistent_worker_script.endswith("paddleocr_persistent_worker.py")

    def test_persistent_failure_falls_back_to_subprocess(self):
        service = OCRService()
        service._config.worker_mode = "persistent"  # noqa: SLF001
        service._persistent_worker = MagicMock()  # noqa: SLF001
        service._persistent_worker.run.side_effect = RuntimeError("worker died")  # noqa: SLF001
        image = Image.new("RGB", (100, 80), color="white")
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = '{"texts":["搜索"],"bboxes_xyxy":[[10,10,80,30]],"confidences":[0.95],"time":0.33}'
        completed.stderr = ""

        with patch("src.perception.ocr_service.subprocess.run", return_value=completed):
            result = service.extract_with_metadata(image)

        assert result.success is True
        assert result.worker_mode == "subprocess"
        assert result.worker_reused is False
        assert "worker died" in (result.fallback_reason or "")

    def test_persistent_worker_copies_non_ascii_script_path_to_ascii_temp(self, tmp_path):
        source_dir = tmp_path / "项目"
        source_dir.mkdir()
        source = source_dir / "paddleocr_persistent_worker.py"
        source.write_text("print('ready')\n", encoding="utf-8")
        worker = _PersistentOCRWorker("python", str(source), timeout_seconds=3)

        resolved = worker._script_path_for_subprocess()

        assert resolved.exists()
        assert resolved.name.startswith(source.stem)
        assert resolved.suffix == source.suffix
        str(resolved).encode("ascii")

    def test_warm_up_runs_tiny_ocr_request(self):
        service = OCRService()
        with patch.object(service, "extract_with_metadata") as extract:
            service.warm_up()

        assert extract.called is True
        image = extract.call_args.kwargs["image"]
        assert image.size == (1, 1)
