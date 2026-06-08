"""Tests for U3-shadow: OCR/Visual Density Region Proposal."""

from src.perception.ocr_visual_shadow import OcrVisualShadowPass


class TestOcrVisualShadowPass:
    def setup_method(self):
        self.pass_ = OcrVisualShadowPass()

    def test_no_ocr_blocks_returns_empty(self):
        result = self.pass_.run(ocr_blocks=None, window_width=800, window_height=600)
        assert result.ocr_text_rows_detected == 0
        assert "no_ocr_blocks" in result.rejected_reasons

    def test_no_window_returns_empty(self):
        result = self.pass_.run(ocr_blocks=[{"text": "test", "bbox": [0, 0, 50, 20]}], window_width=0, window_height=0)
        assert result.ocr_text_rows_detected == 0
        assert "screenshot_missing" in result.rejected_reasons

    def test_single_ocr_block_clustering(self):
        """Single wide OCR block should form one row."""
        blocks = [{"text": "Hello World", "bbox": [10, 10, 200, 30], "confidence": 0.9}]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.ocr_text_rows_detected == 1

    def test_horizontal_row_clustering(self):
        """Multiple blocks at similar Y should cluster into one row."""
        blocks = [
            {"text": "文件", "bbox": [10, 10, 100, 30], "confidence": 0.9},
            {"text": "编辑", "bbox": [110, 12, 200, 32], "confidence": 0.9},
            {"text": "查看", "bbox": [210, 11, 300, 31], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.ocr_text_rows_detected == 1
        assert result.ocr_text_rows[0]["text_count"] == 3

    def test_vertical_separation_creates_multiple_rows(self):
        """Blocks at different Y should form separate rows."""
        blocks = [
            {"text": "Menu Bar Text", "bbox": [10, 10, 200, 30], "confidence": 0.9},
            {"text": "Content Area Text", "bbox": [10, 200, 200, 220], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.ocr_text_rows_detected == 2

    def test_proposals_generated(self):
        """With OCR blocks, proposals should be generated."""
        blocks = [
            {"text": "文件", "bbox": [10, 10, 50, 30], "confidence": 0.9},
            {"text": "编辑", "bbox": [60, 10, 100, 30], "confidence": 0.9},
            {"text": "Content line 1", "bbox": [10, 100, 200, 120], "confidence": 0.9},
            {"text": "Content line 2", "bbox": [10, 130, 200, 150], "confidence": 0.9},
            {"text": "Content line 3", "bbox": [10, 160, 200, 180], "confidence": 0.9},
            {"text": "Content line 4", "bbox": [10, 190, 200, 210], "confidence": 0.9},
            {"text": "Content line 5", "bbox": [10, 220, 200, 240], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.proposals_count >= 1

    def test_whitespace_gap_detection(self):
        """Large vertical gap between OCR blocks should be detected."""
        blocks = [
            {"text": "Top", "bbox": [10, 10, 100, 30], "confidence": 0.9},
            {"text": "Bottom", "bbox": [10, 400, 100, 420], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert len(result.whitespace_gaps) >= 1

    def test_result_to_dict_serializable(self):
        """Result should be JSON serializable."""
        import json
        blocks = [{"text": "test", "bbox": [10, 10, 100, 30], "confidence": 0.9}]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        json.dumps(result.to_dict())

    def test_useful_neutral_noisy_counts(self):
        """useful + neutral + noisy should equal proposals_count."""
        blocks = [
            {"text": "文件", "bbox": [10, 10, 100, 30], "confidence": 0.9},
            {"text": "编辑", "bbox": [110, 10, 200, 30], "confidence": 0.9},
            {"text": "Content text", "bbox": [10, 200, 300, 220], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.useful_proposals + result.neutral_proposals + result.noisy_proposals == result.proposals_count

    def test_dense_text_area_is_neutral(self):
        """dense_text_area should be classified as neutral, not noisy."""
        # Place blocks in middle of window (not top/bottom bands)
        blocks = [
            {"text": "This is a long content line of text", "bbox": [10, 300, 300, 320], "confidence": 0.9},
            {"text": "Another long content line here", "bbox": [10, 330, 300, 350], "confidence": 0.9},
            {"text": "More content text in the middle", "bbox": [10, 360, 300, 380], "confidence": 0.9},
            {"text": "Even more content lines here", "bbox": [10, 390, 300, 410], "confidence": 0.9},
            {"text": "Fifth line of content text", "bbox": [10, 420, 300, 440], "confidence": 0.9},
            {"text": "Sixth line of content text", "bbox": [10, 450, 300, 470], "confidence": 0.9},
        ]
        result = self.pass_.run(ocr_blocks=blocks, window_width=800, window_height=600)
        assert result.neutral_proposals >= 1
        assert result.noisy_proposals == 0
