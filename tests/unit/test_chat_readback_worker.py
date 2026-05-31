from src.chat.readback_models import ChatMessageEvent, ChatReadbackResult, MessageAttachment
from src.chat.readback_worker import ChatReadbackWorker
from PIL import Image, ImageDraw


def test_chat_readback_result_separates_text_and_attachment_events():
    text_event = ChatMessageEvent(
        message_id="m1",
        sender="peer",
        message_type="text",
        text="你好",
        bounds=(100, 200, 300, 240),
        confidence=0.91,
        source="ocr_crop",
    )
    image_event = ChatMessageEvent(
        message_id="m2",
        sender="peer",
        message_type="image",
        text=None,
        bounds=(120, 260, 360, 420),
        confidence=0.82,
        source="local_attachment_shape",
        attachments=[
            MessageAttachment(
                attachment_type="image",
                bounds=(120, 260, 360, 420),
                thumbnail_crop_path="artifacts/sample/image.png",
                needs_vlm=True,
            )
        ],
    )

    result = ChatReadbackResult(window_id="hwnd-1", app_process="weixin.exe", events=[text_event, image_event])

    assert result.has_text("你好")
    assert result.events[1].attachments[0].needs_vlm is True


def test_chat_readback_result_matches_text_when_ocr_drops_spaces():
    result = ChatReadbackResult(
        window_id="hwnd-1",
        app_process="weixin.exe",
        events=[
            ChatMessageEvent(
                message_id="m1",
                sender="me",
                message_type="text",
                text="DeskCanvasdefault readback WX 1906",
                bounds=(100, 200, 360, 240),
                confidence=0.91,
                source="ocr_crop",
            )
        ],
    )

    assert result.has_text("DeskCanvas default readback WX 1906")


class FakeOCR:
    def recognize_image(self, image):
        return [
            {"text": "DeskCanvas send probe", "bounds": (610, 120, 820, 160), "confidence": 0.93},
            {"text": "对方回复", "bounds": (80, 220, 260, 260), "confidence": 0.88},
        ]


class EmptyOCR:
    def recognize_image(self, image):
        return []


def test_readback_worker_emits_self_and_peer_text_events(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")
    worker = ChatReadbackWorker(ocr_service=FakeOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="weixin.exe",
        window_id="hwnd-1",
        stream_bounds=(300, 80, 1000, 620),
        crop_origin=(300, 80),
    )

    assert result.has_text("DeskCanvas send probe")
    assert any(event.sender == "peer" and event.text == "对方回复" for event in result.events)


def test_readback_worker_marks_large_image_like_region_for_vlm(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")
    worker = ChatReadbackWorker(ocr_service=EmptyOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="weixin.exe",
        window_id="hwnd-1",
        stream_bounds=(300, 80, 1000, 620),
        crop_origin=(300, 80),
        local_visual_candidates=[
            {"candidate_id": "img_1", "bounds": (360, 180, 560, 320), "semantic_role": "image"},
        ],
    )

    assert result.events[0].message_type == "image"
    assert result.events[0].attachments[0].needs_vlm is True


def test_readback_worker_marks_expected_self_text_as_me_even_when_left_aligned(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")

    class LeftAlignedOCR:
        def recognize_image(self, image):
            return [
                {
                    "text": "DeskCanvas readback probe Feishu 1735",
                    "bounds": (64, 170, 310, 188),
                    "confidence": 0.98,
                }
            ]

    worker = ChatReadbackWorker(ocr_service=LeftAlignedOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="feishu.exe",
        window_id="hwnd-1",
        stream_bounds=(0, 0, 700, 220),
        crop_origin=(0, 0),
        expected_self_texts=["DeskCanvas readback probe Feishu 1735"],
    )

    assert result.events[0].sender == "me"


def test_readback_worker_expected_self_text_ignores_ocr_space_loss(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")

    class SpacelessOCR:
        def recognize_image(self, image):
            return [
                {
                    "text": "DeskCanvasdefault readback WX 1906",
                    "bounds": (64, 170, 310, 188),
                    "confidence": 0.98,
                }
            ]

    worker = ChatReadbackWorker(ocr_service=SpacelessOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="weixin.exe",
        window_id="hwnd-1",
        stream_bounds=(0, 0, 700, 220),
        crop_origin=(0, 0),
        expected_self_texts=["DeskCanvas default readback WX 1906"],
    )

    assert result.events[0].sender == "me"


def test_readback_worker_records_bubble_color_style_for_ocr_text():
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 200, 70), fill=(184, 232, 178))

    class BubbleOCR:
        def recognize_image(self, image):
            return [
                {
                    "text": "DeskCanvas probe",
                    "bounds": (70, 42, 168, 58),
                    "confidence": 0.97,
                }
            ]

    worker = ChatReadbackWorker(ocr_service=BubbleOCR())

    result = worker.read_crop(
        image=image,
        app_process="feishu.exe",
        window_id="hwnd-1",
        stream_bounds=(0, 0, 240, 120),
        crop_origin=(0, 0),
    )

    assert result.events[0].style["bubble_rgb"] == [184, 232, 178]
    assert result.events[0].style["style_source"] == "crop_sample"


def test_readback_worker_can_use_bubble_style_hint_before_geometry_sender():
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 150, 70), fill=(184, 232, 178))

    class LeftBubbleOCR:
        def recognize_image(self, image):
            return [
                {
                    "text": "left aligned self message",
                    "bounds": (45, 42, 138, 58),
                    "confidence": 0.96,
                }
            ]

    worker = ChatReadbackWorker(ocr_service=LeftBubbleOCR())

    result = worker.read_crop(
        image=image,
        app_process="feishu.exe",
        window_id="hwnd-1",
        stream_bounds=(0, 0, 240, 120),
        crop_origin=(0, 0),
        sender_style_hints={"me": [184, 232, 178], "peer": [245, 245, 245]},
    )

    assert result.events[0].sender == "me"
    assert result.events[0].style["sender_style_match"] == "me"


def test_readback_worker_filters_chat_metadata_and_low_confidence_symbol_noise(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")

    class NoisyQQGroupOCR:
        def recognize_image(self, image):
            return [
                {"text": "22:58", "bounds": (239, 1, 276, 16), "confidence": 0.98},
                {"text": "LV1小刀", "bounds": (378, 30, 433, 48), "confidence": 0.96},
                {"text": "消息测试123", "bounds": (339, 61, 423, 78), "confidence": 0.99},
                {"text": "LV1群主", "bounds": (94, 115, 143, 133), "confidence": 0.97},
                {"text": "消息测试456", "bounds": (90, 146, 175, 163), "confidence": 0.99},
                {"text": "♀", "bounds": (77, 359, 273, 386), "confidence": 0.50},
                {"text": "z", "bounds": (81, 120, 93, 130), "confidence": 0.55},
                {"text": "①", "bounds": (448, 360, 472, 384), "confidence": 0.04},
                {"text": "好", "bounds": (90, 410, 112, 430), "confidence": 0.97},
            ]

    worker = ChatReadbackWorker(ocr_service=NoisyQQGroupOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="qq.exe",
        window_id="hwnd-qq-group",
        stream_bounds=(0, 0, 782, 530),
        crop_origin=(0, 0),
    )

    assert [event.text for event in result.events] == ["消息测试123", "消息测试456", "好"]


def test_readback_worker_warns_when_crop_only_contains_loading_indicator(tmp_path):
    image_path = tmp_path / "stream.png"
    image_path.write_bytes(b"fake")

    class LoadingOCR:
        def recognize_image(self, image):
            return [{"text": "正在加载...", "bounds": (238, 32, 301, 49), "confidence": 0.88}]

    worker = ChatReadbackWorker(ocr_service=LoadingOCR())

    result = worker.read_crop(
        image_path=image_path,
        app_process="feishu.exe",
        window_id="hwnd-feishu",
        stream_bounds=(0, 0, 520, 600),
        crop_origin=(0, 0),
    )

    assert result.status == "warn"
    assert "message_stream_loading" in result.warnings
