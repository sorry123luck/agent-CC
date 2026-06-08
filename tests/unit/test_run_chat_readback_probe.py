import importlib.util
import json
from pathlib import Path

from src.chat.readback_models import ChatMessageEvent, ChatReadbackResult


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_chat_readback_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_chat_readback_probe", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_readback_report(tmp_path: Path):
    module = _load_module()
    result = ChatReadbackResult(
        window_id="hwnd-1",
        app_process="weixin.exe",
        events=[
            ChatMessageEvent(
                message_id="m1",
                sender="me",
                message_type="text",
                text="OpenClaw send probe",
                bounds=(700, 400, 920, 450),
                confidence=0.92,
                source="ocr_crop",
            )
        ],
    )

    module.write_readback_report(tmp_path, [result])

    data = json.loads((tmp_path / "chat_readback_report.json").read_text(encoding="utf-8"))
    assert data["overall"] == "pass"
    assert data["results"][0]["events"][0]["text"] == "OpenClaw send probe"
    assert (tmp_path / "chat_readback_report.md").exists()


def test_find_sent_message_crops_from_send_probe(tmp_path: Path):
    module = _load_module()
    crop_dir = tmp_path / "message_crops"
    crop_dir.mkdir()
    crop = crop_dir / "wechat_sent_message.png"
    crop.write_bytes(b"fake")
    actions = [{"sample": "wechat", "process_name": "weixin.exe", "hwnd": 12, "sent_message_crop": str(crop)}]
    (tmp_path / "send_probe_actions.json").write_text(json.dumps(actions), encoding="utf-8")

    crops = module.find_sent_message_crops(tmp_path)

    assert crops == [
        {
            "sample": "wechat",
            "process_name": "weixin.exe",
            "hwnd": 12,
            "text": "",
            "sent_message_crop": str(crop),
        }
    ]


def test_find_sent_message_crops_preserves_expected_text(tmp_path: Path):
    module = _load_module()
    crop_dir = tmp_path / "message_crops"
    crop_dir.mkdir()
    crop = crop_dir / "feishu_sent_message.png"
    crop.write_bytes(b"fake")
    actions = [
        {
            "sample": "feishu",
            "process_name": "feishu.exe",
            "hwnd": 44,
            "text": "OpenClaw readback probe Feishu 1735",
            "sent_message_crop": str(crop),
        }
    ]
    (tmp_path / "send_probe_actions.json").write_text(json.dumps(actions), encoding="utf-8")

    crops = module.find_sent_message_crops(tmp_path)

    assert crops[0]["text"] == "OpenClaw readback probe Feishu 1735"
