from src.chat.readback_style import build_sender_style_baseline, extract_style_samples


def test_extract_style_samples_keeps_sender_bubble_rgb_events():
    result = {
        "app_process": "weixin.exe",
        "events": [
            {
                "sender": "me",
                "message_type": "text",
                "text": "DeskCanvas probe",
                "style": {"bubble_rgb": [146, 224, 148]},
            },
            {
                "sender": "unknown",
                "message_type": "text",
                "text": "19:14",
                "style": {"bubble_rgb": [245, 245, 246]},
            },
            {
                "sender": "peer",
                "message_type": "image",
                "text": None,
                "style": {"bubble_rgb": [255, 255, 255]},
            },
        ],
    }

    samples = extract_style_samples([result])

    assert len(samples) == 1
    assert samples[0]["app_process"] == "weixin.exe"
    assert samples[0]["sender"] == "me"
    assert samples[0]["bubble_rgb"] == [146, 224, 148]


def test_build_sender_style_baseline_uses_median_rgb_and_min_samples():
    samples = [
        {"app_process": "qq.exe", "sender": "me", "bubble_rgb": [35, 167, 255]},
        {"app_process": "qq.exe", "sender": "me", "bubble_rgb": [37, 165, 253]},
        {"app_process": "qq.exe", "sender": "peer", "bubble_rgb": [240, 240, 240]},
        {"app_process": "weixin.exe", "sender": "me", "bubble_rgb": [146, 224, 148]},
    ]

    baseline = build_sender_style_baseline(samples, min_samples=2)

    assert baseline["apps"]["qq.exe"]["sender_style_hints"]["me"] == [36, 166, 254]
    assert baseline["apps"]["qq.exe"]["sample_counts"] == {"me": 2, "peer": 1}
    assert baseline["apps"]["qq.exe"]["usable"] is True
    assert baseline["apps"]["weixin.exe"]["usable"] is False


def test_build_sender_style_baseline_marks_close_me_peer_colors_ambiguous():
    samples = [
        {"app_process": "feishu.exe", "sender": "me", "bubble_rgb": [195, 212, 239]},
        {"app_process": "feishu.exe", "sender": "peer", "bubble_rgb": [197, 214, 241]},
    ]

    baseline = build_sender_style_baseline(samples, min_samples=1, min_sender_distance=40)

    assert baseline["apps"]["feishu.exe"]["usable"] is False
    assert baseline["apps"]["feishu.exe"]["warnings"] == ["sender_style_ambiguous"]
