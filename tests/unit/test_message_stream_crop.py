from src.chat.message_stream_crop import (
    classify_message_side,
    input_crop_box,
    message_stream_crop_box,
    sent_message_crop_box,
)


def test_input_crop_expands_above_composer_toolbar():
    box = input_crop_box(window_rect=(100, 50, 1100, 780), input_bounds=(320, 610, 990, 720))

    assert box == (420, 620, 1090, 770)


def test_sent_message_crop_stays_above_input_and_right_side():
    box = sent_message_crop_box(window_rect=(100, 50, 1100, 780), input_bounds=(320, 610, 990, 720))

    assert box[0] >= 420
    assert box[1] < 660
    assert box[2] <= 1090
    assert box[3] <= 650


def test_message_stream_crop_uses_area_above_composer():
    box = message_stream_crop_box(window_rect=(100, 50, 1100, 780), input_bounds=(320, 610, 990, 720))

    assert box == (400, 130, 1100, 660)


def test_message_side_uses_bubble_center():
    assert classify_message_side(bounds=(720, 420, 980, 470), stream_bounds=(300, 80, 1000, 620)) == "me"
    assert classify_message_side(bounds=(330, 420, 590, 470), stream_bounds=(300, 80, 1000, 620)) == "peer"


def test_message_side_marks_middle_bubble_unknown():
    assert classify_message_side(bounds=(560, 420, 730, 470), stream_bounds=(300, 80, 1000, 620)) == "unknown"
