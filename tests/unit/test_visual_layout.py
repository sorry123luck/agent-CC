from __future__ import annotations

from PIL import Image, ImageDraw

from src.perception.visual_layout import VisualLayoutSegmenter


def test_detects_vertical_split_on_two_panel_image():
    image = Image.new("RGB", (300, 180), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 79, 179), fill=(225, 228, 235))
    draw.rectangle((80, 0, 299, 179), fill=(255, 255, 255))
    draw.line((80, 0, 80, 179), fill=(205, 208, 215), width=2)

    regions = VisualLayoutSegmenter().segment(image)

    assert any(
        "vertical_separator" in region.evidence
        and region.bounds[0] <= 82
        and region.bounds[2] >= 79
        for region in regions
    )


def test_detects_bottom_bordered_input_rectangle():
    image = Image.new("RGB", (360, 240), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((92, 180, 342, 226), radius=8, outline=(210, 214, 222), width=2, fill=(255, 255, 255))

    regions = VisualLayoutSegmenter().segment(image)

    input_regions = [
        region
        for region in regions
        if region.role_hint == "input_like_region"
        and "bordered_rectangle" in region.evidence
    ]
    assert input_regions
    left, top, right, bottom = input_regions[0].bounds
    assert left <= 96
    assert top <= 184
    assert right >= 338
    assert bottom >= 222


def test_visual_regions_are_valid_and_semantic_only():
    image = Image.new("RGB", (180, 120), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 40, 119), fill=(220, 220, 225))
    draw.rectangle((48, 82, 170, 112), outline=(190, 190, 195), width=1)

    regions = VisualLayoutSegmenter().segment(image)

    assert regions
    for region in regions:
        left, top, right, bottom = region.bounds
        assert 0 <= left < right <= 180
        assert 0 <= top < bottom <= 120
        assert region.actionability == "semantic_only"
        assert region.source == "visual_layout"


def test_detects_bottom_input_strip_on_medium_height_image():
    image = Image.new("RGB", (500, 600), (246, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 120, 599), fill=(224, 228, 235))
    draw.line((130, 505, 490, 505), fill=(210, 214, 222), width=2)
    draw.rounded_rectangle((145, 520, 480, 575), radius=10, outline=(204, 208, 216), width=2, fill=(234, 236, 240))
    draw.text((158, 540), "message", fill=(60, 60, 60))
    draw.rectangle((430, 535, 470, 564), fill=(20, 180, 100))

    regions = VisualLayoutSegmenter().segment(image)

    strips = [region for region in regions if region.role_hint == "bottom_input_strip"]
    assert strips
    left, top, right, bottom = strips[0].bounds
    assert top >= 470
    assert right - left >= 200
    assert bottom <= 600


def test_detects_app_shell_and_card_grid_regions():
    image = Image.new("RGB", (830, 718), (24, 22, 22))
    draw = ImageDraw.Draw(image)
    # Window chrome / title band.
    draw.rectangle((0, 0, 829, 38), fill=(55, 45, 45))
    # Left navigation rail.
    draw.rectangle((0, 38, 79, 717), fill=(35, 31, 32))
    # Header / category tabs.
    draw.rectangle((80, 38, 829, 144), fill=(24, 22, 22))
    for x, label in [(112, "tab"), (220, "tab"), (340, "tab"), (470, "tab"), (610, "tab")]:
        draw.text((x, 112), label, fill=(235, 235, 235))
    # Card grid with repeated bordered cards.
    for row in range(4):
        for col in range(3):
            x1 = 96 + col * 242
            y1 = 160 + row * 114
            x2 = x1 + 232
            y2 = y1 + 104
            fill = (30, 27, 27) if not (row == 0 and col == 1) else (78, 66, 66)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=10, outline=(66, 58, 58), width=2, fill=fill)
            draw.text((x1 + 16, y1 + 18), "node", fill=(245, 245, 245))
            draw.text((x1 + 16, y1 + 74), "98 ms", fill=(70, 220, 100))
    # Floating action button.
    draw.rounded_rectangle((690, 645, 812, 704), radius=18, fill=(234, 204, 207))

    regions = VisualLayoutSegmenter().segment(image)
    by_role = {}
    for region in regions:
        by_role.setdefault(region.role_hint, []).append(region)

    assert "side_panel_region" in by_role
    assert "toolbar_like_region" in by_role
    assert "card_grid_region" in by_role
    assert "floating_region" in by_role

    grid = by_role["card_grid_region"][0]
    assert grid.bounds[0] <= 105
    assert grid.bounds[1] <= 165
    assert grid.bounds[2] >= 805
    assert grid.bounds[3] >= 600


def test_detects_three_column_chat_shell_regions():
    image = Image.new("RGB", (1002, 731), (246, 248, 251))
    draw = ImageDraw.Draw(image)
    # Window chrome.
    draw.rectangle((0, 0, 1001, 31), fill=(218, 219, 222))
    # Left navigation rail and conversation list.
    draw.rectangle((0, 32, 72, 730), fill=(228, 231, 236))
    draw.rectangle((73, 32, 300, 730), fill=(238, 241, 246))
    draw.line((72, 32, 72, 730), fill=(178, 181, 186), width=2)
    draw.line((300, 32, 300, 730), fill=(198, 201, 206), width=2)
    # Main header and message area.
    draw.rectangle((301, 32, 1001, 86), fill=(248, 249, 251))
    draw.line((301, 87, 1001, 87), fill=(204, 207, 212), width=2)
    draw.rectangle((301, 88, 1001, 609), fill=(248, 249, 251))
    # Chat bubbles make the main region textured but not a card grid.
    draw.rounded_rectangle((410, 112, 780, 345), radius=8, fill=(238, 239, 242))
    draw.text((420, 124), "long message content", fill=(32, 32, 32))
    draw.rounded_rectangle((408, 406, 778, 584), radius=8, fill=(238, 239, 242))
    draw.text((420, 420), "reply content", fill=(32, 32, 32))
    # Conversation rows in the middle list.
    for row in range(8):
        top = 92 + row * 58
        draw.rectangle((74, top, 299, top + 56), fill=(238, 241, 246))
        draw.text((116, top + 14), f"chat {row}", fill=(48, 48, 48))
    # Bottom composer on the right only.
    draw.line((301, 610, 1001, 610), fill=(218, 221, 226), width=2)
    draw.rounded_rectangle((348, 636, 988, 713), radius=10, outline=(208, 211, 216), width=2, fill=(255, 255, 255))
    draw.text((364, 675), "message", fill=(42, 42, 42))

    regions = VisualLayoutSegmenter().segment(image)
    by_role = {}
    for region in regions:
        by_role.setdefault(region.role_hint, []).append(region)

    assert "side_panel_region" in by_role
    assert "list_panel_region" in by_role
    assert "content_like_region" in by_role
    assert "bottom_input_strip" in by_role

    list_panel = by_role["list_panel_region"][0]
    assert list_panel.bounds[0] <= 76
    assert 285 <= list_panel.bounds[2] <= 320

    content = by_role["content_like_region"][0]
    assert content.bounds[0] >= 295
    assert content.bounds[1] <= 95
    assert content.bounds[3] <= 620

    composer = by_role["bottom_input_strip"][0]
    assert composer.bounds[0] >= 300
    assert composer.bounds[1] >= 600


def test_three_column_shell_prefers_long_subtle_boundaries_over_local_edges():
    image = Image.new("RGB", (1002, 731), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1001, 31), fill=(218, 219, 222))
    draw.rectangle((0, 32, 59, 730), fill=(230, 232, 236))
    draw.rectangle((60, 32, 300, 730), fill=(238, 240, 244))
    draw.rectangle((301, 32, 1001, 730), fill=(250, 250, 250))

    # Local strong vertical edges inside the list should not become the shell boundary.
    for top in range(48, 620, 72):
        draw.rectangle((72, top, 108, top + 36), fill=(80, 120, 180))
    # Subtle but long vertical shell separators.
    draw.line((59, 32, 59, 730), fill=(212, 214, 218), width=1)
    draw.line((300, 32, 300, 730), fill=(210, 212, 216), width=1)

    # Subtle long header/content split plus stronger local text-like edges below it.
    draw.rectangle((301, 32, 1001, 79), fill=(250, 250, 250))
    draw.line((301, 80, 1001, 80), fill=(236, 236, 236), width=1)
    for left in range(390, 760, 44):
        draw.rectangle((left, 87, left + 22, 98), fill=(90, 90, 90))

    draw.line((301, 610, 1001, 610), fill=(218, 221, 226), width=2)
    draw.rounded_rectangle((348, 636, 988, 713), radius=10, outline=(208, 211, 216), width=2, fill=(255, 255, 255))

    regions = VisualLayoutSegmenter().segment(image)
    by_role = {}
    for region in regions:
        by_role.setdefault(region.role_hint, []).append(region)

    side_panel = by_role["side_panel_region"][0]
    list_panel = by_role["list_panel_region"][0]
    content = by_role["content_like_region"][0]

    assert 56 <= side_panel.bounds[2] <= 64
    assert 56 <= list_panel.bounds[0] <= 64
    assert 295 <= list_panel.bounds[2] <= 305
    assert 78 <= content.bounds[1] <= 83
