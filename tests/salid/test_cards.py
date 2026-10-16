from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from salid import cards
from salid.pdf_report import FontSet


def test_raqm_is_available_for_thai():
    assert cards.thai_ready(), (
        "Pillow has no raqm: install libfribidi (brew install fribidi / apt libfribidi0) "
        "and run tests with DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib; report it, do not "
        "work around it"
    )


def test_bundled_kanit_is_used():
    f = cards.card_fonts()
    assert f.bold.endswith("Kanit-Bold.ttf") and f.regular.endswith("Kanit-Regular.ttf")


def test_falls_back_to_find_fonts_when_bundle_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cards, "FONTS_DIR", tmp_path)
    cards.card_fonts.cache_clear()
    monkeypatch.setattr(cards, "find_fonts", lambda configured="": FontSet("/r.ttf", None, True))
    try:
        assert cards.card_fonts() == cards.CardFonts(bold="/r.ttf", regular="/r.ttf")
    finally:
        cards.card_fonts.cache_clear()


def test_fonts_use_raqm_layout():
    assert cards.font(cards.card_fonts().bold, 40).layout_engine == ImageFont.Layout.RAQM


def test_thai_is_shaped_not_drawn_glyph_by_glyph():
    # RAQM applies the font's mark positioning (stacked tone marks); BASIC does not
    def render(f):
        img = Image.new("L", (900, 140), 0)
        ImageDraw.Draw(img).text((10, 20), "ผู้ใหญ่ น้ำแข็ง ปั้น", font=f, fill=255)
        return img

    bold = cards.card_fonts().bold
    basic = ImageFont.truetype(bold, 60, layout_engine=ImageFont.Layout.BASIC)
    assert ImageChops.difference(render(cards.font(bold, 60)), render(basic)).getbbox() is not None


def test_drawable_drops_characters_without_glyphs():
    assert cards.drawable("ลด 20% 🔥", cards.card_fonts().bold) == "ลด 20%"
    assert cards.drawable("Latte เย็น 59.-", cards.card_fonts().bold) == "Latte เย็น 59.-"


# ------------------------------------------------------------------ primitives
BOLD = cards.card_fonts().bold


def test_never_breaks_inside_a_thai_syllable():
    assert not cards.break_ok("น", "้")  # tone mark (Mn)
    assert not cards.break_ok("้", "ำ")  # SARA AM U+0E33
    assert not cards.break_ok("ก", "า")  # following vowel U+0E30-0E3A
    assert not cards.break_ok("เ", "ก")  # after a leading vowel U+0E40-0E44
    assert not cards.break_ok("ก", "ๆ")  # U+0E45-0E4E
    assert cards.break_ok("า", "ก") and cards.break_ok("a", "b")


def test_fit_prefers_shrinking_to_splitting():
    f = cards.fit_text("โปรพิเศษ", BOLD, 952, 2, (96, 48))
    assert f.lines == ["โปรพิเศษ"] and f.size == 96


def test_fit_breaks_at_spaces_and_keeps_every_line_inside():
    f = cards.fit_text("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", BOLD, 600, 2, (96, 48))
    assert 1 <= len(f.lines) <= 2 and f.width <= 600
    assert all(cards.font(BOLD, f.size).getlength(line) <= 600 for line in f.lines)


def test_two_lines_are_balanced_not_greedy():
    f = cards.fit_text("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20%", BOLD, 952, 2, (80, 40))
    assert f.size == 80 and f.lines == ["ลาเต้เย็น หอมกรุ่น", "ลดพิเศษ 20%"]


def test_fit_splits_a_long_spaceless_thai_run_only_at_syllable_boundaries():
    text = "ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือนนี้เท่านั้น"  # 40+ chars, no spaces
    f = cards.fit_text(text, BOLD, 500, 2, (96, 48))
    assert "".join(f.lines) == text and len(f.lines) == 2 and f.width <= 500
    for a, b in zip(f.lines, f.lines[1:], strict=False):
        assert cards.break_ok(a[-1], b[0])


def test_line_height_leaves_room_for_marks():
    # Kanit's ascent + descent (1.5 em) already spans stacked Thai marks: no extra leading
    fnt = cards.font(BOLD, 80)
    asc, desc = fnt.getmetrics()
    assert cards.line_height(fnt) == asc + desc


def test_shadow_keeps_white_text_readable_on_light_backgrounds():
    f = cards.fit_text("ลดพิเศษ 20%", BOLD, 952, 1, (80, 40))
    plain, shaded = (Image.new("RGB", (1080, 300), "white") for _ in range(2))
    cards.draw_lines(plain, 64, 50, f, BOLD, fill="white")
    cards.draw_lines(shaded, 64, 50, f, BOLD, fill="white", shadow=6)
    assert plain.convert("L").getextrema()[0] == 255  # white on white: invisible
    assert shaded.convert("L").getextrema()[0] < 160  # the shadow outlines the glyphs


def test_drawn_block_stays_in_its_box():
    img = Image.new("RGB", (1080, 600), "white")
    f = cards.fit_text("ผู้ใหญ่ญี่ปุ่น น้ำแข็งปั้น", BOLD, 952, 2, (96, 48))
    x0, y0, x1, y1 = cards.draw_lines(img, 64, 100, f, BOLD, fill="black")
    assert x0 >= 64 and x1 <= 64 + 952 and y0 == 100 and y1 == 100 + f.height
    ink = ImageOps.invert(img.convert("L")).getbbox()
    assert ink[1] >= 100 and ink[3] <= y1


def test_scrim_darkens_towards_alpha1():
    img = Image.new("RGB", (100, 200), "white")
    cards.scrim(img, 0, 200, 0, 200)
    assert img.getpixel((50, 2))[0] > img.getpixel((50, 197))[0]


def test_badge_and_pill_geometry():
    b = cards.measure_badge("59.-", BOLD, (72, 40), 952)
    assert b.w == b.fitted.width + 64 and b.h == b.fitted.height + 32
    img = Image.new("RGB", (1080, 1080))
    assert cards.draw_badge(img, b, 100, 200, BOLD, fill="#E53935", text_fill="#FFFFFF") == (
        100,
        200,
        100 + b.w,
        200 + b.h,
    )


def test_cover_gives_the_exact_size_from_any_aspect():
    for src in ((512, 512), (2000, 800), (1536, 2752)):
        assert cards.cover(Image.new("RGB", src), (1080, 1350)).size == (1080, 1350)


def test_jpeg_under_the_cap():
    noisy = Image.effect_noise((1080, 1920), 120).convert("RGB")
    data = cards.to_jpeg(noisy)
    assert data[:2] == b"\xff\xd8" and len(data) < 1_500_000


def test_contrast_ratio():
    assert round(cards.contrast("#FFFFFF", "#000000"), 1) == 21.0
    assert round(cards.contrast("#E53935", "#FFFFFF"), 2) == 4.23
