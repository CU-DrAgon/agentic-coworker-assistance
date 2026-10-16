from PIL import Image, ImageChops, ImageDraw, ImageFont

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
