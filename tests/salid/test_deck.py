"""Slide decks: outline models, validation, filenames, rendering and charts (salid.deck).
Offline: rendered files are read back with python-pptx."""

import io
from datetime import date

import pytest
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from pydantic import ValidationError

from salid.deck import (
    LATIN_FONT,
    THAI_FONT,
    ChartQuery,
    DeckSpec,
    deck_filename,
    render_deck,
)


def bullets(n=2, text="ข้อ"):
    return {"kind": "bullets", "title": "หัวข้อ", "bullets": [f"{text} {i}" for i in range(n)]}


# ------------------------------------------------------------------ outline models
def test_valid_outline_parses():
    spec = DeckSpec.model_validate({"title": "สรุปเดือนตุลาคม", "slides": [bullets()]})
    assert spec.slides[0].kind == "bullets" and spec.subtitle == ""


@pytest.mark.parametrize(
    "bad, field",
    [
        ({"title": "x" * 81, "slides": [bullets()]}, "title"),
        ({"title": "t", "slides": []}, "slides"),
        ({"title": "t", "slides": [bullets()] * 13}, "slides"),
        ({"title": "t", "slides": [bullets(7)]}, "bullets"),
        (
            {"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": ["x" * 141]}]},
            "bullets",
        ),
        ({"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": []}]}, "bullets"),
        ({"title": "t", "slides": [{"kind": "chart", "title": "t"}]}, "chart"),
        ({"title": "t", "slides": [{"kind": "video", "title": "t"}]}, "kind"),
    ],
)
def test_invalid_outlines_name_the_field(bad, field):
    with pytest.raises(ValidationError) as e:
        DeckSpec.model_validate(bad)
    assert field in str(e.value)


@pytest.mark.parametrize(
    "q, ok",
    [
        ({"group_by": "category", "start_date": "2026-10-01", "end_date": "2026-10-31"}, True),
        (
            {"group_by": "day", "start_date": "2026-09-01", "end_date": "2026-10-31"},
            True,
        ),  # 61 days
        ({"group_by": "day", "start_date": "2026-08-01", "end_date": "2026-10-31"}, False),  # 92
        ({"group_by": "category", "start_date": "2026-10-31", "end_date": "2026-10-01"}, False),
        ({"group_by": "category", "start_date": "31/10/2026", "end_date": "2026-10-31"}, False),
        (
            {
                "group_by": "category",
                "start_date": "2026-10-01",
                "end_date": "2026-10-31",
                "top": 11,
            },
            False,
        ),
    ],
)
def test_chart_query_dates_and_limits(q, ok):
    if ok:
        ChartQuery.model_validate(q)
    else:
        with pytest.raises(ValidationError):
            ChartQuery.model_validate(q)


def test_deck_filename_slug_and_fallback():
    d = date(2026, 10, 16)
    assert (
        deck_filename("Spending report: Oct 2026!", d) == "spending-report-oct-2026-20261016.pptx"
    )
    assert deck_filename("สรุปค่าใช้จ่าย", d) == "salid-deck-20261016.pptx"
    assert deck_filename("🎉 Q4 / ../plan", d) == "q4-plan-20261016.pptx"
    assert len(deck_filename("word " * 40, d)) <= 64


# ------------------------------------------------------------------ rendering: text slides
def render(spec_dict, lang="th", charts=None):
    data = render_deck(
        DeckSpec.model_validate(spec_dict), charts or {}, lang=lang, today=date(2026, 10, 16)
    )
    return Presentation(io.BytesIO(data))


def texts(slide):
    return [s.text_frame.text for s in slide.shapes if s.has_text_frame]


def test_title_slide_then_one_slide_per_outline_slide():
    prs = render(
        {
            "title": "สรุปประชุม",
            "subtitle": "ทีมขาย",
            "slides": [
                {"kind": "section", "title": "ภาพรวม"},
                {
                    "kind": "bullets",
                    "title": "สิ่งที่ตกลง",
                    "bullets": ["เลื่อนเปิดร้านเป็นวันศุกร์"],
                    "notes": "รายละเอียด",
                },
                {"kind": "two_column", "title": "ก่อน / หลัง", "bullets": ["เดิม"], "right": ["ใหม่"]},
            ],
        }
    )
    assert len(prs.slides) == 4 and prs.slide_width == Inches(13.333)
    cover = " ".join(texts(prs.slides[0]))
    assert "สรุปประชุม" in cover and "ทีมขาย" in cover and "16 ตุลาคม 2569" in cover
    assert "สลิด" in cover
    assert "เลื่อนเปิดร้านเป็นวันศุกร์" in " ".join(texts(prs.slides[2]))
    assert prs.slides[2].notes_slide.notes_text_frame.text == "รายละเอียด"
    assert {"เดิม", "ใหม่"} <= set(" ".join(texts(prs.slides[3])).split())


def test_every_run_sets_the_thai_font_after_latin():
    prs = render({"title": "ทดสอบ", "slides": [bullets()]})
    runs = [
        r
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
        for p in shape.text_frame.paragraphs
        for r in p.runs
    ]
    assert runs
    for r in runs:
        rpr = r._r.find(qn("a:rPr"))
        tags = [c.tag for c in rpr]
        assert rpr.find(qn("a:latin")).get("typeface") == LATIN_FONT
        assert rpr.find(qn("a:cs")).get("typeface") == THAI_FONT
        assert tags.index(qn("a:cs")) == tags.index(qn("a:latin")) + 1


def _bullet_size(prs):
    """Largest run size on the first content slide, ignoring its title ("t")."""
    return max(
        r.font.size
        for s in prs.slides[1].shapes
        if s.has_text_frame
        for p in s.text_frame.paragraphs
        if p.text != "t"
        for r in p.runs
    )


def test_long_text_gets_a_smaller_font():
    short = render({"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": ["สั้น"]}]})
    long_ = render(
        {
            "title": "t",
            "slides": [{"kind": "bullets", "title": "t", "bullets": ["ยาวมาก" * 23] * 6}],
        }
    )
    assert _bullet_size(long_) < _bullet_size(short) and _bullet_size(long_) >= Pt(16)


def test_control_characters_and_newlines_are_cleaned():
    prs = render(
        {
            "title": "a\x00b",
            "slides": [{"kind": "bullets", "title": "t", "bullets": ["line1\nline2\x0b end"]}],
        }
    )
    body = " ".join(texts(prs.slides[1]))
    assert "line1 line2 end" in body and "\x0b" not in body
    assert "ab" in " ".join(texts(prs.slides[0]))


def test_english_fixed_strings():
    cover = " ".join(texts(render({"title": "Q4", "slides": [bullets()]}, lang="en").slides[0]))
    assert "Made with Salid" in cover and "16 October 2026" in cover


def test_page_numbers_on_content_slides():
    prs = render({"title": "t", "slides": [bullets(), bullets()]})
    assert "2" in texts(prs.slides[1]) and "3" in texts(prs.slides[2])
