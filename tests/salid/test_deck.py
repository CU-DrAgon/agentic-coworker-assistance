"""Slide decks: outline models, validation, filenames, rendering and charts (salid.deck).
Offline: rendered files are read back with python-pptx."""

from datetime import date

import pytest
from pydantic import ValidationError

from salid.deck import ChartQuery, DeckSpec, deck_filename


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
