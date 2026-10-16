"""Slide decks: the outline the chat agent sends (`DeckSpec`) and its rendering to `.pptx`.

The agent's `create_deck` tool (salid.qa) validates the outline with these models, fetches chart
numbers from the ledger itself (the model never types numbers into a chart) and calls the
renderer. Everything here is pure: no network, no model calls, nothing written to disk.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

MAX_TITLE_CHARS = 80
MAX_SUBTITLE_CHARS = 120
MAX_SLIDES = 12
MAX_BULLETS = 6
MAX_BULLET_CHARS = 140
MAX_NOTES_CHARS = 1000
MAX_TOP = 10
MAX_DAY_RANGE = 62  # days one day-by-day chart may span
MAX_DECK_BYTES = 2_000_000
MAX_SLUG_CHARS = 40

Bullet = Annotated[str, StringConstraints(max_length=MAX_BULLET_CHARS)]


class ChartQuery(BaseModel):
    """Which ledger numbers a chart slide shows; the numbers themselves are fetched in code."""

    group_by: Literal["category", "vendor", "submitter", "day"]
    start_date: str
    end_date: str
    style: Literal["bar", "pie", "line"] = "bar"
    top: int = Field(8, ge=1, le=MAX_TOP)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso(cls, v: str) -> str:
        try:
            return date.fromisoformat(v).isoformat()
        except ValueError as e:
            raise ValueError("must be an ISO date YYYY-MM-DD") from e

    @model_validator(mode="after")
    def _range(self) -> ChartQuery:
        a, b = date.fromisoformat(self.start_date), date.fromisoformat(self.end_date)
        if b < a:
            raise ValueError("end_date is before start_date")
        if self.group_by == "day" and (b - a).days + 1 > MAX_DAY_RANGE:
            raise ValueError(f"a day-by-day chart may span at most {MAX_DAY_RANGE} days")
        return self


class Slide(BaseModel):
    kind: Literal["section", "bullets", "two_column", "chart"]
    title: str = Field(max_length=MAX_TITLE_CHARS)
    bullets: list[Bullet] = Field(default_factory=list, max_length=MAX_BULLETS)
    right: list[Bullet] = Field(default_factory=list, max_length=MAX_BULLETS)
    chart: ChartQuery | None = None
    notes: str = Field("", max_length=MAX_NOTES_CHARS)

    @model_validator(mode="after")
    def _kind_fields(self) -> Slide:
        if self.kind == "chart" and self.chart is None:
            raise ValueError("chart is required for a chart slide")
        if self.kind in ("bullets", "two_column") and not self.bullets:
            raise ValueError("bullets needs at least one item")
        return self


class DeckSpec(BaseModel):
    title: str = Field(max_length=MAX_TITLE_CHARS)
    subtitle: str = Field("", max_length=MAX_SUBTITLE_CHARS)
    slides: list[Slide] = Field(min_length=1, max_length=MAX_SLIDES)


def deck_filename(title: str, today: date) -> str:
    """An ASCII slug of the title plus the date: "spending-oct-2026-20261016.pptx", or
    "salid-deck-20261016.pptx" when the title has no ASCII words (e.g. all Thai)."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = ""
    for word in re.findall(r"[a-z0-9]+", ascii_title.lower()):
        if len(slug) + len(word) + (1 if slug else 0) > MAX_SLUG_CHARS:
            break
        slug = f"{slug}-{word}" if slug else word
    return f"{slug or 'salid-deck'}-{today:%Y%m%d}.pptx"
