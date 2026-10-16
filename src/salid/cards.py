"""Image cards with Thai text drawn correctly: the shared renderer behind promo kits (and later
visual features such as chart cards or digest images).

Thai needs text shaping: tone marks and upper/lower vowels are positioned by the font's GSUB/GPOS
tables. Pillow does that only with `ImageFont.Layout.RAQM`; its default BASIC layout draws each
code point on its own and leaves dotted circles or stacked marks. Pillow's wheels bundle raqm and
HarfBuzz but load the system libfribidi at runtime (Docker: `libfribidi0`), so `thai_ready()`
gates every feature that draws text. Fonts: the bundled Kanit (SIL OFL 1.1, `fonts/`), else the
PDF font search (`salid.pdf_report.find_fonts`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont, features

from .pdf_report import find_fonts

FONTS_DIR = Path(__file__).parent / "fonts"


@dataclass(frozen=True)
class CardFonts:
    bold: str
    regular: str


@functools.lru_cache(maxsize=1)
def card_fonts() -> CardFonts:
    """Bundled Kanit when both files are present, else the PDF fonts (bold falls back to regular)."""
    bold, regular = FONTS_DIR / "Kanit-Bold.ttf", FONTS_DIR / "Kanit-Regular.ttf"
    if bold.is_file() and regular.is_file():
        return CardFonts(bold=str(bold), regular=str(regular))
    fs = find_fonts()
    reg = fs.regular or ""
    return CardFonts(bold=fs.bold or reg, regular=reg)


def thai_ready() -> bool:
    """True when Pillow can shape complex scripts (raqm + libfribidi found)."""
    return bool(features.check("raqm"))


@functools.lru_cache(maxsize=64)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


@functools.lru_cache(maxsize=8)
def _cmap(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    return frozenset((TTFont(path, lazy=True).getBestCmap() or {}).keys())


def drawable(text: str, path: str) -> str:
    """`text` without the characters the font cannot draw (emoji and the like would be boxes)."""
    cmap = _cmap(path)
    kept = "".join(c for c in text if c.isspace() or ord(c) in cmap)
    return " ".join(kept.split())
