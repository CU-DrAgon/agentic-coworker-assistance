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
import io
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, features

from .pdf_report import find_fonts

FONTS_DIR = Path(__file__).parent / "fonts"
LEADING = 1.0  # line height = (ascent + descent) * LEADING; Kanit's 1.5 em already fits Thai marks
SHADOW_ALPHA = 170  # opacity of the soft shadow under overlay text
JPEG_QUALITIES = (90, 85, 80)


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


# ------------------------------------------------------------------ text fitting
def break_ok(prev: str, nxt: str) -> bool:
    """May a line break fall between `prev` and `nxt`? Never before a combining mark or a Thai
    following vowel / sign, never after a Thai leading vowel (so a syllable stays together)."""
    if unicodedata.category(nxt) in ("Mn", "Mc", "Me"):
        return False
    if "ะ" <= nxt <= "ฺ" or "ๅ" <= nxt <= "๎":
        return False
    return not ("เ" <= prev <= "ไ")


@dataclass
class Fitted:
    size: int
    lines: list[str]
    width: int
    height: int
    line_height: int


def line_height(f: ImageFont.FreeTypeFont) -> int:
    asc, desc = f.getmetrics()
    return round((asc + desc) * LEADING)


def _split_word(word: str, f: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """`word` cut into pieces that fit, only where `break_ok` allows (a piece may still be too
    wide when no allowed break fits; it is never dropped)."""
    pieces = []
    while f.getlength(word) > max_width:
        cut = 0
        for i in range(1, len(word)):
            if break_ok(word[i - 1], word[i]):
                if f.getlength(word[:i]) > max_width:
                    break
                cut = i
        if cut == 0:  # no allowed break fits: take the first allowed one, else keep it whole
            cut = next((i for i in range(1, len(word)) if break_ok(word[i - 1], word[i])), 0)
            if cut == 0:
                break
        pieces.append(word[:cut])
        word = word[cut:]
    return [*pieces, word]


def _wrap(text: str, f: ImageFont.FreeTypeFont, max_width: int, split: bool) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if f.getlength(cand) <= max_width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        pieces = _split_word(word, f, max_width) if split else [word]
        lines += pieces[:-1]
        cur = pieces[-1]
    if cur:
        lines.append(cur)
    return lines


def _balance(text: str, lines: list[str], f: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Two greedy lines re-split at the space that makes them most even (no lone last word)."""
    words = text.split()
    if len(lines) != 2 or len(words) < 3 or " ".join(lines) != " ".join(words):
        return lines  # not two lines of whole words (a split word stays as it is)
    best = lines
    best_w = max(f.getlength(line) for line in lines)
    for k in range(1, len(words)):
        cand = [" ".join(words[:k]), " ".join(words[k:])]
        widest = max(f.getlength(line) for line in cand)
        if widest <= max_width and widest < best_w:
            best, best_w = cand, widest
    return best


def fit_text(
    text: str, path: str, max_width: int, max_lines: int, sizes: tuple[int, int]
) -> Fitted:
    """The largest size (from `sizes` = (max, min), 4 px steps) at which `text` wraps at spaces
    into at most `max_lines` lines of at most `max_width` px. At the minimum size too-wide words
    are split at syllable boundaries; text is never truncated."""
    hi, lo = sizes
    steps = [*range(hi, lo, -4), lo]
    for size in steps:
        f = font(path, size)
        last = size == steps[-1]
        lines = _balance(text, _wrap(text, f, max_width, split=last), f, max_width)
        widths = [f.getlength(line) for line in lines] or [0.0]
        fits = max(widths) <= max_width and len(lines) <= max_lines
        if fits or last:
            lh = line_height(f)
            return Fitted(size, lines, math.ceil(max(widths)), lh * len(lines), lh)
    raise AssertionError("unreachable")


def draw_lines(
    img: Image.Image,
    x: int,
    y: int,
    fitted: Fitted,
    path: str,
    *,
    fill: str,
    align: str = "left",
    width: int = 0,
    shadow: int = 0,
) -> tuple[int, int, int, int]:
    """Draw the fitted lines with their top-left at (x, y); returns the block box. Any extra
    leading is split above and below each line so tall Thai marks stay inside it. `shadow` > 0
    puts a soft dark shadow (blur radius in px) under the text, for text over photos."""
    f = font(path, fitted.size)
    asc, desc = f.getmetrics()
    lh = fitted.line_height
    block_w = width or fitted.width
    spots = []
    for i, line in enumerate(fitted.lines):
        free = block_w - f.getlength(line)
        dx = 0 if align == "left" else free if align == "right" else free / 2
        spots.append(((x + dx, y + i * lh + (lh - asc - desc) // 2 + asc), line))
    if shadow > 0:
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        for (sx, sy), line in spots:
            md.text((sx, sy + max(1, shadow // 3)), line, font=f, fill=SHADOW_ALPHA, anchor="ls")
        img.paste(
            Image.new(img.mode, img.size, 0), (0, 0), mask.filter(ImageFilter.GaussianBlur(shadow))
        )
    draw = ImageDraw.Draw(img)
    for xy, line in spots:
        draw.text(xy, line, font=f, fill=fill, anchor="ls")
    return (x, y, x + block_w, y + fitted.height)


# ------------------------------------------------------------------ shapes
def scrim(img: Image.Image, y0: int, y1: int, alpha0: int, alpha1: int) -> None:
    """Darken rows y0..y1 of `img` (RGB, in place) with black, alpha going alpha0 -> alpha1."""
    w = img.width
    y0, y1 = max(0, y0), min(img.height, y1)
    if y1 <= y0:
        return
    ramp = Image.linear_gradient("L").resize((w, y1 - y0))
    mask = ramp.point(lambda v: round(alpha0 + (alpha1 - alpha0) * v / 255))
    region = img.crop((0, y0, w, y1))
    img.paste(Image.composite(Image.new("RGB", region.size, 0), region, mask), (0, y0))


@dataclass
class Badge:
    w: int
    h: int
    fitted: Fitted


def measure_badge(
    text: str, path: str, sizes: tuple[int, int], max_width: int, pad: tuple[int, int] = (32, 16)
) -> Badge:
    fitted = fit_text(text, path, max_width - 2 * pad[0], 1, sizes)
    return Badge(fitted.width + 2 * pad[0], fitted.height + 2 * pad[1], fitted)


def draw_badge(
    img: Image.Image,
    badge: Badge,
    x: int,
    y: int,
    path: str,
    *,
    fill: str,
    text_fill: str,
    radius: int | None = None,
) -> tuple[int, int, int, int]:
    """A rounded rectangle (a pill when `radius` is None) with the text centred; returns its box."""
    box = (x, y, x + badge.w, y + badge.h)
    r = badge.h // 2 if radius is None else radius
    ImageDraw.Draw(img).rounded_rectangle(box, radius=r, fill=fill)
    pad_x, pad_y = (badge.w - badge.fitted.width) // 2, (badge.h - badge.fitted.height) // 2
    draw_lines(img, x + pad_x, y + pad_y, badge.fitted, path, fill=text_fill)
    return box


# ------------------------------------------------------------------ images
def cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover `size` and crop the centre: exactly `size`, RGB."""
    return ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)


def to_jpeg(img: Image.Image, max_bytes: int = 1_500_000) -> bytes:
    """JPEG at quality 90, then 85, 80 until under `max_bytes` (the last try is returned anyway)."""
    data = b""
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        data = buf.getvalue()
        if len(data) < max_bytes:
            break
    return data


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio of two #RRGGBB colours."""

    def lum(h: str) -> float:
        def ch(v: int) -> float:
            c = v / 255
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b_ = (int(h[i : i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b_)

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)
