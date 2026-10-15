"""Accounting and meeting documents as PDF (fpdf2 + uharfbuzz text shaping so Thai vowels and
tone marks render correctly).

Fonts: a Thai-capable TTF/OTF is looked up at runtime (`SALID_PDF_FONT`, then Sarabun / Noto Sans
Thai / the TLWG fonts of Debian's `fonts-thai-tlwg`). Without one the PDF still builds with a
Latin font, Thai text will show as boxes, and a warning is logged.

Money is never added across currencies: totals are listed per currency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .i18n import t
from .models import Receipt
from .store import LOW_CONF, cur_of

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "fonts"
# (regular, bold or None) in order of preference.
KNOWN_FONTS: list[tuple[str, str | None]] = [
    (str(BUNDLED_DIR / "Sarabun-Regular.ttf"), str(BUNDLED_DIR / "Sarabun-Bold.ttf")),
    ("/usr/share/fonts/truetype/google-sarabun/Sarabun-Regular.ttf",
     "/usr/share/fonts/truetype/google-sarabun/Sarabun-Bold.ttf"),
    ("/usr/share/fonts/opentype/tlwg/Loma.otf", "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"),
    ("/usr/share/fonts/truetype/tlwg/Loma.ttf", "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf"),
    ("/usr/share/fonts/truetype/tlwg/Garuda.ttf", "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf"),
    ("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
     "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"),
    ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"),
]  # fmt: skip
LATIN_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


@dataclass
class FontSet:
    regular: str | None
    bold: str | None
    thai: bool

    @property
    def family(self) -> str:
        return Path(self.regular).stem.split("-")[0] if self.regular else "helvetica"


def supports_thai(path: str) -> bool:
    """True when the font file maps Thai letters and the tone mark (fontTools cmap)."""
    try:
        from fontTools.ttLib import TTFont

        cmap = TTFont(path, lazy=True).getBestCmap() or {}
        return all(ord(c) in cmap for c in "กขคมา่ั")
    except Exception:
        return False


def find_fonts(configured: str = "") -> FontSet:
    """`configured` (SALID_PDF_FONT, a regular face; a sibling `-Bold` is used when present)
    first, then the known paths. A configured font without Thai glyphs is still used (the
    operator chose it) but logged."""
    candidates: list[tuple[str, str | None]] = []
    if configured:
        p = Path(configured)
        bold = next(
            (str(b) for b in (p.with_name(p.stem + "-Bold" + p.suffix),) if b.exists()), None
        )
        candidates.append((configured, bold))
    candidates += KNOWN_FONTS
    for reg, bold in candidates:
        if Path(reg).is_file():
            return FontSet(reg, bold if bold and Path(bold).is_file() else None, supports_thai(reg))
    for reg in LATIN_FALLBACKS:
        if Path(reg).is_file():
            logger.warning("no Thai font found for PDFs (set SALID_PDF_FONT); Thai text will break")
            return FontSet(reg, None, False)
    logger.warning("no TrueType font found for PDFs; using the built-in Latin font")
    return FontSet(None, None, False)


def fmt_amount(amount: float | None) -> str:
    return "-" if amount is None else f"{amount:,.2f}"


def totals_by_currency(rows: list[Receipt]) -> dict[str, float]:
    by: dict[str, float] = {}
    for r in rows:
        by[cur_of(r)] = by.get(cur_of(r), 0.0) + (r.total or 0.0)
    return {c: round(v, 2) for c, v in sorted(by.items(), key=lambda kv: (kv[0] != "THB", kv[0]))}


def fmt_totals_plain(by: dict[str, float]) -> str:
    return ", ".join(f"{c} {fmt_amount(v)}" for c, v in by.items()) or "-"


class _Pdf(FPDF):
    def __init__(self, fonts: FontSet, org: str, lang: str, generated_at: datetime):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.fonts_, self.org, self.lang, self.generated_at = fonts, org, lang, generated_at
        self.family = "Doc"
        if fonts.regular:
            self.add_font(self.family, "", fonts.regular)
            self.add_font(self.family, "B", fonts.bold or fonts.regular)
            try:
                self.set_text_shaping(True)
            except Exception:  # uharfbuzz missing: still a valid PDF, Thai marks may misplace
                logger.warning("PDF text shaping unavailable")
        else:
            self.family = "helvetica"
        self.set_auto_page_break(True, margin=16)
        self.set_margins(14, 14, 14)
        self.alias_nb_pages()

    def _txt(self, s: str) -> str:
        if self.fonts_.regular is None:  # core font: Latin-1 only
            return s.encode("latin-1", "replace").decode("latin-1")
        return s

    def header(self) -> None:
        self.set_font(self.family, "B", 10)
        self.set_text_color(90)
        self.cell(0, 6, self._txt(self.org), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(180)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self.family, "", 8)
        self.set_text_color(110)
        stamp = t("pdf_generated", self.lang, at=f"{self.generated_at:%Y-%m-%d %H:%M}")
        self.cell(130, 6, self._txt(stamp))
        self.cell(
            0, 6, self._txt(t("pdf_page", self.lang, n=self.page_no(), total="{nb}")), align="R"
        )

    # small helpers
    def h1(self, text: str) -> None:
        self.set_font(self.family, "B", 18)
        self.multi_cell(0, 9, self._txt(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h2(self, text: str) -> None:
        self.ln(3)
        self.set_font(self.family, "B", 12)
        self.multi_cell(0, 7, self._txt(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def p(self, text: str, size: int = 10, bold: bool = False) -> None:
        self.set_font(self.family, "B" if bold else "", size)
        self.multi_cell(0, 5.5, self._txt(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, head: list[str], rows: list[list[str]], widths: list[float], right=()) -> None:
        self.set_font(self.family, "B", 9)
        self.set_fill_color(235, 238, 245)
        for h, w in zip(head, widths, strict=True):
            self.cell(w, 7, self._txt(h), border=1, fill=True)
        self.ln()
        self.set_font(self.family, "", 9)
        for row in rows:
            if self.get_y() > 270:
                self.add_page()
                self.set_font(self.family, "B", 9)
                for h, w in zip(head, widths, strict=True):
                    self.cell(w, 7, self._txt(h), border=1, fill=True)
                self.ln()
                self.set_font(self.family, "", 9)
            for i, (c, w) in enumerate(zip(row, widths, strict=True)):
                self.cell(
                    w, 6.5, self._txt(self._clip(c, w)), border=1, align="R" if i in right else "L"
                )
            self.ln()

    def _clip(self, text: str, width: float) -> str:
        """Single-line cells: shorten text that would overflow the column."""
        text = (text or "").replace("\n", " ")
        while text and self.get_string_width(text) > width - 2:
            text = text[:-2] + "…" if text.endswith("…") else text[:-1] + "…"
        return text


def _finish(pdf: _Pdf) -> bytes:
    return bytes(pdf.output())


def _flags(r: Receipt, lang: str) -> str:
    out = []
    if (r.confidence or 0) < LOW_CONF:
        out.append(t("pdf_flag_low", lang))
    if r.possible_duplicate_of:
        out.append(t("pdf_flag_dup", lang, id=r.possible_duplicate_of))
    return ", ".join(out)


def expense_pdf(
    *,
    org: str,
    title: str,
    start: str,
    end: str,
    receipts: list[Receipt],
    void_count: int = 0,
    scope_note: str = "",
    notes: str = "",
    kind: str = "expense_summary",
    lang: str = "th",
    generated_at: datetime,
    fonts: FontSet,
) -> bytes:
    """`receipts` are the ACTIVE receipts of the period (void ones are only counted)."""
    pdf = _Pdf(fonts, org, lang, generated_at)
    pdf.add_page()
    pdf.h1(title or t(f"pdf_title_{kind}", lang))
    pdf.p(t("pdf_period", lang, start=start, end=end))
    if scope_note:
        pdf.p(scope_note)
    if notes:
        pdf.p(notes)
    by = totals_by_currency(receipts)
    if kind == "expense_summary":
        pdf.h2(t("pdf_totals", lang))
        pdf.table(
            [t("pdf_currency", lang), t("pdf_amount", lang)],
            [[c, fmt_amount(v)] for c, v in by.items()] or [["-", "-"]],
            [40, 50],
            right=(1,),
        )
        pdf.p(t("pdf_no_fx", lang), size=8)
        for label_key, key in (("pdf_by_category", "category"), ("pdf_by_submitter", "submitter")):
            groups: dict[tuple[str, str], float] = {}
            for r in receipts:
                g = (r.category if key == "category" else r.submitter_name) or "-"
                groups[(g, cur_of(r))] = groups.get((g, cur_of(r)), 0.0) + (r.total or 0.0)
            pdf.h2(t(label_key, lang))
            rows = [
                [g, c, fmt_amount(round(v, 2))]
                for (g, c), v in sorted(groups.items(), key=lambda kv: (kv[0][1] != "THB", -kv[1]))
            ]
            pdf.table(
                [t("pdf_name", lang), t("pdf_currency", lang), t("pdf_amount", lang)],
                rows or [["-", "-", "-"]],
                [90, 30, 50],
                right=(2,),
            )
    pdf.h2(t("pdf_receipts", lang, n=len(receipts)))
    head = [
        t("pdf_date", lang), t("pdf_vendor", lang), t("pdf_category", lang),
        t("pdf_amount", lang), t("pdf_cur", lang), t("pdf_flags", lang),
    ]  # fmt: skip
    rows = [
        [
            r.effective_date, r.vendor or "-", r.category or "-", fmt_amount(r.total), cur_of(r),
            _flags(r, lang),
        ]
        for r in receipts
    ]  # fmt: skip
    pdf.table(head, rows or [["-"] * 6], [22, 52, 28, 28, 14, 38], right=(3,))
    if kind == "receipt_list" and by:
        pdf.p(t("pdf_total_line", lang, total=fmt_totals_plain(by)), bold=True)
    if void_count:
        pdf.p(t("pdf_void_note", lang, n=void_count), size=8)
    return _finish(pdf)


def minutes_pdf(
    *,
    org: str,
    title: str,
    when: str,
    attendees: list[str],
    summary: str,
    attendance: dict[str, str] | None = None,
    notes: str = "",
    lang: str = "th",
    generated_at: datetime,
    fonts: FontSet,
) -> bytes:
    pdf = _Pdf(fonts, org, lang, generated_at)
    pdf.add_page()
    pdf.h1(title or t("pdf_title_meeting_minutes", lang))
    pdf.p(t("pdf_when", lang, when=when))
    if attendees:
        pdf.p(t("pdf_attendees", lang, names=", ".join(attendees)))
    if attendance:
        pdf.h2(t("pdf_attendance", lang))
        for name, note in attendance.items():
            pdf.p(f"- {name}: {note}")
    if notes:
        pdf.p(notes)
    pdf.h2(t("pdf_summary", lang))
    pdf.p(summary or t("pdf_no_summary", lang))
    return _finish(pdf)


def custom_pdf(
    *,
    org: str,
    title: str,
    body: str,
    lang: str = "th",
    generated_at: datetime,
    fonts: FontSet,
) -> bytes:
    pdf = _Pdf(fonts, org, lang, generated_at)
    pdf.add_page()
    pdf.h1(title or t("pdf_title_custom", lang))
    pdf.p(body)
    return _finish(pdf)
