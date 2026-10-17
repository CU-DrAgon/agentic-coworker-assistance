"""Concept studio: AI concept images of a product / screen / logo and web price checks.

`concept_image` is a text-to-image call to Nano Banana (`settings.promo_image_model`), optionally
with a reference image (the photo of the request, or the previous concept for a revision). The
prompt is built here (`concept_prompt`, pure) and always forbids text, logos and faces: Thai (or
any) text in the image is never trusted to the model. The `turnaround` view asks for ONE image
with four views of the same object on a 2x2 grid, and the labels ("front", "side"...) are drawn
by `cards`. The result is a concept render, not a 3D model or an engineering drawing.

Budget: one unit per image taken BEFORE the call (blocked calls are billed too) from its own
per-guild daily counter (`SALID_CONCEPT_IMAGES_PER_DAY`, separate from the promo kits). When over
the cap, blocked or failed the user gets an honest line and no image. The last concept of each
(guild, channel, user) is kept in process memory for 30 minutes for revisions ("rounder", "make
it green"). With an `idea` the JPEG is also stored in the idea's project document (see
`ideas.IdeaRepo.add_image`) and goes into its PDF export.

`price_check` runs ONE Google-grounded search for current prices and returns a table that code
rebuilt from the answer: a row without a price is dropped, the range comes from the numbers, the
sources are the grounding chunks. It never invents a price.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from google.genai import types
from PIL import Image

from . import cards
from .cards import CardFonts
from .i18n import lang_name, t
from .ideas import MAX_STORED_IMAGE_BYTES, IdeaRepo, ascii_name
from .llm import generate
from .media import DailyBudget
from .promo import (
    BLOCKED,
    PHOTO_MIMES,
    PROMO_TIMEOUT_MS,
    PhotoTooLarge,
    clean,
    prepare_photo,
    take_unit,
)
from .web_read import grounded_search
from .web_search import today_bangkok

logger = logging.getLogger(__name__)

IMAGE_KINDS = ("product", "scene", "ui", "logo", "moodboard")
VIEWS = ("front", "turnaround", "set")
MAX_BRIEF = 600
MAX_STYLE = 200
MAX_JPEG_BYTES = 1_500_000
_THAI = re.compile("[฀-๿]")
ASPECT = {
    "product": "1:1",
    "logo": "1:1",
    "moodboard": "1:1",
    "ui": "16:9",
    "scene": "16:9",
}
NO_TEXT = (
    "No text, letters, numbers, captions, watermarks or brand logos anywhere in the image "
    "(use plain shapes or grey bars where text would be). No people, hands or faces unless the "
    "brief asks for them."
)
KIND_PROMPT = {
    "product": (
        "A professional product design concept render of: {brief}. One single object, centred "
        "with generous margin on a plain seamless light-grey studio background, soft studio "
        "lighting, believable materials and finish, a soft shadow under the object."
    ),
    "scene": (
        "A realistic lifestyle scene showing this product or object in use: {brief}. Natural "
        "light, believable setting, the product clearly visible."
    ),
    "ui": (
        "A clean, modern user-interface mockup of: {brief}. A flat screen design (not a photo of "
        "a device) with a real layout: navigation, cards, lists, buttons, icons and images. "
        "Every piece of text is shown as grey placeholder bars, never as readable words."
    ),
    "logo": (
        "A logo design sheet for: {brief}. Three different logo symbol ideas side by side on a "
        "plain white background, simple flat vector style, bold shapes, a symbol only (no "
        "lettering)."
    ),
    "moodboard": (
        "A moodboard collage for: {brief}. A tidy flat-lay grid of photographs of textures, "
        "materials, colours and patterns that express the idea, plus a few colour swatches."
    ),
}
VIEW_PROMPT = {
    "front": "Show one view only: the main hero angle.",
    "turnaround": (
        "ONE image with exactly four views of the SAME object (identical design, colours and "
        "proportions) in a 2x2 grid of equal cells separated by plain empty background: "
        "top-left the front view, top-right the side view, bottom-left a three-quarter view, "
        "bottom-right the top view. Keep each view in the upper 80% of its cell and leave the "
        "bottom of every cell empty for a label."
    ),
    "set": (
        "ONE image with exactly three variations of the idea side by side (left, middle, right), "
        "clearly different from each other in shape, colour or detail, same camera angle and "
        "background, equal spacing, the bottom of the image empty for labels."
    ),
}
REF_PROMPT = {
    "ui": "The attached image shows the current page or sketch: keep its content and structure as the base.",
    "logo": "The attached image is a reference for the idea; do not copy lettering.",
    "moodboard": "The attached image is a reference for mood and colour.",
}
REF_DEFAULT = (
    "The attached image is a reference: keep this object's overall shape, proportions and "
    "main details."
)


def normalize_kind(kind: str) -> str:
    k = (kind or "product").strip().casefold()
    k = {"packshot": "product", "render": "product", "mockup": "ui", "screen": "ui"}.get(k, k)
    return k if k in IMAGE_KINDS else ""


def normalize_views(views: str, kind: str) -> str:
    v = (views or "front").strip().casefold()
    v = {"3d": "turnaround", "multi": "turnaround", "variations": "set", "options": "set"}.get(v, v)
    if v not in VIEWS:
        return ""
    # a turnaround (the "3D-ish" look) only makes sense for one object
    return "front" if v == "turnaround" and kind not in ("product",) else v


def aspect_for(kind: str, views: str) -> str:
    base = ASPECT[kind]
    return "16:9" if views == "set" and base == "1:1" else base


def concept_prompt(
    brief: str,
    kind: str = "product",
    views: str = "front",
    style: str = "",
    *,
    reference: bool = False,
    change: str = "",
) -> str:
    """The text-to-image prompt. `reference`: an image is attached. `change`: a revision, `brief`
    is then the previous brief and the attached image is the previous result."""
    kind = normalize_kind(kind) or "product"
    views = normalize_views(views, kind) or "front"
    parts = [KIND_PROMPT[kind].format(brief=clean(brief)), VIEW_PROMPT[views]]
    if style:
        parts.append(f"Style: {clean(style)}.")
    if change:
        parts.append(
            "The attached image is the previous version of this concept. Apply ONLY this change "
            f"and keep everything else the same: {clean(change)}."
        )
    elif reference:
        parts.append(REF_PROMPT.get(kind, REF_DEFAULT))
    parts.append(NO_TEXT)
    parts.append(f"Aspect ratio {aspect_for(kind, views)}.")
    return "\n".join(parts)


def generate_concept(
    client,
    model: str,
    prompt: str,
    aspect: str,
    size: str,
    reference: bytes | None = None,
) -> tuple[Image.Image | None, str]:
    """One text-to-image call: (image, "") or (None, reason) with reason "blocked", "timeout" or
    "error". `reference`: JPEG bytes passed as an image part before the prompt."""
    if client is None:
        return None, "error"
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect, image_size=size),
        http_options=types.HttpOptions(timeout=PROMO_TIMEOUT_MS),
    )  # no sampling params: an API error on Nano Banana 2.1
    contents: list[Any] = []
    if reference:
        contents.append(types.Part.from_bytes(data=reference, mime_type="image/jpeg"))
    contents.append(prompt)
    try:
        resp = generate(client, model, contents, config, delays=())
    except httpx.TimeoutException:
        logger.warning("concept image timed out")
        return None, "timeout"
    except Exception as e:  # never log the prompt
        logger.warning("concept image failed (%s)", type(e).__name__)
        return None, "error"
    cand = (resp.candidates or [None])[0]
    if cand is None or cand.finish_reason in BLOCKED:
        return None, "blocked"
    for part in (cand.content.parts if cand.content else None) or []:
        blob = part.inline_data
        if blob is not None and blob.data and (blob.mime_type or "").startswith("image/"):
            try:
                return Image.open(io.BytesIO(blob.data)).convert("RGB"), ""
            except Exception:
                logger.warning("concept image could not be decoded")
                return None, "error"
    return None, "blocked"


# ------------------------------------------------------------------ labels and files
def label_texts(views: str, lang: str) -> list[str]:
    """The labels drawn on a turnaround (4, grid order) or a set (3, left to right)."""
    if views == "turnaround":
        return [
            t("concept_lbl_front", lang),
            t("concept_lbl_side", lang),
            t("concept_lbl_three_quarter", lang),
            t("concept_lbl_top", lang),
        ]
    if views == "set":
        return [t("concept_lbl_option", lang, n=i) for i in (1, 2, 3)]
    return []


def add_labels(img: Image.Image, views: str, lang: str, fonts: CardFonts) -> Image.Image:
    """Labels at the bottom of each cell of a turnaround grid / each third of a set, typeset by
    `cards` (the image model never draws text). Returns `img` unchanged when there is nothing to
    label or Thai shaping is not available."""
    texts = label_texts(views, lang)
    if not texts or not cards.thai_ready() or not fonts.regular:
        return img
    img = img.convert("RGB")
    w, h = img.size
    scale = w / 1024
    sizes = (max(14, round(34 * scale)), max(12, round(20 * scale)))
    pad = (round(22 * scale), round(10 * scale))
    if views == "turnaround":
        cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
        cw, ch = w / 2, h / 2
        spots = [(cx * cw, (cy + 1) * ch, cw) for cx, cy in cells]
    else:
        cw = w / 3
        spots = [(i * cw, float(h), cw) for i in range(3)]
    margin = round(18 * scale)
    for text, (x0, y1, cwidth) in zip(texts, spots, strict=False):
        badge = cards.measure_badge(
            text, fonts.bold or fonts.regular, sizes, round(cwidth - 2 * margin), pad
        )
        x = round(x0 + (cwidth - badge.w) / 2)
        cards.draw_badge(
            img, badge, x, round(y1 - badge.h - margin), fonts.bold or fonts.regular,
            fill="#1F2937", text_fill="#FFFFFF",
        )  # fmt: skip
    return img


def fit_jpeg(img: Image.Image, max_bytes: int = MAX_JPEG_BYTES) -> bytes:
    """JPEG under `max_bytes`: the quality ladder of `cards.to_jpeg`, then smaller sizes."""
    img = img.convert("RGB")
    data = cards.to_jpeg(img, max_bytes)
    for _ in range(6):
        if len(data) <= max_bytes:
            break
        img = img.resize(
            (round(img.width * 0.8), round(img.height * 0.8)), Image.Resampling.LANCZOS
        )
        data = cards.to_jpeg(img, max_bytes)
    return data


# ------------------------------------------------------------------ revision memory
Key = tuple[int, int, int]  # (guild id, channel id, user id)


@dataclass
class LastConcept:
    brief: str
    kind: str
    views: str
    style: str
    idea: str
    jpeg: bytes  # the image as sent (labels included), the reference of a revision
    n: int


class ConceptMemory:
    """The last concept per person and channel, for revisions. Process memory only, `ttl_s` after
    the last image, at most `max_entries` (least recently used dropped). Thread-safe."""

    def __init__(
        self,
        ttl_s: float = 1800.0,
        max_entries: int = 50,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl_s, self.max_entries, self.clock = ttl_s, max_entries, clock
        self._items: OrderedDict[Key, tuple[float, LastConcept]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: Key, last: LastConcept) -> None:
        with self._lock:
            self._items[key] = (self.clock(), last)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, key: Key) -> LastConcept | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            if self.clock() - hit[0] > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return hit[1]


# ------------------------------------------------------------------ price check
COUNTRIES = {
    "TH": "Thailand (prices in THB, say whether VAT is included)",
    "US": "the United States (USD)",
    "GB": "the United Kingdom (GBP)",
    "SG": "Singapore (SGD)",
    "JP": "Japan (JPY)",
    "CN": "China (CNY)",
}
MAX_ROWS = 8
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CUR = (
    ("฿", re.compile(r"฿|บาท|\bTHB\b", re.I)),
    ("$", re.compile(r"\$|\bUSD\b", re.I)),
    ("€", re.compile(r"€|\bEUR\b", re.I)),
    ("£", re.compile(r"£|\bGBP\b", re.I)),
    ("¥", re.compile(r"¥|\b(?:JPY|CNY|RMB)\b", re.I)),
)


def price_columns(purpose: str, lang: str) -> list[str]:
    cols = [
        "แหล่ง|รุ่น/สเปก|ราคา|หน่วย",
        "Source|Model / spec|Price|Unit",
    ][0 if lang == "th" else 1].split("|")
    if purpose == "procure":
        cols += ["MOQ", "Lead time"]
    return [*cols, "วันที่/หมายเหตุ" if lang == "th" else "Date / note"]


def price_instruction(
    item: str, country: str, quantity: str, purpose: str, today: date, lang: str
) -> str:
    cols = price_columns(purpose, lang)
    where = COUNTRIES.get(country.upper(), country or "Thailand")
    extra = (
        " This is for PROCUREMENT: prefer suppliers, wholesalers and catalogue pages; fill MOQ "
        "(minimum order) and Lead time only when the source shows them, else write -."
        if purpose == "procure"
        else ""
    )
    qty = f" Quantity wanted: {quantity}; prefer prices that match it." if quantity else ""
    return (
        f"Today is {today.isoformat()} (Bangkok). Use Google Search to find CURRENT prices for "
        f"this item from shops, marketplaces or supplier catalogues in {where}: {item}.{qty}{extra}\n"
        f"Output ONLY a Markdown table with exactly these columns: | {' | '.join(cols)} |. At most "
        f"{MAX_ROWS} rows, one per distinct offer, cheapest first. The price cell holds the number "
        "as shown by the source with its currency. If a source shows no price, do NOT write a "
        "row for it. Never estimate or invent a price, model, shop or date. Put the listing date "
        "or a short remark (VAT included or not, shipping) in the last column. After the table "
        "write one line: RANGE: <lowest>-<highest> <currency>. If you found nothing reliable, "
        f"write only NONE. Write in {'Thai' if lang == 'th' else lang_name(lang)}."
    )


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_price_table(text: str, purpose: str, lang: str) -> tuple[list[list[str]], str]:
    """(rows, the model's RANGE line): rows rebuilt to our columns; header, separator and rows
    without a number in the price cell are dropped; at most MAX_ROWS."""
    ncols = len(price_columns(purpose, lang))
    rows: list[list[str]] = []
    rng = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = re.match(r"^\W*RANGE\W*:\s*(.+)$", line, re.I)
        if m:
            rng = re.sub(r"[*`]", "", m.group(1)).strip()
            continue
        if not line.startswith("|") or re.match(r"^\|?\s*:?-{2,}", line.strip("| ")):
            continue
        cells = _cells(line)
        if len(cells) < 3 or set("".join(cells)) <= set("-: "):
            continue
        price = cells[2]
        if not _NUM.search(price):  # the header row and anything without a price
            continue
        row = [c.replace("\n", " ") for c in cells[:ncols]]
        row += ["-"] * (ncols - len(row))
        if not row[0].strip("- "):
            continue
        if row not in rows:
            rows.append(row)
    return rows[:MAX_ROWS], rng


def _amount(cell: str) -> tuple[str, float] | None:
    m = _NUM.search(cell)
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    cur = next((sym for sym, rx in _CUR if rx.search(cell)), "")
    return cur, value


def price_range(rows: list[list[str]]) -> str:
    """ "฿1,290-2,490" from the price cells when they share one currency and one unit, else ""."""
    amounts = [a for r in rows if (a := _amount(r[2]))]
    units = {r[3].strip().casefold() for r in rows if r[3].strip("- ")}
    if (
        not amounts
        or len(amounts) != len(rows)
        or len({c for c, _ in amounts}) != 1
        or len(units) > 1
    ):
        return ""
    cur = amounts[0][0]
    lo, hi = min(v for _, v in amounts), max(v for _, v in amounts)

    def fmt(v: float) -> str:
        return f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"

    return f"{cur}{fmt(lo)}" if lo == hi else f"{cur}{fmt(lo)}-{fmt(hi)}"


def to_markdown_table(rows: list[list[str]], purpose: str, lang: str) -> str:
    cols = price_columns(purpose, lang)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *("| " + " | ".join(c.replace("|", "/") for c in r) + " |" for r in rows),
    ]
    return "\n".join(lines)


def to_lines(
    rows: list[list[str]], purpose: str, lang: str, rng: str = "", sources: list[dict] | None = None
) -> str:
    """The chat rendering (Discord does not draw Markdown tables): one numbered line per row,
    `1. source — model/spec: price unit (note)`, then the range line and the numbered sources.
    Procurement rows add MOQ and lead time inside the brackets."""
    out: list[str] = []
    for i, r in enumerate(rows, 1):
        extras = [c for c in r[4:-1] if c.strip("- ")]
        if purpose == "procure":
            extras = [
                f"{n} {c}"
                for n, c in zip(("MOQ", "lead time"), r[4:-1], strict=False)
                if c.strip("- ")
            ]
        note = ", ".join([*extras, *([r[-1]] if r[-1].strip("- ") else [])])
        price = " ".join(c for c in (r[2], r[3] if r[3].strip("- ") else "") if c)
        out.append(f"{i}. {r[0]} — {r[1]}: {price}" + (f" ({note})" if note else ""))
    if rng:
        out += ["", f"{t('concept_price_range', lang)}: {rng}"]
    if sources:
        out.append("")
        out += [f"{s['n']}. {s['title']} — {s['link']}".rstrip(" —") for s in sources]
    return "\n".join(out)


# ------------------------------------------------------------------ the agent tools
NOTE_IMAGE = (
    "Reply in the user's language, short: say the image is attached and that it is an AI "
    "concept render (not a 3D model or an engineering drawing). "
)
NOTE_TURNAROUND = (
    "It shows the same object from the front, side, three-quarter and top; say it is a concept "
    "render, not a real 3D model. "
)
NOTE_NEXT = (
    "Offer ONE next step in one line (change the look, another view, the brand theme, website copy "
    "or the marketing plan)."
)


@dataclass
class ConceptCtx:
    """One request's concept tools, bound by the bot (guild, channel and author are never model
    input). `photos`: (attachment index, data) of this message's usable photos."""

    guild_id: int
    day: str
    lang: str
    asker_id: int
    asker_name: str
    key: Key
    image_client: Any
    image_model: str
    image_size: str
    budget: DailyBudget
    memory: ConceptMemory
    repo: IdeaRepo | None
    photos: list[tuple[int, bytes]] = field(default_factory=list)
    fonts: CardFonts | None = None
    web_client: Any = None
    web_model: str = ""
    today: Callable[[], date] = today_bangkok
    outbox: list[tuple[str, bytes]] = field(default_factory=list)

    # ---- concept_image
    def concept_image(
        self,
        brief: str,
        kind: str = "product",
        views: str = "front",
        style: str = "",
        idea: str = "",
        revise: bool = False,
        file_index: int = -1,
    ) -> dict:
        try:
            return self._concept_image(brief, kind, views, style, idea, revise, file_index)
        except PhotoTooLarge as e:
            return {"status": "error", "error": f"{e}; ask for a smaller photo"}

    def _concept_image(self, brief, kind, views, style, idea, revise, file_index) -> dict:
        brief, style = clean(brief), clean(style)
        idea = " ".join((idea or "").split())
        last = self.memory.get(self.key) if revise else None
        if revise and last is None:
            return {
                "status": "no_previous",
                "note": "There is no concept from the last 30 minutes to change: ask the user what "
                "to draw (or call again with revise=false and a full brief).",
            }
        k = normalize_kind(kind if not last else last.kind)
        if not k:
            return {"status": "error", "error": f"kind must be one of: {', '.join(IMAGE_KINDS)}"}
        v = normalize_views(views if (views or "front") != "front" or not last else last.views, k)
        if not v:
            return {"status": "error", "error": f"views must be one of: {', '.join(VIEWS)}"}
        style = style or (last.style if last else "")
        if len(brief) < 3:
            return {
                "status": "error",
                "error": "brief is required: describe the object or screen in English",
            }
        for name, text in (("brief", brief), ("style", style)):
            if _THAI.search(text):  # the image model must never get Thai text (it would draw it)
                return {"status": "error", "error": f"{name} must be written in English"}
        if len(brief) > MAX_BRIEF or len(style) > MAX_STYLE:
            return {
                "status": "error",
                "error": f"brief max {MAX_BRIEF}, style max {MAX_STYLE} characters",
            }
        idea = idea or (last.idea if last else "")

        change = brief if last else ""
        full_brief = f"{last.brief}; {brief}"[: MAX_BRIEF * 2] if last else brief
        reference: bytes | None = None
        if last:
            reference = last.jpeg
        else:
            photo = self._photo(file_index)
            if photo is not None:
                reference = photo
        prompt = concept_prompt(
            last.brief if last else brief,
            k,
            v,
            style,
            reference=reference is not None,
            change=change,
        )

        t0 = time.monotonic()
        reason = take_unit(self.budget, self.guild_id, self.day)
        image = None
        if not reason:
            image, reason = generate_concept(
                self.image_client,
                self.image_model,
                prompt,
                aspect_for(k, v),
                self.image_size,
                reference,
            )
        if image is None:
            logger.info("concept image guild=%s outcome=%s", self.guild_id, reason)
            return {
                "status": "unavailable",
                "reason": reason,
                "message": t(f"concept_img_{reason}", self.lang),
                "note": "Tell the user this message (shortly, in their language); no image was made.",
            }
        if self.fonts is not None:
            try:
                image = add_labels(image, v, self.lang, self.fonts)
            except Exception:
                logger.warning("concept labels failed", exc_info=True)
        jpeg = fit_jpeg(image)
        n = self._next_number(idea, last)
        name = f"concept-{ascii_name(idea or full_brief)}-{k}-{n}.jpg"
        stored = self._store(idea, name, k, full_brief, image)
        self.outbox.append((name, jpeg))
        self.memory.put(self.key, LastConcept(full_brief, k, v, style, idea, jpeg, n))
        logger.info(
            "concept image guild=%s kind=%s views=%s ms=%d bytes=%d",
            self.guild_id,
            k,
            v,
            (time.monotonic() - t0) * 1000,
            len(jpeg),
        )
        note = NOTE_IMAGE + (NOTE_TURNAROUND if v == "turnaround" else "")
        if stored:
            note += f"It is also saved in the project document “{stored}”. "
        elif idea:
            note += "It could not be saved in the project document; say so. "
        note += NOTE_NEXT
        return {
            "status": "ok",
            "file": name,
            "kind": k,
            "views": v,
            "saved_in_idea": stored,
            "images_left_today": max(
                0, int(self.budget.cap - self.budget.used(self.guild_id, self.day))
            ),
            "note": note,
        }

    def _photo(self, file_index: int) -> bytes | None:
        """The reference photo of the request (JPEG, upright, <= 2048 px) or None."""
        if not self.photos:
            return None
        if file_index is not None and file_index >= 0:
            data = next((d for i, d in self.photos if i == file_index), None)
        else:
            data = self.photos[0][1]
        photo = prepare_photo(data) if data else None
        return photo.jpeg if photo else None

    def _next_number(self, idea: str, last: LastConcept | None) -> int:
        top = last.n if last else 0
        if idea and self.repo is not None:
            doc = self.repo.get(self.guild_id, self.asker_id, _slug(idea))
            for im in doc.images if doc else []:
                m = re.search(r"-(\d+)\.jpg$", im.get("name", ""))
                top = max(top, int(m.group(1)) if m else 0)
        return top + 1

    def _store(self, idea: str, name: str, kind: str, brief: str, image: Image.Image) -> str:
        """Keep the image under the idea; the idea's title, or "" when nothing was stored."""
        if not idea or self.repo is None:
            return ""
        caption = (
            f"{t('concept_kind_' + kind, self.lang)} · {self.today().isoformat()} · {brief[:140]}"
        )
        try:
            doc = self.repo.add_image(
                self.guild_id,
                self.asker_id,
                self.asker_name,
                idea,
                name,
                caption,
                fit_jpeg(image, MAX_STORED_IMAGE_BYTES),
            )
        except Exception:
            logger.warning("concept image not stored", exc_info=True)
            return ""
        return doc.title

    # ---- price_check
    def price_check(
        self,
        item: str,
        country: str = "TH",
        quantity: str = "",
        purpose: str = "buy",
        idea: str = "",
    ) -> dict:
        item, quantity = clean(item)[:200], clean(quantity)[:80]
        if len(item) < 2:
            return {"status": "error", "error": "item is required (what to price)"}
        if self.web_client is None:
            return {"status": "error", "error": "web search is off for this server"}
        country = (clean(country) or "TH")[:40]
        purpose = (
            "procure"
            if (purpose or "").strip().casefold() in ("procure", "procurement", "buy_bulk", "b2b")
            else "buy"
        )
        today = self.today()
        res = grounded_search(
            self.web_client,
            self.web_model,
            f"Current prices of: {item}",
            price_instruction(item, country, quantity, purpose, today, self.lang),
        )
        if "error" in res and not res.get("text"):
            return {"status": "error", "error": res["error"]}
        rows, model_range = parse_price_table(res.get("text", ""), purpose, self.lang)
        sources = [
            {"n": i, "title": s.get("title") or "", "link": s.get("uri") or ""}
            for i, s in enumerate(res.get("sources") or [], 1)
        ]
        if not rows:
            return {
                "status": "none",
                "item": item,
                "message": t("concept_price_none", self.lang),
                "sources": sources,
                "note": "Say this message; do not guess a price.",
            }
        table = to_markdown_table(rows, purpose, self.lang)
        rng = price_range(rows) or ""
        lines = to_lines(rows, purpose, self.lang, rng or model_range, sources)
        out: dict[str, Any] = {
            "status": "ok",
            "item": item,
            "country": country,
            "date": today.isoformat(),
            "rows": len(rows),
            "table": table,
            "lines": lines,
            "range": rng,
            "sources": sources,
        }
        if not rng and model_range:
            out["range_note"] = model_range  # units or currencies differ: only the model's line
        saved = self._save_sourcing(idea, item, today, table, rng, sources)
        out["saved_in_idea"] = saved
        out["instruction"] = (
            "In Discord chat relay `lines` verbatim (it already has the price range and the "
            "numbered sources; Discord does not render Markdown tables, so never show `table` "
            "there; do not change any number). "
            + t("concept_price_recheck", self.lang, date=today.isoformat())
            + (f" Say it was saved in the project document “{saved}”." if saved else "")
        )
        return out

    def _save_sourcing(self, idea, item, today, table, rng, sources) -> str:
        idea = " ".join((idea or "").split())
        if not idea or self.repo is None:
            return ""
        lines = [f"### {today.isoformat()} · {item}", "", table, ""]
        if rng:
            lines += [f"{t('concept_price_range', self.lang)}: {rng}", ""]
        lines += [f"[{s['n']}] {s['title']} - {s['link']}" for s in sources]
        try:
            doc, _key, _offer = self.repo.save_section(
                self.guild_id,
                self.asker_id,
                self.asker_name,
                idea,
                "sourcing",
                "\n".join(lines),
                append=True,
            )
        except Exception:
            logger.warning("sourcing not saved", exc_info=True)
            return ""
        return doc.title


def _slug(title: str) -> str:
    from .wiki import slugify

    return slugify(" ".join((title or "").split())[:120])


def concept_tools(ctx: ConceptCtx, *, images: bool = True) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions).
    `images=False` (voice): only `price_check`, an image cannot be spoken. Files are collected in
    `ctx.outbox` (`outbox` attribute: see `qa.tool_files`)."""

    def concept_image(
        brief: str,
        kind: str = "product",
        views: str = "front",
        style: str = "",
        idea: str = "",
        revise: bool = False,
        file_index: int = -1,
    ) -> dict:
        """Draw a CONCEPT IMAGE with AI (text to image): the look of a product or object, the
        object in use, a web / app screen mockup, logo ideas or a moodboard. It is a concept
        render, NOT a 3D model or an engineering drawing; say so. All text in the image is added
        by Salid, never drawn by the model. Use for "ออกแบบหน้าตาสินค้า", "ขอภาพคอนเซปต์",
        "ลองวาดโลโก้", "show me what it could look like". For a change to the last image
        ("ทำให้โค้งมนกว่านี้", "เปลี่ยนเป็นสีเขียว") call again with revise=true and in `brief` ONLY the
        change (the last image is remembered 30 minutes). If the message has a photo it is used
        as a reference for the shape.

        Args:
            brief: what to draw, IN ENGLISH, max 600 characters: the object, shape, materials,
                colours, mood, key details. For a screen: the page and its blocks. Never include
                words that should appear in the image.
            kind: product (default; a studio render on a plain background) | scene (the product
                in use) | ui (web / app mockup) | logo (3 symbol ideas) | moodboard.
            views: front (default; one image) | turnaround (ONE image with front, side,
                three-quarter and top views of the same product on a grid: the "3D-ish" look) |
                set (three variations side by side).
            style: optional style words IN ENGLISH, e.g. "minimal Scandinavian, matte white and
                oak" (max 200 characters).
            idea: the title of the user's saved project / idea when the image belongs to it (the
                image is then kept in that project document and exported with it). Empty = just
                send the image.
            revise: true to change the last image of this chat; `brief` is then only the change.
            file_index: which attached photo is the reference (the number in its label); -1 = the
                first photo if the message has one.
        """
        try:
            return ctx.concept_image(brief, kind, views, style, idea, revise, file_index)
        except Exception:
            logger.exception("concept_image failed")
            return {"status": "error", "error": "the concept image failed; try again"}

    def price_check(
        item: str, country: str = "TH", quantity: str = "", purpose: str = "buy", idea: str = ""
    ) -> dict:
        """Check CURRENT prices of a product or material on the internet (Google Search): shops,
        marketplaces and supplier catalogues. Returns `lines` (numbered rows: source, model / spec,
        price, unit, note; then the price range and numbered sources) for chat, and `table`
        (Markdown, for documents only). Discord shows no tables: relay `lines` as returned and add that prices must be re-checked before buying. Rows without a real
        price are dropped. Use for "ราคาเท่าไร", "เช็คราคา", "หาซัพพลายเออร์", "how much does X cost".
        For a supplier's catalogue page use web_search / read_web instead.

        Args:
            item: what to price, with brand / model / size when known, in the user's language
                (max 200 characters). Never put team data in it: it goes to Google.
            country: ISO country code, default "TH" (Thailand, THB).
            quantity: optional, e.g. "100 pcs".
            purpose: "buy" (default; retail prices) or "procure" (suppliers, adds MOQ and lead
                time columns when the sources show them).
            idea: optional title of the user's project; the result is then appended, with the
                date, to its "Sourcing & prices" section.
        """
        try:
            return ctx.price_check(item, country, quantity, purpose, idea)
        except Exception:
            logger.exception("price_check failed")
            return {"status": "error", "error": "the price check failed; try again"}

    tools = [concept_image, price_check] if images else [price_check]
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]
    return tools


__all__ = [
    "PHOTO_MIMES",
    "BLOCKED",
    "ConceptCtx",
    "ConceptMemory",
    "concept_prompt",
    "concept_tools",
]
