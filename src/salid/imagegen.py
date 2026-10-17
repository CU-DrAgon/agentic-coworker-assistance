"""Image generation with Nano Banana (`settings.image_model`): the one image-model call shared by
promo kits, `create_image` and the video ad stills, and the guards every caller needs.

- `generate_images`: `count` parallel calls (the model returns one image per call), one attempt
  each (no retry: exactly one bill per image), person and prominent-people settings fixed here,
  no sampling parameters (an API error on Nano Banana 2.1).
- Blocked = a blocked prompt, a blocked finish reason or an answer without an image.
- `english_only`: prompts must never carry Thai (the model would draw it); Thai text is typeset
  by Salid with `salid.cards`.
- `open_capped`: the decode guard for images decoded in the shared bot process (2 GiB).
- `create_image` (`image_tools`, bound by `imagegen_bot.ImageMixin`): 1-4 images from an English
  prompt, an attached photo or a remembered result as the reference, optional Thai/English text
  typeset by `overlay` on a scrim band (top for 9:16, bottom otherwise), PNG files (the whole
  batch as JPEG when the PNGs pass `ImageCtx.max_bytes`, which follows the server's upload
  limit, at most OUTBOX_MAX_BYTES; still over it: `fit_batch` lowers the quality, then the size,
  then drops trailing images and the result says so). Daily limits per guild and per person (any
  `DayBudget`: `media.DailyBudget`, or the DocStore-backed one the bot uses,
  `imagegen_bot.StoredDailyBudget`, whose `pair` reads both counts at once), taken per image
  before the calls and given back only when no model call is made (no client, or a prompt
  refused before the call): blocked, timed-out and failed calls stay charged, and when the text
  overlay (or a later step) fails the billed images are sent without the text, with a note.
  References are only photos of
  products, logos, food or places, never of a real person (the tool description and NOTE tell
  the agent; spec section 5). `ImageMemory` keeps the last batch per (guild, channel, user) for
  revisions, in process memory only.

Never logs prompts or images.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from google.genai import types
from PIL import Image, ImageOps

from . import cards
from .cards import CardFonts
from .llm import generate

logger = logging.getLogger(__name__)

MAX_DECODE_PIXELS = 50_000_000  # bigger canvases are refused before decoding (bot memory is 2 GiB)
IMAGE_TIMEOUT_MS = 60_000
MAX_PARALLEL = 4
BLOCKED: set[types.FinishReason] = {
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    types.FinishReason.IMAGE_RECITATION,
    types.FinishReason.IMAGE_OTHER,
    types.FinishReason.NO_IMAGE,
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
}
_THAI = re.compile(r"[\u0E00-\u0E7F]")
_REASON_ORDER = ("blocked", "timeout", "error")  # the reason reported when no call gave an image


class ImageTooLarge(ValueError):
    """The canvas is bigger than the pixel cap: refused from the header, before decoding."""


def english_only(text: str) -> bool:
    """False if `text` has any Thai character (U+0E00-U+0E7F)."""
    return _THAI.search(text or "") is None


def open_capped(
    data: bytes, *, formats: list[str] | None = None, max_pixels: int | None = None
) -> Image.Image:
    """The image opened (header read, pixels not decoded yet), optionally restricted to Pillow
    `formats`. Raises ImageTooLarge when the header declares more than `max_pixels` (default
    MAX_DECODE_PIXELS; a tiny file can claim a huge canvas) or Pillow refuses it as a
    decompression bomb, and ValueError("unreadable image") for anything else it cannot open."""
    limit = MAX_DECODE_PIXELS if max_pixels is None else max_pixels
    too_large = f"image is larger than {limit // 1_000_000} megapixels"
    try:
        im = Image.open(io.BytesIO(data), formats=formats)
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(too_large) from e
    except Exception as e:
        raise ValueError("unreadable image") from e
    if im.width * im.height > limit:
        im.close()
        raise ImageTooLarge(too_large)
    return im


def _mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _pick(resp: types.GenerateContentResponse) -> tuple[Image.Image | None, str]:
    """The last non-thought image of the answer, or (None, "blocked" | "error")."""
    feedback = resp.prompt_feedback
    if feedback is not None and feedback.block_reason:
        return None, "blocked"
    cand = (resp.candidates or [None])[0]
    if cand is None or cand.finish_reason in BLOCKED:
        return None, "blocked"
    data = None
    for part in (cand.content.parts if cand.content else None) or []:
        blob = part.inline_data
        if part.thought or blob is None or not blob.data:
            continue
        if (blob.mime_type or "").startswith("image/"):
            data = blob.data
    if data is None:
        return None, "blocked"  # a text-only answer is a refusal
    try:
        with open_capped(data) as im:
            return im.convert("RGB"), ""
    except Exception:
        logger.warning("generated image could not be decoded")
        return None, "error"


def _one(client, model: str, contents: list, config) -> tuple[Image.Image | None, str]:
    try:
        resp = generate(client, model, contents, config, delays=())
    except httpx.TimeoutException:
        logger.warning("image call timed out")
        return None, "timeout"
    except Exception as e:  # never log the prompt
        logger.warning("image call failed (%s)", type(e).__name__)
        return None, "error"
    return _pick(resp)


def generate_images(
    client,
    model: str,
    prompt: str,
    *,
    refs: Sequence[bytes] = (),
    aspect: str = "1:1",
    size: str = "1K",
    count: int = 1,
) -> tuple[list[Image.Image], str]:
    """`count` images of `prompt` (reference images `refs` go before it), one call each, in
    parallel. Returns (images, "") when at least one came back, else ([], reason) with reason
    "blocked" (safety / no image), "timeout" or "error". A prompt with Thai in it is refused
    ("error") without any call: the model would draw the Thai text."""
    if client is None or count < 1:
        return [], "error"
    if not english_only(prompt):
        logger.warning("image prompt refused: not English")
        return [], "error"
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect,
            image_size=size,
            person_generation="ALLOW_ADULT",
            prominent_people="BLOCK_PROMINENT_PEOPLE",
        ),
        http_options=types.HttpOptions(timeout=IMAGE_TIMEOUT_MS),
    )  # no temperature / top_p / top_k / seed / logprobs: an API error on Nano Banana 2.1
    contents = [*(types.Part.from_bytes(data=r, mime_type=_mime(r)) for r in refs), prompt]
    with ThreadPoolExecutor(max_workers=min(count, MAX_PARALLEL)) as pool:
        results = list(pool.map(lambda _: _one(client, model, contents, config), range(count)))
    images = [img for img, _ in results if img is not None]
    if images:
        return images, ""
    reasons = {reason for _, reason in results}
    return [], next(r for r in _REASON_ORDER if r in reasons)


# ------------------------------------------------------------------ create_image
ASPECTS = ("1:1", "4:5", "9:16", "16:9")
SIZES = ("1K", "2K")
MAX_IMAGES = 4
PROMPT_MAX = 600
HEADLINE_MAX = 40
SUBTITLE_MAX = 60
REF_SIDE = 2048  # an attached reference photo is shrunk to this long side (like promo photos)
MEMORY_SIDE = 1376  # remembered results: 1K kept as made, 2K halved (a reference needs no more)
MEMORY_BYTES = 128 * 1024 * 1024  # all remembered results together, least recently used dropped
OUTBOX_MAX_BYTES = 20 * 1024 * 1024  # one reply's files at most; the bot lowers it per server
NOTE = (
    "Reply briefly in the user's language: the images are attached and were made with AI. Do not "
    "describe people in them as real. Reference photos (file_index, revise) are only for photos "
    "of products, logos, food or places, never a photo of a real person: refuse that and offer "
    "an illustration instead."
)
NOTE_TRIMMED = " Only {n} of {asked} could be made today (daily limit)."
NOTE_FAILED = (
    " The image service refused or failed for some of them; suggest a different description."
)
NOTE_LIMIT = "Today's image limit for this server (or this person) is used up."
NOTE_DROPPED = " Only {n} of {made} images fit this server's upload limit; say so."
NOTE_NO_TEXT = (
    " The text could not be added, so the images are sent without the text; say so and offer to"
    " try again."
)
FAILED = {
    "blocked": "the image service refused this description; suggest a different one",
    "timeout": "the image service timed out; suggest trying again",
    "error": "the image service failed; suggest trying again",
    "too_large": "the images are too large for this server's upload limit; suggest fewer or 1K",
}
# Tried in order while a JPEG batch is still over the upload budget: (quality, share of the
# long side). After the last one, trailing images are dropped.
SHRINK_STEPS = ((75, 1.0), (60, 1.0), (60, 0.75), (50, 0.5), (40, 0.35))
_BUDGET_LOCK = threading.Lock()  # check-and-take of both daily budgets (tools run in threads)

Key = tuple[int, int, int]  # (guild id, channel id, user id)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _reference_png(data: bytes) -> bytes:
    """An attached photo as a reference image: refused from the header when too large
    (ImageTooLarge), upright (EXIF), long side at most REF_SIDE, PNG. ValueError when it cannot
    be read."""
    with open_capped(data) as im:
        try:
            im.draft("RGB", (REF_SIDE, REF_SIDE))  # JPEG: decode at a reduced scale
            im.thumbnail((REF_SIDE, REF_SIDE), Image.Resampling.LANCZOS)
            img = ImageOps.exif_transpose(im)
            alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            return _png(img.convert("RGBA" if alpha else "RGB"))
        except Exception as e:
            raise ValueError("unreadable image") from e


def _jpeg(img: Image.Image, quality: int, scale: float = 1.0) -> bytes:
    """`img` as a JPEG at `quality`, its sides scaled by `scale` (never up)."""
    if scale < 1:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _total(files: list[tuple[str, bytes]]) -> int:
    return sum(len(d) for _, d in files)


def fit_batch(images: list[Image.Image], max_bytes: int) -> list[tuple[str, bytes]]:
    """The batch as `image-N.jpg` files that together take at most `max_bytes`: each image by
    `cards.to_jpeg` first, then the whole batch at each of SHRINK_STEPS (lower quality, then a
    smaller long side), and last the trailing images dropped (all of them when not even one
    fits)."""
    each = max_bytes // max(1, len(images))
    files = [
        (f"image-{i}.jpg", cards.to_jpeg(img, max_bytes=each)) for i, img in enumerate(images, 1)
    ]
    for quality, scale in SHRINK_STEPS:
        if _total(files) <= max_bytes:
            return files
        files = [
            (name, _jpeg(img, quality, scale)) for (name, _), img in zip(files, images, strict=True)
        ]
    while files and _total(files) > max_bytes:
        files.pop()
    return files


def overlay(
    img: Image.Image, headline: str, subtitle: str, accent: str, fonts: CardFonts, *, top: bool
) -> tuple[Image.Image, list[tuple[str, tuple[int, int, int, int]]]]:
    """`img` with the headline (white, centred, at most 2 lines) and the subtitle (in an accent
    pill) on a scrim band at the top (`top`) or the bottom; sizes follow the width. Returns the new
    image and the placed boxes [("headline" | "subtitle", (x0, y0, x1, y1))]."""
    from .promo import ACCENTS  # promo imports this module

    out = img.convert("RGB")  # a copy
    w, h = out.size
    margin, gap = round(0.06 * w), round(0.02 * w)
    content_w = w - 2 * margin
    head = cards.drawable(headline, fonts.bold)
    sub = cards.drawable(subtitle, fonts.bold)
    fitted = (
        cards.fit_text(head, fonts.bold, content_w, 2, (round(0.085 * w), round(0.04 * w)))
        if head
        else None
    )
    pad = (round(0.03 * w), round(0.012 * w))
    badge = (
        cards.measure_badge(sub, fonts.bold, (round(0.05 * w), round(0.03 * w)), content_w, pad)
        if sub
        else None
    )
    block = (fitted.height if fitted else 0) + (badge.h if badge else 0)
    block += gap if fitted and badge else 0
    placed: list[tuple[str, tuple[int, int, int, int]]] = []
    if not block:
        return out, placed
    if top:
        y = margin
        cards.scrim(out, 0, round((y + block + margin) * 1.5), 190, 0)
    else:
        y = h - margin - block
        cards.scrim(out, h - round((h - y + margin) * 1.5), h, 0, 190)
    if fitted:
        box = cards.draw_lines(
            out,
            margin,
            y,
            fitted,
            fonts.bold,
            fill="#FFFFFF",
            align="center",
            width=content_w,
            shadow=max(2, fitted.size // 12),
        )
        placed.append(("headline", box))
        y += fitted.height + gap
    if badge:
        fill, text_fill = ACCENTS.get(accent, ACCENTS["red"])
        radius = None if len(badge.fitted.lines) == 1 else gap  # a pill, or rounded when wrapped
        x = max(margin, (w - badge.w) // 2)
        box = cards.draw_badge(
            out, badge, x, y, fonts.bold, fill=fill, text_fill=text_fill, radius=radius
        )
        placed.append(("subtitle", box))
    return out, placed


class ImageMemory:
    """The last batch of images (PNG bytes, without overlay text) per person and channel, for
    `revise`. Process memory only (never written anywhere), `ttl_s` after the last batch, at most
    `max_entries` and `max_bytes` (least recently used dropped). Thread-safe: tools run in worker
    threads."""

    def __init__(
        self,
        ttl_s: float = 1800.0,
        max_entries: int = 50,
        clock: Callable[[], float] = time.monotonic,
        max_bytes: int = MEMORY_BYTES,
    ):
        self.ttl_s, self.max_entries, self.clock = ttl_s, max_entries, clock
        self.max_bytes = max_bytes
        self._items: OrderedDict[Key, tuple[float, list[bytes]]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _drop(self, key: Key) -> None:
        _, images = self._items.pop(key)
        self._bytes -= sum(len(b) for b in images)

    def put(self, key: Key, images: list[bytes]) -> None:
        images = list(images)
        with self._lock:
            if key in self._items:
                self._drop(key)
            self._items[key] = (self.clock(), images)
            self._bytes += sum(len(b) for b in images)
            while self._items and (
                len(self._items) > self.max_entries or self._bytes > self.max_bytes
            ):
                self._drop(next(iter(self._items)))

    def get(self, key: Key) -> list[bytes] | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            if self.clock() - hit[0] > self.ttl_s:
                self._drop(key)
                return None
            self._items.move_to_end(key)
            return list(hit[1])


def _as_int(value) -> int | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _error(message: str) -> dict:
    return {"status": "error", "error": message}


class DayBudget(Protocol):
    """Units used per key (guild or person) per day: `media.DailyBudget` (memory) or
    `imagegen_bot.StoredDailyBudget` (DocStore, survives restarts). A guild budget may also have
    `pair(person_budget, guild_id)`, returning `Budgets` that read and change both counts at
    once (or None)."""

    cap: float

    def used(self, key: int, day: str) -> float: ...

    def add(self, key: int, day: str, amount: float) -> None: ...

    def give_back(self, key: int, day: str, amount: float) -> None: ...


class Budgets(Protocol):
    """The guild's and the person's budget of one request, used together."""

    def counts(self, guild_id: int, user_id: int, day: str) -> tuple[float, float]: ...

    def take(self, guild_id: int, user_id: int, day: str, amount: int) -> tuple[int, int]: ...

    def give_back(self, guild_id: int, user_id: int, day: str, amount: int) -> None: ...


@dataclass(frozen=True)
class _Separate:
    """Two budgets with nothing shared (`media.DailyBudget`), each read and changed on its own
    (the caller holds _BUDGET_LOCK around `take`)."""

    guild: DayBudget
    person: DayBudget

    def counts(self, guild_id: int, user_id: int, day: str) -> tuple[float, float]:
        return self.guild.used(guild_id, day), self.person.used(user_id, day)

    def take(self, guild_id: int, user_id: int, day: str, amount: int) -> tuple[int, int]:
        """Take `amount` from both, trimmed to what both have left: (taken, left after)."""
        g, u = self.counts(guild_id, user_id, day)
        left = max(0, int(min(self.guild.cap - g, self.person.cap - u)))
        n = max(0, min(amount, left))
        if n:
            self.guild.add(guild_id, day, n)
            self.person.add(user_id, day, n)
        return n, left - n

    def give_back(self, guild_id: int, user_id: int, day: str, amount: int) -> None:
        self.guild.give_back(guild_id, day, amount)
        self.person.give_back(user_id, day, amount)


@dataclass
class ImageCtx:
    """One request's create_image tool, bound by the bot (guild, channel and author are never
    model input). `photos`: (attachment index, data) of this message's usable photos; the user
    budget is keyed by `user_id`."""

    guild_id: int
    user_id: int
    day: str
    photos: list[tuple[int, bytes]]
    memory: ImageMemory
    key: Key
    client: Any
    model: str
    guild_budget: DayBudget
    user_budget: DayBudget
    fonts: CardFonts
    max_bytes: int = OUTBOX_MAX_BYTES  # this reply's files together (the server's upload limit)
    outbox: list[tuple[str, bytes]] = field(default_factory=list)

    def budgets(self) -> Budgets:
        """Both budgets together: the guild budget's `pair` when it has one (one read per
        check, one read-modify-write per take), else each on its own."""
        pair = getattr(self.guild_budget, "pair", None)
        joint = pair(self.user_budget, self.guild_id) if callable(pair) else None
        return joint if joint is not None else _Separate(self.guild_budget, self.user_budget)

    def left(self) -> int:
        """Images this person may still make here today: the smaller of both budgets."""
        g, u = self.budgets().counts(self.guild_id, self.user_id, self.day)
        return max(0, int(min(self.guild_budget.cap - g, self.user_budget.cap - u)))

    def make(
        self,
        prompt: str = "",
        aspect: str = "1:1",
        count: int = 1,
        size: str = "1K",
        headline: str = "",
        subtitle: str = "",
        accent: str = "",
        file_index: int = -1,
        revise: int = 0,
    ) -> dict:
        from .promo import ACCENTS, clean  # promo imports this module

        prompt, headline, subtitle = clean(prompt), clean(headline), clean(subtitle)
        aspect, size, accent = clean(aspect), clean(size).upper(), clean(accent).casefold() or "red"
        n_asked, index, rev = _as_int(count), _as_int(file_index), _as_int(revise)
        if not prompt:
            return _error("prompt is required: describe the image in English")
        if len(prompt) > PROMPT_MAX:
            return _error(f"prompt is longer than {PROMPT_MAX} characters; shorten it")
        if not english_only(prompt):  # the model would draw the Thai text
            return _error("prompt must be English; put Thai text in headline/subtitle")
        if aspect not in ASPECTS:
            return _error(f"aspect must be one of: {', '.join(ASPECTS)}")
        if size not in SIZES:
            return _error(f"size must be one of: {', '.join(SIZES)}")
        if n_asked is None or not 1 <= n_asked <= MAX_IMAGES:
            return _error(f"count must be 1-{MAX_IMAGES}")
        if len(headline) > HEADLINE_MAX:
            return _error(f"headline is longer than {HEADLINE_MAX} characters; shorten it")
        if len(subtitle) > SUBTITLE_MAX:
            return _error(f"subtitle is longer than {SUBTITLE_MAX} characters; shorten it")
        if accent not in ACCENTS:
            return _error(f"accent must be one of: {', '.join(ACCENTS)}")
        if index is None:
            return _error("file_index must be a file number, or -1 for none")
        if rev is None or not 0 <= rev <= MAX_IMAGES:
            return _error(f"revise must be 0 (new images) or 1-{MAX_IMAGES}")
        if self.left() <= 0:  # one read, before decoding any photo; the take checks again
            return {"status": "limit", "note": NOTE_LIMIT}
        refs: list[bytes] = []
        if rev:
            last = self.memory.get(self.key)
            if not last:
                return _error("nothing to revise: no images from the last 30 minutes; use revise=0")
            if rev > len(last):
                return _error(f"revise must be 1-{len(last)} (the last batch had {len(last)})")
            refs.append(last[rev - 1])
        if index >= 0:
            data = next((d for i, d in self.photos if i == index), None)
            if data is None:
                return _error(f"file #{index} is not a usable photo (PNG, JPEG or WebP)")
            try:
                refs.append(_reference_png(data))
            except ImageTooLarge as e:
                return _error(f"{e}; ask for a smaller photo")
            except ValueError:
                return _error("the photo could not be read")
        budgets = self.budgets()
        with _BUDGET_LOCK:  # taken per image before the calls (one read-modify-write)
            n, left = budgets.take(self.guild_id, self.user_id, self.day, n_asked)
        if n <= 0:
            return {"status": "limit", "note": NOTE_LIMIT}
        t0 = time.monotonic()
        if self.client is None or not english_only(prompt):
            # generate_images makes no call then, so nothing is billed: the units go back. Every
            # real call stays charged (blocked, timed out or failed, Vertex may bill it).
            budgets.give_back(self.guild_id, self.user_id, self.day, n)
            left += n
            images, reason = [], "error"
        else:
            images, reason = generate_images(
                self.client, self.model, prompt, refs=refs, aspect=aspect, size=size, count=n
            )
        extra = ""
        try:
            files, kept = self._files(images, headline, subtitle, accent, aspect == "9:16")
        except Exception as e:
            if not (headline or subtitle):
                raise  # nothing to leave out; the images stay charged (they were billed)
            # billed images are never thrown away: send them without the text
            logger.warning("image overlay failed (%s): sending without text", type(e).__name__)
            files, kept = self._files(images, "", "", accent, aspect == "9:16")
            extra = NOTE_NO_TEXT
        if images and not files:
            reason = "too_large"
        if files:
            self.outbox[:] = files  # a second call in the same turn replaces the first batch
            self.memory.put(self.key, kept[: len(files)])
        logger.info(
            "image guild=%s made=%d asked=%d ms=%d",
            self.guild_id,
            len(files),
            n_asked,
            (time.monotonic() - t0) * 1000,
        )
        if not files:
            return {
                **_error(FAILED.get(reason, FAILED["error"])),
                "reason": reason,
                "count_made": 0,
                "images_left_today": left,
            }
        note = NOTE
        if n < n_asked:
            note += NOTE_TRIMMED.format(n=n, asked=n_asked)
        if len(images) < n:
            note += NOTE_FAILED
        if len(files) < len(images):
            note += NOTE_DROPPED.format(n=len(files), made=len(images))
        note += extra
        return {
            "status": "ok",
            "files": [name for name, _ in files],
            "count_made": len(files),
            "reason": reason,
            "images_left_today": left,
            "note": note,
        }

    def _files(
        self, images: list[Image.Image], headline: str, subtitle: str, accent: str, top: bool
    ) -> tuple[list[tuple[str, bytes]], list[bytes]]:
        """(the files to send, the clean PNGs to remember for `revise`): PNGs with the overlay
        text, or the whole batch as JPEGs that fit `max_bytes` (`fit_batch`, which may drop
        trailing images)."""
        files, kept, shown = [], [], []
        for i, img in enumerate(images, 1):
            small = img.copy()
            small.thumbnail((MEMORY_SIDE, MEMORY_SIDE), Image.Resampling.LANCZOS)
            kept.append(_png(small))  # the reference for `revise`: never the overlay text
            if headline or subtitle:
                img, _ = overlay(img, headline, subtitle, accent, self.fonts, top=top)
                data = _png(img)
            else:
                data = kept[-1] if small.size == img.size else _png(img)
            shown.append(img)
            files.append((f"image-{i}.png", data))
        if _total(files) > self.max_bytes:  # the whole batch, one format
            files = fit_batch(shown, self.max_bytes)
        return files, kept


def image_tools(ctx: ImageCtx) -> list[Callable]:
    """The function `_agent_kwargs` appends to `extra_tools` (its docstring is the tool
    description). Files are collected in `ctx.outbox` (`outbox` attribute: see `qa.tool_files`)."""

    def create_image(
        prompt: str = "",
        aspect: str = "1:1",
        count: int = 1,
        size: str = "1K",
        headline: str = "",
        subtitle: str = "",
        accent: str = "",
        file_index: int = -1,
        revise: int = 0,
    ) -> dict:
        """Create images from a description (illustrations, posters, product scenes, social posts). Use for
        "วาดรูป…", "สร้างภาพ…", "ทำโปสเตอร์…", "make an image of…". Write `prompt` IN ENGLISH, never
        describing text; put Thai words the user wants on the image in `headline` / `subtitle` (Salid
        typesets them). For a product photo the user attached, pass its number as `file_index`. To change
        one of the last results ("เอาแบบที่ 2 แต่โทนอุ่นขึ้น") pass `revise` = its number (1-4) with the new
        prompt; the last results are remembered for 30 minutes. Never depict real public figures or children.
        `file_index` and `revise` are only for photos of products, logos, food or places,
        never a photo of a real person: refuse that and offer an illustration instead.

        Args:
            prompt: English description, max 600 characters.
            aspect: "1:1" (default), "4:5", "9:16" or "16:9".
            count: 1-4 variations (default 1).
            size: "1K" (default) or "2K".
            headline: optional text on the image, max 40 characters, in the user's language.
            subtitle: optional second line, max 60 characters.
            accent: subtitle pill colour: red, orange, green, blue, gold or black.
            file_index: attached photo (a product, logo, food or place) to use as a reference; -1 = none.
            revise: number of a remembered result to edit; 0 = new images.
        """
        try:
            return ctx.make(
                prompt, aspect, count, size, headline, subtitle, accent, file_index, revise
            )
        except Exception:
            logger.exception("create_image failed")
            return _error("image creation failed; try again")

    create_image.outbox = ctx.outbox  # type: ignore[attr-defined]
    return [create_image]
