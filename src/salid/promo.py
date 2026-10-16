"""Promo post kits: a product photo becomes three ready-to-post images (1:1, 4:5, 9:16).

The image model (Nano Banana, `settings.promo_image_model`) only restyles the scene around the
product: the product stays as photographed and the model is told never to draw text. All text
(headline, price, call to action, shop name) is typeset here with `salid.cards`, because Thai is
not a language the image models render reliably. One image-model call per kit at most; the daily
unit is taken before the call (blocked calls are billed too). When the call is blocked, fails or
the cap is reached, the kit is built from the original photo instead, so the user always gets
images. The photo and request of the last kit are kept in process memory only (revisions).
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
from typing import Any

import httpx
from google.genai import types
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel

from . import cards
from .cards import CardFonts
from .llm import generate
from .media import DailyBudget

logger = logging.getLogger(__name__)

ACCENTS: dict[str, tuple[str, str]] = {  # badge fill, badge text
    "red": ("#E53935", "#FFFFFF"),
    "orange": ("#E65100", "#FFFFFF"),
    "green": ("#2E7D32", "#FFFFFF"),
    "blue": ("#1E5EC8", "#FFFFFF"),
    "gold": ("#F2C14E", "#1A1A1A"),
    "black": ("#111111", "#FFFFFF"),
}
LIMITS = {"headline": 40, "scene": 300, "price": 20, "call_to_action": 24, "shop_name": 30}
MIN_SCENE = 3
PHOTO_MIMES = {"image/png", "image/jpeg", "image/webp"}
MAX_PHOTO_SIDE = 2048
LOW_QUALITY_SIDE = 400
PROMO_TIMEOUT_MS = 60_000
PROMO_RETRY_DELAYS = (3.0,)  # one retry: a stuck call ends in the fallback kit after ~2 min
BLOCKED = {
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    types.FinishReason.NO_IMAGE,
    types.FinishReason.IMAGE_RECITATION,
    types.FinishReason.IMAGE_OTHER,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SAFETY,
}
SCENE_PROMPT = (
    "Edit this product photo into a professional advertising photo.\n"
    "Keep the product exactly as it is: same shape, colours, label, logo, packaging, text printed "
    "on the product and quantity. Do not add, remove or redraw any part of the product.\n"
    "Change only the background, the surface it stands on, the lighting and small props.\n"
    "New scene: {scene}.\n"
    "Composition: vertical 9:16, product centred and fully visible in the middle half of the "
    "height, the top 20% and bottom 25% of the frame plain, calm background with no objects.\n"
    "No text, letters, numbers, logos, watermarks, price tags or signs anywhere in the new scene.\n"
    "No people, hands or faces."
)
_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff\ud800-\udfff]")


# ------------------------------------------------------------------ request
class PromoSpec(BaseModel):
    headline: str = ""
    scene: str = ""
    price: str = ""
    call_to_action: str = ""
    shop_name: str = ""
    accent: str = "red"


def clean(text: str) -> str:
    """Control and XML-invalid characters removed, whitespace collapsed."""
    return " ".join(_INVALID.sub("", str(text or "")).split())


def merge_spec(
    prev: PromoSpec | None,
    *,
    headline: str = "",
    scene: str = "",
    price: str = "",
    call_to_action: str = "",
    shop_name: str = "",
    accent: str = "",
    default_shop: str = "",
) -> PromoSpec:
    """The kit request: given fields win, empty ones keep `prev` (a revision) or take the default,
    "-" removes an optional field. Raises ValueError naming the field the agent must fix."""
    given = {
        "headline": headline,
        "scene": scene,
        "price": price,
        "call_to_action": call_to_action,
        "shop_name": shop_name,
        "accent": accent,
    }
    defaults = {"shop_name": clean(default_shop)[: LIMITS["shop_name"]], "accent": "red"}
    out: dict[str, str] = {}
    for name, raw in given.items():
        value = clean(raw)
        if value == "-":
            value = ""
        elif not value:
            value = getattr(prev, name) if prev is not None else defaults.get(name, "")
        out[name] = value
    out["accent"] = out["accent"].casefold() or "red"
    if out["accent"] not in ACCENTS:
        raise ValueError(f"accent must be one of: {', '.join(ACCENTS)}")
    for name, limit in LIMITS.items():
        if len(out[name]) > limit:
            raise ValueError(f"{name} is longer than {limit} characters; shorten it")
    if not out["headline"]:
        raise ValueError("headline is required (the main line on the image)")
    if len(out["scene"]) < MIN_SCENE:
        raise ValueError("scene is required: describe the setting around the product in English")
    return PromoSpec(**out)


# ------------------------------------------------------------------ photo
@dataclass
class Photo:
    jpeg: bytes
    short_side: int


def prepare_photo(data: bytes) -> Photo | None:
    """Upright (EXIF), transparency on white, long side at most 2048 px, JPEG; None if unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            img = ImageOps.exif_transpose(im)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                img = Image.new("RGB", rgba.size, "white")
                img.paste(rgba, mask=rgba.getchannel("A"))
            img = img.convert("RGB")
            img.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=90)
            return Photo(jpeg=buf.getvalue(), short_side=min(img.size))
    except Exception:
        return None


# ------------------------------------------------------------------ scene
def scene_prompt(scene: str) -> str:
    return SCENE_PROMPT.format(scene=scene)


def generate_scene(client, model: str, photo: Photo, scene: str) -> tuple[Image.Image | None, str]:
    """One Nano Banana edit of the photo into the new scene: (image, "") or (None, reason) with
    reason "blocked" (safety / no image), "timeout" or "error"."""
    if client is None:
        return None, "error"
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="9:16", image_size="2K"),
        http_options=types.HttpOptions(timeout=PROMO_TIMEOUT_MS),
    )  # no temperature / top_p / top_k / seed / logprobs: an API error on Nano Banana 2.1
    contents = [types.Part.from_bytes(data=photo.jpeg, mime_type="image/jpeg"), scene_prompt(scene)]
    try:
        resp = generate(client, model, contents, config, delays=PROMO_RETRY_DELAYS)
    except httpx.TimeoutException:
        logger.warning("promo scene timed out")
        return None, "timeout"
    except Exception as e:  # never log contents
        logger.warning("promo scene failed (%s)", type(e).__name__)
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
                logger.warning("promo scene image could not be decoded")
                return None, "error"
    return None, "blocked"


def take_unit(budget: DailyBudget, guild_id: int, day: str) -> str:
    """Take one AI-scene unit before the call: "" when taken, else "disabled" / "daily_limit"."""
    if budget.cap <= 0:
        return "disabled"
    if not budget.available(guild_id, day):
        return "daily_limit"
    budget.add(guild_id, day, 1)
    return ""


# ------------------------------------------------------------------ layouts
MASTER = (1536, 2752)  # the 9:16 scene the layouts are cut from (Nano Banana 2K)
MARGIN = 64  # side margin of every format; the content box is x in [64, 1016]
BADGE_PAD = (32, 16)
PRICE_RADIUS = 24
GAP = 24


@dataclass(frozen=True)
class Format:
    name: str
    size: tuple[int, int]
    safe_top: int
    safe_bottom: int
    headline: tuple[int, int]  # font sizes (max, min)
    price: tuple[int, int]
    cta: tuple[int, int]
    shop: tuple[int, int]


FORMATS: tuple[Format, ...] = (
    Format("promo-1x1.jpg", (1080, 1080), 48, 48, (80, 40), (64, 36), (40, 32), (32, 24)),
    Format("promo-4x5.jpg", (1080, 1350), 48, 48, (88, 44), (64, 36), (40, 32), (32, 24)),
    # Stories / Reels / TikTok draw their own buttons over the top 14% and bottom 20%
    Format("promo-9x16.jpg", (1080, 1920), 270, 384, (96, 48), (72, 40), (44, 32), (36, 28)),
)


@dataclass
class Placed:
    kind: str  # shop | headline | price | cta
    box: tuple[int, int, int, int]


def fallback_scene(photo: Photo) -> Image.Image:
    """The original photo on a blurred, darkened copy of itself, 9:16: the scene when the AI one is
    unavailable (product centred in the same band the AI prompt asks for)."""
    with Image.open(io.BytesIO(photo.jpeg)) as im:
        src = im.convert("RGB")
    canvas = cards.cover(src, MASTER).filter(ImageFilter.GaussianBlur(40))
    canvas = ImageEnhance.Brightness(canvas).enhance(0.65)
    box_w, box_h = MASTER[0] - 2 * 96, int(MASTER[1] * 0.75) - int(MASTER[1] * 0.20)
    fit = ImageOps.contain(src, (box_w, box_h), Image.Resampling.LANCZOS)
    canvas.paste(
        fit, (96 + (box_w - fit.width) // 2, int(MASTER[1] * 0.20) + (box_h - fit.height) // 2)
    )
    return canvas


def render_format(
    scene: Image.Image, spec: PromoSpec, fmt: Format, fonts: CardFonts
) -> tuple[Image.Image, list[Placed]]:
    """One format: the scene cut to size, scrims for contrast, then shop name, headline, price
    badge and call-to-action pill. Returns the image and where each text block was drawn."""
    w, h = fmt.size
    content_w = w - 2 * MARGIN
    img = cards.cover(scene, fmt.size)
    fill, text_fill = ACCENTS[spec.accent]
    headline = cards.drawable(spec.headline, fonts.bold)
    price = cards.drawable(spec.price, fonts.bold)
    cta = cards.drawable(spec.call_to_action, fonts.bold)
    shop = cards.drawable(spec.shop_name, fonts.regular)
    placed: list[Placed] = []

    def badge(text: str, sizes: tuple[int, int]) -> cards.Badge:
        return cards.measure_badge(text, fonts.bold, sizes, content_w, BADGE_PAD)

    def text(kind: str, f: cards.Fitted, path: str, y: int) -> None:
        """White text straight on the photo: a soft shadow keeps it readable on light areas."""
        box = cards.draw_lines(img, MARGIN, y, f, path, fill="#FFFFFF", shadow=max(2, f.size // 12))
        placed.append(Placed(kind, box))

    def pill(kind: str, b: cards.Badge, x: int, y: int, radius: int | None = None) -> None:
        box = cards.draw_badge(
            img, b, x, y, fonts.bold, fill=fill, text_fill=text_fill, radius=radius
        )
        placed.append(Placed(kind, box))

    if fmt.safe_top > 48:  # 9:16: everything stacks down from under the top safe zone
        cards.scrim(img, 0, int(0.40 * h), 170, 0)
        cards.scrim(img, int(0.60 * h), h, 0, 190)
        y = fmt.safe_top + GAP
        if shop:
            f = cards.fit_text(shop, fonts.regular, content_w, 1, fmt.shop)
            text("shop", f, fonts.regular, y)
            y += f.height + 12
        f = cards.fit_text(headline, fonts.bold, content_w, 2, fmt.headline)
        text("headline", f, fonts.bold, y)
        y += f.height + GAP
        if price:
            b = badge(price, fmt.price)
            pill("price", b, w - MARGIN - b.w, y, PRICE_RADIUS)
        if cta:
            b = badge(cta, fmt.cta)
            pill("cta", b, (w - b.w) // 2, h - fmt.safe_bottom - 40 - b.h)
        return img, placed

    # feed (4:5, 1:1): price top-right, headline and a last row (shop | call to action) at the bottom
    cards.scrim(img, int(0.55 * h), h, 0, 190)
    if price:
        b = badge(price, fmt.price)
        pill("price", b, w - MARGIN - b.w, fmt.safe_top, PRICE_RADIUS)
    bottom = h - fmt.safe_bottom
    cta_b = badge(cta, fmt.cta) if cta else None
    shop_f = None
    if shop:
        room = content_w - (cta_b.w + GAP if cta_b else 0)
        shop_f = cards.fit_text(shop, fonts.regular, room, 1, fmt.shop)
    row_h = max(cta_b.h if cta_b else 0, shop_f.height if shop_f else 0)
    row_top = bottom - row_h
    if cta_b:
        pill("cta", cta_b, w - MARGIN - cta_b.w, bottom - cta_b.h)
    if shop_f:
        text("shop", shop_f, fonts.regular, row_top + (row_h - shop_f.height) // 2)
    f = cards.fit_text(headline, fonts.bold, content_w, 2, fmt.headline)
    text("headline", f, fonts.bold, (row_top - GAP if row_h else bottom) - f.height)
    return img, placed


def render_kit(scene: Image.Image, spec: PromoSpec, fonts: CardFonts) -> list[tuple[str, bytes]]:
    """The three JPEGs, in FORMATS order."""
    return [(f.name, cards.to_jpeg(render_format(scene, spec, f, fonts)[0])) for f in FORMATS]


# ------------------------------------------------------------------ revision memory
Key = tuple[int, int, int]  # (guild id, channel id, user id)


class PromoMemory:
    """The last kit's photo and request per person and channel, for revisions without a new
    upload. Process memory only (never written anywhere), `ttl_s` after the last kit, at most
    `max_entries` (least recently used dropped). Thread-safe: tools run in worker threads."""

    def __init__(
        self,
        ttl_s: float = 1800.0,
        max_entries: int = 50,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl_s, self.max_entries, self.clock = ttl_s, max_entries, clock
        self._items: OrderedDict[Key, tuple[float, Photo, PromoSpec]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: Key, photo: Photo, spec: PromoSpec) -> None:
        with self._lock:
            self._items[key] = (self.clock(), photo, spec)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, key: Key) -> tuple[Photo, PromoSpec] | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            if self.clock() - hit[0] > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return hit[1], hit[2]


# ------------------------------------------------------------------ the agent tool
NOTE_ALWAYS = (
    "Reply in the user's language: say the three images are attached (1:1 for feed and LINE, 4:5 "
    "for Facebook/Instagram feed, 9:16 for Stories, Reels and TikTok), then give a ready-to-paste "
    "caption with 3-5 hashtags. Do not list the overlay text again."
)
NOTE_AI = (
    " Mention briefly that the background was made with AI and the product is their own photo."
)
NOTE_FALLBACK = {
    "blocked": " The AI scene was blocked by safety filters, so the kit uses the original photo; "
    "say so and suggest a different scene.",
    "error": " The AI scene service failed, so the kit uses the original photo; say so and that "
    "they can try again.",
    "daily_limit": " Today's AI scene limit for this server is used up, so the kit uses the "
    "original photo; say so.",
    "disabled": " AI scenes are turned off here, so the kit uses the original photo.",
}
NOTE_FALLBACK["timeout"] = NOTE_FALLBACK["error"]
NOTE_SMALL = (
    " The photo is small (under 400 px); suggest sending a larger photo for sharper results."
)


@dataclass
class PromoCtx:
    """One request's promo tool, bound by the bot (guild, channel and author are never model
    input). `photos`: (attachment index, data) of this message's usable photos."""

    guild_id: int
    day: str
    default_shop: str
    photos: list[tuple[int, bytes]]
    memory: PromoMemory
    key: Key
    client: Any
    model: str
    budget: DailyBudget
    fonts: CardFonts
    outbox: list[tuple[str, bytes]] = field(default_factory=list)

    def make_kit(
        self,
        headline: str = "",
        scene: str = "",
        price: str = "",
        call_to_action: str = "",
        shop_name: str = "",
        accent: str = "",
        file_index: int = -1,
    ) -> dict:
        prev: PromoSpec | None = None
        if file_index is not None and file_index >= 0:
            data = next((d for i, d in self.photos if i == file_index), None)
            if data is None:
                return {
                    "status": "error",
                    "error": f"file #{file_index} is not a usable photo (PNG, JPEG or WebP)",
                }
            photo = prepare_photo(data)
        elif self.photos:
            photo = prepare_photo(self.photos[0][1])
        else:
            remembered = self.memory.get(self.key)
            if remembered is None:
                return {
                    "status": "no_photo",
                    "note": "Ask the user to send the product photo with the request.",
                }
            photo, prev = remembered
        if photo is None:
            return {"status": "error", "error": "the photo could not be read"}
        try:
            spec = merge_spec(
                prev,
                headline=headline,
                scene=scene,
                price=price,
                call_to_action=call_to_action,
                shop_name=shop_name,
                accent=accent,
                default_shop=self.default_shop,
            )
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        t0 = time.monotonic()
        reason = take_unit(self.budget, self.guild_id, self.day)
        image = None
        if not reason:
            image, reason = generate_scene(self.client, self.model, photo, spec.scene)
        t1 = time.monotonic()
        files = render_kit(image if image is not None else fallback_scene(photo), spec, self.fonts)
        self.outbox[:] = files  # a second call in the same turn replaces the first kit
        self.memory.put(self.key, photo, spec)
        logger.info(
            "promo kit guild=%s outcome=%s model_ms=%d render_ms=%d bytes=%d",
            self.guild_id,
            "ai" if image is not None else f"fallback:{reason}",
            (t1 - t0) * 1000,
            (time.monotonic() - t1) * 1000,
            sum(len(d) for _, d in files),
        )
        note = NOTE_ALWAYS + (NOTE_AI if image is not None else NOTE_FALLBACK.get(reason, ""))
        if photo.short_side < LOW_QUALITY_SIDE:
            note += NOTE_SMALL
        return {
            "status": "ok",
            "files": [name for name, _ in files],
            "ai_scene": image is not None,
            "reason": reason,
            "kits_left_today": max(
                0, int(self.budget.cap - self.budget.used(self.guild_id, self.day))
            ),
            "note": note,
        }


def promo_tools(ctx: PromoCtx) -> list[Callable]:
    """The function `_agent_kwargs` appends to `extra_tools` (its docstring is the tool
    description). Files are collected in `ctx.outbox` (`outbox` attribute: see `qa.tool_files`)."""

    def make_promo_kit(
        headline: str = "",
        scene: str = "",
        price: str = "",
        call_to_action: str = "",
        shop_name: str = "",
        accent: str = "",
        file_index: int = -1,
    ) -> dict:
        """Make a promo post kit from the product photo the user sent: the product stays exactly as
        photographed, AI creates a new scene around it, and Salid typesets the text, giving three images
        (1:1, 4:5, 9:16). Use for requests like "ทำโพสต์โปรจากรูปนี้", "ทำภาพโฆษณาลาเต้ 59 บาท",
        "make an ad for this". For a change to the last kit ("ขอโทนอุ่นกว่านี้", "เปลี่ยนราคาเป็น 49")
        pass only what changes; the photo and the other fields are remembered for 30 minutes.

        Args:
            headline: main line on the image, max 40 characters, short and catchy, in the user's
                language; put a space between phrases (Thai too) so it can wrap onto two lines.
            scene: the setting to create around the product, IN ENGLISH, max 300 characters, e.g.
                "wooden cafe table, soft morning window light, a few coffee beans". Never describe text.
            price: optional, max 20 characters, e.g. "59.-" or "ลด 20%". "-" removes it.
            call_to_action: optional, max 24 characters, e.g. "สั่งเลยวันนี้". "-" removes it.
            shop_name: optional, max 30 characters; default is the server name. "-" removes it.
            accent: badge colour: red (default), orange, green, blue, gold or black.
            file_index: which attached file (the number in its label); -1 = the first photo.
        """
        try:
            return ctx.make_kit(
                headline, scene, price, call_to_action, shop_name, accent, file_index
            )
        except Exception:
            logger.exception("make_promo_kit failed")
            return {"status": "error", "error": "the promo kit failed; try again"}

    make_promo_kit.outbox = ctx.outbox  # type: ignore[attr-defined]
    return [make_promo_kit]
