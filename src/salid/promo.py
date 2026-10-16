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
from dataclasses import dataclass

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
    for field, raw in given.items():
        value = clean(raw)
        if value == "-":
            value = ""
        elif not value:
            value = getattr(prev, field) if prev is not None else defaults.get(field, "")
        out[field] = value
    out["accent"] = out["accent"].casefold() or "red"
    if out["accent"] not in ACCENTS:
        raise ValueError(f"accent must be one of: {', '.join(ACCENTS)}")
    for field, limit in LIMITS.items():
        if len(out[field]) > limit:
            raise ValueError(f"{field} is longer than {limit} characters; shorten it")
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
