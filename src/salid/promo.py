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
from PIL import Image, ImageOps
from pydantic import BaseModel

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
