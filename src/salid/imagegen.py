"""Image generation with Nano Banana (`settings.image_model`): the one image-model call shared by
promo kits, `create_image` and the video ad stills, and the guards every caller needs.

- `generate_images`: `count` parallel calls (the model returns one image per call), one attempt
  each (no retry: exactly one bill per image), person and prominent-people settings fixed here,
  no sampling parameters (an API error on Nano Banana 2.1).
- Blocked = a blocked prompt, a blocked finish reason or an answer without an image.
- `english_only`: prompts must never carry Thai (the model would draw it); Thai text is typeset
  by Salid with `salid.cards`.
- `open_capped`: the decode guard for images decoded in the shared bot process (2 GiB).

Never logs prompts or images.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx
from google.genai import types
from PIL import Image

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
    "blocked" (safety / no image), "timeout" or "error"."""
    if client is None or count < 1:
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
