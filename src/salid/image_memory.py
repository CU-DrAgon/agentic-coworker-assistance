"""Image memory: a cheap multimodal model describes every image posted in a remembered channel and
the description (with the text found in the image) is stored in the message's log record, so
questions and day summaries can use it.

Blocking (run with asyncio.to_thread). Image bytes are only held in memory while describing and
are never stored. Failures raise; the caller keeps the plain "[image: filename]" descriptor and
logs a warning without any content.
"""

from __future__ import annotations

import json
import logging

from google.genai import types
from pydantic import BaseModel

from .llm import generate

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_TOKENS = 450
NOTE_CHARS = 1200
BACKFILL_IMAGES_PER_CHANNEL = 50
KINDS = "receipt/invoice, document, screenshot, chart/table, photo, meme, other"
_EXT_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "heic": "image/heic",
    "heif": "image/heif",
}

PROMPT = (
    "You describe one image posted in a team's Discord chat so it can be searched later. "
    "Reply as JSON with: description (1-2 sentences), text (ALL legible text transcribed "
    "verbatim, Thai or English, keeping numbers, amounts and dates exactly; at most about 600 "
    "characters, empty if none), type (one of: " + KINDS + "; a bank transfer slip, payment "
    "slip, tax invoice or bill is receipt/invoice). Do not guess unreadable text."
)


class ImageNote(BaseModel):
    description: str = ""
    text: str = ""
    type: str = "other"

    @property
    def is_receipt_like(self) -> bool:
        """The describer thinks this is a receipt / invoice / bill / transfer slip."""
        kind = (self.type or "").casefold()
        return any(w in kind for w in ("receipt", "invoice", "bill", "slip", "ใบเสร็จ", "สลิป"))


def image_mime(att) -> str | None:
    """The image MIME type of a Discord attachment (content_type image/*, else the file
    extension), or None when it is not an image."""
    ct = str(getattr(att, "content_type", "") or "").split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return ct
    name = str(getattr(att, "filename", "") or "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _EXT_MIME.get(ext)


def describe_image(client, model: str, data: bytes, mime: str) -> ImageNote:
    resp = generate(
        client,
        model,
        [types.Part.from_bytes(data=data, mime_type=mime), PROMPT],
        types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=ImageNote,
        ),
    )
    return ImageNote.model_validate(json.loads(resp.text))


def format_note(note: ImageNote) -> str:
    """ "[รูป: screenshot] description | ข้อความในรูป: text" (the text part only when there is some)."""
    kind = (note.type or "other").strip()[:30] or "other"
    out = f"[รูป: {kind}] {note.description.strip()}"
    if note.text.strip():
        out += f" | ข้อความในรูป: {' '.join(note.text.split())}"
    return out[:NOTE_CHARS]


def receipt_note(r) -> str:
    """Short text for an image the receipt pipeline already read (no second model call)."""
    total = f"{r.total:,.2f}".removesuffix(".00") if r.total is not None else "?"
    return (
        f"[ใบเสร็จ] {r.vendor or '?'} | ยอด {total} {r.currency or ''} | "
        f"{r.effective_date} | {r.category}"
    ).replace("  ", " ")
