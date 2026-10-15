"""Recognising Salid's name in free text (Thai and Latin, misspellings included)."""

from __future__ import annotations

import re

# Thai spellings of สลิด/สะลิด/สลิต..., "ซาลิด" and the Latin forms. "สลิป" (a payment slip) has a
# different final consonant, so it never matches. Latin forms must not sit inside a longer word.
_WAKE_RE = re.compile(
    r"(?:ส[ะั]?ลิ[ดตทธ]ร?[์๊๋]?|สลิ[ดตท][ร]?|ซา\s?ลิด"
    r"|(?<![a-z])(?:salid|saleed|salit|sa\s?lid)(?![a-z]))",
    re.I,
)
# Custom emoji (<:salid:123>) and links carry names that are not people talking to Salid.
_NOISE_RE = re.compile(r"<a?:\w+:\d+>|https?://\S+")


def mentions_name(text: str) -> bool:
    """True when Salid's name appears in `text`, with or without an @mention."""
    return _WAKE_RE.search(_NOISE_RE.sub(" ", text or "")) is not None
