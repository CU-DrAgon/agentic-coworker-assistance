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


_LEAD_JUNK = " \t\n,.:;!?-–—…/@"
_NAME_AT_START = re.compile(rf"^(?:{_WAKE_RE.pattern})", _WAKE_RE.flags)
# Politeness / filler words that may trail a typed command ("เข้าประชุมหน่อยนะครับ", "be quiet please").
_TRAIL_POLITE = re.compile(
    r"(?:[\s,.!?…]+|^)(?:ครับ|ค่ะ|คะ|นะ|จ้ะ|จ๊ะ|จ้า|หน่อย|ด้วย|เถอะ|ซิ|เลย|please|pls|plz)$"
    r"|(?:ครับ|ค่ะ|คะ|นะ|จ้ะ|จ๊ะ|จ้า|หน่อย|ด้วย|เถอะ|เลย)$",
    re.I,
)


def strip_leading_name(text: str) -> tuple[str, bool]:
    """Remove Salid's name from the START of `text` ("Salid เข้าประชุมหน่อย" -> "เข้าประชุมหน่อย").
    Returns (rest, True) when a name was removed, else (text, False). Does not touch trailing
    words; see `strip_trailing_polite` for those."""
    s = (text or "").lstrip(_LEAD_JUNK)
    m = _NAME_AT_START.match(s)
    if not m:
        return text, False
    return s[m.end() :].lstrip(_LEAD_JUNK), True


def strip_trailing_polite(text: str) -> str:
    """Drop trailing politeness words (หน่อย ด้วย ครับ ค่ะ นะ please ...), repeatedly."""
    s = (text or "").strip()
    while True:
        new = _TRAIL_POLITE.sub("", s).rstrip(" \t\n,.!?…")
        if new == s or not new:
            return new or s
        s = new


def command_text(text: str) -> tuple[str, bool]:
    """Text to feed the typed-command parsers: a leading name and trailing politeness removed.
    The bool says whether a leading name was present (an untagged command needs one)."""
    rest, named = strip_leading_name(text)
    return strip_trailing_polite(rest), named
