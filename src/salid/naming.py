"""Recognising Salid's name in free text (Thai and Latin, misspellings included)."""

from __future__ import annotations

import re

# Thai spellings of สลิด/สะลิด/สลิต..., "ซาลิด" and the Latin forms. "สลิป" (a payment slip) has a
# different final consonant, so it never matches. Latin forms must not sit inside a longer word.
_WAKE_RE = re.compile(
    r"(?:ส[ะั]?ห?ลิ[ดตทธ]ร?[์๊๋]?|สลิ[ดตท][ร]?|ซ[าะ]\s?ลิ[ดต]"
    r"|(?<![a-z])(?:salid|saleed|salit|sa\s?lid)(?![a-z]))",
    re.I,
)
# Custom emoji (<:salid:123>) and links carry names that are not people talking to Salid.
_NOISE_RE = re.compile(r"<a?:\w+:\d+>|https?://\S+")


def mentions_name(text: str) -> bool:
    """True when Salid's name appears in `text`, with or without an @mention."""
    return _WAKE_RE.search(_NOISE_RE.sub(" ", text or "")) is not None


# Misheard names in SPEECH transcripts: สนิท (also "close / intimate"), สิริ / ซิริ / Siri. They count
# only as an address: with a greeting, a politeness particle, "น้อง", or alone at the start / end
# of the utterance, never in the middle of a sentence ("สนิทกันมาก").
# "โซรี่ / ซิรี่" are how the speech model writes "Siri" after "เฮ้". They count only right
# after a greeting (below), never alone.
_FUZZY = r"(?:สนิท|สิริ|ซิริ|ซีริ|siri)"
_FUZZY_GREETED = r"(?:โซรี่|โซรี|ซิรี่|ซิริ|ซีริ)"
_GREET = r"(?:หวัดดี|สวัสดี|ว่าไง|เฮ้|hey|hi|hello|ok)"
_PART = r"(?:ครับ|ค่ะ|คะ|นะ|จ๋า|จ้า)"
_FUZZY_RE = re.compile(
    rf"(?:{_GREET}\s*{_PART}?\s*(?P<a>{_FUZZY}|{_FUZZY_GREETED})(?![ก-๙a-z])"
    rf"|(?<![ก-๙a-z])(?P<b>{_FUZZY})\s*{_GREET}"
    rf"|น้อง\s?(?P<c>{_FUZZY})"
    rf"|(?<![ก-๙a-z])(?P<d>{_FUZZY})\s*{_PART}(?![ก-๙])"
    rf"|^[\s,.!?]*(?P<e>{_FUZZY})(?=[\s,.!?]|$)"
    rf"|(?:^|[\s,])(?P<f>{_FUZZY})[\s.!?]*$)",
    re.I,
)


def find_call(text: str) -> tuple[int, int] | None:
    """(start, end) of Salid's name in a spoken transcript, else None. The usual spellings always
    count; the risky mishearings (สนิท, สิริ, Siri...) only when used as an address."""
    text = text or ""
    m = _WAKE_RE.search(text)
    if m is not None:
        return m.start(), m.end()
    m = _FUZZY_RE.search(text)
    if m is None:
        return None
    g = next(k for k in "abcdef" if m.group(k) is not None)
    return m.start(g), m.end(g)


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
