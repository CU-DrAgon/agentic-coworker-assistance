"""One company language, personal languages added for the people involved (pure, no I/O).

Two levels. The *company language* (the server setting chosen with `language_ask` /
`set_server_language`) is THE language of everything Salid says and produces. Each person may set a
*personal language* (profile); unset = the company language. A message addressed to people carries
the company language FIRST, then the personal language of each person involved when it differs, so
both sides understand. Documents / reports / minutes / `/pm` stay in the company language only.

An executive can force "company language only" (org policy `language_mode`, or per meeting): then
`langs_for` returns just the company language and personal languages are ignored. Listening never
changes (Thai and English are always understood); only the settings decide what Salid says, never
the language somebody happened to write or speak.

An UNCHOSEN server (company `auto`) keeps today's behaviour: `langs_for` returns `["auto"]` and
callers follow the words people write, until the language question has been answered.
"""

from __future__ import annotations

import re

from .i18n import CATALOG, DEFAULT_LANGUAGE, LANG_NAMES, LANGS, detect_lang, lang_name, t

MODE_NOTICES = (
    True  # the answering-mode statements (join, first reply of the day); tests switch off
)
MAX_LANGS = 3  # at most three languages in one message
MODES = ("audience", "company_only")
DEFAULT_MODE = "audience"

CATALOG.update(
    {
        "aud_mode_one": {
            "th": "สวัสดีค่ะ สลิดจะจดและตอบเป็นภาษา{lname}นะคะ",
            "en": "Hi, I'll take notes and answer in {lname}.",
        },
        "aud_mode_multi": {
            "th": "สวัสดีค่ะทุกคน ในห้องนี้มีคนที่ใช้ภาษา{rest} สลิดจะตอบเป็นภาษา{first}แล้วตามด้วยภาษา{rest}นะคะ",
            "en": "Hi everyone. Someone here prefers {rest}, so I'll answer in {first} and then in {rest}.",
        },
        "aud_mode_only": {
            "th": "ห้องนี้ตอบภาษา{lname}อย่างเดียวค่ะ",
            "en": "I'll answer in {lname} only in this room.",
        },
        "aud_adding": {
            "th": "เพิ่มภาษา{lname}ตั้งแต่นี้ไป",
            "en": "adding {lname} from now on",
        },
        "aud_change_hint": {
            "th": "ถ้าอยากเปลี่ยนภาษา บอกสลิดว่า “คุยกับฉันเป็นภาษาอังกฤษ” ได้เลยค่ะ",
            "en": 'To change the language, tell Salid: "talk to me in Thai".',
        },
        "aud_and": {"th": " และ ", "en": " and "},
        "aud_personal_set": {
            "th": "ได้ค่ะ จากนี้สลิดจะตอบคุณเป็นภาษา{lname}ต่อจากภาษาบริษัทนะคะ",
            "en": "Done. From now on I'll add {lname} after the company language when I talk to you.",
        },
        "aud_personal_same": {
            "th": "ภาษา{lname}เป็นภาษาบริษัทอยู่แล้วค่ะ ไม่ต้องเพิ่มอะไร",
            "en": "{lname} is already the company language, so nothing to add.",
        },
        "aud_personal_unset": {
            "th": "ได้ค่ะ จากนี้สลิดจะใช้ภาษาบริษัทกับคุณอย่างเดียวนะคะ",
            "en": "Done. From now on I'll use only the company language with you.",
        },
        "aud_personal_off": {
            "th": "องค์กรนี้ปิดการตั้งค่าส่วนตัวไว้ค่ะ ใช้ภาษาบริษัทอย่างเดียว ถามแอดมินได้นะคะ",
            "en": "This organisation turned personal settings off, so everyone uses the company language. Ask an admin.",
        },
        "aud_only_denied": {
            "th": "เฉพาะผู้บริหารหรือแอดมินเท่านั้นที่ตั้งเรื่องนี้ได้ค่ะ",
            "en": "Only an executive or admin can set this.",
        },
        "aud_mode_set_only": {
            "th": "รับทราบค่ะ ต่อไปนี้สลิดตอบเป็นภาษาบริษัท ({lname}) อย่างเดียว",
            "en": "Understood. From now on I'll answer only in the company language ({lname}).",
        },
        "aud_mode_set_audience": {
            "th": "รับทราบค่ะ สลิดจะตอบเป็นภาษาบริษัท ({lname}) แล้วเพิ่มภาษาส่วนตัวของแต่ละคนให้",
            "en": "Understood. I'll answer in the company language ({lname}) and add each person's own language.",
        },
        "aud_meet_set_only": {
            "th": "ประชุมนี้ตอบภาษา{lname}อย่างเดียวค่ะ",
            "en": "This meeting: I'll answer in {lname} only.",
        },
        "aud_meet_set_audience": {
            "th": "ประชุมนี้ใช้ภาษา{lname}แล้วเพิ่มภาษาของผู้ร่วมประชุมค่ะ",
            "en": "This meeting: {lname} plus the languages of the people in the room.",
        },
        "aud_meet_denied": {
            "th": "เฉพาะผู้เชิญสลิดหรือผู้บริหารในห้องเท่านั้นที่ตั้งภาษาของการประชุมได้ค่ะ",
            "en": "Only whoever invited Salid, or an executive in the room, can set the meeting's language.",
        },
        "aud_meet_none": {
            "th": "ตอนนี้ไม่มีการประชุมที่คุณอยู่ในห้องค่ะ",
            "en": "There is no meeting running in a room you are in.",
        },
    }
)


# ----------------------------------------------------------------------------- the language set
def company_lang(settings) -> str:
    """The server's chosen main language, else `auto` (settings: GuildSettings or its string)."""
    if isinstance(settings, str):
        return settings if settings in LANGS else "auto"
    value = getattr(settings, "language_setting", None)
    return value if value in LANGS else "auto"


def person_lang(profile, *, allowed: bool = True) -> str:
    """A person's own language, "" when unset, `auto`, or personal settings are off."""
    if not allowed:
        return ""
    lang = profile if isinstance(profile, str) else getattr(profile, "language", None)
    return lang if lang in LANG_NAMES else ""


def valid_mode(mode: str | None) -> str:
    return mode if mode in MODES else DEFAULT_MODE


def langs_for(
    company: str, *people_langs: str | None, limit: int = MAX_LANGS, mode: str = DEFAULT_MODE
) -> list[str]:
    """Company language first, then the distinct personal languages that differ from it, capped at
    `limit`. `auto` company -> `["auto"]` (today's path); `company_only` -> `[company]`."""
    if company not in LANGS or company == "auto":
        return ["auto"]
    out = [company]
    if valid_mode(mode) == "company_only":
        return out
    for lang in people_langs:
        if lang and lang in LANG_NAMES and lang not in out:
            out.append(lang)
        if len(out) >= max(1, limit):
            break
    return out


def voice_langs(
    company: str, attendees, *, limit: int = MAX_LANGS, mode: str = DEFAULT_MODE
) -> list[str]:
    """The languages of a voice room: company + the personal languages of the people in it now.
    `attendees`: profiles or plain language strings (None / "" = the company language)."""
    return langs_for(company, *(person_lang(a) for a in attendees), limit=limit, mode=mode)


def primary(langs: list[str]) -> str:
    return langs[0] if langs else DEFAULT_LANGUAGE


def multi(langs: list[str]) -> bool:
    return len(langs) > 1 and "auto" not in langs


def added(old: list[str], new: list[str]) -> list[str]:
    """Languages in `new` that `old` did not have."""
    return [x for x in new if x not in old]


# ----------------------------------------------------------------------------- rendering
def _render_lang(lang: str) -> str:
    return lang if lang in LANGS and lang != "auto" else DEFAULT_LANGUAGE


def tt(key: str, langs: list[str], **kw) -> str:
    """`t(key, lang)` for each language joined with a blank line; identical renderings are
    dropped (a key that is the same in every language is shown once)."""
    seen: list[str] = []
    out: list[str] = []
    for lang in langs or [DEFAULT_LANGUAGE]:
        r = _render_lang(lang)
        # a callable value is asked for its text in each language (e.g. a language's name)
        text = t(key, r, **{k: (v(r) if callable(v) else v) for k, v in kw.items()})
        if text not in seen:
            seen.append(text)
            out.append(text if not out else translation_label(lang) + text)
    return "\n\n".join(out)


def translation_label(lang: str) -> str:
    """The English header that marks a second-language block as a translation, so a reader who
    does not know that script sees it is a translation and not a different message."""
    return f"*({lang_name(lang, 1)} translation)*\n"


def render(fn, langs: list[str]) -> str:
    """`fn(lang)` for each language of the audience, joined like `tt` (identical results once).
    For messages built by a renderer that takes a language (cards, reports, lines)."""
    blocks = {lang: fn(_render_lang(lang)) for lang in dict.fromkeys(langs or [DEFAULT_LANGUAGE])}
    return join_blocks(blocks, list(blocks))


def name_in(code: str):
    """For `tt`: the name of language `code` as written in the rendering language."""
    return lambda render: lang_name(code, 2 if render == "th" else 1)


def join_blocks(blocks: dict[str, str], langs: list[str]) -> str:
    """Per-language texts (already formatted) in audience order, identical ones once. A language
    without a block is skipped; no match at all -> the first block available."""
    seen: list[str] = []
    out: list[str] = []
    for lang in langs:
        text = blocks.get(lang)
        if text and text not in seen:
            seen.append(text)
            out.append(text if not out else translation_label(lang) + text)
    if not out and blocks:
        out.append(next(iter(blocks.values())))
    return "\n\n".join(out)


def _names(langs: list[str], render: str) -> list[str]:
    which = 2 if render == "th" else 1
    return [lang_name(x, which) for x in langs]


def _name_list(langs: list[str], render: str) -> str:
    names = _names(langs, render)
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + t("aud_and", render) + names[-1]


def mode_line(
    langs: list[str], line_langs: list[str] | None = None, *, company_only: bool = False
) -> str:
    """The announcement of the answering mode (once, on joining): one sentence per language of
    `line_langs` (default `langs`), company language first."""
    langs = [x for x in langs if x != "auto"] or [DEFAULT_LANGUAGE]
    lines = line_langs or langs
    blocks: list[str] = []
    for render in lines:
        r = _render_lang(render)
        if company_only or len(langs) == 1:
            key = "aud_mode_only" if company_only else "aud_mode_one"
            text = t(key, r, lname=lang_name(langs[0], 2 if r == "th" else 1))
        else:
            text = t(
                "aud_mode_multi",
                r,
                first=_name_list(langs[:1], r),
                rest=_name_list(langs[1:], r),
            )
        if text not in blocks:
            blocks.append(text)
    return "\n".join(blocks)


def mode_line_with_hint(langs: list[str], *, company_only: bool = False) -> str:
    """`mode_line` plus how to change it, for a private channel's first message."""
    base = [x for x in langs if x != "auto"] or [DEFAULT_LANGUAGE]
    hint = tt("aud_change_hint", base[:1])
    return mode_line(langs, company_only=company_only) + "\n" + hint


def adding_line(new_langs: list[str]) -> str:
    """The text line when a person joins and the set grows: English then Thai, not spoken."""
    en = t("aud_adding", "en", lname=_name_list(new_langs, "en"))
    th = t("aud_adding", "th", lname=_name_list(new_langs, "th"))
    return f"{en} / {th}"


def second_language_rule(langs: list[str], prompt_lang: str = "en") -> str:
    """The prompt sentence telling the model to repeat its answer in the audience's other
    languages ("" with one language or `auto`)."""
    if not multi(langs):
        return ""
    if prompt_lang == "th":
        first = lang_name(langs[0], 2)
        rest = " และ ".join(lang_name(x, 2) for x in langs[1:])
        return (
            f"\nผู้ฟังคำตอบนี้ใช้หลายภาษา: ตอบเป็นภาษา{first}ก่อน แล้วเว้นบรรทัดว่างและตอบซ้ำเป็นภาษา{rest} "
            "(ถ้ามีหลายภาษา แยกเป็นย่อหน้าตามลำดับ) คำตอบสั้นให้ซ้ำทั้งหมด คำตอบยาวให้ภาษาถัดไปเฉพาะประเด็นสำคัญ "
            "ห้ามผสมสองภาษาในย่อหน้าเดียว และห้ามตัดสินใจเปลี่ยนภาษาเองจากภาษาที่ผู้ใช้พิมพ์มา"
        )
    first = lang_name(langs[0])
    rest = " and then ".join(lang_name(x) for x in langs[1:])
    return (
        f"\nThe people who will read or hear this answer use more than one language. Answer in {first} "
        f"first, then repeat it in {rest} after a blank line (one paragraph block per language, in "
        "that order). Short answers: repeat them in full. Long answers: in the later language(s) give "
        "only the key points. Never mix two languages inside one paragraph, and never choose a "
        "language yourself from the language the person wrote in."
    )


def company_rule(company: str, prompt_lang: str = "en") -> str:
    """The hard rule that the settings, not the person's words, decide the language."""
    if company in ("", "auto"):
        return ""
    if prompt_lang == "th":
        return (
            f"กฎเหล็ก: ตอบเป็นภาษา{lang_name(company, 2)}เสมอ ไม่ว่าผู้ใช้จะพิมพ์หรือพูดภาษาอะไรมา "
            "เปลี่ยนภาษาได้เฉพาะเมื่อผู้ใช้ขอชัดเจน (เช่น “ตอบเป็นภาษาไทย”) เท่านั้น"
        )
    return (
        f"Hard rule: reply in {lang_name(company)} whatever language the person writes or speaks in; "
        'change language only when they explicitly ask for it (e.g. "answer in Thai").'
    )


# ----------------------------------------------------------------------------- speaking blocks
def split_blocks(text: str, langs: list[str]) -> list[tuple[str, str]]:
    """A bilingual answer as (language, text) pieces for text-to-speech: paragraphs, each given the
    language its script shows when that language is in `langs`, else the company language."""
    pieces: list[tuple[str, str]] = []
    company = langs[0] if langs else DEFAULT_LANGUAGE
    for para in re.split(r"\n\s*\n", text or ""):
        para = para.strip()
        if not para:
            continue
        seen = detect_lang(para)
        lang = seen if seen in langs else company
        if pieces and pieces[-1][0] == lang:
            pieces[-1] = (lang, pieces[-1][1] + "\n\n" + para)
        else:
            pieces.append((lang, para))
    return pieces


# ----------------------------------------------------------------------------- typed forms
_NAME_TO_CODE: dict[str, str] = {}
for _code, (_native, _english, _thai) in LANG_NAMES.items():
    for _n in (_native, _english, _thai, f"ภาษา{_thai}"):
        _NAME_TO_CODE[_n.casefold()] = _code
_NAME_TO_CODE.update({"thai": "th", "english": "en", "ไทย": "th", "อังกฤษ": "en"})
_LANG_ALT = "|".join(sorted((re.escape(n) for n in _NAME_TO_CODE), key=len, reverse=True))
_POLITE = r"(?:ครับ|ค่ะ|คะ|นะ|หน่อย|ด้วย|เลย|please|pls)?"
_TAIL = rf"(?:\s*{_POLITE})*[\s.!?]*$"
_NAME = r"(?:@?\S*salid\S*|สลิด|สะลิด|ซาลิด)"
_HEAD = rf"^\s*(?:@\S+\s+|(?:hey\s+|hi\s+)?{_NAME}[\s,:]+)?"
_PERSONAL_SET = [
    re.compile(
        rf"{_HEAD}(?:คุย|พูด|ตอบ|ใช้)\s?กับ\s?(?:ฉัน|ผม|หนู|เรา|ดิฉัน)\s?(?:เป็น|ด้วย)?\s?(?:ภาษา\s?)?"
        rf"(?P<l>{_LANG_ALT}){_TAIL}",
        re.I,
    ),
    re.compile(
        rf"{_HEAD}(?:talk|speak|reply|answer|respond)\s+(?:to|with)\s+me\s+in\s+(?P<l>{_LANG_ALT}){_TAIL}",
        re.I,
    ),
    re.compile(
        rf"{_HEAD}my\s+(?:personal\s+)?language\s+(?:is|=|:)\s*(?P<l>{_LANG_ALT}){_TAIL}", re.I
    ),
    re.compile(
        rf"{_HEAD}ภาษาของ\s?(?:ฉัน|ผม|หนู|เรา)\s?(?:คือ|เป็น)\s?(?:ภาษา\s?)?(?P<l>{_LANG_ALT}){_TAIL}"
    ),
]
_PERSONAL_UNSET = re.compile(
    rf"{_HEAD}(?:ใช้ภาษาบริษัท(?!อย่างเดียว)|ใช้ภาษากลาง|use\s+the\s+company\s+language|"
    rf"use\s+company\s+language|reset\s+my\s+language|ล้างภาษาของ(?:ฉัน|ผม|หนู)){_TAIL}",
    re.I,
)
_ONLY = re.compile(
    rf"{_HEAD}(?:(?:ใช้|ตอบ)?ภาษาบริษัทอย่างเดียว|ใช้ภาษาเดียว|ตอบภาษาเดียว|"
    rf"company\s+language\s+only|(?:{_LANG_ALT})\s+only|only\s+(?:the\s+)?company\s+language|"
    rf"one\s+language\s+only){_TAIL}",
    re.I,
)
_ALLOW = re.compile(
    rf"{_HEAD}(?:ตอบหลายภาษาได้|ตอบได้หลายภาษา|ใช้สองภาษา|ใช้หลายภาษา|allow\s+personal\s+languages|"
    rf"both\s+languages|multiple\s+languages|use\s+both\s+languages){_TAIL}",
    re.I,
)
_MEETING = re.compile(
    r"(?:ประชุมนี้|การประชุมนี้|in\s+this\s+meeting|this\s+meeting|for\s+this\s+meeting|in\s+this\s+room|ห้องนี้)",
    re.I,
)


def parse_personal_language(text: str) -> tuple[str, str | None] | None:
    """("set", code) for "talk to me in Thai" / "สลิด คุยกับฉันเป็นภาษาไทย" / "my language is Thai",
    ("unset", None) for "use the company language" / "ใช้ภาษาบริษัท", else None."""
    s = (text or "").strip()
    if not s or len(s) > 120:
        return None
    for rx in _PERSONAL_SET:
        m = rx.match(s)
        if m:
            code = _NAME_TO_CODE.get(m.group("l").casefold())
            if code:
                return ("set", code)
    if _PERSONAL_UNSET.match(s):
        return ("unset", None)
    return None


def parse_language_mode(text: str) -> str | None:
    """ "ใช้ภาษาบริษัทอย่างเดียว" / "company language only" / "English only" -> company_only;
    "ตอบหลายภาษาได้" / "allow personal languages" -> audience; else None. Not for a meeting
    (see `parse_meeting_mode`)."""
    s = (text or "").strip()
    if not s or len(s) > 120 or _MEETING.search(s):
        return None
    if _ONLY.match(s):
        return "company_only"
    if _ALLOW.match(s):
        return "audience"
    return None


def parse_meeting_mode(text: str) -> str | None:
    """The chair's per-meeting form: "Salid, English only in this meeting" / "สลิด ประชุมนี้ใช้ภาษาอังกฤษ
    อย่างเดียว" -> company_only; "both languages in this meeting" / "ประชุมนี้ใช้สองภาษา" -> audience."""
    s = (text or "").strip()
    if not s or len(s) > 140 or not _MEETING.search(s):
        return None
    low = s.casefold()
    if re.search(
        rf"(?:{_LANG_ALT})\s+only|only\s+(?:{_LANG_ALT})|อย่างเดียว|ภาษาเดียว|company\s+language\s+only",
        low,
    ):
        return "company_only"
    if re.search(
        r"both\s+languages|two\s+languages|สองภาษา|หลายภาษา|personal\s+languages|ทั้งสองภาษา", low
    ):
        return "audience"
    return None
