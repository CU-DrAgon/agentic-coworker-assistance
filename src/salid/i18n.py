"""Salid's reply language: Thai, English or auto, plus the catalog of fixed user-facing strings.

Language setting values: `th`, `en`, `auto` (reply in the language the person wrote in). A
resolved language is always one of `LANGS` minus `auto`. English is the default (a server picks
its main language when Salid joins); Thai entries are the original wording.
"""

from __future__ import annotations

import re

PRIMARY = ("th", "en")
EXTRA = ("zh", "hi", "ms", "id", "vi", "ko", "ja")
LANGS = PRIMARY + EXTRA + ("auto",)
DEFAULT_LANGUAGE = "en"
# lang -> (native name, English name, Thai name). Thai and English are hand-written; the others
# are the second tier (model-translated catalogs in locales/<lang>.json, model replies).
LANG_NAMES: dict[str, tuple[str, str, str]] = {
    "th": ("ไทย", "Thai", "ไทย"),
    "en": ("English", "English", "อังกฤษ"),
    "zh": ("中文", "Chinese", "จีน"),
    "hi": ("हिन्दी", "Hindi", "ฮินดี"),
    "ms": ("Bahasa Melayu", "Malay", "มาเลย์"),
    "id": ("Bahasa Indonesia", "Indonesian", "อินโดนีเซีย"),
    "vi": ("Tiếng Việt", "Vietnamese", "เวียดนาม"),
    "ko": ("한국어", "Korean", "เกาหลี"),
    "ja": ("日本語", "Japanese", "ญี่ปุ่น"),
}
LANGUAGE_LABELS = {
    **{k: f"{v[0]} · {v[1]}" if k != "th" else "ไทย · Thai" for k, v in LANG_NAMES.items()},
    "en": "English · อังกฤษ",
    "auto": "อัตโนมัติ · Auto",
}
_THAI_RE = re.compile(r"[฀-๿]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_VI_RE = re.compile(
    "[ăâđêôơưĂÂĐÊÔƠƯạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]"
)


def has_thai(text: str | None) -> bool:
    return _THAI_RE.search(text or "") is not None


def base_lang(lang: str | None) -> str:
    """The hand-written language a resolved language builds on: th for Thai, en for everything
    else (the second-tier languages use the English prompt blocks, particles and formats)."""
    return "th" if lang == "th" else "en"


def map_language_code(lang: str | None) -> str:
    """BCP-47 code for Google Maps / Weather APIs (th, en, zh-CN, hi, ms, id, vi, ko, ja)."""
    return {"zh": "zh-CN"}.get(lang or "", lang if lang in LANG_NAMES else "en")


def lang_name(lang: str | None, which: int = 1) -> str:
    """Name of a language: 0 native, 1 English (default), 2 Thai."""
    return LANG_NAMES.get(lang or "", LANG_NAMES["en"])[which]


def reply_language_line(lang: str | None) -> str:
    """The mandatory first line of the persona for a second-tier language ("" for th / en)."""
    if lang not in EXTRA:
        return ""
    native, english, _ = LANG_NAMES[lang]
    return (
        f"Reply ONLY in {english} ({native}). Keep the name Salid. Polite, natural register for "
        "that language. Fixed strings the system posts may be in English; do not translate them.\n"
    )


def detect_lang(text: str | None) -> str | None:
    """The language of `text` by script: Thai -> th, kana -> ja, Hangul -> ko, Han only -> zh,
    Devanagari -> hi, Latin with Vietnamese diacritics -> vi, other letters -> en, no letters ->
    None. Malay and Indonesian are never detected (set them explicitly)."""
    s = text or ""
    if _THAI_RE.search(s):
        return "th"
    if _KANA_RE.search(s):
        return "ja"
    if _HANGUL_RE.search(s):
        return "ko"
    if _HAN_RE.search(s):
        return "zh"
    if _DEVANAGARI_RE.search(s):
        return "hi"
    if _VI_RE.search(s):
        return "vi"
    return "en" if any(c.isalpha() for c in s) else None


def resolve_lang(setting: str | None, text: str | None = None) -> str:
    """The language to reply in. `auto` follows `text` by script (see detect_lang); no text (or
    no letters, e.g. only emoji) -> the default language (English)."""
    if setting in PRIMARY or setting in EXTRA:
        return setting  # type: ignore[return-value]
    if setting == "auto":
        return detect_lang(text) or DEFAULT_LANGUAGE
    return DEFAULT_LANGUAGE


def locale_lang(locale: str | None) -> str:
    """The language of a Discord locale (`th`, `en-US`, `zh-CN`, `ja`...): a known language of
    `LANGS` by its primary subtag, English when the locale is empty or unknown."""
    code = str(locale or "").lower().replace("_", "-").split("-")[0]
    return code if code in LANGS and code != "auto" else DEFAULT_LANGUAGE


def resolve_language(
    channel_id: int,
    *,
    channel_languages: dict[str, str],
    default_language: str = DEFAULT_LANGUAGE,
) -> str:
    """The guild's language (th/en/auto). Per-channel overrides are no longer used: language is
    set for the whole server, and per person (the owner's rule, 2026-10-17)."""
    return default_language if default_language in LANGS else DEFAULT_LANGUAGE


_FILLER = r"(?:\s*(?:ครับ|ค่ะ|คะ|นะ|หน่อย|ด้วย|เถอะ|เลย|please|pls|plz|now|mode|してください|ください|吧|一下|"
_FILLER += r"해줘|해주세요|하세요|lah|ya|ya|nhé|nha))*[\s.!?]*"
_VERB_EN = r"(?:please\s+)?(?:speak|reply|answer|talk|respond)(?:\s+in)?"
# lang -> (Thai names, English names, native phrases). The Thai names are matched after
# "ภาษา" or a verb (พูด/ตอบ/คุย); English names stand alone or after a verb; native phrases
# stand alone (the whole message must be the command).
_LANG_WORDS: dict[str, tuple[str, str, str]] = {
    "en": (r"อังกฤษ", r"english", r""),
    "th": (r"ไทย", r"thai", r""),
    "zh": (
        r"จีน(?:กลาง)?|แมนดาริน",
        r"chinese|mandarin",
        r"(?:说|讲|用|請說|请说)?(?:中文|汉语|漢語|普通话|国语)(?:说|吧)?",
    ),
    "hi": (
        r"ฮินดี|ฮินดู|อินเดีย",
        r"hindi",
        r"(?:हिं?दी|हिन्दी)(?:\s+में(?:\s+(?:बोलो|बात\s+करो|जवाब\s+दो))?|\s+बोलो)?",
    ),
    "ms": (
        r"มาเลย์|มลายู|มาเลเซีย",
        r"malay|bahasa\s+(?:melayu|malaysia)",
        r"(?:cakap|guna|gunakan|bercakap|balas)\s+(?:dalam\s+)?bahasa\s+(?:melayu|malaysia)"
        r"|bahasa\s+(?:melayu|malaysia)|cakap\s+melayu|bahasa\s+melayu",
    ),
    "id": (
        r"อินโดนีเซีย|อินโด",
        r"indonesian|bahasa\s+indonesia|indonesia",
        r"(?:bicara|pakai|gunakan|balas|jawab)\s+(?:dalam\s+)?bahasa\s+indonesia|bahasa\s+indonesia",
    ),
    "vi": (
        r"เวียดนาม",
        r"vietnamese",
        r"(?:nói|trả\s+lời|dùng)\s+(?:bằng\s+)?tiếng\s+việt(?:\s+nhé)?|tiếng\s+việt",
    ),
    "ko": (
        r"เกาหลี",
        r"korean",
        r"한국어(?:로|로\s*(?:말해|대답해|답해)(?:줘|주세요)?)?|한국말(?:로)?|한글로",
    ),
    "ja": (
        r"ญี่ปุ่น",
        r"japanese",
        r"日本語(?:で(?:話して|答えて|返事して|お願い)?(?:ください)?)?|にほんご(?:で)?",
    ),
}


def _lang_pattern(lang: str) -> str:
    th, en, native = _LANG_WORDS[lang]
    parts = [
        rf"(?:(?:พูด|ตอบ|คุย|ใช้)\s?)?ภาษา\s?(?:{th})",
        rf"(?:พูด|ตอบ|คุย|ใช้)\s?(?:{th})",
        rf"(?:{_VERB_EN}\s+(?:{en}))",
        rf"(?:in\s+)?(?:{en})",
        rf"use\s+(?:{en})",
    ]
    if native:
        parts.append(native)
    return "|".join(parts)


_LANG_COMMANDS = (
    *((lang, _lang_pattern(lang)) for lang in ("en", "th", *EXTRA)),
    (
        "auto",
        r"(?:(?:พูด|ตอบ|คุย)\s?)?ภาษา\s?(?:อัตโนมัติ|ออโต้|auto)|อัตโนมัติ"
        r"|auto(?:matic)?(?:\s+language)?|(?:language|lang)\s+auto(?:matic)?",
    ),
)
_SERVER_VERB = (
    r"(?:ใช้|เปลี่ยน|ตั้ง|สลับ|พูด|ตอบ|คุย|ปรับ|ให้|switch|change|set|use|make|put|speak|reply|answer|"
    r"talk|respond|please|go|run|turn)"
)
_SERVER_SCOPE = re.compile(
    r"ทั้งเซิร์?ฟ(?:เวอร์)?|ทั้งหมด|ทุกห้อง|ทุกที่|ทั้งเซิ[ร์ฟ]*|เซิร์?ฟเวอร์|ทั้งบริษัท|ทุกอย่าง"
    r"|\bserver\b|server-?wide|everywhere|all\s+channels|every\s+channel|\bwhole\b|\bentire\b"
    r"|everything|all\s+of\s+salid",
    re.I,
)
_SERVER_LANG_RES = [
    (
        lang,
        re.compile(
            rf"ภาษา\s?(?:{th})|(?:{th})|\b(?:{en})\b" + (rf"|{native}" if native else ""), re.I
        ),
    )
    for lang, (th, en, native) in _LANG_WORDS.items()
]
_SERVER_START = re.compile(rf"^\s*(?:@\S+\s+)?{_SERVER_VERB}", re.I)
_LANG_RES = [(v, re.compile(rf"^(?:{p})(?:{_FILLER})$", re.I)) for v, p in _LANG_COMMANDS]


def parse_language_command(text: str) -> str | None:
    """ "พูดภาษาอังกฤษ" / "speak English" -> "en", "พูดไทย" / "speak Thai" -> "th", "พูดญี่ปุ่น" /
    "日本語で" / "한국어로" / "nói tiếng Việt" -> ja / ja / ko / vi (zh hi ms id likewise),
    "ภาษาอัตโนมัติ" / "auto language" -> "auto", anything else None. The whole (mention-stripped)
    message must be the command, so a question that merely starts with "English" is not one."""
    s = (text or "").strip().strip(" \t\n/,.:;!-–—…")
    if not s or len(s) > 40:
        return None
    for value, rx in _LANG_RES:
        if rx.match(s):
            return value
    return None


def parse_server_language_command(text: str) -> str | None:
    """ "ใช้ภาษาอังกฤษทั้งเซิร์ฟเวอร์" / "switch the whole server to English" -> "en", the same with
    Thai (or Japanese, Korean...) -> that language, anything else None. Needs a verb at the start,
    a scope word (server, everywhere, ทั้งหมด...) and a language; the last language named wins
    ("from Thai to English")."""
    s = (text or "").strip().strip(" \t\n/,.:;!-–—…")
    if not s or len(s) > 90 or not _SERVER_START.match(s) or not _SERVER_SCOPE.search(s):
        return None
    if s.rstrip("?？").lower() != s.lower():
        return None
    best: tuple[int, str] | None = None
    for lang, rx in _SERVER_LANG_RES:
        for m in rx.finditer(s):
            if best is None or m.start() >= best[0]:
                best = (m.start(), lang)
    return best[1] if best else None


def _bi(th: str, en: str) -> dict[str, str]:
    both = f"{th} · {en}"
    return {"th": both, "en": both}


_LOCALES: dict[str, dict[str, str]] = {}


def _locale(lang: str) -> dict[str, str]:
    """The generated strings of a second-tier language (empty when there is no catalog file)."""
    if lang not in _LOCALES:
        import json
        from pathlib import Path

        path = Path(__file__).parent / "locales" / f"{lang}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("strings", {})
        except (OSError, ValueError, AttributeError):
            data = {}
        _LOCALES[lang] = {k: v for k, v in data.items() if isinstance(v, str)}
    return _LOCALES[lang]


def t(key: str, lang: str = DEFAULT_LANGUAGE, **fmt) -> str:
    """Catalog text for a resolved language. th / en are hand-written; the second-tier languages
    read locales/<lang>.json and fall back to English (never Thai); anything else is English."""
    entry = CATALOG[key]
    if lang in EXTRA:
        text = _locale(lang).get(key) or entry["en"]
    else:
        text = entry["th"] if lang == "th" else entry["en"]
    return text.format(**fmt) if fmt else text


CATALOG: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- quotations
    "co_kind_profile": {"th": "ข้อมูลหัวบริษัท", "en": "company details"},
    "docs_font_missing": {
        "th": "ฟอนต์ภาษา{language}ยังไม่ได้ติดตั้งบนเซิร์ฟเวอร์ เลยทำเอกสารเป็นภาษาอังกฤษแทนค่ะ",
        "en": "Fonts for {language} are not installed on this server, so the document was made in English.",
    },
    "quote_title": {"th": "ใบเสนอราคา", "en": "QUOTATION"},
    "quote_no": {"th": "เลขที่", "en": "No."},
    "quote_rev": {"th": "ฉบับแก้ไขครั้งที่ {n}", "en": "Revision {n}"},
    "quote_date": {"th": "วันที่", "en": "Date"},
    "quote_valid_until": {"th": "ยืนราคาถึงวันที่", "en": "Valid until"},
    "quote_prepared_by": {"th": "ผู้จัดทำ", "en": "Prepared by"},
    "quote_customer": {"th": "เสนอต่อ", "en": "To"},
    "quote_attn": {"th": "ผู้ติดต่อ", "en": "Attn"},
    "quote_tax_id": {"th": "เลขประจำตัวผู้เสียภาษี", "en": "Tax ID"},
    "quote_phone": {"th": "โทร", "en": "Tel"},
    "quote_email": {"th": "อีเมล", "en": "Email"},
    "quote_web": {"th": "เว็บไซต์", "en": "Web"},
    "quote_col_no": {"th": "ลำดับ", "en": "No."},
    "quote_col_desc": {"th": "รายการ", "en": "Description"},
    "quote_col_qty": {"th": "จำนวน", "en": "Qty"},
    "quote_col_unit": {"th": "หน่วย", "en": "Unit"},
    "quote_col_price": {"th": "ราคาต่อหน่วย", "en": "Unit price"},
    "quote_col_disc": {"th": "ส่วนลด", "en": "Discount"},
    "quote_col_amount": {"th": "จำนวนเงิน", "en": "Amount"},
    "quote_subtotal": {"th": "รวมเป็นเงิน", "en": "Subtotal"},
    "quote_discount": {"th": "ส่วนลด", "en": "Discount"},
    "quote_discount_pct": {"th": "ส่วนลด {pct}%", "en": "Discount {pct}%"},
    "quote_after_discount": {"th": "ราคาหลังหักส่วนลด", "en": "After discount"},
    "quote_before_vat": {"th": "ราคาก่อนภาษีมูลค่าเพิ่ม", "en": "Amount before VAT"},
    "quote_vat": {"th": "ภาษีมูลค่าเพิ่ม {rate}%", "en": "VAT {rate}%"},
    "quote_vat_incl": {
        "th": "ภาษีมูลค่าเพิ่ม {rate}% (รวมอยู่ในราคาแล้ว)",
        "en": "VAT {rate}% (included in the prices)",
    },
    "quote_no_vat": {"th": "ไม่คิดภาษีมูลค่าเพิ่ม", "en": "VAT not applied"},
    "quote_total": {"th": "จำนวนเงินรวมทั้งสิ้น", "en": "Grand total"},
    "quote_words": {"th": "ตัวอักษร", "en": "In words"},
    "quote_terms": {"th": "เงื่อนไข", "en": "Terms"},
    "quote_payment": {"th": "การชำระเงิน", "en": "Payment"},
    "quote_default_payment": {
        "th": "ชำระภายใน 30 วันหลังรับใบแจ้งหนี้",
        "en": "Payment within 30 days of receiving the invoice",
    },
    "quote_validity": {
        "th": "ยืนราคา {days} วัน (ถึง {until})",
        "en": "Prices valid for {days} days (until {until})",
    },
    "quote_notes": {"th": "หมายเหตุ", "en": "Notes"},
    "quote_bank": {"th": "บัญชีรับชำระเงิน", "en": "Bank account"},
    "quote_accepted_by": {"th": "ผู้สั่งซื้อ (ลงนามยืนยัน)", "en": "Customer acceptance"},
    "quote_signed_by": {"th": "ผู้เสนอราคา", "en": "Authorised signatory"},
    "quote_date_line": {"th": "วันที่ ____ / ____ / ________", "en": "Date ____ / ____ / ________"},
    "quote_missing": {"th": "[ยังไม่ได้ระบุ: {field}]", "en": "[not set: {field}]"},
    "quote_f_legal_name": {"th": "ชื่อบริษัท", "en": "company name"},
    "quote_f_address": {"th": "ที่อยู่", "en": "address"},
    "quote_f_tax_id": {"th": "เลขประจำตัวผู้เสียภาษี", "en": "tax ID"},
    "quote_f_signer_name": {"th": "ชื่อผู้ลงนาม", "en": "signer name"},
    "quote_refusal": {
        "th": "การออกใบเสนอราคาในนามบริษัททำได้เฉพาะผู้บริหาร{roles}ค่ะ สิทธิ์นี้มาจากบทบาทใน Discord ที่แอดมินของเซิร์ฟเวอร์กำหนด สลิดให้หรือเปลี่ยนสิทธิ์ให้ใครเองไม่ได้นะคะ ถ้าต้องการออกใบเสนอราคา ขอให้ผู้บริหารเป็นคนสั่ง หรือให้แอดมินเพิ่มบทบาทให้ค่ะ",
        "en": "Only executives{roles} can issue quotations in the company's name. That right comes from Discord roles set by the server admins; Salid cannot grant or change anyone's rights. Ask an executive to issue it, or an admin to add the role.",
    },
    "quote_refusal_roles": {
        "th": " และสมาชิกที่มีบทบาท {roles}",
        "en": " and members with the role {roles}",
    },
    "quote_teach_hint": {
        "th": "สอนสลิดได้เลย เช่น “ที่อยู่บริษัทคือ …” “เลขผู้เสียภาษี …” “ผู้ลงนามคือ … ตำแหน่ง …”",
        "en": "Just tell Salid, e.g. “the company address is …”, “our tax ID is …”, “the signer is … , title …”",
    },
    # ---------------------------------------------------------------- company learning
    "co_btn_ok": {"th": "อนุมัติ", "en": "Approve"},
    "co_btn_no": {"th": "ไม่อนุมัติ", "en": "Decline"},
    "co_kind_rule": {"th": "กฎของบริษัท", "en": "company rule"},
    "co_kind_persona": {"th": "แนวทางบุคลิก/น้ำเสียง", "en": "tone / persona note"},
    "co_kind_glossary": {"th": "ศัพท์เฉพาะ", "en": "glossary term"},
    "co_kind_sop": {"th": "ขั้นตอนงาน", "en": "procedure"},
    "co_kind_faq": {"th": "คำถามที่พบบ่อย", "en": "FAQ"},
    "co_kind_about": {"th": "ข้อมูลบริษัท", "en": "company info"},
    "co_kind_industry": {"th": "ประเภทธุรกิจ", "en": "kind of business"},
    "co_suggest_dm": {
        "th": "{name} เสนอให้สลิดจำเรื่องนี้ไว้ในบริษัท ({kind}) ค่ะ\n“{text}”{detail}{link}\nอนุมัติไหมคะ",
        "en": '{name} suggests Salid remembers this for the company ({kind}).\n"{text}"{detail}{link}\nApprove?',
    },
    "co_approved": {"th": "อนุมัติแล้วค่ะ สลิดจำไว้แล้ว", "en": "Approved. Salid has learned it."},
    "co_declined": {"th": "ไม่อนุมัติค่ะ", "en": "Declined."},
    "co_handled": {"th": "รายการนี้ถูกจัดการไปแล้วค่ะ", "en": "This one was already handled."},
    "co_not_admin": {
        "th": "เฉพาะผู้ดูแลเท่านั้นที่อนุมัติได้ค่ะ",
        "en": "Only admins can approve this.",
    },
    "co_result_dm_ok": {
        "th": "{who} อนุมัติแล้วค่ะ สลิดจำไว้แล้ว: “{text}”",
        "en": '{who} approved it. Salid has learned: "{text}"',
    },
    "co_result_dm_no": {
        "th": "{who} ยังไม่อนุมัติเรื่องนี้ค่ะ: “{text}”",
        "en": '{who} did not approve this one: "{text}"',
    },
    # ---------------------------------------------------------------- bot: general
    "qa_off": {
        "th": "ตอนนี้ปิดการถามตอบอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Q&A is switched off right now. You can turn it on in Salid's command center.",
    },
    "deep_off": {
        "th": "ตอนนี้ปิดงานใหญ่อยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Big tasks are switched off right now. You can turn them on in Salid's command center.",
    },
    "unlicensed": {
        "th": "เซิร์ฟเวอร์นี้ยังไม่ได้เปิดใช้สลิดค่ะ ให้ผู้ดูแลสมัครที่ {url}",
        "en": "Salid is not activated on this server yet. An admin can subscribe at {url}",
    },
    "dm": {
        "th": "สลิดแยกข้อมูลของแต่ละเซิร์ฟเวอร์ค่ะ รบกวนแท็ก {mention} ถามในห้องแชตของเซิร์ฟเวอร์ที่ต้องการนะคะ",
        "en": "Salid keeps each server's data separate. Please tag {mention} in the chat channel of "
        "the server you mean.",
    },
    "dmg_ask": {
        "th": "คุยเรื่องบริษัทไหนคะ?",
        "en": "Which company is this about?",
    },
    "dmg_switched": {
        "th": "ได้ค่ะ ต่อไปนี้สลิดคุยเรื่องบริษัท {name} นะคะ",
        "en": "Done. I'll talk about {name} from now on.",
    },
    "dmg_confirm": {
        "th": "เปลี่ยนไปคุยเรื่องบริษัท {name} ใช่ไหมคะ?",
        "en": "Switch to talking about {name}?",
    },
    "dmg_btn_ok": {"th": "ใช่ค่ะ", "en": "Yes"},
    "dmg_btn_no": {"th": "ยกเลิก", "en": "Cancel"},
    "dmg_kept": {"th": "ได้ค่ะ คุยเรื่องเดิมต่อนะคะ", "en": "OK, staying on the current company."},
    "pc_category": {"th": "คุยกับสลิด (ส่วนตัว)", "en": "Salid private chats"},
    "pc_topic": {
        "th": "ห้องส่วนตัวของคุณกับสลิด",
        "en": "Your private room with Salid",
    },
    "pc_welcome": {
        "th": "สวัสดีค่ะ {mention} นี่คือห้องส่วนตัวของคุณกับสลิดค่ะ "
        "ห้องนี้เห็นแค่คุณกับสลิด (และแอดมินของเซิร์ฟเวอร์ที่มีสิทธิ์ Administrator) "
        "คุยได้เลยไม่ต้องแท็ก สลิดจะส่งสรุป เตือนงาน และขออนุมัติมาที่นี่ด้วยค่ะ "
        "อยากรับทาง DM แทนก็บอกได้ หรือพิมพ์ “ลบห้องส่วนตัว” เพื่อลบห้องนี้ค่ะ",
        "en": "Hi {mention}, this is your private room with Salid. Only you and Salid can see it "
        "(plus server admins who have the Administrator permission). Just talk, no tag needed. "
        "Salid will send your briefings, reminders and approval requests here. Prefer DMs? Say so. "
        'Type "delete my private channel" to remove this room.',
    },
    "pc_created": {
        "th": "สร้างห้องส่วนตัวให้แล้วค่ะ {channel} (เซิร์ฟเวอร์ {server})",
        "en": "Your private channel is ready: {channel} ({server})",
    },
    "pc_exists": {
        "th": "คุณมีห้องส่วนตัวอยู่แล้วค่ะ {channel} (เซิร์ฟเวอร์ {server})",
        "en": "You already have a private channel: {channel} ({server})",
    },
    "pc_failed": {
        "th": "สร้างห้องส่วนตัวไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ",
        "en": "I couldn't create the private channel. Please try again.",
    },
    "pc_off": {
        "th": "แอดมินปิดห้องส่วนตัวไว้ค่ะ",
        "en": "Private channels are switched off by the admins.",
    },
    "pc_cap": {
        "th": "เซิร์ฟเวอร์นี้ห้องเต็มใกล้ขีดจำกัดของ Discord แล้ว สลิดจะส่งทาง DM แทนนะคะ",
        "en": "This server is near Discord's channel limit, so I'll use DMs instead.",
    },
    "pc_cap_admin": {
        "th": "ห้องในเซิร์ฟเวอร์ใกล้ถึงขีดจำกัดของ Discord (500) สลิดเลยหยุดสร้างห้องส่วนตัวให้สมาชิกเพิ่ม และจะส่งทาง DM แทนค่ะ",
        "en": "The server is close to Discord's channel limit (500), so I've paused creating private "
        "channels and will use DMs for the rest.",
    },
    "pc_none": {"th": "คุณยังไม่มีห้องส่วนตัวค่ะ", "en": "You don't have a private channel."},
    "pc_delete_ask": {
        "th": "ลบห้องส่วนตัวของคุณใช่ไหมคะ? ตอบ “ยืนยัน” เพื่อลบค่ะ",
        "en": 'Delete your private channel? Reply "confirm" to delete it.',
    },
    "pc_deleted": {"th": "ลบห้องส่วนตัวให้แล้วค่ะ", "en": "Your private channel is deleted."},
    "pc_offer": {
        "th": "หรือจะคุยในห้องส่วนตัวในเซิร์ฟเวอร์ก็ได้นะคะ (พิมพ์ “ขอห้องส่วนตัว” ในเซิร์ฟเวอร์นั้น)",
        "en": 'Or talk in your private channel inside the server (say "make me a private channel").',
    },
    "pc_offer_link": {"th": "ห้องของคุณใน {server}: {url}", "en": "Your room in {server}: {url}"},
    "cmd_private": _bi("สร้างห้องคุยส่วนตัวกับสลิด", "Create your private chat with Salid"),
    "dmg_footer": {"th": "— บริษัท: {name}", "en": "— Company: {name}"},
    "dmg_current": {"th": "ตอนนี้: {name}", "en": "Now: {name}"},
    "dmg_not_yours": {
        "th": "ปุ่มนี้ของคนอื่นค่ะ",
        "en": "These buttons are for someone else.",
    },
    "dmg_gone": {
        "th": "ดูเหมือนคุณไม่ได้อยู่ในเซิร์ฟเวอร์นั้นแล้วค่ะ เลือกใหม่ได้เลยนะคะ",
        "en": "You no longer seem to be in that server. Please pick again.",
    },
    "dmg_none": {
        "th": "สลิดไม่พบเซิร์ฟเวอร์ที่คุณอยู่ร่วมกับสลิดค่ะ",
        "en": "I can't find a server we share.",
    },
    "cmd_company": _bi("เลือกบริษัทที่สลิดจะคุยด้วยใน DM", "Choose which company Salid talks about in DMs"),
    "cmd_company_arg": _bi("ชื่อบริษัท (ไม่ใส่ = เลือกจากปุ่ม)", "Company name (empty = pick with buttons)"),
    "welcome": {
        "th": "สวัสดีค่ะ สลิด เลขาประจำทีมค่ะ 🐱 ทักสลิดได้เลย พิมพ์หรือส่งไฟล์ รูป เสียงมาได้หมด "
        "ห้องนี้คุยกับสลิดได้เลยไม่ต้องแท็ก ห้องอื่นแท็ก {mention} หรือเรียกชื่อสลิดก็พอ ({chat})\n"
        "สลิดจำแค่ห้องที่สลิดเห็นได้ ({days} วัน) ไม่อยากให้จำห้องไหน บอกสลิดได้เลยว่า “ห้องนี้ปิดไว้นะ”",
        "en": "Hi, I'm Salid, your team's secretary 🐱 Just talk to me: type, or send files, photos "
        "or voice, anything goes. In {chat} you can chat with me without tagging; elsewhere tag "
        "{mention} or call my name.\n"
        "I only remember channels I'm allowed to see ({days} days). If you'd rather I skip one, "
        "just tell me “switch this channel off”.",
    },
    "chat_topic": {
        "th": "คุยกับน้องสลิดได้ทุกเรื่อง พิมพ์ได้เลยไม่ต้องแท็ก",
        "en": "Chat with Salid about anything, no tag needed",
    },
    "greeting": {
        "th": "ว่าไงค่ะ มีอะไรให้ช่วยไหม 🙂",
        "en": "Hey, what can I do for you? 🙂",
    },
    "greeting_2": {
        "th": "ค่ะ ว่าเลยค่ะ",
        "en": "Yes? I'm all ears.",
    },
    "greeting_3": {
        "th": "สลิดอยู่ค่ะ มีอะไรให้ช่วยบอกได้เลย",
        "en": "Salid here, tell me what you need.",
    },
    "ack_doc_1": {
        "th": "ขอเวลาอ่านเอกสารให้เข้าใจก่อนนะคะ",
        "en": "Give me a moment, I want to read the document properly first",
    },
    "ack_doc_2": {"th": "เดี๋ยวเปิดอ่านไฟล์นี้ก่อนนะคะ", "en": "Let me open the file and read it through"},
    "ack_doc_3": {
        "th": "ขออ่านเอกสารแป๊บนึงนะคะ เดี๋ยวมาตอบค่ะ",
        "en": "Reading it now, back in a moment",
    },
    "ack_doc_4": {
        "th": "รับไฟล์แล้วค่ะ ขอไล่อ่านก่อนนะคะ",
        "en": "Got the file, let me go through it first",
    },
    "ack_doc_5": {"th": "สักครู่นะคะ กำลังอ่านเอกสารให้อยู่", "en": "One sec, I'm reading the document"},
    "ack_doc_6": {"th": "ขออ่านรายละเอียดในไฟล์ก่อนนะคะ", "en": "Let me read the details first"},
    "ack_doc_7": {"th": "เดี๋ยวไล่ดูเอกสารให้เลยค่ะ", "en": "I'll go through this for you now"},
    "ack_doc_8": {
        "th": "ขอเวลาอ่านไฟล์นี้ให้ละเอียดหน่อยนะคะ",
        "en": "Give me a minute to read it carefully",
    },
    "ack_image_1": {"th": "ขอดูรูปแป๊บนึงนะคะ", "en": "Let me take a look at the picture"},
    "ack_image_2": {"th": "เดี๋ยวดูรูปให้ชัดๆ ก่อนนะคะ", "en": "One sec, looking at it closely"},
    "ack_image_3": {"th": "สักครู่นะคะ กำลังดูรูปอยู่", "en": "Checking the image now"},
    "ack_image_4": {"th": "ขอเพ่งดูรูปนี้หน่อยนะคะ", "en": "Let me have a closer look"},
    "ack_image_5": {
        "th": "รับรูปแล้วค่ะ ขอดูรายละเอียดก่อนนะคะ",
        "en": "Got the picture, let me see the details",
    },
    "ack_image_6": {"th": "เดี๋ยวดูรูปให้เลยค่ะ", "en": "I'll take a look right now"},
    "ack_image_7": {"th": "ขอเวลาดูรูปให้ละเอียดหน่อยนะคะ", "en": "Give me a moment to study the image"},
    "ack_image_8": {"th": "แป๊บนึงนะคะ ขอดูรูปให้ชัดก่อน", "en": "Looking at it now"},
    "ack_audio_1": {"th": "ขอฟังคลิปก่อนนะคะ", "en": "Let me listen to the clip first"},
    "ack_audio_2": {"th": "เดี๋ยวฟังให้จบก่อนนะคะ", "en": "I'll hear it through first"},
    "ack_audio_3": {"th": "ขอเวลาฟังแป๊บนึงนะคะ", "en": "Give me a minute to listen to it"},
    "ack_audio_4": {"th": "สักครู่นะคะ กำลังฟังคลิปอยู่", "en": "Listening now, back in a moment"},
    "ack_audio_5": {"th": "รับคลิปแล้วค่ะ ขอฟังก่อนนะคะ", "en": "Got the clip, let me listen"},
    "ack_audio_6": {"th": "เดี๋ยวเปิดฟังให้เลยค่ะ", "en": "Let me play it and listen properly"},
    "ack_audio_7": {"th": "ขอฟังเนื้อหาในคลิปก่อนนะคะ", "en": "One sec, I'm listening to it"},
    "ack_audio_8": {"th": "แป๊บนึงนะคะ ขอฟังให้ครบก่อน", "en": "Let me hear the whole thing first"},
    "ack_link_1": {"th": "ขอเปิดลิงก์ดูแป๊บนึงค่ะ", "en": "Let me open that link real quick"},
    "ack_link_2": {"th": "เดี๋ยวเปิดลิงก์อ่านก่อนนะคะ", "en": "I'll read the link first"},
    "ack_link_3": {"th": "สักครู่นะคะ กำลังอ่านลิงก์อยู่", "en": "Reading the page now, one sec"},
    "ack_link_4": {"th": "ขอเข้าไปอ่านหน้านั้นก่อนนะคะ", "en": "Let me pull up that page"},
    "ack_link_5": {"th": "รับลิงก์แล้วค่ะ ขออ่านก่อนนะคะ", "en": "Got the link, let me read it"},
    "ack_link_6": {"th": "เดี๋ยวเข้าไปดูลิงก์ให้เลยค่ะ", "en": "I'll go and take a look at it"},
    "ack_link_7": {"th": "ขอเวลาอ่านลิงก์นี้หน่อยนะคะ", "en": "Give me a moment to read that link"},
    "ack_link_8": {"th": "แป๊บนึงนะคะ ขอเปิดอ่านลิงก์ก่อน", "en": "Opening it now"},
    "ack_deep_1": {"th": "เดี๋ยวสลิดไล่ดูให้นะคะ", "en": "I'll go through it for you"},
    "ack_deep_2": {
        "th": "รับทราบค่ะ ขอเวลาไล่ดูข้อมูลก่อนนะคะ",
        "en": "On it, give me a bit to go through the data",
    },
    "ack_deep_3": {
        "th": "สักครู่นะคะ ขอไล่เช็กข้อมูลให้ครบก่อน",
        "en": "Let me work through all of it first",
    },
    "ack_deep_4": {
        "th": "ได้เลยค่ะ เดี๋ยวสลิดรวบรวมให้นะคะ",
        "en": "Sure, I'll pull it together for you",
    },
    "ack_deep_5": {
        "th": "ขอเวลาไล่ดูข้อมูลทั้งหมดก่อนนะคะ",
        "en": "Give me a little while to go through everything",
    },
    "ack_deep_6": {
        "th": "เรื่องนี้ต้องดูเยอะหน่อย ขอเวลาสักครู่นะคะ",
        "en": "This one needs a closer look, back soon",
    },
    "ack_deep_7": {"th": "เดี๋ยวสลิดไล่ข้อมูลให้ละเอียดเลยค่ะ", "en": "I'll dig into the data properly"},
    "ack_deep_8": {"th": "รับเรื่องแล้วค่ะ กำลังไล่ดูให้อยู่นะคะ", "en": "Got it, working through it now"},
    "ack_slow_1": {"th": "แป๊บนึงนะคะ กำลังดูให้อยู่", "en": "One moment, I'm on it"},
    "ack_slow_2": {"th": "ขอเวลาเช็กให้แป๊บนึงนะคะ", "en": "Give me a second to check that"},
    "ack_slow_3": {"th": "สักครู่นะคะ กำลังดูให้อยู่", "en": "Looking into it, one moment"},
    "ack_slow_4": {"th": "เดี๋ยวดูให้เลยค่ะ", "en": "Let me check that for you"},
    "ack_slow_5": {"th": "ขอเช็กให้แป๊บนะคะ", "en": "Checking now, back in a sec"},
    "ack_slow_6": {"th": "กำลังหาข้อมูลให้อยู่ค่ะ รอสักครู่นะคะ", "en": "Pulling that up for you now"},
    "ack_slow_7": {"th": "ได้ค่ะ ขอเวลานิดนึงนะคะ", "en": "Sure, just a moment"},
    "ack_slow_8": {"th": "เดี๋ยวสลิดเช็กให้ค่ะ", "en": "I'll look that up for you"},
    "ack_research_1": {
        "th": "รับทราบค่ะ ขอเวลาค้นข้อมูลให้ก่อนนะคะ",
        "en": "On it, give me a little while to research this",
    },
    "ack_research_2": {"th": "เดี๋ยวสลิดค้นข้อมูลให้เลยค่ะ", "en": "I'll look into this for you now"},
    "ack_research_3": {
        "th": "ได้เลยค่ะ ขอเวลาหาข้อมูลให้ละเอียดหน่อยนะคะ",
        "en": "Sure, let me dig up the information",
    },
    "ack_research_4": {"th": "กำลังค้นข้อมูลให้อยู่นะคะ", "en": "Researching it now"},
    "ack_research_5": {
        "th": "เรื่องนี้ขอไปหาข้อมูลมาให้ก่อนนะคะ",
        "en": "Let me go and find the details on this",
    },
    "ack_research_6": {
        "th": "สักครู่นะคะ สลิดกำลังรวบรวมข้อมูลให้ค่ะ",
        "en": "One moment, I'm gathering the information",
    },
    "ack_research_7": {"th": "เดี๋ยวสลิดไปค้นมาให้นะคะ", "en": "I'll go and look that up for you"},
    "ack_research_8": {"th": "รับเรื่องแล้วค่ะ เดี๋ยวหาข้อมูลมาให้", "en": "Got it, I'll find what I can"},
    "ack_voice_1": {"th": "แป๊บนึงนะคะ", "en": "One moment."},
    "ack_voice_2": {"th": "ขอดูให้แป๊บนึงนะคะ", "en": "One moment, let me check."},
    "ack_voice_3": {"th": "เดี๋ยวสลิดเช็กให้ค่ะ", "en": "Let me look that up."},
    "ack_voice_4": {"th": "รอสักครู่นะคะ", "en": "Give me a second."},
    "ack_voice_5": {"th": "สักครู่นะคะ กำลังดูให้อยู่", "en": "Sure, checking now."},
    "ack_voice_6": {"th": "เดี๋ยวดูให้เลยค่ะ", "en": "Let me see."},
    "ack_voice_7": {"th": "ขอเช็กให้แป๊บนะคะ", "en": "Just a sec, I'm on it."},
    "ack_voice_8": {"th": "ได้ค่ะ ขอเวลานิดนึงนะคะ", "en": "Okay, one moment."},
    "ack_again_1": {"th": "ยังดูอยู่นะคะ ขออีกนิดค่ะ", "en": "Still working on it, bear with me."},
    "ack_again_2": {"th": "กำลังรวบรวมอยู่ค่ะ รอสักครู่นะคะ", "en": "Still on it, just a bit longer."},
    "ack_again_3": {
        "th": "ข้อมูลเยอะหน่อยนะคะ ขอเวลาอีกนิดค่ะ",
        "en": "Thanks for waiting, still checking.",
    },
    "ack_again_4": {"th": "ยังทำอยู่นะคะ ขออีกสักครู่ค่ะ", "en": "A lot to go through, one more moment."},
    "ack_again_5": {"th": "ขอบคุณที่รอนะคะ ยังเช็กอยู่ค่ะ", "en": "Still looking, sorry for the wait."},
    "ack_again_6": {"th": "อีกนิดเดียวนะคะ ยังหาอยู่ค่ะ", "en": "Hang on, I'm still on it."},
    "ack_again_7": {"th": "ขอโทษที่ให้รอนะคะ ยังดูอยู่ค่ะ", "en": "Not quite done yet, still working."},
    "ack_again_8": {
        "th": "ยังไม่เสร็จนะคะ แต่กำลังทำอยู่ค่ะ",
        "en": "Bear with me, still pulling it together.",
    },
    "relay_dm": {
        "th": "ข้อความจากคุณ {sender} ({server}) ผ่านสลิดค่ะ: {text}",
        "en": "A message from {sender} ({server}), passed on by Salid: {text}",
    },
    "bc_post": {"th": "📢 แจ้งจากคุณ{sender}: {text}", "en": "📢 From {sender}: {text}"},
    "bc_preview_post": {
        "th": "จะโพสต์ใน #{channel} {ping}\n“{text}”\nยืนยันไหมคะ",
        "en": "I'll post in #{channel} {ping}\n“{text}”\nShall I go ahead?",
    },
    "bc_cant_see": {
        "th": "ไม่แท็ก {names} เพราะมองไม่เห็นห้องนี้ ให้สลิดส่ง DM บอกแทนไหมคะ",
        "en": "Not tagging {names}: they can't see this channel. Shall I DM them instead?",
    },
    "bc_preview_dm": {
        "th": "จะส่ง DM ถึง {count} คน: {names}\n“{text}”\nยืนยันไหมคะ",
        "en": "I'll DM {count} people: {names}\n“{text}”\nShall I go ahead?",
    },
    "bc_report": {
        "th": "ส่ง DM เสร็จแล้วค่ะ: ส่งสำเร็จ {sent} คน",
        "en": "DMs done: sent to {sent}",
    },
    "bc_report_closed": {
        "th": "ปิดรับ DM {count} คน ({names})",
        "en": "DMs closed for {count} ({names})",
    },
    "bc_report_failed": {
        "th": "ส่งไม่สำเร็จ {count} คน ({names})",
        "en": "failed for {count} ({names})",
    },
    "dm_closed": {
        "th": "ส่ง DM หา {name} ไม่ได้ค่ะ น่าจะปิดรับข้อความส่วนตัวจากสมาชิกในเซิร์ฟเวอร์ไว้ "
        "ให้เปิดที่ ชื่อเซิร์ฟเวอร์ ({server}) > ความเป็นส่วนตัว (Privacy Settings) > อนุญาตข้อความส่วนตัว (Direct Messages) แล้วบอกสลิดอีกที",
        "en": "I couldn't DM {name}; they probably have server DMs turned off. They can switch them "
        "on under {server} > Privacy Settings > Direct Messages, then ask me again.",
    },
    "signoff": {
        "th": "ถ้ามีอะไรก็ทักน้องสลิดมาได้เลยนะคะ 👋",
        "en": "If you need anything, just ping Salid 👋",
    },
    "snooze": {
        "th": "ได้เลยค่ะ สลิดเงียบในห้องนี้ 1 ชั่วโมง 🤫 พิมพ์ “@Salid คุยได้แล้ว” เมื่อไรก็เรียกกลับได้ค่ะ",
        "en": "Sure, Salid will stay quiet in this channel for 1 hour 🤫 Type “@Salid you can talk” "
        "any time to call me back.",
    },
    "unsnooze": {"th": "กลับมาคุยแล้วค่ะ 😊", "en": "I'm back 😊"},
    "not_snoozed": {
        "th": "สลิดคุยได้ตามปกติอยู่แล้วค่ะ 🙂",
        "en": "Salid is already chatting as usual 🙂",
    },
    "history_meeting": {
        "th": "บันทึกการประชุมล่าสุด (ใครพูดอะไร):\n{meeting}\n\nแชต:\n{chat}",
        "en": "Latest meeting transcript (who said what):\n{meeting}\n\nChat:\n{chat}",
    },
    "correction_denied": {
        "th": "แก้ไขได้เฉพาะผู้ส่งใบเสร็จหรือผู้บริหารเท่านั้น",
        "en": "Only the person who sent the receipt, or an executive, can edit it.",
    },
    "deep_failed": {
        "th": "❌ งานใหญ่ล้มเหลว ลองใหม่อีกครั้งภายหลังค่ะ",
        "en": "❌ The big task failed. Please try again later.",
    },
    # ---------------------------------------------------------------- bot: slash commands
    "hidden_none": {
        "th": "สลิดเห็นทุกห้องแล้วค่ะ 👍",
        "en": "Salid can see every channel now 👍",
    },
    "hidden_header": {
        "th": "🔒 ห้องที่สลิดยังมองไม่เห็น ({n} ห้อง)",
        "en": "🔒 Channels Salid cannot see yet ({n})",
    },
    "hidden_reason_view": {"th": "ไม่เห็นห้อง", "en": "cannot see the channel"},
    "hidden_reason_history": {"th": "อ่านย้อนหลังไม่ได้", "en": "cannot read history"},
    "hidden_more": {
        "th": "และอีก {m} ห้อง (ดูทั้งหมดในหน้า Hub)",
        "en": "and {m} more (see all of them in the Hub)",
    },
    "hidden_steps": {
        "th": "วิธีให้สลิดเข้าห้อง:\n"
        "1. กดชื่อห้องด้านบน แล้วกด ⚙️ (Edit Channel)\n"
        "2. ไปที่ Permissions → กด + (Add members or roles)\n"
        "3. พิมพ์ Salid แล้วเลือก\n"
        "4. ติ๊ก ✅ View Channel และ ✅ Read Message History แล้วกด Save",
        "en": "How to let Salid in:\n"
        "1. Click the channel name above, then click ⚙️ (Edit Channel)\n"
        "2. Go to Permissions → click + (Add members or roles)\n"
        "3. Type Salid and pick it\n"
        "4. Tick ✅ View Channel and ✅ Read Message History, then click Save",
    },
    "hidden_footer": {
        "th": "สลิดจะเริ่มอ่านห้องนั้นและประวัติย้อนหลังให้เองทันที ห้องลับที่ไม่อยากให้สลิดรู้ ไม่ต้องเพิ่มค่ะ",
        "en": "Salid will start reading that channel and its history by itself right away. "
        "Private channels you don't want Salid to know about: just leave them out.",
    },
    "cmd_channels": _bi(
        "ดูห้องที่สลิดยังมองไม่เห็น และวิธีเพิ่ม", "Channels Salid cannot see, and how to add it"
    ),
    "setup_denied": {
        "th": "คำสั่งนี้ใช้ได้เฉพาะเจ้าของเซิร์ฟเวอร์หรือผู้ที่มีสิทธิ์จัดการเซิร์ฟเวอร์ค่ะ",
        "en": "This command is only for the server owner or people who can manage the server.",
    },
    "setup_message": {
        "th": "ลิงก์เชื่อม {name} กับศูนย์บัญชาการของสลิด 🔗\n{url}\n"
        "เปิดลิงก์แล้วเข้าสู่ระบบด้วยบัญชี Google ของคุณ ลิงก์ใช้ได้ครั้งเดียวและหมดอายุใน 15 นาที "
        "อย่าส่งต่อให้คนอื่นค่ะ",
        "en": "Link to connect {name} to Salid's command center 🔗\n{url}\n"
        "Open the link and sign in with your Google account. It works once and expires in 15 "
        "minutes. Please do not share it.",
    },
    "setup_failed": {
        "th": "สร้างลิงก์ไม่สำเร็จ ลองพิมพ์ /ตั้งค่า อีกครั้งภายหลังค่ะ",
        "en": "Could not create the link. Please try /setup again later.",
    },
    "mode_denied": {
        "th": "ปรับโหมดได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change the mode.",
    },
    "mode_failed": {
        "th": "ตั้งโหมดไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not set the mode. Please try again later.",
    },
    "mode_current": {
        "th": "โหมดของห้องนี้ตอนนี้: {label}",
        "en": "This channel's mode right now: {label}",
    },
    "mode_snoozed": {
        "th": "(สลิดกำลังพักเงียบชั่วคราว ทำตัวเหมือนโหมดฟังเงียบ)",
        "en": "(Salid is snoozed for now and acts like Listen mode)",
    },
    "mode_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /โหมด แล้วเลือกโหมดค่ะ",
        "en": "Admins or executives can change it with /mode and then pick a mode.",
    },
    "mode_set": {"th": "ตั้งห้องนี้เป็น {label} แล้วค่ะ", "en": "This channel is now set to {label}."},
    "lang_current": {
        "th": "ภาษาที่สลิดตอบในห้องนี้ตอนนี้: {label}",
        "en": "Salid's reply language in this channel right now: {label}",
    },
    "lang_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /ภาษา แล้วเลือกภาษาค่ะ "
        "(หรือแท็กสลิดแล้วพิมพ์ “speak English” / “พูดไทย”)",
        "en": "Admins or executives can change it with /language and then pick a language "
        "(or tag Salid and say “speak English” / “speak Thai”).",
    },
    "lang_denied": {
        "th": "เปลี่ยนภาษาได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change the language.",
    },
    "lang_ask_done": {
        "th": "ได้ค่ะ จากนี้สลิดจะใช้ภาษา{label}เป็นภาษาหลักของเซิร์ฟเวอร์นี้ค่ะ "
        "(เปลี่ยนได้ทุกเมื่อ แค่บอกสลิดว่า “พูดภาษา…ทั้งเซิร์ฟเวอร์”)",
        "en": "Done: from now on I'll use {label} as this server's main language. "
        "You can change it any time, just tell me “speak … everywhere”.",
    },
    "lang_ask_denied": {
        "th": "ให้ผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารเป็นคนเลือกภาษาหลักนะคะ",
        "en": "Please ask a server admin or executive to pick the main language.",
    },
    "lang_failed": {
        "th": "ตั้งภาษาไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not set the language. Please try again later.",
    },
    "lang_set": {
        "th": "ตั้งภาษาของห้องนี้เป็น {label} แล้วค่ะ",
        "en": "This channel's language is now {label}.",
    },
    "server_lang_set": {
        "th": "เปลี่ยนสลิดเป็นภาษา{label}ทั้งเซิร์ฟเวอร์แล้วค่ะ",
        "en": "Salid now speaks {label} across the whole server.",
    },
    "server_lang_set_default": {
        "th": "ตั้งภาษาหลักของเซิร์ฟเวอร์เป็น{label}แล้วค่ะ (ห้องที่ตั้งภาษาไว้เองยังคงเดิม)",
        "en": "The server's default language is now {label} (channels with their own language are unchanged).",
    },
    # ---------------------------------------------------------------- tone presets (salid.tone)
    "tone_label_cheerful": {"th": "สดใส ร่าเริง", "en": "Bright & cheerful"},
    "tone_label_friendly": {"th": "เป็นกันเอง สบายๆ", "en": "Friendly & relaxed"},
    "tone_label_professional": {"th": "สุภาพ ทางการ", "en": "Professional & formal"},
    "tone_label_concise": {"th": "กระชับ", "en": "Concise"},
    "tone_label_calm": {"th": "ใจเย็น อบอุ่น", "en": "Calm & warm"},
    "tone_label_default": {"th": "ตามค่าเริ่มต้น", "en": "Default"},
    "tone_desc_cheerful": {
        "th": "ขี้เล่น อบอุ่น มีอีโมจิเบาๆ (ค่าเริ่มต้น)",
        "en": "playful and warm, light emoji (the default)",
    },
    "tone_desc_friendly": {
        "th": "คุยสบายๆ เหมือนเพื่อนร่วมงาน ไม่น่ารักเกินจริง อีโมจิไม่เกิน 1 ตัว",
        "en": "natural and relaxed like a coworker, no cutesy words, at most one emoji",
    },
    "tone_desc_professional": {
        "th": "ภาษาสุภาพแบบงานธุรกิจ ไม่มีอีโมจิ ไม่เล่นมุก ประโยคครบ",
        "en": "polite business English, no emoji, no jokes, complete sentences",
    },
    "tone_desc_concise": {
        "th": "ตอบสั้นที่สุดที่ยังครบ ใช้บูลเล็ตเมื่อมี 3 ข้อขึ้นไป ไม่มีอีโมจิ ไม่คุยเล่น",
        "en": "the shortest correct answer, bullets for 3+ items, no emoji, no small talk",
    },
    "tone_desc_calm": {
        "th": "ปลอบให้อุ่นใจ อดทน จังหวะช้า เหมาะกับห้องที่เครียด",
        "en": "reassuring and patient with a slower rhythm, good for stressful rooms",
    },
    "tone_list_line": {"th": "• {label} — {desc}", "en": "• {label} — {desc}"},
    "tone_current": {
        "th": "โทนของสลิดในห้องนี้ตอนนี้: {label}",
        "en": "Salid's tone in this channel right now: {label}",
    },
    "tone_current_me": {
        "th": "โทนที่สลิดใช้คุยกับคุณในห้องส่วนตัว/DM ตอนนี้: {label}",
        "en": "The tone Salid uses with you in your private channel / DMs right now: {label}",
    },
    "tone_current_note": {
        "th": "หมายเหตุของเซิร์ฟเวอร์: “{note}”",
        "en": "Server note: “{note}”",
    },
    "tone_presets_h": {"th": "โทนที่เลือกได้:", "en": "Available tones:"},
    "tone_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /โทน (เลือกโทน และ scope: เซิร์ฟเวอร์ ถ้าต้องการทั้งเซิร์ฟเวอร์) "
        "หรือแท็กสลิดแล้วพิมพ์ “เปลี่ยนโทนเป็นทางการ” / “change tone to professional” "
        "ส่วนในห้องส่วนตัวหรือ DM ทุกคนเปลี่ยนโทนของตัวเองได้",
        "en": "Admins or executives can change it with /tone (pick a tone, and scope: server for the whole "
        "server) or tag Salid and say “change tone to professional” / “เปลี่ยนโทนเป็นทางการ”. "
        "In your private channel or DMs anyone can change their own tone.",
    },
    "tone_denied": {
        "th": "เปลี่ยนโทนของห้องหรือของเซิร์ฟเวอร์ได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ "
        "(โทนของตัวเองเปลี่ยนได้ในห้องส่วนตัวหรือ DM)",
        "en": "Only server admins or executives can change the tone of a channel or the server "
        "(you can change your own tone in your private channel or DMs).",
    },
    "tone_failed": {
        "th": "ตั้งโทนไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not set the tone. Please try again later.",
    },
    "tone_unknown": {
        "th": "ไม่รู้จักโทนนั้นค่ะ เลือกได้: สดใส, เป็นกันเอง, ทางการ, กระชับ, ใจเย็น",
        "en": "I don't know that tone. Choose one of: cheerful, friendly, professional, concise, calm.",
    },
    "tone_note_too_long": {
        "th": "หมายเหตุยาวเกิน 200 ตัวอักษรค่ะ ช่วยย่อให้สั้นลงหน่อยนะคะ",
        "en": "The note is longer than 200 characters. Please shorten it.",
    },
    "tone_note_unsafe": {
        "th": "หมายเหตุนี้แตะเรื่องสิทธิ์ ความลับ หรือตัวตนของสลิด สลิดรับไว้ไม่ได้ค่ะ "
        "(หมายเหตุใช้ได้เฉพาะเรื่องวิธีพูด เช่น การเรียกลูกค้า)",
        "en": "That note touches permissions, secrets or Salid's identity, so I can't accept it "
        "(a note can only be about how to speak, e.g. how to address customers).",
    },
    "tone_note_scope": {
        "th": "หมายเหตุตั้งได้ที่ระดับทั้งเซิร์ฟเวอร์เท่านั้น (scope: เซิร์ฟเวอร์) ค่ะ",
        "en": "A note can only be set for the whole server (scope: server).",
    },
    "tone_note_set": {
        "th": "จดหมายเหตุเรื่องวิธีพูดของเซิร์ฟเวอร์ไว้แล้วค่ะ: “{note}”",
        "en": "Saved the server's wording note: “{note}”",
    },
    "tone_note_cleared": {
        "th": "ลบหมายเหตุเรื่องวิธีพูดของเซิร์ฟเวอร์แล้วค่ะ",
        "en": "Removed the server's wording note.",
    },
    "tone_reset_channel": {
        "th": "ได้ค่ะ ห้องนี้จะใช้โทนตามค่าเริ่มต้นของเซิร์ฟเวอร์ ({label}) นะคะ",
        "en": "Okay, this channel now follows the server's default tone ({label}).",
    },
    "tone_reset_me": {
        "th": "ได้ค่ะ ต่อไปจะใช้โทนตามห้องและเซิร์ฟเวอร์ ({label}) กับคุณนะคะ",
        "en": "Okay, I'll go back to the channel and server tone ({label}) with you.",
    },
    "tone_me_elsewhere": {
        "th": "(ใช้ในห้องส่วนตัวและ DM ของคุณ ส่วนในห้องรวมสลิดใช้โทนของห้อง)",
        "en": "(This applies in your private channel and DMs; in shared rooms I use the room's tone.)",
    },
    "tone_cmd_scope_choice_channel": {"th": "ห้องนี้", "en": "This channel"},
    "tone_cmd_scope_choice_server": {"th": "ทั้งเซิร์ฟเวอร์", "en": "Whole server"},
    "cmd_tone": _bi("ดูหรือตั้งโทนการพูดของสลิด", "View or set Salid's tone of voice"),
    "cmd_tone_arg": _bi("โทนใหม่ (ไม่เลือก = ดูโทนปัจจุบัน)", "New tone (empty = view)"),
    "cmd_tone_scope_arg": _bi(
        "ขอบเขต: ห้องนี้ (ค่าเริ่มต้น) หรือทั้งเซิร์ฟเวอร์", "Scope: this channel (default) or the whole server"
    ),
    "cmd_tone_note_arg": _bi(
        "หมายเหตุวิธีพูดของเซิร์ฟเวอร์ ไม่เกิน 200 ตัวอักษร (scope: เซิร์ฟเวอร์)",
        "Server wording note, up to 200 characters (scope: server)",
    ),
    "tone_set_cheerful_channel": {
        "th": "ได้เลยค่ะ ต่อจากนี้สลิดจะพูดแบบสดใสร่าเริงในห้องนี้นะคะ 😊",
        "en": "Sure thing! I'll keep it bright and cheerful in this channel from now on 😊",
    },
    "tone_set_cheerful_server": {
        "th": "ได้เลยค่ะ ต่อจากนี้สลิดจะพูดแบบสดใสร่าเริงทั้งเซิร์ฟเวอร์นะคะ 😊",
        "en": "Sure thing! I'll keep it bright and cheerful across the whole server from now on 😊",
    },
    "tone_set_cheerful_me": {
        "th": "ได้เลยค่ะ ต่อจากนี้สลิดจะคุยกับคุณแบบสดใสร่าเริงนะคะ 😊",
        "en": "Sure thing! I'll be bright and cheerful with you from now on 😊",
    },
    "tone_set_friendly_channel": {
        "th": "โอเคค่ะ ต่อไปจะคุยแบบสบายๆ เป็นกันเองในห้องนี้นะคะ",
        "en": "Okay, I'll keep things relaxed and casual in this channel from now on.",
    },
    "tone_set_friendly_server": {
        "th": "โอเคค่ะ ต่อไปจะคุยแบบสบายๆ เป็นกันเองทั้งเซิร์ฟเวอร์นะคะ",
        "en": "Okay, I'll keep things relaxed and casual across the whole server from now on.",
    },
    "tone_set_friendly_me": {
        "th": "โอเคค่ะ ต่อไปจะคุยกับคุณแบบสบายๆ เป็นกันเองนะคะ",
        "en": "Okay, I'll keep things relaxed and casual with you from now on.",
    },
    "tone_set_professional_channel": {
        "th": "รับทราบค่ะ ต่อจากนี้สลิดจะพูดแบบสุภาพทางการในห้องนี้ค่ะ",
        "en": "Done. I'll keep a professional tone in this channel from now on.",
    },
    "tone_set_professional_server": {
        "th": "รับทราบค่ะ ต่อจากนี้สลิดจะพูดแบบสุภาพทางการทั้งเซิร์ฟเวอร์ค่ะ",
        "en": "Done. I'll keep a professional tone across the whole server from now on.",
    },
    "tone_set_professional_me": {
        "th": "รับทราบค่ะ ต่อจากนี้สลิดจะพูดกับคุณแบบสุภาพทางการค่ะ",
        "en": "Done. I'll keep a professional tone with you from now on.",
    },
    "tone_set_concise_channel": {
        "th": "รับทราบค่ะ ห้องนี้จะตอบสั้นกระชับค่ะ",
        "en": "Got it. Short answers in this channel from now on.",
    },
    "tone_set_concise_server": {
        "th": "รับทราบค่ะ ทั้งเซิร์ฟเวอร์จะตอบสั้นกระชับค่ะ",
        "en": "Got it. Short answers across the whole server from now on.",
    },
    "tone_set_concise_me": {
        "th": "รับทราบค่ะ กับคุณจะตอบสั้นกระชับค่ะ",
        "en": "Got it. Short answers with you from now on.",
    },
    "tone_set_calm_channel": {
        "th": "ได้ค่ะ ต่อไปนี้สลิดจะคุยอย่างใจเย็นและอบอุ่นในห้องนี้นะคะ ไม่ต้องรีบนะคะ",
        "en": "Of course. I'll keep a calm, gentle tone in this channel from now on. No rush.",
    },
    "tone_set_calm_server": {
        "th": "ได้ค่ะ ต่อไปนี้สลิดจะคุยอย่างใจเย็นและอบอุ่นทั้งเซิร์ฟเวอร์นะคะ ไม่ต้องรีบนะคะ",
        "en": "Of course. I'll keep a calm, gentle tone across the whole server from now on. No rush.",
    },
    "tone_set_calm_me": {
        "th": "ได้ค่ะ ต่อไปนี้สลิดจะคุยกับคุณอย่างใจเย็นและอบอุ่นนะคะ ไม่ต้องรีบนะคะ",
        "en": "Of course. I'll keep a calm, gentle tone with you from now on. No rush.",
    },
    "lang_label_th": {"th": "ไทย", "en": "Thai"},
    "lang_label_en": {"th": "อังกฤษ (English)", "en": "English"},
    "lang_label_zh": {"th": "จีน (中文)", "en": "Chinese (中文)"},
    "lang_label_hi": {"th": "ฮินดี (हिन्दी)", "en": "Hindi (हिन्दी)"},
    "lang_label_ms": {"th": "มาเลย์ (Bahasa Melayu)", "en": "Malay (Bahasa Melayu)"},
    "lang_label_id": {"th": "อินโดนีเซีย (Bahasa Indonesia)", "en": "Indonesian (Bahasa Indonesia)"},
    "lang_label_vi": {"th": "เวียดนาม (Tiếng Việt)", "en": "Vietnamese (Tiếng Việt)"},
    "lang_label_ko": {"th": "เกาหลี (한국어)", "en": "Korean (한국어)"},
    "lang_label_ja": {"th": "ญี่ปุ่น (日本語)", "en": "Japanese (日本語)"},
    "lang_label_auto": {
        "th": "อัตโนมัติ (ตอบตามภาษาที่พิมพ์มา)",
        "en": "Auto (reply in the language you write in)",
    },
    "mode_label_off": {"th": "🔇 ปิด", "en": "🔇 Off"},
    "mode_label_listen": {"th": "👂 ฟังเงียบ", "en": "👂 Listen"},
    "mode_label_call": {"th": "🙋 เรียกแล้วคุย", "en": "🙋 Call"},
    "mode_label_join": {"th": "🎧 ร่วมวง", "en": "🎧 Join"},
    "mode_label_chat": {"th": "💬 ตอบทุกข้อความ", "en": "💬 Chat"},
    "mode_help_off": {
        "th": "สลิดไม่ยุ่งกับห้องนี้เลย (ไม่อ่านใบเสร็จ ไม่ตอบ แท็กก็ไม่ตอบ) และไม่บันทึกข้อความ",
        "en": "Salid stays out of this channel completely (no receipts, no replies, even if tagged) "
        "and nothing is recorded",
    },
    "mode_help_listen": {
        "th": "ฟังเงียบ ไม่พูดเอง อ่านใบเสร็จแล้วแปะอีโมจิ ตอบเมื่อถูกแท็กเท่านั้น (ไม่ตอบเมื่อเรียกชื่อเฉยๆ) "
        "สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Listens quietly and never speaks unprompted: reads receipts with an emoji and replies "
        "only when tagged (not when just called by name). Salid remembers this channel's messages "
        "for {days} days",
    },
    "mode_help_call": {
        "th": "ตอบเมื่อถูกแท็กหรือถูกเรียกชื่อ แล้วคุยต่อได้ 3 นาทีโดยไม่ต้องแท็ก สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Replies when tagged or called by name, then keeps chatting for 3 minutes without a tag. "
        "Salid remembers this channel's messages for {days} days",
    },
    "mode_help_join": {
        "th": "เหมือนโหมดเรียกแล้วคุย และสลิดอาจเข้าร่วมวงเองเมื่อมีคำถามที่ช่วยได้ (จำกัดชั่วโมงละ 3 ครั้ง) "
        "สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Like Call, and Salid may chime in on its own when it can help with a question "
        "(at most 3 times an hour). Salid remembers this channel's messages for {days} days",
    },
    "mode_help_chat": {
        "th": "ตอบทุกข้อความในห้อง สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Replies to every message in the channel. Salid remembers this channel's messages "
        "for {days} days",
    },
    "cmd_memory": _bi("ดูหรือตั้งจำนวนวันที่สลิดจำข้อความ", "View or set how long Salid remembers chats"),
    "cmd_memory_arg": _bi("จำนวนวัน 1-3650 (ไม่ใส่ = ดูสถานะ)", "Days 1-3650 (empty = view status)"),
    "memory_on": {"th": "เปิด", "en": "on"},
    "memory_off": {"th": "ปิดอยู่", "en": "off"},
    "memory_status": {
        "th": "ความจำของทีม: {state} · เก็บข้อความไว้ {days} วัน",
        "en": "Team memory: {state} · messages are kept for {days} days",
    },
    "memory_import": {
        "th": "นำเข้าประวัติแล้ว {done}/{total} ห้อง, {messages} ข้อความ, รูป {images}/{cap}",
        "en": "History imported: {done}/{total} channels, {messages} messages, images {images}/{cap}",
    },
    "memory_modes": {
        "th": "ห้องที่ถูกบันทึก: ทุกโหมดยกเว้น {off} (โหมด {recorded})",
        "en": "Recorded channels: every mode except {off} (modes {recorded})",
    },
    "memory_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนจำนวนวันได้ด้วย /ความจำ days:<จำนวนวัน> "
        "ปิดห้องเดียวด้วย /โหมด 🔇 หรือปิดทั้งเซิร์ฟเวอร์ที่หน้าศูนย์บัญชาการ",
        "en": "Admins or executives can change the days with /memory days:<number>. Switch one "
        "channel off with /mode 🔇, or the whole server in the command center.",
    },
    "memory_set": {
        "th": "ตั้งให้สลิดจำข้อความไว้ {days} วันแล้วค่ะ (ใช้กับข้อความเก่าด้วยในการล้างรอบถัดไป)",
        "en": "Salid will now keep messages for {days} days (this also applies to existing "
        "messages at the next daily cleanup).",
    },
    "memory_denied": {
        "th": "เปลี่ยนจำนวนวันที่จำข้อความได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change how long messages are kept.",
    },
    "memory_bad_days": {
        "th": "ใส่จำนวนวันเป็นตัวเลข {lo} ถึง {hi} ค่ะ",
        "en": "Please give a number of days from {lo} to {hi}.",
    },
    "memory_failed": {
        "th": "ตั้งค่าไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not save the setting. Please try again later.",
    },
    "memory_notice": {
        "th": "📝 ต่อไปนี้สลิดจะจำข้อความในห้องนี้ไว้ {days} วัน เพื่อช่วยตอบคำถามอย่าง “เมื่อวานคุยเรื่องอะไร” "
        "และจะให้ข้อมูลจากห้องนี้เฉพาะกับคนที่เห็นห้องนี้เท่านั้น "
        "ถ้าไม่ต้องการ พิมพ์ /โหมด แล้วเลือก 🔇 ปิด",
        "en": "📝 From now on Salid will remember messages in this channel for {days} days to help "
        "answer questions like “what did we discuss yesterday?”, and will only share information "
        "from it with people who can see this channel. To stop, type /mode and pick 🔇 Off.",
    },
    # ---------------------------------------------------------------- slash command texts
    # (Discord shows one description, so these are bilingual in both columns)
    "cmd_setup": _bi("เชื่อมเซิร์ฟเวอร์กับศูนย์บัญชาการ (ผู้ดูแล)", "Link server to command center (admins)"),
    "cmd_mode": _bi("ดูหรือตั้งโหมดการคุยของสลิดในห้องนี้", "View or set Salid's mode here"),
    "cmd_mode_arg": _bi("โหมดใหม่ (ไม่เลือก = ดูโหมดปัจจุบัน)", "New mode (empty = view)"),
    "cmd_language": _bi(
        "ภาษาส่วนตัวของคุณ (สลิดจะตอบคุณเพิ่มเป็นภาษานี้)", "Your personal language (Salid adds it for you)"
    ),
    "cmd_company_language": _bi(
        "ภาษาบริษัท ใช้ทั้งเซิร์ฟเวอร์ (ผู้บริหาร)", "Company language for the whole server (executives)"
    ),
    "cmd_language_arg": _bi("ภาษาใหม่ (ไม่เลือก = ดูภาษาปัจจุบัน)", "New language (empty = view)"),
    "cmd_show_captions": _bi(
        "แสดงข้อความถอดเสียงสดในแชตประชุมนี้", "Show live captions (ASR text) in this meeting"
    ),
    "cmd_hide_captions": _bi("หยุดแสดงข้อความถอดเสียงในประชุมนี้", "Hide live captions in this meeting"),
    "cmd_captions_default": _bi(
        "ค่าเริ่มต้นคำบรรยายสดของทุกประชุม", "Default for live captions in every meeting"
    ),
    "cmd_captions_on": _bi("เปิดคำบรรยายสดแล้วค่ะ", "Live captions are on."),
    "cmd_captions_off": _bi("ปิดคำบรรยายสดแล้วค่ะ", "Live captions are off."),
    "cmd_voice": _bi(
        "เสียงสลิดในประชุมนี้: ฟังอย่างเดียว/พูดได้",
        "Salid's voice here: listen only / may talk",
    ),
    "cmd_voice_arg": _bi("โหมด", "Mode"),
    "cmd_quiet": _bi("ให้สลิดเงียบจนกว่าจะเรียกชื่อ", "Salid goes quiet until called by name"),
    "cmd_quiet_done": _bi(
        "เงียบแล้วค่ะ เรียกชื่อสลิดเมื่อต้องการ", "Quiet now. Call my name when you need me."
    ),
    "cmd_no_meeting": _bi(
        "ตอนนี้ไม่มีประชุมที่สลิดอยู่ในห้องนี้", "There is no meeting with Salid here right now."
    ),
    "cmd_join": _bi("ให้สลิดเข้าห้องเสียงเพื่อจดบันทึกประชุม", "Salid joins your voice channel"),
    "cmd_leave": _bi("ให้สลิดออกจากห้องเสียงและสรุปการประชุม", "Salid leaves and summarizes"),
    "cmd_summary": _bi("สรุปสิ่งที่คุยกันในห้อง", "Summarize what was discussed"),
    "cmd_summary_period": _bi("ช่วงเวลา (ค่าเริ่มต้น วันนี้)", "Period (default today)"),
    "cmd_summary_channel": _bi(
        "ห้องที่จะสรุป (ค่าเริ่มต้น ห้องนี้)", "Channel to summarize (default this one)"
    ),
    "cmd_summary_all": _bi("ทุกห้องที่ฉันเห็น", "All channels I can see"),
    "cmd_summary_topic": _bi("เรื่องที่อยากให้เน้น (ไม่บังคับ)", "Topic to focus on (optional)"),
    "cmd_summary_private": _bi("เห็นคนเดียว", "Only visible to me"),
    "period_today": _bi("วันนี้", "Today"),
    "period_yesterday": _bi("เมื่อวาน", "Yesterday"),
    "period_7d": _bi("7 วัน", "Last 7 days"),
    "period_30d": _bi("30 วัน", "Last 30 days"),
    "menu_summarize": {"th": "สรุปด้วยสลิด", "en": "Summarize with Salid"},
    "menu_translate": {"th": "แปลด้วยสลิด", "en": "Translate with Salid"},
    "translate_nothing": _bi(
        "ข้อความนี้ไม่มีตัวอักษรให้แปลค่ะ", "There is no text in that message to translate."
    ),
    "translate_failed": _bi(
        "แปลไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ", "I couldn't translate that, please try again."
    ),
    "summary_q_channel": {
        "th": "สรุปสิ่งที่คุยกันใน #{channel} ช่วง {dates}: หัวข้อ, สิ่งที่ตัดสินใจ, ใครรับงานอะไร, "
        "ตัวเลขสำคัญ, เรื่องที่ค้าง",
        "en": "Summarize what was discussed in #{channel} for {dates}: topics, decisions made, "
        "who took which task, key numbers, open items",
    },
    "summary_q_all": {
        "th": "สรุปสิ่งที่คุยกันในทุกห้องที่ฉันเห็น ช่วง {dates} แยกตามห้อง: หัวข้อ, สิ่งที่ตัดสินใจ, "
        "ใครรับงานอะไร, ตัวเลขสำคัญ, เรื่องที่ค้าง",
        "en": "Summarize what was discussed in all the channels I can see for {dates}, per channel: "
        "topics, decisions made, who took which task, key numbers, open items",
    },
    "summary_focus": {"th": " [เน้น: {topic}]", "en": " [focus: {topic}]"},
    "summary_dates_one": {"th": "{start} (เขตเวลา {tz})", "en": "{start} (time zone {tz})"},
    "summary_dates_range": {
        "th": "{start} ถึง {end} (เขตเวลา {tz})",
        "en": "{start} to {end} (time zone {tz})",
    },
    "summary_msg_q": {
        "th": "สรุปข้อความนี้ของ {author}: หัวข้อ ประเด็นสำคัญ ตัวเลข และสิ่งที่ต้องทำต่อ "
        "(ถ้ามีไฟล์แนบหรือลิงก์ ให้อ่านและสรุปด้วย อย่าบันทึกเป็นใบเสร็จ)",
        "en": "Summarize this message from {author}: topic, key points, numbers and next steps "
        "(read and summarize any attached files or links too; do not record anything as a receipt)",
    },
    "summary_msg_text": {"th": "ข้อความที่ต้องสรุป:\n{text}", "en": "Message to summarize:\n{text}"},
    "summary_unavailable": {
        "th": "สรุปไม่ได้ตอนนี้ เพราะการถามตอบหรือความจำของทีมถูกปิดอยู่ (เปิดได้ที่หน้าศูนย์บัญชาการ)",
        "en": "I can't summarize right now: Q&A or team memory is switched off (turn it on in the command center).",
    },
    "summary_no_access": {
        "th": "คุณไม่มีสิทธิ์เห็นห้องนั้น หรือสลิดไม่ได้จดห้องนั้นไว้ค่ะ",
        "en": "You can't see that channel, or Salid isn't recording it.",
    },
    "summary_nothing": {
        "th": "ไม่มีข้อความที่สรุปได้ค่ะ (ข้อความว่าง ไม่มีไฟล์หรือลิงก์)",
        "en": "There is nothing to summarize (no text, files or links).",
    },
    # ---------------------------------------------------------------- meeting
    "meet_announce": {
        "th": "สลิดเข้ามาฟังและจดบันทึกการประชุมค่ะ 🎙️ "
        "(เก็บเป็นข้อความในหน่วยความจำเท่านั้น ไม่บันทึกเสียง) "
        "เรียก “สลิด” แล้วถามได้เลย หรือพิมพ์ /ออกประชุม เพื่อสรุปและให้สลิดออก",
        "en": "Salid has joined to listen and take meeting notes 🎙️ "
        "(kept as text in memory only, no audio is recorded). "
        "Say “Salid” and ask away, or type /leave-meeting to get the summary and let me go.",
    },
    # Fallbacks only: the consent question is normally generated. ค่ะ / คะ = voice particles.
    "meet_consent_ask_1": {
        "th": "สลิดขออนุญาตจดประชุมนี้นะคะ ถ้าไม่สะดวกบอกได้เลยค่ะ",
        "en": "Salid would like to take notes of this meeting. Just say if you'd rather not.",
    },
    "meet_consent_ask_2": {
        "th": "ขออนุญาตจดบันทึกการประชุมนะคะ ถ้าไม่ต้องการ บอกสลิดได้เลยนะคะ",
        "en": "May I take notes here? Tell me if that's not okay.",
    },
    "meet_consent_ask_3": {
        "th": "สลิดนั่งฟังและจดการประชุมอยู่นะคะ ถ้าไม่สะดวกบอกได้เลยค่ะ",
        "en": "I'm listening in to take notes. Let me know if you'd prefer I didn't.",
    },
    "meet_dismiss_ack": {"th": "ได้ค่ะ", "en": "Okay!"},
    # Visible state of the meeting (one short line in the meeting's text chat on every change)
    "voice_state_join": {
        "th": "👂 สลิดเข้ามาฟังและจดประชุมแล้วค่ะ เรียกชื่อ “สลิด” เมื่อต้องการนะคะ",
        "en": "👂 Salid joined to listen and take notes. Say “Salid” when you need me.",
    },
    "voice_state_engaged_name": {
        "th": "💬 สลิดกำลังคุยกับคุณ{name}อยู่ค่ะ",
        "en": "💬 Salid is talking with {name}.",
    },
    "voice_state_engaged": {
        "th": "💬 สลิดกำลังคุยอยู่ค่ะ",
        "en": "💬 Salid is in the conversation.",
    },
    "voice_state_listening": {
        "th": "👂 สลิดกลับไปฟังอย่างเดียวแล้วค่ะ",
        "en": "👂 Salid is back to listening only.",
    },
    # why a live-mode state line changed (appended to the line above)
    "voice_why_called": {"th": " · เรียกชื่อ", "en": " · called by name"},
    "voice_why_ended": {"th": " · จบบทสนทนา", "en": " · conversation ended"},
    "voice_why_thanks": {"th": " · ขอบคุณแล้ว", "en": " · said thanks"},
    "voice_why_closed": {"th": " · ปิดบทสนทนาแล้ว", "en": " · closed it"},
    # {mins} = engaged_timeout_seconds in minutes (built by MeetingManager._engage_min)
    "voice_why_quiet": {"th": " · เงียบไป {mins} นาที", "en": " · quiet {mins} min"},
    "voice_why_bare": {"th": " · ไม่มีใครพูดต่อ", "en": " · no one went on"},
    "voice_why_hushed": {"th": " · ถูกบอกให้เงียบ", "en": " · told to be quiet"},
    "voice_answer_echo": {"th": "💬 **สลิด:** {text}", "en": "💬 **Salid:** {text}"},
    "voice_tool_failed": {
        "th": "⚠️ สลิดตรวจสอบให้ไม่ได้ค่ะ: {tool} ขัดข้อง ({reason})",
        "en": "⚠️ Salid could not check that: {tool} failed ({reason})",
    },
    "voice_tool_failed_say": {
        "th": "ขออภัยค่ะ ตรวจสอบให้ไม่ได้ตอนนี้ เครื่องมือค้นข้อมูลขัดข้องค่ะ",
        "en": "Sorry, I can't check that right now. The lookup tool is not working.",
    },
    "voice_working_text": {
        "th": "⏳ กำลังหาข้อมูล: {question}",
        "en": "⏳ Looking into it: {question}",
    },
    "voice_handoff_text": {
        "th": "⏳ ส่งไปทำพื้นหลังแล้ว: {question} (จะแจ้งเมื่อเสร็จ)",
        "en": "⏳ Sent to the background: {question} (I'll tell you when it's done)",
    },
    "voice_still_working_text": {
        "th": "⏳ ยังทำอยู่ค่ะ: {question}",
        "en": "⏳ Still working on it: {question}",
    },
    "voice_job_done": {"th": "✅ เสร็จแล้ว: {question}", "en": "✅ Done: {question}"},
    "voice_still_suffix": {"th": " · ยังทำอยู่", "en": " · still working"},
    "voice_bg_1": {
        "th": "งานนี้ใช้เวลาหน่อยค่ะ สลิดทำให้อยู่ เดี๋ยวได้แล้วจะบอกนะคะ คุยเรื่องอื่นต่อได้เลยค่ะ",
        "en": "This one takes a little while. I'm on it and I'll tell you when it's ready; keep talking meanwhile.",
    },
    "voice_bg_2": {
        "th": "ขอเวลาทำสักพักนะคะ เสร็จแล้วสลิดจะแจ้งค่ะ ระหว่างนี้คุยเรื่องอื่นได้เลยค่ะ",
        "en": "Give me a little while. I'll announce it as soon as it's done, so go on with whatever you like.",
    },
    "voice_bg_3": {
        "th": "อันนี้นานหน่อยค่ะ สลิดทำอยู่เบื้องหลังนะคะ ได้ผลแล้วจะบอกค่ะ",
        "en": "That will take some time. I'm working on it in the background and will say when I have it.",
    },
    "voice_still_working": {
        "th": "ยังทำอยู่ค่ะ",
        "en": "Still working on it.",
    },
    "voice_jobs_full": {
        "th": "ขอให้งานก่อนหน้าเสร็จก่อนนะคะ",
        "en": "Let me finish the earlier jobs first.",
    },
    "voice_files_echo": {"th": "📎 ไฟล์ที่ขอไว้ค่ะ", "en": "📎 The files you asked for."},
    "voice_files_say": {
        "th": "ส่งไฟล์ไว้ในแชตแล้วนะคะ",
        "en": "I put the file in the chat.",
    },
    "ack_longer_1": {
        "th": "ขออภัยค่ะ รอบนี้นานกว่าปกตินะคะ ยังหาอยู่ค่ะ",
        "en": "Sorry, this is taking longer than usual. Still on it.",
    },
    "ack_longer_2": {
        "th": "ช้ากว่าปกติหน่อยนะคะ ขอบคุณที่รอค่ะ ยังทำอยู่ค่ะ",
        "en": "It's taking longer than usual, thanks for waiting. Still working.",
    },
    "meet_not_recording": {
        "th": "งั้นสลิดไม่จดนะคะ มีอะไรเรียกได้เลยค่ะ",
        "en": "Then I won't take notes. Just call me if you need anything.",
    },
    "meet_consent_no_1": {
        "th": "ได้ค่ะ งั้นสลิดขอตัวนะคะ",
        "en": "Okay, I'll leave you to it.",
    },
    "meet_consent_no_2": {
        "th": "รับทราบค่ะ สลิดไม่จดแล้วนะคะ ขอตัวก่อน",
        "en": "Understood, no notes. I'll head out.",
    },
    "meet_consent_no_3": {
        "th": "ไม่เป็นไรค่ะ สลิดออกไปก่อนนะคะ",
        "en": "No problem, I'll step out.",
    },
    "minutes_set_spoken": {
        "th": "ได้เลยค่ะ เดี๋ยวส่งสรุปไปที่ห้อง {channel} นะคะ",
        "en": "Sure, I'll send the minutes to {channel}.",
    },
    "minutes_not_found_spoken": {
        "th": "หาห้องที่บอกไม่เจอค่ะ ลองบอกชื่อห้องอีกทีนะคะ",
        "en": "I couldn't find that channel. Please say its name again.",
    },
    "minutes_ambiguous_spoken": {
        "th": "มีหลายห้องที่ชื่อใกล้เคียงกันค่ะ ช่วยบอกชื่อให้ชัดขึ้นหน่อยนะคะ",
        "en": "Several channels match. Please be more specific.",
    },
    "minutes_not_allowed_spoken": {
        "th": "ห้องนั้นคนในที่ประชุมบางคนยังมองไม่เห็นค่ะ เลยส่งสรุปไปที่นั่นไม่ได้",
        "en": "Not everyone here can see that channel, so I can't send the minutes there.",
    },
    "minutes_forwarded": {
        "th": "📝 สรุปการประชุม #{room} ({when}) ส่งต่อโดย {who}\n",
        "en": "📝 Minutes of the meeting in #{room} ({when}), forwarded by {who}\n",
    },
    "meet_not_in_voice": {
        "th": "ต้องเข้าห้องเสียง (Voice Channel) ก่อน แล้วค่อยเรียกสลิดเข้าประชุมนะคะ",
        "en": "Please join a voice channel first, then ask Salid to join the meeting.",
    },
    "meet_already": {
        "th": "สลิดกำลังอยู่ในการประชุมของเซิร์ฟเวอร์นี้แล้วค่ะ ({channel}) พิมพ์ /ออกประชุม เพื่อจบก่อนนะคะ",
        "en": "Salid is already in a meeting on this server ({channel}). Type /leave-meeting to end it first.",
    },
    "meet_joining": {"th": "กำลังเข้า", "en": "joining now"},
    "meet_none": {
        "th": "ตอนนี้สลิดไม่ได้อยู่ในการประชุมค่ะ",
        "en": "Salid is not in a meeting right now.",
    },
    "meet_leave_denied": {
        "th": "ต้องอยู่ในห้องประชุมเดียวกัน (หรือเป็นผู้ดูแลเซิร์ฟเวอร์) ถึงจะสั่งให้สลิดออกได้ค่ะ",
        "en": "You need to be in the same meeting channel (or be a server admin) to tell Salid to leave.",
    },
    "meet_off": {
        "th": "ตอนนี้ปิดฟังการประชุมอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Meeting listening is switched off right now. You can turn it on in Salid's command center.",
    },
    "meet_connect_failed": {
        "th": "เข้าห้องเสียงไม่สำเร็จค่ะ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not join the voice channel. Please try again later.",
    },
    "meet_joined": {"th": "สลิดเข้าห้อง {channel} แล้วค่ะ", "en": "Salid has joined {channel}."},
    "meet_left": {
        "th": "สลิดออกจากการประชุมแล้วค่ะ กำลังสรุปให้ในแชตของห้องเสียง",
        "en": "Salid has left the meeting. The summary is coming in the voice channel's chat.",
    },
    "meet_left_room": {"th": "👋 ออกจากห้องแล้วค่ะ", "en": "👋 Left the room"},
    "meet_wake_empty": {"th": "คะ? ว่ามาได้เลยค่ะ", "en": "Yes? Go ahead."},
    "meet_wake_spoken": {"th": "ค่ะ? ว่ามาได้เลยค่ะ", "en": "Yes? Go ahead."},
    "meet_greet_spoken": {"th": "สวัสดีค่ะ มีอะไรให้ช่วยคะ", "en": "Hi! How can I help?"},
    "meet_bye_spoken": {"th": "แล้วเจอกันนะคะ", "en": "Bye for now!"},
    "meet_no_transcript": {
        "th": "จบการประชุมแล้วค่ะ แต่ยังไม่ได้ยินเนื้อหาให้สรุปเลย 🙂",
        "en": "The meeting is over, but I did not hear anything to summarize 🙂",
    },
    "meet_summary_failed": {
        "th": "จบการประชุมแล้วค่ะ แต่สรุปไม่สำเร็จ ขออภัยด้วยนะคะ",
        "en": "The meeting is over, but the summary failed. Sorry about that.",
    },
    "meet_summary_header": {
        "th": "📝 สรุปการประชุม ({reason})\n",
        "en": "📝 Meeting summary ({reason})\n",
    },
    "meet_participant": {"th": "ผู้ร่วมประชุม {n}", "en": "Participant {n}"},
    # Attendance log (who entered / left the voice room, and when)
    "attend_joined": {
        "th": "{icon} {time} {names} เข้าห้องประชุม",
        "en": "{icon} {time} {names} joined the meeting",
    },
    "attend_left": {
        "th": "{icon} {time} {names} ออกจากห้องประชุม",
        "en": "{icon} {time} {names} left the meeting",
    },
    "attend_header": {"th": "👥 ผู้เข้าร่วม", "en": "👥 Attendees"},
    "attend_total": {"th": "ทั้งหมด {n} คน", "en": "Total: {n}"},
    "attend_minutes": {"th": "{n} นาที", "en": "{n} min"},
    "attend_lt_minute": {"th": "ไม่ถึง 1 นาที", "en": "under 1 min"},
    "mdoc_attendance": {"th": "ผู้เข้าร่วม", "en": "Attendance"},
    "mdoc_col_present": {"th": "ช่วงเวลาในห้อง", "en": "Time in the room"},
    "mdoc_col_total": {"th": "รวม (นาที)", "en": "Total (min)"},
    "mdoc_title": {"th": "รายงานการประชุม", "en": "Meeting minutes"},
    "mdoc_room": {"th": "ห้องประชุม", "en": "Meeting room"},
    "mdoc_date": {"th": "วันที่", "en": "Date"},
    "mdoc_time": {"th": "เวลา", "en": "Time"},
    "mdoc_minutes": {"th": "{n} นาที", "en": "{n} min"},
    "mdoc_attendees": {"th": "ผู้เข้าร่วม", "en": "Attendees"},
    "mdoc_transcript": {"th": "บทพูดของแต่ละคน", "en": "Transcript"},
    "mdoc_note": {
        "th": "ถอดความจากเสียงอัตโนมัติ อาจมีคำผิดหรือคลาดเคลื่อน",
        "en": "Auto-transcribed from speech; it may contain errors.",
    },
    "mdoc_note_clean": {
        "th": "ถอดความจากเสียงอัตโนมัติ แล้วแก้คำผิดชัดเจนและตัดคำฟุ่มเฟือย โดยไม่เปลี่ยนความหมาย อาจยังมีคลาดเคลื่อน",
        "en": "Auto-transcribed from speech, then obvious errors and filler words were tidied without changing the meaning; it may still contain errors.",
    },
    "mdoc_no_transcript": {
        "th": "ไม่มีบทพูดของการประชุมนี้ (ไม่ได้เก็บไว้หรือหมดอายุแล้ว)",
        "en": "No transcript is available for this meeting (not kept or expired).",
    },
    "mdoc_stats": {"th": "สัดส่วนการพูด", "en": "Speaker stats"},
    "mdoc_col_person": {"th": "ผู้พูด", "en": "Speaker"},
    "mdoc_col_turns": {"th": "จำนวนครั้งที่พูด", "en": "Turns"},
    "mdoc_col_share": {"th": "สัดส่วนโดยประมาณ", "en": "Approx. share"},
    "mdoc_salid": {"th": "สลิด", "en": "Salid"},
    "mdoc_private_caption": {
        "th": "📎 รายงานการประชุม #{room} พร้อมบทพูดของแต่ละคนค่ะ (ส่งให้คุณคนเดียว)",
        "en": "📎 Minutes of the meeting in #{room} with the transcript (sent to you only)",
    },
    "end_command": {
        "th": "ผู้ใช้สั่งให้ออกจากการประชุม",
        "en": "someone told Salid to leave",
    },
    "end_empty": {"th": "ทุกคนออกจากห้องแล้ว", "en": "everyone left the channel"},
    "end_timeout": {"th": "ประชุมครบ 6 ชั่วโมง", "en": "the meeting reached 6 hours"},
    "captions_suffix": {
        "th": "📝 คำบรรยายเปิดอยู่ · พูด “ซ่อนคำบรรยาย” เพื่อปิด",
        "en": "📝 captions on · say “hide captions” to stop",
    },
    "captions_default_on": {
        "th": "ได้ค่ะ ประชุมครั้งต่อไปสลิดจะแสดงคำบรรยายสดในแชตให้ทุกครั้ง (ซ่อนเฉพาะรอบได้ด้วย “ซ่อนคำบรรยาย”)",
        "en": "Done: every meeting from now on starts with live captions in the chat (say “hide captions” to hide them in one meeting).",
    },
    "captions_default_off": {
        "th": "ได้ค่ะ ประชุมครั้งต่อไปจะไม่แสดงคำบรรยายสดเอง (สั่งเปิดเฉพาะรอบได้ด้วย “แสดงคำบรรยาย”)",
        "en": "Done: meetings no longer start with live captions (say “show captions” to turn them on in one meeting).",
    },
    "captions_default_denied": {
        "th": "เฉพาะแอดมินหรือผู้บริหารที่ตั้งค่าคำบรรยายของทุกประชุมได้ค่ะ",
        "en": "Only admins and executives can change the captions default for every meeting.",
    },
    "voice_talk_listen": {
        "th": "🔇 ฟังและจดอย่างเดียว (สลิดจะไม่พูด ถามได้ทางข้อความ)",
        "en": "🔇 Listening only: Salid will not speak; ask in text",
    },
    "voice_talk_may": {
        "th": "🔊 สลิดพูดได้เมื่อถูกเรียก",
        "en": "🔊 Salid may talk when called",
    },
    "voice_talk_denied": {
        "th": "เฉพาะผู้จัดประชุมหรือผู้บริหารที่สลับโหมดเสียงของประชุมนี้ได้ค่ะ",
        "en": "Only the person who started this meeting or an executive can change Salid's voice here.",
    },
    "voice_talk_default_listen": {
        "th": "ได้ค่ะ ประชุมครั้งต่อไปสลิดจะฟังและจดอย่างเดียว ไม่พูด (สั่ง “สลิด พูดได้” เฉพาะรอบได้)",
        "en": "Done: in meetings from now on Salid listens and takes notes only (say “Salid, you may talk” in one meeting to change it).",
    },
    "voice_talk_default_may": {
        "th": "ได้ค่ะ ประชุมครั้งต่อไปสลิดพูดได้เมื่อถูกเรียก",
        "en": "Done: in meetings from now on Salid may talk when called.",
    },
    "voice_talk_default_denied": {
        "th": "เฉพาะแอดมินหรือผู้บริหารที่ตั้งค่าเสียงของสลิดในทุกประชุมได้ค่ะ",
        "en": "Only admins and executives can change Salid's voice default for every meeting.",
    },
    "meet_rejoined": {
        "th": "สลิดกลับเข้ามาจดประชุมต่อแล้วค่ะ (หลุดไปครู่หนึ่ง)",
        "en": "Salid is back and taking notes again (it dropped out for a moment).",
    },
    "meet_reconnected": {
        "th": "สลิดหลุดไปครู่หนึ่ง แต่กลับเข้าห้องและจดประชุมต่อแล้วค่ะ",
        "en": "Salid dropped out for a moment but is back in the room and still taking notes.",
    },
    "meet_reconnect_failed": {
        "th": "สลิดต่อเข้าห้องเสียงกลับไม่ได้ค่ะ ขอปิดการประชุมและสรุปให้ตอนนี้",
        "en": "Salid could not reconnect to the voice channel, so it is ending the meeting now.",
    },
    "meet_max_warning": {
        "th": "ประชุมใกล้ครบ 6 ชั่วโมงแล้ว สลิดจะออกและสรุปในอีก 10 นาทีค่ะ",
        "en": "The meeting hits the 6-hour limit in 10 minutes; Salid will then leave and summarize.",
    },
    "end_disconnected": {"th": "สลิดหลุดจากห้องเสียง", "en": "Salid was disconnected"},
    "end_shutdown": {"th": "สลิดกำลังปิดระบบ", "en": "Salid is shutting down"},
    # ---------------------------------------------------------------- digest
    "digest_title": {
        "th": "📒 สรุปค่าใช้จ่ายประจำวันที่ {day}",
        "en": "📒 Daily expense summary for {day}",
    },
    "digest_none": {
        "th": "ไม่มีใบเสร็จที่บันทึกในวันนี้",
        "en": "No receipts were recorded today",
    },
    "digest_total": {
        "th": "ยอดรวม: {total} ({count} ใบเสร็จ)",
        "en": "Total: {total} ({count} receipts)",
    },
    "digest_by_category": {"th": "แยกตามหมวด:", "en": "By category:"},
    "digest_top_vendors": {
        "th": "ร้านที่ใช้จ่ายสูงสุด 3 อันดับ:",
        "en": "Top 3 vendors by spending:",
    },
    "digest_flagged": {
        "th": "⚠️ รายการที่ต้องตรวจสอบ ({n}):",
        "en": "⚠️ Needs review ({n}):",
    },
    "digest_dup": {"th": "อาจซ้ำ", "en": "possible duplicate"},
    "digest_dups": {
        "th": "🔁 อาจซ้ำ {n} รายการ รวม {amount} รอตรวจ",
        "en": "🔁 Possible duplicates: {n} receipts, {amount} awaiting review",
    },
    "delete_denied": {
        "th": "ลบได้เฉพาะคนที่สลิดตอบอยู่ ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหารเท่านั้นค่ะ",
        "en": "Only the person Salid was answering, a server manager or an executive can delete this.",
    },
    "receipt_void_ack": {
        "th": "ยกเลิกใบเสร็จ #{id} แล้ว (กู้คืนได้)",
        "en": "Receipt #{id} is cancelled (it can be restored).",
    },
    "cancel_done": {
        "th": "ยกเลิกงานใหญ่ที่กำลังทำอยู่แล้วค่ะ",
        "en": "The running big task has been cancelled.",
    },
    "research_ack": {
        "th": "กำลังค้นข้อมูลให้นะคะ ประมาณ 1–2 นาที เสร็จแล้วสลิดส่งรายงานพร้อมไฟล์ PDF ให้ที่ห้องนี้ค่ะ (สั่ง “ยกเลิกงาน” ได้ถ้าไม่ต้องการแล้ว)",
        "en": "I'm researching this now, about 1-2 minutes. I'll post a report with a PDF here when it's done (say “cancel task” to stop).",
    },
    "research_busy": {
        "th": "ตอนนี้สลิดกำลังค้นข้อมูลเชิงลึกเรื่องอื่นอยู่ค่ะ รอให้เสร็จก่อนนะคะ",
        "en": "I'm already running another deep research here. Please wait until it finishes.",
    },
    "research_limit": {
        "th": "วันนี้ค้นข้อมูลเชิงลึกครบ {n} ครั้งแล้วค่ะ ลองใหม่พรุ่งนี้ หรือให้สลิดค้นแบบสั้นๆ (web_search) แทนก็ได้นะคะ",
        "en": "The daily limit of {n} deep researches has been reached. Try again tomorrow, or I can do a quick web search instead.",
    },
    "research_done": {
        "th": "ค้นข้อมูลเสร็จแล้วค่ะ รายงานฉบับเต็มอยู่ในไฟล์ PDF:",
        "en": "The research is done. The full report is in the attached PDF:",
    },
    "research_failed": {
        "th": "ขอโทษค่ะ ค้นข้อมูลไม่สำเร็จ ลองใหม่อีกครั้งนะคะ",
        "en": "Sorry, the research failed. Please try again.",
    },
    "research_cancelled": {
        "th": "หยุดการค้นข้อมูลให้แล้วค่ะ",
        "en": "Research stopped.",
    },
    "research_nothing": {
        "th": "## สรุป\nค้นเว็บแล้วไม่ได้ข้อมูลที่ใช้ได้ค่ะ ลองถามให้เฉพาะเจาะจงขึ้นอีกนิดนะคะ",
        "en": "## Summary\nThe web search returned nothing usable. Try a more specific question.",
    },
    "research_sources": {"th": "แหล่งอ้างอิง", "en": "Sources"},
    "research_raw_head": {
        "th": "ผลการค้นหา (ยังไม่ได้สรุป)",
        "en": "Search results (not summarised)",
    },
    "research_title": {"th": "รายงานการค้นคว้า", "en": "Research report"},
    "research_dated": {"th": "ข้อมูล ณ วันที่ {at}", "en": "Information as of {at}"},
    # ---------------------------------------------------------------- video ads (videoad_bot)
    "video_started": {
        "th": "กำลังทำวิดีโอโฆษณาให้ค่ะ ใช้เวลาสักครู่ เสร็จแล้วจะส่งไว้ที่ห้องนี้นะคะ",
        "en": "Making your video ad. It takes a few minutes; I'll post it here when it's ready.",
    },
    "video_step_storyboard": {"th": "กำลังวางสตอรีบอร์ด…", "en": "Planning the storyboard…"},
    "video_step_stills": {"th": "กำลังทำภาพแต่ละฉาก…", "en": "Making the scene pictures…"},
    "video_step_hero": {"th": "กำลังทำช็อตเคลื่อนไหว…", "en": "Making the motion shots…"},
    "video_step_audio": {"th": "กำลังใส่เสียงบรรยาย…", "en": "Recording the narration…"},
    "video_step_render": {"th": "กำลังตัดต่อวิดีโอ…", "en": "Editing the video…"},
    "video_done": {
        "th": "วิดีโอโฆษณาเสร็จแล้วค่ะ (สร้างด้วย AI)",
        "en": "Your video ad is ready (made with AI).",
    },
    "video_scenes": {"th": "ลำดับฉาก:", "en": "Scenes:"},
    "video_caption": {"th": "แคปชันแนะนำ:", "en": "Suggested caption:"},
    "video_failed": {
        "th": "ขอโทษค่ะ ทำวิดีโอไม่สำเร็จ ({reason}) ลองสั่งใหม่อีกครั้งได้นะคะ",
        "en": "Sorry, the video ad could not be made ({reason}). Please try again.",
    },
    "video_reason_storyboard": {
        "th": "วางสตอรีบอร์ดไม่ได้",
        "en": "the storyboard did not work out",
    },
    "video_reason_render": {"th": "ตัดต่อวิดีโอไม่สำเร็จ", "en": "the video could not be rendered"},
    "video_reason_error": {"th": "เกิดข้อผิดพลาด", "en": "something went wrong"},
    "video_upload_refused": {
        "th": "วิดีโอทำเสร็จแล้ว แต่แนบไฟล์ในห้องนี้ไม่ได้ค่ะ (สลิดอาจไม่มีสิทธิ์แนบไฟล์ หรือไฟล์ใหญ่เกินไป)",
        "en": "The video was made, but I couldn't attach it here (I may not have permission to "
        "attach files, or it is too large).",
    },
    # burned into every video ad, and the end of its suggested caption
    "video_ai_label": {"th": "สร้างด้วย AI", "en": "Made with AI"},
    "video_ai_hashtag": {"th": "#สร้างด้วยAI", "en": "#MadeWithAI"},
    "video_cancelling": {
        "th": "กำลังยกเลิกการทำวิดีโอค่ะ…",
        "en": "Cancelling the video ad…",
    },
    "video_cancelled": {"th": "ยกเลิกการทำวิดีโอแล้วค่ะ", "en": "The video ad was cancelled."},
    "video_restarted": {
        "th": "วิดีโอโฆษณาที่สั่งไว้ทำไม่เสร็จเพราะสลิดรีสตาร์ต รบกวนสั่งใหม่อีกครั้งนะคะ",
        "en": "Your video ad was not finished because Salid restarted. Please ask again.",
    },
    "video_busy": {
        "th": "ตอนนี้สลิดกำลังทำวิดีโออยู่ค่ะ รอให้เสร็จก่อนแล้วค่อยสั่งใหม่นะคะ",
        "en": "Salid is already making a video right now. Please try again when it is done.",
    },
    "video_month_limit": {
        "th": "เดือนนี้เซิร์ฟเวอร์นี้ทำวิดีโอโฆษณาครบ {n} ชิ้นแล้วค่ะ",
        "en": "This server has used all {n} video ads for this month.",
    },
    "video_cancel_button": {"th": "ยกเลิก", "en": "Cancel"},
    "video_not_yours": {
        "th": "เฉพาะคนที่สั่งหรือผู้ดูแลเซิร์ฟเวอร์เท่านั้นที่ยกเลิกได้ค่ะ",
        "en": "Only the person who asked or a server admin can cancel this.",
    },
    "cancel_none": {
        "th": "ตอนนี้ไม่มีงานใหญ่ที่กำลังทำอยู่ค่ะ",
        "en": "There is no big task running right now.",
    },
    "cancel_denied": {
        "th": "ยกเลิกงานใหญ่ได้เฉพาะคนที่สั่งงานนั้น ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหารเท่านั้นค่ะ",
        "en": "Only the person who started the task, a server manager or an executive can cancel it.",
    },
    "cmd_cancel": _bi("ยกเลิกงานใหญ่ที่กำลังทำอยู่", "Cancel the running big task"),
    "corr_void_done": {
        "th": "🗑️ หักใบเสร็จ #{id} ({vendor} {total}) ออกจากยอดแล้ว (ถือว่าซ้ำ) ตอบ “กู้คืน” เพื่อนำกลับมา",
        "en": "🗑️ Receipt #{id} ({vendor} {total}) is deducted from the totals (marked as a "
        'duplicate). Reply "restore" to bring it back.',
    },
    "corr_restore_done": {
        "th": "♻️ กู้คืนใบเสร็จ #{id} ({vendor} {total}) กลับเข้ายอดแล้ว",
        "en": "♻️ Receipt #{id} ({vendor} {total}) is counted again.",
    },
    "corr_already_void": {
        "th": "ใบเสร็จ #{id} ถูกหักออกไปแล้ว ตอบ “กู้คืน” เพื่อนำกลับมา",
        "en": 'Receipt #{id} is already deducted. Reply "restore" to bring it back.',
    },
    "corr_not_void": {
        "th": "ใบเสร็จ #{id} ยังนับอยู่ในยอด ไม่ต้องกู้คืน",
        "en": "Receipt #{id} is still counted, nothing to restore.",
    },
    "digest_unclear": {"th": "ข้อมูลไม่ชัด", "en": "unclear data"},
    "digest_no_total": {"th": "ไม่มียอด", "en": "no total"},
    "digest_flag_line": {
        "th": "• #{id} {vendor} {amount} ({why}) โดย {who}",
        "en": "• #{id} {vendor} {amount} ({why}) by {who}",
    },
    # ---------------------------------------------------------------- receipts
    "unknown": {"th": "ไม่ทราบ", "en": "unknown"},
    "rc_check": {"th": "⚠️ กรุณาตรวจสอบ", "en": "⚠️ Please check"},
    "rc_saved": {"th": "✅ บันทึกแล้ว", "en": "✅ Saved"},
    "rc_vendor": {"th": "ร้าน: {v}", "en": "Vendor: {v}"},
    "rc_date": {"th": "วันที่: {v}", "en": "Date: {v}"},
    "rc_date_unknown": {"th": "ไม่ทราบ (ใช้วันที่ส่ง)", "en": "unknown (using the date sent)"},
    "rc_total": {"th": "ยอดรวม: {v}", "en": "Total: {v}"},
    "rc_category": {"th": "หมวด: {v}", "en": "Category: {v}"},
    "rc_unclear": {
        "th": 'อ่านข้อมูลไม่ชัด ตอบข้อความนี้เพื่อแก้ไข เช่น "ยอดจริง 450" หรือ "หมวด ค่าเดินทาง"',
        "en": 'Some details were hard to read. Reply to this message to fix them, e.g. "total 450" '
        'or "category ค่าเดินทาง"',
    },
    "rc_possible_dup": {
        "th": "⚠️ อาจซ้ำกับใบเสร็จ #{id} (ร้าน/วันที่/ยอดเท่ากัน)",
        "en": "⚠️ Possible duplicate of receipt #{id} (same vendor, date and total)",
    },
    "fwd_header": {
        "th": "📎 ใบเสร็จจาก {user} (ส่งมาใน #{channel}) · [ข้อความต้นฉบับ]({url})",
        "en": "📎 Receipt from {user} (sent in #{channel}) · [original message]({url})",
    },
    "fwd_header_dm": {
        "th": "📎 ใบเสร็จจาก {user} (ส่งมาทาง DM)",
        "en": "📎 Receipt from {user} (sent by DM)",
    },
    "fwd_ack": {
        "th": "ส่งต่อไปที่ #{home} ให้แล้วค่ะ (ข้อความเดิมลบได้ถ้าต้องการ)",
        "en": "Forwarded to #{home} (you can delete your original message if you like).",
    },
    "rc_duplicate": {
        "th": "ℹ️ ใบเสร็จนี้เคยบันทึกแล้ว (#{id} ร้าน {vendor} ยอด {total}) ไม่บันทึกซ้ำ",
        "en": "ℹ️ This receipt was already recorded (#{id}, vendor {vendor}, total {total}). Not "
        "saving it again.",
    },
    "corr_unclear": {
        "th": 'ไม่เข้าใจคำแก้ไข ลองพิมพ์ เช่น "ยอดจริง 450", "หมวด ค่าเดินทาง", "ร้าน ABC" หรือ "วันที่ 2026-10-12"',
        "en": 'I did not understand that correction. Try e.g. "total 450", "category ค่าเดินทาง", '
        '"vendor ABC" or "date 2026-10-12"',
    },
    "corr_total": {"th": "ยอดรวม", "en": "total"},
    "corr_vat": {"th": "VAT", "en": "VAT"},
    "corr_vendor": {"th": "ร้าน", "en": "vendor"},
    "corr_date": {"th": "วันที่", "en": "date"},
    "corr_category": {"th": "หมวด", "en": "category"},
    "corr_notes": {"th": "หมายเหตุ", "en": "notes"},
    "corr_done": {
        "th": "✏️ แก้ไขใบเสร็จ #{id} แล้ว: {changed}\nตอนนี้: {now}",
        "en": "✏️ Receipt #{id} updated: {changed}\nNow: {now}",
    },
    "qa_failed": {
        "th": "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้งนะคะ",
        "en": "Sorry, I can't answer right now. Please try asking again.",
    },
    # ---------------------------------------------------------------- deep tasks
    "deep_disabled": {
        "th": "ฟีเจอร์งานใหญ่ยังไม่ได้เปิดใช้งานค่ะ (ยังไม่ได้ตั้งค่า SALID_GEMINI_CLI_PATH) ลองถามคำถามสั้นๆ แทนได้เลย",
        "en": "Big tasks are not enabled yet (SALID_GEMINI_CLI_PATH is not set). Please ask a "
        "short question instead.",
    },
    "deep_not_exec": {
        "th": "ขออภัยค่ะ งานใหญ่ใช้ได้เฉพาะผู้บริหารเท่านั้น ถามคำถามสั้นๆ เรื่องใบเสร็จของคุณได้เลยค่ะ",
        "en": "Sorry, big tasks are for executives only. You can still ask short questions about "
        "your own receipts.",
    },
    "deep_cli_failed": {
        "th": "Gemini CLI ทำงานไม่สำเร็จ ลองใหม่อีกครั้งภายหลังค่ะ",
        "en": "Gemini CLI failed. Please try again later.",
    },
    "deep_no_result": {
        "th": "ไม่ได้รับผลลัพธ์จากงานนี้ค่ะ",
        "en": "This task returned no result.",
    },
    "deep_ack": {
        "th": "เดี๋ยวสลิดไล่ดูให้นะคะ งานนี้ใหญ่หน่อย อาจใช้เวลา 1-5 นาที เสร็จแล้วส่งผลในห้องนี้ค่ะ",
        "en": "I'll go through it for you. This one's a big job, so it may take 1-5 minutes; I'll "
        "post the result right here.",
    },
    "deep_busy": {
        "th": "ตอนนี้สลิดกำลังทำงานใหญ่อีกงานอยู่ค่ะ รอให้เสร็จก่อนแล้วสั่งใหม่อีกครั้งนะคะ",
        "en": "Salid is busy with another big task. Please wait for it to finish and ask again.",
    },
    "deep_usage": {
        "th": "บอกโจทย์ด้วยค่ะ เช่น งานใหญ่: วิเคราะห์ค่าใช้จ่ายเดือนนี้ หาหมวดที่โตผิดปกติ แล้วเขียนรายงาน",
        "en": "Please tell me the task, e.g. deep task: analyze this month's expenses, find "
        "categories that grew unusually, and write a report",
    },
    "deep_attached": {
        "th": "(รายงานฉบับเต็มแนบเป็นไฟล์ {name})",
        "en": "(The full report is attached as {name})",
    },
    # ---------------------------------------------------------------- slide decks (salid.deck)
    "deck_footer": {
        "th": "สร้างโดยสลิด · {date}",
        "en": "Made with Salid · {date}",
    },
    "deck_other": {
        "th": "อื่นๆ",
        "en": "Other",
    },
    "deck_no_data": {
        "th": "ไม่มีข้อมูลในช่วงนี้",
        "en": "No data for this period",
    },
    "attach_failed": {
        "th": "(แนบไฟล์ไม่ได้ อาจเพราะสลิดไม่มีสิทธิ์แนบไฟล์ในห้องนี้ หรือไฟล์ใหญ่เกินไป)",
        "en": "(I couldn't attach the file. I may not have permission to attach files here, or it is too large.)",
    },
    "deep_stats": {
        "th": "⏱ {seconds:.0f} วินาที · tokens {tokens:,} · โมเดล: {models}",
        "en": "⏱ {seconds:.0f} s · tokens {tokens:,} · model: {models}",
    },
    "deep_report_lang": {
        "th": "เป็นภาษาไทย",
        "en": "เป็นภาษาอังกฤษ (English)",
    },
    "deep_report_lang_auto": {
        "th": "ด้วยภาษาเดียวกับที่เขียนโจทย์ (ไทยหรืออังกฤษ)",
        "en": "ด้วยภาษาเดียวกับที่เขียนโจทย์ (ไทยหรืออังกฤษ)",
    },
    "pdf_generated": {
        "th": "สร้างเมื่อ {at}",
        "en": "Generated {at}",
    },
    "pdf_page": {
        "th": "หน้า {n}/{total}",
        "en": "Page {n}/{total}",
    },
    "pdf_title_expense_summary": {
        "th": "สรุปค่าใช้จ่าย",
        "en": "Expense summary",
    },
    "pdf_title_receipt_list": {
        "th": "รายการใบเสร็จ",
        "en": "Receipt list",
    },
    "pdf_title_meeting_minutes": {
        "th": "รายงานการประชุม",
        "en": "Meeting minutes",
    },
    "pdf_title_custom": {
        "th": "เอกสาร",
        "en": "Document",
    },
    "pdf_period": {
        "th": "ช่วงเวลา: {start} ถึง {end}",
        "en": "Period: {start} to {end}",
    },
    "pdf_scope_own": {
        "th": "แสดงเฉพาะใบเสร็จของ {name}",
        "en": "Showing only receipts submitted by {name}",
    },
    "pdf_totals": {
        "th": "ยอดรวมแยกตามสกุลเงิน",
        "en": "Totals by currency",
    },
    "pdf_currency": {
        "th": "สกุลเงิน",
        "en": "Currency",
    },
    "pdf_amount": {
        "th": "จำนวนเงิน",
        "en": "Amount",
    },
    "pdf_no_fx": {
        "th": "ไม่รวมต่างสกุลเงินเข้าด้วยกันและไม่แปลงอัตราแลกเปลี่ยน",
        "en": "Currencies are never added together and no exchange rate is applied.",
    },
    "pdf_by_category": {
        "th": "แยกตามหมวด",
        "en": "By category",
    },
    "pdf_by_submitter": {
        "th": "แยกตามผู้ส่ง",
        "en": "By submitter",
    },
    "pdf_name": {
        "th": "ชื่อ",
        "en": "Name",
    },
    "pdf_receipts": {
        "th": "รายการใบเสร็จ ({n} ใบ)",
        "en": "Receipts ({n})",
    },
    "pdf_date": {
        "th": "วันที่",
        "en": "Date",
    },
    "pdf_vendor": {
        "th": "ร้านค้า",
        "en": "Vendor",
    },
    "pdf_category": {
        "th": "หมวด",
        "en": "Category",
    },
    "pdf_cur": {
        "th": "สกุล",
        "en": "Cur.",
    },
    "pdf_flags": {
        "th": "หมายเหตุ",
        "en": "Flags",
    },
    "pdf_flag_low": {
        "th": "ความมั่นใจต่ำ",
        "en": "low confidence",
    },
    "pdf_flag_dup": {
        "th": "อาจซ้ำกับ #{id}",
        "en": "possible duplicate of #{id}",
    },
    "pdf_total_line": {
        "th": "รวม: {total}",
        "en": "Total: {total}",
    },
    "pdf_void_note": {
        "th": "ไม่รวมใบเสร็จที่ถูกหักออก (void) {n} ใบ",
        "en": "Excludes {n} voided receipt(s).",
    },
    "pdf_when": {
        "th": "เวลา: {when}",
        "en": "When: {when}",
    },
    "pdf_attendees": {
        "th": "ผู้เข้าร่วม: {names}",
        "en": "Attendees: {names}",
    },
    "pdf_attendance": {
        "th": "การเข้าร่วมและเหตุผลที่ไม่เข้า",
        "en": "Attendance and absence reasons",
    },
    "pdf_summary": {
        "th": "สรุปการประชุม",
        "en": "Summary",
    },
    "pdf_no_summary": {
        "th": "(ยังไม่มีสรุป)",
        "en": "(no summary yet)",
    },
    "sched_invite_dm": {
        "th": "สวัสดีค่ะ มีนัดประชุมใหม่ในเซิร์ฟเวอร์ {guild} จาก {by}\n📅 {title}\n🕑 {when}\n🔗 {link}",
        "en": "Hi! You are invited to a meeting on {guild} by {by}\n📅 {title}\n🕑 {when}\n🔗 {link}",
    },
    "flight_delay_dm": {
        "th": "✈️ เที่ยวบิน {flight} ({date}) ล่าช้าประมาณ {minutes} นาทีค่ะ เวลาออกใหม่ประมาณ {time}",
        "en": "✈️ Flight {flight} ({date}) is delayed by about {minutes} minutes. New departure about {time}.",
    },
    "flight_gate_dm": {
        "th": "✈️ เที่ยวบิน {flight} ({date}) เปลี่ยนประตูขึ้นเครื่องเป็น {gate} ค่ะ",
        "en": "✈️ Flight {flight} ({date}) now boards at gate {gate}.",
    },
    "flight_cancelled_dm": {
        "th": "✈️ เที่ยวบิน {flight} ({date}) ถูกยกเลิกค่ะ รบกวนติดต่อสายการบินด่วน",
        "en": "✈️ Flight {flight} ({date}) has been cancelled. Please contact the airline.",
    },
    "flight_landed_dm": {
        "th": "✈️ เที่ยวบิน {flight} ({date}) ลงจอดแล้วค่ะ",
        "en": "✈️ Flight {flight} ({date}) has landed.",
    },
    "place_offer_dm": {
        "th": "สลิดเห็นว่าคุณพิมพ์ที่อยู่ “{name}” ไว้ในห้องแชต: {address}\nให้สลิดจำไว้เป็นที่ส่วนตัวของคุณไหมคะ (เห็นได้คนเดียว แชร์ให้ใครทีหลังได้ ลืมได้ทุกเมื่อ) ตอบ “จำไว้” หรือ “ไม่ต้อง” ได้เลยค่ะ สลิดจะถามแค่ครั้งนี้ครั้งเดียว",
        "en": "Salid noticed you wrote the address “{name}” in a channel: {address}\nShall Salid remember it as your own private place (only you can see it; you can share it later and I can forget it any time)? Just reply “remember it” or “no thanks”. I will only ask this once.",
    },
    "bf_leave_line": {
        "th": "ควรออกจาก{origin} ประมาณ {leave} (ใช้เวลา ~{minutes} นาที) ไปประชุม “{title}” ที่{place}",
        "en": "Leave {origin} around {leave} (~{minutes} min) for “{title}” at {place}",
    },
    "bf_weather_line": {
        "th": "อากาศที่{place}วันนี้: {summary}",
        "en": "Weather at {place} today: {summary}",
    },
    "place_event_dm": {
        "th": "📍 {where}\n🗺️ {maps}",
        "en": "📍 {where}\n🗺️ {maps}",
    },
    "sched_cancelled_dm": {
        "th": "การประชุม “{title}” ถูกยกเลิกแล้วค่ะ",
        "en": "The meeting “{title}” has been cancelled.",
    },
    "sched_moved_dm": {
        "th": "การประชุม “{title}” เลื่อนเวลาเป็น {when}\n🔗 {link}",
        "en": "The meeting “{title}” was moved to {when}\n🔗 {link}",
    },
    "sched_remind_pre": {
        "th": "⏰ อีก {mins} นาทีถึงการประชุม “{title}” ({when})\nเข้าร่วม: {link}",
        "en": "⏰ “{title}” starts in {mins} minutes ({when})\nJoin: {link}",
    },
    "sched_remind_start": {
        "th": "🔔 การประชุม “{title}” เริ่มแล้วค่ะ\nเข้าร่วม: {link}",
        "en": "🔔 “{title}” is starting now\nJoin: {link}",
    },
    "sched_reply_if_stuck": {
        "th": "ถ้าเข้าไม่ได้ ตอบกลับสลิดได้เลยค่ะ",
        "en": "If you can't join, just reply to me.",
    },
    "sched_nudge_dm": {
        "th": "การประชุม “{title}” เริ่มแล้วค่ะ มีปัญหาอะไรหรือเปล่า? ถ้าหาลิงก์ไม่เจอ กดเข้าตรงนี้ได้เลย: {link}\nถ้าเข้าไม่ได้ ตอบกลับสลิดได้เลยค่ะ",
        "en": "“{title}” has started. Is something wrong? If you can't find the link, join here: {link}\nIf you can't make it, just reply to me.",
    },
    "sched_attended": {
        "th": "✅ เข้าร่วม: {names}",
        "en": "✅ Attended: {names}",
    },
    "sched_absent": {
        "th": "🚫 ไม่ได้เข้า: {names}",
        "en": "🚫 Absent: {names}",
    },
    "sched_summary_link": {
        "th": "📝 สรุปการประชุม: {url}",
        "en": "📝 Meeting summary: {url}",
    },
    "sched_absence_note": {
        "th": "{name} แจ้งว่า: {reason} (การประชุม “{title}”)",
        "en": "{name} says: {reason} (meeting “{title}”)",
    },
    "sched_absence_ack": {
        "th": "รับทราบค่ะ แจ้งผู้จัดการประชุม “{title}” แล้ว",
        "en": "Got it, I let the organiser of “{title}” know.",
    },
    "perm_missing": {
        "th": "ตอนนี้สลิดยังขาดสิทธิ์ {what} อยู่ค่ะ ให้แอดมินเปิดลิงก์นี้ครั้งเดียวแล้วเลือกเซิร์ฟเวอร์นี้ก็เรียบร้อย {url}",
        "en": "I'm missing the {what} permission here. An admin just needs to open this link once "
        "and pick this server: {url}",
    },
    "perm_missing_nolink": {
        "th": "ตอนนี้สลิดยังขาดสิทธิ์ {what} อยู่ค่ะ ให้แอดมินเปิดสิทธิ์นี้ให้บทบาทของสลิดหน่อยนะคะ",
        "en": "I'm missing the {what} permission here. An admin needs to allow it for my role.",
    },
    "email_not_configured": {
        "th": "ยังไม่ได้ตั้งค่าการส่งอีเมล (บันทึกร่างไว้แล้ว แต่ยังไม่ได้ส่งจริง)",
        "en": "Email sending is not configured yet (the draft was recorded but nothing was sent).",
    },
    "meet_announce_scheduled": {
        "th": "สลิดเข้ามาจดประชุมตามนัดค่ะ 🎙️ (เก็บเป็นข้อความในความจำเท่านั้น ไม่บันทึกเสียง)",
        "en": "Salid has joined to take notes for the scheduled meeting 🎙️ (kept as text only, no audio is recorded).",
    },
    "meet_announce_auto": {
        "th": "สลิดเข้ามาจดประชุมนะคะ ไม่ต้องการพิมพ์ “@Salid ออกประชุม” ได้เลย "
        "(เก็บเป็นข้อความในความจำเท่านั้น ไม่บันทึกเสียง)",
        "en": "Salid has joined to take meeting notes. If you don't need me, just type "
        "“@Salid leave meeting” (kept as text only, no audio is recorded).",
    },
    "meet_preempted": {
        "th": "ขอไปช่วยประชุมที่นัดไว้ก่อนนะคะ ส่วนที่คุยมาสรุปไว้ให้แล้ว",
        "en": "I have to go help with a scheduled meeting, sorry! The summary of what was said is below.",
    },
    "end_preempted": {
        "th": "สลิดต้องไปประชุมที่สำคัญกว่า",
        "en": "Salid moved to a more important meeting",
    },
    "meet_busy": {
        "th": "ตอนนี้ร่างของสลิดติดประชุมอยู่ครบแล้วค่ะ ({rooms}) ถ้ามีผู้ช่วยเสียงเพิ่มจะแบ่งร่างไปได้อีก",
        "en": "All of Salid's voices are in meetings right now ({rooms}). "
        "More voice helpers would let Salid split further.",
    },
    "meet_leave_which": {
        "th": "สลิดอยู่ในหลายห้อง ให้พิมพ์สั่งจากในห้องนั้นหรือเข้าห้องที่ต้องการก่อนนะคะ: {rooms}",
        "en": "Salid is in several rooms. Type it from that room or join the room first: {rooms}",
    },
    "voice_bodies_title": {
        "th": "ร่างเสียงของสลิด",
        "en": "Salid's voices",
    },
    "voice_body_in": {"th": "อยู่ในเซิร์ฟเวอร์นี้", "en": "in this server"},
    "voice_body_out": {"th": "ยังไม่ได้เชิญเข้าเซิร์ฟเวอร์นี้", "en": "not invited to this server"},
    "voice_body_down": {"th": "ยังไม่พร้อมใช้งาน", "en": "unavailable"},
    # ---------------------------------------------------------------- coordinator: disclosure
    "disc_where_dm": {"th": "บันทึก DM ส่วนตัว", "en": "a private DM note"},
    "disc_desc": {
        "th": "{n} ข้อความที่คุณเขียนไว้ ช่วงวันที่ {span} ({where})",
        "en": "{n} message(s) you wrote, dated {span} ({where})",
    },
    "disc_ask": {
        "th": "{asker} ถามสลิดเรื่อง “{topic}”\nคำถามเดิม: “{question}”\n"
        "สลิดมีข้อมูลที่เป็นของคุณที่เกี่ยวข้อง: {description}\n"
        "(สลิดยังไม่ได้บอกเนื้อหาให้ใคร) อนุญาตให้สลิดบอก {asker} เรื่องนี้ไหมคะ?",
        "en": "{asker} asked Salid about “{topic}”.\nOriginal question: “{question}”\n"
        "Salid holds related information of yours: {description}\n"
        "(Nothing has been shared yet.) May Salid tell {asker} about it?",
    },
    "disc_btn_ok": {"th": "อนุญาต", "en": "Approve"},
    "disc_btn_no": {"th": "ไม่อนุญาต", "en": "Decline"},
    "disc_owner_approved": {
        "th": "✅ อนุญาตแล้ว สลิดจะบอก {asker} เฉพาะข้อมูลชุดนี้ค่ะ",
        "en": "✅ Approved. Salid will tell {asker} only this information.",
    },
    "disc_owner_declined": {
        "th": "🚫 ไม่อนุญาต สลิดจะไม่บอก {asker} ค่ะ",
        "en": "🚫 Declined. Salid will not tell {asker}.",
    },
    "disc_handled": {
        "th": "คำขอนี้ตัดสินไปแล้วค่ะ",
        "en": "This request has already been decided.",
    },
    "disc_not_yours": {
        "th": "คำขอนี้ไม่ใช่ของคุณค่ะ",
        "en": "This request is not yours to answer.",
    },
    "disc_asker_declined": {
        "th": "เรื่อง “{topic}”: เจ้าของข้อมูลยังไม่อนุญาตให้บอกค่ะ ขอโทษด้วยนะคะ ลองถามเจ้าของโดยตรงได้เลย",
        "en": "About “{topic}”: the owner of that information did not allow sharing it, sorry. "
        "You can ask them directly.",
    },
    "disc_asker_approved": {
        "th": "เรื่อง “{topic}”: เจ้าของข้อมูลอนุญาตแล้วค่ะ\n\n",
        "en": "About “{topic}”: the owner approved sharing it.\n\n",
    },
    "disc_asker_failed": {
        "th": "เจ้าของข้อมูลอนุญาตแล้ว แต่สลิดเรียบเรียงคำตอบไม่สำเร็จ ลองถามอีกครั้งทาง DM ได้เลยค่ะ",
        "en": "The owner approved, but Salid could not put the answer together. Please ask again in a DM.",
    },
    # ---------------------------------------------------------------- coordinator: DMs
    "dm_notice": {
        "th": "ℹ️ เรื่องงานที่คุณบอกสลิดทาง DM สลิดจะจดไว้ให้ทีม และผู้ที่ตำแหน่ง (role) สูงกว่าคุณอาจเห็นได้ "
        "ถ้าเรื่องไหนเป็นความลับ พิมพ์ “เรื่องนี้เก็บเป็นความลับ” แล้วสลิดจะเก็บให้คุณคนเดียวค่ะ",
        "en": "ℹ️ Work information you tell Salid in a DM is noted for the team, and people with a "
        "higher role than yours may see it. If something is confidential, type “keep this private” "
        "and Salid will keep it to you alone.",
    },
    "dm_notice_room": {
        "th": "ℹ️ เรื่องงานที่คุณบอกสลิดในห้องส่วนตัวนี้ สลิดจะจดไว้ให้ทีม และผู้ที่ตำแหน่ง (role) สูงกว่าคุณอาจเห็นได้ "
        "ถ้าเรื่องไหนเป็นความลับ พิมพ์ “เรื่องนี้เก็บเป็นความลับ” แล้วสลิดจะเก็บให้คุณคนเดียวค่ะ",
        "en": "ℹ️ Work information you tell Salid in this private channel is noted for the team, and "
        "people with a higher role than yours may see it. If something is confidential, type "
        "“keep this private” and Salid will keep it to you alone.",
    },
    "long_answer_note": {
        "th": "คำตอบยาว สลิดสรุปสั้นๆ ไว้ข้างบน ส่งฉบับเต็มเป็นไฟล์แนบให้แล้วค่ะ 📎",
        "en": "This answer is long, so here is a short summary; the full text is attached. 📎",
    },
    "dm_private_marked": {
        "th": "รับทราบค่ะ เรื่องนี้เก็บเป็นความลับ เห็นเฉพาะคุณคนเดียว สลิดจะไม่เอาไปรายงานหรือบอกใครคะ 🔒",
        "en": "Understood. That stays confidential: only you can see it, and Salid will not report "
        "it or tell anyone. 🔒",
    },
    "dm_private_note": {
        "th": "(ผู้ใช้ระบุว่าเรื่องนี้เป็นความลับ เก็บไว้ให้เขาคนเดียว)",
        "en": "(The user said this is confidential: keep it to them alone.)",
    },
    # ---------------------------------------------------------------- coordinator: follow-ups
    "fu_header": {
        "th": "สวัสดีค่ะ {name} สลิดขอติดตามงานที่ถึงกำหนดหรือเลยกำหนดค่ะ:",
        "en": "Hi {name}, Salid is following up on tasks that are due or overdue:",
    },
    "fu_line": {"th": "• #{id} {title} (กำหนด {due})", "en": "• #{id} {title} (due {due})"},
    "fu_footer": {
        "th": "ตอบกลับที่นี่ได้เลยว่าถึงไหนแล้ว เช่น “#{id} เสร็จแล้ว” หรือ “ติดเรื่อง…” แล้วสลิดจะอัปเดตให้ค่ะ",
        "en": "Just reply here with the status, e.g. “#{id} done” or “blocked on…”, and Salid will "
        "update it.",
    },
    # ---------------------------------------------------------------- coordinator: reports
    "rep_title": {"th": "**📋 รายงานประจำวันที่ {day}**", "en": "**📋 Daily report for {day}**"},
    "rep_brief": {"th": "**ภาพรวมบริษัท**", "en": "**Company brief**"},
    "rep_spend": {
        "th": "- ใช้จ่ายวันนี้ {today} ({n} ใบ) · เดือนนี้ {month}",
        "en": "- Spent today {today} ({n} receipts) · this month {month}",
    },
    "rep_review": {
        "th": "- ใบเสร็จที่ต้องตรวจ {n} ใบ · อาจซ้ำ {d} ใบ",
        "en": "- Receipts to review: {n} · possible duplicates: {d}",
    },
    "rep_meetings": {"th": "**ประชุมวันนี้**", "en": "**Meetings today**"},
    "rep_people": {"th": "**งานของแต่ละคน**", "en": "**Work by person**"},
    "rep_person": {
        "th": "- **{name}**: {msgs} ข้อความ · งานค้าง {open}",
        "en": "- **{name}**: {msgs} messages · {open} open task(s)",
    },
    "rep_opened": {"th": "เปิดใหม่: ", "en": "opened: "},
    "rep_done": {"th": "✅ เสร็จ: ", "en": "✅ done: "},
    "rep_overdue": {"th": "⚠️ เลยกำหนด: ", "en": "⚠️ overdue: "},
    "rep_nodue": {"th": "🗓️ **งานไม่มีกำหนด**", "en": "🗓️ **No deadline**"},
    "rep_more": {"th": "…และอีก {n} งาน", "en": "+{n} more"},
    "rep_risks": {"th": "**ความเสี่ยง / สิ่งที่ควรตาม**", "en": "**Risks / to follow up**"},
    "pr_title": {"th": "**วันนี้ของคุณ ({day})**", "en": "**Your day ({day})**"},
    "pr_where": {"th": "คุณเขียนใน: {where}", "en": "You wrote in: {where}"},
    "pr_none": {
        "th": "วันนี้ยังไม่พบข้อความของคุณในห้องที่สลิดจดไว้ค่ะ",
        "en": "Salid has not recorded any message from you today.",
    },
    "pr_done": {"th": "งานที่เสร็จวันนี้: ", "en": "Tasks done today: "},
    "pr_open": {"th": "งานที่ยังค้าง:", "en": "Open tasks:"},
    "report_channel_topic": {
        "th": "รายงานประจำวันจากสลิด สำหรับเจ้าของ แอดมิน และผู้บริหาร",
        "en": "Daily reports from Salid for the owner, admins and executives",
    },
    "report_need_channel": {
        "th": "สลิดอยากส่งรายงานเย็นให้ผู้บริหารในห้องส่วนตัว “{name}” แต่ไม่มีสิทธิ์จัดการช่อง (Manage Channels) "
        "รบกวนให้สิทธิ์สลิด หรือสร้างห้องที่เห็นเฉพาะผู้บริหารแล้วเลือกเป็นห้องรายงานในหน้า Hub ค่ะ",
        "en": "Salid would like to post the evening report for leadership in a private channel "
        "“{name}” but lacks the Manage Channels permission. Please grant it, or create a "
        "leadership-only channel and pick it as the report channel in the hub.",
    },
    "report_denied": {
        "th": "รายงานบริษัทดูได้เฉพาะเจ้าของ แอดมิน และผู้บริหารค่ะ (ดูของตัวเองด้วย /รายงาน ของฉัน)",
        "en": "The company report is for the owner, admins and executives (use /report mine for yours).",
    },
    "report_bad_date": {
        "th": "วันที่ต้องเป็นรูปแบบ ปี-เดือน-วัน เช่น 2026-10-14 ค่ะ",
        "en": "The date must look like 2026-10-14.",
    },
    "report_off": {
        "th": "ตอนนี้ปิดความจำหรือรายงานอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Memory or reports are switched off. You can turn them on in Salid's command center.",
    },
    "report_posted": {
        "th": "ส่งรายงานวันที่ {day} เข้า {channel} แล้วค่ะ",
        "en": "Posted the report for {day} in {channel}.",
    },
    "report_failed": {
        "th": "สร้างรายงานไม่สำเร็จ ลองใหม่ภายหลังค่ะ",
        "en": "Could not build the report. Please try again later.",
    },
    "cmd_report": _bi("รายงานประจำวันจากสลิด", "Daily report from Salid"),
    "cmd_report_date": _bi("วันที่ (ปี-เดือน-วัน) ไม่ใส่ = วันนี้", "Date (YYYY-MM-DD), default today"),
    "cmd_report_mine": _bi("ดูของฉันเอง", "Show only my own day"),
    # ---------------------------------------------------------------- briefings and alerts
    "bf_title_morning": {
        "th": "**สรุปเช้าของคุณ {name} ({day})**",
        "en": "**Good morning {name}, your brief for {day}**",
    },
    "bf_title_evening": {
        "th": "**สรุปเย็นของคุณ {name} ({day})**",
        "en": "**Evening brief for {name} ({day})**",
    },
    "bf_meetings": {"th": "**นัดวันนี้**", "en": "**Today's meetings**"},
    "bf_meetings_tomorrow": {"th": "**นัดพรุ่งนี้**", "en": "**Tomorrow's meetings**"},
    "bf_meeting_line": {"th": "• {time} {title} {link}", "en": "• {time} {title} {link}"},
    "bf_tasks_due": {"th": "**งานที่ครบกำหนดหรือเลยกำหนด**", "en": "**Tasks due or overdue**"},
    "bf_task_line": {
        "th": "• #{id} {title} (กำหนด {due})",
        "en": "• #{id} {title} (due {due})",
    },
    "bf_docs": {"th": "**เอกสารสำคัญที่ได้รับ**", "en": "**Important documents received**"},
    "bf_doc_line": {
        "th": "• {kind}: {summary} ({channel}) {link}",
        "en": "• {kind}: {summary} ({channel}) {link}",
    },
    "bf_asks": {"th": "**เรื่องที่รอคุณตอบ**", "en": "**Waiting for your reply**"},
    "bf_ask_line": {
        "th": "• {who} ({channel}): {text} {link}",
        "en": "• {who} ({channel}): {text} {link}",
    },
    "bf_attended": {"th": "**ประชุมที่คุณเข้าวันนี้**", "en": "**Meetings you attended today**"},
    "bf_new_tasks": {"th": "**งานใหม่ที่ได้รับวันนี้**", "en": "**New tasks assigned today**"},
    "bf_done": {"th": "**งานที่เสร็จวันนี้**", "en": "**Done today**"},
    "bf_open": {"th": "**งานที่ยังค้าง**", "en": "**Still open**"},
    "bf_footer": {
        "th": "ปรับหรือปิดสรุปนี้ได้โดยบอกสลิด เช่น “ไม่ต้องส่งสรุปเย็น”",
        "en": "To change or stop these briefs just tell Salid, e.g. “no evening brief”.",
    },
    # ---------------------------------------------------------------- catch-up ("what did I miss")
    "cu_title": {
        "th": "**สิ่งที่คุณพลาดไปตั้งแต่ {since}**",
        "en": "**What you missed since {since}**",
    },
    "cu_none": {
        "th": "ตั้งแต่ {since} ยังไม่มีอะไรใหม่ที่เกี่ยวกับคุณในห้องที่คุณอ่านได้ค่ะ",
        "en": "Nothing new for you since {since} in the channels you can read.",
    },
    "cu_mentions": {"th": "**มีคนแท็กหรือพูดถึงคุณ**", "en": "**Mentions of you**"},
    "cu_mention_line": {
        "th": "• {who} ({channel}): {text} {link}",
        "en": "• {who} ({channel}): {text} {link}",
    },
    "cu_tasks": {"th": "**งานของคุณ**", "en": "**Your tasks**"},
    "cu_task_line": {
        "th": "• #{id} {title} (กำหนด {due}) {flag} {link}",
        "en": "• #{id} {title} (due {due}) {flag} {link}",
    },
    "cu_late": {"th": "⚠️ เลยกำหนด", "en": "⚠️ overdue"},
    "cu_new": {"th": "ใหม่", "en": "new"},
    "cu_decisions": {"th": "**มติและเรื่องที่ตกลงกัน**", "en": "**Decisions and agreements**"},
    "cu_missed": {"th": "**ประชุมที่คุณไม่ได้เข้า**", "en": "**Meetings you missed**"},
    "cu_upcoming": {"th": "**นัดที่กำลังจะถึง**", "en": "**Coming up**"},
    "cu_meeting_line": {"th": "• {time} {title} {link}", "en": "• {time} {title} {link}"},
    "cu_no_minutes": {"th": "(ยังไม่มีสรุปประชุม)", "en": "(no minutes yet)"},
    "cu_channels": {"th": "**ไฮไลต์รายห้อง**", "en": "**Highlights by channel**"},
    "cu_channel_line": {
        "th": "• {channel} ({n} ข้อความ): {text} {link}",
        "en": "• {channel} ({n} messages): {text} {link}",
    },
    "cu_files": {"th": "**ไฟล์ที่เกี่ยวกับคุณ**", "en": "**Files for you**"},
    "cu_ask_more": {
        "th": "อยากดูละเอียดห้องไหนบอกได้นะคะ",
        "en": "Tell me which channel you want in detail.",
    },
    "cu_s_mentions": {"th": "แท็กคุณ {n}", "en": "{n} mention(s)"},
    "cu_s_tasks": {"th": "งาน {n}", "en": "{n} task(s)"},
    "cu_s_missed": {"th": "ประชุมที่พลาด {n}", "en": "{n} missed meeting(s)"},
    "cu_s_decisions": {"th": "มติ {n}", "en": "{n} decision(s)"},
    "cu_off": {
        "th": "ตอนนี้ปิดสรุปสิ่งที่พลาดหรือความจำอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Catch-up or memory is switched off. You can turn it on in Salid's command center.",
    },
    "cu_failed": {
        "th": "สรุปสิ่งที่พลาดไม่สำเร็จ ลองใหม่ภายหลังนะคะ",
        "en": "Could not build your catch-up. Please try again later.",
    },
    "cu_bad_since": {
        "th": "ไม่เข้าใจช่วงเวลานี้ค่ะ ลองเช่น “เมื่อวาน” “3 วัน” “จันทร์” หรือ 2026-10-12",
        "en": "I did not understand that period. Try “yesterday”, “3 days”, “monday” or 2026-10-12.",
    },
    "cmd_catchup": _bi("สรุปสิ่งที่คุณพลาดไป", "Catch up on what you missed"),
    "cmd_catchup_since": _bi(
        "ตั้งแต่เมื่อไร เช่น เมื่อวาน, 3 วัน, จันทร์ (ไม่ใส่ = ตั้งแต่ที่คุณหายไป)",
        "Since when, e.g. yesterday, 3 days, monday (empty = since you were last around)",
    ),
    "alert_doc": {
        "th": "**เอกสารสำคัญ** ({kind}): {summary}\nห้อง {channel} · {link}",
        "en": "**Important document** ({kind}): {summary}\nChannel {channel} · {link}",
    },
    "ask_reminder": {
        "th": "ขอเตือนเบาๆ ค่ะ {who} ฝากเรื่องนี้ไว้ในห้อง {channel} และยังไม่มีคำตอบจากคุณ: “{text}” {link}",
        "en": "A gentle reminder: {who} asked you something in {channel} and there is no reply "
        "from you yet: “{text}” {link}",
    },
    # ---------------------------------------------------------------- deliver_file
    "deliver_footer": {
        "th": "ส่งต่อจาก {who} · [ต้นฉบับ]({url})",
        "en": "Forwarded from {who} · [original]({url})",
    },
    "deliver_oversize": {
        "th": "ไฟล์ใหญ่เกินกว่าจะแนบ (เกิน 25 MB) ดูได้ที่ต้นฉบับ: {url}",
        "en": "The file is too large to attach (over 25 MB); see the original: {url}",
    },
    "deliver_email_body": {
        "th": "เรียนทุกท่าน\nขอส่งไฟล์ {name} มาพร้อมอีเมลนี้ค่ะ",
        "en": "Hello,\nPlease find the file {name} attached.",
    },
    "deliver_footer_plain": {"th": "ส่งต่อจาก {who}", "en": "Forwarded from {who}"},
    "deliver_original": {"th": "[ต้นฉบับ]({url})", "en": "[original]({url})"},
    "deliver_dm_from": {
        "th": "ส่งต่อจาก {who} ({server}) ผ่านสลิดค่ะ",
        "en": "Forwarded by Salid from {who} ({server}).",
    },
    "deliver_warn_audience": {
        "th": "เอกสารนี้มาจาก #{source} ซึ่ง {who} มองไม่เห็น ยืนยันจะส่งไหมคะ",
        "en": "This document comes from #{source}, which {who} cannot see. Send it anyway?",
    },
    # ---------------------------------------------------------------- polls and forms
    "poll_result": {
        "th": "ผลโหวต “{question}”: {winner} ชนะค่ะ ({votes} จาก {total} โหวต)",
        "en": "Poll result for “{question}”: {winner} wins ({votes} of {total} votes).",
    },
    "poll_result_tie": {
        "th": "ผลโหวต “{question}”: เสมอกันระหว่าง {winners} ({votes} โหวตเท่ากัน) ค่ะ",
        "en": "Poll result for “{question}”: a tie between {winners} ({votes} votes each).",
    },
    "poll_result_none": {
        "th": "โหวต “{question}” จบแล้ว แต่ยังไม่มีใครโหวตเลยค่ะ",
        "en": "The poll “{question}” has ended, but nobody voted.",
    },
    "poll_memory": {
        "th": "[ผลโหวต] {line}",
        "en": "[Poll result] {line}",
    },
    "form_invite": {
        "th": "สวัสดีค่ะ {creator} ({server}) ฝากให้สลิดถามเรื่อง “{title}” มี {n} ข้อ "
        "ตอบกลับในแชตนี้ได้เลยค่ะ (พิมพ์ “ข้าม” ถ้าไม่อยากตอบข้อไหน หรือ “ไม่ขอตอบ” เพื่อหยุด) "
        "ตอบภายใน {deadline} นะคะ",
        "en": "Hi! {creator} ({server}) asked me to collect answers for “{title}”: {n} questions. "
        "Just reply here (type “skip” to pass a question, or “stop” to opt out). "
        "Please answer by {deadline}.",
    },
    "form_question": {
        "th": "ข้อ {i}/{n}: {question}",
        "en": "Question {i}/{n}: {question}",
    },
    "form_thanks": {
        "th": "ขอบคุณค่ะ ตอบครบแล้ว สลิดส่งคำตอบให้ {creator} เรียบร้อยค่ะ",
        "en": "Thank you, that's everything. I'll pass your answers to {creator}.",
    },
    "form_declined_ack": {
        "th": "รับทราบค่ะ ไม่ถามเรื่องนี้อีกแล้วนะคะ",
        "en": "Got it, I won't ask about this again.",
    },
    "form_reminder": {
        "th": "เตือนความจำเบาๆ ค่ะ “{title}” ของ {creator} ใกล้ถึงกำหนด ({deadline}) แล้ว "
        "ยังตอบไม่ครบนะคะ ตอบกลับในแชตนี้ได้เลยค่ะ",
        "en": "Gentle reminder: “{title}” from {creator} closes soon ({deadline}) and I still "
        "have some unanswered questions. Just reply here.",
    },
    "form_progress": {
        "th": "“{title}” ตอบแล้ว {done} จาก {total} คน",
        "en": "“{title}”: {done} of {total} people have answered.",
    },
    "form_closed": {
        "th": "แบบสอบถาม “{title}” ปิดแล้วค่ะ ตอบแล้ว {done} จาก {total} คน "
        "ขอดูผลได้เลยนะคะ (บอกว่า “ขอผลแบบสอบถาม”)",
        "en": "The form “{title}” has closed: {done} of {total} people answered. "
        "Ask me for the results any time.",
    },
    "form_col_name": {"th": "ชื่อ", "en": "Name"},
    "form_col_state": {"th": "สถานะ", "en": "Status"},
    "form_pdf_people": {"th": "คำตอบรายคน", "en": "Answers by person"},
    "form_results_dm": {
        "th": "ผลแบบสอบถามส่งทาง DM แล้วค่ะ",
        "en": "I sent the results to you by DM.",
    },
    # ---------------------------------------------------------------- contests (salid.contests_bot)
    "contest_remind": {
        "th": "⏳ {name}: “{label}” เหลืออีก {left} ({when}) ยังเหลืออีก {open} ข้อที่ต้องส่งค่ะ",
        "en": "⏳ {name}: “{label}” is due in {left} ({when}). {open} item(s) still open.",
    },
    "contest_remind_nolist": {
        "th": "⏳ {name}: “{label}” เหลืออีก {left} ({when}) ค่ะ",
        "en": "⏳ {name}: “{label}” is due in {left} ({when}).",
    },
    "contest_brief": {
        "th": "⏳ {name} ส่งงานอีก {days} วัน (เหลือ {open} ข้อ)",
        "en": "⏳ {name}: {days} day(s) to submit ({open} item(s) left)",
    },
    "contest_brief_today": {
        "th": "⏳ {name} ส่งงานวันนี้ (เหลือ {open} ข้อ)",
        "en": "⏳ {name}: due today ({open} item(s) left)",
    },
    "contest_brief_nolist": {
        "th": "⏳ {name} ส่งงานอีก {days} วัน",
        "en": "⏳ {name}: {days} day(s) to submit",
    },
    "contest_brief_today_nolist": {
        "th": "⏳ {name} ส่งงานวันนี้",
        "en": "⏳ {name}: due today",
    },
    # ---------------------------------------------------------------- wiki (salid.wiki_bot)
    "wiki_post_footer": {
        "th": "อัปเดตล่าสุด {date} · เขียนโดย {who} · สลิด",
        "en": "Last updated {date} · written by {who} · via Salid",
    },
    # ---- english-audit A (proactive)
    "translate_nothing_pure": {
        "th": "ข้อความนี้ไม่มีตัวอักษรให้แปลค่ะ · There is no text in that message to translate.",
        "en": "There is no text in that message to translate.",
    },
    "translate_failed_pure": {
        "th": "แปลไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ · I couldn't translate that, please try again.",
        "en": "I couldn't translate that, please try again.",
    },
    "rc_unclear_pure": {
        "th": 'อ่านข้อมูลไม่ชัด ตอบข้อความนี้เพื่อแก้ไข เช่น "ยอดจริง 450" หรือ "หมวด ค่าเดินทาง"',
        "en": 'Some details were hard to read. Reply to this message to fix them, e.g. "total 450" '
        'or "category Travel"',
    },
    "corr_unclear_pure": {
        "th": 'ไม่เข้าใจคำแก้ไข ลองพิมพ์ เช่น "ยอดจริง 450", "หมวด ค่าเดินทาง", "ร้าน ABC" หรือ "วันที่ 2026-10-12"',
        "en": 'I did not understand that correction. Try e.g. "total 450", "category Travel", '
        '"vendor ABC" or "date 2026-10-12"',
    },
    # ---- english-audit B (voice, meeting)
    # ---- english-audit C (documents, tools)
    "deep_timeout": {
        "th": "งานใหญ่ใช้เวลานานเกิน {seconds} วินาที จึงยกเลิก",
        "en": "The big task ran longer than {seconds} seconds, so it was cancelled",
    },
    # ---------------------------------------------------------------- reminders (reminders.py)
    "rem_fire": {
        "th": "⏰ เตือนค่ะ: {message}",
        "en": "⏰ Reminder: {message}",
    },
    "rem_fire_from": {
        "th": "⏰ {who} ฝากเตือนค่ะ: {message}",
        "en": "⏰ Reminder from {who}: {message}",
    },
    "rem_late": {
        "th": "(เลยเวลาที่ตั้งไว้ {when} แล้วค่ะ)",
        "en": "(this was due {when})",
    },
    "rem_footer": {
        "th": '({every} · ครั้งต่อไป {next} · บอกสลิดว่า "ยกเลิกเตือน {id}" ได้)',
        "en": '({every} · next {next} · tell Salid "cancel reminder {id}" to stop)',
    },
    "rem_once": {
        "th": "ครั้งเดียว",
        "en": "once",
    },
    "rem_every_n": {
        "th": "ทุก {n} {unit}",
        "en": "every {n} {unit}",
    },
    "rem_unit_m": {
        "th": "นาที",
        "en": "min",
    },
    "rem_unit_h": {
        "th": "ชั่วโมง",
        "en": "h",
    },
    "rem_unit_d": {
        "th": "วัน",
        "en": "d",
    },
    "rem_every_day": {
        "th": "ทุกวัน",
        "en": "every day",
    },
    "rem_every_weekday": {
        "th": "ทุกวันจันทร์-ศุกร์",
        "en": "every weekday",
    },
    "rem_every_on": {
        "th": "ทุก {days}",
        "en": "every {days}",
    },
    "rem_today": {
        "th": "วันนี้ {hm}",
        "en": "today {hm}",
    },
    "rem_tomorrow": {
        "th": "พรุ่งนี้ {hm}",
        "en": "tomorrow {hm}",
    },
    "rem_err_empty": {
        "th": "ไม่รู้ว่าจะให้เตือนเรื่องอะไรค่ะ",
        "en": "There is nothing to remind about.",
    },
    "rem_err_too_long": {
        "th": "ข้อความเตือนยาวเกินไปค่ะ (ไม่เกิน {n} ตัวอักษร)",
        "en": "That reminder is too long (max {n} characters).",
    },
    "rem_err_no_time": {
        "th": "ยังไม่ได้บอกเวลาที่จะให้เตือนค่ะ",
        "en": "No time was given for the reminder.",
    },
    "rem_err_bad_time": {
        "th": "อ่านเวลาไม่ออกค่ะ",
        "en": "Could not read that time.",
    },
    "rem_err_past": {
        "th": "เวลานั้นผ่านไปแล้วค่ะ",
        "en": "That time has already passed.",
    },
    "rem_err_too_far": {
        "th": "ตั้งเตือนล่วงหน้าได้ไม่เกิน 1 ปีค่ะ",
        "en": "Reminders can be set at most 1 year ahead.",
    },
    "rem_err_bad_every": {
        "th": "อ่านความถี่ไม่ออกค่ะ (ใช้ เช่น 30m, 2h, 1d)",
        "en": "Could not read that interval (use e.g. 30m, 2h, 1d).",
    },
    "rem_err_min_interval": {
        "th": "เตือนซ้ำถี่สุดทุก 15 นาทีค่ะ",
        "en": "The shortest repeat interval is 15 minutes.",
    },
    "rem_err_both": {
        "th": "เลือกอย่างใดอย่างหนึ่งค่ะ: เตือนทุกกี่ชั่วโมง หรือเตือนตามวันในสัปดาห์",
        "en": "Choose either a repeat interval or weekdays, not both.",
    },
    "rem_err_bad_weekdays": {
        "th": "อ่านวันในสัปดาห์ไม่ออกค่ะ",
        "en": "Could not read those weekdays.",
    },
    "rem_err_weekly_needs_time": {
        "th": "เตือนรายวัน/รายสัปดาห์ต้องบอกเวลาเป็น HH:MM ด้วยค่ะ",
        "en": "Daily or weekly reminders need a time of day (HH:MM).",
    },
    "rem_err_bad_until": {
        "th": "วันสิ้นสุดไม่ถูกต้องค่ะ (ต้องไม่ก่อนวันนี้ และไม่ก่อนการเตือนครั้งแรก)",
        "en": "The end date is not valid (not before today or the first reminder).",
    },
    "rem_err_cap": {
        "th": "มีเตือนที่ยังใช้งานอยู่ครบ {n} อันแล้วค่ะ ยกเลิกบางอันก่อนนะคะ",
        "en": "There are already {n} active reminders; cancel some first.",
    },
    "rem_err_exec_only": {
        "th": "ตั้งเตือนให้คนอื่นได้เฉพาะผู้บริหารค่ะ",
        "en": "Only executives can set reminders for other people.",
    },
    "rem_err_person_not_found": {
        "th": "ไม่เจอคนชื่อ “{person}” ในเซิร์ฟเวอร์ค่ะ",
        "en": "I could not find anyone named “{person}”.",
    },
    "rem_err_ambiguous_person": {
        "th": "มีหลายคนชื่อ “{person}” ค่ะ: {names}",
        "en": "Several people match “{person}”: {names}.",
    },
    "rem_err_bad_where": {
        "th": "ส่งเตือนได้แค่ทางช่องส่วนตัว (me) หรือในห้องนี้ (here) ค่ะ",
        "en": "Reminders go to the private channel (me) or this channel (here) only.",
    },
    "rem_err_sensitive": {
        "th": "ข้อความเกี่ยวกับการโอนเงินหรือรหัสผ่านฝากเตือนคนอื่นไม่ได้ค่ะ",
        "en": "Money or credential text cannot be set as a reminder for someone else.",
    },
    "rem_err_not_found": {
        "th": "ไม่เจอเตือนที่ตรงกันค่ะ",
        "en": "No matching reminder.",
    },
    "rem_err_ambiguous": {
        "th": "มีหลายอันที่ตรงกันค่ะ บอกรหัส (เช่น R-0012) ได้ไหมคะ",
        "en": "Several reminders match; please give the id (e.g. R-0012).",
    },
    # ---------------------------------------------------------------- meeting votes (votes.py)
    "vote_ballot": {
        "th": "🗳️ มติ: {question} — จากการประชุม {room}, ปิดโหวต {time}",
        "en": "🗳️ Vote: {question} — from the meeting in {room}, closes {time}",
    },
    "vote_ballot_chat": {
        "th": "🗳️ มติ: {question} — ขอโดย {opener}, ปิดโหวต {time}",
        "en": "🗳️ Vote: {question} — asked by {opener}, closes {time}",
    },
    "vote_chosen": {
        "th": "🗳️ {question}\n✅ คุณเลือก: {option} (เปลี่ยนได้จนถึง {time})",
        "en": "🗳️ {question}\n✅ You chose: {option} (you can change it until {time})",
    },
    "vote_you_chose": {"th": "คุณเลือก: {option}", "en": "You chose: {option}"},
    "vote_you_skipped": {"th": "คุณไม่ได้ลงคะแนน", "en": "You did not vote"},
    "vote_ballot_closed": {
        "th": "🔒 ปิดโหวตแล้ว: {question} — {choice}",
        "en": "🔒 Voting closed: {question} — {choice}",
    },
    "vote_press_closed": {"th": "โหวตนี้ปิดแล้วค่ะ", "en": "This vote is closed."},
    "vote_press_not_voter": {
        "th": "คุณไม่ได้อยู่ในรายชื่อผู้ลงคะแนนของมตินี้ค่ะ",
        "en": "You are not on the voter list for this vote.",
    },
    "vote_press_unknown": {"th": "ไม่พบมตินี้แล้วค่ะ", "en": "That vote no longer exists."},
    "vote_tally": {
        "th": "🗳️ มติ: {question} — โหวตแล้ว {done}/{total} ({counts}) ปิด {time}",
        "en": "🗳️ Vote: {question} — {done}/{total} voted ({counts}) closes {time}",
    },
    "vote_tally_closed": {
        "th": "🔒 มติ: {question} — ปิดแล้ว โหวต {done}/{total} ({counts})",
        "en": "🔒 Vote: {question} — closed, {done}/{total} voted ({counts})",
    },
    "vote_result": {
        "th": "{icon}มติ: {question} → {counts} ({verdict}, เสียงข้างมากของผู้ลงคะแนน)",
        "en": "{icon}Vote: {question} → {counts} ({verdict}, simple majority of ballots cast)",
    },
    "vote_v_pass": {"th": "ผ่าน", "en": "passed"},
    "vote_v_fail": {"th": "ไม่ผ่าน", "en": "rejected"},
    "vote_v_tie": {"th": "ไม่ได้ข้อสรุป", "en": "no conclusion"},
    "vote_v_none": {"th": "ไม่มีผู้ลงคะแนน", "en": "nobody voted"},
    "vote_v_win": {"th": "{option} ชนะ", "en": "{option} wins"},
    "vote_anon_tag": {"th": "[ไม่เปิดเผยชื่อ]", "en": "[anonymous]"},
    "vote_names": {"th": "{option}: {names}", "en": "{option}: {names}"},
    "vote_nonvoters": {"th": "ไม่ได้ลงคะแนน: {names}", "en": "Did not vote: {names}"},
    "vote_cue": {
        "th": "(ปิดโหวตแล้ว — แจ้งผลสั้น ๆ โดยพูดตามนี้ ห้ามเรียบเรียงใหม่: {line})",
        "en": "(The vote has closed. Announce the result briefly, say it as given, do not rephrase: {line})",
    },
    "vote_unreached": {
        "th": "ส่งปุ่มโหวตให้ไม่ถึง: {names} ค่ะ",
        "en": "Could not reach: {names}.",
    },
    "vote_minutes_header": {"th": "🗳️ มติที่ลงคะแนน", "en": "🗳️ Votes taken"},
    "vote_offer": {
        "th": "🗳️ ให้สลิดเปิดโหวตเรื่อง “{question}” ไหมคะ ({options})",
        "en": "🗳️ Shall I open a vote on “{question}”? ({options})",
    },
    "vote_offer_open": {"th": "เปิดโหวต", "en": "Open vote"},
    "vote_offer_no": {"th": "ไม่ต้อง", "en": "No need"},
    "vote_offer_opened": {
        "th": "🗳️ เปิดโหวตเรื่อง “{question}” แล้วค่ะ ส่งปุ่มไปที่ห้องส่วนตัวของแต่ละคน",
        "en": "🗳️ Vote on “{question}” is open. Buttons went to everyone's private channel.",
    },
    "vote_offer_declined": {"th": "ได้ค่ะ ไม่เปิดโหวต", "en": "Okay, no vote."},
    "vote_offer_gone": {"th": "ข้อเสนอนี้หมดอายุแล้วค่ะ", "en": "That offer is no longer valid."},
    # ---------------------------------------------------------------- concept studio
    "concept_img_disabled": {
        "th": "ตอนนี้ปิดการสร้างภาพคอนเซปต์ไว้ค่ะ ให้แอดมินเปิดก่อนนะคะ",
        "en": "Concept images are turned off here. An admin can turn them on.",
    },
    "concept_img_daily_limit": {
        "th": "วันนี้สร้างภาพคอนเซปต์ครบโควตาของเซิร์ฟเวอร์แล้วค่ะ ลองใหม่พรุ่งนี้นะคะ",
        "en": "Today's concept image limit for this server is used up. Please try again tomorrow.",
    },
    "concept_img_blocked": {
        "th": "ระบบความปลอดภัยของโมเดลไม่ยอมสร้างภาพนี้ค่ะ ลองเปลี่ยนคำอธิบายให้ต่างออกไปนะคะ",
        "en": "The image model's safety filter declined this image. Try describing it differently.",
    },
    "concept_img_timeout": {
        "th": "สร้างภาพนานเกินไปค่ะ ลองอีกครั้งได้เลยนะคะ",
        "en": "The image took too long. Please try again.",
    },
    "concept_img_error": {
        "th": "สร้างภาพไม่สำเร็จค่ะ ลองอีกครั้งได้เลยนะคะ",
        "en": "Could not make the image. Please try again.",
    },
    "concept_price_none": {
        "th": "ไม่พบราคาที่เชื่อถือได้ค่ะ ลองระบุรุ่น/ยี่ห้อ",
        "en": "No reliable price found. Try giving a brand or model.",
    },
    "concept_price_recheck": {
        "th": "บอกด้วยว่าราคาเป็นของวันที่ {date} จากเว็บ ต้องตรวจซ้ำกับร้านก่อนสั่งซื้อจริง",
        "en": "Say the prices are from the web as of {date} and must be re-checked with the shop before buying.",
    },
    "concept_price_range": {"th": "ช่วงราคา", "en": "Price range"},
    "concept_lbl_front": {"th": "ด้านหน้า", "en": "Front"},
    "concept_lbl_side": {"th": "ด้านข้าง", "en": "Side"},
    "concept_lbl_three_quarter": {"th": "มุมเฉียง", "en": "3/4 view"},
    "concept_lbl_top": {"th": "ด้านบน", "en": "Top"},
    "concept_lbl_option": {"th": "ตัวเลือก {n}", "en": "Option {n}"},
    "concept_kind_product": {"th": "ภาพสินค้า", "en": "Product render"},
    "concept_kind_scene": {"th": "ภาพการใช้งาน", "en": "In use"},
    "concept_kind_ui": {"th": "ภาพหน้าจอ", "en": "Screen mockup"},
    "concept_kind_logo": {"th": "ไอเดียโลโก้", "en": "Logo ideas"},
    "concept_kind_moodboard": {"th": "มู้ดบอร์ด", "en": "Moodboard"},
}
