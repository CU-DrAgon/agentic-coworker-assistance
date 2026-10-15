from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date

from google.genai import types
from pydantic import BaseModel

from .config import Settings
from .i18n import t
from .llm import generate
from .models import CATEGORIES, Receipt, ReceiptExtraction
from .store import LOW_CONF, Store

logger = logging.getLogger(__name__)

MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
    "pdf": "application/pdf",
}

PROMPT = """You extract data from a receipt / invoice / bill image or PDF (usually Thai restaurant or shop expenses).
- If it is NOT a receipt, invoice or bill (selfie, food photo, screenshot of chat, etc.) set is_receipt=false and leave the rest empty.
- receipt_date: ISO YYYY-MM-DD in the Gregorian calendar. Thai Buddhist years (e.g. 2569 or 69) must be converted (BE - 543). null if not visible.
- currency default THB. Numbers are plain floats (no thousands separators). total is the final amount paid incl. VAT.
- category must be one of: {cats}. Pick the best fit (ingredients/food/ice/meat/veg -> วัตถุดิบ; tools, machines -> อุปกรณ์; fuel/taxi/toll -> ค่าเดินทาง; water/electric/internet -> สาธารณูปโภค; rent -> ค่าเช่า; wages -> ค่าแรง; ads/printing -> การตลาด).
- confidence: 0-1, how sure you are that vendor, date and total are correct. Lower it for blurry or partial images.
- Keep Thai text as Thai. The submitter's note below is extra context, not data to copy blindly.
Submitter note: {note}"""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guess_mime(filename: str, content_type: str | None = None) -> str | None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    ct = (content_type or "").split(";")[0].strip().lower()
    return ct if ct in MIME_BY_EXT.values() else None


def _fix_date(d: str | None) -> str | None:
    if not d:
        return None
    try:
        parsed = date.fromisoformat(d.strip()[:10])
    except ValueError:
        return None
    if parsed.year > 2400:  # Buddhist era slipped through
        parsed = parsed.replace(year=parsed.year - 543)
    return parsed.isoformat()


def extract_receipt(
    client, settings: Settings, data: bytes, mime: str, note: str = ""
) -> ReceiptExtraction:
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ReceiptExtraction,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    prompt = PROMPT.format(cats=", ".join(CATEGORIES), note=note or "(none)")
    resp = generate(
        client,
        settings.gemini_model,
        [types.Part.from_bytes(data=data, mime_type=mime), prompt],
        cfg,
    )
    ex = ReceiptExtraction.model_validate_json(resp.text)
    ex.receipt_date = _fix_date(ex.receipt_date)
    ex.confidence = min(max(ex.confidence, 0.0), 1.0)
    return ex


# ---------------------------------------------------------------- ingest


@dataclass
class IngestResult:
    status: str  # created | duplicate | ignored
    reply: str | None = None
    reaction: str | None = None
    receipt: Receipt | None = None


def fmt_money(x: float | None, currency: str = "THB", lang: str = "th") -> str:
    if x is None:
        return t("unknown", lang)
    s = f"{x:,.2f}".removesuffix(".00")
    return f"฿{s}" if currency == "THB" else f"{s} {currency}"


def confirmation_text(r: Receipt, dup: Receipt | None, needs_check: bool, lang: str = "th") -> str:
    lines = [
        f"{t('rc_check' if needs_check else 'rc_saved', lang)} (#{r.id})",
        t("rc_vendor", lang, v=r.vendor or t("unknown", lang)),
        t("rc_date", lang, v=r.receipt_date or t("rc_date_unknown", lang)),
        t("rc_total", lang, v=fmt_money(r.total, r.currency, lang)),
        t("rc_category", lang, v=r.category),
    ]
    if needs_check:
        lines.append(t("rc_unclear", lang))
    if dup:
        lines.append(t("rc_possible_dup", lang, id=dup.id))
    return "\n".join(lines)


def ingest_attachment(
    store: Store,
    client,
    settings: Settings,
    *,
    guild_id: int,
    data: bytes,
    filename: str,
    content_type: str | None,
    submitter_id: int,
    submitter_name: str,
    channel_id: int,
    message_id: int,
    note: str = "",
    archive=None,
    lang: str = "th",
) -> IngestResult:
    mime = guess_mime(filename, content_type)
    if not mime:
        return IngestResult("ignored")
    sha = sha256_hex(data)
    existing = store.get_by_sha(guild_id, sha)
    if existing:
        return IngestResult(
            "duplicate",
            t(
                "rc_duplicate",
                lang,
                id=existing.id,
                vendor=existing.vendor or t("unknown", lang),
                total=fmt_money(existing.total, existing.currency, lang),
            ),
            "♻️",
            existing,
        )
    ex = extract_receipt(client, settings, data, mime, note)
    if not ex.is_receipt:
        return IngestResult("ignored")
    now = settings.now()
    r = Receipt(
        guild_id=guild_id,
        submitter_id=submitter_id,
        submitter_name=submitter_name,
        channel_id=channel_id,
        message_id=message_id,
        sha256=sha,
        vendor=ex.vendor,
        receipt_date=ex.receipt_date,
        effective_date=ex.receipt_date or now.date().isoformat(),
        currency=ex.currency or "THB",
        subtotal=ex.subtotal,
        vat=ex.vat,
        total=ex.total,
        category=ex.category,
        items=ex.items,
        payment_method=ex.payment_method,
        confidence=ex.confidence,
        notes=ex.notes,
        context=note or None,
        created_at=now.isoformat(timespec="seconds"),
    )
    if archive is not None:  # best effort: a storage outage must not lose the receipt
        try:
            r.original_uri = archive.save(guild_id, sha, data, mime)
        except Exception:
            logger.exception("receipt archive upload failed (receipt still saved)")
    dup = store.find_possible_duplicate(r)
    if dup:
        r.possible_duplicate_of = dup.id
    r = store.add_receipt(r)
    needs_check = ex.confidence < LOW_CONF or ex.total is None
    return IngestResult(
        "created",
        confirmation_text(r, dup, needs_check, lang),
        "⚠️" if needs_check else "✅",
        r,
    )


# ---------------------------------------------------------------- corrections

_NUM = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_FIELD_RES = [
    (
        "total",
        re.compile(r"(?:ยอดจริง|ยอดรวม|ยอดเงิน|ยอด|total)\s*(?:คือ|เป็น|=|:)?\s*฿?\s*" + _NUM, re.I),
    ),
    ("vat", re.compile(r"(?:vat|ภาษี)\s*(?:คือ|เป็น|=|:)?\s*฿?\s*" + _NUM, re.I)),
    (
        "date",
        re.compile(
            r"(?:วันที่|date)\s*(?:คือ|เป็น|=|:)?\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})",
            re.I,
        ),
    ),
    ("category", re.compile(r"(?:หมวดหมู่|หมวด|category)\s*(?:คือ|เป็น|=|:)?\s*(\S+)", re.I)),
    ("vendor", re.compile(r"(?:ร้าน|vendor)\s*(?:คือ|เป็น|=|:)?\s*(.+?)(?:\n|$)", re.I)),
]


def _parse_date(s: str) -> str | None:
    try:
        if "-" in s:
            y, m, d = (int(x) for x in s.split("-"))
        else:
            d, m, y = (int(x) for x in s.split("/"))
        if y < 100:
            y += 2000
        if y > 2400:
            y -= 543
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_correction_simple(text: str) -> dict:
    """Regex parse of Thai/English corrections into store field updates. Returns {} if nothing recognised."""
    out: dict = {}
    for name, rx in _FIELD_RES:
        m = rx.search(text)
        if not m:
            continue
        v = m.group(1).strip()
        if name in ("total", "vat"):
            out[name] = float(v.replace(",", ""))
        elif name == "date":
            d = _parse_date(v)
            if d:
                out["receipt_date"] = d
        elif name == "category":
            cat = next((c for c in CATEGORIES if c == v or c in v), None)
            if cat:
                out["category"] = cat
        else:
            out["vendor"] = v
    return out


class _Corr(BaseModel):
    total: float | None = None
    vat: float | None = None
    vendor: str | None = None
    receipt_date: str | None = None
    category: str | None = None
    notes: str | None = None


def parse_correction(client, settings: Settings, text: str, receipt: Receipt | None = None) -> dict:
    fields = parse_correction_simple(text)
    if fields or client is None:
        return fields
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_Corr,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    prompt = (
        f"A user is correcting a saved receipt. Current: vendor={receipt.vendor if receipt else None}, total={receipt.total if receipt else None}, "
        f"date={receipt.receipt_date if receipt else None}, category={receipt.category if receipt else None}.\n"
        f"Categories: {', '.join(CATEGORIES)}. Dates ISO (convert Buddhist era). Return only the fields the user wants to change, null otherwise. "
        f"If the message is not a correction, return all null.\nUser message: {text}"
    )
    ex = _Corr.model_validate_json(generate(client, settings.gemini_model, prompt, cfg).text)
    out = {k: v for k, v in ex.model_dump().items() if v is not None}
    if "category" in out and out["category"] not in CATEGORIES:
        del out["category"]
    if "receipt_date" in out:
        d = _fix_date(out["receipt_date"])
        out.pop("receipt_date") if not d else out.update(receipt_date=d)
    return out


def apply_correction(
    store: Store, client, settings: Settings, receipt: Receipt, text: str, lang: str = "th"
) -> str:
    fields = parse_correction(client, settings, text, receipt)
    if not fields:
        return t("corr_unclear", lang)
    r = store.update_receipt(receipt.id, {**fields, "confidence": 1.0})  # human-verified
    names = {
        "total": t("corr_total", lang),
        "vat": t("corr_vat", lang),
        "vendor": t("corr_vendor", lang),
        "receipt_date": t("corr_date", lang),
        "category": t("corr_category", lang),
        "notes": t("corr_notes", lang),
    }
    changed = ", ".join(
        f"{names[k]}={fmt_money(v, lang=lang) if k in ('total', 'vat') else v}"
        for k, v in fields.items()
    )
    now = (
        f"{r.vendor or t('unknown', lang)} | {r.receipt_date or r.effective_date} | "
        f"{fmt_money(r.total, r.currency, lang)} | {r.category}"
    )
    return t("corr_done", lang, id=r.id, changed=changed, now=now)
