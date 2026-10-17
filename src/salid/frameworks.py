"""Business frameworks (BMC, SWOT, pricing...) with a fixed, complete structure.

`generate_framework` asks the model for the whole document in Markdown with exactly the headings of
the kind, checks that every one is there, asks once more for the missing ones, and as a last resort
adds the missing headings with a placeholder so the structure is never incomplete. The chat only
shows `short_summary` (<= 8 lines); the full text goes to a PDF and into the idea document.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from google.genai import types

from .i18n import lang_name
from .llm import generate

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 8000
SUMMARY_LINES = 8
SUMMARY_LINE_CHARS = 110

# kind -> (display name th, en, [(heading th, heading en), ...])
KINDS: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "bmc": (
        "Business Model Canvas",
        "Business Model Canvas",
        [
            ("กลุ่มลูกค้า", "Customer Segments"),
            ("คุณค่าที่นำเสนอ", "Value Propositions"),
            ("ช่องทาง", "Channels"),
            ("ความสัมพันธ์กับลูกค้า", "Customer Relationships"),
            ("แหล่งรายได้", "Revenue Streams"),
            ("ทรัพยากรหลัก", "Key Resources"),
            ("กิจกรรมหลัก", "Key Activities"),
            ("พันธมิตรหลัก", "Key Partnerships"),
            ("โครงสร้างต้นทุน", "Cost Structure"),
        ],
    ),
    "lean_canvas": (
        "Lean Canvas",
        "Lean Canvas",
        [
            ("ปัญหา", "Problem"),
            ("กลุ่มลูกค้า", "Customer Segments"),
            ("จุดขายที่แตกต่าง", "Unique Value Proposition"),
            ("ทางออก", "Solution"),
            ("ช่องทาง", "Channels"),
            ("แหล่งรายได้", "Revenue Streams"),
            ("โครงสร้างต้นทุน", "Cost Structure"),
            ("ตัวชี้วัดหลัก", "Key Metrics"),
            ("ความได้เปรียบที่ลอกเลียนยาก", "Unfair Advantage"),
        ],
    ),
    "swot": (
        "SWOT",
        "SWOT",
        [
            ("จุดแข็ง", "Strengths"),
            ("จุดอ่อน", "Weaknesses"),
            ("โอกาส", "Opportunities"),
            ("อุปสรรค", "Threats"),
            ("แนวทางที่ควรทำ", "Recommended Actions"),
        ],
    ),
    "pestel": (
        "PESTEL",
        "PESTEL",
        [
            ("การเมือง", "Political"),
            ("เศรษฐกิจ", "Economic"),
            ("สังคม", "Social"),
            ("เทคโนโลยี", "Technological"),
            ("สิ่งแวดล้อม", "Environmental"),
            ("กฎหมาย", "Legal"),
        ],
    ),
    "porter5": (
        "Porter's Five Forces",
        "Porter's Five Forces",
        [
            ("ภัยคุกคามจากผู้เข้าใหม่", "Threat of New Entrants"),
            ("อำนาจต่อรองของซัพพลายเออร์", "Bargaining Power of Suppliers"),
            ("อำนาจต่อรองของผู้ซื้อ", "Bargaining Power of Buyers"),
            ("ภัยคุกคามจากสินค้าทดแทน", "Threat of Substitutes"),
            ("การแข่งขันในอุตสาหกรรม", "Competitive Rivalry"),
        ],
    ),
    "value_proposition": (
        "Value Proposition Canvas",
        "Value Proposition Canvas",
        [
            ("งานที่ลูกค้าต้องทำ", "Customer Jobs"),
            ("ความเจ็บปวด", "Pains"),
            ("ผลประโยชน์ที่ต้องการ", "Gains"),
            ("สินค้าและบริการ", "Products & Services"),
            ("ตัวลดความเจ็บปวด", "Pain Relievers"),
            ("ตัวสร้างผลประโยชน์", "Gain Creators"),
            ("ประโยคคุณค่า", "Value Proposition Statement"),
        ],
    ),
    "pricing_tiers": (
        "แพ็กเกจและราคา",
        "Pricing tiers",
        [
            ("หลักการตั้งราคา", "Pricing Approach"),
            ("แพ็กเกจเริ่มต้น", "Starter Package"),
            ("แพ็กเกจมาตรฐาน", "Standard Package"),
            ("แพ็กเกจพรีเมียม", "Premium Package"),
            ("บริการเสริม", "Add-ons"),
            ("ตารางเปรียบเทียบแพ็กเกจ", "Package Comparison"),
            ("VAT และการเรียกเก็บเงิน", "VAT and Billing"),
            ("สมมติฐานที่ต้องทดสอบ", "Assumptions to Validate"),
        ],
    ),
    "gtm": (
        "แผนเข้าสู่ตลาด (Go-to-Market)",
        "Go-to-Market plan",
        [
            ("กลุ่มเป้าหมายแรก", "Beachhead Segment"),
            ("จุดยืนและข้อความสื่อสาร", "Positioning and Messaging"),
            ("ช่องทางขาย", "Sales Channels"),
            ("ขั้นตอนการขาย", "Sales Process"),
            ("แผนการตลาด", "Marketing Plan"),
            ("พันธมิตร", "Partnerships"),
            ("แผนนำร่อง 90 วัน", "90-day Pilot Plan"),
            ("ตัวชี้วัดและเป้าหมาย", "KPIs and Targets"),
        ],
    ),
    "pitch_outline": (
        "โครงนำเสนอ (Pitch)",
        "Pitch outline",
        [
            ("ปัญหา", "Problem"),
            ("ทางออก", "Solution"),
            ("ขนาดตลาด", "Market Size"),
            ("ผลิตภัณฑ์", "Product"),
            ("โมเดลธุรกิจ", "Business Model"),
            ("ความคืบหน้าและหลักฐาน", "Traction"),
            ("คู่แข่ง", "Competition"),
            ("แผนเข้าสู่ตลาด", "Go-to-Market"),
            ("ทีม", "Team"),
            ("การเงินและสิ่งที่ขอ", "Financials and Ask"),
        ],
    ),
}
# the document section each kind is saved under in the idea document (see ideas.SECTIONS)
KIND_SECTION = {
    "bmc": "bmc",
    "lean_canvas": "lean_canvas",
    "swot": "swot",
    "pestel": "pestel",
    "porter5": "porter5",
    "value_proposition": "value_proposition",
    "pricing_tiers": "pricing",
    "gtm": "gtm",
    "pitch_outline": "pitch_outline",
}
ALIASES = {
    "business model canvas": "bmc",
    "canvas": "bmc",
    "lean": "lean_canvas",
    "lean canvas": "lean_canvas",
    "porter": "porter5",
    "porter5": "porter5",
    "five forces": "porter5",
    "vpc": "value_proposition",
    "value proposition": "value_proposition",
    "pricing": "pricing_tiers",
    "price": "pricing_tiers",
    "pricing tiers": "pricing_tiers",
    "go-to-market": "gtm",
    "go to market": "gtm",
    "pitch": "pitch_outline",
    "pitch outline": "pitch_outline",
}

PRICING_DEFAULTS = (
    "Thai market defaults for pricing: prices in Thai baht (THB, ฿); say clearly whether each price "
    "includes or excludes VAT 7% (state B2B prices as ex-VAT and add the VAT-inclusive figure for "
    "one tier as an example); monthly price with an annual option (typically 15-20% off); round to "
    "psychological Thai price points (e.g. 990, 1,990, 4,990, 9,900); small businesses (SMEs) pay "
    "little per month, so offer a free pilot or low-cost starter; mention PromptPay/bank transfer, "
    "invoice and withholding tax 3% for B2B; mark every number as an ASSUMPTION to validate with "
    "customers. Never present invented market data as fact."
)


@dataclass
class FrameworkResult:
    kind: str
    title: str
    markdown: str
    complete: bool  # every heading was produced by the model (no placeholder added)
    missing: list[str] = field(default_factory=list)  # headings added as placeholders


def resolve_kind(kind: str) -> str | None:
    k = (kind or "").strip().casefold().replace("-", "_")
    k = ALIASES.get(k, ALIASES.get(k.replace("_", " "), k))
    return k if k in KINDS else None


def kind_title(kind: str, lang: str) -> str:
    th, en, _ = KINDS[kind]
    return th if lang == "th" else en


def heading_line(pair: tuple[str, str], lang: str) -> str:
    th, en = pair
    return f"## {th} ({en})" if lang == "th" else f"## {en}"


def required_headings(kind: str, lang: str) -> list[str]:
    """The exact `## ...` heading lines the document must contain, in order."""
    return [heading_line(p, lang) for p in KINDS[kind][2]]


def _norm(s: str) -> str:
    return re.sub(r"[^\w฀-๿]+", " ", s.casefold()).strip()


def missing_headings(markdown: str, kind: str) -> list[tuple[str, str]]:
    """Required headings without a heading line (`#`..`####`) naming them in Thai or English."""
    heads = [_norm(m) for m in re.findall(r"^\s{0,3}#{1,4}\s+(.+?)\s*#*\s*$", markdown, re.M)]
    out = []
    for th, en in KINDS[kind][2]:
        t_, e_ = _norm(th), _norm(en)
        if not any(t_ in h or e_ in h for h in heads):
            out.append((th, en))
    return out


def build_prompt(kind: str, idea: str, context: str, lang: str, existing: str = "") -> str:
    th, en, pairs = KINDS[kind]
    name = th if lang == "th" else en
    lines = "\n".join(heading_line(p, lang) for p in pairs)
    language = lang_name(lang) if lang != "th" else "Thai"
    extra = PRICING_DEFAULTS if kind == "pricing_tiers" else ""
    if kind in ("bmc", "lean_canvas"):
        extra += (
            " Under every heading write 3-5 concrete bullet points for THIS idea (no generic "
            "filler); the bullets are placed in a one-page canvas, so keep each under 90 characters."
        )
    return (
        f"Write a complete {name} document in {language} Markdown for the business idea below.\n"
        f"Start with one title line `# <short title>` and a 2-sentence intro, then use EXACTLY these "
        f"section headings, in this order, each with real, specific content (bullets, tables where "
        f"useful). Do not skip, merge or rename any heading, do not stop early, add no sections "
        f"after the last heading except an optional `## Assumptions` line, and no text before the "
        f"title.\n{lines}\n\n"
        f"{extra}\n"
        "Be concrete for this idea and the Thai market where relevant. State assumptions as "
        "assumptions; never invent statistics, sources or numbers presented as facts.\n"
        "Output only the Markdown.\n\n"
        f"IDEA: {idea}\n"
        f"CONTEXT FROM THE USER: {context or '-'}\n"
        + (
            f"WHAT IS ALREADY IN THE PROJECT DOCUMENT (stay consistent):\n{existing[:6000]}\n"
            if existing
            else ""
        )
    )


def _clean(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", text).strip()


def _call(client, model: str, prompt: str) -> str:
    resp = generate(
        client,
        model,
        prompt,
        types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ),
    )
    return _clean(getattr(resp, "text", "") or "")


def generate_framework(
    client,
    model: str,
    kind: str,
    idea: str,
    context: str = "",
    lang: str = "th",
    existing: str = "",
) -> FrameworkResult:
    """The full Markdown of a framework with every required heading (see module doc)."""
    prompt = build_prompt(kind, idea, context, lang, existing)
    text = _call(client, model, prompt)
    gone = missing_headings(text, kind)
    if gone:  # re-ask once, naming what is missing
        logger.info("framework %s: %d headings missing, asking again", kind, len(gone))
        again = (
            prompt
            + "\n\nYOUR PREVIOUS ANSWER WAS INCOMPLETE. These headings were missing:\n"
            + "\n".join(heading_line(p, lang) for p in gone)
            + "\nWrite the COMPLETE document again from the title, with ALL headings.\n"
            + f"PREVIOUS ANSWER:\n{text[:6000]}"
        )
        try:
            retry = _call(client, model, again)
        except Exception:
            logger.warning("framework retry failed", exc_info=True)
            retry = ""
        if retry and len(missing_headings(retry, kind)) <= len(gone):
            text = retry
        gone = missing_headings(text, kind)
    placeholder = (
        "(ยังไม่มีข้อมูลในส่วนนี้ ขอให้สลิดเติมให้ได้)"
        if lang == "th"
        else "(not filled in yet; ask Salid to complete it)"
    )
    for pair in gone:
        text = f"{text.rstrip()}\n\n{heading_line(pair, lang)}\n{placeholder}"
    title = kind_title(kind, lang)
    if not re.match(r"^\s*#\s+\S", text):
        text = f"# {title}: {idea[:80]}\n\n{text}"
    return FrameworkResult(kind, title, text, not gone, [p[1] for p in gone])


# ------------------------------------------------------------------ parsing / summary
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.M)


def split_h2(markdown: str) -> list[tuple[str, str]]:
    """[(heading text, body)] for every `##` section."""
    matches = list(_H2.finditer(markdown))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        out.append((m.group(1), markdown[m.end() : end].strip()))
    return out


def _first_point(body: str) -> str:
    for line in body.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        s = re.sub(r"[*_`]+", "", s)
        if s and not s.startswith("|") and not s.startswith("#"):
            return s if len(s) <= SUMMARY_LINE_CHARS else s[: SUMMARY_LINE_CHARS - 1].rstrip() + "…"
    return ""


def short_summary(markdown: str, lang: str = "th", lines: int = SUMMARY_LINES) -> str:
    """At most `lines` lines: the first point of each section (the rest is in the file)."""
    secs = [(h, _first_point(b)) for h, b in split_h2(markdown)]
    secs = [(re.sub(r"\s*\(.*?\)\s*$", "", h) if lang == "th" else h, p) for h, p in secs]
    rows = [f"• {h}: {p}" if p else f"• {h}" for h, p in secs]
    if len(rows) > lines:
        more = len(rows) - (lines - 1)
        tail = f"… และอีก {more} หัวข้อในไฟล์แนบ" if lang == "th" else f"… and {more} more in the file"
        rows = [*rows[: lines - 1], tail]
    return "\n".join(rows)
