"""Built-in domain packs: what a good secretary already knows about common kinds of business.

Each pack is short, Thai-first, plain data: what the business is, its vocabulary, typical tasks,
KPIs and hints for which Salid tools help. A guild picks up to three (`company_brain`), and only a
<= 600 character summary reaches the prompt. Packs are general background, never company facts:
health and legal packs carry a "no medical / legal advice" rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PACKS = 3
SUMMARY_CHARS = 600


@dataclass(frozen=True)
class Pack:
    id: str
    th: str
    en: str
    keywords: tuple[str, ...]
    about: str
    vocab: tuple[str, ...]  # "term = meaning"
    tasks: tuple[str, ...]
    kpis: tuple[str, ...]
    hints: tuple[str, ...]
    caution: str = ""


PACKS: dict[str, Pack] = {
    p.id: p
    for p in (
        Pack(
            "startup",
            "สตาร์ทอัพ/เทคโนโลยี",
            "Startup / tech",
            ("startup", "สตาร์ทอัพ", "software", "ซอฟต์แวร์", "แอป", "saas", "แพลตฟอร์ม", "tech", "ai"),
            "ทีมเล็กทำผลิตภัณฑ์ดิจิทัล ปรับเร็ว เน้นหาลูกค้ากลุ่มแรกและระดมทุน",
            (
                "MVP = ผลิตภัณฑ์ขั้นต่ำที่ทดสอบตลาดได้",
                "sprint = รอบพัฒนา 1-2 สัปดาห์",
                "MRR = รายได้ประจำรายเดือน",
                "churn = อัตราลูกค้าเลิกใช้",
                "runway = เงินสดที่พอใช้กี่เดือน",
            ),
            (
                "วางแผน sprint และ demo",
                "ติดตาม bug/release",
                "คุยลูกค้าและเก็บ feedback",
                "เตรียม pitch และงบ",
                "สมัครแข่ง/ทุน",
            ),
            ("MRR", "churn", "CAC/LTV", "runway", "active users"),
            ("business_framework", "idea_save", "market_benchmark", "list_tasks"),
        ),
        Pack(
            "restaurant",
            "ร้านอาหาร/คาเฟ่",
            "Restaurant / cafe",
            ("ร้านอาหาร", "คาเฟ่", "ร้านกาแฟ", "restaurant", "cafe", "coffee", "bar", "ครัว", "อาหาร"),
            "ขายอาหารเครื่องดื่มหน้าร้านและเดลิเวอรี ต้นทุนวัตถุดิบและกะพนักงานเป็นหัวใจ",
            (
                "food cost = ต้นทุนวัตถุดิบต่อยอดขาย",
                "GP = กำไรขั้นต้น",
                "เมนูขายดี/ตาย",
                "FIFO = ของเข้าก่อนใช้ก่อน",
                "เดลิเวอรี = Grab/LINE MAN/Robinhood",
            ),
            ("สั่งวัตถุดิบ/สต็อก", "จัดกะพนักงาน", "ปิดยอดรายวัน", "ตรวจความสะอาด", "โปรโมชันและเมนูใหม่"),
            ("food cost %", "ยอดขายต่อวัน", "ยอดบิลเฉลี่ย", "ของเสีย", "รีวิวลูกค้า"),
            ("freetime", "polls", "list_tasks", "ใบเสร็จวัตถุดิบ"),
        ),
        Pack(
            "juristic",
            "นิติบุคคลอาคารชุด/หอพัก/โรงแรม",
            "Hotel / apartment / condo juristic",
            (
                "นิติบุคคล",
                "คอนโด",
                "หอพัก",
                "อพาร์ตเมนต์",
                "โรงแรม",
                "รีสอร์ต",
                "juristic",
                "condo",
                "hotel",
                "apartment",
                "resort",
            ),
            "ดูแลอาคาร ผู้พักอาศัย/แขก ส่วนกลาง และงานซ่อมบำรุง",
            (
                "ค่าส่วนกลาง = ค่าบำรุงรักษาพื้นที่ร่วม",
                "แจ้งซ่อม = คำร้องซ่อมจากลูกบ้าน",
                "check-in/out",
                "occupancy = อัตราเข้าพัก",
                "ADR/RevPAR = ราคาห้องเฉลี่ย/รายได้ต่อห้อง",
            ),
            (
                "รับเรื่องแจ้งซ่อมและตามช่าง",
                "เก็บค่าส่วนกลาง/ค่าห้อง",
                "ประกาศถึงลูกบ้าน",
                "ประชุมคณะกรรมการ",
                "ตรวจระบบ PM ลิฟต์ ดับเพลิง",
            ),
            ("เวลาปิดงานซ่อม", "ค้างชำระ", "occupancy", "ข้อร้องเรียน"),
            ("broadcast", "list_tasks", "approvals", "deliver_file"),
        ),
        Pack(
            "retail",
            "ค้าปลีก/อีคอมเมิร์ซ",
            "Retail / e-commerce",
            (
                "ค้าปลีก",
                "ร้านค้า",
                "ออนไลน์",
                "อีคอมเมิร์ซ",
                "shopee",
                "lazada",
                "retail",
                "ecommerce",
                "e-commerce",
                "สินค้า",
            ),
            "ขายสินค้าหน้าร้านและออนไลน์ ต้องคุมสต็อก ราคา โปรโมชัน และการจัดส่ง",
            (
                "SKU = รหัสสินค้า",
                "สต็อก = จำนวนคงเหลือ",
                "COD = เก็บเงินปลายทาง",
                "AOV = ยอดต่อออเดอร์",
                "คืนสินค้า = return/refund",
            ),
            (
                "ตรวจสต็อกและสั่งเติม",
                "ตั้งโปรโมชัน/แคมเปญ",
                "แพ็กและส่งพัสดุ",
                "ตอบแชตลูกค้า",
                "กระทบยอดแพลตฟอร์ม",
            ),
            ("ยอดขาย", "AOV", "อัตราคืนสินค้า", "สต็อกหมุนเวียน", "กำไรต่อ SKU"),
            ("unit_economics", "market_benchmark", "ใบเสร็จ"),
        ),
        Pack(
            "travel",
            "สายการบิน/ท่องเที่ยว",
            "Airline / travel",
            (
                "สายการบิน",
                "ท่องเที่ยว",
                "ทัวร์",
                "ตั๋วเครื่องบิน",
                "airline",
                "travel",
                "tour",
                "flight",
                "ตัวแทนท่องเที่ยว",
            ),
            "ขายและดูแลการเดินทาง เที่ยวบิน ที่พัก ทัวร์ ต้องเช็กเวลาและเงื่อนไขตั๋วเสมอ",
            (
                "PNR = รหัสการจอง",
                "layover = เปลี่ยนเครื่อง",
                "no-show = ไม่มาตามนัด",
                "fare rules = เงื่อนไขตั๋ว",
                "ใบกำกับทัวร์ = voucher",
            ),
            (
                "ติดตามสถานะเที่ยวบิน",
                "ทำใบเสนอราคา/แพ็กเกจ",
                "ยืนยันที่พัก",
                "แจ้งตารางเดินทาง",
                "ดูแลลูกค้าเมื่อเที่ยวบินเปลี่ยน",
            ),
            ("ยอดจอง", "อัตราตรงเวลา", "คะแนนความพึงพอใจ", "กำไรต่อทริป"),
            ("track_flight", "weather", "currency", "places"),
        ),
        Pack(
            "clinic",
            "คลินิก/สถานพยาบาล",
            "Clinic / healthcare",
            (
                "คลินิก",
                "โรงพยาบาล",
                "หมอ",
                "ทันตกรรม",
                "สุขภาพ",
                "clinic",
                "hospital",
                "dental",
                "healthcare",
            ),
            "นัดหมายผู้ป่วย ตารางแพทย์ เวชภัณฑ์ และการเงิน เน้นความลับของคนไข้",
            (
                "OPD = ผู้ป่วยนอก",
                "นัด follow-up = นัดติดตามอาการ",
                "สิทธิ์ประกัน/เบิกจ่าย",
                "เวชภัณฑ์ = ยาและอุปกรณ์",
                "PDPA = ข้อมูลส่วนบุคคล",
            ),
            (
                "จัดตารางแพทย์และนัดผู้ป่วย",
                "เตือนนัด",
                "เช็กสต็อกเวชภัณฑ์",
                "ออกใบเสร็จ/เคลมประกัน",
                "ปิดยอดรายวัน",
            ),
            ("จำนวนนัด", "no-show", "เวลารอ", "รายได้ต่อวัน"),
            ("scheduling", "freetime", "approvals"),
            "ห้ามให้คำแนะนำทางการแพทย์หรือวินิจฉัย ให้ส่งต่อแพทย์ และไม่เปิดเผยข้อมูลคนไข้",
        ),
        Pack(
            "education",
            "การศึกษา/กวดวิชา",
            "Education / tutoring",
            (
                "โรงเรียน",
                "กวดวิชา",
                "สอนพิเศษ",
                "การศึกษา",
                "tutor",
                "school",
                "education",
                "คอร์ส",
                "course",
            ),
            "จัดคอร์ส ตารางสอน ผู้เรียน/ผู้ปกครอง และการชำระค่าเรียน",
            ("คอร์ส = ชุดการสอน", "ตารางสอน", "ค่าเรียน/งวด", "สอนชดเชย", "ผลสอบ = คะแนนทดสอบ"),
            ("จัดห้องและครู", "แจ้งผู้ปกครอง", "ตามค่าเรียน", "นัดสอนชดเชย", "ประกาศตารางสอบ"),
            ("จำนวนผู้เรียน", "อัตราต่อคอร์ส", "ค้างชำระ", "ความพึงพอใจ"),
            ("freetime", "broadcast", "polls", "forms"),
        ),
        Pack(
            "accounting",
            "สำนักงานบัญชี/การเงิน",
            "Accounting / finance office",
            ("บัญชี", "สำนักงานบัญชี", "ภาษี", "การเงิน", "accounting", "tax", "audit", "ตรวจสอบบัญชี"),
            "ทำบัญชี ภาษี และเอกสารการเงินให้ลูกค้า มีกำหนดยื่นที่ขยับไม่ได้",
            (
                "ภ.พ.30 = แบบ VAT รายเดือน",
                "ภ.ง.ด.3/53 = หัก ณ ที่จ่าย",
                "งบทดลอง",
                "ใบกำกับภาษี",
                "สปส. = ประกันสังคม",
            ),
            (
                "รวบรวมเอกสารลูกค้า",
                "บันทึกบัญชีรายเดือน",
                "ยื่นภาษีตามกำหนด",
                "กระทบยอดธนาคาร",
                "ตามเอกสารที่ขาด",
            ),
            ("งานเสร็จทันกำหนด", "เอกสารค้าง", "ค่าธรรมเนียมที่เก็บได้"),
            ("ใบเสร็จ", "list_tasks", "approvals", "investment_metrics"),
            "เป็นข้อมูลทั่วไป ตัวเลขภาษีต้องให้ผู้รับผิดชอบตรวจ",
        ),
        Pack(
            "construction",
            "ก่อสร้าง/อสังหาริมทรัพย์",
            "Construction / property",
            (
                "ก่อสร้าง",
                "รับเหมา",
                "อสังหา",
                "บ้าน",
                "construction",
                "property",
                "real estate",
                "โครงการ",
            ),
            "งานโครงการ ผู้รับเหมา วัสดุ งวดงานและการส่งมอบ",
            (
                "BOQ = ตารางปริมาณและราคา",
                "งวดงาน = การเบิกเงินตามความคืบหน้า",
                "variation order = งานเพิ่ม/ลด",
                "site = หน้างาน",
                "ส่งมอบ/ตรวจรับงาน",
            ),
            ("ติดตามความคืบหน้าหน้างาน", "สั่งวัสดุ", "เบิกงวดงาน", "ตรวจรับงาน", "ประสานผู้รับเหมา"),
            ("ความคืบหน้า % เทียบแผน", "งบเกิน", "งานแก้ (punch list)", "อุบัติเหตุ"),
            ("approvals", "list_tasks", "deliver_file", "freetime"),
        ),
        Pack(
            "security",
            "รักษาความปลอดภัย/CCTV",
            "Security / CCTV services",
            ("รปภ", "รักษาความปลอดภัย", "cctv", "กล้องวงจรปิด", "security", "ยาม", "access control"),
            "ให้บริการรปภ. ติดตั้งและดูแลกล้อง ระบบเข้า-ออก และตอบเหตุ",
            (
                "NVR/DVR = เครื่องบันทึกภาพ",
                "access control = ระบบควบคุมการเข้าออก",
                "รายงานเหตุ = incident report",
                "กะ รปภ.",
                "SLA = เวลาตอบสนองที่สัญญา",
            ),
            ("จัดกะ รปภ.", "รับแจ้งเหตุ/กล้องเสีย", "ตั้งนัดช่างติดตั้ง", "ส่งรายงานเหตุให้ลูกค้า", "ต่อสัญญาบริการ"),
            ("เวลาตอบสนอง", "จำนวนเหตุ", "กล้องออนไลน์ %", "ต่อสัญญา"),
            ("freetime", "list_tasks", "broadcast", "security_check"),
            "ภาพ/ข้อมูลกล้องเป็นความลับ แชร์เฉพาะผู้มีสิทธิ์",
        ),
        Pack(
            "manufacturing",
            "โรงงาน/การผลิต",
            "Manufacturing",
            ("โรงงาน", "การผลิต", "อุตสาหกรรม", "manufacturing", "factory", "production", "ผลิต"),
            "ผลิตสินค้าเป็นล็อต คุมวัตถุดิบ คุณภาพ เครื่องจักร และกำหนดส่ง",
            (
                "lot = ล็อตการผลิต",
                "QC = ตรวจคุณภาพ",
                "downtime = เครื่องหยุด",
                "MOQ = ขั้นต่ำสั่งซื้อ",
                "lead time = เวลานำ",
            ),
            ("วางแผนการผลิต", "สั่งวัตถุดิบ", "บำรุงรักษาเครื่อง", "ตรวจ QC", "นัดส่งสินค้า"),
            ("OEE", "ของเสีย %", "ส่งตรงเวลา", "สต็อกวัตถุดิบ"),
            ("unit_economics", "list_tasks", "approvals"),
        ),
        Pack(
            "legal",
            "สำนักงานกฎหมาย",
            "Legal office",
            ("กฎหมาย", "ทนาย", "สำนักงานกฎหมาย", "law", "legal", "attorney", "สัญญา"),
            "จัดการคดี สัญญา กำหนดวันศาล และเอกสารของลูกความ",
            (
                "ลูกความ = client",
                "วันนัดศาล",
                "คำฟ้อง/คำให้การ",
                "due diligence",
                "NDA = สัญญารักษาความลับ",
            ),
            (
                "ติดตามวันนัดและกำหนดยื่น",
                "ร่าง/ตรวจสัญญา (ให้ทนายตรวจ)",
                "นัดลูกความ",
                "เก็บเอกสารคดี",
                "ออกใบแจ้งหนี้",
            ),
            ("ชั่วโมงที่เรียกเก็บ", "งานเลยกำหนด", "ค้างชำระ"),
            ("scheduling", "deliver_file", "approvals"),
            "สลิดไม่ใช่ที่ปรึกษากฎหมาย ไม่ให้คำแนะนำทางกฎหมาย ให้ทนายตัดสินเสมอ",
        ),
        Pack(
            "marketing",
            "เอเจนซีการตลาด/โฆษณา",
            "Marketing agency",
            (
                "การตลาด",
                "โฆษณา",
                "เอเจนซี",
                "คอนเทนต์",
                "marketing",
                "agency",
                "advertising",
                "content",
                "แบรนด์",
            ),
            "ทำแคมเปญ คอนเทนต์ และโฆษณาให้ลูกค้า มีดีลเลย์ด้านอนุมัติงานบ่อย",
            (
                "brief = โจทย์จากลูกค้า",
                "KV = key visual",
                "ROAS = รายได้ต่อค่าโฆษณา",
                "CPC/CPM = ต้นทุนต่อคลิก/พันครั้ง",
                "content calendar = ปฏิทินคอนเทนต์",
            ),
            ("รับ brief และแตกงาน", "ส่งงานให้ลูกค้าอนุมัติ", "นัดถ่ายทำ", "รายงานผลแคมเปญ", "เสนอราคา"),
            ("ROAS", "reach/engagement", "งานส่งตรงเวลา", "revisions ต่อชิ้น"),
            ("translate", "approvals", "business_framework", "market_benchmark"),
        ),
        Pack(
            "logistics",
            "ขนส่ง/โลจิสติกส์",
            "Logistics",
            ("ขนส่ง", "โลจิสติกส์", "คลังสินค้า", "logistics", "shipping", "warehouse", "พัสดุ"),
            "รับ-ส่งสินค้า คลัง เส้นทาง และติดตามสถานะพัสดุ",
            (
                "POD = หลักฐานส่งมอบ",
                "last mile = ส่งไมล์สุดท้าย",
                "เส้นทาง = route",
                "WMS = ระบบคลัง",
                "ค่าระวาง = freight",
            ),
            ("จัดรถและเส้นทาง", "ติดตามพัสดุ", "ตอบลูกค้าเรื่องล่าช้า", "นับสต็อกคลัง", "เก็บ POD/ใบเสร็จ"),
            ("ส่งตรงเวลา %", "พัสดุเสียหาย", "ต้นทุนต่อเที่ยว", "รอบรถ"),
            ("freetime", "places", "maps", "weather"),
        ),
    )
}


def get_pack(pack_id: str) -> Pack | None:
    return PACKS.get((pack_id or "").strip().casefold())


def pack_name(p: Pack, lang: str = "th") -> str:
    return p.en if lang != "th" else p.th


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold())


def pick_packs(text: str, limit: int = MAX_PACKS) -> list[str]:
    """Pack ids that fit a free-text description of a business ("บริษัทเราเป็นร้านอาหาร"), best
    first: a pack id or name named outright, else keyword hits. [] when nothing matches."""
    s = _norm(text)
    if not s:
        return []
    scored: list[tuple[int, str]] = []
    for p in PACKS.values():
        score = 0
        if re.search(rf"\b{re.escape(p.id)}\b", s) or p.th.casefold() in s or p.en.casefold() in s:
            score += 5
        for kw in p.keywords:
            k = kw.casefold()
            if k.isascii():
                if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", s):
                    score += 2
            elif k in s:
                score += 2
        if score:
            scored.append((score, p.id))
    scored.sort(key=lambda x: -x[0])
    return [pid for _, pid in scored[: max(0, limit)]]


def pack_text(p: Pack, lang: str = "th") -> str:
    """The full pack as text (for the user: 'what do you know about restaurants?')."""
    en = lang != "th"
    lines = [f"{pack_name(p, lang)}: {p.about}"]
    lines.append(("Vocabulary: " if en else "คำศัพท์: ") + "; ".join(p.vocab))
    lines.append(("Common tasks: " if en else "งานที่พบบ่อย: ") + "; ".join(p.tasks))
    lines.append("KPI: " + ", ".join(p.kpis))
    lines.append(("Helpful tools: " if en else "เครื่องมือที่ช่วยได้: ") + ", ".join(p.hints))
    if p.caution:
        lines.append(("Caution: " if en else "ข้อควรระวัง: ") + p.caution)
    return "\n".join(lines)


def pack_summary(ids: list[str], limit: int = SUMMARY_CHARS) -> str:
    """A compact line for the prompt about the chosen packs (<= `limit` characters)."""
    packs = [p for p in (get_pack(i) for i in ids[:MAX_PACKS]) if p]
    if not packs:
        return ""
    per = max(120, limit // len(packs))
    parts = []
    for p in packs:
        head = f"{p.th}: {p.about}"
        extra = " | ระวัง: " + p.caution if p.caution else ""  # first: never the part cut off
        extra += " | KPI: " + ", ".join(p.kpis[:4]) + " | ศัพท์: " + "; ".join(p.vocab[:3])
        s = head + extra
        parts.append(s if len(s) <= per else s[: per - 1].rstrip() + "…")
    out = " // ".join(parts)
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"
