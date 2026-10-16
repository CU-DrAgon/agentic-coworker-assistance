"""Phishing / scam checks for links and messages, and payment-slip consistency checks.

Pure heuristics, offline and deterministic (the optional Google Safe Browsing Lookup call takes an
injectable `post`). Nothing here fetches a page. The verdicts are Salid's ESTIMATE:

* `safe`        น่าจะปลอดภัย  - nothing suspicious found (never "guaranteed safe");
* `suspicious`  น่าสงสัย     - some signs; verify through another channel;
* `dangerous`   อันตราย      - strong signs (or Safe Browsing says so); do not click / pay.

`link_guard_hits` is the passive check used by the bot: only DANGEROUS links with high-confidence
signs, never an allowlisted domain. The Discord glue is `security_bot.py`.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from urllib.parse import unquote, urlparse

from .i18n import CATALOG, t

# ------------------------------------------------------------------ data
# Hosts that are never flagged (exact or any subdomain). Bank list = official sites only.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "google.com", "google.co.th", "youtube.com", "youtu.be", "discord.com", "discordapp.com",
        "discord.gg", "github.com", "githubusercontent.com", "microsoft.com", "office.com",
        "live.com", "apple.com", "icloud.com", "zoom.us", "slack.com", "notion.so", "line.me",
        "wikipedia.org", "linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
        "go.th", "gov.th", "ac.th", "bot.or.th", "set.or.th", "sec.or.th", "rd.go.th",
        "kasikornbank.com", "kbank.com", "scb.co.th", "bangkokbank.com", "krungthai.com",
        "krungsri.com", "ttbbank.com", "gsb.or.th", "baac.or.th", "uob.co.th", "cimbthai.com",
        "lhbank.co.th", "tisco.co.th", "kkpfg.com", "ibank.co.th", "truemoney.com",
        "promptpay.io", "thaipost.com", "kerryexpress.com", "flashexpress.com", "dhl.com",
    }
)  # fmt: skip
# Official domains of brands scammers imitate: brand keyword -> official registrable domains.
BRANDS: dict[str, tuple[str, ...]] = {
    "kasikorn": ("kasikornbank.com", "kbank.com", "kbank.co.th"),
    "kbank": ("kasikornbank.com", "kbank.com", "kbank.co.th"),
    "scb": ("scb.co.th",),
    "krungthai": ("krungthai.com",),
    "krungsri": ("krungsri.com",),
    "bangkokbank": ("bangkokbank.com",),
    "ttb": ("ttbbank.com",),
    "gsb": ("gsb.or.th",),
    "promptpay": ("promptpay.io",),
    "truemoney": ("truemoney.com",),
    "paypal": ("paypal.com",),
    "google": ("google.com",),
    "facebook": ("facebook.com",),
    "instagram": ("instagram.com",),
    "microsoft": ("microsoft.com", "office.com", "live.com"),
    "apple": ("apple.com", "icloud.com"),
    "netflix": ("netflix.com",),
    "line": ("line.me",),
    "shopee": ("shopee.co.th", "shopee.com"),
    "lazada": ("lazada.co.th", "lazada.com"),
    "kerry": ("kerryexpress.com",),
    "thailandpost": ("thaipost.com",),
    "flashexpress": ("flashexpress.com",),
    "revenue": ("rd.go.th",),
    "discord": ("discord.com", "discordapp.com"),
}
SHORTENERS = frozenset(
    {
        "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "shorturl.at", "s.id",
        "ow.ly", "buff.ly", "goo.gl", "t.ly", "tiny.cc", "rebrand.ly", "lnkd.in", "v.gd",
        "shorturl.asia", "qr.ae", "bl.ink", "short.io",
    }
)  # fmt: skip
SUSPICIOUS_TLDS = frozenset(
    {
        "zip", "mov", "top", "xyz", "click", "link", "icu", "cfd", "sbs", "rest", "gq", "tk", "ml",
        "ga", "cf", "buzz", "monster", "cyou", "work", "support", "country", "kim", "loan",
        "men", "bid", "date", "download", "stream", "racing", "win", "vip",
    }
)  # fmt: skip
SECOND_LEVEL = frozenset({"co", "or", "go", "ac", "in", "com", "net", "org", "ne", "gov", "edu"})
RISKY_FILES = (".apk", ".exe", ".scr", ".bat", ".msi", ".cmd", ".jar", ".vbs", ".ps1", ".dmg")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "$": "s", "@": "a", "7": "t"})

# text patterns: id -> (weight, regex). Weights add up; see `verdict_of`.
_P = re.IGNORECASE
TEXT_PATTERNS: dict[str, tuple[int, re.Pattern]] = {
    "otp": (
        3,
        re.compile(
            r"(ขอ|ส่ง|แจ้ง|บอก|ให้).{0,12}(otp|รหัส\s*(ผ่าน|ยืนยัน|ครั้งเดียว)|pin)|"
            r"(send|share|tell|give|enter).{0,20}(otp|one[- ]time (code|password)|verification code|pin)",
            _P,
        ),
    ),
    "gov_threat": (
        3,
        re.compile(
            r"(ตำรวจ|ดีเอสไอ|dsi|ปปง|ศาล|อัยการ|สรรพากร|ศุลกากร|ธนาคารแห่งประเทศไทย).{0,40}"
            r"(ฟอกเงิน|หมายจับ|อายัด|ระงับ|ผิดกฎหมาย|ยาเสพติด|ค้างชำระ)|"
            r"(police|court|revenue dept|customs|dsi).{0,40}(money laundering|warrant|freeze|arrest|unpaid)",
            _P,
        ),
    ),
    "urgency": (
        1,
        re.compile(
            r"(ด่วน|ภายใน\s*\d+\s*(ชั่วโมง|ชม\.?|นาที|วัน)|ทันที|วันนี้เท่านั้น|หมดเขต|ไม่เช่นนั้น|มิฉะนั้น)|"
            r"(urgent|immediately|within \d+ ?(hours?|minutes?)|act now|last chance|final notice|expires? (today|soon))",
            _P,
        ),
    ),
    "account_threat": (
        2,
        re.compile(
            r"(บัญชี|แอคเคาท์|account).{0,30}(ถูกระงับ|ถูกล็อก|ถูกปิด|ถูกอายัด|มีปัญหา|ผิดปกติ|suspended|locked|"
            r"disabled|compromised|unusual activity|will be (closed|terminated|suspended))",
            _P,
        ),
    ),
    "verify_credentials": (
        2,
        re.compile(
            r"(ยืนยันตัวตน|ยืนยันบัญชี|อัปเดตข้อมูล|เข้าสู่ระบบ|ล็อกอิน|กรอก(ข้อมูล)?(บัตร|รหัส)|เลขบัตรประชาชน|"
            r"verify your (account|identity)|confirm your (password|identity|details)|log ?in to (verify|confirm|restore)|"
            r"update your (payment|billing|account))",
            _P,
        ),
    ),
    "prize": (
        2,
        re.compile(
            r"(ยินดีด้วย|คุณได้รับรางวัล|ถูกรางวัล|รับรางวัล|สุ่มได้|แจกฟรี|รับเงินคืน|คืนภาษี|เงินคืน)|"
            r"(congratulations.{0,30}(won|winner|selected)|you(?:'ve| have) won|claim your (prize|reward|refund)|tax refund)",
            _P,
        ),
    ),
    "pay_first": (
        2,
        re.compile(
            r"(โอน|จ่าย|ชำระ).{0,20}(ค่าธรรมเนียม|ค่าประกัน|ค่าดำเนินการ|ค่าปลดล็อก|มัดจำ|ค่าส่ง).{0,30}(ก่อน|เพื่อรับ|เพื่อปลดล็อก)|"
            r"(pay|transfer).{0,20}(fee|deposit|insurance).{0,30}(first|to (receive|unlock|release|claim))",
            _P,
        ),
    ),
    "invest_promise": (
        2,
        re.compile(
            r"(ผลตอบแทน|กำไร|รายได้|ลงทุน).{0,15}(\d+\s*%\s*(ต่อ)?\s*(วัน|สัปดาห์)|วันละ\s*\d|ชัวร์|การันตี|ไม่มีความเสี่ยง)|"
            r"(guaranteed (returns?|profit)|risk[- ]free|\d+ ?% (daily|per day|weekly)|double your (money|bitcoin|crypto))",
            _P,
        ),
    ),
    "job_scam": (
        2,
        re.compile(
            r"(งานพาร์ทไทม์|งานออนไลน์|ทำงานที่บ้าน).{0,40}(วันละ|รายได้|ค่าตอบแทน)\s*\d|กดไลก์|กดรับออเดอร์|รีวิวสินค้า.{0,15}(รายได้|ค่าคอม)|"
            r"(part[- ]time (job|work).{0,40}(\d+ ?(baht|thb|usd|\$))|like and earn|task[- ]based job)",
            _P,
        ),
    ),
    "loan_scam": (
        2,
        re.compile(
            r"(กู้เงิน|ให้ยืมเงิน|สินเชื่อ).{0,25}(ไม่เช็ค(เครดิต|บูโร)|ไม่ต้องค้ำ|อนุมัติทันที|ไม่ตรวจ)|"
            r"(loan.{0,25}(no credit check|instant approval|guaranteed approval))",
            _P,
        ),
    ),
    "family_money": (
        2,
        re.compile(
            r"(พ่อ|แม่|ลูก|พี่|น้อง|หลาน)\s*(เอง)?.{0,25}(ขอยืมเงิน|โอนเงินให้|เบอร์ใหม่|ไลน์ใหม่|เปลี่ยนเบอร์)|"
            r"(this is your (son|daughter|boss|ceo|manager).{0,30}(new number|gift ?cards?|wire|transfer))",
            _P,
        ),
    ),
    "install_app": (
        2,
        re.compile(
            r"(ติดตั้ง|ดาวน์โหลด|โหลด).{0,25}(แอป|แอพ|ไฟล์|apk|โปรแกรม).{0,40}(ธนาคาร|ตำรวจ|กรม|ยืนยัน|รับเงิน|รับสิทธิ์)|"
            r"(install|download).{0,25}(app|apk|file).{0,40}(bank|police|verify|claim)|remote (desktop|access|control)|anydesk|teamviewer",
            _P,
        ),
    ),
    "parcel": (
        1,
        re.compile(
            r"(พัสดุ|สินค้า).{0,25}(ตกค้าง|ติดศุลกากร|ถูกตีกลับ|ไม่สามารถจัดส่ง|ที่อยู่ไม่ถูกต้อง)|"
            r"(parcel|package).{0,25}(held|on hold|could not be delivered|customs fee|address (is )?incomplete)",
            _P,
        ),
    ),
    "crypto_wallet": (
        2,
        re.compile(
            r"(seed phrase|recovery phrase|private key|wallet connect|connect (your )?wallet|คำกู้คืนกระเป๋า|"
            r"กรอกคีย์กระเป๋า|12 ?(คำ|words))",
            _P,
        ),
    ),
}
URL_RE = re.compile(
    r"https?://[^\s<>\"'`)\]]+|(?<![\w@.-])(?:[a-z0-9-]+\.)+(?:com|net|org|co\.th|"
    r"in\.th|or\.th|go\.th|th|xyz|top|click|link|icu|info|site|online|cc|ly|me|io)(?:/[^\s<>\"'`)\]]*)?",
    _P,
)
MASKED_RE = re.compile(r"\[([^\]\n]{3,200})\]\((https?://[^)\s]+)\)")

# ------------------------------------------------------------------ strings (th / en)
REASONS: dict[str, tuple[str, str]] = {
    "userinfo": ("ลิงก์มี @ ซ่อนปลายทางจริงไว้", "The link has an @ that hides the real destination"),
    "ip_host": ("ใช้เลข IP แทนชื่อเว็บ", "Uses a raw IP address instead of a website name"),
    "punycode": (
        "ชื่อเว็บใช้อักขระพิเศษ (punycode) ที่ทำให้ดูเหมือนเว็บจริง",
        "Host uses punycode, often a lookalike of a real site",
    ),
    "mixed_script": (
        "ชื่อเว็บผสมอักษรหลายภาษา อาจเลียนแบบเว็บจริง",
        "Host mixes scripts, which can imitate a real site",
    ),
    "shortener": (
        "เป็นลิงก์ย่อ มองไม่เห็นปลายทางจริง",
        "A link shortener: the real destination is hidden",
    ),
    "bad_tld": ("นามสกุลโดเมนที่พบบ่อยในลิงก์หลอกลวง", "A domain ending common in scam links"),
    "brand_lookalike": (
        "ชื่อเว็บคล้ายแบรนด์/ธนาคาร แต่ไม่ใช่เว็บทางการ",
        "Looks like a brand or bank name but is not its official site",
    ),
    "brand_typo": (
        "สะกดคล้ายเว็บทางการ (ตัวอักษรสลับ/แทนที่)",
        "Spelled like an official site with swapped or replaced letters",
    ),
    "keywords": (
        "ชื่อเว็บมีคำล่อให้ล็อกอิน/ยืนยัน/ชำระเงิน",
        "Host contains login / verify / payment bait words",
    ),
    "many_labels": (
        "ชื่อเว็บซับซ้อนผิดปกติ (หลายส่วนย่อย/ขีดกลางมาก)",
        "Unusually long host (many subdomains or hyphens)",
    ),
    "no_https": ("ไม่ใช้ HTTPS", "Not using HTTPS"),
    "risky_file": (
        "ลิงก์ไปที่ไฟล์ติดตั้ง/โปรแกรม (เช่น .apk)",
        "Links straight to an installable file (such as .apk)",
    ),
    "masked_mismatch": (
        "ข้อความลิงก์ที่เห็นไม่ตรงกับปลายทางจริง",
        "The visible link text does not match the real destination",
    ),
    "safe_browsing": (
        "Google Safe Browsing ระบุว่าเป็นลิงก์อันตราย",
        "Google Safe Browsing lists it as dangerous",
    ),
    "otp": ("ขอรหัส OTP/PIN", "Asks for an OTP / PIN"),
    "gov_threat": ("อ้างหน่วยงานรัฐ/ตำรวจเพื่อข่มขู่", "Threatens in the name of police or authorities"),
    "urgency": ("เร่งรัดให้รีบทำ", "Pressures you to act fast"),
    "account_threat": (
        "อ้างว่าบัญชีมีปัญหา/จะถูกระงับ",
        "Claims your account is suspended or has a problem",
    ),
    "verify_credentials": (
        "ให้ยืนยันตัวตน/กรอกรหัส/ข้อมูลบัตร",
        "Asks you to verify identity or enter passwords / card data",
    ),
    "prize": ("อ้างว่าได้รางวัล/เงินคืน", "Says you won a prize or a refund"),
    "pay_first": (
        "ให้จ่ายค่าธรรมเนียมก่อนจึงจะได้รับเงิน/สิ่งของ",
        "Asks for a fee before you receive money or goods",
    ),
    "invest_promise": ("สัญญาผลตอบแทนสูง/การันตี", "Promises high or guaranteed returns"),
    "job_scam": ("งานออนไลน์รายได้สูงผิดปกติ", "Too-good online job offer"),
    "loan_scam": ("เงินกู้อนุมัติทันที ไม่เช็คเครดิต", "Instant loan with no credit check"),
    "family_money": (
        "แอบอ้างคนใกล้ตัว/เจ้านายขอเงิน",
        "Pretends to be family or your boss asking for money",
    ),
    "install_app": (
        "ให้ติดตั้งแอป/ควบคุมเครื่องระยะไกล",
        "Asks you to install an app or allow remote control",
    ),
    "parcel": ("อ้างพัสดุตกค้าง/ต้องจ่ายค่าธรรมเนียม", "Claims a held parcel needing a fee"),
    "crypto_wallet": (
        "ขอ seed phrase/คีย์กระเป๋าคริปโต",
        "Asks for a crypto seed phrase or private key",
    ),
}
STRINGS: dict[str, dict[str, str]] = {
    "sec_safe": {"th": "น่าจะปลอดภัย", "en": "Looks safe"},
    "sec_suspicious": {"th": "น่าสงสัย", "en": "Suspicious"},
    "sec_dangerous": {"th": "อันตราย", "en": "Dangerous"},
    "sec_head": {
        "th": "ผลตรวจความปลอดภัย (ประเมินโดยสลิด ไม่ใช่การรับรอง)",
        "en": "Security check (Salid's estimate, not a guarantee)",
    },
    "sec_nothing": {
        "th": "ไม่พบสัญญาณน่าสงสัยจากการตรวจเบื้องต้น แต่ไม่ได้เปิดดูหน้าเว็บจริง จึงรับรองไม่ได้",
        "en": "No warning signs in the basic checks, but the page itself was not opened, so this is no guarantee.",
    },
    "sec_todo_dangerous": {
        "th": "อย่ากดลิงก์ อย่ากรอกข้อมูล/OTP อย่าโอนเงิน และอย่าติดตั้งไฟล์ ถ้าเผลอกรอกข้อมูลแล้วให้โทรธนาคาร (เบอร์หลังบัตร) ทันที เปลี่ยนรหัสผ่าน และแจ้งความออนไลน์ที่ thaipoliceonline.go.th หรือโทร 1441",
        "en": "Do not click, enter data or an OTP, pay, or install anything. If you already did, call your bank (number on the card) now, change passwords, and report at thaipoliceonline.go.th or call 1441 in Thailand.",
    },
    "sec_todo_suspicious": {
        "th": "อย่าเพิ่งกดหรือกรอกข้อมูล ตรวจสอบกับผู้ส่งผ่านช่องทางอื่นที่เชื่อถือได้ หรือเข้าเว็บ/แอปทางการด้วยการพิมพ์ชื่อเอง",
        "en": "Do not click or enter data yet. Confirm with the sender through another trusted channel, or open the official site/app by typing its address yourself.",
    },
    "sec_todo_safe": {
        "th": "ถ้ามีการขอรหัส/เงินในภายหลัง ให้ตรวจซ้ำอีกครั้งนะคะ",
        "en": "If it later asks for codes or money, check again.",
    },
    "sec_nothing_to_check": {"th": "ไม่พบลิงก์หรือข้อความให้ตรวจ", "en": "No link or message to check."},
    "sec_sb_on": {
        "th": "ตรวจกับ Google Safe Browsing แล้ว",
        "en": "Checked with Google Safe Browsing",
    },
    "sec_guard_warn": {
        "th": "ระวังลิงก์นี้นะคะ สลิดตรวจเบื้องต้นแล้วเห็นสัญญาณหลอกลวง ({reasons}) อย่าเพิ่งกดหรือกรอกข้อมูล ถ้าอยากให้ตรวจละเอียดบอกสลิดได้เลยค่ะ",
        "en": "Careful with this link: a quick check shows scam signs ({reasons}). Please do not click or enter data. Ask me for a full check if you like.",
    },
    "slip_head": {
        "th": "เช็กลิสต์สลิดโอนเงิน (ตรวจความสอดคล้อง ไม่ใช่ยืนยันกับธนาคาร)",
        "en": "Payment slip checklist (consistency only, not a bank confirmation)",
    },
    "slip_ok": {
        "th": "ไม่พบความผิดปกติจากข้อมูลบนสลิดที่อ่านได้",
        "en": "No inconsistencies found in the slip data that could be read",
    },
    "slip_issues": {"th": "พบข้อสังเกต {n} ข้อ", "en": "{n} issue(s) found"},
    "slip_cannot": {
        "th": "สลิดยืนยันกับธนาคารไม่ได้ ก่อนส่งของหรือให้บริการ ให้เช็กยอดเข้าในแอปธนาคารของตัวเอง หรือสแกน QR บนสลิดผ่านเมนู 'ตรวจสอบสลิด' ในแอปธนาคารทางการ สลิดปลอมที่ตัดต่อมาดูเหมือนจริงได้มาก",
        "en": "Salid cannot confirm with the bank. Before handing over goods or service, check the money arrived in your own bank app, or scan the slip's QR with the official bank app's slip verification. Edited fake slips can look very real.",
    },
    "slip_missing": {
        "th": "อ่านไม่ได้: {fields} (ถ้าสลิดชัดกว่านี้จะตรวจได้มากขึ้น)",
        "en": "Could not read: {fields} (a clearer slip allows more checks)",
    },
    "slip_f_future": {
        "th": "วันที่/เวลาบนสลิดอยู่ในอนาคต",
        "en": "The date/time on the slip is in the future",
    },
    "slip_f_old": {
        "th": "สลิดเก่ามาก ({days} วันก่อน) ตรงกับรายการที่เรียกเก็บหรือไม่",
        "en": "The slip is old ({days} days ago): does it match this payment?",
    },
    "slip_f_amount": {
        "th": "ยอดบนสลิด {got} ไม่ตรงกับยอดที่ต้องการ {want}",
        "en": "Slip amount {got} differs from the expected {want}",
    },
    "slip_f_ref_format": {
        "th": "เลขอ้างอิงมีรูปแบบผิดปกติ ({ref})",
        "en": "The reference number looks malformed ({ref})",
    },
    "slip_f_dup": {
        "th": "เลขอ้างอิงนี้ซ้ำกับรายการ #{id} ในบัญชี (อาจใช้สลิดซ้ำ)",
        "en": "This reference already appears on ledger entry #{id} (slip may be reused)",
    },
    "slip_f_same_party": {
        "th": "ผู้โอนและผู้รับเป็นชื่อเดียวกัน",
        "en": "Sender and receiver are the same name",
    },
    "slip_f_receiver": {
        "th": "ชื่อผู้รับบนสลิดไม่ตรงกับที่คาดไว้ ({got} / {want})",
        "en": "Receiver on the slip differs from the expected one ({got} / {want})",
    },
    "slip_f_amount_odd": {"th": "ยอดเงินผิดปกติ ({amount})", "en": "The amount looks odd ({amount})"},
    "slip_f_time_odd": {"th": "เวลาไม่ถูกรูปแบบ ({time})", "en": "The time is not valid ({time})"},
    "slip_f_bank_unknown": {
        "th": "ไม่รู้จักชื่อธนาคาร ({bank})",
        "en": "Unrecognised bank name ({bank})",
    },
}
CATALOG.update(STRINGS)


# ------------------------------------------------------------------ URL analysis
@dataclass
class Finding:
    id: str
    weight: int
    strong: bool = False  # high-confidence on its own (used by the passive link guard)


@dataclass
class UrlReport:
    url: str
    host: str
    allowlisted: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(f.weight for f in self.findings)

    @property
    def strong(self) -> bool:
        return any(f.strong for f in self.findings)

    @property
    def verdict(self) -> str:
        return verdict_of(self.score, self.strong, self.allowlisted)


def verdict_of(score: int, strong: bool = False, allowlisted: bool = False) -> str:
    if allowlisted:
        return "safe"
    if strong or score >= 5:
        return "dangerous"
    if score >= 2:
        return "suspicious"
    return "safe"


def host_of(url: str) -> str:
    u = url if "://" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def registrable(host: str) -> str:
    """The registrable domain without the public-suffix list: the last two labels, or three when
    the second-level label is co/or/go/ac/com... (example.co.th)."""
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def is_allowlisted(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in ALLOWLIST)


def _skeleton(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().translate(LEET)
    return s.replace("rn", "m").replace("vv", "w").replace("-", "")


def _skeleton_keep(s: str) -> str:
    """Like `_skeleton` but keeps the dashes (they separate brand words)."""
    return (
        unicodedata.normalize("NFKC", s)
        .lower()
        .translate(LEET)
        .replace("rn", "m")
        .replace("vv", "w")
    )


def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]]
            and a[diff[1]] == b[diff[0]]
        )
    s, lng = (a, b) if len(a) < len(b) else (b, a)
    return any(lng[:i] + lng[i + 1 :] == s for i in range(len(lng)))


def _scripts(label: str) -> set[str]:
    out = set()
    for ch in label:
        if ch.isascii():
            out.add("latin" if ch.isalpha() else "")
        else:
            try:
                out.add(unicodedata.name(ch).split()[0].lower())
            except ValueError:
                out.add("other")
    out.discard("")
    return out


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    return bool(re.fullmatch(r"0x[0-9a-f]+|\d{8,10}", host))  # hex / decimal IPv4 forms


@lru_cache(maxsize=1)
def _official_names() -> frozenset[str]:
    return frozenset(_skeleton(d.split(".")[0]) for ds in BRANDS.values() for d in ds)


def analyze_url(url: str, display_text: str | None = None) -> UrlReport:
    """Heuristics for one URL. `display_text`: the visible text of a masked link, if any."""
    host = host_of(url)
    rep = UrlReport(url=url, host=host, allowlisted=bool(host) and is_allowlisted(host))
    if not host:
        return rep
    parsed = urlparse(url if "://" in url else "http://" + url)
    fs = rep.findings
    if display_text:
        shown = [
            host_of(u)
            for u in re.findall(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}", display_text, _P)
        ]
        if shown and registrable(shown[0]) != registrable(host) and not is_allowlisted(host):
            fs.append(Finding("masked_mismatch", 5, True))
        elif shown and registrable(shown[0]) != registrable(host):
            fs.append(Finding("masked_mismatch", 3))
    if rep.allowlisted:
        # an allowlisted host can still hide a risky payload only through masking (above)
        return rep
    if parsed.username is not None:
        fs.append(Finding("userinfo", 4, True))
    if _is_ip(host):
        fs.append(Finding("ip_host", 3))
    labels = host.split(".")
    if any(lb.startswith("xn--") for lb in labels):
        fs.append(Finding("punycode", 4, True))
    elif any(len(_scripts(lb)) > 1 for lb in labels) or (
        not host.isascii() and any("latin" in _scripts(lb) for lb in labels)
    ):
        fs.append(Finding("mixed_script", 4, True))
    reg = registrable(host)
    if reg in SHORTENERS or host in SHORTENERS:
        fs.append(Finding("shortener", 2))
    if labels[-1] in SUSPICIOUS_TLDS:
        fs.append(Finding("bad_tld", 2))
    # brand lookalikes: a brand word in a host that is not that brand's official domain
    official = any(host == d or host.endswith("." + d) for ds in BRANDS.values() for d in ds)
    if not official:
        tokens = [x for x in re.split(r"[.\-]", _skeleton_keep(host)) if x]
        word = next(
            (br for br in BRANDS if br in tokens or (len(br) >= 7 and br in "".join(tokens))), None
        )
        if word:
            fs.append(Finding("brand_lookalike", 4, True))
        else:
            name = _skeleton(reg.split(".")[0])
            if any(
                len(o) >= 5 and name != o and _edit_distance_le1(o, name) for o in _official_names()
            ):
                fs.append(Finding("brand_typo", 5, True))
    if re.search(
        r"(login|signin|verify|secure|account|update|confirm|wallet|bonus|claim|payment|bank|otp)",
        host,
    ) and (host.count("-") >= 1 or len(labels) >= 4):
        fs.append(Finding("keywords", 2))
    if len(labels) >= 5 or host.count("-") >= 3 or len(host) > 50:
        fs.append(Finding("many_labels", 1))
    if parsed.scheme == "http":
        fs.append(Finding("no_https", 1))
    path = unquote(parsed.path or "").lower()
    if path.endswith(RISKY_FILES):
        fs.append(Finding("risky_file", 3, path.endswith(".apk")))
    return rep


def find_links(text: str) -> list[tuple[str, str | None]]:
    """[(url, display_text or None)]: masked markdown links first, then bare URLs. Max 10."""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    masked = MASKED_RE.findall(text or "")
    for shown, url in masked:
        out.append((url.rstrip(".,;:!?"), shown))
        seen.add(url)
    plain = MASKED_RE.sub(" ", text or "")
    for m in URL_RE.findall(plain):
        u = m.rstrip(".,;:!?’”")
        if u not in seen and len(u) > 3:
            out.append((u, None))
            seen.add(u)
    return out[:10]


# ------------------------------------------------------------------ text analysis
def analyze_text(text: str) -> list[Finding]:
    fs: list[Finding] = []
    s = (text or "")[:6000]  # no NFKC: it splits Thai sara am (ำ)
    for pid, (w, rx) in TEXT_PATTERNS.items():
        if rx.search(s):
            fs.append(Finding(pid, w))
    return fs


# ------------------------------------------------------------------ Safe Browsing
SB_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"


def safe_browsing_lookup(
    urls: list[str], api_key: str, post: Callable[..., dict] | None = None
) -> set[str]:
    """URLs Google Safe Browsing lists as dangerous. {} when no key or on any failure (fail open:
    the heuristics still run). Never logs the key or the URLs."""
    if not api_key or not urls:
        return set()
    body = {
        "client": {"clientId": "salid", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls[:20]],
        },
    }  # fmt: skip
    try:
        if post is None:
            import httpx

            r = httpx.post(SB_URL, params={"key": api_key}, json=body, timeout=5.0)
            r.raise_for_status()
            data = r.json()
        else:
            data = post(SB_URL, api_key, body)
    except Exception:  # noqa: BLE001  (never log: the exception text may contain the key)
        return set()
    return {m.get("threat", {}).get("url", "") for m in (data or {}).get("matches", [])} - {""}


# ------------------------------------------------------------------ verdict + rendering
@dataclass
class CheckResult:
    verdict: str
    score: int
    reasons: list[str]  # reason ids
    links: list[UrlReport]
    safe_browsing: bool = False


def check(
    link_or_text: str,
    *,
    safe_browsing_key: str = "",
    post: Callable[..., dict] | None = None,
) -> CheckResult | None:
    """None when there is nothing to check."""
    s = (link_or_text or "").strip()
    if not s:
        return None
    reports = [analyze_url(u, shown) for u, shown in find_links(s)]
    text_fs = analyze_text(s)
    sb_bad: set[str] = set()
    if safe_browsing_key and reports:
        sb_bad = safe_browsing_lookup([r.url for r in reports], safe_browsing_key, post)
        for r in reports:
            if r.url in sb_bad:
                r.findings.append(Finding("safe_browsing", 10, True))
    score = sum(r.score for r in reports if not r.allowlisted or r.findings) + sum(
        f.weight for f in text_fs
    )
    strong = any(r.strong for r in reports)
    # a message with a risky-looking text AND a non-allowlisted link is worse than the sum
    if text_fs and any(not r.allowlisted for r in reports) and sum(f.weight for f in text_fs) >= 1:
        score += 1
    all_allow = bool(reports) and all(r.allowlisted and not r.findings for r in reports)
    verdict = verdict_of(score, strong, False)
    if all_allow and not text_fs:
        verdict = "safe"
    ids: list[str] = []
    for f in [f for r in reports for f in r.findings] + text_fs:
        if f.id not in ids:
            ids.append(f.id)
    return CheckResult(verdict, score, ids, reports, safe_browsing=bool(safe_browsing_key))


def reason_text(rid: str, lang: str) -> str:
    th, en = REASONS.get(rid, (rid, rid))
    return en if lang == "en" else th


def render_check(res: CheckResult | None, lang: str = "th") -> str:
    if res is None:
        return t("sec_nothing_to_check", lang)
    icon = {"safe": "🟢", "suspicious": "🟡", "dangerous": "🔴"}[res.verdict]
    L = [f"{icon} {t('sec_head', lang)}", f"{t('sec_' + res.verdict, lang)}"]
    if res.reasons:
        L += [f"- {reason_text(r, lang)}" for r in res.reasons[:8]]
    else:
        L.append(t("sec_nothing", lang))
    hosts = [r.host for r in res.links if r.host]
    if hosts:
        L.append(
            ("เว็บที่ตรวจ: " if lang != "en" else "Checked hosts: ") + ", ".join(dict.fromkeys(hosts))
        )
    if res.safe_browsing:
        L.append(t("sec_sb_on", lang))
    L.append(t("sec_todo_" + res.verdict, lang))
    return "\n".join(L)


def link_guard_hits(text: str, *, safe_browsing_key: str = "", post=None) -> list[UrlReport]:
    """DANGEROUS links with a high-confidence sign only, never allowlisted: what the passive
    guard may warn about. Masked links count too."""
    links = find_links(text)
    reports = [analyze_url(u, shown) for u, shown in links]
    if safe_browsing_key and reports:
        bad = safe_browsing_lookup(
            [r.url for r in reports if not r.allowlisted], safe_browsing_key, post
        )
        for r in reports:
            if r.url in bad:
                r.findings.append(Finding("safe_browsing", 10, True))
    return [r for r in reports if not r.allowlisted and r.strong and r.verdict == "dangerous"]


def guard_message(hits: list[UrlReport], lang: str) -> str:
    ids: list[str] = []
    for r in hits:
        for f in r.findings:
            if f.id not in ids:
                ids.append(f.id)
    reasons = ", ".join(reason_text(i, lang) for i in ids[:2])
    return t("sec_guard_warn", lang, reasons=reasons)


# ------------------------------------------------------------------ payment slips
BANKS: dict[str, tuple[str, ...]] = {
    "kbank": ("กสิกร", "kasikorn", "kbank", "k plus", "kplus"),
    "scb": ("ไทยพาณิชย์", "scb", "siam commercial"),
    "bbl": ("กรุงเทพ", "bangkok bank", "bbl", "bualuang"),
    "ktb": ("กรุงไทย", "krungthai", "ktb", "krungthai next"),
    "bay": ("กรุงศรี", "krungsri", "bay", "kma"),
    "ttb": ("ทีเอ็มบีธนชาต", "ttb", "tmb", "thanachart"),
    "gsb": ("ออมสิน", "gsb", "government savings"),
    "baac": ("ธ.ก.ส", "ธกส", "baac"),
    "uob": ("ยูโอบี", "uob"),
    "cimb": ("ซีไอเอ็มบี", "cimb"),
    "lhb": ("แลนด์ แอนด์ เฮ้าส์", "lh bank", "lhbank"),
    "tisco": ("ทิสโก้", "tisco"),
    "kkp": ("เกียรตินาคิน", "kiatnakin", "kkp"),
    "ghb": ("อาคารสงเคราะห์", "ghb", "government housing"),
    "promptpay": ("พร้อมเพย์", "promptpay"),
    "truemoney": ("ทรูมันนี่", "truemoney", "true money"),
    "ibank": ("อิสลาม", "islamic bank", "ibank"),
}
_REF_OK = re.compile(r"^[A-Za-z0-9]{10,32}$")


def bank_key(name: str | None) -> str | None:
    low = (name or "").strip().lower()
    if not low:
        return None
    for key, aliases in BANKS.items():
        if any(a in low for a in aliases):
            return key
    return None


def _amount(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(re.sub(r"[,\s฿]|THB|บาท", "", str(x), flags=re.I))
    except ValueError:
        return None


def _parse_dt(d: str | None, tm: str | None, tz) -> tuple[date | None, str | None]:
    day = None
    if d:
        s = d.strip()
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            y, mo, da = int(m[1]), int(m[2]), int(m[3])
            if y > 2400:
                y -= 543
            try:
                day = date(y, mo, da)
            except ValueError:
                day = None
    bad_time = None
    if tm:
        mt = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", tm)
        if not mt or int(mt[1]) > 23 or int(mt[2]) > 59:
            bad_time = tm.strip()[:20]
    return day, bad_time


def verify_slip_data(
    *,
    bank: str | None = None,
    amount=None,
    date_iso: str | None = None,
    time_hhmm: str | None = None,
    ref_no: str | None = None,
    sender: str | None = None,
    receiver: str | None = None,
    expected_amount=None,
    expected_receiver: str | None = None,
    ledger_refs: Callable[[str], int | None] | None = None,
    now: datetime,
    lang: str = "th",
) -> dict:
    """The slip checklist from fields already read off the image. `ledger_refs(ref) -> receipt id
    | None` finds the same reference in the ledger. Returns the findings and a rendered text; it
    never says a slip is genuine."""
    issues: list[str] = []
    missing: list[str] = []
    amt = _amount(amount)
    if amt is None:
        missing.append("amount")
    elif amt <= 0 or amt > 100_000_000 or round(amt, 2) != amt:
        issues.append(t("slip_f_amount_odd", lang, amount=str(amount)))
    day, bad_time = _parse_dt(date_iso, time_hhmm, None)
    if day is None:
        missing.append("date")
    else:
        today = now.date()
        if day > today:
            issues.append(t("slip_f_future", lang))
        elif (today - day).days > 60:
            issues.append(t("slip_f_old", lang, days=(today - day).days))
        elif day == today and time_hhmm and not bad_time:
            mt = re.match(r"^\s*(\d{1,2}):(\d{2})", time_hhmm)
            if mt and (int(mt[1]), int(mt[2])) > (
                now.hour,
                now.minute + 5 if now.minute < 55 else 59,
            ):
                issues.append(t("slip_f_future", lang))
    if bad_time:
        issues.append(t("slip_f_time_odd", lang, time=bad_time))
    if not time_hhmm:
        missing.append("time")
    if not bank:
        missing.append("bank")
    elif bank_key(bank) is None:
        issues.append(t("slip_f_bank_unknown", lang, bank=bank[:30]))
    ref = (ref_no or "").strip()
    if not ref:
        missing.append("ref")
    else:
        compact = re.sub(r"[\s-]", "", ref)
        if not _REF_OK.match(compact) or len(set(compact)) <= 3:
            issues.append(t("slip_f_ref_format", lang, ref=ref[:30]))
        if ledger_refs is not None:
            rid = ledger_refs(compact)
            if rid is not None:
                issues.append(t("slip_f_dup", lang, id=rid))
    if not sender:
        missing.append("sender")
    if not receiver:
        missing.append("receiver")
    if sender and receiver and _name_key(sender) == _name_key(receiver):
        issues.append(t("slip_f_same_party", lang))
    exp = _amount(expected_amount)
    if exp is not None and amt is not None and abs(exp - amt) > 0.005:
        issues.append(t("slip_f_amount", lang, got=f"{amt:,.2f}", want=f"{exp:,.2f}"))
    if expected_receiver and receiver and not _names_match(expected_receiver, receiver):
        issues.append(t("slip_f_receiver", lang, got=receiver[:40], want=expected_receiver[:40]))
    verdict = "issues" if issues else "no_issues_found"
    L = [t("slip_head", lang)]
    L += [f"- {i}" for i in issues] if issues else [t("slip_ok", lang)]
    if issues:
        L.insert(1, t("slip_issues", lang, n=len(issues)))
    if missing:
        L.append(t("slip_missing", lang, fields=", ".join(missing)))
    L.append(t("slip_cannot", lang))
    return {
        "status": verdict,
        "issues": issues,
        "missing_fields": missing,
        "can_confirm_with_bank": False,
        "text": "\n".join(L),
    }


def _name_key(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "").lower()
    s = re.sub(r"\b(mr|mrs|ms|miss|นาย|นาง|นางสาว|น\.ส\.|บจก\.?|บริษัท|หจก\.?)\b\.?", " ", s)
    return re.sub(r"[^\w]+", "", s)


def _names_match(a: str, b: str) -> bool:
    ka, kb = _name_key(a), _name_key(b)
    if not ka or not kb:
        return True  # nothing to compare
    return ka in kb or kb in ka or _edit_distance_le1(ka[:8], kb[:8])


# ------------------------------------------------------------------ passive guard rate limit
class GuardLimiter:
    """Keeps the passive link guard quiet: one warning per host per 6 h per guild, one per channel
    per 10 min, 20 per guild per hour. In memory (one bot process)."""

    HOST_S, CHANNEL_S, GUILD_S, GUILD_MAX = 6 * 3600.0, 600.0, 3600.0, 20

    def __init__(self, clock: Callable[[], float]):
        self._clock = clock
        self._hosts: dict[tuple[int, str], float] = {}
        self._channels: dict[int, float] = {}
        self._guild: dict[int, list[float]] = {}

    def allow(self, guild_id: int, channel_id: int, hosts: list[str]) -> bool:
        """True (and records the warning) when a warning for these hosts may be sent now."""
        now = self._clock()
        if now - self._channels.get(channel_id, -1e12) < self.CHANNEL_S:
            return False
        fresh = [h for h in hosts if now - self._hosts.get((guild_id, h), -1e12) >= self.HOST_S]
        if not fresh:
            return False
        recent = [x for x in self._guild.get(guild_id, []) if now - x < self.GUILD_S]
        if len(recent) >= self.GUILD_MAX:
            self._guild[guild_id] = recent
            return False
        recent.append(now)
        self._guild[guild_id] = recent
        self._channels[channel_id] = now
        for h in hosts:
            self._hosts[(guild_id, h)] = now
        if len(self._hosts) > 5000:  # bounded memory
            for k in sorted(self._hosts, key=self._hosts.get)[:2500]:
                del self._hosts[k]
        return True
