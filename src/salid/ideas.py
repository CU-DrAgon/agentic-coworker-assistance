"""Idea / project workspace: a living document per idea, kept by Salid while a member develops it.

A member brainstorms a product idea over many messages and asks for who benefits, a Business Model
Canvas, SWOT, competitors, how to sell, pricing... Each answer is saved as a SECTION of one project
document (DocStore kind `salid_ideas`, SQLite or Firestore), which can be read back, shared and
exported as PDF / DOCX / Markdown. `business_framework` writes a complete framework (see
`salid.frameworks`), attaches it as a PDF and saves it as a section.

Who can read an idea (decided here in code, never by the model):
* the owner always; `private`: only the owner;
* `normal` (default, like a DM note): the owner and the people who outrank the owner;
* `team`: everyone in the server; `shared_with`: the people named by the owner.
Only the owner writes. The command center library shows ideas to executives (not `private`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from . import frameworks as fw
from .coo_store import IDEAS, DocStore
from .ideas_pdf import markdown_docx, markdown_pdf
from .pdf_report import FontSet
from .wiki import slugify

logger = logging.getLogger(__name__)

MAX_IDEAS_PER_USER = 50
MAX_TITLE = 120
MAX_SECTION_CHARS = 30000
MAX_LIST = 30
FORMATS = ("pdf", "docx", "md")
VISIBILITIES = ("normal", "private", "team")

# canonical sections in document order: key -> (th, en)
SECTIONS: dict[str, tuple[str, str]] = {
    "problem": ("ปัญหาและไอเดีย", "Problem / Idea"),
    "customers": ("กลุ่มลูกค้าเป้าหมาย", "Target customers"),
    "bmc": ("Business Model Canvas", "Business Model Canvas"),
    "lean_canvas": ("Lean Canvas", "Lean Canvas"),
    "value_proposition": ("Value Proposition", "Value Proposition"),
    "swot": ("SWOT", "SWOT"),
    "pestel": ("PESTEL", "PESTEL"),
    "porter5": ("Porter's Five Forces", "Porter's Five Forces"),
    "competitors": ("คู่แข่งและความใหม่ของไอเดีย", "Competitors & novelty"),
    "gtm": ("แผนเข้าสู่ตลาด / วิธีขาย", "Go-to-market"),
    "pricing": ("แพ็กเกจและราคา", "Pricing"),
    "pitch_outline": ("โครงนำเสนอ (Pitch)", "Pitch outline"),
    "next_steps": ("ขั้นตอนถัดไป", "Next steps"),
    "notes": ("บันทึกเพิ่มเติม", "Notes"),
}
_SECTION_WORDS = {
    "problem": "problem idea problem/idea ปัญหา ไอเดีย แนวคิด overview summary concept",
    "customers": "customers target customers target-customers who benefits users audience beneficiaries "
    "ลูกค้า กลุ่มลูกค้า กลุ่มเป้าหมาย ใครได้ประโยชน์ ผู้ใช้ segments personas",
    "bmc": "bmc business model canvas business-model-canvas โมเดลธุรกิจ canvas",
    "lean_canvas": "lean lean canvas lean-canvas",
    "value_proposition": "value proposition value-proposition vpc คุณค่า",
    "swot": "swot",
    "pestel": "pestel pest",
    "porter5": "porter porter5 five forces porters five forces",
    "competitors": "competitors competitor competition novelty competitors & novelty market "
    "คู่แข่ง ความใหม่ ตลาด ไอเดียใหม่แค่ไหน landscape",
    "gtm": "gtm go-to-market go to market sales how to sell วิธีขาย การขาย การตลาด marketing แผนขาย",
    "pricing": "pricing price prices pricing_tiers packages ราคา แพ็กเกจ แพ็คเกจ",
    "pitch_outline": "pitch pitch_outline pitch outline นำเสนอ",
    "next_steps": "next steps next-steps next_steps todo actions ขั้นตอนถัดไป สิ่งที่ต้องทำ",
    "notes": "notes note บันทึก อื่นๆ other",
}
_SECTION_LOOKUP = {w: k for k, words in _SECTION_WORDS.items() for w in words.split()}
_SECTION_LOOKUP.update(
    {"problem/idea": "problem", "go-to-market": "gtm", "target customers": "customers"}
)


def ascii_name(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:40] or "idea"


def section_key(name: str | None) -> tuple[str | None, str]:
    """(canonical key or None, display title) for a section name from the model or the user."""
    raw = " ".join((name or "").split())[:80]
    if not raw:
        return None, ""
    low = raw.casefold()
    for cand in (low, low.replace("_", " "), low.replace("-", " "), low.split("(")[0].strip()):
        if cand in _SECTION_LOOKUP:
            return _SECTION_LOOKUP[cand], raw
    for key, (th, en) in SECTIONS.items():
        if low in (th.casefold(), en.casefold(), key):
            return key, raw
    return None, raw


class IdeaSection(BaseModel):
    title: str
    body: str = ""
    updated_at: str = ""


class IdeaDoc(BaseModel):
    slug: str
    title: str
    owner_id: int
    owner_name: str = ""
    sections: dict[str, IdeaSection] = Field(default_factory=dict)
    visibility: str = "private"  # owner only until they share it
    shared_with: list[int] = Field(default_factory=list)
    shared_names: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    offered: bool = False  # the "keep it all as one project document?" question was asked
    version: int = 1

    @property
    def key(self) -> str:
        return f"{self.owner_id}_{self.slug}"

    def ordered(self) -> list[tuple[str, IdeaSection]]:
        canon = [k for k in SECTIONS if k in self.sections]
        custom = [k for k in self.sections if k not in SECTIONS]
        return [(k, self.sections[k]) for k in [*canon, *custom]]


def can_read(
    doc: IdeaDoc, user_id: int, outranked: frozenset[int] | set[int] = frozenset()
) -> bool:
    """May `user_id` read the idea? `outranked`: the owners whose notes this user outranks."""
    if doc.owner_id == user_id or user_id in doc.shared_with:
        return True
    if doc.visibility == "team":
        return True
    if doc.visibility == "private":  # the default: owner (and people they named) only
        return False
    return doc.owner_id in outranked


def library_visible(doc: IdeaDoc, is_exec: bool) -> bool:
    """The command center library: `team` ideas to everyone, the rest (not `private`) to
    executives."""
    if doc.visibility == "team":
        return True
    return is_exec and doc.visibility != "private"


def render_markdown(doc: IdeaDoc, lang: str = "th", *, owner_line: bool = True) -> str:
    """The whole project document: title, owner / date line, then every section (headings inside
    a section are moved one level down)."""
    out = [f"# {doc.title}", ""]
    if owner_line:
        day = (doc.updated_at or doc.created_at)[:10]
        by = "โดย" if lang == "th" else "by"
        upd = "อัปเดต" if lang == "th" else "updated"
        out += [f"{by} {doc.owner_name or '-'} · {upd} {day}", ""]
    for key, sec in doc.ordered():
        title = SECTIONS[key][0 if lang == "th" else 1] if key in SECTIONS else sec.title
        out += [f"## {title}", "", demote(sec.body), ""]
    if not doc.sections:
        out.append("(ยังไม่มีเนื้อหา)" if lang == "th" else "(empty)")
    return "\n".join(out).rstrip() + "\n"


def demote(md: str) -> str:
    """Section body for the full document: its headings go one level down (h6 stays)."""
    out, fenced = [], False
    for line in md.strip().split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced and re.match(r"^#{1,5}\s", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def strip_h1(md: str) -> str:
    return re.sub(r"\A\s*#\s+[^\n]*\n+", "", md.strip(), count=1)


class IdeaRepo:
    def __init__(self, docs: DocStore, clock: Callable[[], datetime] | None = None):
        self.docs = docs
        self._clock = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def _doc(d: dict) -> IdeaDoc | None:
        try:
            return IdeaDoc(**{k: v for k, v in d.items() if k in IdeaDoc.model_fields})
        except ValueError:
            return None

    def all(self, guild_id: int) -> list[IdeaDoc]:
        return [x for d in self.docs.find(IDEAS, guild_id) if (x := self._doc(d))]

    def get(self, guild_id: int, owner_id: int, slug: str) -> IdeaDoc | None:
        d = self.docs.get(IDEAS, guild_id, f"{owner_id}_{slug}")
        return self._doc(d) if d else None

    def put(self, guild_id: int, doc: IdeaDoc) -> IdeaDoc:
        doc.updated_at = self._clock().isoformat(timespec="seconds")
        self.docs.put(IDEAS, guild_id, doc.key, doc.model_dump())
        return doc

    def delete(self, guild_id: int, doc: IdeaDoc) -> None:
        self.docs.delete(IDEAS, guild_id, doc.key)

    def mine(self, guild_id: int, owner_id: int) -> list[IdeaDoc]:
        return [d for d in self.all(guild_id) if d.owner_id == owner_id]

    def save_section(
        self,
        guild_id: int,
        owner_id: int,
        owner_name: str,
        title: str,
        section: str | None,
        content: str,
        *,
        append: bool = False,
    ) -> tuple[IdeaDoc, str, bool]:
        """(doc, section key, offer): store `content` as a section of the owner's idea `title`
        (created on first use). `offer`: true once, when the document first holds 2+ sections."""
        title = " ".join((title or "").split())[:MAX_TITLE]
        if not title:
            raise ValueError("title is empty")
        content = (content or "").strip()
        if not content:
            raise ValueError("content is empty")
        slug = slugify(title)
        doc = self.get(guild_id, owner_id, slug)
        if doc is None:
            if len(self.mine(guild_id, owner_id)) >= MAX_IDEAS_PER_USER:
                raise ValueError(f"too many ideas (max {MAX_IDEAS_PER_USER}): delete one first")
            now = self._clock().isoformat(timespec="seconds")
            doc = IdeaDoc(
                slug=slug, title=title, owner_id=owner_id, owner_name=owner_name, created_at=now
            )
        key, shown = section_key(section)
        if key is None and not shown:
            key = "notes" if doc.sections else "problem"
        if key is None:
            key = "x-" + slugify(shown)
        sec_title = SECTIONS[key][0] if key in SECTIONS else shown
        old = doc.sections.get(key)
        body = f"{old.body}\n\n{content}" if (append and old) else content
        doc.sections[key] = IdeaSection(
            title=sec_title,
            body=body[:MAX_SECTION_CHARS],
            updated_at=self._clock().isoformat(timespec="seconds"),
        )
        doc.owner_name = owner_name or doc.owner_name
        doc.version += 1
        offer = len(doc.sections) >= 2 and not doc.offered
        if offer:
            doc.offered = True
        self.put(guild_id, doc)
        return doc, key, offer


# One block for the persona (added to the system prompt when the idea tools exist).
IDEAS_PROMPT = (
    "\nโปรเจกต์ไอเดีย: เมื่อสมาชิกกำลังพัฒนาไอเดียผลิตภัณฑ์/ธุรกิจต่อเนื่องหลายข้อความ ให้ตั้งชื่อไอเดียสั้นๆ ชื่อเดียวตลอด "
    "(title เดิมทุกครั้ง) แล้วเก็บเป็นเอกสารโปรเจกต์ที่ต่อยอดได้ (ส่วนตัวของเจ้าของ). "
    "ขอ BMC/Lean Canvas/SWOT/PESTEL/Porter 5/Value Proposition/ราคาและแพ็กเกจ/แผนขาย (GTM)/โครงพิตช์ ให้เรียก business_framework "
    "(kind=bmc|lean_canvas|swot|pestel|porter5|value_proposition|pricing_tiers|gtm|pitch_outline) ครั้งเดียว: เครื่องมือเขียนเอกสารครบทุกหัวข้อ แนบ PDF และเก็บเข้าโปรเจกต์ให้เอง "
    "ตอบแค่ summary ที่ได้ (ไม่เกิน 8 บรรทัด) แล้วบอกว่าแนบไฟล์เต็มไว้ ห้ามวางเอกสารทั้งฉบับในแชต ห้ามเรียก idea_save ซ้ำสำหรับเฟรมเวิร์กนั้น. "
    "คำตอบอื่นของไอเดียนั้น (ใครได้ประโยชน์ -> customers, ปัญหา/แนวคิด -> problem, คู่แข่งและความใหม่ -> competitors "
    "(ค้น web_search ก่อน และใส่ลิงก์แหล่งอ้างอิงทุกข้อ), วิธีขาย -> gtm, ขั้นตอนถัดไป -> next_steps) ให้ตอบตามปกติแล้วเรียก idea_save(title, content_markdown, section) "
    "เก็บฉบับเต็มที่ตอบไป. เมื่อผลลัพธ์มี offer_project=true ให้ถามผู้ใช้ครั้งเดียวว่า “เก็บรวมเป็นเอกสารโปรเจกต์ให้ไหมคะ” "
    "(ถ้าตกลงเรียก idea_export เป็น pdf/docx/md); ห้ามถามซ้ำทุกข้อความ. "
    "idea_get / idea_list ดูเอกสารที่เก็บไว้ ใช้ก่อนทำต่อเมื่อผู้ใช้กลับมาคุยไอเดียเดิม. "
    '“แชร์ไอเดียนี้ให้ทีม / ให้คุณ A” -> idea_share(title, to="team" หรือชื่อคน); “เก็บเป็นความลับ” -> to="private"; '
    "ไอเดียเป็นของผู้ถามเอง ค่าเริ่มต้นเห็นเฉพาะเจ้าของและผู้ที่ตำแหน่งสูงกว่า (เหมือนโน้ตใน DM). "
    "ห้ามแต่งตัวเลขตลาด/สถิติ/แหล่งอ้างอิงเอง; ราคาและตัวเลขที่เสนอเป็นสมมติฐานต้องบอกว่าเป็นสมมติฐาน.\n"
)


@dataclass
class IdeasCtx:
    guild_id: int
    guild_name: str
    asker_id: int
    asker_name: str
    is_exec: bool
    repo: IdeaRepo
    client: object  # genai client (None: frameworks unavailable)
    model: str
    lang: str
    now: Callable[[], datetime]
    outranked: Callable[[], frozenset[int]] = lambda: frozenset()
    private: Callable[[], bool] = lambda: True  # a DM / the asker's private channel (not a room)
    resolve_person: Callable[[str], list[tuple[int, str]]] = lambda name: []
    fonts: Callable[[], FontSet] = lambda: FontSet(None, None, False)
    outbox: list[tuple[str, bytes]] = field(default_factory=list)  # files the reply attaches

    # ---- reading
    def readable(self) -> list[IdeaDoc]:
        out = self.outranked()
        docs = [d for d in self.repo.all(self.guild_id) if can_read(d, self.asker_id, out)]
        docs.sort(key=lambda d: d.updated_at, reverse=True)  # newest first, own ideas before others
        docs.sort(key=lambda d: d.owner_id != self.asker_id)
        return docs

    def _blocked(self, doc: IdeaDoc) -> bool:
        """Not for a shared room: outside a private chat only `team` ideas may be shown."""
        return doc.visibility != "team" and not self.private()

    def _private_only(self, doc: IdeaDoc) -> dict:
        return {
            "status": "use_private_chat",
            "title": doc.title,
            "instruction": (
                "This idea is not shared with the team, so do not show it here. Ask the user to "
                "continue in their private Salid channel or a DM (or share it with the team first)."
            ),
        }

    def find(self, title: str) -> tuple[IdeaDoc | None, list[IdeaDoc]]:
        """(the idea, candidates): own ideas first; an exact title or slug wins, else a unique
        partial match; several partial matches are returned as candidates."""
        docs = self.readable()
        slug, low = slugify(title), " ".join((title or "").split()).casefold()
        if not low:
            return None, docs
        for pool in ([d for d in docs if d.owner_id == self.asker_id], docs):
            exact = [d for d in pool if d.slug == slug or d.title.casefold() == low]
            if exact:
                return exact[0], []
        part = [d for d in docs if low in d.title.casefold() or d.slug in slug or slug in d.slug]
        if len(part) == 1:
            return part[0], []
        return None, part

    # ---- tools
    def idea_save(self, title: str, content_markdown: str, section: str | None = None) -> dict:
        try:
            doc, key, offer = self.repo.save_section(
                self.guild_id, self.asker_id, self.asker_name, title, section, content_markdown
            )
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        return self._saved(doc, key, offer)

    def _saved(self, doc: IdeaDoc, key: str, offer: bool) -> dict:
        res = {
            "status": "saved",
            "title": doc.title,
            "section": doc.sections[key].title,
            "sections_now": [s.title for _k, s in doc.ordered()],
            "visibility": doc.visibility,
            "offer_project": offer,
        }
        if offer:
            res["instruction"] = (
                "Ask the user ONCE: เก็บรวมเป็นเอกสารโปรเจกต์ให้ไหมคะ (offer idea_export as pdf, docx or md)"
                if self.lang == "th"
                else "Ask the user ONCE whether to compile everything into one project document "
                "(offer idea_export as pdf, docx or md)."
            )
        return res

    def idea_get(self, title: str) -> dict:
        doc, cands = self.find(title)
        if doc is None:
            return {
                "status": "not_found",
                "ideas": [self._row(d) for d in cands][:MAX_LIST],
            }
        if self._blocked(doc):
            return self._private_only(doc)
        return {
            "status": "ok",
            **self._row(doc),
            "markdown": render_markdown(doc, self.lang),
        }

    def _row(self, d: IdeaDoc) -> dict:
        return {
            "title": d.title,
            "owner": "you" if d.owner_id == self.asker_id else d.owner_name,
            "sections": [s.title for _k, s in d.ordered()],
            "updated_at": d.updated_at,
            "visibility": d.visibility,
            "shared_with": d.shared_names if d.owner_id == self.asker_id else [],
        }

    def idea_list(self) -> dict:
        docs = self.readable()
        shown = [d for d in docs if not self._blocked(d)]
        out = {"count": len(shown), "ideas": [self._row(d) for d in shown[:MAX_LIST]]}
        if len(shown) < len(docs):
            out["hidden_here"] = len(docs) - len(shown)  # private ideas: ask in a private chat
        return out

    def idea_share(self, title: str, to: str) -> dict:
        doc = next(
            (d for d in self.repo.mine(self.guild_id, self.asker_id) if self._is(d, title)), None
        )
        if doc is None:
            mine = self.repo.mine(self.guild_id, self.asker_id)
            return {"status": "not_found", "your_ideas": [d.title for d in mine][:MAX_LIST]}
        who = (to or "").strip()
        low = who.casefold()
        if low in ("team", "ทีม", "everyone", "ทุกคน", "all"):
            doc.visibility = "team"
            note = "everyone in the server can now read it (not edit)"
        elif low in ("private", "ส่วนตัว", "ความลับ", "secret", "confidential"):
            doc.visibility, doc.shared_with, doc.shared_names = "private", [], []
            note = "only you can read it now; not even people with a higher role"
        elif low in ("me", "none", "default", "ฉัน", "ผม", "เฉพาะฉัน", "reset"):
            doc.visibility, doc.shared_with, doc.shared_names = "private", [], []
            note = "back to the default: only you can read it"
        elif low in ("superiors", "boss", "หัวหน้า", "ผู้บริหาร", "managers", "higher roles"):
            doc.visibility = "normal"
            note = "you and people with a higher role can read it"
        else:
            found = [p for p in self.resolve_person(who) if p[0] != self.asker_id]
            if not found:
                return {"status": "person_not_found", "name": who}
            if len(found) > 1:
                return {"status": "ambiguous", "candidates": [n for _i, n in found][:8]}
            uid, name = found[0]
            if uid not in doc.shared_with:
                doc.shared_with.append(uid)
                doc.shared_names.append(name)
            note = f"{name} can now read it (not edit); nobody was notified"
        self.repo.put(self.guild_id, doc)
        return {
            "status": "ok",
            "title": doc.title,
            "visibility": doc.visibility,
            "shared_with": doc.shared_names,
            "note": note,
        }

    @staticmethod
    def _is(d: IdeaDoc, title: str) -> bool:
        low = " ".join((title or "").split()).casefold()
        return bool(low) and (d.title.casefold() == low or d.slug == slugify(title))

    def idea_export(self, title: str, format: str = "pdf") -> dict:  # noqa: A002
        fmt = (format or "pdf").casefold().lstrip(".")
        fmt = {"markdown": "md", "word": "docx", "doc": "docx"}.get(fmt, fmt)
        if fmt not in FORMATS:
            return {"status": "error", "error": "format must be pdf, docx or md"}
        doc, cands = self.find(title)
        if doc is None:
            return {"status": "not_found", "ideas": [self._row(d) for d in cands][:MAX_LIST]}
        if self._blocked(doc):
            return self._private_only(doc)
        if not doc.sections:
            return {"status": "error", "error": "the idea has no content yet"}
        md = render_markdown(doc, self.lang)
        name = f"{ascii_name(doc.title)}.{fmt}"
        try:
            if fmt == "md":
                data = md.encode("utf-8")
            elif fmt == "docx":
                data = markdown_docx(doc.title, md)
            else:
                data = self._pdf(doc, md)
        except Exception:
            logger.exception("idea export failed")
            return {"status": "error", "error": "could not build the file"}
        self.outbox.append((name, data))
        return {"status": "ok", "file": name, "format": fmt, "title": doc.title}

    def _pdf(self, doc: IdeaDoc, md: str) -> bytes:
        bmc = doc.sections.get("bmc")
        return markdown_pdf(
            org=self.guild_name,
            title=doc.title,
            markdown=md,
            lang=self.lang,
            generated_at=self.now(),
            fonts=self.fonts(),
            canvas=bmc.body if bmc else None,
        )

    def business_framework(self, kind: str, idea: str, context: str = "") -> dict:
        k = fw.resolve_kind(kind)
        if k is None:
            return {"status": "error", "error": "unknown kind", "kinds": list(fw.KINDS)}
        if self.client is None:
            return {"status": "error", "error": "no model available"}
        title = " ".join((idea or "").split())[:MAX_TITLE]
        if not title:
            return {"status": "error", "error": "idea is empty"}
        mine = self.repo.get(self.guild_id, self.asker_id, slugify(title))
        existing = render_markdown(mine, self.lang, owner_line=False) if mine else ""
        try:
            res = fw.generate_framework(
                self.client, self.model, k, title, context, self.lang, existing
            )
        except Exception:
            logger.warning("framework generation failed", exc_info=True)
            return {"status": "error", "error": "could not write the document; try again"}
        section = fw.KIND_SECTION[k]
        try:
            doc, key, offer = self.repo.save_section(
                self.guild_id,
                self.asker_id,
                self.asker_name,
                title,
                section,
                fw_body(res.markdown),
            )
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        file = f"{ascii_name(title)}-{k}.pdf"
        attached = True
        try:
            pdf = markdown_pdf(
                org=self.guild_name,
                title=f"{res.title}: {title}",
                markdown=res.markdown,
                lang=self.lang,
                generated_at=self.now(),
                fonts=self.fonts(),
                canvas=res.markdown if k == "bmc" else None,
            )
            self.outbox.append((file, pdf))
        except Exception:
            logger.warning("framework PDF failed", exc_info=True)
            attached = False
            self.outbox.append((file.replace(".pdf", ".md"), res.markdown.encode("utf-8")))
        out = self._saved(doc, key, offer)
        out.update(
            {
                "framework": res.title,
                "summary": fw.short_summary(res.markdown, self.lang),
                "attached_file": file if attached else file.replace(".pdf", ".md"),
                "complete": res.complete,
                "instruction": (
                    "Reply with the summary only (max 8 lines) and say the full document is attached "
                    "and saved in the project. Do not paste the document. "
                )
                + (out.get("instruction") or ""),
            }
        )
        return out


def fw_body(markdown: str) -> str:
    """The saved section: the framework without its own `# title` line."""
    return strip_h1(markdown)


def ideas_tools(ctx: IdeasCtx) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions).
    Attached files are collected in `ctx.outbox` (`outbox` attribute: see `qa.ask_async`)."""

    def safe(fn: Callable[[], dict]) -> dict:
        try:
            return fn()
        except Exception:
            logger.exception("idea tool failed")
            return {"status": "error", "error": "the idea tool failed; try again"}

    def idea_save(title: str, content_markdown: str, section: str = "") -> dict:
        """Save (replace) one section of the user's idea / project document, creating the document
        on first use. Call it after you answered an idea question that is not a framework
        (business_framework saves its own): who benefits (section "customers"), problem and concept
        ("problem"), competitors and novelty with sources ("competitors"), how to sell ("gtm"),
        next steps ("next_steps"). Save the FULL answer you gave, in Markdown, so it can be reused.
        When the result has offer_project=true ask the user once whether to compile everything
        into one project document.

        Args:
            title: short name of the idea, the same every time for the same idea.
            content_markdown: the complete section in Markdown (headings, bullets, source links).
            section: problem | customers | bmc | swot | competitors | gtm | pricing | next_steps |
                notes, or any other section name.
        """
        return safe(lambda: ctx.idea_save(title, content_markdown, section or None))

    def idea_get(title: str) -> dict:
        """Read an idea / project document (the user's own, or one shared with or visible to them)
        as Markdown, to continue working on it or answer questions about it.

        Args:
            title: the idea's title (a partial title is fine).
        """
        return safe(lambda: ctx.idea_get(title))

    def idea_list() -> dict:
        """List the idea / project documents the user may read: title, owner, sections, last
        update, who it is shared with."""
        return safe(ctx.idea_list)

    def idea_export(title: str, format: str = "pdf") -> dict:  # noqa: A002
        """Export an idea / project document as a file attached to the reply: the whole document
        with all saved sections (a Business Model Canvas page is included when it has one).

        Args:
            title: the idea's title.
            format: "pdf" (default), "docx" or "md".
        """
        return safe(lambda: ctx.idea_export(title, format))

    def idea_share(title: str, to: str) -> dict:
        """Change who can read the user's own idea document. Only its owner can. Nobody is
        notified; tell the owner that. By default an idea is visible to its owner only.

        Args:
            title: the idea's title.
            to: "team" (everyone in the server), "superiors" (people with a higher role), a
                person's name (read access for them), or "private" / "me" (owner only, the
                default).
        """
        return safe(lambda: ctx.idea_share(title, to))

    def business_framework(kind: str, idea: str, context: str = "") -> dict:
        """Write a COMPLETE business framework document for an idea with all its sections, attach
        it to the reply as a PDF (a Business Model Canvas also gets a 9-box canvas page) and save
        it in the idea's project document. Reply with the returned `summary` only (it is at most 8
        lines) and say the full file is attached. Takes 10-30 seconds.

        Args:
            kind: bmc | lean_canvas | swot | pestel | porter5 | value_proposition | pricing_tiers
                (Thai market: THB, VAT 7% note) | gtm | pitch_outline.
            idea: the idea's short title, the same every time for the same idea.
            context: everything the user said about the idea so far (the concept, customers,
                numbers, constraints), in a few sentences.
        """
        return safe(lambda: ctx.business_framework(kind, idea, context))

    tools = [idea_save, idea_get, idea_list, idea_export, idea_share, business_framework]
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]  # qa.ask_async attaches these files
    return tools
