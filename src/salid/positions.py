"""Positions: ONE catalog of job positions per company, and who holds which (pure logic).

The catalog (`Position`: key, title, level, department, the Discord role that tags it, aliases)
lives in the DocStore kind `salid_positions` key "catalog"; the holders (`Assignment`) in key
"people". It is mirrored to the central org (`orgs.Member.title / level`) and to Discord roles by
`positions_bot.py`; the stated company fact is the source of truth, never a guess from chat.

Levels: 1 founder / owner (Founder, CEO, Co-founder), 2 C-level (CTO, COO, CFO, CMO...),
3 director / VP / head, 4 manager / lead, 5 senior, 6 staff, 7 intern / trainee, 0 unknown.
A title whose level cannot be told from its words is level 0 (an executive can set it).

Sources of an assignment, strongest first (`people.ROLE_RANK`): stated (an executive said so) >
org (edited in the central system) > discord (a Discord role with that name) > heard. A weaker
source never overwrites a stronger one.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .coo_store import DocStore
from .people import ROLE_RANK, clean

POSITIONS = "salid_positions"
CATALOG_KEY = "catalog"
PEOPLE_KEY = "people"
MAX_POSITIONS = 60
MAX_ALIASES = 8
MAX_TITLE = 80
DEFAULT_EXEC_MAX_LEVEL = 3
SOURCES = ("stated", "org", "discord", "heard")


class Position(BaseModel):
    key: str
    title: str
    level: int = 0
    department: str = ""
    discord_role_id: int = 0  # the Discord role that tags it (0 = none yet)
    aliases: list[str] = Field(default_factory=list)


class Assignment(BaseModel):
    user_id: int
    position_key: str
    department: str = ""
    source: str = "stated"  # stated | org | discord | heard
    at: str = ""
    mirrored: str = ""  # the title last written into the central org member ("" = never)


def norm_title(text: str) -> str:
    """Case-folded letters and digits only, so "Co-Founder", "co founder" and "ผจก." compare equal
    (Thai combining marks are kept)."""
    out = []
    for ch in unicodedata.normalize("NFKC", text or "").casefold():
        if unicodedata.category(ch)[0] in "LNM":
            out.append(ch)
    return "".join(out)


def _p(key, title, level, *aliases) -> Position:
    return Position(key=key, title=title, level=level, aliases=list(aliases))


DEFAULT_CATALOG: tuple[Position, ...] = (
    _p("founder", "Founder", 1, "ผู้ก่อตั้ง", "owner", "เจ้าของ", "เจ้าของกิจการ"),
    _p("co-founder", "Co-founder", 1, "ผู้ร่วมก่อตั้ง", "cofounder"),
    _p("ceo", "CEO", 1, "ซีอีโอ", "chief executive officer", "ประธานเจ้าหน้าที่บริหาร"),
    _p("cto", "CTO", 2, "ซีทีโอ", "chief technology officer"),
    _p("coo", "COO", 2, "ซีโอโอ", "chief operating officer"),
    _p("cfo", "CFO", 2, "ซีเอฟโอ", "chief financial officer"),
    _p("cmo", "CMO", 2, "ซีเอ็มโอ", "chief marketing officer"),
    _p("director", "Director", 3, "ผู้อำนวยการ", "ผอ", "vp", "vice president", "รองประธาน"),
    _p("head", "Head of Department", 3, "หัวหน้าฝ่าย", "หัวหน้าแผนก", "head of"),
    _p("manager", "Manager", 4, "ผู้จัดการ", "ผจก", "หัวหน้า", "หัวหน้าทีม", "team lead", "lead"),
    _p("senior", "Senior", 5, "ซีเนียร์", "อาวุโส", "senior staff"),
    _p("staff", "Staff", 6, "พนักงาน", "employee"),
    _p("intern", "Intern", 7, "นักศึกษาฝึกงาน", "ฝึกงาน", "trainee", "เด็กฝึกงาน"),
)
# Thai role names for the built-in titles that have a common Thai form (used when the company
# language is Thai); CEO / CTO... are written the same in both.
DEFAULT_TH = {
    "founder": "ผู้ก่อตั้ง",
    "co-founder": "ผู้ร่วมก่อตั้ง",
    "director": "ผู้อำนวยการ",
    "head": "หัวหน้าฝ่าย",
    "manager": "ผู้จัดการ",
    "senior": "อาวุโส",
    "staff": "พนักงาน",
    "intern": "นักศึกษาฝึกงาน",
}

_L1 = {"founder", "cofounder", "ceo", "owner", "president", "managingdirector"}
_L1_TH = tuple(
    norm_title(w) for w in ("ผู้ก่อตั้ง", "ผู้ร่วมก่อตั้ง", "ซีอีโอ", "เจ้าของ", "กรรมการผู้จัดการ", "ประธานบริษัท")
)
_L3 = {"director", "vp", "head"}
_L3_TH = tuple(
    norm_title(w) for w in ("ผู้อำนวยการ", "ผอ", "รองประธาน", "หัวหน้าฝ่าย", "หัวหน้าแผนก", "รองกรรมการ")
)
_L4 = {"manager", "lead", "supervisor", "leader"}
_L4_TH = tuple(norm_title(w) for w in ("ผู้จัดการ", "ผจก", "หัวหน้า", "ซูเปอร์ไวเซอร์"))
_L5 = {"senior", "sr"}
_L5_TH = tuple(norm_title(w) for w in ("ซีเนียร์", "อาวุโส"))
_L6 = {"staff", "employee", "junior", "associate"}
_L6_TH = tuple(norm_title(w) for w in ("พนักงาน", "จูเนียร์"))
_L7 = {"intern", "trainee", "apprentice"}
_L7_TH = tuple(norm_title(w) for w in ("ฝึกงาน", "เด็กฝึกงาน"))
_CLEVEL = re.compile(r"^c[a-z]o$")


def level_of(title: str) -> int:
    """The level a free-text title implies; the most senior word wins ("senior manager" = 4).
    0 when nothing in it says (never guessed)."""
    text = unicodedata.normalize("NFKC", title or "").casefold()
    tokens = re.findall(r"[a-z0-9]+", text)
    joined = "".join(tokens)
    thai = norm_title(re.sub(r"[a-z0-9]+", " ", text))
    levels: list[int] = []
    if _L1 & set(tokens) or "managingdirector" in joined or "chiefexecutive" in joined:
        levels.append(1)
    if any(w in thai for w in _L1_TH):
        levels.append(1)
    if "chief" in tokens or any(_CLEVEL.match(w) for w in tokens):
        levels.append(2)
    if re.search(r"\bc[a-z]o\b", text) or "ซีทีโอ" in thai or "ซีโอโอ" in thai:
        levels.append(2)
    if _L3 & set(tokens) or "vicepresident" in joined:
        levels.append(3)
    if any(w in thai for w in _L3_TH):
        levels.append(3)
    if _L4 & set(tokens) or any(w in thai for w in _L4_TH):
        levels.append(4)
    if _L5 & set(tokens) or any(w in thai for w in _L5_TH):
        levels.append(5)
    if _L6 & set(tokens) or any(w in thai for w in _L6_TH):
        levels.append(6)
    if _L7 & set(tokens) or any(w in thai for w in _L7_TH):
        levels.append(7)
    return min(levels) if levels else 0


def is_exec_level(level: int, exec_max_level: int = DEFAULT_EXEC_MAX_LEVEL) -> bool:
    return 1 <= int(level or 0) <= int(exec_max_level)


def key_of(title: str) -> str:
    """A stable catalog key from a title (Thai stays Thai)."""
    out: list[str] = []
    for ch in (title or "").casefold():
        if unicodedata.category(ch)[0] in "LNM":
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:40].strip("-") or "position"


def _same(a: Position, title: str) -> bool:
    n = norm_title(title)
    return bool(n) and (n == norm_title(a.title) or any(n == norm_title(x) for x in a.aliases))


class CatalogFull(Exception):
    """The catalog already holds MAX_POSITIONS positions."""


class PositionBook:
    """Catalog and holders of one company on a `DocStore` (SQLite or Firestore alike)."""

    def __init__(self, docs: DocStore, clock: Callable[[], datetime] | None = None):
        self.docs = docs
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()

    def now(self) -> str:
        return self._clock().isoformat(timespec="seconds")

    # ------------------------------------------------------------- catalog
    def catalog(self, gid: int) -> list[Position]:
        """The stored catalog, else a copy of the defaults (nothing is written until it changes)."""
        d = self.docs.get(POSITIONS, gid, CATALOG_KEY)
        if not d:
            return [p.model_copy(deep=True) for p in DEFAULT_CATALOG]
        out = []
        for x in d.get("positions") or []:
            try:
                out.append(Position(**x))
            except ValueError:
                continue
        return out[:MAX_POSITIONS]

    def put_catalog(self, gid: int, positions: list[Position]) -> None:
        self.docs.put(
            POSITIONS, gid, CATALOG_KEY,
            {"positions": [p.model_dump() for p in positions[:MAX_POSITIONS]]},
        )  # fmt: skip

    def find(self, gid: int, title: str, catalog: list[Position] | None = None) -> Position | None:
        for p in catalog if catalog is not None else self.catalog(gid):
            if _same(p, title):
                return p
        return None

    def get(self, gid: int, key: str) -> Position | None:
        return next((p for p in self.catalog(gid) if p.key == key), None)

    def add(
        self,
        gid: int,
        title: str,
        level: int | None = None,
        aliases: list[str] | tuple[str, ...] = (),
        department: str = "",
    ) -> tuple[Position, bool]:
        """(position, created). An existing title / alias is returned as is (new aliases are
        merged in; a given level replaces the old one). Raises CatalogFull."""
        title = clean(title, MAX_TITLE)
        if not title:
            raise ValueError("a title is needed")
        with self._lock:
            cat = self.catalog(gid)
            found = self.find(gid, title, cat)
            extra = [clean(a, MAX_TITLE) for a in aliases if clean(a, MAX_TITLE)]
            if found is not None:
                changed = False
                for a in extra:
                    if len(found.aliases) < MAX_ALIASES and not _same(found, a):
                        found.aliases.append(a)
                        changed = True
                if level is not None and 0 <= int(level) <= 7 and found.level != int(level):
                    found.level, changed = int(level), True
                if department and not found.department:
                    found.department, changed = clean(department, 60), True
                if changed:
                    self.put_catalog(gid, cat)
                return found, False
            if len(cat) >= MAX_POSITIONS:
                raise CatalogFull
            key = key_of(title)
            taken = {p.key for p in cat}
            base, n = key, 2
            while key in taken:
                key, n = f"{base}-{n}", n + 1
            lv = level_of(title) if level is None else max(0, min(int(level), 7))
            pos = Position(
                key=key, title=title, level=lv, department=clean(department, 60),
                aliases=[a for a in dict.fromkeys(extra)][:MAX_ALIASES],
            )  # fmt: skip
            cat.append(pos)
            self.put_catalog(gid, cat)
            return pos, True

    def remove(self, gid: int, key: str) -> bool:
        """Drop a position and everyone's assignment to it. The Discord role is not touched."""
        with self._lock:
            cat = self.catalog(gid)
            rest = [p for p in cat if p.key != key]
            if len(rest) == len(cat):
                return False
            self.put_catalog(gid, rest)
            held = self.assignments(gid)
            mine = {u: a for u, a in held.items() if a.position_key == key}
            if mine:
                self._write(gid, {u: a for u, a in held.items() if u not in mine})
            return True

    def set_role_id(self, gid: int, key: str, role_id: int) -> None:
        with self._lock:
            cat = self.catalog(gid)
            for p in cat:
                if p.key == key and p.discord_role_id != role_id:
                    p.discord_role_id = int(role_id)
                    self.put_catalog(gid, cat)
                    return

    def rename(self, gid: int, key: str, title: str) -> bool:
        title = clean(title, MAX_TITLE)
        with self._lock:
            cat = self.catalog(gid)
            for p in cat:
                if p.key == key and title and p.title != title:
                    p.title = title
                    self.put_catalog(gid, cat)
                    return True
        return False

    def managed_role_ids(self, gid: int) -> set[int]:
        return {p.discord_role_id for p in self.catalog(gid) if p.discord_role_id}

    @staticmethod
    def discord_role_name(position: Position, lang: str = "en") -> str:
        """The Discord role name: the title as typed; the built-in Thai form for an untouched
        built-in title when the company speaks Thai."""
        default = next((d for d in DEFAULT_CATALOG if d.key == position.key), None)
        if lang == "th" and default is not None and position.title == default.title:
            return DEFAULT_TH.get(position.key, position.title)
        return position.title

    def dept_roles(self, gid: int) -> dict[str, int]:
        """Department name -> the Discord role Salid made for it (flag `department_roles`)."""
        d = self.docs.get(POSITIONS, gid, "depts") or {}
        return {str(k): int(v) for k, v in (d.get("roles") or {}).items()}

    def set_dept_role(self, gid: int, dept: str, role_id: int) -> None:
        with self._lock:
            roles = self.dept_roles(gid)
            roles[dept] = int(role_id)
            self.docs.put(POSITIONS, gid, "depts", {"roles": roles})

    # ------------------------------------------------------------- holders
    def assignments(self, gid: int) -> dict[int, Assignment]:
        d = self.docs.get(POSITIONS, gid, PEOPLE_KEY) or {}
        out: dict[int, Assignment] = {}
        for x in (d.get("items") or {}).values():
            try:
                a = Assignment(**x)
            except ValueError:
                continue
            out[a.user_id] = a
        return out

    def _write(self, gid: int, items: dict[int, Assignment]) -> None:
        self.docs.put(
            POSITIONS, gid, PEOPLE_KEY,
            {"items": {str(u): a.model_dump() for u, a in items.items()}},
        )  # fmt: skip

    def assign(
        self,
        gid: int,
        user_id: int,
        position_key: str,
        source: str,
        department: str = "",
        *,
        force: bool = False,
    ) -> bool:
        """Give `user_id` the position. A weaker source never overwrites a stronger one (unless
        `force`, an explicit new edit in the central system). True when something changed."""
        rank = ROLE_RANK.get(source, 0)
        dept = clean(department, 60)
        with self._lock:
            held = self.assignments(gid)
            cur = held.get(user_id)
            if cur is not None and not force and rank < ROLE_RANK.get(cur.source, 0):
                return False
            if (
                cur is not None
                and cur.position_key == position_key
                and cur.source == source
                and (not dept or dept == cur.department)
            ):
                return False
            new = Assignment(
                user_id=user_id, position_key=position_key, source=source, at=self.now(),
                department=dept or (cur.department if cur else ""),
                mirrored=cur.mirrored if cur else "",
            )  # fmt: skip
            held[user_id] = new
            self._write(gid, held)
            return True

    def unassign(self, gid: int, user_id: int, only_source: str = "") -> bool:
        with self._lock:
            held = self.assignments(gid)
            cur = held.get(user_id)
            if cur is None or (only_source and cur.source != only_source):
                return False
            del held[user_id]
            self._write(gid, held)
            return True

    def set_mirrored(self, gid: int, user_id: int, title: str) -> None:
        with self._lock:
            held = self.assignments(gid)
            cur = held.get(user_id)
            if cur is not None and cur.mirrored != title:
                cur.mirrored = title
                self._write(gid, held)

    def of(self, gid: int, user_id: int) -> tuple[Assignment, Position] | None:
        a = self.assignments(gid).get(user_id)
        if a is None:
            return None
        pos = self.get(gid, a.position_key)
        return (a, pos) if pos is not None else None

    def by_level(self, gid: int) -> list[tuple[int, list[tuple[Assignment, Position]]]]:
        """Holders grouped by level, most senior first (level 0 = unknown, last)."""
        cat = {p.key: p for p in self.catalog(gid)}
        groups: dict[int, list[tuple[Assignment, Position]]] = {}
        for a in self.assignments(gid).values():
            p = cat.get(a.position_key)
            if p is not None:
                groups.setdefault(p.level, []).append((a, p))
        order = sorted(groups, key=lambda lv: (lv == 0, lv))
        return [(lv, sorted(groups[lv], key=lambda x: (x[1].title, x[0].user_id))) for lv in order]

    def level_of_user(self, gid: int, user_id: int) -> int:
        """The level of the person's position; 0 = none or unknown."""
        hit = self.of(gid, user_id)
        return hit[1].level if hit else 0
