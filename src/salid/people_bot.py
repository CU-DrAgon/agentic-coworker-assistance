"""Discord glue of company knowledge (logic: `people.py`, `people_listen.py`, `people_tools.py`,
`handover.py`).

`PeopleMixin` gives `SecretaryBot`:
- the listener (`people_listener`, chained after the project listener by `coo._on_summary_all`) and
  the finished-task feed;
- `_people_tools`: the agent tools, appended to `extra_tools` by `_agent_kwargs` (chat and voice);
- the nightly company-map refresh (`_people_due`, bot tick): roles from Discord and the org member
  list, project owners, then `CompanyPlaybook.org / who_to_ask / key_contacts`, built only from what
  the whole server can read;
- leaving: `on_member_remove` / the `person_left` tool mark the dossier `left`, list the open tasks,
  write the handover with the main model, keep it in the dossier and the wiki, and send the PDF to
  the executives' private channels;
- the extra welcome line for a new member's private channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from datetime import timedelta

import discord
from google.genai import types

from . import audience as aud
from .company_brain_bot import MAX_ADMINS_DM
from .handover import (
    collect_digest,
    evidence_channels,
    fallback_handover,
    handover_prompt,
)
from .i18n import CATALOG, t
from .llm import generate
from .orgs import OrgStore
from .pdf_report import find_fonts
from .people import (
    MAX_HANDOVER,
    Dossier,
    PeopleRepo,
    RosterEntry,
    build_key_contacts,
    build_org,
    build_who_to_ask,
    department_label,
    is_bot_name,
    set_role,
)
from .people_listen import PeopleListener
from .people_tools import PeopleCtx, brief_page, people_tools
from .projects import Person, can_read
from .research import markdown_pdf
from .wiki import WikiPage, page_visible, slugify

logger = logging.getLogger(__name__)

ROSTER_TTL_S = 60.0
DECISION_DAYS = 120
STRINGS = {
    "pe_welcome": {
        "th": "ถามสลิดได้ว่าบริษัททำอะไร ใครทำอะไร ควรถามใครเรื่องไหน",
        "en": "Ask me what the company does, who does what, and whom to ask about anything.",
    },
    "pe_welcome_brief": {
        "th": "อยากได้คู่มือต้อนรับของแผนก {dept} ไหมคะ บอกสลิดว่า “ขอคู่มือต้อนรับ” ได้เลย",
        "en": "Want an onboarding brief for {dept}? Just ask me for an “onboarding brief”.",
    },
    "pe_left_dm": {
        "th": "{name} ออกจากบริษัทแล้วค่ะ สลิดเก็บสิ่งที่{name}ทำและวิธีทำงานไว้ให้ และเขียนเอกสารส่งมอบงานแนบมาด้วยค่ะ",
        "en": "{name} has left the company. Salid kept what they did and how they worked, and wrote a handover (attached).",
    },
    "pe_left_tasks": {
        "th": "\nงานที่ยังค้าง {n} งาน: {items}\nบอกสลิดได้เลยว่า “ย้ายงาน … ให้ …” เพื่อมอบหมายใหม่ค่ะ",
        "en": "\n{n} open task(s): {items}\nTell me “reassign … to …” to hand them over.",
    },
    "pe_left_notasks": {"th": "\nไม่มีงานค้างค่ะ", "en": "\nNo open tasks."},
    "pe_handover_page": {"th": "ส่งมอบงาน: {name}", "en": "Handover: {name}"},
}
CATALOG.update(STRINGS)


class PeopleMixin:
    """Needs `self.docs`, `self.llm`, `self.s`, `self.team`, `self.tasks`, `self.wiki`,
    `self.company`, `self.memory_store`, `_people`, `_ranks`, `_members`. Call `_people_init()`
    once after `_coo_init`, `_wiki_init` and `_company_init`."""

    def _people_init(self) -> None:
        self.people = PeopleRepo(self.docs, clock=self.s.now)
        self.people.bot_check = self._people_is_bot
        self.people_listener = PeopleListener(
            self.people,
            self.s,
            self.llm,
            people_of=lambda gid: self._people(self._guild(gid)) if self._guild(gid) else {},
            kind_of=self._channel_kind,
            enabled=lambda gid: self._people_on(gid) and self.team.enabled(gid, "memory"),
            excluded=self._people_excluded,
        )
        self.tasks.on_done = self.people_listener.on_task_done
        self._people_roster_cache: dict[int, tuple[float, list[RosterEntry]]] = {}
        self._people_day: dict[int, str] = {}

    def _people_is_bot(self, uid: int, name: str) -> bool:
        """The assistant itself (id, or its display name in any guild): never a person."""
        me = getattr(self, "user", None)
        if me is not None and getattr(me, "id", None) == uid:
            return True
        mine = {getattr(me, "name", ""), getattr(me, "display_name", "")} - {""}
        for g in list(getattr(self, "guilds", ()) or ()):
            m = getattr(g, "me", None)
            mine |= {getattr(m, "display_name", ""), getattr(m, "name", "")} - {""}
            if m is not None and getattr(m, "id", None) == uid:
                return True
        return is_bot_name(name, mine)

    def _people_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "company_knowledge")

    def _people_excluded(self, gid: int, cid: int) -> bool:
        """Never learn from a member's own private Salid channel."""
        try:
            return self.private_chats.owner_of(gid, cid) is not None
        except Exception:
            return True  # fail closed

    # ------------------------------------------------------------- the roster
    def _org_departments(self, gid: int) -> dict[int, str]:
        """Discord user id -> department from the org's member list (web members linked to a
        Discord account). Blocking; {} without an org."""
        try:
            org = OrgStore(self.docs).org_of_guild(gid)
        except Exception:
            return {}
        out: dict[int, str] = {}
        for m in getattr(org, "members", None) or []:
            uid = str(m.discord_user_id or "")
            if uid.isdigit() and m.department:
                out[int(uid)] = m.department
        return out

    @staticmethod
    def _discord_role(guild, m) -> str:
        role = getattr(m, "top_role", None)
        name = getattr(role, "name", "") or ""
        if not name or name == "@everyone" or role == getattr(guild, "default_role", None):
            return ""
        return name

    def _people_roster(self, guild, fresh: bool = False) -> list[RosterEntry]:
        """Everybody who is here now: department (stated > org > none), title (stated / heard /
        Discord role), rank. Blocking, cached for a minute."""
        hit = self._people_roster_cache.get(guild.id)
        if hit is not None and not fresh and self._clock() - hit[0] < ROSTER_TTL_S:
            return hit[1]
        dossiers = {d.user_id: d for d in self.people.list(guild.id)}
        depts = self._org_departments(guild.id)
        ranks = self._ranks(guild)
        out: list[RosterEntry] = []
        for m in self._members(guild):
            d = dossiers.get(m.id)
            if d is not None and d.status == "left":
                continue
            if self._people_is_bot(m.id, getattr(m, "display_name", "")):
                continue
            out.append(
                RosterEntry(
                    user_id=m.id,
                    name=getattr(m, "display_name", str(m.id)),
                    department=(d.department if d and d.department else depts.get(m.id, "")),
                    title=(d.role_title if d and d.role_title else self._discord_role(guild, m)),
                    rank=ranks.get(m.id, 0),
                )
            )
        self._people_roster_cache[guild.id] = (self._clock(), out)
        return out

    def _public_channel_ids(self, guild) -> set[int]:
        """Channels every human member of the server can read. One channel that raises is skipped
        (treated as not public). With a partial member cache (no chunking) the cached members
        alone prove nothing, so the channel must also be readable by @everyone."""
        everyone = {m.id for m in self._members(guild)}
        if not everyone:
            return set()
        total = getattr(guild, "member_count", None)
        everyone_role = getattr(guild, "default_role", None)
        partial = bool(
            isinstance(total, int) and everyone_role is not None and total > len(everyone)
        )
        out: set[int] = set()
        for c in self._all_channels(guild):
            try:
                if not everyone <= set(self._channel_viewers(guild, c)):
                    continue
                if partial and not self._can_read(c, everyone_role):
                    continue
                out.add(c.id)
            except Exception:
                logger.warning(
                    "people map: channel skipped guild=%s channel=%s", guild.id, getattr(c, "id", "?"),
                    exc_info=True,
                )  # fmt: skip
        return out

    # ------------------------------------------------------------- nightly refresh
    async def _people_due(self) -> None:
        """Once a day per server: roles and project owners into the dossiers, then the company map."""
        today = self.s.now().date().isoformat()
        for guild in list(self.guilds):
            if self._people_day.get(guild.id) == today:
                continue
            try:
                await asyncio.to_thread(self.team.current, guild.id)
                if not self._people_on(guild.id):
                    continue
                self._people_day[guild.id] = today
                logger.info("people refresh: start guild=%s", guild.id)
                await asyncio.to_thread(self._people_refresh, guild)
            except Exception:
                logger.warning("people refresh failed", exc_info=True)

    def _people_refresh(self, guild) -> None:
        """Blocking. Three stages (roles, projects, map), each in its own try: a failure in one
        is logged with its name and never stops the next, so the map is always rebuilt."""
        for stage, fn in (
            ("roles", self._people_stage_roles),
            ("projects", self._people_stage_projects),
            ("map", self.rebuild_company_map),
        ):
            try:
                fn(guild)
            except Exception:
                logger.warning(
                    "people refresh: stage=%s failed guild=%s", stage, guild.id, exc_info=True
                )

    def _people_stage_roles(self, guild) -> None:
        gid = guild.id
        depts = self._org_departments(gid)
        for m in self._members(guild):
            name = getattr(m, "display_name", str(m.id))
            if self._people_is_bot(m.id, name):
                continue
            role = self._discord_role(guild, m)
            dept = depts.get(m.id)

            def apply(d: Dossier, role=role, dept=dept) -> bool:
                a = set_role(d, role, "discord") if role else False
                b = set_role(d, "", "org", dept) if dept else False
                return a or b

            try:
                self.people.update(gid, m.id, name, apply)
            except Exception:
                logger.warning(
                    "people refresh: stage=roles member failed guild=%s", gid, exc_info=True
                )

    def _people_stage_projects(self, guild) -> None:
        if self.team.enabled(guild.id, "projects"):
            self.people_listener.refresh_projects(guild.id, self.project_repo.list(guild.id))

    def rebuild_company_map(self, guild) -> None:
        """Departments, who-to-ask and key contacts into the playbook. Blocking. Each part is
        built in its own try and the playbook is ALWAYS saved (even with an empty org or no public
        channel), with `org_built_at` set; one info line says what came out."""
        gid = guild.id
        pb = self.company.get(gid)
        try:
            pb.org = build_org(self._people_roster(guild, fresh=True))
        except Exception:
            logger.warning("people map: stage=org failed guild=%s", gid, exc_info=True)
        try:
            dossiers = self.people.list(gid)
        except Exception:
            logger.warning("people map: stage=dossiers failed guild=%s", gid, exc_info=True)
            dossiers = []
        try:
            public = self._public_channel_ids(guild)
        except Exception:
            logger.warning("people map: stage=public failed guild=%s", gid, exc_info=True)
            public = set()
        try:
            pb.who_to_ask = build_who_to_ask(dossiers, public)
        except Exception:
            logger.warning("people map: stage=who_to_ask failed guild=%s", gid, exc_info=True)
        try:
            pb.key_contacts = build_key_contacts(dossiers, public)
        except Exception:
            logger.warning("people map: stage=contacts failed guild=%s", gid, exc_info=True)
        pb.org_built_at = self.s.now().isoformat(timespec="seconds")
        self.company.save(gid, pb)
        members = self._members(guild)
        logger.info(
            "people map: guild=%s people=%d topics=%d contacts=%d departments=%d "
            "public_channels=%d members=%d/%s",
            gid, len(dossiers), len(pb.who_to_ask), len(pb.key_contacts), len(pb.org),
            len(public), len(members), getattr(guild, "member_count", "?"),
        )  # fmt: skip

    # ------------------------------------------------------------- agent tools
    def _people_tools(self, message, guild, asker, is_exec: bool, memory, lang: str) -> list:
        try:
            gid = guild.id
            if not self._people_on(gid):
                return []
            loop = asyncio.get_running_loop()
            allowed = memory.allowed if memory is not None else {}
            vis = getattr(memory, "vis", None)
            alone = bool(vis is not None and vis.scope.dm)
            ctx = PeopleCtx(
                guild_id=gid,
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                lang=lang,
                repo=self.people,
                company=self.company,
                now=self.s.now,
                guild_name=getattr(guild, "name", "") or "",
                readable=lambda: frozenset(allowed),
                see_all=lambda: bool(is_exec and alone),
                channel_name=lambda cid: self._channel_label(guild, cid),
                roster=lambda: self._people_roster(guild),
                channels=lambda: [f"#{n}" for n in list(allowed.values())[:12]],
                wiki_pages=lambda: self._people_wiki_pages(gid, memory),
                projects=lambda: self._people_projects(guild, asker, is_exec),
                decisions=lambda topic: self._people_decisions(gid, memory, topic),
                resolve_member=lambda name: self._people_match(guild, name),
                gen_handover=lambda uid, name, succ: self._people_handover_text(
                    gid, uid, name, succ
                ),
                on_left=lambda uid, name, succ: self._people_on_left(guild, loop, uid, name, succ),
                pdf=lambda title, body: markdown_pdf(
                    org=getattr(guild, "name", "") or "Salid",
                    title=title,
                    body=body,
                    lang=lang,
                    generated_at=self.s.now(),
                    fonts=find_fonts(self.s.pdf_font, lang),
                ),
                save_wiki=lambda title, body, chans: self._people_save_page(
                    gid, asker, title, body, chans
                ),
            )
            return people_tools(ctx)
        except Exception:
            logger.warning("company knowledge tools unavailable", exc_info=True)
            return []

    def _people_match(self, guild, name: str) -> list[tuple[int, str]]:
        low = (name or "").strip().casefold().lstrip("@")
        if not low:
            return []
        rows = [(m.id, getattr(m, "display_name", str(m.id))) for m in self._members(guild)]
        exact = [r for r in rows if r[1].casefold() == low]
        if exact:
            return exact
        return [r for r in rows if len(low) >= 2 and low in r[1].casefold()]

    def _people_wiki_pages(self, gid: int, memory) -> list[WikiPage]:
        if memory is None or not self.team.enabled(gid, "wiki"):
            return []
        vis = memory.vis
        authors = vis.scope.authors if vis else frozenset()
        readable = frozenset(memory.allowed)
        return [p for p in self.wiki.list(gid) if page_visible(p, readable, authors)]

    def _people_projects(self, guild, asker, is_exec: bool) -> list[dict]:
        if not self.team.enabled(guild.id, "projects"):
            return []
        me = Person(discord_user_id=str(asker.id), name=getattr(asker, "display_name", ""))
        out = []
        for p in self.project_repo.list(guild.id):
            if p.status in ("completed", "cancelled") or not can_read(p, me, is_exec):
                continue
            out.append({"name": p.name, "status": p.status, "department": p.department})
        return out

    def _people_decisions(self, gid: int, memory, topic: str) -> list[dict]:
        if memory is None:
            return []
        from .people import topic_match

        end = self.s.now().date()
        start = end - timedelta(days=DECISION_DAYS)
        try:
            rows = memory.store.summaries(
                gid, set(memory.allowed), start.isoformat(), end.isoformat()
            )
        except Exception:
            return []
        out = []
        for s in sorted(rows, key=lambda r: r.local_date, reverse=True):
            for x in (s.summary or {}).get("decisions") or []:
                if topic_match(topic, str(x)) >= 1:
                    out.append(
                        {
                            "text": str(x),
                            "day": s.local_date,
                            "channel_id": s.channel_id,
                            "kind": "meeting",
                        }
                    )
        return out[:5]

    def _people_save_page(self, gid: int, asker, title: str, body: str, chans: list[int]) -> dict:
        if not self.team.enabled(gid, "wiki"):
            return {"error": "the wiki is off for this server"}
        page = brief_page(title, body, chans, asker.id, getattr(asker, "display_name", ""))
        saved = self.wiki.save(gid, page)
        return {"status": "saved", "title": saved.title, "slug": saved.slug}

    # ------------------------------------------------------------- welcome
    def _people_welcome(self, guild, member, langs: list[str]) -> str:
        """The extra line(s) of a new member's private-channel welcome. Blocking. "" when off."""
        try:
            if not self._people_on(guild.id):
                return ""
            text = aud.tt("pe_welcome", langs)
            d = self.people.get(guild.id, member.id)
            dept = (d.department if d else "") or self._org_departments(guild.id).get(member.id, "")
            if dept:
                text += "\n" + aud.tt("pe_welcome_brief", langs, dept=department_label(dept))
            return text
        except Exception:
            logger.warning("people welcome failed", exc_info=True)
            return ""

    # ------------------------------------------------------------- leaving
    async def on_member_remove(self, member) -> None:
        await super().on_member_remove(member)
        if getattr(member, "bot", False):
            return
        try:
            guild = member.guild
            await asyncio.to_thread(self.team.current, guild.id)
            if not self._people_on(guild.id):
                return
            name = getattr(member, "display_name", str(member.id))
            res = await asyncio.to_thread(
                self._people_mark_left, guild.id, member.id, name, 0, True
            )
            if res is None or res["already"]:
                return
            await self._people_after_left(guild, member.id, name, res["tasks"])
        except Exception:
            logger.warning("people: leaving flow failed", exc_info=True)

    def _people_mark_left(
        self, gid: int, uid: int, name: str, successor: int, only_known: bool = False
    ) -> dict | None:
        """Dossier -> left; the open tasks. None when `only_known` and there is nothing to hand
        over. Blocking."""
        tasks = self.tasks.list(gid, assignee_id=uid)
        existing = self.people.get(gid, uid)
        if only_known and existing is None and not tasks:
            return None
        before = existing.status if existing else "active"
        now = self.s.now().isoformat(timespec="seconds")

        def mark(d: Dossier) -> bool:
            if d.status != "left":
                d.status, d.left_at = "left", now
            if successor:
                d.successor_id = successor
            return True

        self.people.update(gid, uid, name, mark)
        self._people_roster_cache.pop(gid, None)
        return {
            "already": before == "left",
            "tasks": [{"id": x.id, "title": x.title, "due": x.due} for x in tasks[:25]],
        }

    def _people_on_left(self, guild, loop, uid: int, name: str, successor: int) -> dict:
        """The `person_left` tool (agent worker thread)."""
        res = self._people_mark_left(guild.id, uid, name, successor)
        loop.call_soon_threadsafe(
            lambda: self._spawn(self._people_after_left(guild, uid, name, res["tasks"]))
        )
        return {
            "status": "recorded",
            "name": name,
            "open_tasks": res["tasks"],
            "reassign_hint": "tell the executive they can say 'reassign <task> to <person>'",
            "handover": "being written; the executives get it as a PDF in their private channels",
        }

    async def _people_after_left(self, guild, uid: int, name: str, tasks: list[dict]) -> None:
        """Handover document, wiki page, PDF to the executives."""
        gid = guild.id
        done = await asyncio.to_thread(self._people_handover_text, gid, uid, name, 0)
        text = done.get("text", "")
        lang = self._guild_lang(gid)
        files = []
        if text:
            with contextlib.suppress(Exception):
                pdf = await asyncio.to_thread(
                    markdown_pdf,
                    org=getattr(guild, "name", "") or "Salid",
                    title=t("pe_handover_page", lang, name=name),
                    body=text,
                    lang=lang,
                    generated_at=self.s.now(),
                    fonts=find_fonts(self.s.pdf_font, lang),
                )
                files.append((f"handover-{slugify(name)[:40]}.pdf", pdf))
        for admin in self._company_admins(guild, uid)[:MAX_ADMINS_DM]:
            langs = self._deliver_langs(gid, admin.id)
            msg = aud.tt("pe_left_dm", langs, name=name)
            if tasks:
                items = "; ".join(
                    x["title"] + (f" ({x['due']})" if x.get("due") else "") for x in tasks[:8]
                )
                msg += aud.tt("pe_left_tasks", langs, n=len(tasks), items=items)
            else:
                msg += aud.tt("pe_left_notasks", langs)
            attach = [discord.File(io.BytesIO(data), filename=fn) for fn, data in files]
            with contextlib.suppress(Exception):
                await self.deliver_to_person(guild, admin, msg[:1900], files=attach)

    def _people_handover_text(self, gid: int, uid: int, name: str, successor: int) -> dict:
        """Write (model, else straight from the dossier), keep in the dossier and as the wiki page
        "Handover: <name>". Blocking."""
        d = self.people.get(gid, uid) or Dossier(user_id=uid, name=name)
        lang = self._guild_lang(gid)
        today = self.s.now().date()
        tasks = [
            x.title + (f" (due {x.due})" if x.due else "")
            for x in self.tasks.list(gid, assignee_id=uid)[:25]
        ]
        chans = evidence_channels(d)
        names = {c: self._channel_name(gid, c) for c in chans}
        digest = collect_digest(self.memory_store, gid, chans, d.name or name, today, names=names)
        text = ""
        if self.llm is not None:
            try:
                resp = generate(
                    self.llm,
                    self.s.gemini_model,
                    handover_prompt(d, tasks, digest, lang, (d.left_at or today.isoformat())[:10]),
                    types.GenerateContentConfig(temperature=0.2, max_output_tokens=4096),
                )
                text = (resp.text or "").strip()
            except Exception:
                logger.warning("handover writing failed", exc_info=True)
        if len(text) < 40:
            text = fallback_handover(d, tasks, lang)
        text = text[:MAX_HANDOVER]
        now = self.s.now().isoformat(timespec="seconds")

        def keep(x: Dossier) -> bool:
            x.handover, x.handover_at = text, now
            if successor:
                x.successor_id = successor
            return True

        self.people.update(gid, uid, name, keep)
        out = {"text": text}
        if self.team.enabled(gid, "wiki"):
            try:
                page = WikiPage(
                    slug=slugify(f"handover {name}"),
                    title=t("pe_handover_page", lang, name=name),
                    body=text,
                    channel_ids=chans,
                    channel_authors={str(c): [uid] for c in chans},
                    author_name="Salid",
                )
                out["wiki_slug"] = self.wiki.save(gid, page).slug
            except Exception:
                logger.warning("handover wiki page failed", exc_info=True)
        return out
