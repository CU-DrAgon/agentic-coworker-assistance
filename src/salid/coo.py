"""The coordinator side of the Discord bot (a mixin of `SecretaryBot`).

* role hierarchy adapter: member ranks and the `Visibility` handed to the memory tools;
* disclosure requests: DM with persistent Approve / Decline buttons, grants, answer delivery;
* personal DMs: a member's DM to Salid is recorded as a private-scope "dm" note;
* daily task follow-up DMs and the evening report (private channel, `/รายงาน`);
* the agent's coordinator tools (`salid.coo_tools`).

The pure rules live in `memory_access`, `disclosure`, `tasks` and `reports`; this module only
adapts them to Discord objects, schedules them and talks to people. Blocking work runs in threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import discord
from discord import app_commands

from . import reports
from .coo_store import STATE, make_doc_store
from .coo_tools import CooTools, build_coo_tools
from .disclosure import Disclosure, DisclosureRepo, pick_owner
from .i18n import resolve_lang, t
from .memory import build_message, message_text
from .memory_access import (
    Scope,
    Visibility,
    audience_authors,
    dm_scope,
    hierarchy_authors,
    member_rank,
)
from .task_handshake import pending_marker
from .tasks import TaskRepo, TaskTracker, due_for_followup
from .tone import tone_block

logger = logging.getLogger(__name__)

COO_CHANNEL: ContextVar[int] = ContextVar("coo_channel", default=0)
REPORT_CHANNEL_NAME = "รายงานจากสลิด"
REPORT_CHANNEL_NAME_EN = "reports-from-salid"  # in guilds whose default language is English
COO_TICK_S = 30.0
FOLLOWUP_WINDOW = timedelta(hours=3)  # sent when due and the bot is up within this time
REPORT_WINDOW = timedelta(hours=4)
PRIVATE_FOLLOWS = timedelta(minutes=10)  # a bare "keep it private" marks the notes just before
RANKS_TTL_S = 60.0
_PRIVATE = re.compile(
    r"เรื่องนี้เก็บเป็นความลับ|(?:เก็บ)?เป็นความลับ|ความลับนะ|(?:อย่า|ห้าม|ไม่ต้อง)บอกใคร"
    r"|\b(?:keep (?:this|it) (?:private|secret|confidential)|off the record"
    r"|don'?t tell (?:anyone|anybody)|confidential|private|secret)\b",
    re.I,
)
_FILLER = re.compile(r"เรื่องนี้|อันนี้|ข้อนี้|นะ|ครับ|ค่ะ|คะ|ด้วย|please|this|it|pls|[\s.!,?:;\-]+", re.I)


def parse_private_marker(text: str) -> tuple[bool, bool]:
    """(marked, only_marker): the text asks to keep it confidential; and it says nothing else."""
    s = text or ""
    if not _PRIVATE.search(s):
        return False, False
    rest = _FILLER.sub("", _PRIVATE.sub("", s))
    return True, not rest.strip()


def plan_report_overwrites(guild, me, exec_role_names: set[str], exec_user_ids=()) -> dict:
    """Permission overwrites of the reports channel: hidden from @everyone; visible to the guild
    owner, Salid, roles with Administrator / Manage Server, roles named in the executive list and
    executive users listed in the environment."""
    seen = discord.PermissionOverwrite(
        view_channel=True, read_message_history=True, send_messages=True
    )
    out = {guild.default_role: discord.PermissionOverwrite(view_channel=False)}
    for role in getattr(guild, "roles", ()):
        perms = getattr(role, "permissions", None)
        if role == guild.default_role or getattr(role, "managed", False):
            continue
        if (
            getattr(perms, "administrator", False)
            or getattr(perms, "manage_guild", False)
            or role.name in exec_role_names
        ):
            out[role] = seen
    owner = guild.get_member(getattr(guild, "owner_id", 0))
    for m in [owner, *(guild.get_member(u) for u in exec_user_ids)]:
        if m is not None:
            out[m] = seen
    out[me] = discord.PermissionOverwrite(
        view_channel=True,
        read_message_history=True,
        send_messages=True,
        embed_links=True,
        attach_files=True,
    )
    return out


class DisclosureButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:disc:(?P<act>ok|no):(?P<id>[0-9a-f]{16})",
):
    """Approve / Decline on a disclosure DM. The custom id carries the request id and the bot
    registers the class once, so the buttons keep working after a restart."""

    def __init__(self, act: str, request_id: str, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=discord.ButtonStyle.success if act == "ok" else discord.ButtonStyle.danger,
                custom_id=f"salid:disc:{act}:{request_id}",
            )
        )
        self.act, self.request_id = act, request_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], match["id"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_disclosure(interaction, self.act, self.request_id)


class CooMixin:
    # ------------------------------------------------------------- setup
    def _coo_init(self) -> None:
        self.docs = make_doc_store(self.s, self.store)
        self.tasks = TaskRepo(self.docs)
        self.disclosures = DisclosureRepo(self.docs)
        self.task_tracker = TaskTracker(
            self.tasks,
            self.s,
            self.llm,
            people_of=lambda gid: self._people(self._guild(gid)) if self._guild(gid) else {},
            kind_of=self._channel_kind,
            enabled=lambda gid: (
                self.team.enabled(gid, "memory")
                and (self.team.enabled(gid, "followups") or self.team.enabled(gid, "reports"))
            ),
            role_members_of=self._role_members,
        )
        self.memory.scheduler.on_summary = self._on_summary_all
        self._ranks_cache: dict[int, tuple[float, dict[int, int]]] = {}
        self._coo_last = float("-inf")

    def _on_summary_all(self, guild_id, channel_id, day, summary, messages) -> None:
        """The memory scheduler's hook (worker thread): tasks first, then projects heard in chat.
        Each part is guarded so one failing never blocks the other."""
        try:
            created = self.task_tracker.on_summary(guild_id, channel_id, day, summary, messages)
            self._handshake_new_tasks(guild_id, created or [])
        except Exception:
            logger.warning("task tracking failed", exc_info=True)
        listener = getattr(self, "project_listener", None)
        if listener is not None:
            try:
                listener.on_summary(guild_id, channel_id, day, summary, messages)
            except Exception:
                logger.warning("project listening failed", exc_info=True)

    def _role_members(self, guild_id: int, text: str) -> list[int] | None:
        """Human member ids of the guild role a text names ("ทีม tech", "@Tech", "<@&id>");
        None when no role matches."""
        guild = self._guild(guild_id)
        if guild is None:
            return None
        raw = (text or "").strip()
        m = re.fullmatch(r"<@&(\d+)>", raw)
        roles = list(getattr(guild, "roles", ()))
        if m:
            role = next((r for r in roles if r.id == int(m[1])), None)
        else:
            name = re.sub(r"^(?:ทีม|team|แผนก)\s*|\s*team$", "", raw.lstrip("@"), flags=re.I)
            name = name.strip().casefold()
            hits = [r for r in roles if name and r.name.casefold() == name]
            role = hits[0] if len(hits) == 1 else None
        if role is None or role == getattr(guild, "default_role", None):
            return None
        return [x.id for x in self._members(guild) if role in getattr(x, "roles", ())]

    def _guild(self, guild_id: int):
        return next((g for g in self.guilds if g.id == guild_id), None)

    def _channel_kind(self, guild_id: int, channel_id: int) -> str:
        guild = self._guild(guild_id)
        ch = guild.get_channel(channel_id) if guild else None
        return (
            "meeting" if isinstance(ch, discord.VoiceChannel | discord.StageChannel) else "summary"
        )

    # ------------------------------------------------------------- hierarchy adapter
    @staticmethod
    def _members(guild) -> list:
        members = guild.members.values() if isinstance(guild.members, dict) else guild.members
        return [m for m in members if not getattr(m, "bot", False)]

    def _rank_of(self, guild, m) -> int:
        perms = getattr(m, "guild_permissions", None)
        return member_rank(
            is_owner=getattr(guild, "owner_id", None) == m.id,
            is_admin=bool(getattr(perms, "administrator", False)),
            top_role_position=getattr(getattr(m, "top_role", None), "position", 0) or 0,
        )

    def _ranks(self, guild) -> dict[int, int]:
        """Member id -> rank of every human member, cached for a minute."""
        hit = self._ranks_cache.get(guild.id)
        if hit is not None and self._clock() - hit[0] < RANKS_TTL_S:
            return hit[1]
        ranks = {m.id: self._rank_of(guild, m) for m in self._members(guild)}
        self._ranks_cache[guild.id] = (self._clock(), ranks)
        return ranks

    def _people(self, guild) -> dict[int, str]:
        return {m.id: getattr(m, "display_name", str(m.id)) for m in self._members(guild)}

    def _all_channels(self, guild) -> list:
        return [
            *guild.text_channels,
            *getattr(guild, "voice_channels", ()),
            *getattr(guild, "stage_channels", ()),
        ]

    def _coo_scope(
        self,
        guild,
        member,
        answering: int | None,
        allowed: dict[int, str],
        room: frozenset[int] | None = None,
    ):
        """(tool channel map, Visibility) for one asker answering in channel `answering` (None =
        a DM / private reply). `room`: the humans in a voice room (the audience; `member` is the
        lowest ranked of them): the answer reaches all of them, and disclosure grants of one
        person are not used when others listen. Blocking."""
        ranks = self._ranks(guild)
        audience = None
        unknown = False
        if room is not None:
            audience = frozenset(room)
        elif answering is not None:
            where = guild.get_channel(answering)
            unknown = where is None
            audience = frozenset() if unknown else self._channel_viewers(guild, where, fresh=True)
        authors = frozenset() if unknown else hierarchy_authors(member.id, ranks, audience)
        scope = Scope(
            asker_id=member.id,
            channels=frozenset(allowed),
            authors=authors,
            dm_authors=hierarchy_authors(member.id, ranks, None),
            unlocked=(
                frozenset()
                if room is not None and len(room) > 1
                else self.disclosures.unlocked_ids(guild.id, member.id)
            ),
            dm=answering is None and room is None,
        )
        people = self._people(guild)
        read = {
            c.id: c.name
            for c in self._all_channels(guild)
            if self.team.channel_mode(guild.id, c.id, c.name, self.s.ask_channel_ids) != "off"
        }
        read.update({dm_scope(uid): "DM" for uid in ranks})
        dm_names = {dm_scope(a): f"DM {people.get(a, a)}" for a in authors}  # their summaries too
        return allowed, Visibility(scope, read, people, dm_names)

    # ------------------------------------------------------------- agent tools
    def _coo_tools_in(self, channel_id: int, *args) -> list:
        """`_coo_tools` for a request made in `channel_id` (tasks made by add_task remember it)."""
        token = COO_CHANNEL.set(channel_id)
        try:
            return self._coo_tools(*args)
        finally:
            COO_CHANNEL.reset(token)

    def _coo_tools(self, guild, member, is_exec: bool, memory, question: str, lang: str) -> list:
        loop = asyncio.get_running_loop()
        vis = memory.vis if memory is not None else None
        c = CooTools(
            guild_id=guild.id,
            asker_id=member.id,
            is_exec=is_exec,
            scope=vis.scope if vis else None,
            tasks=self.tasks,
            people=self._people(guild),
            today=lambda: self.s.now().date(),
            memory=memory,
            request_disclosure=partial(
                self._request_disclosure, guild, member, question, memory, loop
            ),
            my_day=partial(self._my_day, guild, member),
            reassigned=partial(self._handshake_reassigned, guild.id),
            created=partial(self._handshake_new_tasks, guild.id),
            owner_answered=partial(self._owner_tools_done, guild.id),
            find_project=partial(self._task_project, guild, member, is_exec),
            channel_id=COO_CHANNEL.get(),
        )
        return build_coo_tools(c)

    def _task_project(self, guild, member, is_exec: bool, fragment: str) -> tuple[str, str]:
        """(project id, "") for a project named in words, or ("", why not). The asker must be
        allowed to add work to it."""
        from . import projects as P

        try:
            found = P.find_projects(self.project_repo.list(guild.id), fragment)
        except Exception:
            return "", "projects are not available here"
        if not found:
            return "", f"no project matches {fragment!r}"
        if len(found) > 1:
            return "", "several projects match: " + "; ".join(p.name for p in found[:5])
        who = P.Person(discord_user_id=str(member.id), name=getattr(member, "display_name", ""))
        if not P.can_contribute(found[0], who, is_exec):
            return "", "not allowed to add tasks to that project"
        return found[0].id, ""

    def _memory_day_retained(self, guild_id: int, day: date) -> bool:
        """Whether a memory day is inside the guild's current retention window."""
        days = self.team.fresh_memory_retention_days(guild_id, fail_closed=True)
        return day >= self.s.now().date() - timedelta(days=days)

    def _my_day(self, guild, member, day: date) -> dict:
        channels = {
            ch.id: ch.name
            for ch in self._all_channels(guild)
            if self.team.channel_mode(guild.id, ch.id, ch.name, self.s.ask_channel_ids) != "off"
        }
        return reports.person_day(
            guild_id=guild.id,
            user_id=member.id,
            day=day,
            zone=self.s.zone,
            memory=self.memory_store,
            tasks=self.tasks,
            channels=channels,
            include_memory=self._memory_day_retained(guild.id, day),
        )

    # ------------------------------------------------------------- disclosure
    def _request_disclosure(self, guild, member, question, memory, loop, topic: str) -> dict:
        """Tool body (runs in the agent's worker thread): record a request for the owner of the
        withheld records and DM them. Never returns the content."""
        hits = list(memory.hidden)
        owner = pick_owner(hits, self._ranks(guild))
        if not owner or owner == member.id:
            return {"error": "no owner could be determined for the withheld information"}
        owner_lang = self._person_lang(guild.id, owner)
        doc, created = self.disclosures.request(
            guild_id=guild.id,
            owner_id=owner,
            grantee_id=member.id,
            topic=topic,
            question=question,
            hits=hits,
            names=memory.vis.read_channels if memory.vis else {},
            lang=owner_lang,
        )
        if not created:
            return {"status": "already requested; the owner has not answered yet"}
        try:
            ok = asyncio.run_coroutine_threadsafe(
                self._send_disclosure_dm(guild, doc, getattr(member, "display_name", "")), loop
            ).result(timeout=30)
        except Exception:
            logger.warning("disclosure DM failed", exc_info=True)
            ok = False
        if not ok:
            self.disclosures.decline(doc.id)
            return {"error": "could not message the owner; suggest asking them directly"}
        return {"status": "request sent to the owner of the information; the answer follows by DM"}

    async def _send_disclosure_dm(self, guild, doc: Disclosure, asker_name: str) -> bool:
        owner = guild.get_member(doc.owner_id)
        if owner is None:
            return False
        view = discord.ui.View(timeout=None)
        view.add_item(DisclosureButton("ok", doc.id, t("disc_btn_ok", doc.lang)))
        view.add_item(DisclosureButton("no", doc.id, t("disc_btn_no", doc.lang)))
        text = t(
            "disc_ask",
            doc.lang,
            asker=asker_name or str(doc.grantee_id),
            topic=doc.topic,
            question=doc.question or doc.topic,
            description=doc.description,
        )
        res = await self.deliver_to_person(guild, owner, text, view=view)
        return res.ok

    async def handle_disclosure(self, interaction, act: str, request_id: str) -> None:
        """Approve / Decline click. Only the owner counts; a decided request is not decided twice."""
        doc = await asyncio.to_thread(self.disclosures.get, request_id)
        lang = doc.lang if doc else "en"
        if doc is None or doc.status != "pending":
            await interaction.response.send_message(t("disc_handled", lang), ephemeral=True)
            return
        if interaction.user.id != doc.owner_id:
            await interaction.response.send_message(t("disc_not_yours", lang), ephemeral=True)
            return
        guild = self._guild(doc.guild_id)
        asker = guild.get_member(doc.grantee_id) if guild else None
        name = getattr(asker, "display_name", str(doc.grantee_id))
        decided = await asyncio.to_thread(
            self.disclosures.approve if act == "ok" else self.disclosures.decline, request_id
        )
        if decided is None:
            await interaction.response.send_message(t("disc_handled", lang), ephemeral=True)
            return
        key = "disc_owner_approved" if act == "ok" else "disc_owner_declined"
        original = getattr(getattr(interaction, "message", None), "content", "") or ""
        await interaction.response.edit_message(
            content=f"{original}\n\n{t(key, lang, asker=name)}".strip(), view=None
        )
        if asker is None:
            return
        try:
            if act == "ok":
                await self._deliver_disclosure(guild, asker, decided)
            else:
                await self.deliver_to_person(
                    guild,
                    asker,
                    t(
                        "disc_asker_declined",
                        self._person_lang(doc.guild_id, doc.grantee_id, doc.question),
                        topic=doc.topic,
                    ),
                )
        except Exception:
            logger.warning("could not deliver disclosure answer", exc_info=True)

    async def _deliver_disclosure(self, guild, asker, doc: Disclosure) -> None:
        """Answer the asker by DM with exactly the approved records unlocked."""
        from .qa import ask  # late: qa imports a lot

        lang = self._person_lang(guild.id, asker.id, doc.question or doc.topic)
        await asyncio.to_thread(self.team.current, guild.id)
        memory = await self._memory_tools(SimpleNamespace(guild=None, channel=None), guild, asker)
        q = (
            f"{doc.question or doc.topic}\n\n[The owner of the information has just approved "
            f"sharing what they wrote about “{doc.topic}” with this person. Look it up with "
            "chat_messages / chat_search (person filter) and answer.]"
        )
        try:
            ans = await asyncio.to_thread(
                ask,
                self.s,
                self.store,
                q,
                guild_id=guild.id,
                is_exec=self.is_exec_member(asker),
                user_id=asker.id,
                language=lang,
                memory=memory,
            )
            text = t("disc_asker_approved", lang, topic=doc.topic) + ans.text
        except Exception:
            logger.warning("disclosure answer failed", exc_info=True)
            text = t("disc_asker_failed", lang)
        for i in range(0, len(text), 1900):
            await self.deliver_to_person(guild, asker, text[i : i + 1900])

    # ------------------------------------------------------------- personal DMs
    async def _coo_dm_record(self, message, guild, member) -> tuple[bool, str]:
        """Record a member's DM to Salid as a private-scope note. Returns (handled, note): handled
        True when the message was only "keep it private" (answered here); `note` is a hint for the
        agent when a note was marked confidential."""
        gid = guild.id
        if not self.team.enabled(gid, "memory"):
            return False, ""
        text = message_text(message.content, getattr(message, "attachments", ()))
        if not text:
            return False, ""
        marked, only = parse_private_marker(text)
        lang = self._person_lang(gid, member.id, text)
        uid = member.id
        if only:
            at = getattr(message, "created_at", None) or datetime.now(UTC)
            self._memory_enqueue(partial(self._mark_recent_private, gid, uid, at))
            await message.reply(t("dm_private_marked", lang), mention_author=False)
            return True, ""
        rec = build_message(
            self.s,
            guild_id=gid,
            channel_id=dm_scope(uid),
            message_id=message.id,
            author_id=uid,
            author_name=getattr(member, "display_name", str(uid)),
            created_at=getattr(message, "created_at", None) or datetime.now(UTC),
            text=text,
            kind="dm",
        )
        if marked:
            rec = rec.model_copy(update={"visibility": "private"})
        self._memory_enqueue(partial(self.memory.record, rec))
        return False, t("dm_private_note", lang) if marked else ""

    def _mark_recent_private(self, gid: int, uid: int, at: datetime) -> None:
        end = at.astimezone(self.s.zone).date()
        rows = self.memory_store.messages(
            gid, {dm_scope(uid)}, (end - timedelta(days=1)).isoformat(), end.isoformat(),
            limit=5, newest_first=True,
        )  # fmt: skip
        stamp = at if at.tzinfo else at.replace(tzinfo=UTC)
        for m in rows:
            if m.author_id == uid and stamp - m.created_at <= PRIVATE_FOLLOWS:
                self.memory_store.set_visibility(gid, m.message_id, "private")

    async def _coo_dm_notice(self, message, guild) -> None:
        """Once per member: say that DM work info is noted and how to keep something private."""
        if not self.team.enabled(guild.id, "memory"):
            return
        key = f"dm-notice:{message.author.id}"
        first = await asyncio.to_thread(self._claim, guild.id, key)
        if first:
            with contextlib.suppress(discord.HTTPException):
                # in the member's private channel in a server it is not a DM: say so
                notice = "dm_notice" if message.guild is None else "dm_notice_room"
                await message.channel.send(
                    t(notice, self._person_lang(guild.id, message.author.id, message.content or ""))
                )

    def _claim(self, guild_id: int, key: str) -> bool:
        """True the first time `key` is claimed for the guild (survives restarts)."""
        if self.docs.get(STATE, guild_id, key) is not None:
            return False
        self.docs.put(STATE, guild_id, key, {"at": datetime.now(UTC).isoformat(timespec="seconds")})
        return True

    # ------------------------------------------------------------- timers
    async def _coo_due(self) -> None:
        """Follow-ups (only when briefings are off), the evening report, the personal briefs and the
        pending-ask reminders, checked every half minute."""
        if self._clock() - self._coo_last < COO_TICK_S:
            return
        self._coo_last = self._clock()
        now = self.s.now()
        await self._followups_due(now)
        await self._handshake_tick(now)
        await self._reports_due(now)
        await self._briefings_due(now)
        await self._asks_due(now)

    @staticmethod
    def _in_window(now: datetime, hhmm: str, span: timedelta) -> bool:
        h, m = (int(x) for x in hhmm.split(":"))
        start = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return start <= now < start + span

    async def _followups_due(self, now: datetime) -> int:
        """DM every member with tasks due today or overdue, once a day per guild. Returns DMs."""
        sent = 0
        for guild in list(self.guilds):
            gid = guild.id
            try:
                await asyncio.to_thread(self.team.current, gid)
                if not (self.team.enabled(gid, "followups") and self.team.enabled(gid, "memory")):
                    continue
                if self.team.enabled(gid, "briefings"):
                    continue  # the morning brief carries the due tasks: never both
                if not self._in_window(now, self.team.followup_time(gid), FOLLOWUP_WINDOW):
                    continue
                day = now.date()
                if not await asyncio.to_thread(self._claim, gid, f"followup:{day}"):
                    continue
                sent += await self._send_followups(guild, day)
            except Exception:
                logger.exception("follow-ups for guild %s failed", gid)
        return sent

    async def _send_followups(self, guild, day: date) -> int:
        tasks = await asyncio.to_thread(self.tasks.list, guild.id, status="open")
        n = 0
        for uid, items in due_for_followup(tasks, day).items():
            member = guild.get_member(uid)
            if member is None or getattr(member, "bot", False):
                continue
            lang = self._person_lang(guild.id, uid)
            lines = [t("fu_header", lang, name=getattr(member, "display_name", ""))]
            lines += [
                t("fu_line", lang, id=x.id, title=x.title, due=x.due)
                + (f" {mark}" if (mark := pending_marker(x, lang)) else "")
                for x in items[:10]
            ]
            lines.append(t("fu_footer", lang, id=items[0].id))
            if (await self.deliver_to_person(guild, member, "\n".join(lines))).ok:
                n += 1
        logger.info("follow-ups: guild=%s dms=%d", guild.id, n)
        return n

    async def _reports_due(self, now: datetime) -> int:
        """Post the evening report once a day per guild. Returns reports posted."""
        posted = 0
        for guild in list(self.guilds):
            gid = guild.id
            try:
                await asyncio.to_thread(self.team.current, gid)
                if not (self.team.enabled(gid, "reports") and self.team.enabled(gid, "memory")):
                    continue
                if not self._in_window(now, self.team.report_time(gid), REPORT_WINDOW):
                    continue
                if not await asyncio.to_thread(self._claim, gid, f"report:{now.date()}"):
                    continue
                if await self._post_report(guild, now.date()) is not None:
                    posted += 1
            except Exception:
                logger.exception("report for guild %s failed", gid)
        return posted

    # ------------------------------------------------------------- reports channel
    def _find_report_channel(self, guild):
        cid = self.team.report_channel_id(guild.id)
        if cid:
            return guild.get_channel(cid)
        names = (REPORT_CHANNEL_NAME, REPORT_CHANNEL_NAME_EN)
        return next((c for c in guild.text_channels if c.name in names), None)

    async def _report_channel(self, guild, *, create: bool = True):
        """The private reports channel: the configured/existing one, else created (needs Manage
        Channels), else an admin is asked once a day. None when there is none."""
        found = self._find_report_channel(guild)
        if found is not None or not create or self.team.report_channel_id(guild.id):
            return found
        me = getattr(guild, "me", None)
        perms = getattr(me, "guild_permissions", None)
        lang = resolve_lang(self.team.current(guild.id).language_setting)
        report_name = REPORT_CHANNEL_NAME_EN if lang != "th" else REPORT_CHANNEL_NAME
        if getattr(perms, "manage_channels", False) or getattr(perms, "administrator", False):
            names = self.team.exec_role_names(guild.id, self.s.exec_role_names)
            try:
                channel = await guild.create_text_channel(
                    report_name,
                    overwrites=plan_report_overwrites(guild, me, names, self.s.exec_user_ids),
                    topic=t("report_channel_topic", lang),
                    reason="Salid daily reports",
                )
                self._invalidate_viewers(guild.id, channel.id)
                return channel
            except discord.HTTPException:
                logger.warning("could not create the reports channel", exc_info=True)
        if await asyncio.to_thread(self._claim, guild.id, f"report-ask:{self.s.now().date()}"):
            owner = guild.get_member(getattr(guild, "owner_id", 0))
            if owner is not None:
                with contextlib.suppress(discord.HTTPException):
                    await owner.send(t("report_need_channel", lang, name=report_name))
        return None

    def _report_text(self, guild, audience: set[int], day: date, lang: str) -> str:
        """Build the report for readers `audience` (member ids). Blocking."""
        ranks = self._ranks(guild)
        readable, names = set(), {}
        for c in self._all_channels(guild):
            if self.team.channel_mode(guild.id, c.id, c.name, self.s.ask_channel_ids) == "off":
                continue
            names[c.id] = c.name
            if audience and audience <= self._channel_viewers(guild, c):
                readable.add(c.id)
        scope = Scope(
            asker_id=0,
            channels=frozenset(readable),
            authors=audience_authors(ranks, audience),
            dm=True,
        )
        voice = {
            c.id
            for c in [*getattr(guild, "voice_channels", ()), *getattr(guild, "stage_channels", ())]
        }
        inp = reports.collect_company(
            guild_id=guild.id,
            day=day,
            zone=self.s.zone,
            ledger=self.store,
            memory=self.memory_store,
            tasks=self.tasks,
            channels={i: n for i, n in names.items() if i in readable},
            voice_ids=voice,
            people=self._people(guild),
            scope=scope,
            approvals=self._approval_counts(guild.id, day),
            include_memory=self._memory_day_retained(guild.id, day),
        )
        g = self.team.current(guild.id)
        text = reports.build_company_report(
            self.llm,
            self.s.gemini_model,
            inp,
            lang,
            tone_block(g.default_tone, resolve_lang(lang), g.tone_note),
        )
        news = self._company_news_section(guild, lang)  # followed customers / competitors
        return f"{text}\n\n{news}" if news else text

    async def _post_report(self, guild, day: date):
        """Post the report in the private channel (created if needed). Returns the channel."""
        channel = await self._report_channel(guild)
        if channel is None:
            return None
        lang = resolve_lang(self.team.current(guild.id).language_setting)
        viewers = await asyncio.to_thread(self._channel_viewers, guild, channel)
        text = await asyncio.to_thread(self._report_text, guild, set(viewers), day, lang)
        for i in range(0, len(text), 1900):
            await channel.send(text[i : i + 1900])
        return channel

    # ------------------------------------------------------------- /รายงาน
    def _register_coo_commands(self) -> None:
        async def report_cb(
            interaction: discord.Interaction, date: str | None = None, mine: bool = False
        ):
            await self.handle_report(interaction, date, mine)

        report_cb = app_commands.guild_only()(report_cb)
        report_cb = app_commands.describe(date=t("cmd_report_date"), mine=t("cmd_report_mine"))(
            report_cb
        )
        for name in ("รายงาน", "report"):
            self.tree.add_command(
                app_commands.Command(name=name, description=t("cmd_report"), callback=report_cb)
            )

    async def handle_report(self, interaction, day: str | None = None, mine: bool = False) -> None:
        """/รายงาน: the company report (owner, admins, executives) now, for today or a date; with
        `mine` anyone gets their own day privately."""
        guild, user = interaction.guild, interaction.user
        lang = self._interaction_lang(interaction)
        await asyncio.to_thread(self.team.current, guild.id)
        if not self.team.enabled(guild.id, "memory"):
            await interaction.response.send_message(t("report_off", lang), ephemeral=True)
            return
        try:
            d = date.fromisoformat(day) if day else self.s.now().date()
        except ValueError:
            await interaction.response.send_message(t("report_bad_date", lang), ephemeral=True)
            return
        if not mine and not self.is_exec_member(user):
            await interaction.response.send_message(t("report_denied", lang), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            if mine:
                data = await asyncio.to_thread(self._my_day, guild, user, d)
                text = reports.render_personal(data, lang)
            else:
                channel = await self._report_channel(guild)
                if channel is not None:
                    viewers = await asyncio.to_thread(self._channel_viewers, guild, channel)
                    body = await asyncio.to_thread(self._report_text, guild, set(viewers), d, lang)
                    for i in range(0, len(body), 1900):
                        await channel.send(body[i : i + 1900])
                    await interaction.followup.send(
                        t("report_posted", lang, day=d.isoformat(), channel=channel.mention),
                        ephemeral=True,
                    )
                    return
                text = await asyncio.to_thread(self._report_text, guild, {user.id}, d, lang)
        except Exception:
            logger.exception("report command failed")
            await interaction.followup.send(t("report_failed", lang), ephemeral=True)
            return
        for i in range(0, len(text), 1900):
            await interaction.followup.send(text[i : i + 1900], ephemeral=True)
