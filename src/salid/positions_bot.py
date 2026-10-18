"""Positions <-> Discord roles <-> the central org (logic: `positions.py`).

`PositionsMixin` keeps ONE position catalog per company in step with three places:

Discord -> center (never overrides a stated position):
- a member's Discord role named like a catalog title / alias is that member's position (source
  `discord`); the guild owner is a Founder when nothing else is known; a member whose matching
  role was taken away loses a `discord` position;
- Administrator / Manage Server makes the linked org member `admin` (never lowered; the owner
  stays the owner);
- the org member (linked by `discord_user_id`) gets `title` and `level` mirrored, and `department`
  when it is one of the dashboard codes and still empty. A title edited in the central system
  (differs from what Salid last wrote) becomes the position (source `org`).
Center -> Discord:
- a position stated in Salid (`set_role`, or an org title) gets a Discord role with that title
  (no permissions, mentionable so it can be tagged, hoisted for levels 1-3, just below Salid's own
  top role) that is created once, remembered in the catalog, given to the person and taken
  from the person's previous position role. Roles Salid did not create are only read, except a
  permissionless role that already has the same name, which is used instead of making a twin.
  Nothing is ever deleted. Without Manage Roles: one `perm_missing` DM to the executives a day.
Hierarchy: `_position_rank` (called by `coo._rank_of`) and `_position_is_exec` (by
`bot.is_exec_member`); both are identical to before while nobody has a position.

Runs: once a day per server (`_positions_due`, bot tick), on a member's role change, on a role
change that may affect admins, and on demand (`sync_roles` tool).
"""

from __future__ import annotations

import asyncio
import logging

import discord

from .i18n import t
from .orgs import OrgStore
from .people import department_label, set_role
from .positions import (
    DEFAULT_EXEC_MAX_LEVEL,
    CatalogFull,
    PositionBook,
    is_exec_level,
    norm_title,
)

logger = logging.getLogger(__name__)

RANK_UNIT = 1_000_000  # one level step; an Administrator is worth exactly one step
OWNER_BONUS_LEVELS = 8  # the guild owner stays above every level when positions exist
CACHE_TTL_S = 60.0
DEPT_CODES = ("tech", "business", "finance", "other")
TAG_REASON = "Salid: position"
CENTER_POLL_S = 30.0  # how often the central org document is checked for edits made on the web
SRC_FROM_ASSIGN = {"stated": "stated", "org": "org", "discord": "discord", "heard": "heard"}


def _is_human(m) -> bool:
    return not getattr(m, "bot", False)


class PositionsMixin:
    """Needs `self.docs`, `self.s`, `self.team`, `self.people`, `_members`, `_guild_lang`,
    `_company_admins`, `deliver_to_person`. Call `_positions_init()` once after `_people_init()`."""

    def _positions_init(self) -> None:
        self.positions = PositionBook(self.docs, clock=self.s.now)
        self._positions_day: dict[int, str] = {}
        self._positions_warned: dict[int, str] = {}
        # gid -> (time, {uid: level}, {uid: (title, dossier source)})
        self._positions_cache: dict[int, tuple[float, dict, dict]] = {}

    # ------------------------------------------------------------- settings
    def _positions_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "role_sync")

    def _positions_exec_max(self, gid: int) -> int:
        try:
            v = int(self.team.current(gid).exec_max_level)
        except Exception:
            v = DEFAULT_EXEC_MAX_LEVEL
        return v if 1 <= v <= 4 else DEFAULT_EXEC_MAX_LEVEL

    def _positions_read(self, gid: int) -> tuple[dict[int, int], dict[int, tuple[str, str]]]:
        """(member id -> level, member id -> (title, dossier source)) of everyone with a position,
        cached for a minute (blocking on a miss)."""
        hit = self._positions_cache.get(gid)
        if hit is not None and self._clock() - hit[0] < CACHE_TTL_S:
            return hit[1], hit[2]
        levels: dict[int, int] = {}
        titles: dict[int, tuple[str, str]] = {}
        try:
            lang = self._guild_lang(gid)
            cat = {p.key: p for p in self.positions.catalog(gid)}
            for uid, a in self.positions.assignments(gid).items():
                p = cat.get(a.position_key)
                if p is not None:
                    levels[uid] = p.level
                    titles[uid] = (
                        self.positions.discord_role_name(p, lang),
                        SRC_FROM_ASSIGN.get(a.source, "stated"),
                    )
        except Exception:
            logger.warning("positions: could not read guild %s", gid, exc_info=True)
        self._positions_cache[gid] = (self._clock(), levels, titles)
        return levels, titles

    def _positions_levels(self, gid: int) -> dict[int, int]:
        return self._positions_read(gid)[0]

    def _positions_forget(self, gid: int) -> None:
        self._positions_cache.pop(gid, None)
        ranks = getattr(self, "_ranks_cache", None)
        if isinstance(ranks, dict):
            ranks.pop(gid, None)
        roster = getattr(self, "_people_roster_cache", None)
        if isinstance(roster, dict):
            roster.pop(gid, None)

    # ------------------------------------------------------------- hierarchy hooks
    def _position_rank(self, guild, m, base: int) -> int:
        """`base` (the owner / administrator / role-position rank) plus one step per level above
        'unknown': a CEO outranks an administrator staffer. Unchanged without positions. The
        guild owner is lifted above every level, so the owner is always on top."""
        try:
            if not self._positions_on(guild.id):
                return base
            levels = self._positions_levels(guild.id)
            if not levels:
                return base
            if getattr(guild, "owner_id", None) == m.id:
                return base + OWNER_BONUS_LEVELS * RANK_UNIT
            lv = levels.get(m.id, 0)
            return base + (8 - lv) * RANK_UNIT if 1 <= lv <= 7 else base
        except Exception:
            return base

    def _position_is_exec(self, member) -> bool:
        """True when the member's position level is within the server's executive levels."""
        try:
            gid = member.guild.id
            if not self._positions_on(gid):
                return False
            lv = self._positions_levels(gid).get(member.id, 0)
            return is_exec_level(lv, self._positions_exec_max(gid))
        except Exception:
            return False

    def _position_title(self, gid: int, uid: int) -> tuple[str, str]:
        """(title, dossier source) of a person's position, else ("", "")."""
        if not self._positions_on(gid):
            return "", ""
        return self._positions_read(gid)[1].get(uid, ("", ""))

    # ------------------------------------------------------------- events (called by bot.py)
    async def _positions_on_member_update(self, before, after) -> None:
        if before.roles == after.roles or getattr(after, "bot", False):
            return
        try:
            guild = after.guild
            await asyncio.to_thread(self.team.current, guild.id)
            if not self._positions_on(guild.id):
                return
            rep = _Report()
            await asyncio.to_thread(self._pos_pull_discord, guild, rep, after)
            self._positions_forget(guild.id)
        except Exception:
            logger.warning("positions: member update failed", exc_info=True)

    async def _positions_on_role_update(self, before, after) -> None:
        try:
            guild = after.guild
            await asyncio.to_thread(self.team.current, guild.id)
            if not self._positions_on(guild.id):
                return
            if before.name != after.name and after.id in await asyncio.to_thread(
                self.positions.managed_role_ids, guild.id
            ):
                key = next(
                    (
                        p.key
                        for p in await asyncio.to_thread(self.positions.catalog, guild.id)
                        if p.discord_role_id == after.id
                    ),
                    "",
                )
                if key:  # an admin renamed a position role: the position follows
                    await asyncio.to_thread(self.positions.rename, guild.id, key, after.name)
            if before.permissions != after.permissions:
                await asyncio.to_thread(self._pos_pull_discord, guild, _Report(), None)
            self._positions_forget(guild.id)
        except Exception:
            logger.warning("positions: role update failed", exc_info=True)

    async def _positions_on_role_delete(self, role) -> None:
        try:
            gid = role.guild.id
            for p in await asyncio.to_thread(self.positions.catalog, gid):
                if p.discord_role_id == role.id:
                    await asyncio.to_thread(self.positions.set_role_id, gid, p.key, 0)
            self._positions_forget(gid)
        except Exception:
            logger.warning("positions: role delete failed", exc_info=True)

    # ------------------------------------------------------------- daily job
    def _positions_center_print(self, guild) -> str:
        """What the central system says about this server's people (blocking): one string that
        changes whenever a member's role, title, level, department or Discord link changes."""
        _, org, _ = self._pos_linked(guild)
        if org is None:
            return ""
        return "|".join(
            sorted(
                f"{m.email}:{m.role}:{getattr(m, 'title', '')}:{getattr(m, 'level', 0)}:"
                f"{getattr(m, 'department', '')}:{m.identity('discord') or ''}"
                for m in org.members
            )
        )

    async def _positions_center_watch(self) -> None:
        """An edit on the central website reaches Discord within ``CENTER_POLL_S``."""
        now = self._clock()
        if now - getattr(self, "_positions_polled", 0.0) < CENTER_POLL_S:
            return
        self._positions_polled = now
        seen = self.__dict__.setdefault("_positions_seen", {})
        for guild in list(self.guilds):
            try:
                if not self._positions_on(guild.id):
                    continue
                fp = await asyncio.to_thread(self._positions_center_print, guild)
                if not fp or seen.get(guild.id) == fp:
                    continue
                first = guild.id not in seen
                seen[guild.id] = fp
                if first:
                    await self._change_snapshot(guild)
                    continue  # the daily / startup sync covers the state at start
                await self._change_detect(guild)  # edits the web did not record (change outbox)
                rep = await self.positions_sync(guild)
                seen[guild.id] = await asyncio.to_thread(self._positions_center_print, guild)
                logger.info("positions: center edit guild=%s %s", guild.id, rep.as_dict())
            except Exception:
                logger.warning("positions: center watch failed", exc_info=True)

    async def _positions_due(self) -> None:
        await self._positions_center_watch()
        today = self.s.now().date().isoformat()
        for guild in list(self.guilds):
            if self._positions_day.get(guild.id) == today:
                continue
            try:
                await asyncio.to_thread(self.team.current, guild.id)
                if not self._positions_on(guild.id):
                    continue
                self._positions_day[guild.id] = today
                rep = await self.positions_sync(guild)
                logger.info("positions: sync guild=%s %s", guild.id, rep.as_dict())
            except Exception:
                logger.warning("positions sync failed", exc_info=True)

    # ------------------------------------------------------------- the sync
    async def positions_sync(self, guild, only_uid: int | None = None) -> _Report:
        """Both directions, in this order: Discord -> center, center edits -> positions, positions
        -> Discord roles, positions -> center. Never raises; the report says what was done."""
        rep = _Report()
        gid = guild.id
        if only_uid is None:
            steps = (
                ("discord", self._pos_pull_discord, (guild, rep, None)),
                ("org", self._pos_pull_org, (guild, rep)),
            )
            for name, fn, args in steps:
                try:
                    await asyncio.to_thread(fn, *args)
                except Exception:
                    logger.warning("positions: stage=%s failed guild=%s", name, gid, exc_info=True)
        try:
            await self._pos_push_discord(guild, rep, only_uid)
        except Exception:
            logger.warning("positions: stage=push failed guild=%s", gid, exc_info=True)
        try:
            await asyncio.to_thread(self._pos_mirror_org, guild, rep)
            await asyncio.to_thread(self._pos_dossiers, guild)
        except Exception:
            logger.warning("positions: stage=mirror failed guild=%s", gid, exc_info=True)
        self._positions_forget(gid)
        await self._change_snapshot(guild)  # Salid's own mirror writes are not center edits
        return rep

    # ---- Discord -> center (blocking)
    def _pos_linked(self, guild) -> tuple[OrgStore, object | None, dict[int, object]]:
        """(org store, org, discord user id -> org Member). Blocking; empty without an org."""
        store = OrgStore(self.docs)
        try:
            org = store.org_of_guild(guild.id)
        except Exception:
            return store, None, {}
        out: dict[int, object] = {}
        for m in getattr(org, "members", None) or []:
            uid = str(m.identity("discord") or "")
            if uid.isdigit():
                out[int(uid)] = m
        return store, org, out

    def _pos_pull_discord(self, guild, rep: _Report, only=None) -> None:
        gid = guild.id
        book = self.positions
        cat = book.catalog(gid)
        by_role = {p.discord_role_id: p for p in cat if p.discord_role_id}
        members = [only] if only is not None else self._members(guild)
        managed = book.managed_role_ids(gid)
        admins: list[int] = []
        _, org, linked = self._pos_linked(guild)
        for m in members:
            if not _is_human(m) or self._people_is_bot(m.id, getattr(m, "display_name", "")):
                continue
            if m.id in linked:
                continue  # linked to the central system: that system decides (``_pos_pull_org``)
            best = None  # (level or 99, -position, Position)
            for r in getattr(m, "roles", ()) or ():
                if (
                    r == getattr(guild, "default_role", None)
                    or getattr(r, "name", "") == "@everyone"
                    or getattr(r, "id", 0)
                    in managed  # Salid's own tags show positions, not prove them
                ):
                    continue
                p = by_role.get(getattr(r, "id", 0)) or book.find(gid, getattr(r, "name", ""), cat)
                if p is None:
                    continue
                key = (p.level or 99, -int(getattr(r, "position", 0) or 0))
                if best is None or key < best[0]:
                    best = (key, p)
            if best is not None:
                if book.assign(gid, m.id, best[1].key, "discord"):
                    rep.from_discord += 1
            elif book.unassign(gid, m.id, only_source="discord"):
                rep.dropped += 1
            perms = getattr(m, "guild_permissions", None)
            if (
                getattr(guild, "owner_id", None) == m.id
                or getattr(perms, "administrator", False)
                or getattr(perms, "manage_guild", False)
            ):
                admins.append(m.id)
        self._pos_from_dossiers(guild, cat, rep, only)

    def _pos_from_dossiers(self, guild, cat, rep: _Report, only=None) -> None:
        """A title Salid already knows for a person (said by them or an executive, or heard at work)
        that matches the catalog becomes their position when nothing stronger is known."""
        gid = guild.id
        book = self.positions
        held = book.assignments(gid)
        for d in self.people.list(gid):
            if only is not None and d.user_id != getattr(only, "id", None):
                continue
            # only what was heard at work: a stated title already went through set_role, and a title
            # written back from a position must not bring a removed position back
            if d.user_id in held or not d.role_title or d.role_source != "heard":
                continue
            p = book.find(gid, d.role_title, cat)
            if p is not None and book.assign(gid, d.user_id, p.key, d.role_source):
                rep.from_discord += 1

    def _pos_admins(self, guild, admin_ids: list[int], rep: _Report) -> None:
        """Administrator / Manage Server / owner -> `admin` for the linked org member (never lowered,
        the org owner stays the owner)."""
        if not admin_ids:
            return
        store, org, linked = self._pos_linked(guild)
        if org is None:
            return
        want = {uid for uid in admin_ids if uid in linked}
        if not want:
            return
        done: list[int] = []

        def apply(members) -> bool:
            changed = False
            for m in members:
                uid = str(m.identity("discord") or "")
                if uid.isdigit() and int(uid) in want and m.role == "member":
                    m.role, changed = "admin", True
                    done.append(int(uid))
            return changed

        if store.update_members(org.id, apply):
            rep.admins_set += len(done)

    def _pos_pull_org(self, guild, rep: _Report) -> None:
        """A title edited in the central system (not what Salid last wrote) becomes the position."""
        gid = guild.id
        _, org, linked = self._pos_linked(guild)
        if org is None:
            return
        book = self.positions
        here = {m.id for m in self._members(guild)}
        stale: list[int] = []  # titles Salid itself wrote from a Discord guess: taken back
        for uid, mem in linked.items():
            if uid not in here:
                continue
            title = (getattr(mem, "title", "") or "").strip()
            cur = book.of(gid, uid)
            if cur is not None and cur[0].source in ("discord", "heard"):
                # a linked person's position comes from the central system only
                if title and norm_title(title) == norm_title(cur[0].mirrored):
                    stale.append(uid)
                    title = ""
                if book.unassign(gid, uid):
                    rep.dropped += 1
            if not title:
                # no company position in the central system. Its owner / admin / member is the
                # website access right, NOT a job: it never becomes a position.
                cur = book.of(gid, uid)
                if cur is None:
                    continue
                if cur[0].source == "stated" and not cur[0].mirrored:
                    continue  # named in Salid a moment ago, written to the center on this sync
                if book.unassign(gid, uid):  # the center has no title (or it was cleared there)
                    rep.dropped += 1
                continue
            cur = book.of(gid, uid)
            if cur is not None:
                a, p = cur
                if norm_title(title) in (norm_title(a.mirrored), norm_title(p.title)):
                    continue  # nothing new since Salid wrote it
            try:
                pos = book.find(gid, title)
                if pos is None:
                    pos, _ = book.add(gid, title)
            except CatalogFull:
                rep.skipped_other += 1
                continue
            if book.assign(gid, uid, pos.key, "org", force=True):
                rep.from_org += 1

        if stale:
            ids = {str(u) for u in stale}

            def unset(members) -> bool:
                changed = False
                for m in members:
                    if str(m.identity("discord") or "") in ids and (m.title or m.level):
                        m.title, m.level, changed = "", 0, True
                return changed

            store = OrgStore(self.docs)
            if store.update_members(org.id, unset):
                logger.info("positions: took back %d guessed title(s) guild=%s", len(stale), gid)

    def _pos_unlinked_names(self, guild, linked=None) -> list[str]:
        """Humans of the server with no org member linked to their Discord account. Blocking; []
        without an org (nothing to be linked to)."""
        if linked is None:
            _, org, linked = self._pos_linked(guild)
            if org is None:
                return []
        return sorted(
            getattr(m, "display_name", str(m.id))
            for m in self._members(guild)
            if m.id not in linked and not self._people_is_bot(m.id, getattr(m, "display_name", ""))
        )

    # ---- positions -> org document (blocking)
    def _pos_mirror_org(self, guild, rep: _Report) -> None:
        gid = guild.id
        store, org, linked = self._pos_linked(guild)
        if org is None:
            return
        rep.not_linked = self._pos_unlinked_names(guild, linked)
        book = self.positions
        cat = {p.key: p for p in book.catalog(gid)}
        held = book.assignments(gid)
        want: dict[str, tuple[str, int, str, int]] = {}  # discord id -> title, level, dept, uid
        for uid, a in held.items():
            p = cat.get(a.position_key)
            if p is not None and uid in linked:
                dept = (a.department or "").strip().casefold()
                want[str(uid)] = (p.title, p.level, dept if dept in DEPT_CODES else "", uid)
        if not want:
            return
        changed_uids: list[int] = []

        def apply(mems) -> bool:
            changed = False
            for m in mems:
                uid = str(m.identity("discord") or "")
                w = want.get(uid)
                if w is None:
                    continue
                title, level, dept, u = w
                if m.title != title or m.level != level:
                    m.title, m.level, changed = title, level, True
                    changed_uids.append(u)
                if dept and not m.department:
                    m.department, changed = dept, True
                    changed_uids.append(u)
            return changed

        if store.update_members(org.id, apply):
            rep.mirrored += len(set(changed_uids))
        for uid, a in held.items():
            p = cat.get(a.position_key)
            if p is not None and uid in linked:
                book.set_mirrored(gid, uid, p.title)

    def _pos_dossiers(self, guild) -> None:
        """The position is the source of truth of a person's role in the dossiers."""
        gid = guild.id
        self._positions_forget(gid)
        for uid in list(self.positions.assignments(gid)):
            title, source = self._position_title(gid, uid)
            if not title:
                continue
            m = guild.get_member(uid) if hasattr(guild, "get_member") else None
            name = getattr(m, "display_name", str(uid))
            if m is None or self._people_is_bot(uid, name):
                continue
            try:
                self.people.update(
                    gid, uid, name, lambda d, title=title, source=source: set_role(d, title, source)
                )
            except Exception:
                logger.warning("positions: dossier failed guild=%s", gid, exc_info=True)

    # ---- center -> Discord (async)
    def _pos_can_manage_roles(self, guild) -> bool:
        perms = getattr(getattr(guild, "me", None), "guild_permissions", None)
        return bool(getattr(perms, "manage_roles", False))

    async def _pos_push_discord(self, guild, rep: _Report, only_uid: int | None = None) -> None:
        gid = guild.id
        book = self.positions
        lang = self._guild_lang(gid)
        held = await asyncio.to_thread(book.assignments, gid)
        cat = {p.key: p for p in await asyncio.to_thread(book.catalog, gid)}
        # everybody with a known position wears its tag, without waiting for a command; a position
        # that came FROM a Discord role is already worn (that role is the tag), so it is skipped
        todo = {}
        for uid, a in held.items():
            pos = cat.get(a.position_key)
            if pos is None or (only_uid is not None and uid != only_uid):
                continue
            if True:
                m = guild.get_member(uid)
                worn = m is not None and any(
                    getattr(r, "id", 0) == pos.discord_role_id
                    or book.find(gid, getattr(r, "name", ""), list(cat.values())) is pos
                    for r in getattr(m, "roles", ()) or ()
                    if getattr(r, "name", "") != "@everyone"
                )
                if worn:
                    continue
            todo[uid] = pos
        if self._pos_can_manage_roles(guild):
            managed_ids = await asyncio.to_thread(book.managed_role_ids, gid)
            for m in [guild.get_member(only_uid)] if only_uid is not None else self._members(guild):
                if m is None:
                    continue
                mine = todo.get(m.id) or cat.get(getattr(held.get(m.id), "position_key", ""))
                for r in list(getattr(m, "roles", ()) or ()):
                    if r.id in managed_ids and (mine is None or r.id != mine.discord_role_id):
                        try:
                            await m.remove_roles(r, reason=TAG_REASON)
                            rep.untagged += 1
                        except discord.HTTPException:
                            rep.skipped_other += 1
        dept_on = bool(getattr(self.team.current(gid), "department_roles", False))
        depts = await self._pos_dept_wants(guild, held, only_uid) if dept_on else {}
        if not todo and not depts:
            return
        if not self._pos_can_manage_roles(guild):
            rep.skipped_no_permission += len(todo) + len(depts)
            await self._pos_warn_once(guild, lang)
            return
        for uid, pos in sorted(todo.items(), key=lambda kv: (kv[1].level or 99, kv[0])):
            m = guild.get_member(uid)
            if m is None:
                rep.skipped_other += 1
                continue
            try:
                role = await self._pos_role(guild, pos, rep, lang)
                if role is None:
                    continue
                managed = await asyncio.to_thread(book.managed_role_ids, gid)
                await self._pos_apply(m, role, managed, rep)
            except discord.Forbidden:
                rep.skipped_no_permission += 1
            except discord.HTTPException:
                logger.warning(
                    "positions: Discord refused guild=%s user=%s", gid, uid, exc_info=True
                )
                rep.skipped_other += 1
        for uid, dept in depts.items():
            m = guild.get_member(uid)
            if m is None:
                continue
            try:
                role = await self._pos_dept_role(guild, dept, rep)
                if role is not None and role not in m.roles:
                    await m.add_roles(role, reason=TAG_REASON)
                    rep.tagged += 1
            except discord.HTTPException:
                rep.skipped_no_permission += 1

    async def _pos_apply(self, m, role, managed: set[int], rep: _Report) -> None:
        """Give `role`, take the person's other position roles Salid manages."""
        have = {r.id for r in m.roles}
        if role.id not in have:
            await m.add_roles(role, reason=TAG_REASON)
            rep.tagged += 1
        for r in list(m.roles):
            if r.id in managed and r.id != role.id:
                await m.remove_roles(r, reason=TAG_REASON)
                rep.untagged += 1

    async def _pos_role(self, guild, pos, rep: _Report, lang: str):
        """The Discord role of a position: remembered, else a permissionless one with the same
        name, else a new one. None when it cannot be had (counted in the report)."""
        book = self.positions
        role = guild.get_role(pos.discord_role_id) if pos.discord_role_id else None
        if role is not None:
            return role
        name = book.discord_role_name(pos, lang)
        role = await self._pos_find_or_create(guild, name, hoist=1 <= pos.level <= 3, rep=rep)
        if role is not None:
            await asyncio.to_thread(book.set_role_id, guild.id, pos.key, role.id)
        return role

    async def _pos_find_or_create(self, guild, name: str, hoist: bool, rep: _Report):
        same = [
            r
            for r in guild.roles
            if r != guild.default_role
            and not getattr(r, "managed", False)
            and norm_title(getattr(r, "name", "")) == norm_title(name)
        ]
        for r in same:
            if int(getattr(getattr(r, "permissions", None), "value", 1)) == 0:
                assignable = getattr(r, "is_assignable", None)
                if callable(assignable) and not assignable():
                    rep.skipped_no_permission += 1
                    return None
                return r
        if same:  # a role of that name with real permissions is not a tag: never hand it out
            rep.skipped_other += 1
            return None
        role = await guild.create_role(
            name=name,
            permissions=discord.Permissions.none(),
            hoist=hoist,
            mentionable=True,
            reason=TAG_REASON,
        )
        rep.created += 1
        top = getattr(getattr(guild, "me", None), "top_role", None)
        if top is not None:
            try:
                await role.edit(position=max(int(top.position) - 1, 1), reason=TAG_REASON)
            except discord.HTTPException:
                logger.info("positions: could not place the new role under mine", exc_info=True)
        return role

    async def _pos_dept_wants(self, guild, held, only_uid) -> dict[int, str]:
        org_depts = await asyncio.to_thread(self._org_departments, guild.id)
        out: dict[int, str] = {}
        for m in self._members(guild):
            if only_uid is not None and m.id != only_uid:
                continue
            a = held.get(m.id)
            dept = (a.department if a else "") or org_depts.get(m.id, "")
            if dept:
                out[m.id] = department_label(dept)
        return out

    async def _pos_dept_role(self, guild, dept: str, rep: _Report):
        gid = guild.id
        known = await asyncio.to_thread(self.positions.dept_roles, gid)
        role = guild.get_role(known.get(dept, 0)) if known.get(dept) else None
        if role is not None:
            return role
        role = await self._pos_find_or_create(guild, dept, hoist=False, rep=rep)
        if role is not None:
            await asyncio.to_thread(self.positions.set_dept_role, gid, dept, role.id)
        return role

    async def _pos_warn_once(self, guild, lang: str) -> None:
        """The usual missing-permission line, once a day, to the executives."""
        today = self.s.now().date().isoformat()
        if self._positions_warned.get(guild.id) == today:
            return
        self._positions_warned[guild.id] = today
        try:
            for admin in self._company_admins(guild, 0)[:3]:
                text = t(
                    "perm_missing_nolink",
                    self._deliver_langs(guild.id, admin.id)[0],
                    what="Manage Roles",
                )
                await self.deliver_to_person(guild, admin, text)
        except Exception:
            logger.warning("positions: permission notice failed", exc_info=True)

    # ------------------------------------------------------------- tool glue
    def _pos_tag_now(self, guild, loop, uid: int):
        """`set_role` tool (agent worker thread): tag one person now. True, "pending" (scheduled
        or not finished in time) or "no_permission"."""
        if not self._positions_on(guild.id):
            return "pending"
        if not self._pos_can_manage_roles(guild):
            return "no_permission"
        self._positions_forget(guild.id)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        coro = self.positions_sync(guild, only_uid=uid)
        if running is loop:  # called on the bot's own loop: cannot wait here
            self._spawn(coro)
            return "pending"
        try:
            rep = asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=25)
        except Exception:
            return "pending"
        if rep.skipped_no_permission:
            return "no_permission"
        m = guild.get_member(uid)
        hit = self.positions.of(guild.id, uid)
        ok = (
            m is not None
            and hit is not None
            and bool(hit[1].discord_role_id)
            and any(r.id == hit[1].discord_role_id for r in m.roles)
        )
        return True if ok else "pending"

    def _pos_sync_now(self, guild, loop) -> dict:
        """`sync_roles` tool (agent worker thread)."""
        try:
            rep = asyncio.run_coroutine_threadsafe(self.positions_sync(guild), loop).result(
                timeout=120
            )
        except Exception:
            logger.warning("positions: sync tool failed", exc_info=True)
            return {"status": "error", "error": "the sync did not finish; try again"}
        return rep.as_dict()


class _Report:
    """Counts of one sync run."""

    def __init__(self) -> None:
        self.from_discord = 0
        self.from_org = 0
        self.dropped = 0
        self.admins_set = 0
        self.mirrored = 0
        self.tagged = 0
        self.untagged = 0
        self.created = 0
        self.skipped_no_permission = 0
        self.skipped_other = 0
        self.not_linked: list[str] = []

    def as_dict(self) -> dict:
        return {
            "tagged": self.tagged,
            "untagged": self.untagged,
            "roles_created": self.created,
            "mirrored_to_center": self.mirrored,
            "positions_from_discord": self.from_discord,
            "positions_from_center_edits": self.from_org,
            "positions_dropped": self.dropped,
            "admins_set_in_center": self.admins_set,
            "skipped_no_permission": self.skipped_no_permission,
            "skipped_other": self.skipped_other,
            "not_linked_to_center": self.not_linked[:15],
            "not_linked_to_center_count": len(self.not_linked),
        }
