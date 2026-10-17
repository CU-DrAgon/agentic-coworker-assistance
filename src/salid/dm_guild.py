"""Which company is a DM about? Salid can sit in many Discord servers (one per company) and a DM has
no server. The resolution order for each DM (never a guess from the content):

  1. the person shares exactly one server with Salid -> that one;
  2. the DM is a reply to a Salid DM that belongs to a company (briefing, approval, reminder, form
     question: every DM Salid sends is tagged message id -> guild) -> that company, for that reply
     only (the sticky choice is not changed);
  3. the company the person picked (sticky until they switch; still a member) -> that one;
  4. otherwise Salid asks with buttons listing ONLY the servers this person is in; the pending
     message is kept and answered after the click.

"switch to X" / "เปลี่ยนเป็นบริษัท X" asks "switch to X?" (Confirm / Cancel, or a "yes" reply) before
changing a choice that exists; the slash command /บริษัท /company picks directly. While the
person shares 2+ servers every DM answer ends with "— บริษัท: <name>". Pure rules and storage here;
the Discord glue is `DMGuildMixin`.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from datetime import UTC, datetime

import discord

from .coo_store import DocStore
from .i18n import locale_lang, resolve_lang, t

logger = logging.getLogger(__name__)

ACTIVE = "salid_dm_company"  # key = user id -> {guild_id, set_at}
TAGS = "salid_dm_tags"  # key = Salid DM message id -> {guild_id}
HOME = 0  # these records belong to no guild: the DocStore guild slot is 0
_YES = re.compile(
    r"^\s*(?:ใช่|ใช่ค่ะ|ใช่ครับ|ยืนยัน|ตกลง|โอเค|y|yes|yep|ok|okay|confirm)\s*[.!ค่ะครับนะคะ]*$", re.I
)
MAX_BUTTONS = 20  # Discord: 5 rows of 5, and we keep room for nothing else
SWITCH_COMMANDS = ("บริษัท", "company")

_SWITCH = re.compile(
    r"^\s*(?:เปลี่ยน(?:เป็น|ไปคุย(?:เรื่อง)?|ไป)?|สลับ(?:เป็น|ไป)?|ไปคุย(?:เรื่อง)?|"
    r"switch(?:\s+(?:to|over to))?|change\s+(?:the\s+)?company\s+to|talk\s+about)\s*"
    r"(?P<tag>บริษัท|company)?\s*(?P<name>.+?)\s*[.!?ค่ะครับนะคะ]*$",
    re.I,
)


def parse_switch(text: str) -> tuple[str, bool] | None:
    """("name", said_company_word) for "switch to X" / "เปลี่ยนเป็นบริษัท X"; None for anything else."""
    s = (text or "").strip()
    if not s or len(s) > 80 or "\n" in s:
        return None
    m = _SWITCH.match(s)
    if not m or not m["name"].strip():
        return None
    return m["name"].strip(), bool(m["tag"])


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_.]+", "", (s or "").casefold())


def fuzzy_pick(name: str, names: dict[int, str]) -> int | None:
    """The one server id whose name matches `name` (exact, then prefix/contains, then close
    spelling); None when nothing or several match equally."""
    q = _norm(name)
    if not q:
        return None
    norm = {gid: _norm(n) for gid, n in names.items()}
    for test in (
        lambda n: n == q,
        lambda n: n.startswith(q) or q.startswith(n),
        lambda n: q in n or n in q,
    ):
        hit = [gid for gid, n in norm.items() if n and test(n)]
        if len(hit) == 1:
            return hit[0]
        if hit:
            return None
    scored = sorted(
        ((difflib.SequenceMatcher(None, q, n).ratio(), gid) for gid, n in norm.items()),
        reverse=True,
    )
    if scored and scored[0][0] >= 0.7 and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.1):
        return scored[0][1]
    return None


class DMGuildRepo:
    """Active company per user and DM message id -> company, in the shared DocStore."""

    def __init__(self, docs: DocStore, now=lambda: datetime.now(UTC)):
        self.docs, self.now = docs, now

    def set_active(self, user_id: int, guild_id: int) -> None:
        self.docs.put(
            ACTIVE,
            HOME,
            str(user_id),
            {"guild_id": guild_id, "set_at": self.now().astimezone(UTC).isoformat()},
        )

    def active(self, user_id: int) -> int | None:
        doc = self.docs.get(ACTIVE, HOME, str(user_id))
        if not doc:
            return None
        try:
            return int(doc["guild_id"])
        except (KeyError, ValueError, TypeError):
            return None

    def clear(self, user_id: int) -> None:
        self.docs.delete(ACTIVE, HOME, str(user_id))

    def tag(self, message_id: int, guild_id: int) -> None:
        self.docs.put(TAGS, HOME, str(message_id), {"guild_id": guild_id})

    def tagged(self, message_id: int) -> int | None:
        doc = self.docs.get(TAGS, HOME, str(message_id))
        return int(doc["guild_id"]) if doc else None


def company_footer(name: str, lang: str) -> str:
    return "\n" + t("dmg_footer", lang, name=name)


class CompanyButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:dmg:(?P<gid>[0-9]+):(?P<uid>[0-9]+)",
):
    """One server in the "which company?" question. The custom id carries the server and the
    person, so the buttons keep working after a restart."""

    def __init__(self, guild_id: int, user_id: int, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=(label or str(guild_id))[:80],
                style=discord.ButtonStyle.primary,
                custom_id=f"salid:dmg:{guild_id}:{user_id}",
            )
        )
        self.guild_id, self.user_id = guild_id, user_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(int(match["gid"]), int(match["uid"]), getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_company_pick(interaction, self.guild_id, self.user_id)


class SwitchButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:dmsw:(?P<act>ok|no):(?P<gid>[0-9]+):(?P<uid>[0-9]+)",
):
    """Confirm / Cancel on "switch to company X?"."""

    def __init__(self, act: str, guild_id: int, user_id: int, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=discord.ButtonStyle.success if act == "ok" else discord.ButtonStyle.secondary,
                custom_id=f"salid:dmsw:{act}:{guild_id}:{user_id}",
            )
        )
        self.act, self.guild_id, self.user_id = act, guild_id, user_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], int(match["gid"]), int(match["uid"]), getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_company_switch(
            interaction, self.act, self.guild_id, self.user_id
        )


def company_view(user_id: int, shared) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for g in list(shared)[:MAX_BUTTONS]:
        view.add_item(CompanyButton(g.id, user_id, g.name))
    return view


class DMGuildMixin:
    def _dm_guild_init(self) -> None:
        self.dm_guilds = DMGuildRepo(self.docs, now=lambda: datetime.now(UTC))
        self._dm_held: dict[int, discord.Message] = {}  # user id -> the DM waiting for a pick
        self._dm_switch_pending: dict[int, int] = {}  # user id -> server they were asked about
        self._dm_footers: dict[int, str] = {}  # message id -> footer for the answer

    # ------------------------------------------------------------- tagging Salid's own DMs
    async def _dm_tag(self, sent, guild_id: int) -> None:
        """Remember that this DM Salid sent belongs to `guild_id` (a reply to it resolves there)."""
        mid = getattr(sent, "id", None)
        if mid is None or not guild_id:
            return
        try:
            await asyncio.to_thread(self.dm_guilds.tag, mid, guild_id)
        except Exception:
            logger.warning("could not tag a DM with its company", exc_info=True)

    async def _dm_sent_hook(self, member, sent) -> None:
        guild = getattr(member, "guild", None)
        if guild is not None:
            await self._dm_tag(sent, guild.id)

    def _shared_guilds(self, user_id: int) -> list[tuple]:
        pairs = [(g, g.get_member(user_id)) for g in self.guilds]
        return [(g, m) for g, m in pairs if m is not None]

    # ------------------------------------------------------------- resolution
    async def _dm_resolve(self, message, shared: list[tuple]):
        """The (guild, member) this DM is about, or None when Salid asked / answered instead."""
        if len(shared) == 1:
            return shared[0]
        uid = message.author.id
        by_id = {g.id: (g, m) for g, m in shared}
        names = {g.id: g.name for g, _ in shared}
        lang = self._lang(message)
        active = await asyncio.to_thread(self.dm_guilds.active, uid)
        if active is not None and active not in by_id:
            await asyncio.to_thread(self.dm_guilds.clear, uid)  # they left that server: ask again
            active = None
        pending = self._dm_switch_pending.pop(uid, None)
        text = message.content or ""
        if pending in by_id and _YES.match(text):  # "yes" to "switch to X?"
            await self._dm_switch(message, by_id[pending][0], lang)
            return None
        sw = parse_switch(text)
        if sw is not None:  # "switch to X": only among the servers this person is in
            gid = fuzzy_pick(sw[0], names)
            if gid is not None:
                if active is None or active == gid:  # nothing to confirm
                    await self._dm_switch(message, by_id[gid][0], lang)
                else:
                    await self._dm_confirm_switch(message, by_id[gid][0], lang)
                return None
            if sw[1]:  # said "company" but we do not know which: ask
                await self._dm_ask(message, shared, lang, hold=False)
                return None
        ref = getattr(message, "reference", None)
        ref_id = getattr(ref, "message_id", None)
        if ref_id:  # a reply to a Salid DM that belongs to a company
            gid = await asyncio.to_thread(self.dm_guilds.tagged, ref_id)
            if gid in by_id:
                return by_id[gid]
        if active is not None:
            return by_id[active]
        await self._dm_ask(message, shared, lang)
        return None

    async def _dm_switch(self, message, guild, lang: str) -> None:
        await asyncio.to_thread(self.dm_guilds.set_active, message.author.id, guild.id)
        await message.reply(t("dmg_switched", lang, name=guild.name), mention_author=False)

    async def _dm_confirm_switch(self, message, guild, lang: str) -> None:
        uid = message.author.id
        self._dm_switch_pending[uid] = guild.id
        view = discord.ui.View(timeout=None)
        view.add_item(SwitchButton("ok", guild.id, uid, t("dmg_btn_ok", lang)))
        view.add_item(SwitchButton("no", guild.id, uid, t("dmg_btn_no", lang)))
        await message.reply(
            t("dmg_confirm", lang, name=guild.name), view=view, mention_author=False
        )

    async def _dm_ask(self, message, shared, lang: str, *, hold: bool = True) -> None:
        uid = message.author.id
        if hold:
            self._dm_held[uid] = message
        offer = [t("dmg_ask", lang), t("pc_offer", lang)]
        for g, _ in shared:  # the private channel, where one exists, is the clearest context
            cid = await asyncio.to_thread(self.private_chats.get, g.id, uid)
            if cid is not None and g.get_channel(cid) is not None:
                url = f"https://discord.com/channels/{g.id}/{cid}"
                offer.append(t("pc_offer_link", lang, server=g.name, url=url))
        await message.reply(
            "\n".join(offer), view=company_view(uid, [g for g, _ in shared]), mention_author=False
        )

    def _dm_note_footer(self, message, shared_count: int, guild) -> None:
        """Mark the answer to this DM with the company when the person is in 2+ servers."""
        if shared_count > 1:
            self._dm_footers[message.id] = company_footer(guild.name, self._lang(message))

    # ------------------------------------------------------------- button and slash command
    def _pick_lang(self, interaction, guild_id: int) -> str:
        """The company's language once chosen; before that the Discord locale (as before)."""
        setting = self.team.current(guild_id).language_setting
        return (
            resolve_lang(setting)
            if setting != "auto"
            else locale_lang(getattr(interaction, "locale", ""))
        )

    async def handle_company_pick(self, interaction, guild_id: int, user_id: int) -> None:
        lang = self._pick_lang(interaction, guild_id)
        if interaction.user.id != user_id:
            await interaction.response.send_message(t("dmg_not_yours", lang), ephemeral=True)
            return
        guild = next((g for g in self.guilds if g.id == guild_id), None)
        if guild is None or guild.get_member(user_id) is None:
            await interaction.response.send_message(t("dmg_gone", lang), ephemeral=True)
            return
        await asyncio.to_thread(self.dm_guilds.set_active, user_id, guild_id)
        await interaction.response.edit_message(
            content=t("dmg_switched", lang, name=guild.name), view=None
        )
        held = self._dm_held.pop(user_id, None)
        if held is not None:  # now answer the question that was waiting
            try:
                await self._direct_message(held)
            except Exception:
                logger.exception("could not answer the held DM")

    async def handle_company_switch(self, interaction, act: str, guild_id: int, user_id: int):
        lang = self._pick_lang(interaction, guild_id)
        if interaction.user.id != user_id:
            await interaction.response.send_message(t("dmg_not_yours", lang), ephemeral=True)
            return
        self._dm_switch_pending.pop(user_id, None)
        guild = next((g for g in self.guilds if g.id == guild_id), None)
        if act == "no":
            await interaction.response.edit_message(content=t("dmg_kept", lang), view=None)
        elif guild is None or guild.get_member(user_id) is None:
            await interaction.response.edit_message(content=t("dmg_gone", lang), view=None)
        else:
            await asyncio.to_thread(self.dm_guilds.set_active, user_id, guild_id)
            await interaction.response.edit_message(
                content=t("dmg_switched", lang, name=guild.name), view=None
            )

    async def handle_company(self, interaction, name: str | None = None) -> None:
        lang = self._interaction_lang(interaction)
        uid = interaction.user.id
        shared = [g for g, _ in self._shared_guilds(uid)]
        if not shared:
            await interaction.response.send_message(t("dmg_none", lang), ephemeral=True)
            return
        if name:
            gid = fuzzy_pick(name, {g.id: g.name for g in shared})
            if gid is not None:
                g = next(x for x in shared if x.id == gid)
                await asyncio.to_thread(self.dm_guilds.set_active, uid, gid)
                await interaction.response.send_message(
                    t("dmg_switched", lang, name=g.name), ephemeral=True
                )
                return
        active = await asyncio.to_thread(self.dm_guilds.active, uid)
        current = next((g.name for g in shared if g.id == active), None)
        text = t("dmg_ask", lang) + (f"\n{t('dmg_current', lang, name=current)}" if current else "")
        await interaction.response.send_message(
            text, view=company_view(uid, shared), ephemeral=True
        )
