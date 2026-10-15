from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
from datetime import date, datetime

import discord
from discord import app_commands

from .archive import make_archive
from .config import Settings
from .deep_task import (
    ACK_MESSAGE,
    BUSY_MESSAGE,
    DISABLED_MESSAGE,
    NOT_EXEC_MESSAGE,
    REPORT_NAME,
    USAGE_MESSAGE,
    DeepTaskError,
    parse_deep_trigger,
    reply_parts,
    run_deep_task,
)
from .digest import build_daily_digest
from .extract import apply_correction, guess_mime, ingest_attachment
from .guild_settings import BotGuildSettings, GuildStore, make_guild_store
from .qa import ask
from .store import Store

logger = logging.getLogger(__name__)
ATTACH_LIMIT = 15 * 1024 * 1024
SYNC_MIN_INTERVAL_S = 30.0  # channel-list writes per guild are batched to one per interval
MAX_SYNCED_CHANNELS = 500
RECEIPT_CHANNEL_HINTS = (
    "ใบเสร็จ",
    "receipt",
)  # a channel whose name has one of these takes receipts
QA_OFF_MESSAGE = "ตอนนี้ปิดการถามตอบอยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด"
DEEP_OFF_MESSAGE = "ตอนนี้ปิดงานใหญ่อยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด"
DM_MESSAGE = "สลิดแยกข้อมูลของแต่ละเซิร์ฟเวอร์ครับ รบกวนแท็ก {mention} ถามในห้องแชตของเซิร์ฟเวอร์ที่ต้องการนะครับ"
WELCOME_MESSAGE = (
    "สวัสดีครับ ผม “สลิด” เลขาที่ช่วยทำบัญชีจากใบเสร็จ 📒\n"
    "• ส่งรูปใบเสร็จในห้องที่ชื่อมีคำว่า “ใบเสร็จ” หรือ “receipt” "
    "หรือแท็ก {mention} พร้อมรูป (ห้องไหนก็ได้) ผมจะอ่านยอดและบันทึกให้เอง\n"
    "• ผู้บริหารถามเรื่องค่าใช้จ่ายได้โดยแท็ก {mention} แล้วพิมพ์คำถาม\n"
    "• ผู้ดูแลพิมพ์ /ตั้งค่า เพื่อเชื่อมกับศูนย์บัญชาการ (หน้าเว็บสำหรับตั้งค่า)"
)
GREETING_MESSAGE = (
    'สวัสดีครับ สลิดเองครับ 🐱 ส่งรูปใบเสร็จมาได้เลย หรือแท็กสลิดพร้อมคำถาม เช่น "เดือนนี้จ่ายค่าวัตถุดิบไปเท่าไร" ครับ'
)
# A bare @mention right after the same person's message is treated as a question about it.
FOLLOWUP_WINDOW_SECONDS = 120
# After Salid answers someone, their next messages in that channel need no @mention for this long.
CONVERSATION_SECONDS = 180
# Chat context handed to the model: this person's and Salid's recent messages in the channel.
HISTORY_MESSAGES = 10
HISTORY_SECONDS = 30 * 60
SETUP_DENIED_MESSAGE = "คำสั่งนี้ใช้ได้เฉพาะเจ้าของเซิร์ฟเวอร์หรือผู้ที่มีสิทธิ์จัดการเซิร์ฟเวอร์ครับ"
SETUP_MESSAGE = (
    "ลิงก์เชื่อม {name} กับศูนย์บัญชาการของสลิด 🔗\n{url}\n"
    "เปิดลิงก์แล้วเข้าสู่ระบบด้วยบัญชี Google ของคุณ ลิงก์ใช้ได้ครั้งเดียวและหมดอายุใน 15 นาที "
    "อย่าส่งต่อให้คนอื่นครับ"
)
SETUP_FAILED_MESSAGE = "สร้างลิงก์ไม่สำเร็จ ลองพิมพ์ /ตั้งค่า อีกครั้งภายหลังครับ"
SETUP_COMMANDS = ("ตั้งค่า", "setup")  # Thai name plus an English alias


def chunks(text: str, n: int = 1900):
    while text:
        yield text[:n]
        text = text[n:]


def is_receipt_channel_name(name: str | None) -> bool:
    low = (name or "").casefold()
    return any(h in low for h in RECEIPT_CHANNEL_HINTS)


# Channels named after Salid are for talking to it: every message there is a question.
ASK_CHANNEL_HINTS = ("salid", "สลิด")


def is_ask_channel_name(name: str | None) -> bool:
    low = (name or "").casefold()
    return any(h in low for h in ASK_CHANNEL_HINTS)


def has_receipt_file(message) -> bool:
    return any(guess_mime(a.filename, a.content_type) for a in message.attachments)


def can_manage_guild(member) -> bool:
    """Owner, Administrator or Manage Server: may link the hub and counts as an executive."""
    perms = member.guild_permissions
    return member.guild.owner_id == member.id or perms.administrator or perms.manage_guild


def mentions_bot(message: discord.Message, me: discord.ClientUser | None) -> bool:
    """True for an @mention of the bot user or of its integration role. Discord's autocomplete
    often picks the role, which has the same name as the bot."""
    if me is None:
        return False
    if me in message.mentions:
        return True
    roles = getattr(message, "role_mentions", ())
    return any(getattr(getattr(r, "tags", None), "bot_id", None) == me.id for r in roles)


class SecretaryBot(discord.Client):
    def __init__(
        self, settings: Settings, store: Store, llm, guilds: GuildStore | None = None
    ) -> None:
        intents = discord.Intents(
            guilds=True, messages=True, message_content=True, members=True, dm_messages=True
        )
        super().__init__(intents=intents)
        self.s, self.store, self.llm = settings, store, llm
        # Per-guild settings document (Firestore, or in memory for local dev).
        self.guilds_store = guilds or make_guild_store(settings)
        self.team = BotGuildSettings(self.guilds_store)
        self.archive = make_archive(settings)
        self.tree = app_commands.CommandTree(self)
        self._talking: dict[tuple[int, int], float] = {}  # (channel, user) -> last answer time
        self._digest_task: asyncio.Task | None = None
        self._digest_sent: dict[int, date] = {}
        self._deep_tasks: set[asyncio.Task] = set()  # strong refs so tasks are not GC'd
        self._sync_last: dict[int, float] = {}
        self._sync_pending: dict[int, asyncio.Task] = {}

    async def setup_hook(self):
        self._register_commands()
        try:
            await self.tree.sync()  # global commands; Discord may take a while to show them
        except discord.HTTPException:
            logger.exception("could not sync slash commands")
        self._digest_task = asyncio.create_task(self._digest_loop())

    async def on_ready(self):
        logger.info("logged in as %s in %d guilds", self.user, len(self.guilds))
        for guild in self.guilds:
            await self._sync_guild(guild)

    # ------------------------------------------------------------- guild lifecycle
    @staticmethod
    def _guild_snapshot(guild: discord.Guild) -> dict:
        return {
            "name": guild.name,
            "icon_url": guild.icon.url if guild.icon else "",
            "channels": [{"id": str(c.id), "name": c.name} for c in guild.text_channels][
                :MAX_SYNCED_CHANNELS
            ],
        }

    async def _sync_guild(self, guild: discord.Guild, *, installed: bool = False) -> None:
        """Write the guild's name, icon and text channels (hub-managed fields are untouched)."""
        self._sync_last[guild.id] = time.monotonic()
        try:
            await asyncio.to_thread(
                self.guilds_store.upsert_guild,
                guild.id,
                active=True,
                installed=installed,
                **self._guild_snapshot(guild),
            )
        except Exception:
            logger.exception("could not sync guild %s", guild.id)

    def _request_sync(self, guild: discord.Guild) -> None:
        """Throttled resync for channel events: at most one write per interval, always ending
        with the latest state (a burst of changes schedules one trailing sync)."""
        if guild.id in self._sync_pending:
            return
        wait = max(0.0, SYNC_MIN_INTERVAL_S - (time.monotonic() - self._sync_last.get(guild.id, 0)))

        async def later():
            try:
                await asyncio.sleep(wait)
                self._sync_pending.pop(guild.id, None)
                await self._sync_guild(guild)
            finally:
                self._sync_pending.pop(guild.id, None)

        self._sync_pending[guild.id] = asyncio.create_task(later())

    async def on_guild_join(self, guild: discord.Guild):
        logger.info("joined guild %s", guild.id)
        await self._sync_guild(guild, installed=True)
        target = guild.system_channel
        if target is None or not target.permissions_for(guild.me).send_messages:
            target = next(
                (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None
            )
        if target is not None:
            with contextlib.suppress(discord.HTTPException):
                await target.send(WELCOME_MESSAGE.format(mention=self.user.mention))

    async def on_guild_remove(self, guild: discord.Guild):
        logger.info("removed from guild %s", guild.id)
        try:
            await asyncio.to_thread(self.guilds_store.set_active, guild.id, False)
        except Exception:
            logger.exception("could not mark guild %s inactive", guild.id)

    async def on_guild_channel_create(self, channel):
        self._request_sync(channel.guild)

    async def on_guild_channel_update(self, before, after):
        if before.name != after.name:
            self._request_sync(after.guild)

    async def on_guild_channel_delete(self, channel):
        self._request_sync(channel.guild)

    # ------------------------------------------------------------- permissions
    def is_exec_member(self, member: discord.Member) -> bool:
        """Executives: env user ids, the guild owner, Administrator / Manage Server, or a role named
        in the guild's exec_role_names. Everyone else only sees their own receipts."""
        if member.id in self.s.exec_user_ids or can_manage_guild(member):
            return True
        names = self.team.exec_role_names(member.guild.id, self.s.exec_role_names)
        return bool(names & {r.name for r in member.roles})

    # ------------------------------------------------------------- routing
    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author == self.user:
            return
        try:
            if message.guild is None:
                await self._direct_message(message)
                return
            gid = message.guild.id
            await asyncio.to_thread(self.team.current, gid)  # refresh the cache off the loop
            receipts_on = self.team.enabled(gid, "receipts")
            if receipts_on and await self._maybe_correction(message):
                return
            mentioned = mentions_bot(message, self.user)
            receipt_ids = self.team.receipt_channel_ids(gid, self.s.receipt_channel_ids)
            is_receipt_post = message.attachments and (
                message.channel.id in receipt_ids
                or is_receipt_channel_name(getattr(message.channel, "name", ""))
                or (mentioned and has_receipt_file(message))
            )
            if is_receipt_post:
                if receipts_on:
                    await self._intake(message)
                return
            ask_ids = self.team.ask_channel_ids(gid, self.s.ask_channel_ids)
            if (
                mentioned
                or message.channel.id in ask_ids
                or is_ask_channel_name(getattr(message.channel, "name", ""))
                or self._in_conversation(message)
            ):
                await self._answer(message, message.guild, message.author)
        except Exception:
            logger.exception("handler failed")

    async def _direct_message(self, message: discord.Message):
        """DMs have no guild: answer only when the asker shares exactly one guild with Salid."""
        shared = [(g, g.get_member(message.author.id)) for g in self.guilds]
        shared = [(g, m) for g, m in shared if m is not None]
        if len(shared) != 1:
            await message.reply(DM_MESSAGE.format(mention=self.user.mention), mention_author=False)
            return
        guild, member = shared[0]
        await asyncio.to_thread(self.team.current, guild.id)
        await self._answer(message, guild, member)

    async def _maybe_correction(self, message: discord.Message) -> bool:
        ref = message.reference
        if not ref or not ref.message_id or not message.content.strip():
            return False
        receipt = self.store.get_by_reply_message(ref.message_id)
        if not receipt or receipt.guild_id != message.guild.id:
            return False
        if message.author.id != receipt.submitter_id and not self.is_exec_member(message.author):
            await message.reply("แก้ไขได้เฉพาะผู้ส่งใบเสร็จหรือผู้บริหารเท่านั้น", mention_author=False)
            return True
        async with message.channel.typing():
            text = await asyncio.to_thread(
                apply_correction, self.store, self.llm, self.s, receipt, message.content
            )
        await message.reply(text, mention_author=False)
        return True

    async def _intake(self, message: discord.Message):
        note = message.content.strip()
        for att in message.attachments:
            if att.size > ATTACH_LIMIT:
                continue
            data = await att.read()
            async with message.channel.typing():
                res = await asyncio.to_thread(
                    ingest_attachment,
                    self.store,
                    self.llm,
                    self.s,
                    guild_id=message.guild.id,
                    data=data,
                    filename=att.filename,
                    content_type=att.content_type,
                    submitter_id=message.author.id,
                    submitter_name=message.author.display_name,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    note=note,
                    archive=self.archive,
                )
            if res.status == "ignored":
                continue
            if res.reaction:
                with contextlib.suppress(discord.HTTPException):
                    await message.add_reaction(res.reaction)
            sent = await message.reply(res.reply, mention_author=False)
            if res.status == "created":
                self.store.set_reply_message(res.receipt.id, sent.id)

    def _in_conversation(self, message: discord.Message) -> bool:
        """A reply to Salid, or a follow-up soon after Salid answered this person here.
        Messages that @mention someone else are left alone: they are talking to that person."""
        if any(u != self.user for u in message.mentions):
            return False
        ref = message.reference
        resolved = getattr(ref, "resolved", None) if ref else None
        if self.user is not None and getattr(resolved, "author", None) == self.user:
            return True
        last = self._talking.get((message.channel.id, message.author.id))
        return last is not None and time.monotonic() - last < CONVERSATION_SECONDS

    def _note_talking(self, message: discord.Message) -> None:
        self._talking[(message.channel.id, message.author.id)] = time.monotonic()

    def _strip_mentions(self, text: str, message: discord.Message) -> str:
        if self.user:
            text = text.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        for role in getattr(message, "role_mentions", ()):
            text = text.replace(f"<@&{role.id}>", "")
        return text.strip()

    async def _recent_chat(self, message: discord.Message) -> str:
        """Recent lines between this person and Salid here, oldest first (context only; the
        ledger scope is still decided in code)."""
        history = getattr(message.channel, "history", None)
        if history is None:
            return ""
        lines: list[str] = []
        try:
            async for prev in history(limit=30, before=message):
                if (message.created_at - prev.created_at).total_seconds() > HISTORY_SECONDS:
                    break
                if prev.author.id not in (message.author.id, getattr(self.user, "id", None)):
                    continue
                text = self._strip_mentions(prev.content, prev)
                if text:
                    who = "สลิด" if prev.author == self.user else prev.author.display_name
                    lines.append(f"{who}: {text[:500]}")
                if len(lines) >= HISTORY_MESSAGES:
                    break
        except discord.HTTPException:
            logger.warning("could not read channel history", exc_info=True)
        return "\n".join(reversed(lines))

    async def _previous_text(self, message: discord.Message) -> str:
        """The author's own message just before a bare @mention ("Hi" then "@Salid")."""
        history = getattr(message.channel, "history", None)
        if history is None:
            return ""
        try:
            async for prev in history(limit=5, before=message):
                if prev.author.id != message.author.id:
                    continue
                age = (message.created_at - prev.created_at).total_seconds()
                if age > FOLLOWUP_WINDOW_SECONDS:
                    return ""
                text = self._strip_mentions(prev.content, prev)
                if text:
                    return text
        except discord.HTTPException:
            logger.warning("could not read channel history", exc_info=True)
        return ""

    async def _answer(self, message: discord.Message, guild: discord.Guild, member):
        q = self._strip_mentions(message.content, message)
        if not q:
            q = await self._previous_text(message)
        if not q:
            await message.reply(GREETING_MESSAGE, mention_author=False)
            self._note_talking(message)
            return
        gid = guild.id
        is_exec = self.is_exec_member(member)
        task = parse_deep_trigger(q)
        if task is not None:
            if self.team.enabled(gid, "deep_tasks"):
                await self._deep(message, gid, task, is_exec)
            else:
                await message.reply(DEEP_OFF_MESSAGE, mention_author=False)
            return
        if not self.team.enabled(gid, "qa"):
            await message.reply(QA_OFF_MESSAGE, mention_author=False)
            return
        history = await self._recent_chat(message)
        async with message.channel.typing():
            ans = await asyncio.to_thread(
                ask,
                self.s,
                self.store,
                q,
                guild_id=gid,
                is_exec=is_exec,
                user_id=message.author.id,
                history=history,
            )
        logger.info("answered in %.1fs (%d rounds, exec=%s)", ans.seconds, ans.rounds, is_exec)
        self._note_talking(message)
        for i, part in enumerate(chunks(ans.text)):
            await (
                message.reply(part, mention_author=False) if i == 0 else message.channel.send(part)
            )

    async def _deep(self, message: discord.Message, guild_id: int, task: str, is_exec: bool):
        """Executives only. Acknowledge at once, run Gemini CLI in the background, post the result."""
        if not is_exec:
            await message.reply(NOT_EXEC_MESSAGE, mention_author=False)
        elif not self.s.gemini_cli_path:
            await message.reply(DISABLED_MESSAGE, mention_author=False)
        elif not task:
            await message.reply(USAGE_MESSAGE, mention_author=False)
        elif any(not t.done() for t in self._deep_tasks):
            await message.reply(BUSY_MESSAGE, mention_author=False)
        else:
            await message.reply(ACK_MESSAGE, mention_author=False)
            t = asyncio.create_task(self._run_deep(message, guild_id, task))
            self._deep_tasks.add(t)
            t.add_done_callback(self._deep_tasks.discard)

    async def _run_deep(self, message: discord.Message, guild_id: int, task: str):
        try:
            res = await asyncio.to_thread(
                run_deep_task, self.s, self.store, task, guild_id=guild_id
            )
        except DeepTaskError as e:
            await message.channel.send(f"❌ {e}")
            return
        except Exception:
            logger.exception("deep task crashed")
            await message.channel.send("❌ งานใหญ่ล้มเหลว ลองใหม่อีกครั้งภายหลังครับ")
            return
        logger.info(
            "deep task done in %.0fs, %d tokens, models=%s",
            res.seconds,
            res.tokens,
            list(res.models),
        )
        text, attachment = reply_parts(res)
        file = discord.File(io.BytesIO(attachment), filename=REPORT_NAME) if attachment else None
        await message.reply(text, file=file, mention_author=False)

    # ------------------------------------------------------------- slash commands
    def _register_commands(self) -> None:
        async def setup(interaction: discord.Interaction):
            await self.handle_setup(interaction)

        setup = app_commands.guild_only()(setup)
        setup = app_commands.default_permissions(manage_guild=True)(setup)
        for name in SETUP_COMMANDS:
            self.tree.add_command(
                app_commands.Command(
                    name=name,
                    description="เชื่อมเซิร์ฟเวอร์นี้กับศูนย์บัญชาการของสลิด (ผู้ดูแลเท่านั้น)",
                    callback=setup,
                )
            )

    async def handle_setup(self, interaction: discord.Interaction) -> None:
        """/ตั้งค่า: DM-style ephemeral reply with a one-time link to the hub for this guild."""
        guild, user = interaction.guild, interaction.user
        if guild is None or not hasattr(user, "guild_permissions") or not can_manage_guild(user):
            await interaction.response.send_message(SETUP_DENIED_MESSAGE, ephemeral=True)
            return
        try:
            code = await asyncio.to_thread(
                self.guilds_store.create_link_code, guild.id, user.id, str(user)
            )
        except Exception:
            logger.exception("could not create a link code")
            await interaction.response.send_message(SETUP_FAILED_MESSAGE, ephemeral=True)
            return
        url = f"{self.s.hub_url.rstrip('/')}/hub?link={code}"
        await interaction.response.send_message(
            SETUP_MESSAGE.format(name=guild.name, url=url), ephemeral=True
        )

    # ------------------------------------------------------------- digest
    def due_digests(self, now: datetime) -> list[tuple[discord.Guild, discord.abc.Messageable]]:
        """Guilds whose daily digest is due this minute (and not yet sent today)."""
        due = []
        for guild in self.guilds:
            gid = guild.id
            if self._digest_sent.get(gid) == now.date() or not self.team.enabled(gid, "digest"):
                continue
            channel_id = self.team.digest_channel_id(gid, self.s.digest_channel_id)
            channel = guild.get_channel(channel_id) if channel_id else None
            if channel is None:  # no digest channel (or not in this guild): skip the guild
                continue
            if now.strftime("%H:%M") == self.team.digest_time(gid, self.s.digest_time):
                due.append((guild, channel))
        return due

    async def _send_digests(self, now: datetime) -> None:
        for guild, channel in self.due_digests(now):
            self._digest_sent[guild.id] = now.date()
            try:
                text = await asyncio.to_thread(build_daily_digest, self.store, guild.id, now.date())
                for part in chunks(text):
                    await channel.send(part)
            except Exception:
                logger.exception("digest for guild %s failed", guild.id)

    async def _digest_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            for guild in self.guilds:  # refresh the caches off the event loop
                await asyncio.to_thread(self.team.current, guild.id)
            await self._send_digests(self.s.now())
            await asyncio.sleep(20)  # < 1 minute, so the HH:MM match is never skipped
