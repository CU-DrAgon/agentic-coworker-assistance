from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

import discord
from discord import app_commands

from . import gate
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
from .gate import GateLine, is_join_candidate
from .guild_settings import BotGuildSettings, GuildStore, make_guild_store
from .meeting import MeetingManager, parse_meeting_command
from .modes import (
    CHAT_CHANNEL_NAME,
    MODE_HELP,
    MODE_LABELS,
    MODES,
    is_chat_channel_name,
    parse_snooze_command,
)
from .naming import mentions_name
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
    "สวัสดีครับ ผม “สลิด” เลขาและเพื่อนร่วมทีม 🐱\n"
    "• คุยกับผมได้ทุกเรื่องในห้อง {chat} พิมพ์ได้เลยไม่ต้องแท็ก "
    "ห้องอื่นแท็ก {mention} หรือเรียกชื่อ “สลิด” ได้เลยครับ "
    "(พิมพ์ /โหมด เพื่อดูหรือปรับว่าผมจะคุยในแต่ละห้องแค่ไหน)\n"
    "• ส่งรูปใบเสร็จในห้องที่ชื่อมีคำว่า “ใบเสร็จ” หรือ “receipt” "
    "หรือแท็ก {mention} พร้อมรูป ผมจะอ่านยอดและบันทึกให้เอง\n"
    "• ผู้ดูแลพิมพ์ /ตั้งค่า เพื่อเชื่อมกับศูนย์บัญชาการ (หน้าเว็บสำหรับตั้งค่า)"
)
# CHAT_CHANNEL_NAME (salid.modes) is created on join (needs Manage Channels): mode "chat".
CHAT_CHANNEL_TOPIC = "คุยกับน้องสลิดได้ทุกเรื่อง พิมพ์ได้เลยไม่ต้องแท็ก"
GREETING_MESSAGE = (
    'สวัสดีครับ สลิดเองครับ 🐱 ส่งรูปใบเสร็จมาได้เลย หรือแท็กสลิดพร้อมคำถาม เช่น "เดือนนี้จ่ายค่าวัตถุดิบไปเท่าไร" ครับ'
)
# A bare @mention right after the same person's message is treated as a question about it.
FOLLOWUP_WINDOW_SECONDS = 120
# After Salid answers in a channel, messages there need no @mention for this long (a gate decides
# whether each one is meant for Salid). Silence for this long ends the conversation.
CONVERSATION_SECONDS = 180
SIGNOFF_MESSAGE = "ถ้ามีอะไรก็ทักน้องสลิดมาได้เลยนะครับ 👋"
SIGNOFF_MIN_REPLIES = 2  # fewer replies: only a 👋 reaction on Salid's last reply
GATE_HISTORY = 10  # channel lines the in-window gate reads
JOIN_HISTORY = 15  # channel lines the join-mode batch gate reads
QUIET_SECONDS = 30  # join mode: a channel must be this quiet before the batch gate runs
MAX_INTERJECTIONS_PER_HOUR = 3  # join mode: unprompted answers per channel, rolling hour
MAX_JOIN_CANDIDATES = 5
SNOOZE_SECONDS = 3600
TICK_SECONDS = 5.0
SNOOZE_MESSAGE = "ได้เลยครับ สลิดเงียบในห้องนี้ 1 ชั่วโมง 🤫 พิมพ์ “@Salid คุยได้แล้ว” เมื่อไรก็เรียกกลับได้ครับ"
UNSNOOZE_MESSAGE = "กลับมาคุยแล้วครับ 😊"
NOT_SNOOZED_MESSAGE = "สลิดคุยได้ตามปกติอยู่แล้วครับ 🙂"
MODE_DENIED_MESSAGE = "ปรับโหมดได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารครับ"
MODE_FAILED_MESSAGE = "ตั้งโหมดไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะครับ"
MODE_COMMANDS = ("โหมด", "mode")
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
JOIN_COMMANDS = ("เข้าประชุม", "join")
LEAVE_COMMANDS = ("ออกประชุม", "leave")


def chunks(text: str, n: int = 1900):
    while text:
        yield text[:n]
        text = text[n:]


def is_receipt_channel_name(name: str | None) -> bool:
    low = (name or "").casefold()
    return any(h in low for h in RECEIPT_CHANNEL_HINTS)


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


def mentions_other(message: discord.Message, me: discord.ClientUser | None) -> bool:
    """True when the message @mentions a user or role other than Salid (it talks to someone else)."""
    if any(u != me for u in message.mentions):
        return True
    roles = getattr(message, "role_mentions", ())
    return any(
        getattr(getattr(r, "tags", None), "bot_id", None) != getattr(me, "id", None) for r in roles
    )


@dataclass
class _Convo:
    """A running conversation with Salid in one channel (call/join modes)."""

    channel: object
    last: float  # clock time of Salid's latest answer
    replies: int = 0
    last_reply: object = None  # Salid's latest message, for the 👋 reaction


@dataclass
class _Pending:
    """Join mode: candidate messages waiting for the channel to go quiet."""

    channel: object
    guild: object
    last_activity: float
    candidates: list = field(default_factory=list)


class SecretaryBot(discord.Client):
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm,
        guilds: GuildStore | None = None,
        clock=time.monotonic,
    ) -> None:
        intents = discord.Intents(
            guilds=True,
            messages=True,
            message_content=True,
            members=True,
            dm_messages=True,
            voice_states=True,
        )
        super().__init__(intents=intents)
        self.s, self.store, self.llm = settings, store, llm
        # Per-guild settings document (Firestore, or in memory for local dev).
        self.guilds_store = guilds or make_guild_store(settings)
        self.team = BotGuildSettings(self.guilds_store)
        self.archive = make_archive(settings)
        self.tree = app_commands.CommandTree(self)
        self._clock = clock  # injectable so conversation timers are testable without sleeping
        self._convos: dict[int, _Convo] = {}  # channel -> running conversation
        self._pending: dict[int, _Pending] = {}  # join mode: channel -> candidates
        self._interjections: dict[int, deque[float]] = {}  # join mode budget, per channel
        self._snoozed: dict[int, float] = {}  # channel -> snooze end (in memory only)
        self._tick_task: asyncio.Task | None = None
        self._digest_task: asyncio.Task | None = None
        self._digest_sent: dict[int, date] = {}
        self._deep_tasks: set[asyncio.Task] = set()  # strong refs so tasks are not GC'd
        self._sync_last: dict[int, float] = {}
        self._sync_pending: dict[int, asyncio.Task] = {}
        self.meetings = MeetingManager(self)

    async def setup_hook(self):
        self._register_commands()
        try:
            await self.tree.sync()  # global commands; Discord may take a while to show them
        except discord.HTTPException:
            logger.exception("could not sync slash commands")
        self._digest_task = asyncio.create_task(self._digest_loop())
        self._tick_task = asyncio.create_task(self._tick_loop())

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
        chat = await self._ensure_chat_channel(guild)
        if chat is not None:  # a reused channel must answer everything, whatever its name
            try:
                await asyncio.to_thread(self.team.set_channel_mode, guild.id, chat.id, "chat")
            except Exception:
                logger.exception("could not save chat mode for guild %s", guild.id)
        await self._sync_guild(guild, installed=True)
        target = chat or guild.system_channel
        if target is None or not target.permissions_for(guild.me).send_messages:
            target = next(
                (c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None
            )
        if target is not None:
            where = chat.mention if chat else f"#{CHAT_CHANNEL_NAME}"
            with contextlib.suppress(discord.HTTPException):
                await target.send(WELCOME_MESSAGE.format(mention=self.user.mention, chat=where))

    async def _ensure_chat_channel(self, guild: discord.Guild):
        """The guild's Salid chat channel: an existing one with its name, else a new one."""
        existing = next((c for c in guild.text_channels if is_chat_channel_name(c.name)), None)
        if existing is not None:
            return existing
        perms = getattr(guild.me, "guild_permissions", None)
        if not getattr(perms, "manage_channels", False):
            logger.info("no Manage Channels in guild %s: chat channel not created", guild.id)
            return None
        try:
            return await guild.create_text_channel(
                CHAT_CHANNEL_NAME, topic=CHAT_CHANNEL_TOPIC, reason="Salid chat channel"
            )
        except discord.HTTPException:
            logger.warning("could not create chat channel in guild %s", guild.id, exc_info=True)
            return None

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
            mode = self._mode(message.channel, gid)
            receipts_on = self.team.enabled(gid, "receipts")
            if receipts_on and await self._maybe_correction(message):
                return
            if mode == "off":
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
                    await self._intake(message, quiet=mode == "listen")
                return
            if mentioned and await self._meeting_command(message):
                return
            if mentioned and await self._snooze_command(message):
                return
            if mode == "listen":
                if mentioned:
                    with contextlib.suppress(discord.HTTPException):
                        await message.add_reaction("👂")
                return
            if mode == "chat":
                await self._answer(message, message.guild, message.author)
                return
            await self._call_or_join(message, mode, mentioned)
        except Exception:
            logger.exception("handler failed")

    def _mode(self, channel, guild_id: int) -> str:
        """The channel's mode now: the resolved setting, or listen while snoozed."""
        mode = self.team.channel_mode(
            guild_id, channel.id, getattr(channel, "name", None), self.s.ask_channel_ids
        )
        return "listen" if mode != "off" and self._is_snoozed(channel.id) else mode

    def _is_snoozed(self, channel_id: int) -> bool:
        until = self._snoozed.get(channel_id)
        if until is not None and self._clock() >= until:
            del self._snoozed[channel_id]
            return False
        return until is not None

    async def _call_or_join(self, message: discord.Message, mode: str, mentioned: bool) -> None:
        """call/join: answer when called by tag or name, replied to, or (inside the conversation
        window) when the gate says the message is meant for Salid. Join mode also queues
        questions for the quiet-time batch gate."""
        cid = message.channel.id
        if mentioned or mentions_name(message.content):
            await self._answer(message, message.guild, message.author)
            return
        if self._replying_to_salid(message):
            await self._answer(message, message.guild, message.author)
            return
        if mentions_other(message, self.user):
            return  # talking to someone else
        if self._window_open(cid) and await self._gate_window(message):
            await self._answer(message, message.guild, message.author)
            return
        if mode != "join":
            return
        if message.content and is_join_candidate(message.content):
            pending = self._pending.get(cid)
            if pending is None:
                pending = self._pending[cid] = _Pending(message.channel, message.guild, 0.0)
            pending.candidates = [*pending.candidates, message][-MAX_JOIN_CANDIDATES:]
        if cid in self._pending:
            self._pending[cid].last_activity = self._clock()

    def _replying_to_salid(self, message: discord.Message) -> bool:
        ref = message.reference
        resolved = getattr(ref, "resolved", None) if ref else None
        return self.user is not None and getattr(resolved, "author", None) == self.user

    def _window_open(self, channel_id: int) -> bool:
        convo = self._convos.get(channel_id)
        return convo is not None and self._clock() - convo.last < CONVERSATION_SECONDS

    async def _gate_window(self, message: discord.Message) -> bool:
        history = await self._gate_lines(message.channel, GATE_HISTORY, before=message)
        line = GateLine(
            str(message.id),
            message.author.display_name,
            self._strip_mentions(message.content, message),
        )
        return await asyncio.to_thread(
            gate.addressed_to_salid,
            self.llm,
            self.s.gate_model,
            history,
            line,
            channel_id=message.channel.id,
        )

    async def _gate_lines(self, channel, limit: int, before=None) -> list[GateLine]:
        """The channel's last lines, oldest first, for a gate prompt."""
        history = getattr(channel, "history", None)
        if history is None:
            return []
        lines: list[GateLine] = []
        try:
            kw = {"limit": limit} if before is None else {"limit": limit, "before": before}
            async for prev in history(**kw):
                text = self._strip_mentions(prev.content, prev)
                if text:
                    who = "สลิด" if prev.author == self.user else prev.author.display_name
                    lines.append(GateLine(str(prev.id), who, text))
        except discord.HTTPException:
            logger.warning("could not read channel history", exc_info=True)
        return lines[::-1]

    async def _snooze_command(self, message: discord.Message) -> bool:
        """'@Salid เงียบก่อน' (listen mode for an hour) / '@Salid คุยได้แล้ว'. Open to everyone."""
        cmd = parse_snooze_command(self._strip_mentions(message.content, message))
        if cmd is None:
            return False
        cid = message.channel.id
        if cmd == "snooze":
            self._snoozed[cid] = self._clock() + SNOOZE_SECONDS
            self._convos.pop(cid, None)
            self._pending.pop(cid, None)
            text = SNOOZE_MESSAGE
        else:
            text = (
                UNSNOOZE_MESSAGE
                if self._snoozed.pop(cid, None) is not None
                else NOT_SNOOZED_MESSAGE
            )
        await message.reply(text, mention_author=False)
        return True

    async def _meeting_command(self, message: discord.Message) -> bool:
        """'@Salid เข้าประชุม' / 'ออกประชุม' typed in a text channel."""
        cmd = parse_meeting_command(self._strip_mentions(message.content, message))
        if cmd is None or not hasattr(message.author, "voice"):
            return False
        if cmd == "join":
            reply = await self.meetings.join(message.author, fallback_channel=message.channel)
        else:
            reply = await self.meetings.leave(message.author)
        await message.reply(reply, mention_author=False)
        return True

    async def on_voice_state_update(self, member, before, after):
        """Salid itself was disconnected (kicked, channel deleted): end that meeting."""
        if member == self.user and after.channel is None and before.channel is not None:
            await self.meetings.end(before.channel.guild.id, "disconnected")

    async def close(self):
        if self._tick_task is not None:
            self._tick_task.cancel()
        await self.meetings.shutdown()
        await super().close()

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

    async def _intake(self, message: discord.Message, quiet: bool = False):
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
            if quiet:  # listen mode: reaction only, no text reply
                continue
            sent = await message.reply(res.reply, mention_author=False)
            if res.status == "created":
                self.store.set_reply_message(res.receipt.id, sent.id)

    def _note_answer(self, message: discord.Message, sent) -> None:
        """Salid answered here: open or refresh the channel's conversation (call/join only)."""
        if message.guild is None:
            return
        cid = message.channel.id
        self._pending.pop(cid, None)  # whatever was waiting has just been dealt with
        if self._mode(message.channel, message.guild.id) not in ("call", "join"):
            return
        convo = self._convos.get(cid)
        if convo is None or not self._window_open(cid):
            convo = self._convos[cid] = _Convo(message.channel, self._clock())
        convo.replies += 1
        convo.last, convo.last_reply = self._clock(), sent

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
            sent = await message.reply(GREETING_MESSAGE, mention_author=False)
            self._note_answer(message, sent)
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
        meeting = self.meetings.context_for(message.channel.id)
        if meeting:  # talking in a live meeting's chat: the spoken transcript is context too
            history = f"บันทึกการประชุมล่าสุด (ใครพูดอะไร):\n{meeting}\n\nแชต:\n{history}".rstrip()
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
        sent = None
        for i, part in enumerate(chunks(ans.text)):
            sent = await (
                message.reply(part, mention_author=False) if i == 0 else message.channel.send(part)
            )
        self._note_answer(message, sent)

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

        async def mode_cb(interaction: discord.Interaction, mode: str | None = None):
            await self.handle_mode(interaction, mode)

        mode_cb = app_commands.guild_only()(mode_cb)
        mode_cb = app_commands.describe(mode="โหมดใหม่ของห้องนี้ (ไม่เลือก = ดูโหมดปัจจุบัน)")(mode_cb)
        mode_cb = app_commands.choices(
            mode=[app_commands.Choice(name=MODE_LABELS[m], value=m) for m in MODES]
        )(mode_cb)
        for name in MODE_COMMANDS:
            self.tree.add_command(
                app_commands.Command(
                    name=name,
                    description="ดูหรือตั้งโหมดการคุยของสลิดในห้องนี้",
                    callback=mode_cb,
                )
            )
        for names, handler, desc in (
            (JOIN_COMMANDS, self.handle_join, "ให้สลิดเข้าห้องเสียงที่คุณอยู่ เพื่อฟังและจดบันทึกประชุม"),
            (LEAVE_COMMANDS, self.handle_leave, "ให้สลิดออกจากห้องเสียงและสรุปการประชุม"),
        ):
            cb = self._command_callback(handler)
            for name in names:
                self.tree.add_command(
                    app_commands.Command(name=name, description=desc, callback=cb)
                )

    @staticmethod
    def _command_callback(handler):
        async def cb(interaction: discord.Interaction):
            await handler(interaction)

        return app_commands.guild_only()(cb)

    async def handle_join(self, interaction: discord.Interaction) -> None:
        """/เข้าประชุม: join the invoker's voice channel and start the meeting."""
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.team.current, interaction.guild.id)
        reply = await self.meetings.join(interaction.user, fallback_channel=interaction.channel)
        await interaction.followup.send(reply, ephemeral=True)

    async def handle_leave(self, interaction: discord.Interaction) -> None:
        """/ออกประชุม: leave and post the summary."""
        await interaction.response.send_message(
            await self.meetings.leave(interaction.user), ephemeral=True
        )

    async def handle_mode(self, interaction: discord.Interaction, mode: str | None) -> None:
        """/โหมด: show this channel's mode, or (executives / server managers) change it."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        send = interaction.response.send_message
        if guild is None or channel is None:
            await send(SETUP_DENIED_MESSAGE, ephemeral=True)
            return
        await asyncio.to_thread(self.team.current, guild.id)
        if mode is None:
            current = self._mode(channel, guild.id)
            lines = [f"โหมดของห้องนี้ตอนนี้: {MODE_LABELS[current]}", ""]
            if self._is_snoozed(channel.id):
                lines += ["(สลิดกำลังพักเงียบชั่วคราว ทำตัวเหมือนโหมดฟังเงียบ)", ""]
            lines += [f"{MODE_LABELS[m]} — {MODE_HELP[m]}" for m in MODES]
            lines += ["", "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /โหมด แล้วเลือกโหมดครับ"]
            await send("\n".join(lines), ephemeral=True)
            return
        if mode not in MODES:
            await send(MODE_FAILED_MESSAGE, ephemeral=True)
            return
        if not hasattr(user, "guild_permissions") or not (
            can_manage_guild(user) or self.is_exec_member(user)
        ):
            await send(MODE_DENIED_MESSAGE, ephemeral=True)
            return
        try:
            await asyncio.to_thread(self.team.set_channel_mode, guild.id, channel.id, mode)
        except Exception:
            logger.exception("could not save channel mode")
            await send(MODE_FAILED_MESSAGE, ephemeral=True)
            return
        self._snoozed.pop(channel.id, None)
        self._convos.pop(channel.id, None)
        self._pending.pop(channel.id, None)
        await send(f"ตั้งห้องนี้เป็น {MODE_LABELS[mode]} แล้วครับ")

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

    # ------------------------------------------------------------- timers (sign-off, join batches)
    async def _tick_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(TICK_SECONDS)
            try:
                await self.tick()
            except Exception:
                logger.exception("tick failed")

    async def tick(self) -> None:
        await self._signoff_due()
        await self._join_due()

    async def _signoff_due(self) -> None:
        """Conversations whose window ran out: one short sign-off (or a 👋 reaction after a
        single answer), once."""
        now = self._clock()
        for cid, convo in list(self._convos.items()):
            if now - convo.last < CONVERSATION_SECONDS:
                continue
            del self._convos[cid]
            try:
                if convo.replies >= SIGNOFF_MIN_REPLIES:
                    await convo.channel.send(SIGNOFF_MESSAGE)
                elif convo.last_reply is not None:
                    await convo.last_reply.add_reaction("👋")
            except discord.HTTPException:
                logger.warning("could not sign off in channel %s", cid, exc_info=True)

    def _interjection_allowed(self, channel_id: int) -> bool:
        used = self._interjections.setdefault(channel_id, deque())
        while used and self._clock() - used[0] >= 3600:
            used.popleft()
        return len(used) < MAX_INTERJECTIONS_PER_HOUR

    async def _join_due(self) -> None:
        """Join mode: channels that went quiet get one batch-gate call over their candidates."""
        now = self._clock()
        for cid, pending in list(self._pending.items()):
            if now - pending.last_activity < QUIET_SECONDS:
                continue
            del self._pending[cid]
            if not pending.candidates or self._mode(pending.channel, pending.guild.id) != "join":
                continue
            if not self._interjection_allowed(cid):  # budget first: no tokens when out of budget
                logger.info("join budget used up in channel %s", cid)
                continue
            lines = await self._gate_lines(pending.channel, JOIN_HISTORY)
            ids = [str(m.id) for m in pending.candidates]
            picked = await asyncio.to_thread(
                gate.pick_interjection, self.llm, self.s.gate_model, lines, ids, channel_id=cid
            )
            target = next((m for m in pending.candidates if str(m.id) == picked), None)
            if target is None:
                continue
            self._interjections[cid].append(self._clock())
            try:
                await self._answer(target, pending.guild, target.author)
            except Exception:
                logger.exception("join answer failed")

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
