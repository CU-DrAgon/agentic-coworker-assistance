from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import discord
from discord import app_commands

from . import gate
from .archive import make_archive
from .channel_access import access_flags, hidden_channels, hidden_channels_messages
from .config import Settings
from .deep_task import (
    REPORT_NAME,
    DeepCancelled,
    DeepHandle,
    DeepTaskError,
    parse_deep_trigger,
    reply_parts,
    run_deep_task,
)
from .digest import build_daily_digest
from .docs import doc_kind
from .email_out import make_sender
from .extract import (
    apply_correction,
    apply_void_command,
    guess_mime,
    ingest_attachment,
    parse_void_command,
)
from .gate import GateLine, is_join_candidate
from .guild_settings import (
    MAX_RETENTION_DAYS,
    MIN_RETENTION_DAYS,
    BotGuildSettings,
    GuildStore,
    make_guild_store,
    valid_retention_days,
)
from .i18n import LANGS, LANGUAGE_LABELS, parse_language_command, resolve_lang, t
from .image_memory import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    ImageNote,
    describe_image,
    format_note,
    image_mime,
    receipt_note,
)
from .media import media_candidates
from .media_bot import MediaMixin
from .meeting import MeetingManager, parse_meeting_command
from .memory import (
    MemoryScheduler,
    MemoryService,
    build_message,
    describe_attachment,
    message_text,
    synthetic_id,
)
from .memory_access import ChannelAccess, allowed_channel_ids
from .memory_import import MemoryImporter, Target, import_progress
from .memory_store import MemoryStore, SQLiteMemoryStore, make_memory_store
from .memory_summary import render_summary
from .message_control import AnswerLog, parse_cancel_task, parse_delete_message
from .modes import (
    CHAT_CHANNEL_NAME,
    MODES,
    is_chat_channel_name,
    mode_choice_name,
    mode_help,
    mode_label,
    parse_snooze_command,
)
from .naming import command_text, mentions_name
from .qa import FileActions, FileInput, MemoryTools, ask, file_mime, is_no_reply, select_files
from .receipt_forward import forward_receipt
from .scheduling import DiscordCalendar, MeetingService
from .secretary import SecretaryContext, SecretaryState
from .secretary_store import SecretaryStore, make_secretary_store
from .store import Store
from .summary import (
    PERIODS,
    is_summarize_request,
    message_summary_question,
    summary_question,
)

logger = logging.getLogger(__name__)
ATTACH_LIMIT = 15 * 1024 * 1024
SYNC_MIN_INTERVAL_S = 30.0  # channel-list writes per guild are batched to one per interval
MAX_SYNCED_CHANNELS = 500
RECEIPT_CHANNEL_HINTS = (
    "ใบเสร็จ", "receipt", "invoice", "ใบกำกับ", "สลิป", "slip", "บิล", "bill", "expense",
    "ค่าใช้จ่าย", "เบิก",
)  # fmt: skip  # a channel whose name has one of these takes receipts (case-insensitive)
# CHAT_CHANNEL_NAME (salid.modes) is created on join (needs Manage Channels): mode "chat".
# A bare @mention right after the same person's message is treated as a question about it.
FOLLOWUP_WINDOW_SECONDS = 120
# After Salid answers in a channel, messages there need no @mention for this long (a gate decides
# whether each one is meant for Salid). Silence for this long ends the conversation.
CONVERSATION_SECONDS = 180
SIGNOFF_MIN_REPLIES = 2  # fewer replies: only a 👋 reaction on Salid's last reply
GATE_HISTORY = 10  # channel lines the in-window gate reads
JOIN_HISTORY = 15  # channel lines the join-mode batch gate reads
QUIET_SECONDS = 30  # join mode: a channel must be this quiet before the batch gate runs
MAX_INTERJECTIONS_PER_HOUR = 3  # join mode: unprompted answers per channel, rolling hour
MAX_JOIN_CANDIDATES = 5
INTERJECT_SAME_USER_COOLDOWN = 600  # join mode: no second unprompted answer to one person
SNOOZE_SECONDS = 3600
TICK_SECONDS = 5.0
SCHEDULER_TICK_SECONDS = 15.0  # how often meeting reminders are checked
MEMORY_TICK_SECONDS = 30.0  # how often the memory scheduler looks for due summary updates
MEMORY_QUEUE_MAX = 20000
IMPORT_PAGE_SLEEP_S = 1.0  # pause between history pages (discord.py also rate-limits)
IMPORT_DEBOUNCE_S = 5.0  # access events within this window are folded into one import run
IMPORT_RESUME_S = 600.0  # a run stopped by the guild cap is resumed this much later
IMAGE_NOTES_KEPT = 500  # recent messages whose image notes are merged in memory
VIEWERS_TTL_S = 60.0  # cache of "who can read channel X" used for memory access
MODE_COMMANDS = ("โหมด", "mode")
MEMORY_COMMANDS = ("ความจำ", "memory")
LANGUAGE_COMMANDS = ("ภาษา", "language")
# Chat context handed to the model: this person's and Salid's recent messages in the channel.
# Conversation context for an answer: the whole room (everyone, not only the asker), so Salid
# follows group chat like a person reading back the channel.
HISTORY_MESSAGES = 30
HISTORY_SECONDS = 3 * 3600
HISTORY_CHARS = 8000  # oldest lines are dropped first
HISTORY_LINE_CHARS = 500
CHANNELS_COMMANDS = ("ห้อง", "channels")
SETUP_COMMANDS = ("ตั้งค่า", "setup")  # Thai name plus an English alias
JOIN_COMMANDS = ("เข้าประชุม", "join-meeting")
LEAVE_COMMANDS = ("ออกประชุม", "leave-meeting")
CANCEL_COMMANDS = ("ยกเลิก", "cancel")
SUMMARY_COMMANDS = ("สรุป", "summary")
MENU_SUMMARIZE = "Summarize with Salid"  # context menu; Thai name via SalidTranslator


def chunks(text: str, n: int = 1900):
    while text:
        yield text[:n]
        text = text[n:]


def own_receipt(is_receipt_post, receipts_on: bool) -> bool:
    """True when the receipt pipeline will read this post's images (no second reading)."""
    return bool(is_receipt_post) and receipts_on


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


class SalidTranslator(app_commands.Translator):
    """Localizes the message context menu name (everything else is bilingual in its text)."""

    async def translate(self, string, locale, context):
        if str(string) == MENU_SUMMARIZE and str(locale.value).startswith("th"):
            return t("menu_summarize", "th")
        return None


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


@dataclass
class _DeepJob:
    task: asyncio.Task
    handle: DeepHandle
    starter_id: int


class SecretaryBot(MediaMixin, discord.Client):
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm,
        guilds: GuildStore | None = None,
        clock=time.monotonic,
        memory: MemoryStore | None = None,
        secretary_store: SecretaryStore | None = None,
    ) -> None:
        intents = discord.Intents(
            guilds=True,
            messages=True,
            message_content=True,
            members=True,
            dm_messages=True,
            voice_states=True,
            guild_scheduled_events=True,
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
        self._interjected_at: dict[tuple[int, int], float] = {}  # (channel, user) -> last time
        self._snoozed: dict[int, float] = {}  # channel -> snooze end (in memory only)
        self._tick_task: asyncio.Task | None = None
        self._digest_task: asyncio.Task | None = None
        self._digest_sent: dict[int, date] = {}
        self._agent_msgs: set[int] = set()  # messages whose files went to the agent
        self._deep_tasks: set[asyncio.Task] = set()  # strong refs so tasks are not GC'd
        self._deep_jobs: dict[int, _DeepJob] = {}  # guild -> the running deep task
        self._answers = AnswerLog(clock)  # chunk ids of Salid's answers, ~24 h
        self._sync_last: dict[int, float] = {}
        self._sync_pending: dict[int, asyncio.Task] = {}
        self.meetings = MeetingManager(self)
        # Secretary: scheduled meetings, PDFs, email (see salid.secretary / scheduling / email_out)
        self.secretary_store = secretary_store or make_secretary_store(settings, store)
        self.secretary_state = SecretaryState()
        self.email_sender = make_sender(settings)
        self.scheduler = MeetingService(
            self.secretary_store,
            DiscordCalendar(),
            settings,
            self.team,
            self.meetings,
            self.get_guild,
        )
        self._sched_last = float("-inf")
        # Organisational memory: raw log + day summaries (see salid.memory).
        self.memory_store = memory or self._default_memory_store(settings, store)
        self.memory = MemoryService(
            self.memory_store,
            settings,
            MemoryScheduler(
                self.memory_store,
                settings,
                llm,
                now=settings.now,
                lang_of=lambda gid: resolve_lang(self.team.current(gid).default_language),
                retention_of=self.team.memory_retention_days,
                channel_name_of=self._channel_name,
                enabled=lambda gid: self.team.enabled(gid, "memory"),
            ),
        )
        self._memory_queue: asyncio.Queue | None = None
        self._memory_worker: asyncio.Task | None = None
        self._memory_last = float("-inf")
        self._import_task: asyncio.Task | None = None
        self._import_requested = False
        self._import_debounce: asyncio.Task | None = None
        self._import_resume_at: float | None = None
        self._memory_cfg_seen: dict[int, tuple[bool, int]] = {}  # guild -> (memory on, retention)
        self._bg_tasks: set[asyncio.Task] = set()
        self._image_sem = asyncio.Semaphore(2)  # at most two images described at once
        self._image_notes: dict[int, dict[int, str]] = {}  # message id -> attachment idx -> note
        self.image_stats = {"described": 0, "failed": 0, "skipped": 0}  # cost visibility
        self._init_media()
        self.importer = MemoryImporter(
            self.memory_store,
            settings,
            self.memory,
            llm,
            build_record=self._import_record,
            retention_of=self.team.memory_retention_days,
            lang_of=lambda gid: resolve_lang(self.team.current(gid).default_language),
            channel_name_of=self._channel_name,
            now=settings.now,
            page_sleep=IMPORT_PAGE_SLEEP_S,
        )
        self._seen_mode: dict[int, str] = {}  # channel -> last mode seen (off -> on notice)
        self._viewers: dict[int, tuple[float, frozenset[int]]] = {}

    @staticmethod
    def _default_memory_store(settings: Settings, store) -> MemoryStore:
        if settings.store == "firestore":
            return make_memory_store(settings, store)
        return SQLiteMemoryStore(getattr(store, "path", None) or ":memory:", settings.zone)

    def _channel_name(self, guild_id: int, channel_id: int) -> str:
        guild = next((g for g in self.guilds if g.id == guild_id), None)
        channel = guild.get_channel(channel_id) if guild else None
        return getattr(channel, "name", None) or str(channel_id)

    async def setup_hook(self):
        self._register_commands()
        await self.tree.set_translator(SalidTranslator())
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
        self._start_import()

    async def on_resumed(self):
        self._start_import()  # a gateway outage may have dropped events

    # ------------------------------------------------------------- guild lifecycle
    @staticmethod
    def _guild_snapshot(guild: discord.Guild) -> dict:
        return {
            "name": guild.name,
            "icon_url": guild.icon.url if guild.icon else "",
            "channels": [
                {
                    "id": str(c.id),
                    "name": c.name,
                    "visible": flags[0],
                    "readable_history": flags[1],
                }
                for c in guild.text_channels
                for flags in [access_flags(c, getattr(guild, "me", None))]
            ][:MAX_SYNCED_CHANNELS],
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
        await asyncio.to_thread(self.team.current, guild.id)
        lang = resolve_lang(self.team.current(guild.id).default_language)
        chat = await self._ensure_chat_channel(guild, lang)
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
                await target.send(
                    t(
                        "welcome",
                        lang,
                        mention=self.user.mention,
                        chat=where,
                        days=self.team.memory_retention_days(guild.id),
                    )
                )

    async def _ensure_chat_channel(self, guild: discord.Guild, lang: str = "th"):
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
                CHAT_CHANNEL_NAME, topic=t("chat_topic", lang), reason="Salid chat channel"
            )
        except discord.HTTPException:
            logger.warning("could not create chat channel in guild %s", guild.id, exc_info=True)
            return None

    async def on_guild_remove(self, guild: discord.Guild):
        # Only marks the guild inactive; like the ledger, its memory is kept until retention
        # expires it. `self.memory.delete_guild(guild_id)` wipes it at once if ever wanted.
        logger.info("removed from guild %s", guild.id)
        try:
            await asyncio.to_thread(self.guilds_store.set_active, guild.id, False)
        except Exception:
            logger.exception("could not mark guild %s inactive", guild.id)

    async def on_guild_channel_create(self, channel):
        self._request_sync(channel.guild)
        self._request_import()

    async def on_guild_channel_update(self, before, after):
        if before.name != after.name:
            self._request_sync(after.guild)
        if before.overwrites != after.overwrites:  # permissions changed: maybe newly readable
            self._request_sync(after.guild)
            self._request_import()

    async def on_member_update(self, before, after):
        if self.user is not None and after.id == self.user.id and before.roles != after.roles:
            self._request_sync(after.guild)  # the owner changed Salid's roles
            self._request_import()

    async def on_guild_role_update(self, before, after):
        me = getattr(after.guild, "me", None)
        if me is not None and after in me.roles and before.permissions != after.permissions:
            self._request_sync(after.guild)
            self._request_import()

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
        own = message.author == self.user
        logged = False
        if message.guild is not None and (own or not getattr(message.author, "bot", False)):
            logged = await self._memory_record(message, own)
        if own or getattr(message.author, "bot", False):
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
            if await self._maybe_delete_own(message):
                return
            mentioned = mentions_bot(message, self.user)
            receipt_ids = self.team.receipt_channel_ids(gid, self.s.receipt_channel_ids)
            qa_on = self.team.enabled(gid, "qa")
            # Receipt channels read receipts directly (reliable, quiet in listen mode). Elsewhere a
            # message to Salid with files goes to the agent, which decides what to do; only with
            # Q&A off does "@Salid + receipt file" still mean direct intake.
            is_receipt_post = message.attachments and (
                message.channel.id in receipt_ids
                or is_receipt_channel_name(getattr(message.channel, "name", ""))
                or (mentioned and has_receipt_file(message) and not qa_on)
            )
            describe = not own_receipt(is_receipt_post, receipts_on)
            try:
                await self._route(
                    message, mode, mentioned, bool(is_receipt_post), receipts_on, qa_on
                )
            finally:
                # images -> text for memory (and receipts anywhere nobody is talking to Salid),
                # in the background. A message the agent read is not run through receipt intake.
                self._media_soon(message, logged)
                self._doc_soon(message, logged)
                if describe:
                    to_agent = message.id in self._agent_msgs
                    self._agent_msgs.discard(message.id)
                    self._describe_soon(
                        message,
                        logged,
                        intake_mode=mode
                        if receipts_on and not is_receipt_post and not to_agent
                        else None,
                    )
        except Exception:
            logger.exception("handler failed")

    async def _route(
        self,
        message: discord.Message,
        mode: str,
        mentioned: bool,
        is_receipt_post: bool,
        receipts_on: bool,
        qa_on: bool,
    ) -> None:
        gid = message.guild.id
        if is_receipt_post:
            if receipts_on:
                await self._intake(message, quiet=mode == "listen")
            return
        called = mentioned or (self._starts_with_name(message))
        if called and await self._meeting_command(message):
            return
        if called and await self._snooze_command(message):
            return
        if called and await self._cancel_task_command(message):
            return
        if called and await self._language_command(message):
            return
        # Q&A off and no image describer: images are receipt candidates when Salid is addressed
        if (
            receipts_on
            and not qa_on
            and self._image_candidates(message)
            and not self._images_enabled(gid)
            and (mode == "chat" or (mode in ("call", "join") and called))
            and await self._intake(message)
        ):
            return
        if mode == "listen":
            if mentioned and self._is_snoozed(message.channel.id):
                with contextlib.suppress(discord.HTTPException):
                    await message.add_reaction("👂")
            elif mentioned:  # listen answers a tag, never a bare name, never unprompted
                await self._answer(message, message.guild, message.author)
            return
        if mode == "chat":
            await self._answer(message, message.guild, message.author)
            return
        await self._call_or_join(message, mode, mentioned)

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

    # ------------------------------------------------------------- reply language
    def _lang_setting(self, target) -> str:
        """th / en / auto for the channel of `target` (a message or a channel). DMs: auto."""
        guild = getattr(target, "guild", None)
        if guild is None:
            return "auto"
        channel = getattr(target, "channel", None) or target
        return self.team.channel_language(guild.id, getattr(channel, "id", 0))

    def _lang(self, target, text: str | None = None) -> str:
        """Language (th or en) for Salid's fixed replies to `target`. `auto` follows `text`, by
        default the message's own words."""
        if text is None and hasattr(target, "content"):
            text = self._strip_mentions(target.content or "", target)
        return resolve_lang(self._lang_setting(target), text)

    def _interaction_lang(self, interaction) -> str:
        """Language for a slash command reply: the channel's setting; `auto` has no message to
        follow, so the user's Discord locale (Thai, else English; Thai when unknown)."""
        setting = self._lang_setting(interaction)
        if setting != "auto":
            return setting
        locale = str(getattr(interaction, "locale", "") or "")
        return "en" if locale and not locale.startswith("th") else "th"

    def _starts_with_name(self, message: discord.Message) -> bool:
        """The message opens with Salid's name (no @tag): "Salid เข้าประชุมหน่อย"."""
        return command_text(self._strip_mentions(message.content or "", message))[1]

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

    @staticmethod
    def _files_line(message: discord.Message) -> str:
        """Gate text of a message that is only attachments."""
        return " ".join(f"[{a.filename}]" for a in message.attachments)

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
            self._strip_mentions(message.content, message) or self._files_line(message),
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

    def _command_text(self, message: discord.Message) -> str:
        """Text for the typed-command parsers: tag, a leading name ("Salid ...") and trailing
        politeness (หน่อย / ครับ / please) removed."""
        return command_text(self._strip_mentions(message.content or "", message))[0]

    async def _snooze_command(self, message: discord.Message) -> bool:
        """'@Salid เงียบก่อน' / 'be quiet' (listen mode for an hour) and 'คุยได้แล้ว' / 'you can talk'
        again. Open to everyone."""
        cmd = parse_snooze_command(self._command_text(message))
        if cmd is None:
            return False
        cid = message.channel.id
        if cmd == "snooze":
            self._snoozed[cid] = self._clock() + SNOOZE_SECONDS
            self._convos.pop(cid, None)
            self._pending.pop(cid, None)
            key = "snooze"
        else:
            key = "unsnooze" if self._snoozed.pop(cid, None) is not None else "not_snoozed"
        await message.reply(t(key, self._lang(message)), mention_author=False)
        return True

    async def _language_command(self, message: discord.Message) -> bool:
        """'@Salid พูดภาษาอังกฤษ' / 'speak English' etc.: sets this channel's reply language.
        Executives and server managers only; others get a short refusal."""
        text = self._command_text(message)
        cmd = parse_language_command(text)
        if cmd is None or message.guild is None:
            return False
        user = message.author
        if not hasattr(user, "guild_permissions") or not (
            can_manage_guild(user) or self.is_exec_member(user)
        ):
            await message.reply(t("lang_denied", self._lang(message)), mention_author=False)
            return True
        new = resolve_lang(cmd, text)  # the reply follows the language just chosen
        try:
            await asyncio.to_thread(
                self.team.set_channel_language, message.guild.id, message.channel.id, cmd
            )
        except Exception:
            logger.exception("could not save channel language")
            await message.reply(t("lang_failed", new), mention_author=False)
            return True
        label = t(f"lang_label_{cmd}", new)
        await message.reply(t("lang_set", new, label=label), mention_author=False)
        return True

    async def _meeting_command(self, message: discord.Message) -> bool:
        """'@Salid เข้าประชุม' / 'join' / 'ออกประชุม' / 'leave' typed in a text channel."""
        cmd = parse_meeting_command(self._command_text(message))
        if cmd is None or not hasattr(message.author, "voice"):
            return False
        lang = self._lang(message)
        if cmd == "join":
            reply = await self.meetings.join(
                message.author, fallback_channel=message.channel, lang=lang
            )
        else:
            reply = await self.meetings.leave(message.author, lang=lang)
        await message.reply(reply, mention_author=False)
        return True

    async def on_scheduled_event_create(self, event) -> None:
        """Somebody made a Discord event: track it (reminders, no-show logic for voice events)."""
        await self._sched_event(self.scheduler.track_discord_event, event)

    async def on_scheduled_event_update(self, before, after) -> None:
        await self._sched_event(self.scheduler.update_discord_event, after)

    async def on_scheduled_event_delete(self, event) -> None:
        await self._sched_event(self.scheduler.delete_discord_event, event)

    @staticmethod
    async def _sched_event(fn, event) -> None:
        try:
            await fn(event)
        except Exception:
            logger.exception("could not track scheduled event")

    async def on_voice_state_update(self, member, before, after):
        """Salid itself was disconnected (kicked, channel deleted): end that meeting."""
        if member == self.user and after.channel is None and before.channel is not None:
            await self.meetings.end(before.channel.guild.id, "disconnected")

    async def close(self):
        if self._tick_task is not None:
            self._tick_task.cancel()
        if self._memory_worker is not None:
            self._memory_worker.cancel()
        await self.meetings.shutdown()
        await super().close()

    async def _direct_message(self, message: discord.Message):
        """DMs have no guild: answer only when the asker shares exactly one guild with Salid."""
        shared = [(g, g.get_member(message.author.id)) for g in self.guilds]
        shared = [(g, m) for g, m in shared if m is not None]
        if len(shared) != 1:
            await message.reply(
                t("dm", self._lang(message), mention=self.user.mention), mention_author=False
            )
            return
        guild, member = shared[0]
        await asyncio.to_thread(self.team.current, guild.id)
        ack = await self.scheduler.handle_dm_reply(message.author.id, message.content)
        if ack:  # the reason an attendee gave after a meeting reminder / no-show nudge
            await message.reply(ack, mention_author=False)
            return
        await self._answer(message, guild, member)

    async def _maybe_correction(self, message: discord.Message) -> bool:
        ref = message.reference
        if not ref or not ref.message_id or not message.content.strip():
            return False
        receipt = self.store.get_by_reply_message(ref.message_id)
        if not receipt or receipt.guild_id != message.guild.id:
            return False
        if message.author.id != receipt.submitter_id and not self.is_exec_member(message.author):
            await message.reply(t("correction_denied", self._lang(message)), mention_author=False)
            return True
        action = parse_void_command(message.content)
        delete_kind = parse_delete_message(self._command_text(message))
        if action is None and delete_kind is not None:
            action = "void"  # "remove"/"unsend" on a receipt message cancels the receipt
        if action:
            text = await asyncio.to_thread(
                apply_void_command,
                self.store,
                receipt,
                action,
                f"{message.author.display_name} ({message.author.id})",
                self._lang(message),
            )
            if action == "void" and delete_kind == "message":
                # voiding is the important part; the receipt message itself goes too
                await self._delete_messages(message.channel, [ref.message_id], message.guild.id)
                text = t("receipt_void_ack", self._lang(message), id=receipt.id)
            await message.reply(text, mention_author=False)
            return True
        async with message.channel.typing():
            text = await asyncio.to_thread(
                apply_correction,
                self.store,
                self.llm,
                self.s,
                receipt,
                message.content,
                self._lang(message),
            )
        await message.reply(text, mention_author=False)
        return True

    async def _ingest_one(self, message: discord.Message, att, data: bytes, quiet: bool):
        """Receipt pipeline for one attachment (bytes already read): ingest, reaction, reply."""
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
                note=message.content.strip(),
                archive=self.archive,
                lang=self._lang(message),
            )
        if res.status == "ignored":
            return res
        if res.reaction:
            with contextlib.suppress(discord.HTTPException):
                await message.add_reaction(res.reaction)
        fwd = await self._forward(
            message, message.guild, [(att.filename, data)], res, self._lang(message)
        )
        if not quiet:  # listen mode: reaction only, no text reply
            text = res.reply
            if fwd is not None:
                text += "\n" + t("fwd_ack", self._lang(message), home=fwd.channel.name)
            sent = await message.reply(text, mention_author=False)
            if res.status == "created":
                self.store.set_reply_message(res.receipt.id, sent.id)
        if fwd is not None and res.status == "created":  # replies in the home channel win
            self.store.set_reply_message(res.receipt.id, fwd.id)
        return res

    async def _forward(self, message, guild, files, res, lang):
        """Copy a created / duplicate receipt to the guild's receipt home channel when it was
        posted elsewhere (see salid.receipt_forward). The sent copy, or None."""
        if res.status not in ("created", "duplicate") or guild is None:
            return None
        try:
            return await forward_receipt(
                guild=guild,
                settings=self.team.current(guild.id),
                env_receipt_ids=self.s.receipt_channel_ids,
                name_matches=is_receipt_channel_name,
                source=message.channel if message.guild is not None else None,
                author=message.author,
                jump_url=getattr(message, "jump_url", None),
                files=files,
                text=res.reply,
                lang=lang,
            )
        except Exception:
            logger.exception("receipt forward failed")
            return None

    async def _intake(self, message: discord.Message, quiet: bool = False) -> bool:
        """Read every attachment as a receipt. True when at least one was a receipt (saved or a
        duplicate); images that are not receipts are described for memory instead."""
        found = False
        for idx, att in enumerate(message.attachments):
            if att.size > ATTACH_LIMIT:
                continue
            data = await att.read()
            res = await self._ingest_one(message, att, data, quiet)
            if res.receipt is not None and self.team.enabled(message.guild.id, "memory"):
                self._apply_notes(message, {idx: receipt_note(res.receipt)})
            elif (
                res.status == "ignored"
                and image_mime(att)
                and self._images_enabled(message.guild.id)
            ):  # not a receipt: describe it instead
                self._spawn(self._describe_message(message, [(idx, att)]))
            found = found or res.status != "ignored"
        return found

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
        """The room's recent conversation, oldest first: every participant (not only the asker)
        with local times, plus the message being replied to and today's summary of this channel
        when memory has one. Context only; the ledger and memory scopes are decided in code."""
        history = getattr(message.channel, "history", None)
        if history is None:
            return ""
        zone = self.s.zone
        lines: list[str] = []
        try:
            async for prev in history(limit=HISTORY_MESSAGES + 10, before=message):
                if (message.created_at - prev.created_at).total_seconds() > HISTORY_SECONDS:
                    break
                if getattr(prev.author, "bot", False) and prev.author != self.user:
                    continue
                text = self._strip_mentions(prev.content, prev)
                if not text and getattr(prev, "attachments", None):
                    text = " ".join(f"[{a.filename}]" for a in prev.attachments)
                if text:
                    who = "สลิด" if prev.author == self.user else prev.author.display_name
                    at = prev.created_at.astimezone(zone).strftime("%H:%M")
                    lines.append(f"[{at}] {who}: {text[:HISTORY_LINE_CHARS]}")
                if len(lines) >= HISTORY_MESSAGES:
                    break
        except discord.HTTPException:
            logger.warning("could not read channel history", exc_info=True)
        lines.reverse()
        while lines and sum(len(x) + 1 for x in lines) > HISTORY_CHARS:
            lines.pop(0)
        parts = []
        summary = await self._today_summary(message)
        if summary:
            parts.append(f"สรุปห้องนี้วันนี้ (ก่อนหน้าแชตด้านล่าง):\n{summary}")
        ref = getattr(message, "reference", None)
        replied = getattr(ref, "resolved", None) if ref else None
        if getattr(replied, "content", None):
            who = "สลิด" if replied.author == self.user else replied.author.display_name
            parts.append(f"ข้อความที่ผู้ถามกำลังตอบกลับ:\n{who}: {replied.content[:1000]}")
        if lines:
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    async def _today_summary(self, message: discord.Message) -> str:
        """Today's memory summary of this channel (free: already stored), or ''."""
        guild = getattr(message, "guild", None)
        if guild is None or not self.team.enabled(guild.id, "memory"):
            return ""
        try:
            day = datetime.now(self.s.zone).date().isoformat()
            channel = getattr(message.channel, "parent", None) or message.channel
            got = await asyncio.to_thread(self.memory_store.get_summary, guild.id, channel.id, day)
            return render_summary(got.summary, max_chars=1500) if got and got.summary else ""
        except Exception:
            logger.warning("could not read today's summary", exc_info=True)
            return ""

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

    async def _file_inputs(self, message: discord.Message) -> list[FileInput]:
        """The message's attachments as agent input, in order: readable ones downloaded (within
        the size limits of qa.select_files), the rest carrying the reason they are skipped."""
        atts = list(message.attachments)
        inputs = [
            FileInput(
                i,
                a.filename,
                file_mime(a.filename, getattr(a, "content_type", None)),
                size=int(getattr(a, "size", 0) or 0),
            )
            for i, a in enumerate(atts)
        ]
        media_idx = {i for i, _a in media_candidates(message)}
        office_idx = {
            i for i, a in enumerate(atts) if doc_kind(a) in ("docx", "xlsx", "pptx", "doc", "xls")
        }
        for i in media_idx | office_idx:  # read in code (transcript / text), not as raw bytes
            inputs[i].mime, inputs[i].size = "text/plain", 0
        usable, _skipped = select_files(inputs)
        for f in usable:
            if f.index in media_idx:
                async with message.channel.typing():
                    await self._fill_media_input(message, f, atts[f.index])
                continue
            if f.index in office_idx:
                await self._fill_office_input(f, atts[f.index])
                continue
            try:
                f.data = await atts[f.index].read()
                f.size = len(f.data)
            except Exception as e:  # never log contents
                logger.warning("attachment download failed (%s)", type(e).__name__)
                f.skip = "could not be downloaded"
        return inputs

    def _file_actions(
        self, message, guild, member, is_exec: bool, setting: str, loop, forwards=None
    ) -> FileActions:
        """The side effects the agent may trigger for this message. Guild, submitter, channel and
        message are bound here; the model only chooses which attachment."""
        gid = guild.id
        lang = self._lang(message)

        def record(f: FileInput, note: str):
            res = ingest_attachment(
                self.store,
                self.llm,
                self.s,
                guild_id=gid,
                data=f.data,
                filename=f.filename,
                content_type=f.mime,
                submitter_id=member.id,
                submitter_name=member.display_name,
                channel_id=message.channel.id,
                message_id=message.id,
                note=" ".join(x for x in ((message.content or "").strip(), note.strip()) if x),
                archive=self.archive,
                lang=lang,
            )
            if forwards is not None:
                forwards.append((f, res))  # posted to the receipt home channel by _answer
            return res

        def deep(task: str) -> str:
            status = self._deep_status(gid, task, is_exec)
            if status == "ok":
                loop.call_soon_threadsafe(self._launch_deep, message, gid, task, setting)
                return "started"
            return status

        return FileActions(
            record_receipt=record if self.team.enabled(gid, "receipts") else None,
            start_deep_task=deep if self.team.enabled(gid, "deep_tasks") else None,
        )

    async def _answer(
        self,
        message: discord.Message,
        guild: discord.Guild,
        member,
        *,
        interjection: bool = False,
    ) -> bool:
        """Answer `message`. `interjection`: Salid was not asked (join mode), so the model may
        stay silent (NO_REPLY). Attachments (images, PDFs, text files) go to the agent with the
        text, and it decides what to do with them (record a receipt, summarise, ...); not for an
        interjection, which was not addressed to Salid. Returns True when something was posted."""
        q = self._strip_mentions(message.content, message)
        with_files = bool(message.attachments) and not interjection
        if not q and not with_files:
            q = await self._previous_text(message)
        if not q and not with_files:
            sent = await message.reply(t("greeting", self._lang(message, "")), mention_author=False)
            self._note_answer(message, sent)
            return True
        gid = guild.id
        is_exec = self.is_exec_member(member)
        setting = self._lang_setting(message)  # th / en / auto: auto lets the model follow the user
        lang = resolve_lang(setting, q)
        task = parse_deep_trigger(q)
        if task is not None:
            if self.team.enabled(gid, "deep_tasks"):
                await self._deep(message, gid, task, is_exec, setting)
            else:
                await message.reply(t("deep_off", lang), mention_author=False)
            return True
        if not self.team.enabled(gid, "qa"):
            if interjection or (with_files and not q):  # a bare file is not worth a "Q&A is off"
                return False
            await message.reply(t("qa_off", lang), mention_author=False)
            return True
        if not interjection and is_summarize_request(q):
            target = await self._referenced_message(message)
            if target is not None:  # "สรุป" as a reply: summarise the message it answers
                return await self._reply_summary(message, guild, member, target, setting)
        history = await self._recent_chat(message)
        meeting = self.meetings.context_for(message.channel.id)
        if meeting:  # talking in a live meeting's chat: the spoken transcript is context too
            history = t("history_meeting", lang, meeting=meeting, chat=history).rstrip()
        memory = await self._memory_tools(message, guild, member)
        files: list[FileInput] = []
        forwards: list = []  # (file, IngestResult) of receipts the agent recorded
        actions = None
        if with_files:
            self._agent_msgs.add(message.id)  # the agent reads these: no separate receipt intake
            files = await self._file_inputs(message)
            actions = self._file_actions(
                message, guild, member, is_exec, setting, asyncio.get_running_loop(), forwards
            )
            q, files = self._voice_question(message, q, files)
            lang = resolve_lang(setting, q)
        secretary = (
            None if interjection else self._secretary(message, guild, member, is_exec, setting)
        )
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
                language=setting,
                memory=memory,
                interjection=interjection,
                **({"files": files, "actions": actions} if files else {}),
                **self._web_kw(gid),
                **({"secretary": secretary} if secretary is not None else {}),
            )
        logger.info("answered in %.1fs (%d rounds, exec=%s)", ans.seconds, ans.rounds, is_exec)
        if interjection and is_no_reply(ans.text):
            logger.info("interjection suppressed (nothing useful to add)")
            return False
        sent = first = None
        sent_ids: list[int] = []
        uploads = [
            discord.File(io.BytesIO(data), filename=name)
            for name, data in list(getattr(ans, "attachments", ()))[:10]
        ]
        for i, part in enumerate(chunks(ans.text)):
            extra = {"files": uploads} if i == 0 and uploads else {}
            sent = await (
                message.reply(part, mention_author=False, **extra)
                if i == 0
                else message.channel.send(part)
            )
            first = first or sent
            if getattr(sent, "id", None) is not None:
                sent_ids.append(sent.id)
        if sent_ids:
            self._answers.add(message.channel.id, message.author.id, sent_ids)
        lines = [
            receipt_note(r)
            for rid, _rx in getattr(ans, "receipts", ())
            if (r := self.store.get(rid)) is not None
        ]
        if lines:  # the document note reuses these: a receipt is never ingested twice
            self._agent_receipts[message.id] = lines
        for n, (receipt_id, reaction) in enumerate(getattr(ans, "receipts", ())):
            if n == 0 and first is not None:  # a correction by reply finds this receipt
                self.store.set_reply_message(receipt_id, first.id)
            if reaction:
                with contextlib.suppress(discord.HTTPException):
                    await message.add_reaction(reaction)
        forwarded = None
        for f, res in forwards[:4]:
            fwd = await self._forward(message, guild, [(f.filename, f.data)], res, lang)
            if fwd is not None and res.status == "created":
                self.store.set_reply_message(res.receipt.id, fwd.id)
            forwarded = forwarded or fwd
        if forwarded is not None:
            with contextlib.suppress(discord.HTTPException):
                await message.reply(
                    t("fwd_ack", lang, home=forwarded.channel.name), mention_author=False
                )
        self._note_answer(message, sent)
        return True

    def _secretary(self, message, guild, member, is_exec: bool, setting: str):
        """The secretary tools' bindings for this message (guild, author and channel are fixed
        here, never model input). Discord work is marshalled onto this loop."""
        gid = guild.id
        on = {f: self.team.enabled(gid, f) for f in ("scheduling", "documents", "email")}
        if not any(on.values()):
            return None
        loop = asyncio.get_running_loop()
        senders = set(self.team.current(gid).email_senders)
        can_email = is_exec or bool(senders & {r.name for r in member.roles})
        lang = self._lang(message)
        return SecretaryContext(
            settings=self.s,
            guild_id=gid,
            guild_name=getattr(guild, "name", ""),
            author_id=member.id,
            author_name=member.display_name,
            is_exec=is_exec,
            can_email=can_email,
            lang=lang,
            text_channel_id=message.channel.id if message.guild is not None else None,
            store=self.secretary_store,
            state=self.secretary_state,
            run=lambda coro: asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60),
            service=self.scheduler,
            sender=self.email_sender,
            documents=on["documents"],
            scheduling=on["scheduling"],
            email=on["email"],
        )

    def _web_kw(self, guild_id: int) -> dict:
        """read_web (links + Google Search) for `ask`: only with the `web` feature on."""
        if self.llm is not None and self.team.enabled(guild_id, "web"):
            return {"web_client": self.llm}
        return {}

    async def _referenced_message(self, message) -> discord.Message | None:
        """The message `message` is a reply to (None when it is not a reply or is gone)."""
        ref = getattr(message, "reference", None)
        if ref is None:
            return None
        found = getattr(ref, "resolved", None)
        if found is None and getattr(ref, "message_id", None):
            with contextlib.suppress(discord.HTTPException, AttributeError):
                found = await message.channel.fetch_message(ref.message_id)
        return found if getattr(found, "author", None) is not None else None

    async def _summarize_message(self, target, guild, member, setting: str) -> str | None:
        """Ask for a summary of one message: its text and links in the question, its attachments
        as file parts (no receipt/deep-task actions, no chat memory). None: nothing to summarise."""
        content = self._strip_mentions(getattr(target, "content", "") or "", target)
        files = await self._file_inputs(target) if getattr(target, "attachments", None) else []
        if not content and not files:
            return None
        lang = resolve_lang(setting, content)
        q = message_summary_question(lang, target.author.display_name, content)
        ans = await asyncio.to_thread(
            ask,
            self.s,
            self.store,
            q,
            guild_id=guild.id,
            is_exec=self.is_exec_member(member),
            user_id=member.id,
            language=setting,
            **({"files": files} if files else {}),
            **self._web_kw(guild.id),
        )
        logger.info("summarised a message in %.1fs", ans.seconds)
        return ans.text

    async def _reply_summary(self, message, guild, member, target, setting: str) -> bool:
        async with message.channel.typing():
            text = await self._summarize_message(target, guild, member, setting)
        if text is None:
            text = t("summary_nothing", resolve_lang(setting, message.content))
        sent = None
        for i, part in enumerate(chunks(text)):
            sent = await (
                message.reply(part, mention_author=False) if i == 0 else message.channel.send(part)
            )
        self._note_answer(message, sent)
        return True

    def _deep_status(self, guild_id: int, task: str, is_exec: bool) -> str:
        """Why a deep task cannot start now (not_executive / disabled / empty / busy), else ok."""
        if not is_exec:
            return "not_executive"
        if not self.s.gemini_cli_path:
            return "disabled"
        if not task:
            return "empty"
        if any(not job.done() for job in self._deep_tasks):
            return "busy"
        return "ok"

    def _launch_deep(self, message, guild_id: int, task: str, setting: str = "th") -> None:
        handle = DeepHandle()
        job = asyncio.create_task(self._run_deep(message, guild_id, task, setting, handle))
        self._deep_tasks.add(job)
        self._deep_jobs[guild_id] = _DeepJob(
            job, handle, getattr(getattr(message, "author", None), "id", None)
        )
        job.add_done_callback(self._deep_tasks.discard)
        job.add_done_callback(
            lambda _j, g=guild_id, me=job: (
                self._deep_jobs.pop(g, None)
                if getattr(self._deep_jobs.get(g), "task", None) is me
                else None
            )
        )

    async def _cancel_deep(self, guild_id: int, user, lang: str) -> str:
        """Stop the guild's running deep task (cancel the asyncio task, kill the CLI process).
        Allowed for whoever started it, server managers and executives."""
        job = self._deep_jobs.get(guild_id)
        if job is None or job.task.done():
            return t("cancel_none", lang)
        if user.id != job.starter_id and not (
            hasattr(user, "guild_permissions") and self.is_exec_member(user)
        ):
            return t("cancel_denied", lang)
        job.handle.cancel()
        job.task.cancel()
        return t("cancel_done", lang)

    async def _cancel_task_command(self, message: discord.Message) -> bool:
        """'@Salid ยกเลิกงาน' / 'cancel task'."""
        if message.guild is None or not parse_cancel_task(self._command_text(message)):
            return False
        reply = await self._cancel_deep(message.guild.id, message.author, self._lang(message))
        await message.reply(reply, mention_author=False)
        return True

    async def handle_cancel(self, interaction: discord.Interaction) -> None:
        """/ยกเลิก: cancel the running big task."""
        lang = self._interaction_lang(interaction)
        if interaction.guild is None:
            await interaction.response.send_message(t("cancel_none", lang), ephemeral=True)
            return
        reply = await self._cancel_deep(interaction.guild.id, interaction.user, lang)
        await interaction.response.send_message(reply, ephemeral=True)

    # ------------------------------------------------------------- deleting Salid's messages
    async def _delete_messages(self, channel, ids: list[int], guild_id: int) -> None:
        for mid in ids:
            with contextlib.suppress(discord.HTTPException):
                await channel.get_partial_message(mid).delete()
        # the gateway delete event does the same; doing it here too makes it immediate
        self._memory_enqueue(partial(self.memory.delete, guild_id, list(ids)))

    async def _maybe_delete_own(self, message: discord.Message) -> bool:
        """Reply to one of Salid's messages with just "ลบ" / "delete" / "unsend" ...: Salid deletes
        it (and every chunk of the same answer). Allowed for the person Salid was answering,
        server managers and executives. A thumbs up replaces a confirmation message."""
        ref = message.reference
        if not ref or not ref.message_id or not message.content.strip():
            return False
        if parse_delete_message(self._command_text(message)) is None:
            return False
        rec = self._answers.get(ref.message_id)
        asker_id = rec.asker_id if rec else None
        if rec is None:
            target = getattr(ref, "resolved", None)
            if target is None:
                with contextlib.suppress(discord.HTTPException):
                    target = await message.channel.fetch_message(ref.message_id)
            if target is None or getattr(target, "author", None) != self.user:
                return False  # not a Salid message: nothing to do here
            origin = getattr(getattr(target, "reference", None), "resolved", None)
            asker_id = getattr(getattr(origin, "author", None), "id", None)
        user = message.author
        allowed = user.id == asker_id or (
            hasattr(user, "guild_permissions") and self.is_exec_member(user)
        )
        if not allowed:
            await message.reply(t("delete_denied", self._lang(message)), mention_author=False)
            return True
        ids = list(rec.message_ids) if rec else [ref.message_id]
        if rec:
            self._answers.forget(rec)
        await self._delete_messages(message.channel, ids, message.guild.id)
        with contextlib.suppress(discord.HTTPException):
            await message.add_reaction("👍")
        return True

    async def _deep(
        self,
        message: discord.Message,
        guild_id: int,
        task: str,
        is_exec: bool,
        setting: str = "th",
    ):
        """Executives only. Acknowledge at once, run Gemini CLI in the background, post the result."""
        lang = resolve_lang(setting, task)
        reply = message.reply
        status = self._deep_status(guild_id, task, is_exec)
        if status == "ok":
            await reply(t("deep_ack", lang), mention_author=False)
            self._launch_deep(message, guild_id, task, setting)
        else:
            key = {
                "not_executive": "deep_not_exec",
                "disabled": "deep_disabled",
                "empty": "deep_usage",
                "busy": "deep_busy",
            }[status]
            await reply(t(key, lang), mention_author=False)

    async def _run_deep(
        self,
        message: discord.Message,
        guild_id: int,
        task: str,
        setting: str = "th",
        handle: DeepHandle | None = None,
    ):
        lang = resolve_lang(setting, task)
        try:
            res = await asyncio.to_thread(
                run_deep_task,
                self.s,
                self.store,
                task,
                guild_id=guild_id,
                lang=setting,
                **({"handle": handle} if handle is not None else {}),
            )
        except (asyncio.CancelledError, DeepCancelled):
            if handle is not None:
                handle.cancel()  # make sure the CLI process is gone
            logger.info("deep task cancelled")
            return
        except DeepTaskError as e:
            await message.channel.send(f"❌ {e}")
            return
        except Exception:
            logger.exception("deep task crashed")
            await message.channel.send(t("deep_failed", lang))
            return
        logger.info(
            "deep task done in %.0fs, %d tokens, models=%s",
            res.seconds,
            res.tokens,
            list(res.models),
        )
        text, attachment = reply_parts(res, lang)
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
                    description=t("cmd_setup"),
                    callback=setup,
                )
            )

        async def mode_cb(interaction: discord.Interaction, mode: str | None = None):
            await self.handle_mode(interaction, mode)

        mode_cb = app_commands.guild_only()(mode_cb)
        mode_cb = app_commands.describe(mode=t("cmd_mode_arg"))(mode_cb)
        mode_cb = app_commands.choices(
            mode=[app_commands.Choice(name=mode_choice_name(m), value=m) for m in MODES]
        )(mode_cb)
        for name in MODE_COMMANDS:
            self.tree.add_command(
                app_commands.Command(
                    name=name,
                    description=t("cmd_mode"),
                    callback=mode_cb,
                )
            )

        async def memory_cb(interaction: discord.Interaction, days: int | None = None):
            await self.handle_memory(interaction, days)

        memory_cb = app_commands.guild_only()(memory_cb)
        memory_cb = app_commands.describe(days=t("cmd_memory_arg"))(memory_cb)
        for name in MEMORY_COMMANDS:
            self.tree.add_command(
                app_commands.Command(name=name, description=t("cmd_memory"), callback=memory_cb)
            )

        async def language_cb(interaction: discord.Interaction, language: str | None = None):
            await self.handle_language(interaction, language)

        language_cb = app_commands.guild_only()(language_cb)
        language_cb = app_commands.describe(language=t("cmd_language_arg"))(language_cb)
        language_cb = app_commands.choices(
            language=[app_commands.Choice(name=LANGUAGE_LABELS[x], value=x) for x in LANGS]
        )(language_cb)
        for name in LANGUAGE_COMMANDS:
            self.tree.add_command(
                app_commands.Command(name=name, description=t("cmd_language"), callback=language_cb)
            )

        async def summary_cb(
            interaction: discord.Interaction,
            period: str = "today",
            channel: discord.TextChannel | None = None,
            all_channels: bool = False,
            topic: str | None = None,
            private: bool = False,
        ):
            await self.handle_summary(interaction, period, channel, all_channels, topic, private)

        summary_cb = app_commands.guild_only()(summary_cb)
        summary_cb = app_commands.describe(
            period=t("cmd_summary_period"),
            channel=t("cmd_summary_channel"),
            all_channels=t("cmd_summary_all"),
            topic=t("cmd_summary_topic"),
            private=t("cmd_summary_private"),
        )(summary_cb)
        summary_cb = app_commands.choices(
            period=[app_commands.Choice(name=t(f"period_{p}"), value=p) for p in PERIODS]
        )(summary_cb)
        for name in SUMMARY_COMMANDS:
            self.tree.add_command(
                app_commands.Command(name=name, description=t("cmd_summary"), callback=summary_cb)
            )

        async def summarize_menu(interaction: discord.Interaction, message: discord.Message):
            await self.handle_summarize_menu(interaction, message)

        summarize_menu = app_commands.guild_only()(summarize_menu)
        self.tree.add_command(
            app_commands.ContextMenu(name=MENU_SUMMARIZE, callback=summarize_menu)
        )
        for names, handler, desc in (
            (CANCEL_COMMANDS, self.handle_cancel, t("cmd_cancel")),
            (CHANNELS_COMMANDS, self.handle_channels, t("cmd_channels")),
            (JOIN_COMMANDS, self.handle_join, t("cmd_join")),
            (LEAVE_COMMANDS, self.handle_leave, t("cmd_leave")),
        ):
            cb = self._command_callback(handler)
            for name in names:
                self.tree.add_command(
                    app_commands.Command(name=name, description=desc, callback=cb)
                )

    async def handle_summary(
        self,
        interaction: discord.Interaction,
        period: str = "today",
        channel=None,
        all_channels: bool = False,
        topic: str | None = None,
        private: bool = False,
    ) -> None:
        """/สรุป: summarise a period of chat memory through the normal ask. The channels the
        model may read are computed for the invoker and for where the reply is posted (a private
        reply: just the invoker's own access)."""
        guild, user, here = interaction.guild, interaction.user, interaction.channel
        await interaction.response.defer(ephemeral=bool(private))
        await asyncio.to_thread(self.team.current, guild.id)
        setting = self._lang_setting(interaction)
        lang = self._interaction_lang(interaction)
        send = partial(interaction.followup.send, ephemeral=bool(private))
        if not (self.team.enabled(guild.id, "qa") and self.team.enabled(guild.id, "memory")):
            await send(t("summary_unavailable", lang))
            return
        where = SimpleNamespace(guild=None if private else guild, channel=here)
        memory = await self._memory_tools(where, guild, user)
        target = channel or here
        found = self._memory_channel(target)
        if memory is None or (
            not all_channels and (found is None or found[0] not in memory.allowed)
        ):
            await send(t("summary_no_access", lang))
            return
        q = summary_question(
            lang,
            period=period if period in PERIODS else "today",
            today=self.s.now().date(),
            tz=self.s.tz,
            channel_name=None if all_channels else getattr(target, "name", None),
            topic=topic,
        )
        ans = await asyncio.to_thread(
            ask,
            self.s,
            self.store,
            q,
            guild_id=guild.id,
            is_exec=self.is_exec_member(user),
            user_id=user.id,
            language=setting if setting != "auto" else lang,
            memory=memory,
            **self._web_kw(guild.id),
        )
        logger.info("summary command answered in %.1fs (%d rounds)", ans.seconds, ans.rounds)
        for part in chunks(ans.text):
            await send(part)

    async def handle_summarize_menu(self, interaction: discord.Interaction, message) -> None:
        """Message context menu "Summarize with Salid": summarise the right-clicked message."""
        guild, user = interaction.guild, interaction.user
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.team.current, guild.id)
        setting = self._lang_setting(interaction)
        lang = self._interaction_lang(interaction)
        send = partial(interaction.followup.send, ephemeral=True)
        if not self.team.enabled(guild.id, "qa"):
            await send(t("summary_unavailable", lang))
            return
        text = await self._summarize_message(
            message, guild, user, setting if setting != "auto" else lang
        )
        if text is None:
            text = t("summary_nothing", lang)
        for part in chunks(text):
            await send(part)

    @staticmethod
    def _command_callback(handler):
        async def cb(interaction: discord.Interaction):
            await handler(interaction)

        return app_commands.guild_only()(cb)

    async def handle_join(self, interaction: discord.Interaction) -> None:
        """/เข้าประชุม: join the invoker's voice channel and start the meeting."""
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.team.current, interaction.guild.id)
        reply = await self.meetings.join(
            interaction.user,
            fallback_channel=interaction.channel,
            lang=self._interaction_lang(interaction),
        )
        await interaction.followup.send(reply, ephemeral=True)

    async def handle_leave(self, interaction: discord.Interaction) -> None:
        """/ออกประชุม: leave and post the summary."""
        await asyncio.to_thread(self.team.current, interaction.guild.id)
        reply = await self.meetings.leave(
            interaction.user, lang=self._interaction_lang(interaction)
        )
        await interaction.response.send_message(reply, ephemeral=True)

    def _may_configure(self, user) -> bool:
        return hasattr(user, "guild_permissions") and (
            can_manage_guild(user) or self.is_exec_member(user)
        )

    async def handle_mode(self, interaction: discord.Interaction, mode: str | None) -> None:
        """/โหมด: show this channel's mode, or (executives / server managers) change it."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        send = interaction.response.send_message
        if guild is None or channel is None:
            await send(t("setup_denied", self._interaction_lang(interaction)), ephemeral=True)
            return
        await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        if mode is None:
            current = self._mode(channel, guild.id)
            lines = [t("mode_current", lang, label=mode_label(current, lang)), ""]
            if self._is_snoozed(channel.id):
                lines += [t("mode_snoozed", lang), ""]
            days = self.team.memory_retention_days(guild.id)
            lines += [f"{mode_label(m, lang)} — {mode_help(m, lang, days)}" for m in MODES]
            lines += ["", t("mode_footer", lang)]
            await send("\n".join(lines), ephemeral=True)
            return
        if mode not in MODES:
            await send(t("mode_failed", lang), ephemeral=True)
            return
        if not self._may_configure(user):
            await send(t("mode_denied", lang), ephemeral=True)
            return
        previous = self.team.channel_mode(
            guild.id, channel.id, getattr(channel, "name", None), self.s.ask_channel_ids
        )
        try:
            await asyncio.to_thread(self.team.set_channel_mode, guild.id, channel.id, mode)
        except Exception:
            logger.exception("could not save channel mode")
            await send(t("mode_failed", lang), ephemeral=True)
            return
        self._snoozed.pop(channel.id, None)
        self._convos.pop(channel.id, None)
        self._pending.pop(channel.id, None)
        self._seen_mode[channel.id] = mode
        await send(t("mode_set", lang, label=mode_label(mode, lang)))
        if previous == "off" and mode != "off":
            await self._memory_notice(channel, guild.id)
            self._start_import()

    async def handle_memory(self, interaction: discord.Interaction, days: int | None) -> None:
        """/ความจำ: show the memory status (everyone) or (executives / server managers) set how
        many days this server's chat memory is kept."""
        guild, user = interaction.guild, interaction.user
        send = interaction.response.send_message
        if guild is None:
            await send(t("setup_denied", self._interaction_lang(interaction)), ephemeral=True)
            return
        await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        current = self.team.memory_retention_days(guild.id)
        if days is None:
            state = t("memory_on" if self.team.enabled(guild.id, "memory") else "memory_off", lang)
            recorded = ", ".join(mode_label(m, lang) for m in MODES if m != "off")
            lines = [
                t("memory_status", lang, state=state, days=current),
                t("memory_modes", lang, recorded=recorded, off=mode_label("off", lang)),
            ]
            prog = await asyncio.to_thread(
                import_progress,
                self.memory_store,
                guild.id,
                self.s.memory_import_images_per_day,
                self.s.now().date().isoformat(),
            )
            if prog["channels_total"]:
                lines.append(
                    t(
                        "memory_import",
                        lang,
                        done=prog["channels_done"],
                        total=prog["channels_total"],
                        messages=f"{prog['messages']:,}",
                        images=prog["images_done"],
                        cap=prog["images_cap"],
                    )
                )
            lines.append(t("memory_footer", lang))
            await send("\n".join(lines), ephemeral=True)
            return
        if not self._may_configure(user):
            await send(t("memory_denied", lang), ephemeral=True)
            return
        if valid_retention_days(days) is None:
            await send(
                t("memory_bad_days", lang, lo=MIN_RETENTION_DAYS, hi=MAX_RETENTION_DAYS),
                ephemeral=True,
            )
            return
        try:
            await asyncio.to_thread(self.team.set_memory_retention_days, guild.id, days)
        except Exception:
            logger.exception("could not save memory retention")
            await send(t("memory_failed", lang), ephemeral=True)
            return
        await send(t("memory_set", lang, days=days))

    async def handle_language(self, interaction: discord.Interaction, language: str | None) -> None:
        """/ภาษา: show this channel's reply language, or (executives / server managers) set it."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        send = interaction.response.send_message
        if guild is None or channel is None:
            await send(t("setup_denied", self._interaction_lang(interaction)), ephemeral=True)
            return
        await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        if language is None:
            current = self.team.channel_language(guild.id, channel.id)
            lines = [
                t("lang_current", lang, label=t(f"lang_label_{current}", lang)),
                "",
                t("lang_footer", lang),
            ]
            await send("\n".join(lines), ephemeral=True)
            return
        if language not in LANGS:
            await send(t("lang_failed", lang), ephemeral=True)
            return
        if not self._may_configure(user):
            await send(t("lang_denied", lang), ephemeral=True)
            return
        try:
            await asyncio.to_thread(self.team.set_channel_language, guild.id, channel.id, language)
        except Exception:
            logger.exception("could not save channel language")
            await send(t("lang_failed", lang), ephemeral=True)
            return
        new = lang if language == "auto" else language  # confirm in the language just chosen
        await send(t("lang_set", new, label=t(f"lang_label_{language}", new)))

    async def handle_setup(self, interaction: discord.Interaction) -> None:
        """/ตั้งค่า: DM-style ephemeral reply with a one-time link to the hub for this guild."""
        guild, user = interaction.guild, interaction.user
        if guild is not None:
            await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        if guild is None or not hasattr(user, "guild_permissions") or not can_manage_guild(user):
            await interaction.response.send_message(t("setup_denied", lang), ephemeral=True)
            return
        try:
            code = await asyncio.to_thread(
                self.guilds_store.create_link_code, guild.id, user.id, str(user)
            )
        except Exception:
            logger.exception("could not create a link code")
            await interaction.response.send_message(t("setup_failed", lang), ephemeral=True)
            return
        url = f"{self.s.hub_url.rstrip('/')}/hub?link={code}"
        await interaction.response.send_message(
            t("setup_message", lang, name=guild.name, url=url), ephemeral=True
        )
        await self._send_hidden_channels(interaction, lang, first=False)

    async def _send_hidden_channels(
        self, interaction: discord.Interaction, lang: str, *, first: bool
    ) -> None:
        """Ephemeral "channels Salid cannot see" message(s); `first`: nothing was sent yet."""
        entries = hidden_channels(interaction.guild, interaction.guild.me)
        for text in hidden_channels_messages(entries, lang):
            if first:
                await interaction.response.send_message(text, ephemeral=True)
                first = False
            else:
                await interaction.followup.send(text, ephemeral=True)

    async def handle_channels(self, interaction: discord.Interaction) -> None:
        """/ห้อง: which channels Salid cannot read and how to add it (managers and executives)."""
        guild, user = interaction.guild, interaction.user
        if guild is not None:
            await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        allowed = (
            guild is not None
            and hasattr(user, "guild_permissions")
            and (can_manage_guild(user) or self.is_exec_member(user))
        )
        if not allowed:
            await interaction.response.send_message(t("setup_denied", lang), ephemeral=True)
            return
        await self._send_hidden_channels(interaction, lang, first=True)

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
        await self._memory_due()
        await self._scheduler_due()

    async def _scheduler_due(self) -> None:
        """Meeting reminders, no-show nudges and auto-join (every SCHEDULER_TICK_SECONDS)."""
        now = self._clock()
        if not self.guilds or now - self._sched_last < SCHEDULER_TICK_SECONDS:
            return
        self._sched_last = now
        await self.scheduler.tick()

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
                    await convo.channel.send(t("signoff", self._lang(convo.channel)))
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
            now = self._clock()
            candidates = [  # same person twice in a row: dropped before the gate (no tokens)
                m for m in pending.candidates if not self._user_cooling_down(cid, m.author.id, now)
            ]
            if not candidates:
                logger.info("join candidates all in same-user cooldown in channel %s", cid)
                continue
            lines = await self._gate_lines(pending.channel, JOIN_HISTORY)
            ids = [str(m.id) for m in candidates]
            hints = await self._interjection_hints(pending.guild, candidates)
            kw = {"hints": hints} if hints else {}
            picked = await asyncio.to_thread(
                partial(
                    gate.pick_interjection,
                    self.llm,
                    self.s.gate_model,
                    lines,
                    ids,
                    channel_id=cid,
                    **kw,
                )
            )
            target = next((m for m in candidates if str(m.id) == picked), None)
            if target is None:
                continue
            try:  # budget and cooldown are used only when something was actually posted
                posted = await self._answer(target, pending.guild, target.author, interjection=True)
            except Exception:
                logger.exception("join answer failed")
                continue
            if posted is not False:
                self._interjections.setdefault(cid, deque()).append(self._clock())
                self._interjected_at[(cid, target.author.id)] = self._clock()

    def _user_cooling_down(self, channel_id: int, user_id: int, now: float) -> bool:
        at = self._interjected_at.get((channel_id, user_id))
        if at is not None and now - at >= INTERJECT_SAME_USER_COOLDOWN:
            del self._interjected_at[(channel_id, user_id)]
            return False
        return at is not None

    async def _interjection_hints(self, guild, candidates) -> dict[str, str]:
        """Free stage-2 hints: per candidate a keyword search of memory (same scoping as the
        chat_search tool for that candidate), compact lines, 600 characters in all. No LLM call.
        Empty when memory is off."""
        if not self.team.enabled(guild.id, "memory"):
            return {}
        hints: dict[str, str] = {}
        for m in candidates:
            kws = gate.extract_keywords(self._strip_mentions(m.content or "", m))
            if not kws:
                continue
            try:
                tools = await self._memory_tools(m, guild, m.author)
                if tools is None:
                    continue
                result = await asyncio.to_thread(
                    tools.chat_search, kws, limit=gate.HINT_HITS_PER_CANDIDATE + 2
                )
            except Exception:
                logger.warning("memory hint lookup failed", exc_info=True)
                continue
            if line := gate.hint_line(
                result, exclude_text=self._strip_mentions(m.content or "", m)
            ):
                hints[str(m.id)] = line
        return gate.fit_hints(hints)

    # ------------------------------------------------------------- organisational memory
    def _memory_enqueue(self, job) -> None:
        """Queue a blocking memory write. One worker runs the jobs in order (an edit never
        overtakes its message), off the event loop; a full queue drops the job, never blocks."""
        if self._memory_queue is None:
            self._memory_queue = asyncio.Queue(MEMORY_QUEUE_MAX)
        if self._memory_worker is None or self._memory_worker.done():
            self._memory_worker = asyncio.create_task(self._memory_run())
        try:
            self._memory_queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning("memory queue full; dropped one record")

    async def _memory_run(self) -> None:
        queue = self._memory_queue
        while True:
            job = await queue.get()
            try:
                await asyncio.to_thread(job)
            except Exception:
                logger.warning("memory write failed", exc_info=True)
            finally:
                queue.task_done()

    async def drain_memory(self) -> None:
        """Wait until queued memory writes are done (tests, shutdown)."""
        if self._memory_queue is not None:
            await self._memory_queue.join()

    @staticmethod
    def _memory_channel(channel) -> tuple[int, int | None, str | None] | None:
        """(channel id for access checks, thread id, name for the mode) or None to skip. A thread
        counts as its parent channel; private threads are not recorded."""
        parent_id = getattr(channel, "parent_id", None)
        if parent_id is None:
            return channel.id, None, getattr(channel, "name", None)
        is_private = getattr(channel, "is_private", None)
        if callable(is_private) and is_private():
            return None
        return parent_id, channel.id, getattr(getattr(channel, "parent", None), "name", None)

    def _memory_message(self, message, own: bool, cid: int, thread_id: int | None):
        """The log record for a Discord message, or None when it has no text."""
        text = message_text(message.content, message.attachments)
        if not text:
            return None
        return build_message(
            self.s,
            guild_id=message.guild.id,
            channel_id=cid,
            thread_id=thread_id,
            message_id=message.id,
            author_id=message.author.id,
            author_name="Salid" if own else message.author.display_name,
            created_at=message.created_at,
            text=text,
            jump_url=getattr(message, "jump_url", ""),
            kind="salid" if own else "chat",
        )

    async def _memory_record(self, message: discord.Message, own: bool) -> bool:
        """Layer 1: log the message when the channel's mode is not off. Never raises."""
        try:
            guild = message.guild
            gid = guild.id
            await asyncio.to_thread(self.team.current, gid)
            where = self._memory_channel(message.channel)
            if where is None:
                return
            cid, thread_id, cname = where
            mode = self.team.channel_mode(gid, cid, cname, self.s.ask_channel_ids)
            previous, self._seen_mode[cid] = self._seen_mode.get(cid), mode
            if mode == "off" or not self.team.enabled(gid, "memory"):
                return
            record = self._memory_message(message, own, cid, thread_id)
            if record is not None:
                self._memory_enqueue(partial(self.memory.record, record))
            if previous == "off" and not own:  # switched on elsewhere (hub): tell the channel once
                target = message.channel if thread_id is None else message.channel.parent
                await self._memory_notice(target or message.channel, gid)
                self._start_import()  # and read this channel's history
            return record is not None
        except Exception:
            logger.warning("memory record failed", exc_info=True)
            return False

    async def _memory_notice(self, channel, guild_id: int) -> None:
        """One-time note that Salid now remembers this channel (off -> on)."""
        if not self.team.enabled(guild_id, "memory"):
            return
        with contextlib.suppress(discord.HTTPException):
            await channel.send(
                t(
                    "memory_notice",
                    self._lang(channel),
                    days=self.team.memory_retention_days(guild_id),
                )
            )

    async def record_voice(self, voice_channel, user_id: int, name: str, text: str, at: datetime):
        """A transcribed meeting segment goes into the log as kind "voice" (channel = the voice
        channel). Never raises."""
        try:
            guild = voice_channel.guild
            await asyncio.to_thread(self.team.current, guild.id)
            mode = self.team.channel_mode(
                guild.id, voice_channel.id, getattr(voice_channel, "name", None),
                self.s.ask_channel_ids,
            )  # fmt: skip
            if mode == "off" or not self.team.enabled(guild.id, "memory") or not text.strip():
                return
            self._memory_enqueue(
                partial(
                    self.memory.record,
                    build_message(
                        self.s,
                        guild_id=guild.id,
                        channel_id=voice_channel.id,
                        message_id=synthetic_id(at, user_id % 1000),
                        author_id=user_id,
                        author_name=name,
                        created_at=at,
                        text=text[:4000],
                        jump_url=f"https://discord.com/channels/{guild.id}/{voice_channel.id}",
                        kind="voice",
                    ),
                )
            )
        except Exception:
            logger.warning("memory voice record failed", exc_info=True)

    async def on_raw_message_edit(self, payload) -> None:
        data = getattr(payload, "data", None) or {}
        if payload.guild_id is None or "content" not in data:
            return
        atts = [
            SimpleNamespace(filename=a.get("filename"), content_type=a.get("content_type"))
            for a in data.get("attachments") or []
        ]
        stamp = data.get("edited_timestamp")
        try:
            edited = datetime.fromisoformat(stamp).astimezone(UTC) if stamp else datetime.now(UTC)
        except ValueError:
            edited = datetime.now(UTC)
        self._memory_enqueue(
            partial(
                self.memory.edit,
                payload.guild_id,
                payload.message_id,
                message_text(data["content"], atts),
                edited,
            )
        )

    async def on_raw_message_delete(self, payload) -> None:
        if payload.guild_id is not None:
            self._memory_enqueue(
                partial(self.memory.delete, payload.guild_id, [payload.message_id])
            )

    async def on_raw_bulk_message_delete(self, payload) -> None:
        if payload.guild_id is not None:
            self._memory_enqueue(
                partial(self.memory.delete, payload.guild_id, sorted(payload.message_ids))
            )

    # ---- history import (see salid.memory_import)
    def _start_import(self) -> None:
        """Ask for an import run (startup, reconnect, a channel switched on, retention
        lengthened, a capped run resuming). One background task; requests while it runs are
        folded into one more pass."""
        self._import_requested = True
        if self._import_task is None or self._import_task.done():
            self._import_task = asyncio.create_task(self._import_loop())

    def _request_import(self) -> None:
        """Import soon after Salid may have gained access to channels. Debounced: events that
        arrive while one request is waiting are folded into it, so a burst starts one run."""
        if self._import_debounce is not None and not self._import_debounce.done():
            return

        async def later():
            await asyncio.sleep(IMPORT_DEBOUNCE_S)
            self._start_import()

        self._import_debounce = asyncio.create_task(later())

    async def _import_loop(self) -> None:
        try:
            while self._import_requested:
                self._import_requested = False
                await self._import_all()
        except Exception:
            logger.exception("memory import failed")

    async def _import_all(self) -> None:
        for guild in list(self.guilds):
            await asyncio.to_thread(self.team.current, guild.id)
            if not self.team.enabled(guild.id, "memory"):
                continue
            targets = await self._import_targets(guild)
            res = await self.importer.run_guild(guild.id, targets)
            if res.capped:
                self._import_resume_at = self._clock() + IMPORT_RESUME_S
            if self.team.enabled(guild.id, "image_memory"):
                await self._image_backlog(guild, {t.cid for t in targets})
            await self._media_doc_backlog(guild, {t.cid for t in targets})

    async def _import_targets(self, guild) -> list[Target]:
        """Readable, non-off channels (text, voice chat) and threads (active, and archived ones
        of readable text channels). Private threads are skipped."""
        me = guild.me
        floor = self.s.now() - timedelta(days=self.team.memory_retention_days(guild.id))

        def usable(ch):
            where = self._memory_channel(ch)
            if where is None:
                return None
            cid, thread_id, cname = where
            if self.team.channel_mode(guild.id, cid, cname, self.s.ask_channel_ids) == "off":
                return None
            if not self._can_read(ch, me):
                return None
            return Target(ch, cid, thread_id)

        out: dict[int, Target] = {}
        chans = [*guild.text_channels, *getattr(guild, "voice_channels", ())]
        for ch in [*chans, *getattr(guild, "threads", ())]:
            if (tg := usable(ch)) is not None:
                out[tg.state_id] = tg
        for ch in guild.text_channels:
            if not hasattr(ch, "archived_threads") or ch.id not in {t.cid for t in out.values()}:
                continue
            try:
                async for th in ch.archived_threads(limit=50):
                    stamp = getattr(th, "archive_timestamp", None)
                    if stamp is not None and stamp < floor:
                        break
                    if (tg := usable(th)) is not None:
                        out.setdefault(tg.state_id, tg)
            except discord.HTTPException:
                logger.debug("could not list archived threads of %s", ch.id, exc_info=True)
        return list(out.values())

    def _import_record(self, message, target: Target):
        own = message.author == self.user
        if not own and getattr(message.author, "bot", False):
            return None
        return self._memory_message(message, own, target.cid, target.thread_id)

    def _memory_triggers(self) -> None:
        """Start an import when memory was switched on or retention lengthened (hub or
        command), and resume a run the guild cap stopped."""
        for guild in self.guilds:
            now = (self.team.enabled(guild.id, "memory"), self.team.memory_retention_days(guild.id))
            before = self._memory_cfg_seen.get(guild.id)
            self._memory_cfg_seen[guild.id] = now
            if before is not None and now[0] and (not before[0] or now[1] > before[1]):
                self._start_import()
        if self._import_resume_at is not None and self._clock() >= self._import_resume_at:
            self._import_resume_at = None
            self._start_import()

    # ---- images (see salid.image_memory)
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _image_candidates(self, message) -> list[tuple[int, object]]:
        found = [(i, a) for i, a in enumerate(message.attachments) if image_mime(a)]
        return found[:MAX_IMAGES_PER_MESSAGE]

    def _images_enabled(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "memory") and self.team.enabled(guild_id, "image_memory")

    def _describe_soon(self, message, logged: bool = True, intake_mode: str | None = None) -> None:
        """Describe the message's images in the background (never delays a reply). With
        `intake_mode` set, images the describer takes for receipts also go through receipt
        intake (quietly in listen mode)."""
        cands = self._image_candidates(message)
        if cands and self._images_enabled(message.guild.id) and (logged or intake_mode):
            self._spawn(
                self._describe_message(message, cands, intake_mode=intake_mode, apply=logged)
            )

    async def _read_describe(self, att) -> tuple[bytes | None, ImageNote | None]:
        """One model call per image; the bytes are returned so receipt intake can reuse them."""
        if (getattr(att, "size", 0) or 0) > MAX_IMAGE_BYTES:
            self.image_stats["skipped"] += 1
            return None, None
        data = None
        try:
            data = await att.read()
            async with self._image_sem:
                note = await asyncio.to_thread(
                    describe_image, self.llm, self.s.gate_model, data, image_mime(att)
                )
        except Exception as e:  # never log contents, only the kind of failure
            self.image_stats["failed"] += 1
            logger.warning("image description failed (%s)", type(e).__name__)
            return data, None
        self.image_stats["described"] += 1
        return data, note

    async def _describe_att(self, att) -> str | None:
        _data, note = await self._read_describe(att)
        return format_note(note) if note is not None else None

    async def _describe_message(
        self, message, cands, intake_mode: str | None = None, apply: bool = True
    ) -> tuple[bool, dict[int, str]]:
        """Describe images; receipt-like ones also run receipt intake when `intake_mode` is set.
        Returns (a receipt was handled, notes by attachment index)."""
        notes: dict[int, str] = {}
        handled = False
        for i, att in cands:
            data, note = await self._read_describe(att)
            if note is None:
                continue
            notes[i] = format_note(note)
            if intake_mode and data is not None and note.is_receipt_like:
                try:
                    res = await self._ingest_one(message, att, data, intake_mode == "listen")
                except Exception:
                    logger.exception("receipt intake from image failed")
                    continue
                if res.status != "ignored":
                    handled = True
                    if res.receipt is not None:
                        notes[i] = receipt_note(res.receipt)
        logger.info(
            "memory images guild=%s channel=%s described=%d of %d (totals %s)",
            message.guild.id, message.channel.id, len(notes), len(cands), self.image_stats,
        )  # fmt: skip
        if notes and apply:
            self._apply_notes(message, notes)
        return handled, notes

    def _apply_notes(self, message, notes: dict[int, str]) -> None:
        """Update the message's log record with image descriptions / receipt summaries."""
        merged = self._image_notes.setdefault(message.id, {})
        merged.update(notes)
        while len(self._image_notes) > IMAGE_NOTES_KEPT:
            self._image_notes.pop(next(iter(self._image_notes)))
        text = message_text(message.content, message.attachments, merged)
        self._memory_enqueue(partial(self.memory.edit, message.guild.id, message.id, text, None))

    async def _image_backlog(self, guild, channel_ids: set[int]) -> None:
        """Historical images (records still holding a plain "[image: name]" descriptor), newest
        first, at most `memory_import_images_per_day` per guild per day. Messages are re-fetched
        from Discord; ones that are gone or unreadable are marked "[image-unread: name]"."""
        gid = guild.id
        gstate = self.importer.guild_state(gid)
        cap = self.s.memory_import_images_per_day
        now = self.s.now()
        start = (now - timedelta(days=self.team.memory_retention_days(gid))).date().isoformat()
        rows = await asyncio.to_thread(
            self.memory_store.messages, gid, channel_ids, start, now.date().isoformat(),
            keyword="[image:", limit=5000,
        )  # fmt: skip
        done = 0
        logger.info(
            "memory backlog images guild=%s pending=%d today=%d/%d", gid, len(rows),
            gstate.images_done, cap,
        )  # fmt: skip
        for rec in rows:
            if gstate.images_done >= cap:
                break
            if "[image: " not in rec.text or rec.kind == "salid":
                continue
            new_text, used = await self._backlog_record(guild, rec)
            if used:
                gstate.images_done += used
                done += used
            if new_text != rec.text:
                self._memory_enqueue(partial(self.memory.edit, gid, rec.message_id, new_text, None))
                await asyncio.to_thread(self.memory_store.put_guild_state, gstate)
            await asyncio.sleep(self.importer.page_sleep)
        await asyncio.to_thread(self.memory_store.put_guild_state, gstate)
        if done:
            logger.info(
                "memory backlog images guild=%s described=%d today=%d/%d", gid, done,
                gstate.images_done, cap,
            )  # fmt: skip

    async def _backlog_record(self, guild, rec) -> tuple[str, int]:
        """(new text, model calls used) for one logged message with undescribed images."""
        getter = getattr(guild, "get_channel_or_thread", None) or guild.get_channel
        channel = getter(rec.thread_id or rec.channel_id)
        text, used = rec.text, 0
        try:
            msg = await channel.fetch_message(rec.message_id) if channel else None
        except discord.HTTPException:
            msg = None
        cands = self._image_candidates(msg) if msg is not None else []
        logger.info(
            "memory backlog message=%s fetched=%s images=%d", rec.message_id, msg is not None,
            len(cands),
        )  # fmt: skip
        for _i, att in cands:
            marker = describe_attachment(att)
            if marker not in text:
                continue
            used += 1
            data, note = await self._read_describe(att)
            new = format_note(note) if note is not None else None
            if note is not None and note.is_receipt_like and data is not None:
                receipt = await self._backlog_receipt(msg, att, data)
                if receipt is not None:
                    used += 1  # the extraction call counts against the same daily cap
                    new = receipt_note(receipt)
            text = text.replace(marker, new or marker.replace("[image:", "[image-unread:", 1), 1)
        # anything still plain has no attachment to read (message gone): stop retrying it
        if msg is None or not cands:
            text = text.replace("[image: ", "[image-unread: ")
        return text, used

    async def _backlog_receipt(self, msg, att, data: bytes):
        """Quietly ingest an old receipt-like image into the ledger (no reply, no reaction):
        original author as submitter, the message time as the fallback date, deduped by sha256."""
        gid = msg.guild.id
        if not self.team.enabled(gid, "receipts") or self._mode(msg.channel, gid) == "off":
            return None
        try:
            res = await asyncio.to_thread(
                ingest_attachment,
                self.store,
                self.llm,
                self.s,
                guild_id=gid,
                data=data,
                filename=att.filename,
                content_type=att.content_type,
                submitter_id=msg.author.id,
                submitter_name=msg.author.display_name,
                channel_id=msg.channel.id,
                message_id=msg.id,
                note=(msg.content or "").strip(),
                archive=self.archive,
                lang=resolve_lang(self.team.current(gid).default_language, msg.content),
                at=msg.created_at,
            )
        except Exception as e:  # counts only, never contents
            logger.warning("history receipt intake failed (%s)", type(e).__name__)
            return None
        return res.receipt if res.status in ("created", "duplicate") else None

    async def _memory_due(self) -> None:
        """Layer 2: run due summary updates (and the daily purge), throttled, off the loop."""
        now = self._clock()
        if now - self._memory_last < MEMORY_TICK_SECONDS:
            return
        self._memory_last = now
        try:
            self._memory_triggers()
            await asyncio.to_thread(self.memory.scheduler.tick, [g.id for g in self.guilds])
        except Exception:
            logger.exception("memory tick failed")

    def _can_read(self, channel, member) -> bool:
        perms = channel.permissions_for(member)
        return bool(
            getattr(perms, "read_messages", False) and getattr(perms, "read_message_history", False)
        )

    def _channel_viewers(self, guild, channel) -> frozenset[int]:
        """Humans who can read `channel`, cached for a minute."""
        hit = self._viewers.get(channel.id)
        if hit is not None and self._clock() - hit[0] < VIEWERS_TTL_S:
            return hit[1]
        members = guild.members.values() if isinstance(guild.members, dict) else guild.members
        viewers = frozenset(
            m.id for m in members if not getattr(m, "bot", False) and self._can_read(channel, m)
        )
        self._viewers[channel.id] = (self._clock(), viewers)
        return viewers

    def _memory_scope(self, guild, member, answering_channel_id: int | None) -> dict[int, str]:
        """Channel id -> name the asker may get memory from (see salid.memory_access). Blocking:
        run in a thread."""
        channels = [
            *guild.text_channels,
            *getattr(guild, "voice_channels", ()),
            *getattr(guild, "stage_channels", ()),
        ]
        access = []
        for c in channels:
            if self.team.channel_mode(guild.id, c.id, c.name, self.s.ask_channel_ids) == "off":
                continue
            viewers = self._channel_viewers(guild, c) - {member.id}
            if self._can_read(c, member):  # the asker is checked live, not from the cache
                viewers |= {member.id}
            access.append(ChannelAccess(c.id, c.name, viewers))
        allowed = allowed_channel_ids(access, member.id, answering_channel_id)
        return {a.channel_id: a.name for a in access if a.channel_id in allowed}

    async def _memory_tools(self, message, guild, member) -> MemoryTools | None:
        if not self.team.enabled(guild.id, "memory"):
            return None
        answering = None
        if message.guild is not None:  # in a DM the asker's own readable channels are enough
            where = self._memory_channel(message.channel)
            answering = where[0] if where else -1
        try:
            allowed = await asyncio.to_thread(self._memory_scope, guild, member, answering)
        except Exception:
            logger.warning("could not compute memory scope", exc_info=True)
            return None
        return MemoryTools(
            self.memory_store,
            guild.id,
            allowed,
            self.s,
            self.team.memory_retention_days(guild.id),
            opener=self._make_opener(guild, allowed, asyncio.get_running_loop()),
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
                lang = resolve_lang(self.team.channel_language(guild.id, channel.id))
                text = await asyncio.to_thread(
                    build_daily_digest, self.store, guild.id, now.date(), lang
                )
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
