"""Per-guild (Discord server) Salid settings, written by the bot and edited in the web hub.

One Firestore document per guild, `{SALID_GUILD_COLLECTION}/{guild_id}` (default `salid_guilds`):

    features             {receipts, qa, digest, deep_tasks, voice_meetings, memory, image_memory, web,
                         documents, scheduling, email, media_memory, doc_memory, read_receipts,
                         followups, reports, briefings, catchup, freetime, polls, wiki, approvals, places,
                         private_channels, maps, search, travel, markets, ideas, company_learning, contests, bizfin, link_guard, meeting_transcript_doc, attendance_log, promo, concept, decks, votes, vote_offers, reminders, projects, project_notes}: bool, all
                         default true
    receipt_channel_ids  list[str] extra channels where every attachment is a receipt
    ask_channel_ids      list[str] channels where every message is answered (mode "chat")
    channel_modes        {channel id str: off|listen|call|join|chat}, see salid.modes
    default_mode         mode of channels with no entry above (default "call")
    channel_languages    {channel id str: th|en|auto}, reply language override, see salid.i18n
    default_language     the server's main language, reply language of channels with no entry above
                         (default "en"; counts only once `language_chosen`, until then: auto)
    language_chosen      bool the server picked its main language (join question, /language, hub);
                         a document that already carries a default_language counts as chosen
    language_asked_at    str ISO time the join question was posted (so it is asked only once)
    channel_tones        {channel id str: cheerful|friendly|professional|concise|calm}, see salid.tone
    default_tone         tone of channels with no entry above (default "cheerful")
    tone_note            str <= 200 chars, owner's extra wording note appended to every tone
    receipt_home_channel_id  str main receipt channel ("" = auto-pick), see salid.receipt_forward
    receipt_forward      bool copy receipts posted in other channels to the home channel (default true)
    digest_channel_id    str ("" = no daily digest)
    digest_time          "HH:MM"
    followup_time        "HH:MM" daily task follow-up DMs (default 10:00)
    report_time          "HH:MM" evening report (default 18:30)
    morning_brief_time   "HH:MM" personal morning brief DM (default 08:30; replaces followup DMs)
    evening_brief_time   "HH:MM" personal evening brief DM (default 18:00)
    report_channel_id    str ("" = Salid creates/uses the private channel รายงานจากสลิด)
    exec_role_names      list[str] roles that count as executives (default ผู้บริหาร, Executive)
    admin_emails         list[str] lowercase Google accounts allowed to manage this guild in the hub
    guild_name, icon_url, installed_at, active   written by the bot (join / leave / resync)
    channels             list[{id, name}] text channels, synced by the bot (feeds the hub dropdowns)
    updated_by, updated_at

and one short-lived document per `/ตั้งค่า` link, `salid_link_codes/{code}`:
`{guild_id, discord_user_id, discord_user_name, created_at, expires_at, used}`.
The code is a 32-byte URL-safe random string; redeeming it (hub, after Google sign-in) is a
Firestore transaction that marks it used and adds the verified email to `admin_emails`. That is
how a Discord admin proves control of a guild to the hub with no OAuth secret.

The bot reads settings through `BotGuildSettings` (60 s cache per guild, fails open: a missing
document or a read error means defaults, i.e. every feature on). With `SALID_STORE=sqlite`
(local dev) the same API runs on an in-memory store, so nothing needs Firestore.

Channel rule: channel ids listed in the document are *added* to the env lists
(`SALID_RECEIPT_CHANNEL_IDS` / `SALID_ASK_CHANNEL_IDS`); `digest_channel_id` and, once the hub has
saved the document, `digest_time` and `exec_role_names` *override* the env values.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .i18n import DEFAULT_LANGUAGE, LANGS, resolve_language
from .modes import DEFAULT_MODE, MODES, resolve_mode
from .tone import DEFAULT_TONE, MAX_NOTE, TONES, clean_note, effective_tone, note_problem

logger = logging.getLogger(__name__)

TTL_S = 60.0
LINK_TTL = timedelta(minutes=15)
FEATURES = (
    "receipts",
    "qa",
    "digest",
    "deep_tasks",
    "voice_meetings",
    "auto_join_meetings",
    "memory",
    "image_memory",
    "media_memory",
    "doc_memory",
    "read_receipts",
    "web",
    "documents",
    "scheduling",
    "email",
    "followups",
    "reports",
    "briefings",
    "freetime",
    "polls",
    "catchup",
    "wiki",
    "approvals",
    "places",
    "private_channels",
    "maps",
    "search",
    "travel",
    "markets",
    "ideas",
    "company_learning",
    "contests",
    "bizfin",
    "link_guard",
    "meeting_transcript_doc",
    "attendance_log",
    "promo",
    "concept",
    "decks",
    "quotations",
    "votes",
    "vote_offers",
    "reminders",
    "projects",
    "project_notes",
)
DEFAULT_EXEC_ROLES = ["ผู้บริหาร", "Executive"]
MAX_CHANNEL_MODES = 500
MAX_CHANNEL_LANGUAGES = 500
MAX_CHANNEL_TONES = 500
_ID = re.compile(r"^\d{1,25}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


MIN_RETENTION_DAYS, MAX_RETENTION_DAYS = 1, 3650


def default_retention_days() -> int:
    """SALID_MEMORY_RETENTION_DAYS when valid, else 90 (the default for guilds without a value)."""
    v = os.environ.get("SALID_MEMORY_RETENTION_DAYS", "")
    return int(v) if v.isdigit() and MIN_RETENTION_DAYS <= int(v) <= MAX_RETENTION_DAYS else 90


def valid_retention_days(v) -> int | None:
    """An int within bounds (bools and floats with a fraction rejected), else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        v = int(v.strip())
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and MIN_RETENTION_DAYS <= v <= MAX_RETENTION_DAYS:
        return v
    return None


def _now() -> datetime:
    return datetime.now(UTC)


class Features(BaseModel):
    receipts: bool = True
    qa: bool = True
    digest: bool = True
    deep_tasks: bool = True
    voice_meetings: bool = True
    auto_join_meetings: bool = True  # Salid joins voice channels by itself (needs voice_meetings)
    memory: bool = True
    image_memory: bool = True  # describe images into the memory log (needs `memory`)
    media_memory: bool = True  # transcribe audio/video into the memory log (needs `memory`)
    doc_memory: bool = True  # read PDFs / documents into the memory log (needs `memory`)
    read_receipts: bool = True  # 👀 on files Salid read quietly (not addressed to it)
    web: bool = True  # read links and search the internet (read_web tool)
    documents: bool = True  # make_report_pdf: accounting documents as PDF
    scheduling: bool = True  # schedule_meeting & co: Discord Scheduled Events + reminders
    email: bool = True  # draft_email / send_email (still needs confirmation and a role)
    followups: bool = True  # daily DM to members with open tasks (needs `memory`)
    reports: bool = True  # evening report to the private reports channel (needs `memory`)
    polls: bool = True  # create_poll / DM forms (start_form)
    briefings: bool = True  # personal morning / evening briefs, doc alerts, ask reminders (DM)
    freetime: bool = True  # find_free_time / set_availability: find a time everyone is free
    catchup: bool = True  # "what did I miss" catch-up (tool, /ตามทัน, morning brief section)
    wiki: bool = True  # company knowledge base built from memory (needs `memory`)
    private_channels: bool = True  # a private Salid channel per member (off keeps existing ones)
    approvals: bool = True  # request_approval & co: approval requests with Approve / Decline DMs
    places: bool = True  # remembers head office / branches / consented personal places
    maps: bool = True  # travel time / routes / weather (also needs SALID_MAPS_ENABLED)
    travel: bool = True  # currency conversion + flight status / prices / tracking
    search: bool = True  # search_everything + web_search + deep_research (the last two need `web`)
    markets: bool = (
        True  # stock / gold / news tools, price alerts, market line in the morning brief
    )
    company_learning: bool = True  # company playbook + domain packs (company_brain.py)
    ideas: bool = True  # idea / project documents, business frameworks (BMC, SWOT...), exports
    contests: bool = True  # hackathons / grants / tenders: deadlines, plan, mock judge, reminders
    bizfin: bool = True  # finance calculators (break-even, pricing, NPV/IRR...), market benchmark, security/slip checks
    link_guard: bool = True  # warns once under a message with a clearly dangerous link
    meeting_transcript_doc: bool = True  # minutes PDF with who-said-what at meeting end + tool
    attendance_log: bool = (
        True  # who entered / left the voice meeting and when (chat lines + minutes)
    )
    promo: bool = True  # promo post kits from product photos (AI scene + typeset text)
    concept: bool = (
        True  # concept images + price checks for product / marketing concepts (needs ideas)
    )
    decks: bool = True  # the chat agent may build .pptx slide decks (salid.deck)
    quotations: bool = True  # quotations as PDF + Word with running numbers (salid.quotation)
    votes: bool = (
        True  # call_vote & co: button ballots in each voter's private channel (salid.votes)
    )
    vote_offers: bool = True  # offer a vote from untagged talk (needs `votes`)
    projects: bool = True  # project_create & co: PM dashboard projects (salid.projects)
    project_notes: bool = (
        True  # say "📌 recorded" in the room when a project change is heard in chat
    )
    reminders: bool = True  # set_reminder & co: one-off and recurring reminders (salid.reminders)


class Channel(BaseModel):
    id: str
    name: str
    visible: bool = True  # Salid has View Channel here (synced by the bot)
    readable_history: bool = True  # ... and Read Message History


def _clean_ids(v) -> list[str]:
    out = [str(x).strip() for x in (v or []) if str(x).strip()]
    if any(not _ID.match(x) for x in out):
        raise ValueError("channel ids must be numeric")
    return list(dict.fromkeys(out))


class GuildSettingsUpdate(BaseModel):
    """What the hub may change. Everything else in the document is owned by the bot."""

    features: Features = Field(default_factory=Features)
    receipt_channel_ids: list[str] = Field(default_factory=list, max_length=50)
    ask_channel_ids: list[str] = Field(default_factory=list, max_length=50)
    digest_channel_id: str = ""
    receipt_home_channel_id: str = ""
    receipt_forward: bool = True
    digest_time: str = "18:00"
    followup_time: str = "10:00"
    report_time: str = "18:30"
    morning_brief_time: str = "08:30"
    evening_brief_time: str = "18:00"
    report_channel_id: str = ""
    exec_role_names: list[str] = Field(default_factory=lambda: list(DEFAULT_EXEC_ROLES))
    channel_modes: dict[str, str] = Field(default_factory=dict)
    default_mode: str = DEFAULT_MODE
    channel_languages: dict[str, str] = Field(default_factory=dict)
    default_language: str = DEFAULT_LANGUAGE
    channel_tones: dict[str, str] = Field(default_factory=dict)
    default_tone: str = DEFAULT_TONE
    tone_note: str = ""
    memory_retention_days: int = Field(default_factory=default_retention_days)

    @field_validator("memory_retention_days", mode="before")
    @classmethod
    def _retention(cls, v):
        """1-3650 days; anything else falls back to the default."""
        ok = valid_retention_days(v)
        return default_retention_days() if ok is None else ok

    @field_validator("channel_modes", mode="before")
    @classmethod
    def _modes(cls, v):
        """Keep numeric channel ids with a known mode; drop the rest, bounded like other lists."""
        if not isinstance(v, dict):
            return {}
        out = {str(k).strip(): m for k, m in v.items() if _ID.match(str(k).strip()) and m in MODES}
        return dict(list(out.items())[:MAX_CHANNEL_MODES])

    @field_validator("default_mode", mode="before")
    @classmethod
    def _default_mode(cls, v):
        return v if v in MODES else DEFAULT_MODE

    @field_validator("channel_languages", mode="before")
    @classmethod
    def _languages(cls, v):
        """Keep numeric channel ids with a known language; drop the rest, bounded like modes."""
        if not isinstance(v, dict):
            return {}
        out = {str(k).strip(): m for k, m in v.items() if _ID.match(str(k).strip()) and m in LANGS}
        return dict(list(out.items())[:MAX_CHANNEL_LANGUAGES])

    @field_validator("default_language", mode="before")
    @classmethod
    def _default_language(cls, v):
        return v if v in LANGS else DEFAULT_LANGUAGE

    @field_validator("channel_tones", mode="before")
    @classmethod
    def _tones(cls, v):
        """Keep numeric channel ids with a known tone; drop the rest, bounded like languages."""
        if not isinstance(v, dict):
            return {}
        out = {str(k).strip(): m for k, m in v.items() if _ID.match(str(k).strip()) and m in TONES}
        return dict(list(out.items())[:MAX_CHANNEL_TONES])

    @field_validator("default_tone", mode="before")
    @classmethod
    def _default_tone(cls, v):
        return v if v in TONES else DEFAULT_TONE

    @field_validator("tone_note", mode="before")
    @classmethod
    def _tone_note(cls, v):
        """What the hub may save: one clean line of at most 200 characters that does not try to
        touch permissions, secrets or the persona (otherwise the save is a 422)."""
        raw = " ".join(str(v or "").split())
        if len(raw) > MAX_NOTE:
            raise ValueError(f"tone_note must be at most {MAX_NOTE} characters")
        if note_problem(raw):
            raise ValueError("tone_note is not allowed")
        return clean_note(raw)

    @field_validator("receipt_channel_ids", "ask_channel_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        return _clean_ids(v)

    @field_validator(
        "digest_channel_id", "receipt_home_channel_id", "report_channel_id", mode="before"
    )
    @classmethod
    def _digest_id(cls, v):
        v = "" if v is None else str(v).strip()
        if v and not _ID.match(v):
            raise ValueError("channel id must be numeric")
        return v

    @field_validator(
        "digest_time", "followup_time", "report_time", "morning_brief_time", "evening_brief_time"
    )
    @classmethod
    def _time(cls, v):
        if not _TIME.match(v):
            raise ValueError("digest_time must be HH:MM")
        return v

    @field_validator("exec_role_names", mode="before")
    @classmethod
    def _roles(cls, v):
        out = [str(x).strip() for x in (v or []) if str(x).strip()]
        if len(out) > 20 or any(len(x) > 100 for x in out):
            raise ValueError("too many or too long role names")
        return list(dict.fromkeys(out))


class GuildSettings(GuildSettingsUpdate):
    """A full guild document (what the hub reads). Missing document = these defaults."""

    admin_emails: list[str] = Field(default_factory=list)
    # Role names (besides executives and Manage Server) allowed to send email; edited in Firestore.
    email_senders: list[str] = Field(default_factory=list)
    # Role names (besides executives) allowed to issue quotations in the company's name; edited in
    # Firestore (not in the hub form yet, so a hub save never touches it).
    quote_roles: list[str] = Field(default_factory=list)
    guild_name: str = ""
    icon_url: str = ""
    channels: list[Channel] = Field(default_factory=list)
    installed_at: str = ""
    # Channel ids (text, voice, forum) everyone in the server can see; synced by the bot. Only
    # items from these channels are shareable between workspaces / visible to plain members.
    public_channel_ids: list[str] = Field(default_factory=list)
    active: bool = True
    updated_by: str = ""
    updated_at: str = ""
    # Per-channel languages saved when the whole server was switched, so switching back restores.
    previous_channel_languages: dict[str, str] = Field(default_factory=dict)
    # The default language those overrides were saved under; switching back to it restores them.
    previous_default_language: str = ""
    language_chosen: bool = False
    language_asked_at: str = ""

    @model_validator(mode="before")
    @classmethod
    def _migrate_language_chosen(cls, data):
        """A document that already carries a default_language (set earlier with /language or the
        hub) counts as chosen; no write is needed. A server that never chose reads as `auto`
        (what the hub shows, and what the bot runs on) until it does."""
        if isinstance(data, dict) and "language_chosen" not in data:
            if data.get("default_language") in LANGS:
                data = {**data, "language_chosen": True}
            else:
                data = {**data, "default_language": "auto"}
        elif isinstance(data, dict) and not data["language_chosen"]:
            data = {**data, "default_language": "auto"}
        return data

    @property
    def language_setting(self) -> str:
        """The server's language setting as the bot uses it: the chosen main language, or `auto`
        (follow the words they write, English fallback) until the server has chosen."""
        return self.default_language if self.language_chosen else "auto"  # (see the validator)

    @field_validator("previous_default_language", mode="before")
    @classmethod
    def _previous_default(cls, v):
        return v if v in LANGS else ""

    @field_validator("tone_note", mode="before")
    @classmethod
    def _tone_note(cls, v):
        """A stored note is read leniently: an unsafe or over-long one is dropped, never an error."""
        raw = " ".join(str(v or "").split())
        return "" if len(raw) > MAX_NOTE or note_problem(raw) else clean_note(raw)

    @field_validator("previous_channel_languages", mode="before")
    @classmethod
    def _previous_languages(cls, v):
        return GuildSettingsUpdate._languages(v)


def language_chosen_in(doc: dict | None) -> bool:
    """Has this raw guild document a chosen main language (flag, or a stored default_language)?"""
    d = doc or {}
    return bool(d.get("language_chosen")) or d.get("default_language") in LANGS


class LinkError(Exception):
    """The link code is unknown, expired or already used."""


class GuildStore:
    """Firestore access for guild documents and link codes. The client is created lazily."""

    def __init__(
        self,
        client: Any = None,
        collection: str = "salid_guilds",
        link_collection: str = "salid_link_codes",
        project: str | None = None,
    ) -> None:
        self._client, self._project = client, project
        self.collection, self.link_collection = collection, link_collection

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    def _ref(self, guild_id: int | str) -> Any:
        return self.client.collection(self.collection).document(str(guild_id))

    def _in_transaction(self, fn):
        from google.cloud import firestore

        return firestore.transactional(fn)(self.client.transaction())

    # ------------------------------------------------------------ reads
    def find(self, guild_id: int) -> GuildSettings | None:
        snap = self._ref(guild_id).get()
        return GuildSettings.model_validate(snap.to_dict() or {}) if snap.exists else None

    def get(self, guild_id: int) -> GuildSettings:
        """The stored settings, or defaults when the guild has no document."""
        return self.find(guild_id) or GuildSettings()

    def list_guilds(self, email: str | None = None) -> list[tuple[int, GuildSettings]]:
        """Guilds `email` administers (all guilds when email is None, for super admins)."""
        from google.cloud.firestore_v1.base_query import FieldFilter

        col = self.client.collection(self.collection)
        q = (
            col
            if email is None
            else col.where(filter=FieldFilter("admin_emails", "array_contains", email.lower()))
        )
        out = [(int(s.id), GuildSettings.model_validate(s.to_dict() or {})) for s in q.stream()]
        return sorted(out, key=lambda t: (t[1].guild_name.casefold(), t[0]))

    # ------------------------------------------------------------ bot writes
    def upsert_guild(
        self,
        guild_id: int,
        *,
        name: str,
        icon_url: str,
        channels: list[dict],
        active: bool = True,
        installed: bool = False,
        public_channel_ids: list[str] | None = None,
    ) -> None:
        """Merge what the bot knows about the guild; never touches hub-managed fields."""
        data: dict[str, Any] = {
            "guild_name": name,
            "icon_url": icon_url,
            "channels": channels,
            "active": active,
        }
        if public_channel_ids is not None:
            data["public_channel_ids"] = public_channel_ids
        if installed:
            data["installed_at"] = _now().isoformat(timespec="seconds")
        self._ref(guild_id).set(data, merge=True)

    def set_active(self, guild_id: int, active: bool) -> None:
        self._ref(guild_id).set({"active": active}, merge=True)

    # ------------------------------------------------------------ hub writes
    def save_settings(
        self, guild_id: int, update: GuildSettingsUpdate, updated_by: str
    ) -> GuildSettings:
        data = update.model_dump()
        ref = self._ref(guild_id)
        snap = ref.get()
        if not language_chosen_in(snap.to_dict() if snap.exists else None):
            if data.get("default_language") == "auto":
                # the form only echoed what it showed for a server that never chose: not a choice
                data.pop("default_language", None)
            else:
                data["language_chosen"] = True
        modes = data.pop("channel_modes")
        languages = data.pop("channel_languages")
        tones = data.pop("channel_tones")
        data["updated_by"] = updated_by
        data["updated_at"] = _now().isoformat(timespec="seconds")
        ref.set(data, merge=True)
        # set(merge=True) would deep-merge the map and never remove a channel: replace it whole.
        ref.update({"channel_modes": modes, "channel_languages": languages, "channel_tones": tones})
        return self.get(guild_id)

    def set_channel_mode(self, guild_id: int, channel_id: int, mode: str | None) -> None:
        """Set one channel's mode (None removes the entry so the channel follows the default)."""
        if mode is not None and mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        ref, cid = self._ref(guild_id), str(channel_id)

        def run(txn) -> None:
            snap = ref.get(transaction=txn)
            doc = snap.to_dict() if snap.exists else None
            modes = dict((doc or {}).get("channel_modes") or {})
            if mode is None:
                modes.pop(cid, None)
            else:
                modes[cid] = mode
            if doc is None:
                txn.set(ref, {"channel_modes": modes})
            else:  # update() replaces the whole map, so removals stick
                txn.update(ref, {"channel_modes": modes})

        self._in_transaction(run)

    def set_memory_retention_days(self, guild_id: int, days: int) -> None:
        """Set how long this guild's chat memory is kept (1-3650 days)."""
        ok = valid_retention_days(days)
        if ok is None:
            raise ValueError(f"retention must be {MIN_RETENTION_DAYS}-{MAX_RETENTION_DAYS} days")
        self._ref(guild_id).set({"memory_retention_days": ok}, merge=True)

    def set_channel_language(
        self, guild_id: int, channel_id: int, language: str | None, *, mark_chosen: bool = True
    ) -> None:
        """Set one channel's reply language (None removes the entry: the channel follows the
        guild default). Same map-replace semantics as `set_channel_mode`. `mark_chosen=False` is for
        automatic settings (private channels) that must not answer the main-language question."""
        if language is not None and language not in LANGS:
            raise ValueError(f"unknown language {language!r}")
        ref, cid = self._ref(guild_id), str(channel_id)

        def run(txn) -> None:
            snap = ref.get(transaction=txn)
            doc = snap.to_dict() if snap.exists else None
            langs = dict((doc or {}).get("channel_languages") or {})
            if language is None:
                langs.pop(cid, None)
            else:
                langs[cid] = language
            data: dict = {"channel_languages": langs}
            if mark_chosen and language is not None and not language_chosen_in(doc):
                # a channel command answers the join question: keep following the words people
                # write elsewhere (auto) and never ask again
                data.update(language_chosen=True, default_language="auto")
            if doc is None:
                txn.set(ref, data)
            else:
                txn.update(ref, data)

        self._in_transaction(run)

    def mark_language_asked(self, guild_id: int) -> None:
        """Remember that the main-language question was posted (it is asked only once)."""
        self._ref(guild_id).set(
            {"language_asked_at": _now().isoformat(timespec="seconds")}, merge=True
        )

    def set_channel_tone(self, guild_id: int, channel_id: int, tone: str | None) -> None:
        """Set one channel's tone (None removes the entry: the channel follows the guild default).
        Same map-replace semantics as `set_channel_language`."""
        if tone is not None and tone not in TONES:
            raise ValueError(f"unknown tone {tone!r}")
        ref, cid = self._ref(guild_id), str(channel_id)

        def run(txn) -> None:
            snap = ref.get(transaction=txn)
            doc = snap.to_dict() if snap.exists else None
            tones = dict((doc or {}).get("channel_tones") or {})
            if tone is None:
                tones.pop(cid, None)
            else:
                tones[cid] = tone
            if doc is None:
                txn.set(ref, {"channel_tones": tones})
            else:
                txn.update(ref, {"channel_tones": tones})

        self._in_transaction(run)

    def set_server_tone(self, guild_id: int, tone: str) -> None:
        """Set the guild default tone (per-channel tones stay)."""
        if tone not in TONES:
            raise ValueError(f"unknown tone {tone!r}")
        self._ref(guild_id).set({"default_tone": tone}, merge=True)

    def set_tone_note(self, guild_id: int, note: str) -> None:
        """Set (or, with "", clear) the guild's wording note. Validated like the hub field."""
        note = clean_note(note, MAX_NOTE + 1)
        if note and (len(note) > MAX_NOTE or note_problem(note)):
            raise ValueError("unsafe or too long note")
        self._ref(guild_id).set({"tone_note": note}, merge=True)

    def set_server_language(
        self, guild_id: int, language: str, apply_to_all_channels: bool = True
    ) -> None:
        """Set the guild default language. With `apply_to_all_channels` the per-channel overrides
        are cleared (a copy is kept in `previous_channel_languages` together with the default they
        were saved under); switching back to that default restores the copy. Also marks the
        server's language as chosen."""
        if language not in LANGS or language == "auto":
            raise ValueError(f"unknown language {language!r}")
        ref = self._ref(guild_id)

        def run(txn) -> None:
            snap = ref.get(transaction=txn)
            doc = snap.to_dict() if snap.exists else None
            data: dict = {"default_language": language, "language_chosen": True}
            if apply_to_all_channels:
                d = doc or {}
                current = dict(d.get("channel_languages") or {})
                previous = dict(d.get("previous_channel_languages") or {})
                # overrides saved before previous_default_language existed were saved under Thai
                prev_default = d.get("previous_default_language") or ("th" if previous else "")
                if previous and prev_default == language:
                    data["channel_languages"] = previous
                    data["previous_channel_languages"] = {}
                    data["previous_default_language"] = ""
                else:
                    now_default = d.get("default_language") if language_chosen_in(d) else "auto"
                    data["channel_languages"] = {}
                    if current:
                        data["previous_channel_languages"] = current
                        data["previous_default_language"] = now_default
            if doc is None:
                txn.set(ref, data)
            else:
                txn.update(ref, data)

        self._in_transaction(run)

    # ------------------------------------------------------------ link codes
    def create_link_code(self, guild_id: int, user_id: int, user_name: str) -> str:
        code = secrets.token_urlsafe(32)
        now = _now()
        self.client.collection(self.link_collection).document(code).set(
            {
                "guild_id": guild_id,
                "discord_user_id": user_id,
                "discord_user_name": user_name,
                "created_at": now.isoformat(timespec="seconds"),
                "expires_at": (now + LINK_TTL).isoformat(timespec="seconds"),
                "used": False,
            }
        )
        return code

    def peek_link_code(self, code: str) -> int:
        """The guild a code belongs to, without consuming it. LinkError when unusable."""
        snap = self.client.collection(self.link_collection).document(code).get()
        info = snap.to_dict() if snap.exists else None
        if not info or info.get("used"):
            raise LinkError("code unknown or already used")
        if datetime.fromisoformat(info["expires_at"]) <= _now():
            raise LinkError("code expired")
        return int(info["guild_id"])

    def redeem_link_code(self, code: str, email: str) -> tuple[int, GuildSettings]:
        """Atomically consume `code` and make `email` an admin of its guild.

        Raises LinkError when the code is unknown, used or expired. Runs in one Firestore
        transaction, so two people racing for the same code cannot both win.
        """
        email = email.lower()
        code_ref = self.client.collection(self.link_collection).document(code)

        def run(txn) -> int:
            snap = code_ref.get(transaction=txn)
            info = snap.to_dict() if snap.exists else None
            if not info or info.get("used"):
                raise LinkError("code unknown or already used")
            if datetime.fromisoformat(info["expires_at"]) <= _now():
                raise LinkError("code expired")
            gid = int(info["guild_id"])
            guild_ref = self._ref(gid)
            gsnap = guild_ref.get(transaction=txn)
            admins = list((gsnap.to_dict() or {}).get("admin_emails", [])) if gsnap.exists else []
            if email not in admins:
                admins.append(email)
            txn.update(code_ref, {"used": True, "used_by": email, "used_at": _now().isoformat()})
            txn.set(guild_ref, {"admin_emails": admins}, merge=True)
            return gid

        gid = self._in_transaction(run)
        return gid, self.get(gid)


class MemoryFirestore:
    """Minimal in-memory stand-in for the Firestore calls `GuildStore` makes (local dev, tests).

    Supports collection().document().get/set(merge)/update and a collection stream() with an
    optional array_contains filter.
    """

    class _Snap:
        def __init__(self, doc_id: str, data: dict | None):
            self.id, self._data = doc_id, data

        @property
        def exists(self) -> bool:
            return self._data is not None

        def to_dict(self):
            return None if self._data is None else dict(self._data)

    class _Ref:
        def __init__(self, docs: dict, doc_id: str):
            self.docs, self.id = docs, doc_id

        def get(self, **_):
            return MemoryFirestore._Snap(self.id, self.docs.get(self.id))

        def set(self, data, merge: bool = False):
            self.docs[self.id] = {**self.docs.get(self.id, {}), **data} if merge else dict(data)

        def update(self, fields):
            self.docs[self.id].update(fields)

    class _Col:
        def __init__(self, docs: dict, filters=()):
            self.docs, self.filters = docs, tuple(filters)

        def document(self, doc_id: str):
            return MemoryFirestore._Ref(self.docs, doc_id)

        def where(self, *, filter):
            return MemoryFirestore._Col(self.docs, (*self.filters, filter))

        def stream(self):
            return iter(
                MemoryFirestore._Snap(i, d)
                for i, d in list(self.docs.items())
                if all(f.value in d.get(f.field_path, []) for f in self.filters)
            )

    class _Txn:
        def update(self, ref, fields):
            ref.update(fields)

        def set(self, ref, data, merge: bool = False):
            ref.set(data, merge=merge)

    def __init__(self) -> None:
        self.cols: dict[str, dict] = {}
        self.lock = threading.Lock()

    def collection(self, name: str):
        return MemoryFirestore._Col(self.cols.setdefault(name, {}))


class MemoryGuildStore(GuildStore):
    """`GuildStore` on `MemoryFirestore`: SALID_STORE=sqlite (local dev) and offline tests."""

    def __init__(self, **kw) -> None:
        super().__init__(client=MemoryFirestore(), **kw)

    def _in_transaction(self, fn):
        with self.client.lock:
            return fn(MemoryFirestore._Txn())


class BotGuildSettings:
    """What the bot consults, per guild. Cached for `ttl` seconds and fails open."""

    def __init__(self, store: GuildStore, ttl: float = TTL_S, clock=time.monotonic):
        self.store, self._ttl, self._clock = store, ttl, clock
        # (guild id) -> licensed? Set by the bot; an unlicensed guild has every feature off.
        self.license_gate: Callable[[int], bool] | None = None
        self._cache: dict[int, tuple[float, GuildSettings]] = {}

    def current(self, guild_id: int) -> GuildSettings:
        now = self._clock()
        hit = self._cache.get(guild_id)
        if hit is None or now - hit[0] >= self._ttl:
            try:
                value = self.store.get(guild_id)
            except Exception:
                logger.warning(
                    "could not read settings of guild %s; keeping previous/defaults",
                    guild_id,
                    exc_info=True,
                )
                value = hit[1] if hit else GuildSettings()
            hit = self._cache[guild_id] = (now, value)
        return hit[1]

    def memory_retention_days(self, guild_id: int) -> int:
        return self.current(guild_id).memory_retention_days

    def fresh_memory_retention_days(self, guild_id: int, *, fail_closed: bool = False) -> int:
        """Retention read from the store NOW, bypassing the cache. Read paths that hide expired
        memory use ``fail_closed=True``: the hub writes the store directly, so a failed read must
        not expose data under a stale longer cached value. Purge scheduling keeps the last known
        value instead, because deleting data on an unknown setting would be irreversible."""
        try:
            value = self.store.get(guild_id)
        except Exception:
            logger.warning("could not read retention of guild %s", guild_id, exc_info=True)
            return MIN_RETENTION_DAYS if fail_closed else self.memory_retention_days(guild_id)
        self._cache[guild_id] = (self._clock(), value)
        return value.memory_retention_days

    def set_memory_retention_days(self, guild_id: int, days: int) -> None:
        """Write one guild's retention (validated) and drop the cache so it applies at once."""
        ok = valid_retention_days(days)
        if ok is None:
            raise ValueError(f"retention must be {MIN_RETENTION_DAYS}-{MAX_RETENTION_DAYS} days")
        try:
            self.store.set_memory_retention_days(guild_id, ok)
        finally:
            self._cache.pop(guild_id, None)

    def enabled(self, guild_id: int, feature: str) -> bool:
        if self.license_gate is not None and not self.license_gate(guild_id):
            return False
        return bool(getattr(self.current(guild_id).features, feature))

    def receipt_channel_ids(self, guild_id: int, env_ids: set[int]) -> set[int]:
        return env_ids | {int(x) for x in self.current(guild_id).receipt_channel_ids}

    def ask_channel_ids(self, guild_id: int, env_ids: set[int]) -> set[int]:
        return env_ids | {int(x) for x in self.current(guild_id).ask_channel_ids}

    def channel_mode(
        self, guild_id: int, channel_id: int, channel_name: str | None, env_ask_ids: set[int]
    ) -> str:
        g = self.current(guild_id)
        return resolve_mode(
            channel_id,
            channel_name,
            channel_modes=g.channel_modes,
            ask_channel_ids=self.ask_channel_ids(guild_id, env_ask_ids),
            default_mode=g.default_mode,
        )

    def set_channel_mode(self, guild_id: int, channel_id: int, mode: str | None) -> None:
        """Write one channel's mode and drop the cache so it applies at once."""
        try:
            self.store.set_channel_mode(guild_id, channel_id, mode)
        finally:
            self._cache.pop(guild_id, None)

    def channel_language(self, guild_id: int, channel_id: int) -> str:
        """The channel's reply-language setting: th, en or auto (override, else guild default)."""
        g = self.current(guild_id)
        return resolve_language(
            channel_id,
            channel_languages=g.channel_languages,
            default_language=g.language_setting,
        )

    def language_setting(self, guild_id: int) -> str:
        """The server's main-language setting: the chosen language, `auto` until it has chosen."""
        return self.current(guild_id).language_setting

    def language_chosen(self, guild_id: int) -> bool:
        return self.current(guild_id).language_chosen

    def mark_language_asked(self, guild_id: int) -> None:
        try:
            self.store.mark_language_asked(guild_id)
        finally:
            self._cache.pop(guild_id, None)

    def set_channel_language(
        self, guild_id: int, channel_id: int, language: str | None, *, mark_chosen: bool = True
    ) -> None:
        """Write one channel's language and drop the cache so it applies at once."""
        try:
            self.store.set_channel_language(guild_id, channel_id, language, mark_chosen=mark_chosen)
        finally:
            self._cache.pop(guild_id, None)

    def tone_of(self, guild_id: int, channel_id: int) -> str:
        """The channel's tone for a shared room (override, else guild default). A person's own
        tone is `salid.tone.effective_tone(..., private=True)`."""
        return effective_tone(self.current(guild_id), channel_id)

    def tone_note(self, guild_id: int) -> str:
        return self.current(guild_id).tone_note

    def set_channel_tone(self, guild_id: int, channel_id: int, tone: str | None) -> None:
        try:
            self.store.set_channel_tone(guild_id, channel_id, tone)
        finally:
            self._cache.pop(guild_id, None)

    def set_server_tone(self, guild_id: int, tone: str) -> None:
        try:
            self.store.set_server_tone(guild_id, tone)
        finally:
            self._cache.pop(guild_id, None)

    def set_tone_note(self, guild_id: int, note: str) -> None:
        try:
            self.store.set_tone_note(guild_id, note)
        finally:
            self._cache.pop(guild_id, None)

    def set_server_language(
        self, guild_id: int, language: str, apply_to_all_channels: bool = True
    ) -> None:
        """Switch the whole server's language and drop the cache so it applies at once."""
        try:
            self.store.set_server_language(guild_id, language, apply_to_all_channels)
        finally:
            self._cache.pop(guild_id, None)

    def digest_channel_id(self, guild_id: int, env_id: int | None) -> int | None:
        v = self.current(guild_id).digest_channel_id
        return int(v) if v else env_id

    def digest_time(self, guild_id: int, env_time: str) -> str:
        g = self.current(guild_id)
        # Defaults apply until the hub has saved the document (every save sets updated_at).
        return g.digest_time if g.updated_at else env_time

    def followup_time(self, guild_id: int) -> str:
        return self.current(guild_id).followup_time

    def report_time(self, guild_id: int) -> str:
        return self.current(guild_id).report_time

    def brief_times(self, guild_id: int) -> tuple[str, str]:
        """(morning, evening) default brief times of the guild (HH:MM)."""
        g = self.current(guild_id)
        return g.morning_brief_time, g.evening_brief_time

    def report_channel_id(self, guild_id: int) -> int | None:
        v = self.current(guild_id).report_channel_id
        return int(v) if v else None

    def quote_roles(self, guild_id: int) -> set[str]:
        return set(self.current(guild_id).quote_roles)

    def exec_role_names(self, guild_id: int, env_names: set[str]) -> set[str]:
        g = self.current(guild_id)
        return set(g.exec_role_names) if g.updated_at else set(env_names)


def make_guild_store(settings) -> GuildStore:
    """Firestore only when SALID_STORE=firestore; otherwise an in-memory store."""
    if settings.store != "firestore":
        return MemoryGuildStore(collection=settings.guild_collection)
    return GuildStore(collection=settings.guild_collection, project=settings.gcp_project)
