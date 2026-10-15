"""Salid runtime settings, loaded from `SALID_*` environment variables and `.env`."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Every variable is namespaced with SALID_ so it cannot clash with meeting_scribe's
    # settings (e.g. STORE, GCP_PROJECT) when both packages share one `.env`.
    model_config = SettingsConfigDict(env_prefix="SALID_", env_file=".env", extra="ignore")

    discord_token: str = ""
    # Extra voice-only bot tokens (comma-separated, may be empty): "Salid · 2", "Salid · 3"...
    # Secrets: never logged, never in repr.
    voice_tokens: Annotated[list[str], NoDecode] = Field(default_factory=list, repr=False)
    # Comma-separated in the environment (NoDecode: skip pydantic-settings' JSON parsing).
    receipt_channel_ids: Annotated[set[int], NoDecode] = set()
    ask_channel_ids: Annotated[set[int], NoDecode] = set()
    exec_role_names: Annotated[set[str], NoDecode] = {"ผู้บริหาร", "Executive"}
    exec_user_ids: Annotated[set[int], NoDecode] = set()
    digest_time: str = "18:00"  # HH:MM in `tz`
    digest_channel_id: int | None = None
    store: str = "sqlite"  # sqlite | firestore
    firestore_collection: str = "salid_receipts"
    # Optional original-receipt archive: bucket name (empty = disabled), object prefix.
    receipt_bucket: str = ""
    receipt_prefix: str = "salid-receipts/"
    db_path: str = "./data/ledger.db"
    gcp_project: str = "meeting-scribe-hackathon"
    gemini_location: str = "global"
    gemini_model: str = "gemini-3.5-flash"
    tz: str = "Asia/Bangkok"
    # Deep tasks (Gemini CLI, an external Node binary). Empty path = feature disabled.
    gemini_cli_path: str = ""
    deep_task_timeout: int = 600  # seconds
    deep_task_model: str = "gemini-3-flash-preview"  # see deep_task.build_cli_settings
    # Cheap model for the "is this addressed to Salid?" yes/no gates (see salid.gate).
    gate_model: str = "gemini-3.5-flash-lite"
    low_confidence: float = 0.7
    # Spoken replies in voice meetings (see salid.speech). The TTS call reuses the bot's Vertex
    # client, so it runs in `gemini_location` (global; the Gemini TTS models work there).
    voice_reply: bool = True
    # "cloud": Cloud Text-to-Speech Chirp 3 HD (~1.5 s). "gemini": Gemini TTS (~6 s, also the
    # fallback when Cloud TTS fails).
    tts_backend: str = "cloud"
    cloud_tts_voice: str = "Leda"  # brightest Thai Chirp 3 HD voice, for the cute-cat persona
    tts_speaking_rate: float = 1.08
    tts_pitch: float = 1.06  # ffmpeg pitch-up factor at playback (1.0 = unchanged)
    tts_model: str = "gemini-2.5-flash-tts"
    tts_voice: str = "Leda"
    # How Salid talks in a voice meeting (needs voice_reply). "live": a Gemini Live native-audio
    # session hears the room and answers ~1.5 s after someone stops talking (see salid.live_voice).
    # "classic": wake word -> transcription -> text answer -> spoken rewrite -> TTS (~11 s).
    voice_mode: str = "live"  # live | classic
    live_model: str = "gemini-live-2.5-flash-native-audio"
    live_location: str = "us-central1"
    live_voice: str = "Leda"
    # Voice-meeting transcription runs every few seconds: use the Singapore endpoint (own quota,
    # close to Discord's sin voice servers). Empty = reuse the bot's client (gemini_location).
    transcribe_location: str = "asia-southeast1"
    # Organisational memory (salid.memory): raw chat log + per-channel day summaries.
    memory_retention_days: int = 90
    memory_update_every: int = 30  # new messages in a channel that trigger a summary update
    memory_quiet_minutes: int = 20  # quiet time after which any new messages are summarised
    memory_messages_collection: str = "salid_messages"
    memory_summaries_collection: str = "salid_day_summaries"
    memory_channels_collection: str = "salid_memory_channels"  # history-import state per channel
    memory_state_collection: str = "salid_memory_guilds"  # import budgets per guild
    # History import (see salid.memory_import)
    memory_import_guild_cap: int = 20000  # messages imported per guild per run
    memory_import_images_per_day: int = 300  # historical images described per guild per day
    memory_import_summaries_per_day: int = 200  # past-day summaries generated per guild per day
    # Audio/video understanding (salid.media): minutes of unaddressed audio transcribed per guild/day.
    media_minutes_per_day: int = 120
    # Documents (salid.docs): pages of PDF/office/text files read into memory per guild/day.
    doc_pages_per_day: int = 500
    # Per-guild settings document `{guild_collection}/{guild_id}` (see salid.guild_settings).
    guild_collection: str = "salid_guilds"
    # PDF documents (salid.pdf_report): regular TTF/OTF with Thai glyphs; empty = search known paths.
    pdf_font: str = ""
    # Scheduled meetings (salid.scheduling)
    meeting_remind_minutes: int = 10  # reminder before the start (and again at the start)
    meeting_noshow_minutes: int = 5  # DM attendees missing from the voice channel after this
    meeting_empty_minutes: int = 2  # an auto-joined meeting nobody ever came to ends after this
    # A meeting where people were present ends only after this long with no humans in the channel.
    meeting_leave_empty_minutes: int = 5
    # Auto-join (salid.voice_pool): Salid joins a voice channel as soon as `auto_join_min_humans`
    # people are in it. `auto_join_delay_seconds` > 0 adds a stability wait (default: none).
    auto_join_min_humans: int = 1
    auto_join_delay_seconds: float = 0.0
    auto_join_cooldown_minutes: int = 60  # no auto-join where someone told Salid to leave
    # Email (salid.email_out): Gmail sends only when all three are set (refresh token = secret)
    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_refresh_token: str = ""
    gmail_sender: str = ""  # the From address (the dedicated Google account)
    email_daily_limit: int = 20  # emails per guild per day
    # Briefings (salid.briefings): documents above this total (invoice / quotation) are "important",
    # an unanswered request to a member is reminded after this many hours.
    doc_alert_min_amount: float = 50000.0
    ask_reminder_hours: float = 2.0
    doc_alerts_per_day: int = 10  # real-time document alerts per person per day
    # Where `/ตั้งค่า` links point: the web service that serves /hub.
    hub_url: str = "https://meeting-scribe-638191001920.asia-southeast1.run.app"

    @field_validator("receipt_channel_ids", "ask_channel_ids", "exec_user_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        if isinstance(v, str):
            return {int(x) for x in v.replace(" ", "").split(",") if x.strip().isdigit()}
        return v

    @field_validator("voice_tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("exec_role_names", mode="before")
    @classmethod
    def _parse_names(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",") if x.strip()}
        return v

    @field_validator("digest_channel_id", mode="before")
    @classmethod
    def _parse_optional_id(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return int(v) if v.isdigit() else None
        return v

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def now(self) -> datetime:
        return datetime.now(self.zone)

    def today(self) -> date:
        return self.now().date()

    @classmethod
    def from_env(cls) -> Settings:
        return cls()

    def is_exec(self, user_id: int, role_names: list[str] | set[str] = ()) -> bool:
        return user_id in self.exec_user_ids or bool(self.exec_role_names & set(role_names))
