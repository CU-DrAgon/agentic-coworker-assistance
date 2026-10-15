"""Salid runtime settings, loaded from `SALID_*` environment variables and `.env`."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Every variable is namespaced with SALID_ so it cannot clash with meeting_scribe's
    # settings (e.g. STORE, GCP_PROJECT) when both packages share one `.env`.
    model_config = SettingsConfigDict(env_prefix="SALID_", env_file=".env", extra="ignore")

    discord_token: str = ""
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
    # Per-guild settings document `{guild_collection}/{guild_id}` (see salid.guild_settings).
    guild_collection: str = "salid_guilds"
    # PDF documents (salid.pdf_report): regular TTF/OTF with Thai glyphs; empty = search known paths.
    pdf_font: str = ""
    # Scheduled meetings (salid.scheduling)
    meeting_remind_minutes: int = 10  # reminder before the start (and again at the start)
    meeting_noshow_minutes: int = 5  # DM attendees missing from the voice channel after this
    meeting_empty_minutes: int = 2  # an auto-joined meeting ends after this long with nobody in it
    # Email (salid.email_out): Gmail sends only when all three are set (refresh token = secret)
    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_refresh_token: str = ""
    gmail_sender: str = ""  # the From address (the dedicated Google account)
    email_daily_limit: int = 20  # emails per guild per day
    # Where `/ตั้งค่า` links point: the web service that serves /hub.
    hub_url: str = "https://meeting-scribe-638191001920.asia-southeast1.run.app"

    @field_validator("receipt_channel_ids", "ask_channel_ids", "exec_user_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        if isinstance(v, str):
            return {int(x) for x in v.replace(" ", "").split(",") if x.strip().isdigit()}
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
