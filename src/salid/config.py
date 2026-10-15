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
    # Spoken replies in voice meetings (see salid.speech). The TTS call reuses the bot's Vertex
    # client, so it runs in `gemini_location` (global; the Gemini TTS models work there).
    voice_reply: bool = True
    tts_model: str = "gemini-2.5-flash-tts"
    tts_voice: str = "Puck"  # male voice, matches the persona (ends with ครับ)
    # Per-guild settings document `{guild_collection}/{guild_id}` (see salid.guild_settings).
    guild_collection: str = "salid_guilds"
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
