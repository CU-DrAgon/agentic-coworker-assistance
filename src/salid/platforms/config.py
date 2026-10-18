"""Settings of the chat-platform process (`python -m salid.platforms`).

Kept apart from `salid.config.Settings` (which other people edit) but with the same `SALID_`
prefix and `.env` file. A platform starts only when its secrets are set. Secrets are `repr=False`
and never logged.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SALID_", env_file=".env", extra="ignore")

    # Slack: Socket Mode, so there is no public endpoint.
    slack_bot_token: str = Field(default="", repr=False)  # xoxb-...
    slack_app_token: str = Field(default="", repr=False)  # xapp-... (connections:write)
    slack_exec_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set)
    # User groups (handles like "leadership" or ids like "S0123ABC") whose members are
    # executives; needs usergroups:read. Workspace admins / owners always count.
    slack_exec_usergroups: Annotated[set[str], NoDecode] = Field(default_factory=set)
    slack_import: bool = True  # read channel history within the retention window
    # Remember the AI notes canvas of a huddle (needs canvases:read + the file_shared event). Off
    # for a workspace that must not store huddle transcripts.
    slack_huddle_notes: bool = True

    # LINE Messaging API: a webhook served on `line_host:line_port`.
    line_channel_secret: str = Field(default="", repr=False)
    line_channel_access_token: str = Field(default="", repr=False)
    line_exec_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set)
    line_host: str = (
        "127.0.0.1"  # put TLS in front (reverse proxy / tunnel); 0.0.0.0 in a container
    )
    line_port: int = 8091
    line_reply_token_seconds: float = (
        50.0  # older than this: push instead of a reply token (token lives ~1 min)
    )

    # Telegram Bot API: long polling, so there is no public endpoint.
    telegram_bot_token: str = Field(default="", repr=False)  # from @BotFather
    telegram_exec_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set)

    # WhatsApp Business Platform (Cloud API): a webhook served on `whatsapp_host:whatsapp_port`.
    whatsapp_access_token: str = Field(default="", repr=False)
    whatsapp_app_secret: str = Field(default="", repr=False)  # signs every webhook POST
    whatsapp_verify_token: str = Field(default="", repr=False)  # answers the GET handshake
    whatsapp_phone_number_id: str = ""  # the business number's id = the tenant
    whatsapp_api_version: str = "v26.0"
    whatsapp_exec_numbers: Annotated[set[str], NoDecode] = Field(default_factory=set)
    whatsapp_host: str = "127.0.0.1"  # TLS in front (Meta only calls https); 0.0.0.0 in a container
    whatsapp_port: int = 8092

    # Optional: the Discord bot token, ONLY to read guild roles / channel overwrites over REST so a
    # verified member's cross-platform questions can use the Discord channels they may read
    # (org_scope). Absent: Discord is left out of that scope. Never logged.
    discord_token: str = Field(default="", repr=False)

    # Google Calendar + Meet in chat for verified members who connected it on the web. Same env
    # names as the web service (no SALID_ prefix for the first three); all four must be set.
    google_oauth_client_id: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_OAUTH_CLIENT_ID", "google_oauth_client_id"),
    )
    google_oauth_client_secret: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("GOOGLE_OAUTH_CLIENT_SECRET", "google_oauth_client_secret"),
    )
    google_oauth_redirect_uri: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_OAUTH_REDIRECT_URI", "google_oauth_redirect_uri"),
    )
    salid_token_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("SALID_TOKEN_KEY", "salid_token_key"),
    )

    # Behaviour shared by all platforms.
    platform_window_seconds: float = 180.0  # follow-ups of the person Salid just answered
    platform_thread_seconds: float = 3600.0  # a thread Salid answered in keeps counting this long
    platform_answers_per_minute: int = 8  # per room; more are dropped (loop protection)
    platform_slow_after_seconds: float = 6.0  # "one moment" only if no answer by then
    platform_max_concurrent: int = 6  # model calls running at once

    @field_validator(
        "slack_exec_user_ids",
        "slack_exec_usergroups",
        "line_exec_user_ids",
        "telegram_exec_user_ids",
        "whatsapp_exec_numbers",
        mode="before",
    )
    @classmethod
    def _csv(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",") if x.strip()}
        return v

    @field_validator("whatsapp_exec_numbers", mode="after")
    @classmethod
    def _digits(cls, v: set[str]) -> set[str]:
        """+66 81 234 5678, 66812345678 and (66)812345678 are one wa_id."""
        return {d for d in ("".join(c for c in x if c.isdigit()) for x in v) if d}

    @field_validator("whatsapp_api_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not re.fullmatch(r"v\d{1,3}\.\d{1,3}", v):  # it becomes part of a URL path
            raise ValueError("SALID_WHATSAPP_API_VERSION must look like v26.0")
        return v

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token and self.slack_app_token)

    @property
    def line_enabled(self) -> bool:
        return bool(self.line_channel_secret and self.line_channel_access_token)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token)

    @property
    def whatsapp_enabled(self) -> bool:
        return bool(
            self.whatsapp_access_token
            and self.whatsapp_app_secret
            and self.whatsapp_verify_token
            and self.whatsapp_phone_number_id
        )

    def enabled_platforms(self) -> list[str]:
        flags = (
            ("slack", self.slack_enabled),
            ("line", self.line_enabled),
            ("telegram", self.telegram_enabled),
            ("whatsapp", self.whatsapp_enabled),
        )
        return [n for n, on in flags if on]
