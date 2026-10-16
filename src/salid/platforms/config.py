"""Settings of the chat-platform process (`python -m salid.platforms`).

Kept apart from `salid.config.Settings` (which other people edit) but with the same `SALID_`
prefix and `.env` file. A platform starts only when its secrets are set. Secrets are `repr=False`
and never logged.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SALID_", env_file=".env", extra="ignore")

    # Slack: Socket Mode, so there is no public endpoint.
    slack_bot_token: str = Field(default="", repr=False)  # xoxb-...
    slack_app_token: str = Field(default="", repr=False)  # xapp-... (connections:write)
    slack_exec_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set)
    slack_import: bool = True  # read channel history within the retention window

    # LINE Messaging API: a webhook served on `line_host:line_port`.
    line_channel_secret: str = Field(default="", repr=False)
    line_channel_access_token: str = Field(default="", repr=False)
    line_exec_user_ids: Annotated[set[str], NoDecode] = Field(default_factory=set)
    line_host: str = (
        "127.0.0.1"  # put TLS in front (reverse proxy / tunnel); 0.0.0.0 in a container
    )
    line_port: int = 8091
    line_reply_token_seconds: float = 25.0  # older than this: push instead of a reply token

    # Behaviour shared by both platforms.
    platform_window_seconds: float = 180.0  # follow-ups of the person Salid just answered
    platform_thread_seconds: float = 3600.0  # a thread Salid answered in keeps counting this long
    platform_answers_per_minute: int = 8  # per room; more are dropped (loop protection)
    platform_slow_after_seconds: float = 6.0  # "one moment" only if no answer by then
    platform_max_concurrent: int = 6  # model calls running at once

    @field_validator("slack_exec_user_ids", "line_exec_user_ids", mode="before")
    @classmethod
    def _csv(cls, v):
        if isinstance(v, str):
            return {x.strip() for x in v.split(",") if x.strip()}
        return v

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token and self.slack_app_token)

    @property
    def line_enabled(self) -> bool:
        return bool(self.line_channel_secret and self.line_channel_access_token)

    def enabled_platforms(self) -> list[str]:
        return [n for n, on in (("slack", self.slack_enabled), ("line", self.line_enabled)) if on]
