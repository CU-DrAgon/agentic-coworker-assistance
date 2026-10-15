"""Per-team Salid settings, edited in the web hub and obeyed by the bot.

One Firestore document, `{collection}/{team_id}` (default `salid_teams/default`):

    features             {receipts, qa, digest, deep_tasks}: bool, all default true
    receipt_channel_ids  list[str] of numeric Discord channel ids
    ask_channel_ids      list[str]
    digest_channel_id    str ("" = unset)
    digest_time          "HH:MM"
    updated_by, updated_at

The hub reads and writes it through `TeamSettingsStore`. The bot only reads, through
`BotTeamSettings`, which caches for `TTL_S` seconds and fails open: a missing document, a read
error, or `SALID_STORE=sqlite` (local dev) means every feature on and the env settings untouched.

Channel rule: when the document lists channel ids they are *added* to the env lists
(`SALID_RECEIPT_CHANNEL_IDS` / `SALID_ASK_CHANNEL_IDS`); `digest_channel_id` and `digest_time`
*override* the env values when set. Nothing the hub saves can remove an env-configured channel.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

TTL_S = 60.0
FEATURES = ("receipts", "qa", "digest", "deep_tasks")
_ID = re.compile(r"^\d{1,25}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class Features(BaseModel):
    receipts: bool = True
    qa: bool = True
    digest: bool = True
    deep_tasks: bool = True


class TeamSettings(BaseModel):
    features: Features = Field(default_factory=Features)
    receipt_channel_ids: list[str] = Field(default_factory=list, max_length=50)
    ask_channel_ids: list[str] = Field(default_factory=list, max_length=50)
    digest_channel_id: str = ""
    digest_time: str = "18:00"
    updated_by: str = ""
    updated_at: str = ""

    @field_validator("receipt_channel_ids", "ask_channel_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        out = [str(x).strip() for x in (v or []) if str(x).strip()]
        if any(not _ID.match(x) for x in out):
            raise ValueError("channel ids must be numeric")
        return list(dict.fromkeys(out))

    @field_validator("digest_channel_id", mode="before")
    @classmethod
    def _digest_id(cls, v):
        v = "" if v is None else str(v).strip()
        if v and not _ID.match(v):
            raise ValueError("digest_channel_id must be numeric")
        return v

    @field_validator("digest_time")
    @classmethod
    def _time(cls, v):
        if not _TIME.match(v):
            raise ValueError("digest_time must be HH:MM")
        return v


class TeamSettingsStore:
    """Firestore access for the settings document. The client is created lazily."""

    def __init__(
        self,
        client: Any = None,
        collection: str = "salid_teams",
        team_id: str = "default",
        project: str | None = None,
    ) -> None:
        self._client, self._project = client, project
        self.collection, self.team_id = collection, team_id

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    @property
    def _doc(self) -> Any:
        return self.client.collection(self.collection).document(self.team_id)

    def get(self) -> TeamSettings:
        """The stored settings, or defaults when the document does not exist."""
        snap = self._doc.get()
        return TeamSettings.model_validate(snap.to_dict() or {}) if snap.exists else TeamSettings()

    def save(self, settings: TeamSettings, updated_by: str) -> TeamSettings:
        saved = settings.model_copy(
            update={
                "updated_by": updated_by,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        self._doc.set(saved.model_dump())
        return saved


class BotTeamSettings:
    """What the bot consults. Cached, and fails open (see the module docstring)."""

    def __init__(self, store: TeamSettingsStore | None, ttl: float = TTL_S, clock=time.monotonic):
        self._store, self._ttl, self._clock = store, ttl, clock
        self._cached: TeamSettings | None = None
        self._at = 0.0

    def current(self) -> TeamSettings:
        if self._store is None:
            return TeamSettings()
        now = self._clock()
        if self._cached is None or now - self._at >= self._ttl:
            try:
                self._cached = self._store.get()
            except Exception:
                logger.warning(
                    "could not read team settings; keeping previous/defaults", exc_info=True
                )
                self._cached = self._cached or TeamSettings()
            self._at = now
        return self._cached

    def enabled(self, feature: str) -> bool:
        return bool(getattr(self.current().features, feature))

    def receipt_channel_ids(self, env_ids: set[int]) -> set[int]:
        return env_ids | {int(x) for x in self.current().receipt_channel_ids}

    def ask_channel_ids(self, env_ids: set[int]) -> set[int]:
        return env_ids | {int(x) for x in self.current().ask_channel_ids}

    def digest_channel_id(self, env_id: int | None) -> int | None:
        v = self.current().digest_channel_id
        return int(v) if v else env_id

    def digest_time(self, env_time: str) -> str:
        t = self.current()
        # An untouched document keeps the model default; only treat it as an override when the
        # document actually exists (updated_at is set by every save).
        return t.digest_time if t.updated_at else env_time


def make_bot_team_settings(settings) -> BotTeamSettings:
    """Firestore-backed only when SALID_STORE=firestore; otherwise everything stays on."""
    if settings.store != "firestore":
        return BotTeamSettings(None)
    return BotTeamSettings(
        TeamSettingsStore(
            collection=settings.team_collection,
            team_id=settings.team_id,
            project=settings.gcp_project,
        )
    )
