"""Slack: OAuth v2 install, Events API with signing-secret verification, ingest into memory.

Enabled only when `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET` and `SLACK_SIGNING_SECRET` are set.

ID mapping (documented contract, see `salid.orgs`):
* workspace "guild id" = `workspace_memory_id("slack", team_id)` (namespaced, >= 2**62);
* channel id           = `namespaced_id("slack", channel_id, "channel")`;
* author id            = `namespaced_id("slack", user_id, "user")`;
* message id           = `message_namespaced_id("slack", team_id, channel_id, ts)` (the store keys
  messages by guild + message id with no channel, and a `ts` is only unique per channel, so the
  hash includes the channel).
The Salid bot (`salid.platforms.ids`) calls the very same functions, so one Slack workspace is one
tenant: one ledger, one memory, one licence, whichever integration saw the message.
Only PUBLIC channel messages (`channel_type == "channel"`) are ingested; private channels, group
and direct messages are ignored. Each ingested channel is registered under `SLACK_CHANNELS` so the
web chat can read it.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import UTC, datetime
from urllib.parse import urlencode

import httpx

from salid.memory_store import MemoryMessage
from salid.orgs import message_namespaced_id, namespaced_id, workspace_memory_id

from .base import Connection, ConnectorConfig, ConnectorError, NotConfigured, http_client

logger = logging.getLogger(__name__)

SLACK_CHANNELS = "salid_slack_channels"  # guild = workspace memory id, key = channel int id
AUTHORIZE = "https://slack.com/oauth/v2/authorize"
TOKEN_URL = "https://slack.com/api/oauth.v2.access"
API = "https://slack.com/api"
SCOPES = "channels:history,channels:read,chat:write,users:read"
REPLAY_WINDOW_S = 300


def verify_signature(
    signing_secret: str, body: bytes, timestamp: str, signature: str, now: float | None = None
) -> bool:
    """Slack request signing (v0): HMAC-SHA256 of `v0:{timestamp}:{body}`, 5-minute replay window."""
    if not signing_secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs((time.time() if now is None else now) - ts) > REPLAY_WINDOW_S:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class SlackConnector:
    platform = "slack"

    def __init__(self, cfg: ConnectorConfig, memory=None, docs=None, zone=None, client=None):
        self.cfg, self.memory, self.docs, self.zone = cfg, memory, docs, zone
        self._client = client

    @property
    def http(self) -> httpx.Client:
        if self._client is None:
            self._client = http_client()
        return self._client

    def enabled(self) -> bool:
        c = self.cfg
        return bool(c.slack_client_id and c.slack_client_secret and c.slack_signing_secret)

    def _need(self) -> None:
        if not self.enabled():
            raise NotConfigured("Slack is not configured")

    # ---- OAuth
    def oauth_start(self, org_id: str, state: str = "") -> str:
        self._need()
        q = {
            "client_id": self.cfg.slack_client_id,
            "scope": SCOPES,
            "redirect_uri": self.cfg.redirect_uri("slack"),
            "state": state or org_id,
        }
        return f"{AUTHORIZE}?{urlencode(q)}"

    def oauth_callback(self, params: dict[str, str]) -> list[Connection]:
        self._need()
        if params.get("error") or not params.get("code"):
            raise ConnectorError("Slack authorization was cancelled")
        try:
            resp = self.http.post(
                TOKEN_URL,
                data={
                    "client_id": self.cfg.slack_client_id,
                    "client_secret": self.cfg.slack_client_secret,
                    "code": params["code"],
                    "redirect_uri": self.cfg.redirect_uri("slack"),
                },
            )
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ConnectorError("Could not reach Slack") from exc
        team = data.get("team") or {}
        if not data.get("ok") or not team.get("id") or not data.get("access_token"):
            raise ConnectorError("Slack refused the authorization")
        return [
            Connection(
                "slack",
                str(team["id"]),
                str(team.get("name") or team["id"]),
                token=str(data["access_token"]),
                extra={"bot_user_id": data.get("bot_user_id", "")},
            )
        ]

    # ---- API
    def list_channels(self, token: str, external_id: str) -> list[dict]:
        r = self.http.get(
            f"{API}/conversations.list",
            params={"types": "public_channel", "limit": 200},
            headers={"Authorization": f"Bearer {token}"},
        )
        data = r.json()
        if not data.get("ok"):
            raise ConnectorError("Slack refused the channel list")
        return [{"id": c["id"], "name": c.get("name", "")} for c in data.get("channels", [])]

    def send(self, token: str, channel: str, text: str) -> None:
        r = self.http.post(
            f"{API}/chat.postMessage",
            json={"channel": channel, "text": text},
            headers={"Authorization": f"Bearer {token}"},
        )
        if not r.json().get("ok"):
            raise ConnectorError("Slack refused the message")

    # ---- events
    def verify(self, body: bytes, timestamp: str, signature: str, now: float | None = None) -> bool:
        return verify_signature(self.cfg.slack_signing_secret, body, timestamp, signature, now)

    def ingest(self, event: dict) -> int:
        """One Events API `event_callback` payload -> memory. Returns messages stored (0 or 1)."""
        team = str(event.get("team_id") or "")
        ev = event.get("event") or {}
        if not team or ev.get("type") != "message" or self.memory is None:
            return 0
        if ev.get("subtype") or ev.get("bot_id") or not ev.get("text"):
            return 0  # edits, joins, bot posts: not ingested
        if ev.get("channel_type") != "channel":
            return 0  # only public channels
        ts = str(ev.get("ts") or "")
        if not ts:
            return 0
        wid = workspace_memory_id("slack", team)
        channel = namespaced_id("slack", str(ev["channel"]), "channel")
        at = datetime.fromtimestamp(float(ts), UTC)
        zone = self.zone or UTC
        self.memory.add_message(
            MemoryMessage(
                guild_id=wid,
                channel_id=channel,
                message_id=message_namespaced_id("slack", team, str(ev["channel"]), ts),
                author_id=namespaced_id("slack", str(ev.get("user", "")), "user"),
                author_name=str(ev.get("user", "")),
                created_at=at,
                local_date=at.astimezone(zone).date().isoformat(),
                text=str(ev["text"])[:4000],
                kind="chat",
            )
        )
        if self.docs is not None and self.docs.get(SLACK_CHANNELS, wid, str(channel)) is None:
            self.docs.put(SLACK_CHANNELS, wid, str(channel), {"slack_id": ev["channel"]})
        return 1
