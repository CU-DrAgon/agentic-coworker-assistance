"""Multi-platform seam: every chat / work platform Salid can be connected to is a `Connector`.

A connector knows how to start and finish the platform's OAuth flow, list its channels, ingest an
incoming event into Salid's memory and send a message. Connectors are disabled unless their
credentials are configured (environment variables, see each module), and none of them ever logs or
returns a token.

Non-Discord data lives in the same stores as Discord data under a namespaced "guild id", see
`salid.orgs.workspace_memory_id` / `namespaced_id`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

TOKENS = "salid_connector_tokens"  # guild = workspace memory id, key = platform: {access_token}


class ConnectorError(Exception):
    """OAuth / API failure (message safe to show; never contains a secret)."""


class NotConfigured(ConnectorError):
    """The connector's credentials are not set."""


@dataclass
class ConnectorConfig:
    """Credentials come from the environment (via `meeting_scribe.config.Settings`)."""

    base_url: str = ""  # public URL of the web service, no trailing slash
    slack_client_id: str = ""
    slack_client_secret: str = ""
    slack_signing_secret: str = ""
    clickup_client_id: str = ""
    clickup_client_secret: str = ""
    discord_client_id: str = ""

    def redirect_uri(self, platform: str) -> str:
        return f"{self.base_url.rstrip('/')}/api/center/connectors/{platform}/callback"


@dataclass
class Connection:
    """Result of a finished OAuth flow: what to attach as a workspace."""

    platform: str
    external_id: str
    name: str
    token: str = field(default="", repr=False)
    extra: dict[str, Any] = field(default_factory=dict)


class Connector(Protocol):
    platform: str

    def enabled(self) -> bool: ...
    def oauth_start(self, org_id: str, state: str = "") -> str: ...
    def oauth_callback(self, params: dict[str, str]) -> list[Connection]: ...
    def list_channels(self, token: str, external_id: str) -> list[dict]: ...
    def ingest(self, event: dict) -> int: ...
    def send(self, token: str, channel: str, text: str) -> None: ...


def http_client() -> httpx.Client:
    return httpx.Client(timeout=10.0)
