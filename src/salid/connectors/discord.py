"""Discord: wraps today's flow (invite link + one-time `/ตั้งค่า` link code), no OAuth.

Always available once `SALID_DISCORD_CLIENT_ID` is set. Connecting a server = invite the bot, run
`/ตั้งค่า` in it, paste the code in the command center (`POST .../workspaces/discord/link`).
"""

from __future__ import annotations

from .base import ConnectorConfig, ConnectorError


class DiscordConnector:
    platform = "discord"

    def __init__(self, cfg: ConnectorConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        return bool(self.cfg.discord_client_id)

    def invite_url(self) -> str:
        from meeting_scribe.app import salid_invite_url

        return salid_invite_url(self.cfg.discord_client_id)

    def oauth_start(self, org_id: str, state: str = "") -> str:
        return self.invite_url()

    def oauth_callback(self, params):  # the link code flow replaces it
        raise ConnectorError("Discord is connected with a /ตั้งค่า link code")

    def list_channels(self, token: str, external_id: str) -> list[dict]:
        return []  # channels are synced by the bot itself (GuildSettings.channels)

    def ingest(self, event: dict) -> int:
        return 0  # the bot ingests Discord messages directly

    def send(self, token: str, channel: str, text: str) -> None:
        raise ConnectorError("Discord messages are sent by the bot, not the web service")
