"""Platform connectors (Discord, Slack, ClickUp) behind one seam; see `base.Connector`."""

from .base import Connection, Connector, ConnectorConfig, ConnectorError, NotConfigured
from .clickup import ClickUpConnector
from .discord import DiscordConnector
from .slack import SlackConnector, verify_signature

COMING_SOON = ("google_drive", "gmail", "google_calendar")

__all__ = [
    "COMING_SOON",
    "ClickUpConnector",
    "Connection",
    "Connector",
    "ConnectorConfig",
    "ConnectorError",
    "DiscordConnector",
    "NotConfigured",
    "SlackConnector",
    "verify_signature",
]
