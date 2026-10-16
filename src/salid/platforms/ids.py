"""Stable integer ids for Slack / LINE objects, in ranges that cannot collide with Discord.

Salid's stores key everything by integers (guild, channel, user, message). Slack and LINE ids are
strings, so each one is mapped to an int by hashing; nothing needs to be persisted and the same
input always gives the same id.

* **Tenant** (`tenant_id`, used as `guild_id` everywhere): `2**62 | platform << 60 | hash60`.
  Discord snowflakes are `timestamp_ms_since_2015 << 22`; they stay below `2**62` until about
  October 2049, so ids at or above `2**62` are never Discord guilds. Bits 60-61 hold the platform
  (1 Slack, 2 LINE), the other 60 bits a SHA-256 prefix of the workspace / channel key (a collision
  among N tenants has probability about N**2 / 2**61: negligible). The value fits a signed int64
  (Firestore, SQLite).
* **User / room / message** ids are positive and below `2**62`. They only need to be unique inside
  one tenant (stores always filter by tenant). Negative ids are never produced here: they are
  reserved for the private "DM notes" scope (`memory_access.dm_scope`).
"""

from __future__ import annotations

import hashlib

DISCORD_CEILING = 1 << 62  # Discord snowflakes are below this until ~2049
SLACK, LINE = "slack", "line"
_PLATFORM_CODE = {SLACK: 1, LINE: 2}
_LOW60 = (1 << 60) - 1
_LOW62 = (1 << 62) - 1


def _digest(*parts: str) -> int:
    raw = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def tenant_id(platform: str, key: str) -> int:
    """The `guild_id` of a Slack workspace (`team_id`) or a LINE channel (the bot's user id)."""
    code = _PLATFORM_CODE[platform]
    return DISCORD_CEILING | (code << 60) | (_digest("tenant", platform, key) & _LOW60)


def is_platform_tenant(guild_id: int) -> bool:
    """True for an id made by `tenant_id` (never a Discord guild)."""
    return guild_id >= DISCORD_CEILING


def platform_of(guild_id: int) -> str | None:
    code = (guild_id >> 60) & 0b11 if guild_id >= DISCORD_CEILING else 0
    return {v: k for k, v in _PLATFORM_CODE.items()}.get(code)


def _small(*parts: str) -> int:
    return (_digest(*parts) & _LOW62) or 1


def user_id(platform: str, key: str) -> int:
    return _small("user", platform, key)


def room_id(platform: str, tenant_key: str, key: str) -> int:
    """A channel / group / conversation. Positive: negative ids are DM-note scopes."""
    return _small("room", platform, tenant_key, key)


def message_id(platform: str, tenant_key: str, room_key: str, key: str) -> int:
    return _small("message", platform, tenant_key, room_key, key)


def bot_user_id(platform: str, tenant_key: str) -> int:
    """The author id under which Salid's own replies are remembered."""
    return _small("bot", platform, tenant_key)
