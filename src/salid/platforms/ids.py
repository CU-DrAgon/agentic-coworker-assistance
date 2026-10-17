"""Integer ids for Slack / LINE / Telegram / WhatsApp objects: the one workspace id scheme.

Salid's stores key everything by integers (guild, channel, user, message). Non-Discord ids are
strings, so each one is hashed to an int. There is exactly one implementation, in `salid.orgs`,
shared by this bot package and the command-center connectors (`salid.connectors.slack`), so one
workspace is one tenant whichever integration saw it: same ledger, memory, settings and licence.

* **Tenant** (`tenant_id`, used as `guild_id` everywhere) = `orgs.workspace_memory_id(platform, key)`.
  Channel / user / message ids are `orgs.namespaced_id` and `orgs.message_namespaced_id`. All of
  them are `2**62 + (sha256(platform ...) mod 2**61)`: positive, inside a signed int64 (Firestore,
  SQLite) and, because Discord snowflakes stay below `2**62` until about 2049, never a Discord id.
  The platform is part of every hash, so the same key on two platforms is two different ids.
  The platform cannot be decoded from an id any more (there is no code field).
* A message id hashes workspace + room + message key: the memory store keys messages by
  `(guild_id, message_id)` with no channel, and a Slack `ts` is only unique inside one channel.
* Collisions are still refused, not trusted away: the tenant's settings document records its owner
  (`guild_name`, see core.tenant_label) and a message from a different owner whose key hashes to an
  id already taken is dropped (core.Assistant._owner_ok, fail closed).
* Negative ids are never produced here: they are reserved for the private "DM notes" scope
  (`memory_access.dm_scope`).
* Before 2049 this layout must move (Discord snowflakes reach 2**62 then).
"""

from __future__ import annotations

from salid import orgs

DISCORD_CEILING = orgs.NAMESPACE_BASE  # Discord snowflakes are below this until ~2049
SLACK, LINE, TELEGRAM, WHATSAPP = "slack", "line", "telegram", "whatsapp"


def tenant_id(platform: str, key: str) -> int:
    """The `guild_id` of a Slack workspace (`team_id`) or a LINE channel (the bot's user id)."""
    return orgs.workspace_memory_id(platform, key)


def is_platform_tenant(guild_id: int) -> bool:
    """True for an id made by `tenant_id` (never a Discord guild)."""
    return guild_id >= DISCORD_CEILING


def user_id(platform: str, key: str) -> int:
    return orgs.namespaced_id(platform, key, "user")


def room_id(platform: str, tenant_key: str, key: str) -> int:
    """A channel / group / conversation. Positive: negative ids are DM-note scopes."""
    return orgs.namespaced_id(platform, key, "channel")


def message_id(platform: str, tenant_key: str, room_key: str, key: str) -> int:
    return orgs.message_namespaced_id(platform, tenant_key, room_key, key)


def bot_user_id(platform: str, tenant_key: str) -> int:
    """The author id under which Salid's own replies are remembered."""
    return orgs.namespaced_id(platform, tenant_key, "bot")
