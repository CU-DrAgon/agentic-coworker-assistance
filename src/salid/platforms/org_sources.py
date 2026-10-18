"""Where the org-wide scope gets its facts (the I/O side of `salid.org_scope`).

One `RoomProvider` per platform answers "which channels of THAT workspace can this account read?":

* Slack: `SlackPort.readable_rooms` for the workspace's port in this process (a team this process
  has no port for is not reachable: excluded).
* Discord: the guild's roles, the member and the channel overwrites over the Discord REST API with
  the optional bot token (`SALID_DISCORD_TOKEN`), resolved by `org_scope.readable_channels`.
  No token or any REST error: Discord is excluded (fail closed).
* LINE / Telegram / WhatsApp: only the member's own 1:1 chat on that platform. Groups are never
  included: their members cannot be listed.

Answers are cached per (workspace, account) for five minutes (a permission change shows up
within that time; a failure is cached only for a few seconds so a blip does not blind us).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..org_scope import OrgView, RoomProvider, readable_channels, resolve
from ..orgs import OrgStore, Workspace

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300.0
FAIL_TTL_S = 15.0
DISCORD_API = "https://discord.com/api/v10"
DISCORD_TIMEOUT_S = 8.0


class CachedProvider:
    """Memoise a provider per (workspace id, account) for `CACHE_TTL_S`."""

    def __init__(self, inner: RoomProvider, clock: Callable[[], float]):
        self.inner, self._clock = inner, clock
        self._d: dict[tuple[str, str], tuple[float, float, dict[str, str] | None]] = {}

    async def __call__(self, ws: Workspace, account: str) -> dict[str, str] | None:
        key = (ws.id, account)
        hit = self._d.get(key)
        now = self._clock()
        if hit is not None and now - hit[0] < hit[1]:
            return hit[2]
        try:
            got = await self.inner(ws, account)
        except Exception as e:
            logger.warning("org scope provider failed (%s)", type(e).__name__)
            got = None
        self._d[key] = (now, CACHE_TTL_S if got is not None else FAIL_TTL_S, got)
        if len(self._d) > 5000:
            self._d = {k: v for k, v in self._d.items() if now - v[0] < v[1]}
        return got


async def own_chat(ws: Workspace, account: str) -> dict[str, str] | None:
    """LINE / Telegram / WhatsApp: the member's own 1:1 chat (its room key is the account id)."""
    return {account: ""}


class SlackRooms:
    """Slack: ports by team id; `add` registers the port this process runs for a workspace."""

    def __init__(self) -> None:
        self.ports: dict[str, object] = {}

    def add(self, port) -> None:
        self.ports[port.team_id] = port

    async def __call__(self, ws: Workspace, account: str) -> dict[str, str] | None:
        port = self.ports.get(ws.external_id)
        if port is None:
            return None  # no bot token for that team in this process
        return await port.readable_rooms(account)


class DiscordRest:
    """The three read-only GETs the resolver needs. The token only goes into the header."""

    def __init__(self, token: str, client=None):
        self._token, self._client = token, client

    async def get(self, path: str):
        import httpx

        client = self._client or httpx.AsyncClient(timeout=DISCORD_TIMEOUT_S)
        try:
            r = await client.get(
                f"{DISCORD_API}{path}", headers={"Authorization": f"Bot {self._token}"}
            )
            r.raise_for_status()
            return r.json()
        finally:
            if self._client is None:
                await client.aclose()


class DiscordRooms:
    def __init__(self, rest: DiscordRest):
        self.rest = rest

    async def __call__(self, ws: Workspace, account: str) -> dict[str, str] | None:
        guild = ws.external_id
        got, member, channels = await asyncio.gather(
            self.rest.get(f"/guilds/{guild}"),
            self.rest.get(f"/guilds/{guild}/members/{account}"),
            self.rest.get(f"/guilds/{guild}/channels"),
        )
        if (
            not isinstance(got, dict)
            or not isinstance(member, dict)
            or not isinstance(channels, list)
        ):
            return None
        member.setdefault("user", {"id": account})
        if str((member.get("user") or {}).get("id")) != str(account):
            return None
        return {str(cid): name for cid, name in readable_channels(got, member, channels).items()}


class OrgSources:
    """Resolve the org-wide view of a verified member (see `org_scope.resolve`)."""

    def __init__(
        self,
        orgs: OrgStore,
        *,
        discord_token: str = "",
        clock: Callable[[], float] | None = None,
        rest: DiscordRest | None = None,
        providers: dict[str, RoomProvider] | None = None,
    ):
        import time

        clock = clock or time.monotonic
        self.orgs = orgs
        self.slack = SlackRooms()
        self.providers: dict[str, RoomProvider] = {
            "slack": CachedProvider(self.slack, clock),
            "line": own_chat,
            "telegram": own_chat,
            "whatsapp": own_chat,
        }
        if discord_token or rest is not None:
            self.providers["discord"] = CachedProvider(
                DiscordRooms(rest or DiscordRest(discord_token)), clock
            )
        self.providers.update(providers or {})

    def add_slack_port(self, port) -> None:
        self.slack.add(port)

    def _org(self, org_id: str):
        return self.orgs.get(org_id)

    async def view(
        self,
        org_id: str,
        email: str,
        *,
        asker: str,
        is_direct: bool,
        audience: frozenset[str] | None,
        asking: tuple[str, str] | None = None,
    ) -> OrgView | None:
        org = await asyncio.to_thread(self._org, org_id)
        if org is None:
            return None
        return await resolve(
            org,
            email,
            is_direct=is_direct,
            audience=audience,
            asker=asker,
            providers=self.providers,
            asking=asking,
        )


__all__ = [
    "CachedProvider",
    "DiscordRest",
    "DiscordRooms",
    "OrgSources",
    "SlackRooms",
    "own_chat",
]
