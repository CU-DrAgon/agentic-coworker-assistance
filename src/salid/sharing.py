"""Data sharing between the workspaces of one organization (off unless a `ShareLink` exists).

`ShareLink(from_workspace, to_workspace, categories)` is directional: the destination may read the
source's shared categories. Safety rules (decided here, in code, never by the model):

* only items that EVERYONE in the source workspace could see are shareable: wiki pages and
  company places whose every source channel is in the source guild's `public_channel_ids`
  (private channels, DMs, voice rooms, personal places never cross);
* chat memory never crosses workspaces, there is no category for it;
* the ledger is shared to org owners / admins on the web only, never into another server's chat;
* sharing needs a positive answer: a source with unknown public channels shares nothing.

`shared_sources(guild_id, category)` answers "whose data may this guild's tools read besides its
own?" (memory ids of the source workspaces). `shared_wiki_pages` / `shared_places` are the helpers
the bot tools call (`wiki_search`, `find_place`); the documents API uses `shared_sources` too.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from .orgs import CATEGORIES, Org, OrgStore, Workspace

logger = logging.getLogger(__name__)

WEB_DOCS_CHANNEL = 1  # synthetic memory channel of uploaded document notes (web workspace)
SHARED_DOCS_CHANNEL = 2  # same notes, copied into a Discord workspace via ShareLink "documents"
SHARED_DOCS_NAME = "เอกสารองค์กร"
TTL_S = 60.0
MAX_SHARED_HITS = 5


@dataclass(frozen=True)
class SharedSource:
    memory_id: int
    workspace_id: str
    workspace_name: str
    platform: str


class Sharing:
    def __init__(
        self,
        orgs: OrgStore,
        public_channels: Callable[[int], set[int]],
        clock: Callable[[], float] = time.monotonic,
    ):
        """`public_channels(memory_id)` -> channel ids everyone in that workspace can see."""
        self.orgs, self._public, self._clock = orgs, public_channels, clock
        self._cache: dict[tuple[int, str], tuple[float, list[SharedSource]]] = {}

    def invalidate(self) -> None:
        self._cache.clear()

    # ---- who shares with this guild
    def sources(self, guild_id: int, category: str) -> list[SharedSource]:
        if category not in CATEGORIES:
            return []
        key = (guild_id, category)
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < TTL_S:
            return hit[1]
        out = self._sources(guild_id, category)
        self._cache[key] = (self._clock(), out)
        return out

    def _sources(self, guild_id: int, category: str) -> list[SharedSource]:
        found = self.orgs.org_of_workspace("discord", str(guild_id))
        if not found:
            return []
        org, ws = found
        return sources_in_org(org, ws.id, category)

    def with_shared_docs(self, guild_id: int, allowed: dict, vis=None):
        """Add the org's shared uploads (ShareLink "documents" from the web workspace) to what the
        memory tools of this guild may read: (allowed, visibility). Unchanged when not shared."""
        try:
            shared = any(s.platform == "web" for s in self.sources(guild_id, "documents"))
        except Exception:
            logger.warning("could not resolve shared documents", exc_info=True)
            shared = False
        if not shared:
            return allowed, vis
        allowed = {**allowed, SHARED_DOCS_CHANNEL: SHARED_DOCS_NAME}
        if vis is not None:
            scope = replace(vis.scope, channels=vis.scope.channels | {SHARED_DOCS_CHANNEL})
            vis = replace(vis, scope=scope)
        return allowed, vis

    # ---- helpers for the bot tools
    def public_wiki_pages(self, guild_id: int, wiki_repo) -> list[tuple[object, str]]:
        """Wiki pages other workspaces share with `guild_id`: [(page, workspace name)]."""
        out: list[tuple[object, str]] = []
        for src in self.sources(guild_id, "wiki"):
            public = self._public(src.memory_id)
            for page in wiki_repo.list(src.memory_id):
                if page_is_public(page.channel_ids, public):
                    out.append((page, src.workspace_name))
        return out

    def public_places(self, guild_id: int, place_repo) -> list[tuple[object, str]]:
        """Company places other workspaces share with `guild_id`: [(place, workspace name)]."""
        out: list[tuple[object, str]] = []
        for src in self.sources(guild_id, "places"):
            public = self._public(src.memory_id)
            for place in place_repo.list(src.memory_id):
                if place_is_public(place, public):
                    out.append((place, src.workspace_name))
        return out


def sources_in_org(org: Org, to_ws_id: str, category: str) -> list[SharedSource]:
    """Source workspaces (connected, same org) whose `category` is shared to `to_ws_id`."""
    out: dict[str, SharedSource] = {}
    for link in org.sharing:
        if link.to_workspace != to_ws_id or category not in link.categories:
            continue
        src: Workspace | None = org.workspace(link.from_workspace)
        if src is None or src.status != "connected" or src.id == to_ws_id:
            continue
        out[src.id] = SharedSource(src.memory_id, src.id, src.name or src.id, src.platform)
    return list(out.values())


def page_is_public(channel_ids: list[int], public: set[int]) -> bool:
    """A page is shareable only when it names source channels and ALL of them are public."""
    return bool(channel_ids) and all(c > 0 and c in public for c in channel_ids)


def place_is_public(place, public: set[int]) -> bool:
    """A company place stated only in public text channels (never personal, never a voice room)."""
    if place.personal or not place.sources:
        return False
    for src in place.sources:
        if src.get("members"):
            return False
        cid = int(src.get("channel_id", 0))
        if cid <= 0 or cid not in public:
            return False
    return True


# ------------------------------------------------------------------ process-wide default
_default: Sharing | None = None


def set_default(sharing: Sharing | None) -> None:
    global _default
    _default = sharing


def shared_sources(guild_id: int, category: str) -> list[int]:
    """Memory ids of the workspaces that share `category` with this guild (empty by default)."""
    if _default is None:
        return []
    try:
        return [s.memory_id for s in _default.sources(guild_id, category)]
    except Exception:
        logger.warning("could not resolve shared sources", exc_info=True)
        return []


def make_sharing(orgs: OrgStore, guilds) -> Sharing:
    """`guilds`: a `GuildStore`; the public channels of Discord workspaces come from its docs."""

    def public(memory_id: int) -> set[int]:
        try:
            g = guilds.find(memory_id) if memory_id < 2**62 else None
        except Exception:
            return set()
        return {int(x) for x in (g.public_channel_ids if g else [])}

    return Sharing(orgs, public)
