"""License gate: is this Discord guild allowed to use Salid?

`is_licensed(guild_id)` looks at the license of the guild's organization (`salid.orgs`): status
`active` (or `past_due`) and not past `valid_until` plus a grace period. Answers are cached for
5 minutes. Without an org the guild is unlicensed, except for **grandfathering**: a guild that
already has a settings document from before licensing started (its `installed_at` is not later
than the moment of the first lookup, recorded once as the licensing epoch) gets an automatic org
with a trial license until `SALID_GRANDFATHER_UNTIL` (2026-12-31). The migration is idempotent
(the org id is derived from the guild id) and runs on first lookup. `SALID_LICENSE_ENFORCE=false`
turns the gate off. A failing lookup fails open (keeps the last known answer, else allows) so a
Firestore hiccup never silences paying customers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from .orgs import License, Member, Org, OrgStore, Workspace, stamp

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300.0
GRACE_DAYS = 3
STATE = "salid_license_state"  # guild 0: key "epoch"; guild = id: key "notice" (daily reply)
VALID_STATUSES = ("active", "past_due")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
        if len(value) <= 10:  # a plain date: valid through the end of that day
            dt += timedelta(days=1) - timedelta(seconds=1)
    return dt.astimezone(UTC)


def license_end(lic: License, grace_days: int = GRACE_DAYS) -> datetime | None:
    end = _parse(lic.valid_until)
    return end + timedelta(days=grace_days) if end else None


def license_ok(lic: License, now: datetime, grace_days: int = GRACE_DAYS) -> bool:
    if lic.status not in VALID_STATUSES:
        return False
    end = license_end(lic, grace_days)
    return end is None or now <= end


def days_left(lic: License, now: datetime) -> int | None:
    """Whole days until `valid_until` (negative once past); None when open-ended."""
    end = _parse(lic.valid_until)
    if end is None:
        return None
    return (end - now).days


class Licensing:
    def __init__(
        self,
        orgs: OrgStore,
        guilds=None,
        *,
        enforce: bool = True,
        grandfather_until: str = "2026-12-31",
        grace_days: int = GRACE_DAYS,
        ttl: float = CACHE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        hub_url: str = "",
    ):
        self.orgs, self.guilds, self.enforce = orgs, guilds, enforce
        self.grandfather_until, self.grace_days = grandfather_until, grace_days
        self._ttl, self._clock, self._now = ttl, clock, now
        self.hub_url = hub_url.rstrip("/")
        self._cache: dict[int, tuple[float, bool]] = {}
        self._lock = threading.RLock()

    # ---- the gate
    def is_licensed(self, guild_id: int) -> bool:
        if not self.enforce:
            return True
        t = self._clock()
        hit = self._cache.get(guild_id)
        if hit is not None and t - hit[0] < self._ttl:
            return hit[1]
        try:
            ok = self._check(guild_id)
        except Exception:
            logger.warning("license lookup failed for guild %s", guild_id, exc_info=True)
            ok = hit[1] if hit else True
            self._cache[guild_id] = (t - self._ttl + 30.0, ok)  # retry soon
            return ok
        self._cache[guild_id] = (t, ok)
        return ok

    def invalidate(self, guild_id: int | None = None) -> None:
        if guild_id is None:
            self._cache.clear()
        else:
            self._cache.pop(guild_id, None)

    def _check(self, guild_id: int) -> bool:
        org = self.orgs.org_of_guild(guild_id) or self.grandfather(guild_id)
        return org is not None and license_ok(org.license, self._now(), self.grace_days)

    @property
    def url(self) -> str:
        return f"{self.hub_url}/center" if self.hub_url else "/center"

    # ---- grandfathering
    def epoch(self) -> datetime:
        """When licensing first ran (stored once): guilds installed later are not grandfathered."""
        docs = self.orgs.docs
        with self._lock:
            d = docs.get(STATE, 0, "epoch")
            if d is None:
                d = {"at": stamp(self._now())}
                docs.put(STATE, 0, "epoch", d)
        return _parse(d["at"]) or self._now()

    def grandfather(self, guild_id: int) -> Org | None:
        """Auto-create the org of an existing guild (idempotent). None when not eligible."""
        if self.guilds is None:
            return None
        settings = self.guilds.find(guild_id)
        if settings is None:
            return None
        installed = _parse(settings.installed_at)
        if installed is not None and installed > self.epoch():
            return None  # installed after licensing started: must subscribe
        self.epoch()
        with self._lock:
            found = self.orgs.org_of_guild(guild_id)
            if found:
                return found
            org_id = f"org_g{guild_id}"
            if self.orgs.get(org_id) is not None:
                return None  # the owner detached it on purpose
            emails = [e.lower() for e in settings.admin_emails]
            members = [
                Member(email=e, role="owner" if i == 0 else "admin") for i, e in enumerate(emails)
            ]
            org = self.orgs.create(
                settings.guild_name or f"Server {guild_id}",
                emails[0] if emails else "",
                org_id=org_id,
                members=members,
                grandfathered=True,
                license=License(
                    plan="trial",
                    status="active",
                    seats=25,
                    valid_until=self.grandfather_until,
                    notes="grandfathered: existing server",
                ),
            )
            self.orgs.attach_workspace(
                org.id,
                Workspace(
                    id=f"discord-{guild_id}",
                    platform="discord",
                    external_id=str(guild_id),
                    name=settings.guild_name,
                    icon_url=settings.icon_url,
                    connected_at=stamp(self._now()),
                    connected_by="grandfather",
                ),
            )
            logger.info("grandfathered guild %s into %s", guild_id, org.id)
            return self.orgs.get(org.id)

    # ---- the one daily notice
    def claim_notice(self, guild_id: int, day: date | str) -> bool:
        """True once per guild per day: the caller may send the "not activated" line."""
        key = str(day)
        with self._lock:
            d = self.orgs.docs.get(STATE, guild_id, "notice")
            if d and d.get("day") == key:
                return False
            self.orgs.docs.put(STATE, guild_id, "notice", {"day": key})
            return True


def make_licensing(settings, store=None, guilds=None) -> Licensing:
    from .orgs import make_org_store

    return Licensing(
        make_org_store(settings, store),
        guilds,
        enforce=settings.license_enforce,
        grandfather_until=settings.grandfather_until,
        grace_days=settings.license_grace_days,
        hub_url=settings.hub_url,
    )
