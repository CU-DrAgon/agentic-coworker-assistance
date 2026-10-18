"""Google Calendar + Meet tools in chat on Slack, LINE, Telegram and WhatsApp.

The tools themselves live in `salid.google_tools`; this module only decides WHEN a message may
have them and binds them to the right person and conversation:

* the sender must be a VERIFIED member of the org that owns the tenant (`Assistant.identity_of`)
  and that member must have connected Google Calendar on the web (`GoogleOAuth.is_connected`);
  an unverified sender, a guest, or a member who has not connected gets NO calendar tools at all
  (the model then simply cannot schedule; there is no "how to connect" tool);
* the tools act for the asker only (org + email come from the verified link, never from the
  model), so they are safe in a group; the confirm rule is bound to the conversation
  (platform, tenant, room, thread) and to the message id, so a draft is only acted on by a LATER
  message of the same conversation;
* tokens are read from the DocStore of the org store: the platforms process must run with the
  same store as the web (`SALID_STORE=firestore`) to find tokens connected on the web.

Nothing personal (email, title, attendees) is logged here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from ..google_oauth import GoogleConfig, GoogleOAuth
from ..google_tools import CalendarToolContext, calendar_tools
from ..scheduling import GoogleCalendar
from ..secretary import SecretaryState
from ..secretary_store import new_id

if TYPE_CHECKING:  # pragma: no cover
    from .core import Assistant, Incoming

logger = logging.getLogger(__name__)

CONNECTED_TTL_S = 60.0
RUN_TIMEOUT_S = 30.0
EXEC_ROLES = ("owner", "admin")


class CalendarGlue:
    """One per Assistant: the OAuth store, the Calendar backend and the confirm state."""

    def __init__(
        self,
        oauth: Any,
        calendar: Any,
        zone: Any,
        mono=time.monotonic,
        state: SecretaryState | None = None,
    ) -> None:
        self.oauth, self.calendar, self.zone, self._mono = oauth, calendar, zone, mono
        self.state = state or SecretaryState()
        self._connected: dict[tuple[str, str], tuple[float, bool]] = {}

    @classmethod
    def from_settings(cls, ps: Any, docs: Any, tz: str) -> CalendarGlue | None:
        """None (feature off) unless client id, secret, redirect URI and a valid key are set."""
        cfg = GoogleConfig.from_settings(ps)
        if docs is None or not cfg.enabled():
            return None
        from zoneinfo import ZoneInfo

        oauth = GoogleOAuth(docs, cfg)
        return cls(oauth, GoogleCalendar(oauth, tz), ZoneInfo(tz))

    def is_connected(self, org_id: str, email: str) -> bool:
        """Blocking; cached for `CONNECTED_TTL_S` (the tools re-check on every call anyway)."""
        key = (org_id, email.lower())
        hit = self._connected.get(key)
        now = self._mono()
        if hit and hit[0] > now:
            return hit[1]
        ok = bool(self.oauth.is_connected(org_id, email))
        if len(self._connected) > 2000:
            self._connected = {k: v for k, v in self._connected.items() if v[0] > now}
        self._connected[key] = (now + CONNECTED_TTL_S, ok)
        return ok


def _lookup(a: Assistant, glue: CalendarGlue, msg: Incoming) -> tuple[Any, str, bool] | None:
    """(org, member email, org owner/admin) for a verified AND connected sender. Blocking."""
    who = a.identity_of(msg)
    if who is None or a.identity is None:
        return None
    org_id, email = who
    org = a.identity.orgs.get(org_id)
    if org is None:
        return None
    role = org.role_of(email)
    if role is None or not glue.is_connected(org_id, email):
        return None
    return org, email, role in EXEC_ROLES


async def calendar_tools_for(
    a: Assistant, msg: Incoming, gid: int, rid: int, lang: str, is_exec: bool
) -> list:
    """The calendar tools for this message, or [] (feature off / unverified / not connected)."""
    glue: CalendarGlue | None = getattr(a, "calendar", None)
    if glue is None:
        return []
    try:
        found = await asyncio.to_thread(_lookup, a, glue, msg)
    except Exception as e:
        logger.warning("calendar lookup failed (%s)", type(e).__name__)
        return []
    if found is None:
        return []
    org, email, org_admin = found
    loop = asyncio.get_running_loop()

    def run(coro):  # called from the ask worker thread
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=RUN_TIMEOUT_S)

    request_id = "|".join(msg.key) if msg.message else new_id()
    return calendar_tools(
        CalendarToolContext(
            org=org,
            email=email,
            is_exec=is_exec or org_admin,
            calendar=glue.calendar,
            oauth=glue.oauth,
            state=glue.state,
            run=run,
            zone=glue.zone,
            scope=(msg.platform, str(gid), str(rid), msg.thread or ""),
            lang=lang,
            request_id=request_id,
            now=a._now,
        )
    )
