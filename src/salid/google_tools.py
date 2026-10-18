"""Agent tools for Google Calendar + Meet (not wired into any bot yet).

`calendar_tools(ctx)` returns two plain functions for the agent, bound to the asker's org and email
(neither is a tool argument): `schedule_google_meeting` and `cancel_google_meeting`. Both use the
secretary's confirm rule: the first call only returns a draft / proposal, and only the same call
repeated in a LATER request (different `ctx.request_id`) acts.

Attendees: names are resolved to MEMBERS of the asker's organization only (full email, or the
unique start of an email's local part). An email that belongs to nobody in the org is refused,
unless the asker is an executive (owner / admin / super admin), who may invite outside guests by
full email. The event lives in the asker's own primary calendar; Salid only cancels events it
created itself (marked in the event).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .google_oauth import Account
from .scheduling import (
    MAX_ATTENDEES,
    CalendarError,
    EventSpec,
    GoogleCalendar,
    fmt_when,
    parse_local,
)
from .secretary import SecretaryState, fingerprint
from .secretary_store import new_id

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r"[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+")
MAX_MINUTES = 12 * 60


@dataclass
class CalendarToolContext:
    org: Any  # salid.orgs.Org
    email: str
    is_exec: bool
    calendar: GoogleCalendar
    oauth: Any  # google_oauth.GoogleOAuth (is_connected)
    state: SecretaryState
    run: Callable[[Any], Any]  # run a coroutine and return its result
    zone: Any
    lang: str = "en"
    request_id: str = field(default_factory=new_id)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def resolve_attendees(
    org: Any, names: list[str], allow_external: bool
) -> tuple[list[str], list[str]]:
    """-> (emails, unresolved). Members only; with `allow_external` a full email may be outside."""
    members = [m.email.lower() for m in org.members]
    found: dict[str, None] = {}
    bad: list[str] = []
    for raw in names or []:
        s = (raw or "").strip().strip("<>").lower()
        if not s:
            continue
        hit = None
        if s in members:
            hit = s
        elif _EMAIL.fullmatch(s):
            hit = s if allow_external else None
        else:
            cands = [e for e in members if e.split("@")[0] == s] or [
                e for e in members if e.split("@")[0].startswith(s)
            ]
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            bad.append(raw.strip())
        else:
            found[hit] = None
    return list(found), bad


def calendar_tools(ctx: CalendarToolContext) -> list:
    """The agent-facing functions (docstrings are the tool descriptions)."""
    account = Account(ctx.org.id, ctx.email.strip().lower())

    def guard(fn):
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        except CalendarError as e:
            if e.code == "not_connected":
                return _not_connected()
            return {"error": e.code, "detail": e.detail}
        except Exception:
            logger.exception("google calendar tool failed")
            return {"error": "internal error"}

    def _confirmed(key: tuple, confirm: bool) -> bool:
        """True only for confirm=true in a LATER request than the one that proposed `key`. A draft
        call (confirm=false) always (re)records the proposal for the current request."""
        if confirm:
            return ctx.state.confirm(key, ctx.request_id)
        if ctx.state.confirm(key, ctx.request_id):  # consumed an older proposal: record anew
            ctx.state.confirm(key, ctx.request_id)
        return False

    def _not_connected() -> dict:
        return {
            "status": "not_connected",
            "instruction": (
                "The user has not connected Google Calendar (or the connection expired). Tell them "
                "to open the Salid command center and press 'Connect Google Calendar', then ask "
                "again. Do NOT claim anything was scheduled."
            ),
        }

    def schedule_google_meeting(
        title: str,
        start: str,
        end: str,
        attendees: list[str] | None = None,
        description: str = "",
        confirm: bool = False,
    ) -> dict:
        """Create a Google Calendar event with a Google Meet link in the asker's own calendar and
        invite attendees by email. The first call only returns a draft (status needs_confirmation):
        show it and wait. Call again with EXACTLY the same arguments and confirm=true only after
        the user confirmed in a later message.

        Args:
            title: short meeting title.
            start: ISO 8601 start; without an offset it is read in the company time zone.
            end: ISO 8601 end, after start, at most 12 hours later.
            attendees: members of the organization (email or unique start of the email name).
                Outside emails are accepted from executives only.
            description: optional note shown in the invite.
            confirm: false for the draft, true to create (only after the user confirmed).
        """

        def go() -> dict:
            if not title.strip():
                raise ValueError("title is required")
            s, e = parse_local(start, ctx.zone), parse_local(end, ctx.zone)
            if e <= s or e - s > timedelta(minutes=MAX_MINUTES):
                raise ValueError("end must be after start, at most 12 hours later")
            if s < ctx.now() - timedelta(minutes=5):
                raise ValueError("the start is in the past")
            emails, bad = resolve_attendees(ctx.org, attendees or [], ctx.is_exec)
            if bad:
                return {
                    "error": "unknown_attendees",
                    "unresolved": bad,
                    "instruction": (
                        "Attendees must be members of the organization (use their email). Ask the "
                        "user to correct these; outside guests need an executive."
                    ),
                }
            if len(emails) > MAX_ATTENDEES:
                raise ValueError(f"at most {MAX_ATTENDEES} attendees")
            if not ctx.oauth.is_connected(account.org_id, account.email):
                return _not_connected()
            key = (
                "gmeet",
                account.org_id,
                account.email,
                fingerprint(
                    title.strip(), s.astimezone(UTC).isoformat(), e.isoformat(), emails, description
                ),
            )
            proposal = {
                "title": title.strip(),
                "when": f"{fmt_when(s, ctx.lang, ctx.zone)} → {e.astimezone(ctx.zone):%H:%M}",
                "attendees": emails,
                "where": "Google Meet (link is created with the event)",
                "note": description,
            }
            if not _confirmed(key, confirm):
                return {
                    "status": "needs_confirmation",
                    "proposal": proposal,
                    "instruction": (
                        "Show this draft to the user and ask them to confirm. Do NOT claim it is "
                        "scheduled. After they confirm in a later message, call "
                        "schedule_google_meeting again with exactly the same arguments and "
                        "confirm=true."
                    ),
                }
            ev = ctx.run(
                ctx.calendar.create_event_full(
                    account, EventSpec(title.strip(), s, e, description), emails
                )
            )
            return {
                "status": "scheduled",
                "event_id": ev.id,
                "calendar_url": ev.url,
                "meet_url": ev.meet_url,
                "attendees": emails,
            }

        return guard(go)

    def cancel_google_meeting(event_id: str, confirm: bool = False) -> dict:
        """Cancel a Google Calendar event that Salid created in the asker's calendar; attendees
        are notified. The first call returns what would be cancelled (status needs_confirmation);
        call again with the same event_id and confirm=true only after the user confirmed in a
        later message.

        Args:
            event_id: the event id returned when it was scheduled.
            confirm: false to preview, true to cancel (only after the user confirmed).
        """

        def go() -> dict:
            eid = (event_id or "").strip()
            if not eid:
                raise ValueError("event_id is required")
            if not ctx.oauth.is_connected(account.org_id, account.email):
                return _not_connected()
            ev = ctx.run(ctx.calendar.get_event(account, eid))
            if (ev.get("extendedProperties") or {}).get("private", {}).get("salid") != "1":
                return {"error": "not_created_by_salid", "event_id": eid}
            key = ("gmeet-cancel", account.org_id, account.email, eid)
            if not _confirmed(key, confirm):
                return {
                    "status": "needs_confirmation",
                    "proposal": {
                        "event_id": eid,
                        "title": ev.get("summary", ""),
                        "start": (ev.get("start") or {}).get("dateTime", ""),
                    },
                    "instruction": (
                        "Ask the user to confirm cancelling this meeting (attendees will be "
                        "notified), then call cancel_google_meeting again with confirm=true."
                    ),
                }
            ctx.run(ctx.calendar.delete_event(account, eid))
            return {"status": "cancelled", "event_id": eid}

        return guard(go)

    return [schedule_google_meeting, cancel_google_meeting]
