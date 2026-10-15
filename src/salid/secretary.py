"""The secretary tools of the agent: scheduling, PDF documents and email.

Same pattern as the ledger tools: every tool is bound in code to one guild and one author (they are
not part of any tool schema), so the model cannot widen the scope. Outward-facing actions
(creating / cancelling / moving a meeting, sending an email) are gated in code by
`SecretaryState.confirm`: the first call with given arguments only records a proposal and returns a
preview; the action runs only when the *same* arguments arrive again in a LATER request, i.e. after
the user answered the preview. Within one request it can never run, whatever the model does.

Discord objects live on the bot's event loop while tools run in a worker thread, so everything that
touches Discord goes through `ctx.run(coroutine)` (run_coroutine_threadsafe in the bot).
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .email_out import (
    MAX_ATTACHMENT_BYTES,
    MAX_RECIPIENTS,
    EmailSender,
    OutgoingEmail,
    clean_addresses,
)
from .i18n import t
from .pdf_report import custom_pdf, expense_pdf, find_fonts, minutes_pdf, totals_by_currency
from .permissions import guild_missing, permission_message
from .scheduling import (
    CalendarError,
    MeetingService,
    classify_link,
    fmt_when,
    parse_local,
    parse_range,
    pick_voice_channel,
    resolve_attendees,
)
from .secretary_store import SecretaryStore, new_id, utc_iso

logger = logging.getLogger(__name__)

CONFIRM_TTL_S = 30 * 60
PDF_TTL_S = 60 * 60
PDF_KINDS = ("expense_summary", "receipt_list", "meeting_minutes", "custom")
EMAIL_STATUSES = ("sent", "logged")
MAX_BODY_CHARS = 20000
MAX_SUBJECT_CHARS = 200


class SecretaryState:
    """Per-process memory shared by all requests: proposals awaiting confirmation and the PDFs
    built recently (so an email draft can attach one in a later message)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._pending: dict[tuple, tuple[str, float]] = {}
        self._pdfs: dict[tuple[int, int, str], tuple[bytes, float]] = {}

    def confirm(self, key: tuple, request_id: str) -> bool:
        """True when `key` was proposed in an earlier, still-fresh request (consumes it)."""
        now = self.clock()
        self._pending = {k: v for k, v in self._pending.items() if v[1] > now}
        entry = self._pending.get(key)
        if entry is not None and entry[0] != request_id:
            del self._pending[key]
            return True
        if entry is None:
            self._pending[key] = (request_id, now + CONFIRM_TTL_S)
        return False

    def remember_pdf(self, guild_id: int, user_id: int, name: str, data: bytes) -> None:
        now = self.clock()
        self._pdfs = {k: v for k, v in self._pdfs.items() if v[1] > now}
        self._pdfs[(guild_id, user_id, name)] = (data, now + PDF_TTL_S)

    def recall_pdf(self, guild_id: int, user_id: int, name: str) -> bytes | None:
        hit = self._pdfs.get((guild_id, user_id, name))
        return hit[0] if hit and hit[1] > self.clock() else None


def fingerprint(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class SecretaryContext:
    """Everything the secretary tools are bound to for one request (set by the bot)."""

    settings: Any
    guild_id: int
    guild_name: str
    author_id: int
    author_name: str
    is_exec: bool
    can_email: bool
    lang: str
    text_channel_id: int | None
    store: SecretaryStore
    state: SecretaryState
    run: Callable[[Any], Any]  # run a coroutine on the bot's loop and return its result
    app_id: Any = None  # the bot's application id (builds the re-authorize link)
    ledger: Any = None  # qa.LedgerTools (scoping of receipts); set by qa.ask_async
    service: MeetingService | None = None
    sender: EmailSender | None = None
    documents: bool = True
    scheduling: bool = True
    email: bool = True
    request_id: str = field(default_factory=new_id)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    attachments: list[tuple[str, bytes]] = field(default_factory=list)  # (file name, data)


class SecretaryTools:
    def __init__(self, ctx: SecretaryContext) -> None:
        self.c = ctx

    # ------------------------------------------------------------ helpers
    def _guild(self):
        return self.c.service.get_guild(self.c.guild_id)

    def _lang(self) -> str:
        return self.c.lang if self.c.lang in ("th", "en") else "th"

    def _perm_help(self) -> dict:
        try:
            missing = guild_missing(self._guild(), ("manage_events",))
        except Exception:
            missing = []
        text = permission_message(missing or ["manage_events"], self._lang(), self.c.app_id)
        return {"error": "missing_permission", "how_to_grant": text}

    # ------------------------------------------------------------ scheduling
    async def _prepare(self, attendees: list[str], voice_channel: str | None, link: str):
        guild = self._guild()
        members, unresolved = resolve_attendees(guild, attendees)
        author = guild.get_member(self.c.author_id)
        if author is not None and all(m.id != author.id for m in members):
            members.insert(0, author)
        vc = None
        if voice_channel:
            vc = pick_voice_channel(guild, voice_channel)
        elif not link:
            vc = pick_voice_channel(guild, None)
        return guild, members, unresolved, vc, author

    def schedule_meeting(
        self,
        title: str,
        start: str,
        duration_minutes: int = 60,
        attendees: list[str] | None = None,
        voice_channel: str | None = None,
        description: str = "",
        link: str | None = None,
    ) -> dict:
        c = self.c
        if not c.scheduling or c.service is None:
            return {"status": "disabled"}
        title = (title or "").strip()[:100]
        if not title:
            return {"error": "title is required"}
        try:
            start_dt = parse_local(start, c.settings.zone)
        except ValueError:
            return {"error": "start must be an ISO 8601 date-time, e.g. 2026-10-19T14:00"}
        if start_dt <= c.now():
            return {"error": "start is in the past; ask the user for a future time"}
        try:
            minutes = int(duration_minutes or 60)
        except (TypeError, ValueError):
            minutes = 60
        minutes = max(5, min(minutes, 12 * 60))
        link = (link or "").strip()
        if link and classify_link(link) is None:
            return {"error": "link is not a valid http(s) URL"}
        guild, members, unresolved, vc, _author = c.run(
            self._prepare(attendees or [], voice_channel, link)
        )
        if unresolved:
            return {
                "status": "needs_info",
                "unresolved_attendees": unresolved,
                "instruction": "Ask the user who these people are (exact Discord names or mentions).",
            }
        if voice_channel and vc is None:
            return {
                "status": "needs_info",
                "instruction": f"No single voice channel matches {voice_channel!r}; ask which one.",
            }
        if len(members) > 25:
            return {"error": "too many attendees (max 25)"}
        lang = self._lang()
        zone = c.settings.zone
        key = (
            "schedule",
            c.guild_id,
            c.author_id,
            fingerprint(
                title,
                start_dt.astimezone(UTC).isoformat(),
                minutes,
                sorted(m.id for m in members),
                getattr(vc, "id", None),
                link,
            ),
        )
        proposal = {
            "title": title,
            "when": f"{fmt_when(start_dt, lang, zone)} ({c.settings.tz}), {minutes} min",
            "attendees": [{"name": m.display_name, "mention": f"<@{m.id}>"} for m in members],
            "where": (
                f"voice channel #{vc.name}"
                if vc is not None
                else (link or "Discord (external event)")
            ),
            "note": description,
        }
        if not c.state.confirm(key, c.request_id):
            return {
                "status": "needs_confirmation",
                "proposal": proposal,
                "instruction": (
                    "Show this proposal to the user (mention the attendees) and ask them to "
                    "confirm. Do NOT claim it is scheduled. After they confirm in a later "
                    "message, call schedule_meeting again with exactly the same arguments."
                ),
            }
        try:
            m = c.run(self._create(guild, members, vc, title, start_dt, minutes, link, description))
        except CalendarError as e:
            if e.code == "missing_permission":
                return self._perm_help()
            return {"error": "could not create the event", "detail": e.detail}
        return {
            "status": "created",
            "meeting": c.service.display(m, lang),
            "dm_failed": m.get("dm_failed", []),
        }

    async def _create(self, guild, members, vc, title, start_dt, minutes, link, description):
        c = self.c
        author = guild.get_member(c.author_id)
        organizer = author or members[0]
        return await c.service.create(
            guild,
            title=title,
            start=start_dt,
            duration_minutes=minutes,
            attendees=members,
            voice_channel=vc,
            link=link,
            description=description,
            organizer=organizer,
            text_channel_id=c.text_channel_id,
        )

    def list_meetings(self, range: str = "upcoming") -> dict:  # noqa: A002 (tool parameter name)
        c = self.c
        if not c.scheduling or c.service is None:
            return {"status": "disabled"}
        now = c.now().astimezone(c.settings.zone)
        try:
            a, b = parse_range(range, now)
        except ValueError:
            return {
                "error": "range must be today, tomorrow, week, month, upcoming or YYYY-MM-DD[..YYYY-MM-DD]"
            }
        rows = c.service.upcoming(c.guild_id, a, b)
        lang = self._lang()
        return {"count": len(rows), "meetings": [c.service.display(m, lang) for m in rows[:30]]}

    def _may_change(self, m: dict) -> bool:
        return self.c.is_exec or self.c.author_id in (m.get("organizer_id"),)

    def cancel_meeting(self, event_id: str) -> dict:
        c = self.c
        if not c.scheduling or c.service is None:
            return {"status": "disabled"}
        m = c.service.find(c.guild_id, event_id)
        if m is None or m["status"] not in ("scheduled", "started"):
            return {"error": "meeting not found"}
        if not self._may_change(m):
            return {"error": "not_allowed", "note": "only the organiser or an executive"}
        lang = self._lang()
        key = ("cancel", c.guild_id, c.author_id, m["id"])
        if not c.state.confirm(key, c.request_id):
            return {
                "status": "needs_confirmation",
                "proposal": c.service.display(m, lang),
                "instruction": "Ask the user to confirm cancelling this meeting, then call "
                "cancel_meeting again in a later message.",
            }
        try:
            c.run(c.service.cancel(self._guild(), m))
        except CalendarError as e:
            return self._perm_help() if e.code == "missing_permission" else {"error": e.code}
        return {"status": "cancelled", "meeting": c.service.display(m, lang)}

    def reschedule_meeting(self, event_id: str, new_start: str) -> dict:
        c = self.c
        if not c.scheduling or c.service is None:
            return {"status": "disabled"}
        m = c.service.find(c.guild_id, event_id)
        if m is None or m["status"] not in ("scheduled", "started"):
            return {"error": "meeting not found"}
        if not self._may_change(m):
            return {"error": "not_allowed", "note": "only the organiser or an executive"}
        try:
            start_dt = parse_local(new_start, c.settings.zone)
        except ValueError:
            return {"error": "new_start must be an ISO 8601 date-time"}
        if start_dt <= c.now():
            return {"error": "new_start is in the past"}
        lang = self._lang()
        key = (
            "reschedule",
            c.guild_id,
            c.author_id,
            fingerprint(m["id"], start_dt.astimezone(UTC).isoformat()),
        )
        if not c.state.confirm(key, c.request_id):
            return {
                "status": "needs_confirmation",
                "proposal": {
                    **c.service.display(m, lang),
                    "new_when": f"{fmt_when(start_dt, lang, c.settings.zone)} ({c.settings.tz})",
                },
                "instruction": "Ask the user to confirm the new time, then call "
                "reschedule_meeting again with the same arguments in a later message.",
            }
        try:
            m = c.run(c.service.reschedule(self._guild(), m, start_dt))
        except CalendarError as e:
            return self._perm_help() if e.code == "missing_permission" else {"error": e.code}
        return {"status": "rescheduled", "meeting": c.service.display(m, lang)}

    # ------------------------------------------------------------ documents
    def make_report_pdf(
        self,
        kind: str,
        start_date: str = "",
        end_date: str = "",
        title: str = "",
        notes: str = "",
        meeting_id: str = "",
    ) -> dict:
        c = self.c
        if not c.documents:
            return {"status": "disabled"}
        if kind not in PDF_KINDS:
            return {"error": f"kind must be one of {', '.join(PDF_KINDS)}"}
        lang = self._lang()
        fonts = find_fonts(c.settings.pdf_font)
        now = c.now().astimezone(c.settings.zone)
        title, notes = (title or "").strip()[:150], (notes or "").strip()[:3000]
        out: dict = {"kind": kind}
        if kind in ("expense_summary", "receipt_list"):
            try:
                a, b = (datetime.fromisoformat(x).date() for x in (start_date, end_date))
            except ValueError:
                return {"error": "start_date and end_date are required (YYYY-MM-DD)"}
            if b < a or (b - a).days > 731:
                return {"error": "invalid date range (at most 2 years)"}
            rows = c.ledger.store.all_receipts(
                c.guild_id,
                a.isoformat(),
                b.isoformat(),
                submitter_id=c.ledger.forced,
                include_void=True,
            )
            active = [r for r in rows if (r.status or "active") != "void"]
            data = expense_pdf(
                org=c.guild_name,
                title=title,
                start=a.isoformat(),
                end=b.isoformat(),
                receipts=active,
                void_count=len(rows) - len(active),
                scope_note="" if c.is_exec else t("pdf_scope_own", lang, name=c.author_name),
                notes=notes,
                kind=kind,
                lang=lang,
                generated_at=now,
                fonts=fonts,
            )
            by = totals_by_currency(active)
            out.update(receipts=len(active), totals_by_currency=by)
            base = f"{kind}_{a.isoformat()}_{b.isoformat()}"
        elif kind == "meeting_minutes":
            m = self._minutes_source(meeting_id, start_date, end_date)
            if isinstance(m, dict) and "error" in m and "id" not in m:
                return m
            when = fmt_when(datetime.fromisoformat(m["start_utc"]), lang, c.settings.zone)
            data = minutes_pdf(
                org=c.guild_name,
                title=title or m["title"],
                when=when,
                attendees=m.get("attendee_names", []),
                summary=m.get("summary", ""),
                attendance={
                    v.get("name", k): v.get("note", "")
                    for k, v in (m.get("attendance_notes") or {}).items()
                },
                notes=notes,
                lang=lang,
                generated_at=now,
                fonts=fonts,
            )
            out.update(meeting=m["title"])
            base = f"minutes_{m['start_utc'][:10]}"
        else:
            if not notes and not title:
                return {"error": "custom documents need a title or notes (the body)"}
            data = custom_pdf(
                org=c.guild_name, title=title, body=notes, lang=lang, generated_at=now, fonts=fonts
            )
            base = "document"
        name = f"{base}.pdf"
        c.attachments.append((name, data))
        c.state.remember_pdf(c.guild_id, c.author_id, name, data)
        out.update(status="created", file_name=name, size_bytes=len(data), thai_font=fonts.thai)
        out["note"] = "The PDF is attached to your reply automatically; do not paste its content."
        return out

    def _minutes_source(self, meeting_id: str, start_date: str, end_date: str) -> dict:
        c = self.c
        store = c.store
        m = None
        if meeting_id:
            m = (c.service.find(c.guild_id, meeting_id) if c.service else None) or None
        else:
            try:
                a = datetime.fromisoformat(start_date).replace(tzinfo=c.settings.zone)
                b = datetime.fromisoformat(end_date).replace(tzinfo=c.settings.zone) + timedelta(
                    days=1
                )
            except ValueError:
                return {"error": "give meeting_id, or start_date and end_date"}
            rows = store.list_meetings(
                c.guild_id, utc_iso(a), utc_iso(b), statuses=("done", "started", "scheduled")
            )
            rows = [r for r in rows if r.get("summary") and self._may_see(r)]
            m = rows[-1] if rows else None
        if m is None or not m.get("summary"):
            return {"error": "no meeting summary found for that meeting or period"}
        if not self._may_see(m):
            return {"error": "not_allowed", "note": "only attendees or executives"}
        return m

    def _may_see(self, m: dict) -> bool:
        c = self.c
        return (
            c.is_exec
            or c.author_id == m.get("organizer_id")
            or c.author_id in m.get("attendee_ids", [])
        )

    # ------------------------------------------------------------ email
    def draft_email(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        attach_file: str | None = None,
    ) -> dict:
        c = self.c
        if not c.email:
            return {"status": "disabled"}
        if not c.can_email:
            return {"error": "not_allowed", "note": "only executives and managers may email"}
        to_ok, bad = clean_addresses(to)
        cc_ok, bad_cc = clean_addresses(cc)
        if bad or bad_cc:
            return {"error": "invalid email address", "invalid": bad + bad_cc}
        if not to_ok:
            return {"error": "at least one recipient is required"}
        if len(to_ok) + len(cc_ok) > MAX_RECIPIENTS:
            return {"error": f"too many recipients (max {MAX_RECIPIENTS})"}
        subject = (subject or "").strip()[:MAX_SUBJECT_CHARS]
        body = (body or "").strip()[:MAX_BODY_CHARS]
        if not subject or not body:
            return {"error": "subject and body are required"}
        attach_name = None
        if attach_file:
            data = next((d for n, d in c.attachments if n == attach_file), None) or (
                c.state.recall_pdf(c.guild_id, c.author_id, attach_file)
            )
            if data is None:
                return {"error": f"no PDF named {attach_file!r} was made in this conversation"}
            attach_name = attach_file
        e = {
            "id": new_id(),
            "guild_id": c.guild_id,
            "author_id": c.author_id,
            "author_name": c.author_name,
            "to": to_ok,
            "cc": cc_ok,
            "subject": subject,
            "body": body,
            "attachment": attach_name,
            "status": "draft",
            "request_id": c.request_id,
            "created_at": utc_iso(c.now()),
            "sent_at": "",
            "sender": "",
            "confirmed_by": "",
            "result": "",
            "message_id": "",
        }
        c.store.put_email(e)
        c.state.confirm(("email", c.guild_id, c.author_id, e["id"]), c.request_id)  # propose
        return {
            "status": "drafted",
            "draft_id": e["id"],
            "draft": {k: e[k] for k in ("to", "cc", "subject", "body", "attachment")},
            "instruction": (
                "Show the full draft to the user and ask them to confirm. Do NOT claim it was "
                "sent. After they confirm in a later message, call send_email with draft_id."
            ),
        }

    def send_email(self, draft_id: str) -> dict:
        c = self.c
        if not c.email or c.sender is None:
            return {"status": "disabled"}
        if not c.can_email:
            return {"error": "not_allowed", "note": "only executives and managers may send email"}
        e = c.store.get_email(draft_id)
        if e is None or e["guild_id"] != c.guild_id or e["author_id"] != c.author_id:
            return {"error": "draft not found"}
        if e["status"] != "draft":
            return {"error": f"this draft is already {e['status']}"}
        if not c.state.confirm(("email", c.guild_id, c.author_id, e["id"]), c.request_id):
            return {
                "status": "needs_confirmation",
                "draft": {k: e[k] for k in ("to", "cc", "subject", "body", "attachment")},
                "instruction": "Ask the user to confirm sending; call send_email again in a "
                "later message.",
            }
        now = c.now()
        limit = c.settings.email_daily_limit
        if c.store.count_emails_since(c.guild_id, now - timedelta(days=1), EMAIL_STATUSES) >= limit:
            return {"error": "rate_limited", "limit_per_day": limit}
        attachments = []
        if e.get("attachment"):
            data = c.state.recall_pdf(c.guild_id, c.author_id, e["attachment"])
            if data is None:
                return {"error": "the attached PDF expired; make it again and redraft"}
            if len(data) > MAX_ATTACHMENT_BYTES:
                return {"error": "attachment too large"}
            attachments.append(
                (
                    e["attachment"],
                    data,
                    mimetypes.guess_type(e["attachment"])[0] or "application/pdf",
                )
            )
        mail = OutgoingEmail(e["to"], e["subject"], e["body"], e["cc"], attachments)
        res = c.sender.send(mail)
        e.update(
            status=res.status,
            sent_at=utc_iso(now),
            sender=c.sender.name,
            confirmed_by=f"user {c.author_id}",
            result=res.detail,
            message_id=res.message_id,
        )
        c.store.put_email(e)
        if res.status == "logged":
            return {
                "status": "not_configured",
                "message": t("email_not_configured", self._lang()),
                "recorded": True,
            }
        if not res.ok:
            return {"status": "failed", "detail": res.detail}
        return {"status": "sent", "to": e["to"], "cc": e["cc"], "subject": e["subject"]}


def build_secretary_tools(tools: SecretaryTools, calls: list[tuple[str, dict]]) -> list:
    """The agent-facing functions (docstrings are the tool descriptions)."""

    def logged(name: str, args: dict, fn):
        calls.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        except TimeoutError:
            return {"error": "timed out talking to Discord"}
        except Exception:
            logger.exception("secretary tool %s failed", name)
            return {"error": "internal error"}

    def schedule_meeting(
        title: str,
        start: str,
        duration_minutes: int = 60,
        attendees: list[str] | None = None,
        voice_channel: str | None = None,
        description: str = "",
        link: str | None = None,
    ) -> dict:
        """Schedule a meeting / appointment as a Discord Scheduled Event. The first call only
        returns a proposal (status needs_confirmation): show it to the user, mention the
        attendees, and wait. Call again with EXACTLY the same arguments only after the user
        confirmed in a later message; then it is created, attendees get a DM and reminders.

        Args:
            title: short meeting title.
            start: local start time as ISO 8601 (e.g. 2026-10-19T14:00), resolved by you from
                the user's words using today's date and the time zone in the instructions.
            duration_minutes: length, default 60.
            attendees: names or @mentions of the people to invite (not including the asker).
            voice_channel: name of a Discord voice channel for the meeting, optional.
            description: agenda or note, optional.
            link: a meeting URL (Google Meet, Zoom, Teams or any https link), optional. A link
                makes it an external event at that URL.
        """
        a = dict(
            title=title,
            start=start,
            duration_minutes=duration_minutes,
            attendees=attendees,
            voice_channel=voice_channel,
            description=description,
            link=link,
        )
        return logged("schedule_meeting", a, lambda: tools.schedule_meeting(**a))

    def list_meetings(range: str = "upcoming") -> dict:  # noqa: A002
        """List scheduled meetings of this server.

        Args:
            range: today, tomorrow, week, month, upcoming, or YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD.
        """
        return logged("list_meetings", {"range": range}, lambda: tools.list_meetings(range))

    def cancel_meeting(event_id: str) -> dict:
        """Cancel a scheduled meeting (organiser or executive). Needs the user's confirmation in
        a later message like schedule_meeting.

        Args:
            event_id: the meeting's event_id or id from list_meetings.
        """
        return logged(
            "cancel_meeting", {"event_id": event_id}, lambda: tools.cancel_meeting(event_id)
        )

    def reschedule_meeting(event_id: str, new_start: str) -> dict:
        """Move a meeting to a new start time (same length). Needs the user's confirmation in a
        later message like schedule_meeting.

        Args:
            event_id: the meeting's event_id or id from list_meetings.
            new_start: new local start, ISO 8601.
        """
        a = dict(event_id=event_id, new_start=new_start)
        return logged("reschedule_meeting", a, lambda: tools.reschedule_meeting(**a))

    def make_report_pdf(
        kind: str,
        start_date: str = "",
        end_date: str = "",
        title: str = "",
        notes: str = "",
        meeting_id: str = "",
    ) -> dict:
        """Build a PDF document that is attached to your reply. Executives get the whole team's
        data, everyone else only their own receipts. Totals are per currency.

        Args:
            kind: expense_summary (totals per currency, by category and submitter, receipt
                table), receipt_list (receipt table), meeting_minutes (a stored meeting summary)
                or custom (title + notes as the body).
            start_date: ISO date YYYY-MM-DD (period start; for meeting_minutes the period to look in).
            end_date: ISO date YYYY-MM-DD, inclusive.
            title: optional document title.
            notes: optional note printed in the document (the body for custom).
            meeting_id: for meeting_minutes, the meeting id from list_meetings, optional.
        """
        a = dict(
            kind=kind,
            start_date=start_date,
            end_date=end_date,
            title=title,
            notes=notes,
            meeting_id=meeting_id,
        )
        return logged("make_report_pdf", a, lambda: tools.make_report_pdf(**a))

    def draft_email(
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        attach_file: str | None = None,
    ) -> dict:
        """Draft an email (executives and managers only). Nothing is sent: show the full draft to
        the user and ask for confirmation.

        Args:
            to: recipient email addresses.
            subject: subject line.
            body: plain-text body, written in the language the user wants.
            cc: optional cc addresses.
            attach_file: optional file_name of a PDF made earlier by make_report_pdf.
        """
        a = dict(to=to, subject=subject, body=body, cc=cc, attach_file=attach_file)
        return logged("draft_email", a, lambda: tools.draft_email(**a))

    def send_email(draft_id: str) -> dict:
        """Send a drafted email. Only after the user confirmed THAT draft in a later message than
        the one that showed it. Executives and managers only; 20 emails per day per server.

        Args:
            draft_id: the draft_id returned by draft_email.
        """
        return logged("send_email", {"draft_id": draft_id}, lambda: tools.send_email(draft_id))

    out = []
    if tools.c.scheduling:
        out += [schedule_meeting, list_meetings, cancel_meeting, reschedule_meeting]
    if tools.c.documents:
        out.append(make_report_pdf)
    if tools.c.email and tools.c.can_email:
        out += [draft_email, send_email]
    return out
