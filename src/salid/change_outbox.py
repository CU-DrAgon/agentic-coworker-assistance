"""The change outbox: every change made on the central website reaches every platform and person,
is acknowledged, and is sent again every 15 minutes until it is (owner's rule, 2026-10-18).

Pure logic plus a store (DocStore kind `salid_change_outbox`, guild 0, key = change id; SQLite and
Firestore like the other stores). No Discord, no platform SDK: the consumers are
`change_outbox_bot.ChangeOutboxMixin` (Discord) and `platforms.change_outbox_platform`
(Slack / LINE / Telegram / WhatsApp).

Flow: a producer (the web route in `meeting_scribe.center`, or the Discord bot's watcher as a
safety net) calls `record_org_edit` / `ChangeOutbox.record`. `record` fills the targets: one
`apply` target per connected chat workspace of the org (the platform makes the change real:
Discord roles, ...) and one `notify` target per platform where the subject person is linked (the
person must confirm they saw it). A consumer asks `due()` for what to (re)send, sends, and calls
`mark_attempt`; a button click or a reply "รับทราบ / ok / got it" calls `ack`. After
`max_attempts` unanswered sends (or past the deadline for a platform nobody serves) the target is
`failed`, kept forever as an audit trail, and the org's owners and admins get one alert per platform.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from .coo_store import DocStore
from .i18n import CATALOG, t
from .orgs import Member, Org

logger = logging.getLogger(__name__)

KIND = "salid_change_outbox"  # DocStore kind
GUILD = 0
RESEND_MINUTES = 15
MAX_ATTEMPTS = 8  # about two hours
KEEP_DAYS = 90  # acked changes; a change with a failed target is kept forever
DEDUPE_MINUTES = 10
PRUNE_EVERY_S = 3600.0

KINDS = ("member_added", "member_removed", "role", "title", "level", "department", "link")
FIELD_KINDS = ("role", "title", "level", "department", "link")
OPEN = ("pending", "sent")

# "got it" in words (Thai and English), the reply that confirms a notice
ACK_RE = re.compile(
    r"^\s*(รับทราบ|ทราบแล้ว|ทราบ|รับรู้แล้ว|รับรู้|เห็นแล้ว|โอเค|ok|okay|got it|gotcha|noted|"
    r"seen|ack|acknowledged|confirm|confirmed|👍)\s*(ค่ะ|คะ|ครับ|คับ|นะคะ|นะครับ|นะ)?\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_ack_text(text: str) -> bool:
    return bool(ACK_RE.match(text or ""))


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _parse(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ------------------------------------------------------------------ models
class Target(BaseModel):
    workspace_id: str
    platform: str
    kind: str  # apply | notify
    person: str = ""  # notify: platform user id (or email); apply: ""
    status: str = "pending"  # pending | sent | acked | failed
    attempts: int = 0
    sent_at: str = ""
    acked_at: str = ""
    acked_how: str = ""  # applied | button | reply | nothing_to_apply | ...
    last_error: str = ""
    message_ref: str = ""  # the notice's message id, so a reply-to can be matched


class Change(BaseModel):
    id: str
    org_id: str
    at: str
    by: str = "salid"  # an email, "salid" or "detected"
    kind: str
    subject_email: str
    subject_name: str = ""
    before: str = ""
    after: str = ""
    targets: list[Target] = Field(default_factory=list)
    subject_ids: dict[str, str] = Field(default_factory=dict)  # platform -> user id
    subject_tenants: dict[str, str] = Field(default_factory=dict)  # platform -> tenant key
    alerted_on: list[str] = Field(default_factory=list)  # platforms whose admins were told


# ------------------------------------------------------------------ diff (pure)
def _link(m: Member) -> str:
    return ";".join(f"{p}:{i}" for p, i in sorted(m.all_identities().items()))


def _fields(m: Member) -> dict[str, str]:
    return {
        "role": m.role,
        "title": m.title,
        "level": str(m.level or 0) if m.level else "",
        "department": m.department,
        "link": _link(m),
    }


def _change(org_id: str, m: Member, kind: str, before: str, after: str, by: str, at: str) -> Change:
    return Change(
        id=uuid.uuid4().hex[:16],
        org_id=org_id,
        at=at,
        by=by,
        kind=kind,
        subject_email=m.email.lower(),
        subject_name=m.email.split("@")[0],
        before=before,
        after=after,
        subject_ids=m.all_identities(),
        subject_tenants={p: v for p, v in m.identity_tenants.items() if v},
    )


def diff_members(
    before: Org, after: Org, *, by: str = "salid", at: datetime | None = None
) -> list[Change]:
    """One change per field per member (and one per member added / removed). Pure."""
    stamp = _iso(at or _utcnow())
    old = {m.email.lower(): m for m in before.members}
    new = {m.email.lower(): m for m in after.members}
    out: list[Change] = []
    for email, m in new.items():
        if email not in old:
            out.append(_change(after.id, m, "member_added", "", m.role, by, stamp))
            continue
        was, now = _fields(old[email]), _fields(m)
        for kind in FIELD_KINDS:
            if was[kind] != now[kind]:
                out.append(_change(after.id, m, kind, was[kind], now[kind], by, stamp))
    for email, m in old.items():
        if email not in new:
            out.append(_change(before.id, m, "member_removed", m.role, "", by, stamp))
    return out


# ------------------------------------------------------------------ the store
class ChangeOutbox:
    def __init__(
        self,
        docs: DocStore,
        now: Callable[[], datetime] = _utcnow,
        *,
        resend_minutes: int = RESEND_MINUTES,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.docs, self.now = docs, now
        self.resend = timedelta(minutes=max(int(resend_minutes), 1))
        self.max_attempts = max(int(max_attempts), 1)
        self._pruned = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, docs: DocStore, settings: Any, now=None) -> ChangeOutbox:
        return cls(
            docs,
            now or getattr(settings, "now", _utcnow),
            resend_minutes=getattr(settings, "change_resend_minutes", RESEND_MINUTES),
            max_attempts=getattr(settings, "change_max_attempts", MAX_ATTEMPTS),
        )

    # ---- reading
    def get(self, change_id: str) -> Change | None:
        d = self.docs.get(KIND, GUILD, change_id)
        return Change(**d) if d else None

    def _all(self) -> list[Change]:
        return [Change(**d) for d in self.docs.find(KIND, GUILD)]

    def list_changes(self, org_id: str, limit: int = 50) -> list[Change]:
        mine = [c for c in self._all() if c.org_id == org_id]
        mine.sort(key=lambda c: c.at, reverse=True)
        return mine[: max(int(limit), 0)]

    # ---- recording
    def targets_for(self, change: Change, org: Org) -> list[Target]:
        out: list[Target] = []
        chat = [w for w in org.workspaces if w.platform != "web" and w.status != "disconnected"]
        for w in chat:
            out.append(Target(workspace_id=w.id, platform=w.platform, kind="apply"))
        for w in chat:
            uid = change.subject_ids.get(w.platform, "")
            if not uid:
                continue
            tenant = change.subject_tenants.get(w.platform, "")
            if tenant and tenant != w.external_id and w.platform != "discord":
                continue  # verified in another tenant: that link vouches for no other
            out.append(Target(workspace_id=w.id, platform=w.platform, kind="notify", person=uid))
        return out

    def record(self, changes: list[Change], org: Org) -> list[Change]:
        """Fill the targets and store the changes. A change equal to a recent one (same org,
        subject, kind and new value within `DEDUPE_MINUTES`) is dropped: the web and the Discord
        watcher may both see the same edit. Returns what was stored."""
        if not changes:
            return []
        now = self.now()
        horizon = _iso(now - timedelta(minutes=DEDUPE_MINUTES))
        with self._lock:
            recent = {
                (c.org_id, c.subject_email, c.kind, c.after) for c in self._all() if c.at >= horizon
            }
            stored: list[Change] = []
            for c in changes:
                key = (c.org_id, c.subject_email, c.kind, c.after)
                if key in recent:
                    continue
                recent.add(key)
                c.at = _iso(now)
                c.targets = self.targets_for(c, org)
                self.docs.put(KIND, GUILD, c.id, c.model_dump())
                stored.append(c)
        return stored

    # ---- state changes (each one atomic)
    def _update(self, change_id: str, fn: Callable[[Change], bool]) -> bool:
        def run(tx) -> bool:
            d = tx.get(KIND, GUILD, change_id)
            if d is None:
                return False
            c = Change(**d)
            if not fn(c):
                return False
            tx.put(KIND, GUILD, c.id, c.model_dump())
            return True

        return self.docs.transact(run)

    @staticmethod
    def _find(c: Change, workspace_id: str, person: str) -> Target | None:
        kind = "notify" if person else "apply"
        return next(
            (
                x
                for x in c.targets
                if x.workspace_id == workspace_id and x.kind == kind and x.person == person
            ),
            None,
        )

    def ack(self, change_id: str, workspace_id: str, person: str, how: str) -> bool:
        """Mark a target acknowledged. Idempotent: True only the first time."""

        def fn(c: Change) -> bool:
            x = self._find(c, workspace_id, person)
            if x is None or x.status == "acked":
                return False
            x.status, x.acked_at, x.acked_how = "acked", _iso(self.now()), how
            x.last_error = ""
            return True

        return self._update(change_id, fn)

    def mark_attempt(
        self,
        change_id: str,
        workspace_id: str,
        person: str,
        *,
        error: str = "",
        message_ref: str = "",
    ) -> bool:
        """One send (or apply) attempt: counts, stamps the time, remembers the message."""

        def fn(c: Change) -> bool:
            x = self._find(c, workspace_id, person)
            if x is None or x.status == "acked":
                return False
            x.attempts += 1
            x.sent_at = _iso(self.now())
            x.last_error = error[:200]
            if not error:
                x.status = "sent"
            if message_ref:
                x.message_ref = message_ref
            return True

        return self._update(change_id, fn)

    def open_for(self, workspace_id: str, person: str) -> list[tuple[Change, Target]]:
        """Notices this person was sent and has not confirmed (newest first)."""
        out = [
            (c, x)
            for c in self._all()
            for x in c.targets
            if x.workspace_id == workspace_id
            and x.kind == "notify"
            and x.person == person
            and x.status in ("sent", "failed")
        ]
        out.sort(key=lambda p: p[0].at, reverse=True)
        return out

    # ---- what to send
    def due(self) -> list[tuple[Change, Target]]:
        """Targets to (re)send now. Also fails the ones that ran out of attempts or time."""
        now = self.now()
        out: list[tuple[Change, Target]] = []
        for c in self._all():
            if not any(x.status in OPEN for x in c.targets):
                continue
            born = _parse(c.at) or now
            deadline = born + self.resend * (self.max_attempts + 1)
            failed: list[Target] = []
            for x in c.targets:
                if x.status not in OPEN:
                    continue
                last = _parse(x.sent_at)
                wait_over = last is None or now - last >= self.resend
                spent = x.attempts >= self.max_attempts and wait_over
                if spent or now >= deadline:
                    failed.append(x)
                elif wait_over:
                    out.append((c, x))
            if failed:
                keys = {(x.workspace_id, x.kind, x.person) for x in failed}

                def fn(cc: Change, keys=keys) -> bool:
                    hit = False
                    for x in cc.targets:
                        if (x.workspace_id, x.kind, x.person) in keys and x.status in OPEN:
                            x.status = "failed"
                            x.last_error = x.last_error or "not confirmed in time"
                            hit = True
                    return hit

                self._update(c.id, fn)
                out = [
                    (cc, x)
                    for cc, x in out
                    if cc.id != c.id or (x.workspace_id, x.kind, x.person) not in keys
                ]
        return out

    def to_alert(self, platform: str) -> list[Change]:
        """Changes with a failed target whose admins on `platform` were not told yet."""
        return [
            c
            for c in self._all()
            if platform not in c.alerted_on and any(x.status == "failed" for x in c.targets)
        ]

    def mark_alerted(self, change_id: str, platform: str) -> bool:
        def fn(c: Change) -> bool:
            if platform in c.alerted_on:
                return False
            c.alerted_on.append(platform)
            return True

        return self._update(change_id, fn)

    def prune(self, force: bool = False) -> int:
        """Drop acked changes older than `KEEP_DAYS`; changes with a failed target stay forever."""
        import time

        mono = time.monotonic()
        if not force and mono - self._pruned < PRUNE_EVERY_S and self._pruned:
            return 0
        self._pruned = mono or 1.0
        cutoff = _iso(self.now() - timedelta(days=KEEP_DAYS))
        n = 0
        for c in self._all():
            if c.at < cutoff and not any(x.status in ("failed", *OPEN) for x in c.targets):
                self.docs.delete(KIND, GUILD, c.id)
                n += 1
        return n


# ------------------------------------------------------------------ the web / watcher entry
def record_org_edit(
    docs: DocStore, before: Org, after: Org, by: str = "salid", settings=None
) -> int:
    """Record what changed between two versions of an org. NEVER raises: a failure here must not
    break the request or the sync that called it. Returns how many changes were stored."""
    try:
        changes = diff_members(before, after, by=by)
        if not changes:
            return 0
        outbox = ChangeOutbox.from_settings(docs, settings) if settings else ChangeOutbox(docs)
        return len(outbox.record(changes, after))
    except Exception:
        logger.warning("change outbox: could not record an org edit", exc_info=True)
        return 0


# ------------------------------------------------------------------ texts
STRINGS = {
    "chg_kind_role": {"th": "สิทธิ์", "en": "role"},
    "chg_kind_title": {"th": "ตำแหน่ง", "en": "position"},
    "chg_kind_level": {"th": "ระดับ", "en": "level"},
    "chg_kind_department": {"th": "แผนก", "en": "department"},
    "chg_kind_link": {"th": "บัญชีที่เชื่อม", "en": "linked accounts"},
    "chg_kind_member_added": {"th": "การเป็นสมาชิก", "en": "membership"},
    "chg_kind_member_removed": {"th": "การเป็นสมาชิก", "en": "membership"},
    "chg_by_center": {"th": "ระบบกลาง", "en": "the central system"},
    "chg_card": {
        "th": "{what}ของคุณใน {org} เปลี่ยนแปลง: {before} → {after} (โดย {by}) ค่ะ",
        "en": "Your {what} in {org} changed: {before} → {after} (by {by}).",
    },
    "chg_added": {
        "th": "คุณถูกเพิ่มเข้า {org} เป็น {after} (โดย {by}) ค่ะ",
        "en": "You were added to {org} as {after} (by {by}).",
    },
    "chg_removed": {
        "th": "คุณถูกนำออกจาก {org} (โดย {by}) ค่ะ",
        "en": "You were removed from {org} (by {by}).",
    },
    "chg_reminder": {"th": "(แจ้งเตือนครั้งที่ {n})", "en": "(reminder {n})"},
    "chg_btn": {"th": "รับทราบ", "en": "Got it"},
    "chg_hint": {
        "th": "ตอบ “รับทราบ” เพื่อยืนยันว่าเห็นแล้วค่ะ",
        "en": "Reply OK to confirm you saw this.",
    },
    "chg_thanks": {"th": "ขอบคุณค่ะ บันทึกว่ารับทราบแล้ว", "en": "Thanks, noted."},
    "chg_done": {"th": "✅ รับทราบแล้ว", "en": "✅ Confirmed"},
    "chg_not_yours": {"th": "แจ้งเตือนนี้ไม่ใช่ของคุณค่ะ", "en": "This notice is not for you."},
    "chg_handled": {"th": "รับทราบไปแล้วค่ะ", "en": "Already confirmed."},
    "chg_alert": {
        "th": (
            "⚠️ การเปลี่ยนแปลงใน {org} ยังไม่มีใครยืนยันหลังส่งซ้ำ {n} ครั้ง: "
            "{subject} · {what}: {before} → {after} ยังรออยู่ที่: {where}"
        ),
        "en": (
            "⚠️ A change in {org} is still not confirmed after {n} tries: "
            "{subject} · {what}: {before} → {after}. Still waiting on: {where}"
        ),
    },
}
CATALOG.update(STRINGS)

DASH = "—"


def _what(change: Change, lang: str) -> str:
    return t(f"chg_kind_{change.kind}", lang)


def _by(change: Change, lang: str) -> str:
    return change.by if "@" in change.by else t("chg_by_center", lang)


def change_text(change: Change, org_name: str, lang: str, reminder: int = 0) -> str:
    """The notice a person gets. `reminder` > 0 appends the reminder count."""
    kw = {"org": org_name, "by": _by(change, lang), "after": change.after or DASH}
    if change.kind == "member_added":
        text = t("chg_added", lang, **kw)
    elif change.kind == "member_removed":
        text = t("chg_removed", lang, **kw)
    else:
        text = t("chg_card", lang, what=_what(change, lang), before=change.before or DASH, **kw)
    if reminder > 0:
        text = f"{text} {t('chg_reminder', lang, n=reminder)}"
    return text


def alert_text(change: Change, org_name: str, lang: str, attempts: int) -> str:
    where = ", ".join(
        sorted(
            {
                f"{x.platform}{' ' + x.person if x.person else ''}"
                for x in change.targets
                if x.status == "failed"
            }
        )
    )
    return t(
        "chg_alert",
        lang,
        org=org_name,
        n=attempts,
        subject=change.subject_email,
        what=_what(change, lang),
        before=change.before or DASH,
        after=change.after or DASH,
        where=where or DASH,
    )


def admins_of(org: Org) -> list[Member]:
    return [m for m in org.members if m.role in ("owner", "admin")]


# ------------------------------------------------------------------ agent tool
def change_status_tools(outbox: ChangeOutbox, org_id: Callable[[], str | None]) -> list:
    """`change_status`: who has / has not confirmed the recent changes. Executives only (the
    caller registers it only for them); scoped to the org of the asking server in code."""

    def change_status(name: str = "") -> dict:
        """Recent changes made on the central website (position, role, level, department,
        membership) and who has or has not confirmed seeing them. `name` filters by person
        (part of the name or email). Use for "who hasn't confirmed the position change?"."""
        oid = org_id()
        if not oid:
            return {"error": "this server is not linked to a company in the central system"}
        key = (name or "").strip().lower()
        rows = []
        for c in outbox.list_changes(oid, 100):
            if key and key not in c.subject_email and key not in c.subject_name.lower():
                continue
            notify = [x for x in c.targets if x.kind == "notify"]
            rows.append(
                {
                    "when": c.at,
                    "who": c.subject_email,
                    "what": c.kind,
                    "before": c.before,
                    "after": c.after,
                    "by": c.by,
                    "confirmed": [x.platform for x in notify if x.status == "acked"],
                    "waiting": [x.platform for x in notify if x.status in ("pending", "sent")],
                    "gave_up": [x.platform for x in c.targets if x.status == "failed"],
                    "applied_on": [
                        x.platform for x in c.targets if x.kind == "apply" and x.status == "acked"
                    ],
                }
            )
            if len(rows) >= 15:
                break
        return {"changes": rows, "count": len(rows)}

    return [change_status]
