"""Approval requests (like Lark Approval): expense, purchase, leave and other requests.

A member asks Salid ("ขอเบิกค่าแท็กซี่ 350 บาท"); after the member confirms, Salid DMs the approver
with Approve / Decline buttons and tells the requester the outcome. This module is the pure side:
the record (`Approval`), its storage (`ApprovalRepo` on the shared `DocStore`, so SQLite and
Firestore both work), the approver rules (`pick_approver`), the agent tools (`ApprovalTools`) and
a small read API for other features (`ApprovalRepo.approved_leave`). The Discord side (DMs,
buttons, timers) is in `salid.approvals_bot`.

Money: an approval carries its own amount and currency and never touches the ledger. Linking a
receipt to an approved expense only stores the receipt id, so per-currency totals do not change.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable, Collection
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, Field

from .coo_store import APPROVALS, DocStore
from .i18n import CATALOG
from .secretary import SecretaryState, fingerprint
from .tasks import match_person

KINDS = ("expense", "purchase", "leave", "other")
STATUSES = ("pending", "approved", "declined", "cancelled")
MONEY_KINDS = ("expense", "purchase")
MAX_LIST = 20
STALE_AFTER = timedelta(days=1)  # approvers are reminded of requests pending longer than this
MAX_TEXT = 300

STRINGS: dict[str, dict[str, str]] = {
    "appr_dm": {
        "th": "คำขออนุมัติจาก {name} ({kind}): {title}\n{details}เหตุผล: {reason}\nรหัสคำขอ {id}",
        "en": "Approval request from {name} ({kind}): {title}\n{details}Reason: {reason}\nRequest {id}",
    },
    "appr_btn_ok": {"th": "อนุมัติ", "en": "Approve"},
    "appr_btn_no": {"th": "ไม่อนุมัติ", "en": "Decline"},
    "appr_kind_expense": {"th": "เบิกค่าใช้จ่าย", "en": "expense"},
    "appr_kind_purchase": {"th": "ขอซื้อ", "en": "purchase"},
    "appr_kind_leave": {"th": "ลา", "en": "leave"},
    "appr_kind_other": {"th": "อื่นๆ", "en": "other"},
    "appr_amount": {"th": "จำนวนเงิน: {amount}\n", "en": "Amount: {amount}\n"},
    "appr_dates": {"th": "วันที่: {start} ถึง {end}\n", "en": "Dates: {start} to {end}\n"},
    "appr_receipt": {"th": "ใบเสร็จที่แนบ: #{id}\n", "en": "Attached receipt: #{id}\n"},
    "appr_modal_title": {"th": "เหตุผลที่ไม่อนุมัติ", "en": "Reason for declining"},
    "appr_modal_label": {"th": "เหตุผล (ไม่บังคับ)", "en": "Reason (optional)"},
    "appr_handled": {
        "th": "คำขอนี้ถูกจัดการไปแล้วค่ะ",
        "en": "This request has already been handled.",
    },
    "appr_not_yours": {
        "th": "คำขอนี้ไม่ได้ส่งถึงคุณค่ะ",
        "en": "This request was not sent to you.",
    },
    "appr_owner_approved": {
        "th": "✅ อนุมัติคำขอของ {name} แล้วค่ะ",
        "en": "✅ You approved {name}'s request.",
    },
    "appr_owner_declined": {
        "th": "🚫 ไม่อนุมัติคำขอของ {name} แล้วค่ะ",
        "en": "🚫 You declined {name}'s request.",
    },
    "appr_req_approved": {
        "th": "คำขอ “{title}” ได้รับอนุมัติจาก {who} แล้วค่ะ{note}",
        "en": "Your request “{title}” was approved by {who}.{note}",
    },
    "appr_req_declined": {
        "th": "คำขอ “{title}” ไม่ได้รับอนุมัติจาก {who} ค่ะ{note}",
        "en": "Your request “{title}” was declined by {who}.{note}",
    },
    "appr_note": {"th": " เหตุผล: {note}", "en": " Note: {note}"},
    "appr_cancelled_dm": {
        "th": "{name} ยกเลิกคำขอ “{title}” แล้วค่ะ",
        "en": "{name} cancelled the request “{title}”.",
    },
    "bf_approvals": {
        "th": "คำขออนุมัติที่รอคุณอยู่",
        "en": "Approval requests waiting for you",
    },
    "bf_approval_line": {
        "th": "• #{id} {requester}: {title} (รอมา {days} วัน)",
        "en": "• #{id} {requester}: {title} (waiting {days} d)",
    },
    "rep_approvals": {
        "th": "คำขออนุมัติ: รออยู่ {pending} รายการ (ค้างเกิน 1 วัน {stale}) อนุมัติวันนี้ {approved}",
        "en": "Approvals: {pending} pending ({stale} waiting over a day), {approved} approved today",
    },
}
CATALOG.update(STRINGS)  # one import registers the strings (keeps i18n.py conflict free)


class Approval(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    guild_id: int
    requester_id: int
    requester_name: str = ""
    approver_id: int
    approver_name: str = ""
    kind: str = "other"
    title: str
    amount: float | None = None
    currency: str = ""
    reason: str = ""
    start_date: str = ""  # leave: first day (ISO)
    end_date: str = ""  # leave: last day (ISO, inclusive)
    status: str = "pending"
    created_at: str = ""
    decided_at: str = ""
    decision_note: str = ""
    receipt_id: int | None = None  # the ledger receipt that settled an approved expense
    lang: str = "th"


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def fmt_amount(amount: float | None, currency: str = "") -> str:
    if amount is None:
        return ""
    s = f"{amount:,.2f}".removesuffix(".00")
    return f"{s} {currency}".strip()


# ---------------------------------------------------------------- who approves
def pick_approver(requester_id: int, ranks: dict[int, int], exec_ids: Collection[int]) -> int:
    """The requester's nearest higher-ranked member (Discord role hierarchy; executives win a
    tie), else the highest-ranked executive other than the requester. 0 when nobody qualifies."""
    mine = ranks.get(requester_id)
    higher = {u: r for u, r in ranks.items() if u != requester_id and (mine is None or r > mine)}
    if higher:
        nearest = min(higher.values())
        pool = [u for u, r in higher.items() if r == nearest]
        return min(pool, key=lambda u: (u not in exec_ids, u))
    execs = [u for u in exec_ids if u != requester_id and u in ranks]
    return max(execs, key=lambda u: (ranks[u], -u)) if execs else 0


# ---------------------------------------------------------------- storage
class ApprovalRepo:
    def __init__(self, docs: DocStore, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.docs, self.now = docs, now

    def _save(self, a: Approval) -> Approval:
        self.docs.put(APPROVALS, a.guild_id, a.id, a.model_dump())
        return a

    def create(self, a: Approval) -> Approval:
        a.created_at = a.created_at or _iso(self.now())
        return self._save(a)

    def get(self, approval_id: str) -> Approval | None:
        d = self.docs.find_any(APPROVALS, approval_id)
        return Approval(**d) if d else None

    def list(
        self,
        guild_id: int,
        *,
        requester_id: int | None = None,
        approver_id: int | None = None,
        status: str | None = None,
    ) -> list[Approval]:
        """Newest first."""
        rows = [Approval(**d) for d in self.docs.find(APPROVALS, guild_id)]
        rows = [
            a
            for a in rows
            if (requester_id is None or a.requester_id == requester_id)
            and (approver_id is None or a.approver_id == approver_id)
            and (status is None or a.status == status)
        ]
        return sorted(rows, key=lambda a: a.created_at, reverse=True)

    def decide(self, approval_id: str, status: str, note: str = "") -> Approval | None:
        """Approve / decline / cancel a pending request; None when unknown or already decided."""
        a = self.get(approval_id)
        if a is None or a.status != "pending" or status not in STATUSES[1:]:
            return None
        a.status, a.decision_note = status, (note or "").strip()[:MAX_TEXT]
        a.decided_at = _iso(self.now())
        return self._save(a)

    def attach_receipt(self, approval_id: str, receipt_id: int) -> Approval | None:
        a = self.get(approval_id)
        if a is None:
            return None
        a.receipt_id = receipt_id
        return self._save(a)

    # reminders and reports
    def stale_pending(self, guild_id: int, approver_id: int) -> list[Approval]:
        """Pending requests for this approver waiting longer than STALE_AFTER, oldest first."""
        cutoff = _iso(self.now() - STALE_AFTER)
        rows = self.list(guild_id, approver_id=approver_id, status="pending")
        return sorted((a for a in rows if a.created_at <= cutoff), key=lambda a: a.created_at)

    def counts(self, guild_id: int, day: date, zone) -> dict:
        """pending now, stale (pending over a day), approved / declined on local `day`."""
        cutoff = _iso(self.now() - STALE_AFTER)
        c: Counter = Counter()
        for a in self.list(guild_id):
            if a.status == "pending":
                c["pending"] += 1
                c["stale"] += a.created_at <= cutoff
            elif a.status in ("approved", "declined") and _local_day(a.decided_at, zone) == day:
                c[a.status] += 1
        return {k: c[k] for k in ("pending", "stale", "approved", "declined")}

    # read API for other features
    def approved_leave(
        self, guild_id: int, user_id: int, start: date | str, end: date | str
    ) -> list[tuple[date, date]]:
        """Approved leave intervals (first day, last day, inclusive) of `user_id` that overlap
        [start, end], sorted. Leave outside the range is not returned."""
        lo, hi = _as_date(start), _as_date(end)
        out = []
        for a in self.list(guild_id, requester_id=user_id, status="approved"):
            if a.kind != "leave" or not a.start_date:
                continue
            first, last = (
                date.fromisoformat(a.start_date),
                date.fromisoformat(a.end_date or a.start_date),
            )
            if first <= hi and last >= lo:
                out.append((first, last))
        return sorted(out)


def _as_date(v: date | str) -> date:
    return (
        v
        if isinstance(v, date) and not isinstance(v, datetime)
        else date.fromisoformat(str(v)[:10])
    )


def _local_day(iso_ts: str, zone) -> date | None:
    try:
        return datetime.fromisoformat(iso_ts).astimezone(zone).date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- agent tools
def _view(a: Approval) -> dict:
    return {
        "id": a.id,
        "kind": a.kind,
        "title": a.title,
        "amount": a.amount,
        "currency": a.currency or None,
        "reason": a.reason,
        "from": a.requester_name or a.requester_id,
        "approver": a.approver_name or a.approver_id,
        "status": a.status,
        "created": a.created_at,
        "decided": a.decided_at or None,
        "decision_note": a.decision_note or None,
        "leave": [a.start_date, a.end_date or a.start_date] if a.start_date else None,
        "receipt_id": a.receipt_id,
    }


class ApprovalTools:
    """Tools bound to one guild and one asker (neither is ever a model argument)."""

    def __init__(
        self,
        *,
        guild_id: int,
        asker_id: int,
        asker_name: str,
        is_exec: bool,
        repo: ApprovalRepo,
        people: dict[int, str],
        ranks: Callable[[], dict[int, int]],
        exec_ids: Callable[[], Collection[int]],
        receipts,  # the ledger Store (get by id)
        today: Callable[[], date],
        state: SecretaryState,
        request_id: str,
        lang: str,
        notify: Callable[[Approval], bool],
    ):
        self.guild_id, self.asker_id, self.asker_name = guild_id, asker_id, asker_name
        self.is_exec, self.repo, self.people = is_exec, repo, people
        self.ranks, self.exec_ids, self.receipts = ranks, exec_ids, receipts
        self.today, self.state, self.request_id = today, state, request_id
        self.lang, self.notify = lang, notify

    def _receipt(self, receipt_id) -> tuple[object | None, dict | None]:
        try:
            rid = int(receipt_id)
        except (TypeError, ValueError):
            return None, {"error": "receipt id must be a number"}
        r = self.receipts.get(rid)
        if r is None or r.guild_id != self.guild_id or r.status != "active":
            return None, {"error": f"no receipt #{rid} in this server"}
        if not (self.is_exec or r.submitter_id == self.asker_id):
            return None, {"error": "not allowed: you can only link your own receipts"}
        return r, None

    def request_approval(
        self,
        kind: str,
        title: str,
        reason: str,
        amount: float | None = None,
        currency: str | None = None,
        approver: str | None = None,
        attach_receipt_id: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        kind = (kind or "").strip().lower()
        if kind not in KINDS:
            return {"error": f"kind must be one of {', '.join(KINDS)}"}
        title = " ".join((title or "").split())[:120]
        if not title:
            return {"error": "title is required"}
        reason = " ".join((reason or "").split())[:MAX_TEXT]
        if not reason:
            return {"status": "needs_info", "instruction": "Ask the user for a short reason."}
        amt, cur = None, ""
        if amount is not None:
            try:
                amt = round(float(amount), 2)
            except (TypeError, ValueError):
                return {"error": "amount must be a number"}
            if amt <= 0:
                return {"error": "amount must be positive"}
            cur = (currency or "THB").strip().upper()[:5]
        elif kind in MONEY_KINDS:
            return {"status": "needs_info", "instruction": "Ask the user for the amount."}
        first = last = ""
        if kind == "leave":
            try:
                d1 = date.fromisoformat(start_date or "")
                d2 = date.fromisoformat(end_date) if end_date else d1
            except ValueError:
                return {
                    "status": "needs_info",
                    "instruction": "Ask which day(s) the leave is; pass start_date / end_date as "
                    "YYYY-MM-DD.",
                }
            if d2 < d1 or d2 < self.today() or (d2 - d1).days > 90:
                return {"error": "leave dates are invalid (past, reversed or over 90 days)"}
            first, last = d1.isoformat(), d2.isoformat()
        receipt_id = None
        if attach_receipt_id is not None:
            if kind not in MONEY_KINDS:
                return {"error": "only an expense or purchase can have a receipt"}
            r, err = self._receipt(attach_receipt_id)
            if err:
                return err
            receipt_id = r.id
        who, err = self._approver(approver)
        if err:
            return err
        key = (
            "approval",
            self.guild_id,
            self.asker_id,
            fingerprint(kind, title, amt, cur, reason, who, first, last, receipt_id),
        )
        proposal = {
            "kind": kind,
            "title": title,
            "amount": fmt_amount(amt, cur) or None,
            "reason": reason,
            "leave": [first, last] if first else None,
            "receipt_id": receipt_id,
            "approver": self.people.get(who, str(who)),
            "approver_mention": f"<@{who}>",
        }
        dup = next(
            (
                a
                for a in self.repo.list(self.guild_id, requester_id=self.asker_id, status="pending")
                if (a.kind, a.title, a.amount, a.start_date) == (kind, title, amt, first)
            ),
            None,
        )
        if dup is not None:
            return {"status": "already_pending", "request": _view(dup)}
        if not self.state.confirm(key, self.request_id):
            return {
                "status": "needs_confirmation",
                "proposal": proposal,
                "instruction": (
                    "Read the request back (what, amount, who will be asked) and ask the user "
                    "to confirm (Thai: “ยืนยันไหมคะ”; English: “Shall I send it?”; in a voice room say it "
                    "aloud). Do NOT say "
                    "it was sent. After they confirm in a later message call request_approval "
                    "again with exactly the same arguments."
                ),
            }
        a = self.repo.create(
            Approval(
                guild_id=self.guild_id,
                requester_id=self.asker_id,
                requester_name=self.asker_name,
                approver_id=who,
                approver_name=self.people.get(who, ""),
                kind=kind,
                title=title,
                amount=amt,
                currency=cur,
                reason=reason,
                start_date=first,
                end_date=last,
                receipt_id=receipt_id,
                lang=self.lang,
            )
        )
        if not self.notify(a):
            self.repo.decide(a.id, "cancelled", "approver could not be messaged")
            return {
                "error": "could_not_dm_approver",
                "instruction": "Tell the user the approver could not be messaged (DMs closed or "
                "not in the server) and suggest asking them directly or naming someone else.",
            }
        return {"status": "sent", "request": _view(a)}

    def _approver(self, approver: str | None) -> tuple[int, dict | None]:
        if approver and approver.strip():
            who = match_person(approver, self.people)
            if not who:
                return 0, {
                    "status": "needs_info",
                    "instruction": f"Cannot tell who {approver!r} is; ask for the exact name.",
                }
        else:
            who = pick_approver(self.asker_id, self.ranks(), set(self.exec_ids()))
            if not who:
                return 0, {
                    "status": "needs_info",
                    "instruction": "Nobody above the user can approve; ask who should approve.",
                }
        if who == self.asker_id:
            return 0, {"error": "you cannot approve your own request; name someone else"}
        return who, None

    def list_approvals(self, status: str | None = None, scope: str = "mine") -> dict:
        if scope not in ("mine", "to_me"):
            return {"error": "scope must be mine or to_me"}
        if status in (None, "", "all"):
            status = None
        elif status not in STATUSES:
            return {"error": f"status must be one of {', '.join(STATUSES)} or all"}
        by = {"requester_id": self.asker_id} if scope == "mine" else {"approver_id": self.asker_id}
        rows = self.repo.list(self.guild_id, status=status, **by)
        out: dict = {"count": len(rows), "approvals": [_view(a) for a in rows[:MAX_LIST]]}
        if not rows:
            out["note"] = "nothing matches"
        return out

    def cancel_approval(self, approval_id: str) -> dict:
        a = self.repo.get((approval_id or "").lstrip("#").strip())
        if a is None or a.guild_id != self.guild_id or a.requester_id != self.asker_id:
            return {"error": "no such request of yours"}
        done = self.repo.decide(a.id, "cancelled")
        if done is None:
            return {"error": f"already {a.status}; only a pending request can be cancelled"}
        self.notify(done)  # tells the approver (best effort)
        return {"cancelled": _view(done)}

    def attach_receipt(self, approval_id: str, receipt_id: int) -> dict:
        a = self.repo.get((approval_id or "").lstrip("#").strip())
        if a is None or a.guild_id != self.guild_id or a.requester_id != self.asker_id:
            return {"error": "no such request of yours"}
        if a.kind not in MONEY_KINDS or a.status not in ("pending", "approved"):
            return {
                "error": "a receipt can be linked to an expense or purchase that is not declined"
            }
        r, err = self._receipt(receipt_id)
        if err:
            return err
        done = self.repo.attach_receipt(a.id, r.id)
        out: dict = {"linked": _view(done), "ledger_totals": "unchanged"}
        if a.currency and (r.currency or "THB") != a.currency:
            out["warning"] = f"receipt currency {r.currency} differs from approved {a.currency}"
        elif a.amount is not None and r.total is not None and abs(r.total - a.amount) > 0.01:
            out["warning"] = f"receipt total {r.total} differs from approved amount {a.amount}"
        return out


def build_approval_tools(c: ApprovalTools, calls: list | None = None) -> list:
    """Plain functions bound to `c` (their docstrings are the tool descriptions)."""
    log = calls if calls is not None else []

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    def request_approval(
        kind: str,
        title: str,
        reason: str,
        amount: float | None = None,
        currency: str | None = None,
        approver: str | None = None,
        attach_receipt_id: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Ask someone to approve a request on the asker's behalf (like Lark Approval): an
        expense claim (“ขอเบิกค่าแท็กซี่ 350 บาท”), a purchase (“ขออนุมัติซื้อจอ 2 ตัว 12,000”),
        leave (“ขอลาพรุ่งนี้”) or anything else. The approver gets a private message with
        Approve / Decline buttons and the asker is told the outcome. The first call returns
        needs_confirmation: read the request back and ask the user to confirm, then call again
        with the same arguments after they confirm in a later message. Never say it was sent
        before status is sent. Needs a reason (ask if missing) and, for expense / purchase, an
        amount.

        Args:
            kind: expense, purchase, leave or other.
            title: what it is, a few words (e.g. “ค่าแท็กซี่ไปหาลูกค้า”).
            reason: short why.
            amount: money amount (expense / purchase), optional otherwise.
            currency: THB (default), USD, ...
            approver: a named person (name or @mention); leave out to use the asker's boss by
                the server's role hierarchy. Nobody can approve their own request.
            attach_receipt_id: receipt id from the ledger that already documents it, optional.
            start_date: leave only, first day YYYY-MM-DD (convert “พรุ่งนี้” from today's date).
            end_date: leave only, last day YYYY-MM-DD, default the same day.
        """
        a = dict(
            kind=kind, title=title, reason=reason, amount=amount, currency=currency,
            approver=approver, attach_receipt_id=attach_receipt_id,
            start_date=start_date, end_date=end_date,
        )  # fmt: skip
        return safe("request_approval", a, lambda: c.request_approval(**a))

    def list_approvals(status: str | None = None, scope: str = "mine") -> dict:
        """The asker's approval requests. scope "mine" = requests the asker made (“คำขอของฉันถึงไหน”),
        "to_me" = requests waiting for the asker to decide (“มีอะไรรอฉันอนุมัติบ้าง”; the
        asker approves with the buttons in the DM Salid sent).

        Args:
            status: pending, approved, declined, cancelled or all (default all).
            scope: mine (default) or to_me.
        """
        a = dict(status=status, scope=scope)
        return safe("list_approvals", a, lambda: c.list_approvals(**a))

    def cancel_approval(approval_id: str) -> dict:
        """Withdraw the asker's own pending approval request.

        Args:
            approval_id: the id from list_approvals.
        """
        a = dict(approval_id=approval_id)
        return safe("cancel_approval", a, lambda: c.cancel_approval(**a))

    def attach_receipt_to_approval(approval_id: str, receipt_id: int) -> dict:
        """Link a ledger receipt to an expense / purchase approval, e.g. “ใบเสร็จนี้คือที่อนุมัติ
        ไปเมื่อวาน”. Only stores the link; the ledger totals do not change. Find the approval with
        list_approvals and the receipt id from the ledger tools.

        Args:
            approval_id: the approval id.
            receipt_id: the receipt id.
        """
        a = dict(approval_id=approval_id, receipt_id=receipt_id)
        return safe("attach_receipt_to_approval", a, lambda: c.attach_receipt(**a))

    return [request_approval, list_approvals, cancel_approval, attach_receipt_to_approval]
