"""The Discord side of approval requests (a mixin of `SecretaryBot`).

* `approval_tools`: the agent tools for one speaker (chat and voice share them via
  `SecretaryBot._agent_kwargs`);
* the DM to the approver with persistent Approve / Decline buttons (`ApprovalButton`, a
  DynamicItem like `DisclosureButton`, so they survive a restart; Decline asks for an optional
  reason in a modal), and the outcome DM to the requester;
* numbers for the evening report and the list for the morning-brief reminder.

The rules and storage live in `salid.approvals`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import date, datetime

import discord

from .approvals import (
    Approval,
    ApprovalRepo,
    ApprovalTools,
    build_approval_tools,
    fmt_amount,
)
from .i18n import resolve_lang, t
from .secretary import new_id

logger = logging.getLogger(__name__)


class DeclineModal(discord.ui.Modal):
    """Optional reason for declining."""

    def __init__(self, approval_id: str, lang: str) -> None:
        super().__init__(title=t("appr_modal_title", lang), timeout=None)
        self.approval_id = approval_id
        self.note = discord.ui.TextInput(
            label=t("appr_modal_label", lang),
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=300,
        )
        self.add_item(self.note)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_approval(
            interaction, "no", self.approval_id, str(self.note.value or "")
        )


class ApprovalButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:appr:(?P<act>ok|no):(?P<id>[0-9a-f]{16})",
):
    """Approve / Decline on an approval DM. The custom id carries the request id and the bot
    registers the class once, so the buttons keep working after a restart."""

    def __init__(self, act: str, approval_id: str, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=discord.ButtonStyle.success if act == "ok" else discord.ButtonStyle.danger,
                custom_id=f"salid:appr:{act}:{approval_id}",
            )
        )
        self.act, self.approval_id = act, approval_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], match["id"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.act == "no":
            lang = await interaction.client.approval_lang(self.approval_id)
            await interaction.response.send_modal(DeclineModal(self.approval_id, lang))
        else:
            await interaction.client.handle_approval(interaction, "ok", self.approval_id)


class ApprovalMixin:
    def _approval_init(self) -> None:
        self.approvals = ApprovalRepo(self.docs, now=lambda: self.s.now())

    def _approvals_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "approvals")

    def _approval_user_lang(self, gid: int, uid: int) -> str:
        lang = self._profile_of(gid, uid).language
        return resolve_lang(lang) if lang else self._guild_lang(gid)

    async def approval_lang(self, approval_id: str) -> str:
        a = await asyncio.to_thread(self.approvals.get, approval_id)
        return self._approval_user_lang(a.guild_id, a.approver_id) if a else "th"

    # ------------------------------------------------------------- read API for other features
    def approved_leave(self, guild_id: int, user_id: int, start, end) -> list[tuple[date, date]]:
        """Approved leave of a member overlapping [start, end] (dates or ISO strings), as
        (first day, last day) pairs, sorted. Blocking."""
        return self.approvals.approved_leave(guild_id, user_id, start, end)

    # ------------------------------------------------------------- agent tools
    def _approval_tools(self, guild, member, is_exec: bool, lang: str) -> list:
        if not self._approvals_on(guild.id):
            return []
        loop = asyncio.get_running_loop()
        c = ApprovalTools(
            guild_id=guild.id,
            asker_id=member.id,
            asker_name=getattr(member, "display_name", str(member.id)),
            is_exec=is_exec,
            repo=self.approvals,
            people=self._people(guild),
            ranks=lambda: self._ranks(guild),
            exec_ids=lambda: {m.id for m in self._members(guild) if self.is_exec_member(m)},
            receipts=self.store,
            today=lambda: self.s.now().date(),
            state=self.secretary_state,
            request_id=new_id(),  # one per request: a send confirms only in a later message
            lang=lang,
            notify=lambda a: self._notify_approval(guild, a, loop),
        )
        return build_approval_tools(c)

    def _notify_approval(self, guild, a: Approval, loop) -> bool:
        """Tool body (the agent's worker thread): DM the approver a new request, or tell them it
        was cancelled. True when the DM went out."""
        coro = (
            self._send_approval_dm(guild, a)
            if a.status == "pending"
            else self._send_cancel_dm(guild, a)
        )
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)
        except Exception:
            logger.warning("approval DM failed", exc_info=True)
            return False

    @staticmethod
    def _approval_details(a: Approval, lang: str) -> str:
        out = ""
        if a.amount is not None:
            out += t("appr_amount", lang, amount=fmt_amount(a.amount, a.currency))
        if a.start_date:
            out += t("appr_dates", lang, start=a.start_date, end=a.end_date or a.start_date)
        if a.receipt_id:
            out += t("appr_receipt", lang, id=a.receipt_id)
        return out

    async def _send_approval_dm(self, guild, a: Approval) -> bool:
        approver = guild.get_member(a.approver_id)
        if approver is None:
            return False
        lang = self._approval_user_lang(guild.id, a.approver_id)
        view = discord.ui.View(timeout=None)
        view.add_item(ApprovalButton("ok", a.id, t("appr_btn_ok", lang)))
        view.add_item(ApprovalButton("no", a.id, t("appr_btn_no", lang)))
        text = t(
            "appr_dm",
            lang,
            name=a.requester_name or str(a.requester_id),
            kind=t(f"appr_kind_{a.kind}", lang),
            title=a.title,
            details=self._approval_details(a, lang),
            reason=a.reason,
            id=a.id,
        )
        try:
            await approver.send(text, view=view)
        except discord.HTTPException:
            return False
        return True

    async def _send_cancel_dm(self, guild, a: Approval) -> bool:
        approver = guild.get_member(a.approver_id)
        if approver is None:
            return False
        lang = self._approval_user_lang(guild.id, a.approver_id)
        with contextlib.suppress(discord.HTTPException):
            await approver.send(t("appr_cancelled_dm", lang, name=a.requester_name, title=a.title))
            return True
        return False

    # ------------------------------------------------------------- the approver's click
    async def handle_approval(
        self, interaction, act: str, approval_id: str, note: str = ""
    ) -> None:
        """Approve / Decline. Only the named approver counts; a decided request stays decided."""
        a = await asyncio.to_thread(self.approvals.get, approval_id)
        lang = self._approval_user_lang(a.guild_id, a.approver_id) if a else "th"
        if a is None or a.status != "pending":
            await interaction.response.send_message(t("appr_handled", lang), ephemeral=True)
            return
        if interaction.user.id != a.approver_id or interaction.user.id == a.requester_id:
            await interaction.response.send_message(t("appr_not_yours", lang), ephemeral=True)
            return
        decided = await asyncio.to_thread(
            self.approvals.decide, approval_id, "approved" if act == "ok" else "declined", note
        )
        if decided is None:
            await interaction.response.send_message(t("appr_handled", lang), ephemeral=True)
            return
        key = "appr_owner_approved" if act == "ok" else "appr_owner_declined"
        original = getattr(getattr(interaction, "message", None), "content", "") or ""
        await interaction.response.edit_message(
            content=f"{original}\n\n{t(key, lang, name=a.requester_name)}".strip(), view=None
        )
        guild = self._guild(a.guild_id)
        requester = guild.get_member(a.requester_id) if guild else None
        if requester is None:
            return
        rlang = self._approval_user_lang(a.guild_id, a.requester_id)
        key = "appr_req_approved" if act == "ok" else "appr_req_declined"
        text = t(
            key,
            rlang,
            title=a.title,
            who=getattr(interaction.user, "display_name", a.approver_name),
            note=t("appr_note", rlang, note=decided.decision_note) if decided.decision_note else "",
        )
        with contextlib.suppress(discord.HTTPException):
            await requester.send(text)

    # ------------------------------------------------------------- report and briefing data
    def _approval_counts(self, guild_id: int, day: date) -> dict:
        """Numbers for the evening report ({} when the feature is off). Blocking."""
        if not self._approvals_on(guild_id):
            return {}
        return self.approvals.counts(guild_id, day, self.s.zone)

    def _approvals_waiting(self, guild_id: int, user_id: int, now: datetime) -> list[dict]:
        """What the morning brief reminds an approver of: requests pending over a day."""
        if not self._approvals_on(guild_id):
            return []
        rows = self.approvals.stale_pending(guild_id, user_id)
        return [
            {
                "id": a.id,
                "requester": a.requester_name,
                "title": a.title,
                "amount": fmt_amount(a.amount, a.currency) or None,
                "days": max(0, (now - datetime.fromisoformat(a.created_at)).days),
            }
            for a in rows
        ]
