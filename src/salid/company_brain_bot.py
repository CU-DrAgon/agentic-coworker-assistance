"""Discord glue for company learning (logic in `company_brain.py`).

`CompanyMixin` gives `SecretaryBot`: the agent tools (`_company_tools`, appended to `extra_tools`
by `_agent_kwargs`, so voice has them too), the prompt block (`_company_block`), and the DM to the
admins with persistent Approve / Decline buttons (`PlaybookButton`, a DynamicItem like
`ApprovalButton`) for suggestions made by non-admins.
"""

from __future__ import annotations

import asyncio
import logging

import discord

from .company_brain import (
    CompanyCtx,
    CompanyRepo,
    Suggestion,
    company_block,
    company_tools,
    decide,
    nudge_due,
)
from .i18n import t
from .profiles import ProfileTools

logger = logging.getLogger(__name__)

MAX_ADMINS_DM = 5


class PlaybookButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:co:(?P<act>ok|no):(?P<gid>\d{1,25}):(?P<id>[0-9a-f]{8})",
):
    """Approve / Decline on a company-suggestion DM; survives restarts (the id carries the guild
    and the suggestion)."""

    def __init__(self, act: str, gid: int, sid: str, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=discord.ButtonStyle.success if act == "ok" else discord.ButtonStyle.danger,
                custom_id=f"salid:co:{act}:{gid}:{sid}",
            )
        )
        self.act, self.gid, self.sid = act, gid, sid

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], int(match["gid"]), match["id"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_company_suggestion(
            interaction, self.act, self.gid, self.sid
        )


class CompanyMixin:
    def _company_init(self) -> None:
        self.company = CompanyRepo(self.docs, clock=lambda: self.s.now())

    def _company_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "company_learning")

    # ------------------------------------------------------------- prompt block
    def _company_block(
        self, message, guild, asker, is_exec: bool, question: str, interjection: bool
    ) -> str:
        """The company playbook block for one request ("" when off, empty or on any error).
        Also decides the once-a-week onboarding nudge for an admin."""
        try:
            gid = guild.id
            if not self._company_on(gid):
                return ""
            pb = self.company.get(gid)
            nudge = (
                is_exec
                and not interjection
                and getattr(message, "room", None) is None
                and nudge_due(pb, self.s.now())
            )
            if nudge:
                pb.nudged_at = self.s.now().isoformat(timespec="seconds")
                self.company.save(gid, pb)
            return company_block(pb, question, nudge=nudge)
        except Exception:
            logger.warning("company block unavailable", exc_info=True)
            return ""

    # ------------------------------------------------------------- agent tools
    def _company_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        try:
            if not self._company_on(guild.id):
                return []
            loop = asyncio.get_running_loop()
            profile_tools = ProfileTools(self.profiles, guild.id, asker.id)
            ctx = CompanyCtx(
                guild_id=guild.id,
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                repo=self.company,
                lang=lang,
                link=str(getattr(message, "jump_url", "") or ""),
                notify=lambda s: self._notify_admins(guild, s, loop),
                add_style=lambda text: profile_tools.remember("style_add", text),
            )
            return company_tools(ctx)
        except Exception:
            logger.warning("company tools unavailable", exc_info=True)
            return []

    def _notify_admins(self, guild, s: Suggestion, loop) -> int:
        """Tool body (the agent's worker thread): DM the admins. Returns how many were reached."""
        try:
            return asyncio.run_coroutine_threadsafe(
                self._send_suggestion_dms(guild, s), loop
            ).result(timeout=60)
        except Exception:
            logger.warning("company suggestion DM failed", exc_info=True)
            return 0

    def _company_admins(self, guild, exclude: int) -> list:
        ranks = self._ranks(guild)
        admins = [m for m in self._members(guild) if m.id != exclude and self.is_exec_member(m)]
        admins.sort(key=lambda m: -ranks.get(m.id, 0))
        return admins[:MAX_ADMINS_DM]

    async def _send_suggestion_dms(self, guild, s: Suggestion) -> int:
        sent = 0
        for admin in self._company_admins(guild, s.by_user):
            lang = self._approval_user_lang(guild.id, admin.id)
            view = discord.ui.View(timeout=None)
            view.add_item(PlaybookButton("ok", guild.id, s.id, t("co_btn_ok", lang)))
            view.add_item(PlaybookButton("no", guild.id, s.id, t("co_btn_no", lang)))
            text = t(
                "co_suggest_dm",
                lang,
                name=s.by_name or str(s.by_user),
                kind=t(f"co_kind_{s.kind}", lang),
                text=s.text,
                detail=f"\n→ {s.detail}" if s.detail else "",
                link=f"\n{s.source_message_link}" if s.source_message_link else "",
            )
            if (await self.deliver_to_person(guild, admin, text, view=view)).ok:
                sent += 1
        return sent

    # ------------------------------------------------------------- the admin's click
    async def handle_company_suggestion(self, interaction, act: str, gid: int, sid: str) -> None:
        guild = self._guild(gid)
        user = interaction.user
        member = guild.get_member(user.id) if guild else None
        lang = self._approval_user_lang(gid, user.id)
        if member is None or not self.is_exec_member(member):
            await interaction.response.send_message(t("co_not_admin", lang), ephemeral=True)
            return
        done = await asyncio.to_thread(
            decide, self.company, gid, sid, act == "ok", user.id, getattr(user, "display_name", "")
        )
        if done is None:
            await interaction.response.send_message(t("co_handled", lang), ephemeral=True)
            return
        key = "co_approved" if act == "ok" else "co_declined"
        original = getattr(getattr(interaction, "message", None), "content", "") or ""
        await interaction.response.edit_message(
            content=f"{original}\n\n{t(key, lang)}".strip(), view=None
        )
        suggester = guild.get_member(done.by_user)
        if suggester is not None:
            slang = self._approval_user_lang(gid, done.by_user)
            await self.deliver_to_person(
                guild,
                suggester,
                t(
                    "co_result_dm_ok" if act == "ok" else "co_result_dm_no",
                    slang,
                    text=done.text,
                    who=getattr(user, "display_name", ""),
                ),
            )
