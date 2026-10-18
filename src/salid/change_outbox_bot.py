"""Discord side of the change outbox (logic and store: `change_outbox.py`).

`ChangeOutboxMixin`, every 30 s from the bot tick, for each due target of a Discord workspace this
bot serves:
- `apply`: run the positions sync for that server (the same one the 30 s center watcher runs: it
  also adopts center-edited titles, so it is the whole server, once per tick), ack `applied` when
  it did its work, else keep the error and try again in 15 minutes;
- `notify`: DM the person a short card with a persistent "Got it" button (`chg:<change>:<ws>`).
  The click, or a reply "รับทราบ / ok / got it" in that DM or private channel, acks. A resend is
  the same card with "(reminder N)".
After the last attempt the target fails and the org's owners and admins on Discord get one alert.

Also the safety-net producer: `_change_detect` (called by the positions center watcher) diffs the
org against the last snapshot and records what the web did not (`by="detected"`; `record`
dedupes against the web's own record).
"""

from __future__ import annotations

import asyncio
import logging

import discord

from . import audience as aud
from .change_outbox import (
    ChangeOutbox,
    admins_of,
    alert_text,
    change_status_tools,
    change_text,
    diff_org,
    is_ack_text,
)
from .i18n import base_lang, resolve_lang, t
from .orgs import OrgStore

logger = logging.getLogger(__name__)

TICK_S = 30.0


class ChangeButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"chg:(?P<id>[0-9a-f]{16}):(?P<ws>discord-[0-9]+)",
):
    """ "Got it" on a change notice; survives restarts (registered in `setup_hook`)."""

    def __init__(self, change_id: str, ws_id: str, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or "OK",
                style=discord.ButtonStyle.success,
                custom_id=f"chg:{change_id}:{ws_id}",
            )
        )
        self.change_id, self.ws_id = change_id, ws_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["id"], match["ws"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_change_click(interaction, self.change_id, self.ws_id)


class ChangeOutboxMixin:
    """Needs `self.docs`, `self.s`, `self.team`, `self.guilds`, `_clock`, `deliver_to_person`,
    `positions_sync`, `_positions_on`, `_deliver_langs`, `_monitor_note_queue`. Call
    `_change_outbox_init()` after `_positions_init()`."""

    def _change_outbox_init(self) -> None:
        self.change_outbox = ChangeOutbox.from_settings(self.docs, self.s)
        self._change_polled = float("-inf")
        self._change_snap: dict[int, object] = {}  # guild id -> Org as last seen

    # ------------------------------------------------------------- producer (safety net)
    def _change_load(self, guild):
        try:
            return OrgStore(self.docs).org_of_guild(guild.id)
        except Exception:
            return None

    async def _change_snapshot(self, guild) -> None:
        """Remember the org as it is now (Salid's own writes are not 'center edits')."""
        org = await asyncio.to_thread(self._change_load, guild)
        if org is not None:
            self._change_snap[guild.id] = org

    async def _change_detect(self, guild) -> None:
        """Record what changed in the org since the last snapshot and the web did not record.
        Never raises."""
        try:
            org = await asyncio.to_thread(self._change_load, guild)
            if org is None:
                return
            before = self._change_snap.get(guild.id)
            self._change_snap[guild.id] = org
            if before is None:
                return
            changes = diff_org(before, org, by="detected", at=self.s.now())
            if changes:
                stored = await asyncio.to_thread(self.change_outbox.record, changes, org)
                if stored:
                    logger.info(
                        "change outbox: detected %d change(s) guild=%s", len(stored), guild.id
                    )
        except Exception:
            logger.warning("change outbox: detect failed", exc_info=True)

    # ------------------------------------------------------------- consumer
    def _change_lang(self, gid: int) -> str:
        try:
            return base_lang(resolve_lang(self.team.current(gid).language_setting))
        except Exception:
            return "en"

    async def _change_outbox_due(self) -> None:
        now = self._clock()
        if now - self._change_polled < TICK_S:
            return
        self._change_polled = now
        try:
            await self._change_outbox_run()
        except Exception:
            logger.warning("change outbox: tick failed", exc_info=True)

    async def _change_outbox_run(self) -> None:
        guilds = {f"discord-{g.id}": g for g in self.guilds}
        if not guilds:
            return
        outbox = self.change_outbox
        await asyncio.to_thread(outbox.prune)
        due = await asyncio.to_thread(outbox.due)
        mine = [(c, x) for c, x in due if x.platform == "discord" and x.workspace_id in guilds]
        self._monitor_note_queue("outbox_pending", len(mine))  # the monitor heartbeat's queue
        orgs: dict[str, str] = {}
        synced: dict[str, str] = {}  # workspace id -> "" (ok) or the error

        async def org_name(org_id: str) -> str:
            if org_id not in orgs:
                org = await asyncio.to_thread(OrgStore(self.docs).get, org_id)
                orgs[org_id] = getattr(org, "name", "") or ""
            return orgs[org_id]

        for change, target in mine:
            guild = guilds[target.workspace_id]
            try:
                if target.kind == "apply":
                    await self._change_apply(change, target, guild, synced)
                else:
                    await self._change_notify(change, target, guild, await org_name(change.org_id))
            except Exception as e:
                logger.warning("change outbox: target failed", exc_info=True)
                await asyncio.to_thread(
                    outbox.mark_attempt,
                    change.id,
                    target.workspace_id,
                    target.person,
                    error=type(e).__name__,
                )
        await self._change_alerts(guilds, org_name)

    async def _change_apply(self, change, target, guild, synced: dict[str, str]) -> None:
        outbox, ws = self.change_outbox, target.workspace_id
        if not self._positions_on(guild.id):
            await asyncio.to_thread(outbox.ack, change.id, ws, "", "nothing_to_apply")
            return
        if ws not in synced:
            rep = await self.positions_sync(guild)
            await self._change_snapshot(guild)
            synced[ws] = "" if not rep.skipped_no_permission else "missing Manage Roles"
        if synced[ws]:
            await asyncio.to_thread(outbox.mark_attempt, change.id, ws, "", error=synced[ws])
        else:
            await asyncio.to_thread(outbox.ack, change.id, ws, "", "applied")

    async def _change_notify(self, change, target, guild, org_name: str) -> None:
        outbox = self.change_outbox
        uid = int(target.person) if target.person.isdigit() else 0
        member = guild.get_member(uid) if uid else None
        if member is None:
            await asyncio.to_thread(
                outbox.mark_attempt, change.id, target.workspace_id, target.person,
                error="not in the server",
            )  # fmt: skip
            return
        langs = self._deliver_langs(guild.id, uid)
        text = aud.render(
            lambda lang: change_text(change, org_name, base_lang(lang), target.attempts), langs
        )
        view = discord.ui.View(timeout=None)
        view.add_item(
            ChangeButton(change.id, target.workspace_id, t("chg_btn", base_lang(langs[0])))
        )
        res = await self.deliver_to_person(guild, member, text, view=view)
        ref = str(getattr(getattr(res, "message", None), "id", "") or "")
        await asyncio.to_thread(
            outbox.mark_attempt,
            change.id,
            target.workspace_id,
            target.person,
            error="" if res.ok else "could not message",
            message_ref=ref,
        )

    async def _change_alerts(self, guilds: dict, org_name) -> None:
        outbox = self.change_outbox
        for change in await asyncio.to_thread(outbox.to_alert, "discord"):
            org = await asyncio.to_thread(OrgStore(self.docs).get, change.org_id)
            if org is None:
                continue
            for m in admins_of(org):
                uid = str(m.identity("discord") or "")
                if not uid.isdigit():
                    continue
                for w in org.workspaces:
                    guild = guilds.get(w.id)
                    member = guild.get_member(int(uid)) if guild else None
                    if member is None:
                        continue
                    lang = self._change_lang(guild.id)
                    await self.deliver_to_person(
                        guild, member, alert_text(change, org.name, lang, outbox.max_attempts)
                    )
                    break
            await asyncio.to_thread(outbox.mark_alerted, change.id, "discord")

    # ------------------------------------------------------------- acknowledgement
    async def handle_change_click(self, interaction, change_id: str, ws_id: str) -> None:
        gid = int(ws_id.rsplit("-", 1)[-1]) if ws_id.rsplit("-", 1)[-1].isdigit() else 0
        lang = self._change_lang(gid)
        person = str(interaction.user.id)
        change = await asyncio.to_thread(self.change_outbox.get, change_id)
        mine = change is not None and any(
            x.workspace_id == ws_id and x.kind == "notify" and x.person == person
            for x in change.targets
        )
        if not mine:
            await interaction.response.send_message(t("chg_not_yours", lang), ephemeral=True)
            return
        fresh = await asyncio.to_thread(self.change_outbox.ack, change_id, ws_id, person, "button")
        original = getattr(getattr(interaction, "message", None), "content", "") or ""
        suffix = t("chg_done" if fresh else "chg_handled", lang)
        await interaction.response.edit_message(
            content=f"{original}\n\n{suffix}".strip(), view=None
        )

    async def _change_ack_reply(self, message, guild) -> bool:
        """A reply "รับทราบ / ok / got it" from someone with unconfirmed notices acks them (only the
        notice replied to, when the message is a reply to one). True = handled."""
        if not is_ack_text(getattr(message, "content", "") or ""):
            return False
        ws, person = f"discord-{guild.id}", str(message.author.id)
        open_ = await asyncio.to_thread(self.change_outbox.open_for, ws, person)
        if not open_:
            return False
        ref = getattr(getattr(message, "reference", None), "message_id", None)
        picked = [(c, x) for c, x in open_ if ref and x.message_ref == str(ref)]
        for c, _x in picked or open_:
            await asyncio.to_thread(self.change_outbox.ack, c.id, ws, person, "reply")
        lang = self._change_lang(guild.id)
        try:
            await message.reply(t("chg_thanks", lang), mention_author=False)
        except Exception:
            logger.info("change outbox: could not thank for an ack", exc_info=True)
        return True

    # ------------------------------------------------------------- agent tool
    def _change_tools(self, guild, is_exec: bool) -> list:
        """`change_status` for executives (chat and voice: registered in `_agent_kwargs`)."""
        if not is_exec:
            return []

        def org_id() -> str | None:
            org = OrgStore(self.docs).org_of_guild(guild.id)
            return org.id if org else None

        return change_status_tools(self.change_outbox, org_id)
