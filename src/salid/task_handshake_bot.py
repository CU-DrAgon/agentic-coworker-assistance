"""Task handshake, Discord side (pure texts and decisions in `task_handshake.py`).

`TaskHandshakeMixin` gives `SecretaryBot`: the card sent to the assignee of a task Salid heard
(`_handshake_offer`), the answers (`handle_task_answer`, the edit modal), the line to the person
who gave the task, the 24 hour nudge (`_handshake_due`) and the thread-safe hooks that the summary
worker and the agent tools call. `TaskButton` is a DynamicItem, so the buttons survive restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

import discord

from . import task_handshake as H
from .hints import hint as hint_text
from .hints import with_hint
from .tasks import Task

logger = logging.getLogger(__name__)

NUDGE_CHECK_S = 600.0


class TaskButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:task:(?P<act>ok|edit|no|ap|ae|sb|om|oo|on):(?P<gid>\d{1,25}):(?P<id>[0-9A-Za-z_-]{4,40})",
):
    """Accept / Edit / Not mine on a task card. The custom id carries the guild and the task id."""

    def __init__(
        self, act: str, guild_id: int, task_id: str, label: str | None = None, row=None
    ) -> None:
        style = {
            "ok": discord.ButtonStyle.success,
            "edit": discord.ButtonStyle.primary,
            "no": discord.ButtonStyle.danger,
            "ap": discord.ButtonStyle.success,
            "ae": discord.ButtonStyle.primary,
            "sb": discord.ButtonStyle.secondary,
            "om": discord.ButtonStyle.success,
            "oo": discord.ButtonStyle.primary,
            "on": discord.ButtonStyle.danger,
        }[act]
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=style,
                custom_id=f"salid:task:{act}:{guild_id}:{task_id}",
            ),
            row=row,
        )
        self.act, self.guild_id, self.task_id = act, guild_id, task_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], int(match["gid"]), match["id"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        bot = interaction.client
        if self.act == "oo":
            await bot.handle_owner_modal_open(interaction, self.guild_id, self.task_id)
        elif self.act in ("om", "on"):
            await bot.handle_owner_answer(interaction, self.act, self.guild_id, self.task_id)
        elif self.act in ("edit", "ae", "sb"):
            await bot.handle_task_modal_open(interaction, self.act, self.guild_id, self.task_id)
        elif self.act == "ap":
            await bot.handle_task_approve(interaction, self.guild_id, self.task_id)
        else:
            await bot.handle_task_answer(interaction, self.act, self.guild_id, self.task_id)


class TaskEditModal(discord.ui.Modal):
    def __init__(self, guild_id: int, task: Task, lang: str, boss: bool = False) -> None:
        super().__init__(title=H.t("hs_modal_boss" if boss else "hs_modal_title", lang)[:45])
        self.guild_id, self.task_id, self.boss = guild_id, task.id, boss
        self.f_title = discord.ui.TextInput(
            label=H.t("hs_f_title", lang)[:45], default=task.title, max_length=300
        )
        self.f_due = discord.ui.TextInput(
            label=H.t("hs_f_due", lang)[:45],
            default=task.due,
            required=False,
            max_length=40,
            placeholder=hint_text("ph_due", lang)[:100],
        )
        self.f_note = discord.ui.TextInput(
            label=H.t("hs_f_note", lang)[:45],
            default=task.note,
            required=False,
            max_length=500,
            style=discord.TextStyle.paragraph,
        )
        for f in (self.f_title, self.f_due, self.f_note):
            self.add_item(f)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        handler = (
            interaction.client.handle_task_boss_edit_submit
            if self.boss
            else interaction.client.handle_task_edit_submit
        )
        await handler(
            interaction,
            self.guild_id,
            self.task_id,
            str(self.f_title.value),
            str(self.f_due.value),
            str(self.f_note.value),
        )


class TaskBackModal(discord.ui.Modal):
    """The boss's reason for sending a task back."""

    def __init__(self, guild_id: int, task: Task, lang: str) -> None:
        super().__init__(title=H.t("hs_modal_back", lang)[:45])
        self.guild_id, self.task_id = guild_id, task.id
        self.f_note = discord.ui.TextInput(
            label=H.t("hs_f_back", lang)[:45],
            max_length=300,
            style=discord.TextStyle.paragraph,
        )
        self.add_item(self.f_note)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_task_sendback_submit(
            interaction, self.guild_id, self.task_id, str(self.f_note.value)
        )


class OwnerModal(discord.ui.Modal):
    """Who the task goes to: a name or an @mention (several names make one task each)."""

    def __init__(self, guild_id: int, task: Task, lang: str) -> None:
        super().__init__(title=H.t("hs_owner_modal", lang)[:45])
        self.guild_id, self.task_id = guild_id, task.id
        self.f_who = discord.ui.TextInput(
            label=H.t("hs_owner_field", lang)[:45],
            max_length=100,
            placeholder=hint_text("ph_who", lang)[:100],
        )
        self.add_item(self.f_who)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_owner_answer(
            interaction, "oo", self.guild_id, self.task_id, str(self.f_who.value)
        )


def parse_ref(ref: str) -> tuple[str, str, int]:
    """("channel_id", "message_id", index) from a task's `handshake_message`."""
    bits = (ref or "").split(":")
    if len(bits) < 2:
        return "", "", 0
    return bits[0], bits[1], int(bits[2]) if len(bits) > 2 and bits[2].isdigit() else 0


class TaskHandshakeMixin:
    def _handshake_init(self) -> None:
        self._handshake_last = float("-inf")

    def _handshake_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "task_handshake")

    # ------------------------------------------------------------- hooks (any thread)
    def _handshake_schedule(self, coro) -> None:
        """Run `coro` on the bot loop from a worker thread; never raises."""
        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro))
        except Exception:
            coro.close()
            logger.warning("task handshake could not be scheduled", exc_info=True)

    def _handshake_new_tasks(self, guild_id: int, tasks: list[Task]) -> None:
        """Called by the summary worker with the tasks it created."""
        if tasks:
            self._handshake_schedule(self._handshake_offer(guild_id, tasks))

    # ------------------------------------------------------------- names
    def _handshake_names(self, guild):
        gid = guild.id

        def channel_of(task: Task) -> str:
            return self._channel_name(gid, task.source_channel_id) if task.source_channel_id else ""

        def project_of(task: Task) -> str:
            if not task.project_id:
                return ""
            try:
                p = self.project_repo.get(gid, task.project_id)
            except Exception:
                return ""
            return p.name if p else ""

        return channel_of, project_of

    def _handshake_text(
        self, guild, tasks: list[Task], lang: str, edited=frozenset(), first_time=False
    ) -> str:
        channel_of, project_of = self._handshake_names(guild)
        return H.render_message(
            tasks,
            lang,
            channel_of=channel_of,
            project_of=project_of,
            edited=edited,
            first_time=first_time,
        )

    def _first_card(self, guild_id: int, member_id: int, batch_ids=frozenset()) -> bool:
        """Blocking. True when no task of this person, outside `batch_ids` (the cards being sent
        now, so a nudge of the first card is still a first card), has been through the handshake."""
        return not any(
            x.confirm and x.id not in batch_ids
            for x in self.tasks.list(guild_id, status=None, assignee_id=member_id)
        )

    @staticmethod
    def _handshake_view(guild_id: int, tasks: list[Task], lang: str):
        """Buttons for the tasks still waiting (one row of three each); None when none wait."""
        waiting = [(i, x) for i, x in enumerate(tasks) if x.confirm not in H.ANSWERED]
        if not waiting:
            return None
        view = discord.ui.View(timeout=None)
        many = len(tasks) > 1
        for i, x in waiting:
            pre = f"{i + 1}. " if many else ""
            for act in ("ok", "edit", "no"):
                label = (pre + H.t(f"hs_btn_{act}", lang))[:80]
                view.add_item(TaskButton(act, guild_id, x.id, label, row=i))
        return view

    # ------------------------------------------------------------- sending the cards
    async def _handshake_offer(self, guild_id: int, tasks: list[Task]) -> int:
        """Send each assignee one message with a card per new task (max 5 per message). Tasks
        that name a team nobody could pick: ask the giver. Returns messages sent."""
        guild = self._guild(guild_id)
        if guild is None:
            return 0
        await asyncio.to_thread(self.team.current, guild_id)
        if not self._handshake_on(guild_id):
            return 0
        sent = 0
        by_person: dict[int, list[Task]] = {}
        ownerless: list[Task] = []
        for task in tasks:
            if H.offerable(task):
                by_person.setdefault(task.assignee_id, []).append(task)
            elif getattr(task, "_needs_owner", False):
                await self._handshake_ask_owner(guild, task)
            elif H.ownerless(task):
                ownerless.append(task)
        if ownerless:
            sent += await self._owner_ask(guild, ownerless)
        for uid, items in by_person.items():
            member = guild.get_member(uid)
            if member is None or getattr(member, "bot", False):
                continue
            for batch in H.batches(items):
                try:
                    sent += int(await self._handshake_send(guild, member, batch))
                except Exception:
                    logger.warning("task card not sent", exc_info=True)
        return sent

    async def _handshake_ask_owner(self, guild, task: Task) -> None:
        giver = guild.get_member(task.assigner_id) if task.assigner_id else None
        if giver is None or getattr(giver, "bot", False):
            return
        lang = self._person_lang(guild.id, giver.id)
        await self.deliver_to_person(
            guild,
            giver,
            with_hint(
                H.t("hs_team_unclear", lang, title=task.title, id=task.id),
                "reassign",
                "mention",
                lang=lang,
            ),
        )

    async def _handshake_send(self, guild, member, batch: list[Task], *, nudge: bool = False):
        """One message, one card per task. On success the tasks become `pending` (a nudge keeps
        the original time) and remember the message. Returns whether it was delivered."""
        lang = self._person_lang(guild.id, member.id)
        first = await asyncio.to_thread(
            self._first_card, guild.id, member.id, frozenset(x.id for x in batch)
        )
        text = self._handshake_text(guild, batch, lang, first_time=first)
        res = await self.deliver_to_person(
            guild, member, text, view=self._handshake_view(guild.id, batch, lang)
        )
        if not res.ok:
            return False
        msg = res.message
        ch = getattr(getattr(msg, "channel", None), "id", 0)
        now = self.s.now().isoformat(timespec="seconds")

        def stamp() -> None:
            for i, task in enumerate(batch):
                fresh = self.tasks.get(guild.id, task.id) or task
                if fresh.confirm in H.ANSWERED:
                    continue
                if not nudge:
                    fresh.confirm, fresh.confirm_at = H.PENDING, now
                fresh.handshake_message = f"{ch}:{getattr(msg, 'id', 0)}:{i}"
                self.tasks.save(fresh)

        await asyncio.to_thread(stamp)
        return True

    # ------------------------------------------------------------- answers
    def _handshake_is_exec(self, guild, user) -> bool:
        member = guild.get_member(user.id) if guild else None
        return member is not None and self.is_exec_member(member)

    async def _handshake_check(self, interaction, guild_id: int, task_id: str, *, boss=False):
        """(guild, task, lang) when the press may go on, else None (an ephemeral reply is sent).
        `boss`: a press on the giver's approval card instead of the assignee's card."""
        guild = self._guild(guild_id)
        task = await asyncio.to_thread(self.tasks.get, guild_id, task_id)
        lang = self._person_lang(guild_id, interaction.user.id)
        if guild is None or task is None:
            await interaction.response.send_message(H.t("hs_gone", lang), ephemeral=True)
            return None
        who = task.approver_id if boss else task.assignee_id
        if interaction.user.id != who and not self._handshake_is_exec(guild, interaction.user):
            await interaction.response.send_message(H.t("hs_not_yours", lang), ephemeral=True)
            return None
        states = (H.AWAITING,) if boss else ("", H.PENDING)
        if task.confirm not in states:
            await interaction.response.send_message(H.t("hs_handled", lang), ephemeral=True)
            return None
        return guild, task, lang

    def _now_iso(self) -> str:
        return self.s.now().isoformat(timespec="seconds")

    async def handle_task_answer(self, interaction, act: str, guild_id: int, task_id: str) -> None:
        got = await self._handshake_check(interaction, guild_id, task_id)
        if got is None:
            return
        guild, task, lang = got
        assignee_id = task.assignee_id
        if act == "no":
            now = self._now_iso()

            def decline() -> Task:
                task.confirm, task.confirm_at, task.assignee_id = H.DECLINED, now, 0
                return self.tasks.save(task)  # the written name stays

            task = await asyncio.to_thread(decline)
            await self._handshake_finish(interaction, guild, task, assignee_id, "no", [], lang)
            return
        await self._handshake_accept(interaction, guild, task, assignee_id, "ok", [], lang)

    async def handle_task_modal_open(self, interaction, act: str, guild_id: int, task_id: str):
        got = await self._handshake_check(interaction, guild_id, task_id, boss=act != "edit")
        if got is None:
            return
        _, task, lang = got
        if act == "sb":
            await interaction.response.send_modal(TaskBackModal(guild_id, task, lang))
        else:
            await interaction.response.send_modal(
                TaskEditModal(guild_id, task, lang, boss=act == "ae")
            )

    async def handle_task_edit_submit(
        self, interaction, guild_id: int, task_id: str, title: str, due_text: str, note: str
    ) -> None:
        got = await self._handshake_check(interaction, guild_id, task_id)
        if got is None:
            return
        guild, task, lang = got
        fields, bad_due = H.edited_fields(task, title, due_text, note, self.s.now().date())
        changes = H.describe_changes(task, fields["title"], fields["due"], fields["note"], lang)
        await self._handshake_accept(
            interaction,
            guild,
            task,
            task.assignee_id,
            "edit" if changes else "ok",
            changes,
            lang,
            fields=fields,
            bad_due=bad_due,
        )

    # ------------------------------------------------------------- accepted: notify or approve
    def _handshake_mode(self, guild_id: int, task: Task) -> str:
        """notify | approve for this task: the giver's own mode (when the organisation allows
        personal settings and they set one), else the organisation's."""
        giver = (
            self._profile_of(guild_id, task.assigner_id).task_confirm_mode
            if task.assigner_id > 0
            else ""
        )
        return H.effective_mode(self.team.task_confirm_mode(guild_id), giver, True)

    def _handshake_boss(self, guild, task: Task, assignee_id: int):
        """The member who signs off a task in approve mode: its giver, else the owner of its
        project, else the first executive with a private Salid channel. None = no approval
        (notify): the giver is the assignee, or nobody can be reached."""
        if self._handshake_mode(guild.id, task) != "approve":
            return None
        if task.assigner_id > 0 and task.assigner_id == assignee_id:
            return None  # a note to oneself needs no sign-off

        def usable(uid: int):
            m = guild.get_member(uid) if uid > 0 else None
            return m if m is not None and not getattr(m, "bot", False) else None

        if task.assigner_id > 0:
            boss = usable(task.assigner_id)
            if boss is not None:
                return boss
        if task.project_id:
            try:
                p = self.project_repo.get(guild.id, task.project_id)
            except Exception:
                p = None
            raw = getattr(getattr(p, "owner", None), "discord_user_id", "") or ""
            if raw.isdigit() and int(raw) != assignee_id and usable(int(raw)) is not None:
                return usable(int(raw))
        for m in sorted(self._members(guild), key=lambda x: x.id):
            if (
                m.id != assignee_id
                and self.is_exec_member(m)
                and self.private_chats.get(guild.id, m.id)
            ):
                return m
        logger.info("task %s: approve mode but nobody to approve; treated as notify", task.id)
        return None

    async def _handshake_accept(
        self,
        interaction,
        guild,
        task,
        assignee_id,
        kind,
        changes,
        lang,
        *,
        fields=None,
        bad_due=False,
    ) -> None:
        """The assignee accepted (maybe with edits): notify mode records it and tells the giver;
        approve mode hands the task to the boss first."""
        now = self._now_iso()
        boss = await asyncio.to_thread(self._handshake_boss, guild, task, assignee_id)
        old = {"title": task.title, "due": task.due, "note": task.note}

        def apply(state: str) -> Task:
            if fields:
                task.title, task.due, task.note = fields["title"], fields["due"], fields["note"]
            task.confirm, task.confirm_at = state, now
            if state == H.AWAITING:
                task.approver_id = boss.id
                task.edit_from = json.dumps(old) if changes else ""
            return self.tasks.save(task)

        task = await asyncio.to_thread(apply, H.AWAITING if boss is not None else H.ACCEPTED)
        if boss is not None and not await self._handshake_send_boss(guild, task, boss):
            boss = None  # the boss cannot be reached: record it like notify mode
            task = await asyncio.to_thread(apply, H.ACCEPTED)
        await self._handshake_edit_card(interaction, guild, task, lang, edited=kind == "edit")
        if bad_due:
            with contextlib.suppress(Exception):
                await interaction.followup.send(H.t("hs_bad_due", lang), ephemeral=True)
        if boss is None:
            await self._handshake_tell_giver_safe(guild, task, assignee_id, kind, changes)

    async def _handshake_send_boss(self, guild, task: Task, boss) -> bool:
        lang = self._person_lang(guild.id, boss.id)
        who = guild.get_member(task.assignee_id)
        name = getattr(who, "display_name", "") or task.assignee_name or str(task.assignee_id)
        view = discord.ui.View(timeout=None)
        for act in ("ap", "ae", "sb"):
            view.add_item(TaskButton(act, guild.id, task.id, H.t(f"hs_btn_{act}", lang), row=0))
        res = await self.deliver_to_person(
            guild, boss, H.boss_card_text(task, name, lang), view=view
        )
        if not res.ok:
            return False
        msg = res.message
        ref = f"{getattr(getattr(msg, 'channel', None), 'id', 0)}:{getattr(msg, 'id', 0)}"

        def stamp() -> None:
            fresh = self.tasks.get(guild.id, task.id) or task
            fresh.approval_message = ref
            self.tasks.save(fresh)

        await asyncio.to_thread(stamp)
        return True

    # ------------------------------------------------------------- the boss answers
    async def _handshake_edit_boss_card(self, interaction, guild, task, lang, status_key) -> None:
        who = guild.get_member(task.assignee_id)
        name = getattr(who, "display_name", "") or task.assignee_name
        text = H.boss_card_text(task, name, lang, hint=False) + "\n" + H.t(status_key, lang)
        await self._handshake_edit_message(interaction, text, None, text)

    async def _handshake_edit_message(self, interaction, text, view, fallback) -> None:
        try:
            if getattr(interaction, "message", None) is not None:
                await interaction.response.edit_message(content=text, view=view)
            else:
                await interaction.response.send_message(fallback, ephemeral=True)
        except discord.HTTPException:
            logger.warning("could not edit the task card", exc_info=True)

    async def _handshake_tell_assignee(self, guild, task: Task, key: str, **fmt) -> None:
        member = guild.get_member(task.assignee_id) if task.assignee_id else None
        if member is None or getattr(member, "bot", False):
            return
        lang = self._person_lang(guild.id, member.id)
        text = H.t(key, lang, title=task.title, due=task.due or "-", id=task.id, **fmt)
        await self.deliver_to_person(guild, member, text)

    async def handle_task_approve(self, interaction, guild_id: int, task_id: str) -> None:
        got = await self._handshake_check(interaction, guild_id, task_id, boss=True)
        if got is None:
            return
        guild, task, lang = got
        now = self._now_iso()

        def apply() -> Task:
            task.confirm, task.confirm_at = H.ACCEPTED, now
            return self.tasks.save(task)

        task = await asyncio.to_thread(apply)
        await self._handshake_edit_boss_card(interaction, guild, task, lang, "hs_boss_ok")
        with contextlib.suppress(Exception):
            await self._handshake_tell_assignee(guild, task, "hs_to_assignee_ok")

    async def handle_task_boss_edit_submit(
        self, interaction, guild_id: int, task_id: str, title: str, due_text: str, note: str
    ) -> None:
        """The boss's version is final: it replaces the assignee's and counts as approved."""
        got = await self._handshake_check(interaction, guild_id, task_id, boss=True)
        if got is None:
            return
        guild, task, lang = got
        fields, bad_due = H.edited_fields(task, title, due_text, note, self.s.now().date())
        who = guild.get_member(task.assignee_id)
        a_lang = self._person_lang(guild_id, task.assignee_id) if who is not None else lang
        changes = H.describe_changes(task, fields["title"], fields["due"], fields["note"], a_lang)
        now = self._now_iso()

        def apply() -> Task:
            task.title, task.due, task.note = fields["title"], fields["due"], fields["note"]
            task.confirm, task.confirm_at = H.ACCEPTED, now
            return self.tasks.save(task)

        task = await asyncio.to_thread(apply)
        await self._handshake_edit_boss_card(
            interaction, guild, task, lang, "hs_boss_edited" if changes else "hs_boss_ok"
        )
        if bad_due:
            with contextlib.suppress(Exception):
                await interaction.followup.send(H.t("hs_bad_due", lang), ephemeral=True)
        with contextlib.suppress(Exception):
            if changes:
                await self._handshake_tell_assignee(
                    guild, task, "hs_to_assignee_edit", changes=", ".join(changes)
                )
            else:
                await self._handshake_tell_assignee(guild, task, "hs_to_assignee_ok")

    async def handle_task_sendback_submit(
        self, interaction, guild_id: int, task_id: str, note: str
    ) -> None:
        """Back to the assignee with the boss's note (at most MAX_ROUNDS times)."""
        got = await self._handshake_check(interaction, guild_id, task_id, boss=True)
        if got is None:
            return
        guild, task, lang = got
        now = self._now_iso()
        capped = task.approval_rounds >= H.MAX_ROUNDS

        def apply() -> Task:
            task.confirm, task.confirm_at = H.PENDING, now
            if not capped:
                task.approval_rounds += 1
                task.send_back_note = (note or "").strip()[:300]
            return self.tasks.save(task)

        task = await asyncio.to_thread(apply)
        await self._handshake_edit_boss_card(interaction, guild, task, lang, "hs_boss_back")
        if capped:
            with contextlib.suppress(Exception):
                await interaction.followup.send(
                    H.t("hs_back_cap", lang, n=H.MAX_ROUNDS), ephemeral=True
                )
            await self._handshake_tell_exec_stuck(guild, task, interaction.user.id)
            return
        member = guild.get_member(task.assignee_id)
        if member is not None and not getattr(member, "bot", False):
            await self._handshake_send(guild, member, [task])

    async def _handshake_tell_exec_stuck(self, guild, task: Task, skip_id: int) -> None:
        """Once per task: an executive hears that the task keeps coming back."""
        if not await asyncio.to_thread(self._claim, guild.id, f"handshake-cap:{task.id}"):
            return
        for m in sorted(self._members(guild), key=lambda x: x.id):
            if m.id != skip_id and self.is_exec_member(m):
                lang = self._person_lang(guild.id, m.id)
                text = H.t("hs_back_cap_exec", lang, title=task.title, id=task.id, n=H.MAX_ROUNDS)
                if (await self.deliver_to_person(guild, m, text)).ok:
                    return

    # ------------------------------------------------------------- notify: finish
    async def _handshake_finish(
        self, interaction, guild, task: Task, assignee_id, kind, changes, lang, *, bad_due=False
    ) -> None:
        """Edit the card into its final state and tell the giver."""
        await self._handshake_edit_card(interaction, guild, task, lang, edited=kind == "edit")
        await self._handshake_tell_giver_safe(guild, task, assignee_id, kind, changes)

    async def _handshake_tell_giver_safe(self, guild, task, assignee_id, kind, changes) -> None:
        try:
            await self._handshake_tell_giver(guild, task, assignee_id, kind, changes)
        except Exception:
            logger.warning("could not tell the giver of task %s", task.id, exc_info=True)

    async def _handshake_edit_card(self, interaction, guild, task: Task, lang, *, edited) -> None:
        ref = task.handshake_message
        if ref:
            key = ref.rsplit(":", 1)[0]
            rows = await asyncio.to_thread(
                lambda: [
                    x
                    for x in self.tasks.list(guild.id, status=None)
                    if x.handshake_message.rsplit(":", 1)[0] == key
                ]
            )
            rows.sort(key=lambda x: parse_ref(x.handshake_message)[2])
        else:
            rows = [task]
        if not any(x.id == task.id for x in rows):
            rows.append(task)
        edited_ids = frozenset({task.id}) if edited else frozenset()
        text = self._handshake_text(guild, rows, lang, edited_ids)
        view = self._handshake_view(guild.id, rows, lang)
        await self._handshake_edit_message(
            interaction, text, view, H.status_line(task, lang, edited)
        )

    async def _handshake_tell_giver(
        self, guild, task: Task, assignee_id: int, kind: str, changes: list[str]
    ) -> None:
        giver_id = task.assigner_id
        if not giver_id or giver_id == assignee_id:
            return
        giver = guild.get_member(giver_id)
        if giver is None or getattr(giver, "bot", False):
            return
        who = guild.get_member(assignee_id)
        name = getattr(who, "display_name", "") or task.assignee_name or str(assignee_id)
        group = None
        if task.group_id:
            peers = await asyncio.to_thread(
                lambda: [
                    x for x in self.tasks.list(guild.id, status=None) if x.group_id == task.group_id
                ]
            )
            group = (sum(1 for x in peers if x.confirm == H.ACCEPTED), len(peers))
        lang = self._person_lang(guild.id, giver_id)
        await self.deliver_to_person(
            guild, giver, H.giver_message(kind, task, name, lang, changes, group)
        )

    # ------------------------------------------------------------- typed commands
    async def _task_mode_command(self, message, guild=None, member=None) -> bool:
        """ "โหมดมอบหมายงานแบบอนุมัติ" / "task mode notify" (admins and executives: the server) and
        "my task mode approve" / "งานที่ฉันมอบหมาย ขออนุมัติก่อน" (anyone: their own, unless the
        organisation turned personal settings off)."""
        cmd = H.parse_task_mode_command(self._command_text(message))
        if cmd is None:
            return False
        mode, personal = cmd
        guild = guild or message.guild
        user = member or message.author
        if guild is None:
            return False
        gid = guild.id
        lang = self._person_lang(gid, user.id, message.content or "")
        await asyncio.to_thread(self.team.current, gid)
        if personal:
            if not self._personal_allowed(gid):
                await message.reply(H.t("personal_off", lang), mention_author=False)
                return True
            from .profiles import ProfileTools

            ProfileTools(self.profiles, gid, user.id).remember("task_confirm_mode", mode)
            await message.reply(H.t(f"hs_mode_me_{mode}", lang), mention_author=False)
            return True
        if member is not None:  # in a private channel / DM the server form is not for them
            return False
        if not self._may_configure(user):
            await message.reply(H.t("hs_mode_denied", lang), mention_author=False)
            return True
        await asyncio.to_thread(self.team.set_task_confirm_mode, gid, mode)
        await message.reply(H.t(f"hs_mode_set_{mode}", lang), mention_author=False)
        return True

    # ------------------------------------------------------------- who takes this task?
    def _owner_names_of(self, guild):
        def names(task: Task) -> str:
            if not task.group_id:
                return task.assignee_name
            peers = [
                x for x in self.tasks.list(guild.id, status=None) if x.group_id == task.group_id
            ]
            return ", ".join(dict.fromkeys(x.assignee_name for x in peers if x.assignee_name))

        return names

    def _owner_text(self, guild, rows: list[Task], lang: str) -> str:
        channel_of, _ = self._handshake_names(guild)
        return H.render_owner_message(
            rows, lang, channel_of=channel_of, names_of=self._owner_names_of(guild)
        )

    @staticmethod
    def _owner_view(guild_id: int, rows: list[Task], lang: str):
        waiting = [(i, x) for i, x in enumerate(rows) if H.owner_waiting(x)]
        if not waiting:
            return None
        view = discord.ui.View(timeout=None)
        many = len(rows) > 1
        for i, x in waiting:
            pre = f"{i + 1}. " if many else ""
            for act in ("om", "oo", "on"):
                label = (pre + H.t(f"hs_btn_{act}", lang))[:80]
                view.add_item(TaskButton(act, guild_id, x.id, label, row=i))
        return view

    async def _owner_ask(self, guild, task_list: list[Task]) -> int:
        """Ask each giver, in ONE message per 5 tasks, who the unassigned tasks go to."""
        by_giver: dict[int, list[Task]] = {}
        for task in task_list:
            by_giver.setdefault(task.assigner_id, []).append(task)
        sent = 0
        for uid, items in by_giver.items():
            member = guild.get_member(uid)
            if member is None or getattr(member, "bot", False):
                continue
            for batch in H.batches(items):
                try:
                    sent += int(await self._owner_send(guild, member, batch))
                except Exception:
                    logger.warning("owner question not sent", exc_info=True)
        return sent

    async def _owner_send(self, guild, member, batch: list[Task], *, nudge: bool = False):
        lang = self._person_lang(guild.id, member.id)
        res = await self.deliver_to_person(
            guild,
            member,
            self._owner_text(guild, batch, lang),
            view=self._owner_view(guild.id, batch, lang),
        )
        if not res.ok:
            return False
        msg = res.message
        ch = getattr(getattr(msg, "channel", None), "id", 0)
        now = self._now_iso()

        def stamp() -> None:
            for i, task in enumerate(batch):
                fresh = self.tasks.get(guild.id, task.id) or task
                if fresh.confirm not in ("", H.OWNER_ASKED):
                    continue
                if not nudge:
                    fresh.confirm, fresh.confirm_at = H.OWNER_ASKED, now
                fresh.owner_message = f"{ch}:{getattr(msg, 'id', 0)}:{i}"
                self.tasks.save(fresh)

        await asyncio.to_thread(stamp)
        return True

    def _owner_rows(self, guild, task: Task) -> list[Task]:
        """The tasks shown on the same owner card as `task`, in card order."""
        ref = task.owner_message
        if not ref:
            return [task]
        key = ref.rsplit(":", 1)[0]
        rows = [
            x
            for x in self.tasks.list(guild.id, status=None)
            if x.owner_message.rsplit(":", 1)[0] == key
        ]
        rows.sort(key=lambda x: parse_ref(x.owner_message)[2])
        if not any(x.id == task.id for x in rows):
            rows.append(task)
        return rows

    async def _owner_check(self, interaction, guild_id: int, task_id: str):
        guild = self._guild(guild_id)
        task = await asyncio.to_thread(self.tasks.get, guild_id, task_id)
        lang = self._person_lang(guild_id, interaction.user.id)
        if guild is None or task is None:
            await interaction.response.send_message(H.t("hs_gone", lang), ephemeral=True)
            return None
        if interaction.user.id != task.assigner_id and not self._handshake_is_exec(
            guild, interaction.user
        ):
            await interaction.response.send_message(H.t("hs_owner_not_giver", lang), ephemeral=True)
            return None
        if task.confirm != H.OWNER_ASKED:
            await interaction.response.send_message(H.t("hs_handled", lang), ephemeral=True)
            return None
        return guild, task, lang

    async def handle_owner_modal_open(self, interaction, guild_id: int, task_id: str) -> None:
        got = await self._owner_check(interaction, guild_id, task_id)
        if got is None:
            return
        _, task, lang = got
        await interaction.response.send_modal(OwnerModal(guild_id, task, lang))

    async def handle_owner_answer(
        self, interaction, act: str, guild_id: int, task_id: str, name: str = ""
    ) -> None:
        got = await self._owner_check(interaction, guild_id, task_id)
        if got is None:
            return
        guild, task, lang = got
        kind = {"om": "mine", "on": "none", "oo": "other"}[act]
        res = await self._owner_resolve(guild, task, kind, name)
        if res.unresolved:
            await interaction.response.send_message(
                with_hint(H.t("hs_owner_unknown", lang, name=res.unresolved), "mention", lang=lang),
                ephemeral=True,
            )
            return
        rows = await asyncio.to_thread(self._owner_rows, guild, res.tasks[0])
        await self._handshake_edit_message(
            interaction,
            self._owner_text(guild, rows, lang),
            self._owner_view(guild.id, rows, lang),
            H.owner_status(res.tasks[0], lang, res.tasks[0].assignee_name),
        )
        await self._owner_followup(guild, res)

    async def _owner_resolve(self, guild, task: Task, kind: str, name: str = ""):
        people = self._people(guild)
        now = self._now_iso()
        return await asyncio.to_thread(
            H.apply_owner_answer, self.tasks, task, kind, name, people, now
        )

    async def _owner_followup(self, guild, res) -> None:
        """After an answer: the people who now own a task get the normal card."""
        if res.offer:
            await self._handshake_offer(guild.id, res.offer)

    async def _owner_edit_by_ref(self, guild, task: Task) -> None:
        """Edit the question card of `task` without an interaction (an answer given in words or
        through a tool). Best effort."""
        ch_id, msg_id, _ = parse_ref(task.owner_message)
        if not ch_id.isdigit() or not msg_id.isdigit():
            return
        try:
            rows = await asyncio.to_thread(self._owner_rows, guild, task)
            lang = self._person_lang(guild.id, task.assigner_id)
            ch = self.get_channel(int(ch_id)) or await self.fetch_channel(int(ch_id))
            msg = await ch.fetch_message(int(msg_id))
            await msg.edit(
                content=self._owner_text(guild, rows, lang),
                view=self._owner_view(guild.id, rows, lang),
            )
        except Exception:
            logger.warning("could not edit the owner card", exc_info=True)

    def _owner_tools_done(self, guild_id: int, result_tasks: list[Task], offer: list[Task]) -> None:
        """Hook for the answer_task_question tool (any thread): edit the card, send the cards."""

        async def go() -> None:
            guild = self._guild(guild_id)
            if guild is None:
                return
            await self._owner_edit_by_ref(guild, result_tasks[0])
            if offer:
                await self._handshake_offer(guild_id, offer)

        self._handshake_schedule(go())

    async def _owner_answer_command(self, message, guild, member) -> bool:
        """The giver answers the owner question in words (private channel / DM): "ไม่ใช่งาน",
        "ของฉัน", "ให้บี", "not a task". True when the message was such an answer."""
        if not self._handshake_on(guild.id):
            return False
        pending = [
            x
            for x in await asyncio.to_thread(self.tasks.list, guild.id, status="open")
            if x.confirm == H.OWNER_ASKED and x.assigner_id == member.id
        ]
        if not pending:
            return False
        text = self._command_text(message)
        answer = H.parse_owner_answer(text)
        if answer is None:
            return False
        ref = getattr(getattr(message, "reference", None), "message_id", None)
        cands = [x for x in pending if ref and parse_ref(x.owner_message)[1] == str(ref)] or pending
        if len(cands) > 1:
            named = [x for x in cands if x.id.casefold() in (text or "").casefold()]
            cands = named if len(named) == 1 else cands
        lang = self._person_lang(guild.id, member.id, message.content or "")
        if len(cands) > 1:
            lines = [H.t("hs_owner_which", lang)] + [f"#{x.id} {x.title[:60]}" for x in cands[:5]]
            await message.reply(
                with_hint("\n".join(lines), "task_id", lang=lang), mention_author=False
            )
            return True
        task = cands[0]
        kind, name = answer
        res = await self._owner_resolve(guild, task, kind, name)
        if res.unresolved:
            await message.reply(
                with_hint(H.t("hs_owner_unknown", lang, name=res.unresolved), "mention", lang=lang),
                mention_author=False,
            )
            return True
        await self._owner_edit_by_ref(guild, res.tasks[0])
        await message.reply(H.t(f"hs_owner_say_{kind}", lang), mention_author=False)
        await self._owner_followup(guild, res)
        return True

    # ------------------------------------------------------------- reassign / nudge
    def _handshake_reassigned(self, guild_id: int, task: Task) -> None:
        """A tool moved a task to someone else: send the new person the card (any thread)."""
        if task.assignee_id > 0 and not task.confirm:
            self._handshake_schedule(self._handshake_offer(guild_id, [task]))

    async def _handshake_tick(self, now) -> int:
        if self._clock() - self._handshake_last < NUDGE_CHECK_S:
            return 0
        self._handshake_last = self._clock()
        return await self._handshake_due(now)

    async def _handshake_due(self, now) -> int:
        """ONE nudge per task and assignee: the same card again after 24 h without an answer."""
        sent = 0
        for guild in list(self.guilds):
            gid = guild.id
            try:
                await asyncio.to_thread(self.team.current, gid)
                if not self._handshake_on(gid):
                    continue
                tasks = await asyncio.to_thread(self.tasks.list, gid, status="open")
                for task in tasks:
                    if H.owner_nudge_due(task, now):
                        giver = guild.get_member(task.assigner_id)
                        key = f"owner-nudge:{task.id}"
                        if giver is None or getattr(giver, "bot", False):
                            continue
                        if not await asyncio.to_thread(self._claim, gid, key):
                            continue
                        sent += int(bool(await self._owner_send(guild, giver, [task], nudge=True)))
                        continue
                    if not H.nudge_due(task, now):
                        continue
                    boss_side = task.confirm == H.AWAITING
                    target = task.approver_id if boss_side else task.assignee_id
                    member = guild.get_member(target)
                    if member is None or getattr(member, "bot", False):
                        continue
                    key = f"handshake-nudge:{task.id}:{'boss' if boss_side else 'doer'}:{target}"
                    if not await asyncio.to_thread(self._claim, gid, key):
                        continue
                    if boss_side:
                        ok = await self._handshake_send_boss(guild, task, member)
                    else:
                        ok = await self._handshake_send(guild, member, [task], nudge=True)
                    sent += int(bool(ok))
            except Exception:
                logger.exception("task nudges for guild %s failed", gid)
        return sent
