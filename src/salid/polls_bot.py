"""Polls and forms, Discord side: a `SecretaryBot` mixin.

Agent tools (`_poll_form_tools`, appended by `SecretaryBot._agent_kwargs` so chat and voice share
them): `create_poll`, `poll_results`, `start_form`, `form_results`. Every scope decision (which
channel, who may be DMed, who may read results) is made here in code, never by the model.

Timers (`_polls_forms_due`, once a minute from `tick`): post the result line of a poll Salid
created when it ends (and keep it in team memory); remind non-responders of a form once; close a
form at its deadline. The DM hook (`_form_dm_answer`, called from `_direct_message`) treats a DM
as a form answer only when that person has an open question and a cheap gate says the reply
looks like an answer; anything else is normal chat.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import UTC, datetime, timedelta
from functools import partial

import discord

from . import forms as F
from . import polls as P
from .dm import SENSITIVE_DM_RE, send_dm
from .i18n import t
from .memory import build_message
from .minutes import can_see, can_see_and_post, resolve_channel
from .pdf_report import find_fonts
from .permissions import channel_missing, permission_message
from .secretary import fingerprint

logger = logging.getLogger(__name__)

POLL_FORM_TICK_SECONDS = 60.0
INVITE_GAP_SECONDS = 1.0  # between invitation DMs (Discord rate limits)
POLL_PERMS = ("view_channel", "send_messages", "send_polls")
_RAW_MENTION = re.compile(r"<@[!&]?(\d+)>")


class PollsMixin:
    def _polls_init(self) -> None:
        """Call after `_coo_init` (needs `self.docs`)."""
        self.poll_repo = P.PollRepo(self.docs)
        self.form_repo = F.FormRepo(self.docs)
        self._pf_last = float("-inf")
        self._form_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------- agent tools
    def _poll_form_tools(self, message, guild, member, is_exec: bool, lang: str, request_id: str):
        gid = guild.id
        if not self.team.enabled(gid, "polls"):
            return []
        loop = asyncio.get_running_loop()
        in_dm = message.guild is None
        here = None if in_dm else message.channel

        def run(coro, timeout=60):
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

        def confirmed(*parts) -> bool:
            """True when the same request was proposed in an earlier message of this asker."""
            return self.secretary_state.confirm(
                ("pollform", gid, member.id, fingerprint(*parts)), request_id
            )

        def create_poll(
            question: str,
            options: list[str],
            channel: str | None = None,
            hours: int = 24,
            multiple: bool = False,
        ) -> dict:
            """Post a native Discord poll (people vote with Discord's own buttons). Use for "vote on
            ..." / "โหวตหน่อยว่า ..." with the options the user gave. Posts in the current channel;
            `channel` only if the user named another one (needs a confirmation: if the result is
            needs_confirmation, tell the requester where it will go, ask them to confirm, then call
            again with EXACTLY the same arguments in a later message). If the result has
            how_to_grant, relay that sentence as is.

            Args:
                question: the poll question.
                options: 2 to 10 short answers.
                channel: another channel's name, optional (default: where you were asked).
                hours: how long it stays open, default 24.
                multiple: true to let people pick several options.
            """
            return run(
                self._create_poll(
                    guild, member, here, question, options, channel, hours, multiple, lang,
                    confirmed,
                )
            )  # fmt: skip

        def poll_results(poll: str = "last") -> dict:
            """Votes of a poll Salid created: who is ahead, counts per option, whether it ended.
            Use for "how is the vote going" / "ผลโหวต".

            Args:
                poll: "last" for the most recent poll, or part of its question, or its message link.
            """
            return run(self._poll_results(guild, member, poll))

        def start_form(
            title: str,
            questions: list[str],
            people: list[str] | None = None,
            role: str | None = None,
            channel: str | None = None,
            deadline_hours: float = 48,
        ) -> dict:
            """Survey people by private message: Salid DMs each one the questions one at a time,
            they just reply in the DM; Salid records the answers, reminds non-responders once
            before the deadline, and the asker later reads them with form_results. Use for "ask
            everyone in the sales team who can come Friday" / "make a quick survey". Give the
            recipients as `people` (names or @mentions), a `role` name, or a `channel` whose
            members get it. Needs a confirmation (the result is needs_confirmation: tell the
            requester who will be messaged and the questions, ask them to confirm, then call again
            with EXACTLY the same arguments in a later message). At most 50 people, more than 10
            only for executives / managers. Never use it for money transfers or passwords.

            Args:
                title: short name of the form.
                questions: 1 to 10 short questions, in the language of the people.
                people: names or @mentions of the people to ask, optional.
                role: a role name; its members are asked, optional.
                channel: a channel name; its members are asked, optional.
                deadline_hours: hours until the form closes, default 48.
            """
            return run(
                self._start_form(
                    guild, member, is_exec, title, questions, people, role, channel,
                    deadline_hours, lang, confirmed,
                ),
                timeout=90,
            )  # fmt: skip

        def form_results(form: str = "last", file: str = "none") -> dict:
            """Results of a form the asker created (or that someone below them created): progress,
            who has not answered, and the answers. Shown only to the creator and people above
            them. In a public channel the details are sent to the asker by DM (the result then has
            only counts); in a DM the answers come back here so you can summarise them.

            Args:
                form: "last" for the newest form, or part of its title, or its id.
                file: none, pdf or csv; the file is sent to the asker by DM.
            """
            return run(self._form_results(guild, member, in_dm, form, file, lang), timeout=90)

        return [create_poll, poll_results, start_form, form_results]

    # ------------------------------------------------------------- polls
    async def _create_poll(
        self, guild, member, here, question, options, channel_ref, hours, multiple, lang, confirmed
    ) -> dict:
        clean = P.clean_poll(question, options, hours)
        if isinstance(clean, dict):
            return clean
        q, opts, h = clean
        if channel_ref:
            res = resolve_channel(
                guild.text_channels, channel_ref, lambda c: can_see_and_post(c, member)
            )
            if res.status == "ambiguous":
                return {
                    "error": "ambiguous_channel",
                    "candidates": [c.name for c in res.candidates],
                }
            if res.status == "not_allowed":
                return {"error": "not_allowed", "note": "the requester cannot post in that channel"}
            if res.status != "ok":
                return {"error": "channel_not_found"}
            target = res.channel
        else:
            target = here
        if target is None:
            return {"error": "need_channel", "note": "ask which channel to post the poll in"}
        if target is not here and not can_see_and_post(target, member):
            return {"error": "not_allowed", "note": "the requester cannot post in that channel"}
        bad = channel_missing(target, guild, POLL_PERMS)
        if bad:
            return {
                "error": "missing_permission",
                "channel": target.name,
                "how_to_grant": permission_message(bad, lang, self.application_id),
            }
        if target is not here and not confirmed("poll", target.id, q, opts, h, bool(multiple)):
            return {
                "status": "needs_confirmation",
                "destination": f"#{target.name}",
                "question": q,
                "options": opts,
                "instruction": "Tell the requester where the poll would go, ask them to confirm, "
                "then call create_poll again with EXACTLY the same arguments in a later message. "
                "Do not claim it was posted.",
            }
        poll = discord.Poll(question=q, duration=timedelta(hours=h), multiple=bool(multiple))
        for o in opts:
            poll.add_answer(text=o)
        try:
            sent = await target.send(poll=poll)
        except discord.Forbidden:
            return {
                "error": "missing_permission",
                "channel": target.name,
                "how_to_grant": permission_message(["send_polls"], lang, self.application_id),
            }
        except discord.HTTPException:
            logger.warning("create_poll: posting failed", exc_info=True)
            return {"error": "could_not_post", "channel": target.name}
        rec = P.new_record(
            guild_id=guild.id,
            channel_id=target.id,
            message_id=sent.id,
            creator_id=member.id,
            question=q,
            options=opts,
            hours=h,
            multiple=bool(multiple),
        )
        await asyncio.to_thread(self.poll_repo.put, rec)
        return {
            "status": "posted",
            "channel": target.name,
            "link": getattr(sent, "jump_url", ""),
            "closes_in_hours": h,
            "note": "Salid posts the result when it ends. Reply in one short line.",
        }

    async def _poll_counts(self, guild, rec) -> tuple[list[int], bool] | None:
        """(votes per stored option, ended) read from the poll message; None when it is gone."""
        ch = guild.get_channel(rec["channel_id"])
        if ch is None:
            return None
        try:
            msg = await ch.fetch_message(rec["message_id"])
        except discord.HTTPException:
            return None
        poll = getattr(msg, "poll", None)
        if poll is None:
            return None
        by_text = {a.text: a.vote_count for a in poll.answers}
        counts = [int(by_text.get(o, 0) or 0) for o in rec["options"]]
        return counts, bool(poll.is_finalised())

    async def _poll_results(self, guild, member, ref) -> dict:
        def visible(rec) -> bool:
            ch = guild.get_channel(rec["channel_id"])
            return ch is not None and can_see(ch, member)

        rec, cands = await asyncio.to_thread(self.poll_repo.find, guild.id, ref, visible)
        if rec is None:
            if cands:
                return {"error": "which_poll", "candidates": [c["question"] for c in cands]}
            return {"error": "poll_not_found"}
        got = await self._poll_counts(guild, rec)
        if got is None:
            return {"error": "poll_message_gone", "question": rec["question"]}
        counts, ended = got
        return {
            "question": rec["question"],
            **P.tally(rec["options"], counts),
            "ended": ended,
            "ends_at": rec["ends_at"],
            "note": "Counts only; Discord keeps who voted for what.",
        }

    async def _poll_finish(self, guild, rec, now: datetime) -> None:
        """Post the one-line result of an ended poll and remember it. Retried every tick until
        Discord finalises the poll (or GRACE passed)."""
        ends = datetime.fromisoformat(rec["ends_at"])
        got = await self._poll_counts(guild, rec)
        if got is not None and not got[1] and now < ends + P.GRACE:
            return
        rec["done"] = True
        await asyncio.to_thread(self.poll_repo.put, rec)
        ch = guild.get_channel(rec["channel_id"])
        if got is None or ch is None:
            return
        counts = got[0]
        lang = self._lang(ch)
        line = P.result_line(rec["question"], rec["options"], counts, lang)
        try:
            sent = await ch.send(line, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException:
            logger.warning("poll result could not be posted", exc_info=True)
            return
        if self.team.enabled(guild.id, "memory"):
            msg = build_message(
                self.s,
                guild_id=guild.id,
                channel_id=ch.id,
                message_id=sent.id,
                author_id=getattr(self.user, "id", 0),
                author_name="Salid",
                created_at=datetime.now(UTC),
                text=t("poll_memory", lang, line=line),
                jump_url=getattr(sent, "jump_url", ""),
                kind="salid",
            )
            self._memory_enqueue(partial(self.memory.record, msg))

    # ------------------------------------------------------------- forms: starting
    def _form_recipients(self, guild, member, people, role, channel) -> tuple[dict, dict | None]:
        """({member id: name}, error) from names / a role / a channel. Blocking (member scans)."""
        found: dict[int, str] = {}

        def add(m) -> None:
            if not getattr(m, "bot", False) and m.id != member.id:
                found[m.id] = m.display_name

        for ref in people or []:
            ref = str(ref).strip()
            mention = _RAW_MENTION.fullmatch(ref)
            if mention:
                m = guild.get_member(int(mention.group(1)))
                if m is None:
                    return {}, {"error": "person_not_found", "person": ref}
                add(m)
                continue
            hits = self._match_people(guild, ref)
            if not hits:
                return {}, {"error": "person_not_found", "person": ref}
            if len(hits) > 1:
                return {}, {
                    "error": "ambiguous",
                    "person": ref,
                    "candidates": [m.display_name for m in hits[:10]],
                }
            add(hits[0])
        if role:
            want = str(role).strip().lstrip("@").casefold()
            roles = [r for r in guild.roles if r.name.casefold() == want and not r.is_default()]
            if not roles:
                roles = [r for r in guild.roles if want in r.name.casefold() and not r.is_default()]
            if len(roles) != 1:
                return {}, {
                    "error": "role_not_found" if not roles else "ambiguous_role",
                    "candidates": [r.name for r in roles[:10]],
                }
            for m in roles[0].members:
                add(m)
        if channel:
            ch = self._find_text_channel(guild, channel, None)
            if ch is None:
                return {}, {"error": "channel_not_found"}
            if not can_see(ch, member):
                return {}, {"error": "not_allowed", "note": "the requester cannot see that channel"}
            viewers = self._channel_viewers(guild, ch)
            for uid in viewers:
                m = guild.get_member(uid)
                if m is not None:
                    add(m)
        return found, None

    @staticmethod
    def _match_people(guild, name: str) -> list:
        from .mentions import match_members

        return match_members(guild, name)

    async def _start_form(
        self,
        guild,
        member,
        is_exec,
        title,
        questions,
        people,
        role,
        channel,
        hours,
        lang,
        confirmed,
    ) -> dict:
        qs = F.clean_questions(questions)
        if isinstance(qs, dict):
            return qs
        if SENSITIVE_DM_RE.search(f"{title} " + " ".join(qs)):
            return {
                "error": "refused_sensitive",
                "reason": "money transfer or credential request to other people; decline politely",
            }
        if not (people or role or channel):
            return {"error": "need_recipients", "note": "ask who should be asked"}
        found, err = await asyncio.to_thread(
            self._form_recipients, guild, member, people, role, channel
        )
        if err:
            return err
        manager = is_exec or self._may_configure(member)
        refused = F.recipients_refusal(len(found), manager)
        if refused:
            return refused
        h = F.clamp_hours(hours)
        if not confirmed("form", title, qs, sorted(found), h):
            names = list(found.values())
            return {
                "status": "needs_confirmation",
                "title": title,
                "questions": qs,
                "recipients": names[:20],
                "recipient_count": len(names),
                "closes_in_hours": h,
                "instruction": "Tell the requester who will get a private message and the "
                "questions, ask them to confirm, then call start_form again with EXACTLY the same "
                "arguments in a later message. Do not claim it was sent.",
            }
        form = F.new_form(
            guild_id=guild.id,
            creator_id=member.id,
            creator_name=member.display_name,
            title=title,
            questions=qs,
            people=found,
            hours=h,
            lang=lang,
        )
        await asyncio.to_thread(self.form_repo.put, form)
        self._spawn(self._form_launch(guild, form["id"]))
        return {
            "status": "sending",
            "form_id": form["id"],
            "people": len(found),
            "closes_in_hours": h,
            "note": "Invitations go out now one by one; ask for form_results any time. Anyone "
            "with DMs closed is listed there.",
        }

    def _form_lock(self, form_id: str) -> asyncio.Lock:
        return self._form_locks.setdefault(form_id, asyncio.Lock())

    async def _form_update(self, guild_id: int, form_id: str, fn):
        """Load the form, apply `fn(form)` and save it, serialised per form (DM answers, timers and
        the launch all touch the same document). Returns `(fn's value,)`, or None if the form is gone."""
        async with self._form_lock(form_id):
            form = await asyncio.to_thread(self.form_repo.get, guild_id, form_id)
            if form is None:
                return None
            out = fn(form)
            await asyncio.to_thread(self.form_repo.put, form)
            return (out,)

    def _local_time(self, iso: str) -> str:
        return datetime.fromisoformat(iso).astimezone(self.s.zone).strftime("%d/%m %H:%M")

    async def _form_launch(self, guild, form_id: str) -> None:
        """DM the first question to everyone, one by one. A person with DMs closed is marked
        unreachable. The creator hears about failures."""
        form = await asyncio.to_thread(self.form_repo.get, guild.id, form_id)
        if form is None:
            return
        lang, failed = form["lang"], []
        for uid in [int(u) for u in form["recipients"]]:
            body = "\n\n".join(
                [
                    t(
                        "form_invite",
                        lang,
                        creator=form["creator_name"],
                        server=guild.name,
                        title=form["title"],
                        n=len(form["questions"]),
                        deadline=self._local_time(form["deadline"]),
                    ),
                    t("form_question", lang, i=1, n=len(form["questions"]),
                      question=form["questions"][0]),
                ]
            )  # fmt: skip
            res = await send_dm(guild.get_member(uid), body)
            now = datetime.now(UTC)

            def apply(f, uid=uid, ok=res.ok, now=now):
                if ok:
                    F.mark_asked(f, uid, now)
                else:
                    f["recipients"][str(uid)]["state"] = F.UNREACHABLE

            await self._form_update(guild.id, form_id, apply)
            if not res.ok:
                failed.append(form["recipients"][str(uid)]["name"])
            await self._sleep(INVITE_GAP_SECONDS)
        if failed:
            creator = guild.get_member(form["creator_id"])
            note = (
                f"ส่งแบบสอบถาม “{form['title']}” ไม่ถึง {', '.join(failed[:10])} (น่าจะปิด DM) ค่ะ"
                if lang == "th"
                else f"I couldn't DM {', '.join(failed[:10])} for “{form['title']}” "
                "(their DMs are probably closed)."
            )
            await send_dm(creator, note)

    # ------------------------------------------------------------- forms: answers by DM
    async def _form_dm_answer(self, message, guild, member) -> bool:
        """True when this DM was an answer to an open form question (and has been handled)."""
        if not self.team.enabled(guild.id, "polls"):
            return False
        text = (message.content or "").strip()
        if not text:
            return False
        now = datetime.now(UTC)
        forms = await asyncio.to_thread(self.form_repo.all, guild.id)
        form = F.asking_form(forms, member.id, now)
        if form is None:
            return False
        question = F.current_question(form, member.id)
        if question is None:
            return False
        stop, skip = bool(F.STOP_RE.match(text)), bool(F.SKIP_RE.match(text))
        if not stop and not skip:
            ok = await asyncio.to_thread(
                F.looks_like_answer, self.llm, self.s.gate_model, question, text
            )
            if not ok:
                return False
        lang = form["lang"]
        uid = member.id

        def apply(f):
            if F.current_question(f, uid) != question:  # changed meanwhile (double send)
                return "stale"
            if stop:
                F.decline(f, uid)
                return "stopped"
            return F.record_answer(f, uid, None if skip else text, datetime.now(UTC))

        res = await self._form_update(guild.id, form["id"], apply)
        if res is None:
            return False
        out = res[0]
        if out == "stale":
            return True
        if out == "stopped":
            reply = t("form_declined_ack", lang)
        elif out is None:
            reply = t("form_thanks", lang, creator=form["creator_name"])
        else:
            reply = t(
                "form_question", lang, i=out + 1, n=len(form["questions"]),
                question=form["questions"][out],
            )  # fmt: skip
        with contextlib.suppress(discord.HTTPException):
            await message.reply(reply, mention_author=False)
        return True

    # ------------------------------------------------------------- forms: results
    async def _form_results(self, guild, member, in_dm, ref, file, lang) -> dict:
        ranks = await asyncio.to_thread(self._ranks, guild)

        def visible(f) -> bool:
            return F.can_view_results(member.id, f, ranks)

        form, cands = await asyncio.to_thread(self.form_repo.find, guild.id, ref, visible)
        if form is None:
            if cands:
                return {"error": "which_form", "candidates": [c["title"] for c in cands]}
            return {"error": "form_not_found"}
        fmt = (file or "none").strip().lower()
        fonts = find_fonts(self.s.pdf_font)
        files: list[discord.File] = []
        import io

        if fmt == "pdf":
            data = F.form_pdf(
                form, org=guild.name, lang=lang, generated_at=self.s.now(), fonts=fonts
            )
            files.append(discord.File(io.BytesIO(data), filename=f"form_{form['id']}.pdf"))
        elif fmt == "csv":
            data = F.csv_bytes(form, lang)
            files.append(discord.File(io.BytesIO(data), filename=f"form_{form['id']}.csv"))
        out = F.summary_for_agent(form, lang, with_answers=in_dm)
        if not in_dm or files:
            body = F.text_table(form, lang)
            if len(body) > 1800 and not files:  # too long for one message: send it as a file
                data = F.csv_bytes(form, lang)
                files.append(discord.File(io.BytesIO(data), filename=f"form_{form['id']}.csv"))
            res = await send_dm(member, t("form_results_dm", lang) if files else body[:1900], files)
            if not res.ok:
                return {"error": "could_not_dm_results", "note": "ask them to open their DMs"}
            out["sent_by_dm"] = True
        return out

    # ------------------------------------------------------------- timers
    async def _polls_forms_due(self) -> None:
        now = self._clock()
        if not self.guilds or now - self._pf_last < POLL_FORM_TICK_SECONDS:
            return
        self._pf_last = now
        for guild in list(self.guilds):
            if not self.team.enabled(guild.id, "polls"):
                continue
            try:
                await self._polls_forms_guild(guild)
            except Exception:
                logger.exception("polls / forms timer failed for guild %s", guild.id)

    async def _polls_forms_guild(self, guild) -> None:
        now = datetime.now(UTC)
        for rec in await asyncio.to_thread(self.poll_repo.due, guild.id, now):
            await self._poll_finish(guild, rec, now)
        for form in await asyncio.to_thread(self.form_repo.all, guild.id):
            if form["status"] != "open":
                continue
            if not F.is_open(form, now):
                await self._form_close(guild, form["id"])
                continue
            for uid in F.due_reminders(form, now):
                await self._form_remind(guild, form["id"], uid)

    async def _form_remind(self, guild, form_id: str, uid: int) -> None:
        def apply(f):
            r = f["recipients"].get(str(uid))
            if r is None or r["reminded"] or r["state"] not in F.OPEN_STATES:
                return None
            r["reminded"] = True
            return dict(f)

        res = await self._form_update(guild.id, form_id, apply)
        if not res or not res[0]:
            return
        form = res[0]
        lang = form["lang"]
        text = t(
            "form_reminder", lang, title=form["title"], creator=form["creator_name"],
            deadline=self._local_time(form["deadline"]),
        )  # fmt: skip
        q = F.current_question(form, uid)
        if q:
            text += "\n\n" + t(
                "form_question", lang, i=form["recipients"][str(uid)]["idx"] + 1,
                n=len(form["questions"]), question=q,
            )  # fmt: skip
        await send_dm(guild.get_member(uid), text)

    async def _form_close(self, guild, form_id: str) -> None:
        def apply(f):
            if f["status"] != "open":
                return None
            f["status"] = "closed"
            return dict(f)

        res = await self._form_update(guild.id, form_id, apply)
        if not res or not res[0]:
            return
        form = res[0]
        done, total = F.progress(form)
        text = t("form_closed", form["lang"], title=form["title"], done=done, total=total)
        await send_dm(guild.get_member(form["creator_id"]), text)
