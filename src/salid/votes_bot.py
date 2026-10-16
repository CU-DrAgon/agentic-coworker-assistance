"""Meeting votes, Discord side: a `SecretaryBot` mixin.

Agent tools (`_vote_tools`, appended by `SecretaryBot._agent_kwargs` so chat and voice share them):
`call_vote`, `close_vote`, `vote_status`. Each voter gets ONE ballot message (buttons) in their
private Salid channel (DM fallback) through `send_dm`; a press records or changes the ballot. The
meeting chat gets one tally message (counts only, edited on every ballot); at the end a result
line is posted, recorded in the minutes and (in a voice meeting) announced at the next pause.
Voters, scope and who may close are decided here in code, never by the model.

Offers: in listen / join text channels, and from the classic meeting transcript, a conservative
detector (`votes.looks_like_vote_call` + a cheap yes/no extraction) posts ONE small text offer with
buttons "เปิดโหวต / ไม่ต้อง". Salid stays silent in voice.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import datetime, timedelta

import discord

from . import votes as V
from .dm import SENSITIVE_DM_RE, send_dm
from .i18n import t
from .mentions import match_members

logger = logging.getLogger(__name__)

VOTE_TICK_SECONDS = 5.0
OFFER_VOTERS_MAX = 30
OFFER_ACTIVE_HOURS = 24
_RAW_MENTION = re.compile(r"<@[!&]?(\d+)>")
_EVERYONE = {"everyone", "everybody", "all", "ทุกคน", "ทุกคนในทีม", "ทั้งทีม", "ทั้งหมด"}


class VoteButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:vote:(?P<id>[0-9a-f]{12}):(?P<i>[0-5])",
):
    """One option on a ballot. The custom id carries the vote id and the option index, and the bot
    registers the class once, so ballots keep working after a restart."""

    def __init__(self, vote_id: str, index: int, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=(label or str(index + 1))[:80],
                style=discord.ButtonStyle.primary,
                custom_id=f"salid:vote:{vote_id}:{index}",
            )
        )
        self.vote_id, self.index = vote_id, index

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["id"], int(match["i"]), getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_vote_press(interaction, self.vote_id, self.index)


class VoteOfferButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"salid:voff:(?P<act>go|no):(?P<id>[0-9a-f]{12})",
):
    """เปิดโหวต / ไม่ต้อง on an offer message."""

    def __init__(self, act: str, offer_id: str, label: str | None = None) -> None:
        super().__init__(
            discord.ui.Button(
                label=label or act,
                style=discord.ButtonStyle.success if act == "go" else discord.ButtonStyle.secondary,
                custom_id=f"salid:voff:{act}:{offer_id}",
            )
        )
        self.act, self.offer_id = act, offer_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls(match["act"], match["id"], getattr(item, "label", None))

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.client.handle_vote_offer(interaction, self.act, self.offer_id)


def ballot_view(vote: dict) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for i, o in enumerate(vote["options"]):
        view.add_item(VoteButton(vote["id"], i, o))
    return view


def offer_view(offer_id: str, lang: str) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(VoteOfferButton("go", offer_id, t("vote_offer_open", lang)))
    view.add_item(VoteOfferButton("no", offer_id, t("vote_offer_no", lang)))
    return view


class VotesMixin:
    def _votes_init(self) -> None:
        """Call after `_coo_init` (needs `self.docs`)."""
        self.vote_repo = V.VoteRepo(self.docs)
        self.offer_repo = V.OfferRepo(self.docs)
        self._vote_locks: dict[str, asyncio.Lock] = {}
        self._open_votes: dict[str, tuple[int, datetime]] = {}  # id -> (guild, closes_at)
        self._votes_scanned = False
        self._votes_last = float("-inf")
        self._vote_tasks: set[asyncio.Task] = set()
        self._offer_last: dict[int, float] = {}  # channel -> clock of the last offer
        self._offer_silent: dict[int, float] = {}  # channel -> clock until which offers are off
        self._offer_seen: set[tuple[int, str]] = set()  # (channel, question key) already offered
        self._offer_busy: set[int] = set()

    # ------------------------------------------------------------- helpers
    def _votes_on(self, gid: int) -> bool:
        return self.team.enabled(gid, "votes")

    def _vote_lock(self, vote_id: str) -> asyncio.Lock:
        return self._vote_locks.setdefault(vote_id, asyncio.Lock())

    def _vote_spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._vote_tasks.add(task)
        task.add_done_callback(self._vote_tasks.discard)

    def _vote_tz(self):
        return self.s.now().tzinfo

    def _vote_session(self, guild, member, text_channel_id):
        """The running meeting the asker sits in, or whose text chat this is."""
        ch = getattr(getattr(member, "voice", None), "channel", None)
        s = self.meetings.session_in(guild.id, ch.id) if ch is not None else None
        return s or self.meetings.session_for_channel(text_channel_id)

    def _vote_channel(self, guild, vote: dict):
        s = self.meetings.session_in(guild.id, vote["voice_channel_id"])
        if s is not None and s.text_channel is not None:
            return s.text_channel
        return guild.get_channel(vote["channel_id"])

    def _vote_names(self, guild, vote: dict) -> dict[int, str]:
        out = {}
        for u in vote["voters"]:
            m = guild.get_member(u)
            out[u] = getattr(m, "display_name", str(u)) if m else str(u)
        return out

    # ------------------------------------------------------------- agent tools
    def _vote_tools(self, message, guild, member, is_exec: bool, lang: str):
        if not self._votes_on(guild.id):
            return []
        loop = asyncio.get_running_loop()
        here = message.channel

        def run(coro, timeout=60):
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

        def call_vote(
            question: str,
            options: list[str] | None = None,
            minutes: int = 5,
            anonymous: bool = False,
            voters: list[str] | None = None,
        ) -> dict:
            """Open a vote (ลงมติ): Salid sends every voter a message with one button per option
            in their private channel, counts the presses and posts the result in the meeting chat
            and the minutes. Use for "ขอมติว่า..." / "ลงมติเรื่อง..." / "let's vote on...". It
            returns at once; reply with the `say` line. Defaults: options agree / disagree /
            abstain, 5 minutes, names shown in the result, voters = the people in the voice room
            (or who can read this channel).

            Args:
                question: what is being decided, as a short sentence.
                options: 2 to 6 custom answers, optional.
                minutes: how long it stays open, 1 to 60, default 5.
                anonymous: true when the user says no names ("ไม่เปิดเผยชื่อ", "anonymous").
                voters: names or @mentions, or ["everyone"] / ["ทุกคนในทีม"], optional.
            """
            return run(
                self._call_vote(
                    message, guild, member, here, question, options, minutes, anonymous, voters,
                    lang,
                )
            )  # fmt: skip

        def close_vote(vote: str = "last") -> dict:
            """Close a vote now and post its result. Only the person who opened it or an
            executive. Use for "ปิดโหวต" / "close the vote".

            Args:
                vote: "last" for the newest open vote, or part of its question, or its id.
            """
            return run(self._close_vote_tool(guild, member, is_exec, vote))

        def vote_status(vote: str = "last") -> dict:
            """How a vote is going: how many voted and the counts (names only after it closes and
            only when it is not anonymous). For "โหวตถึงไหนแล้ว" / "how is the vote going".

            Args:
                vote: "last" for the newest vote, or part of its question, or its id.
            """
            return run(self._vote_status(guild, member, is_exec, vote))

        return [call_vote, close_vote, vote_status]

    def _resolve_voters(self, guild, member, here, session, voters) -> list | dict:
        """Members who vote, or an error dict. Default: the humans in the voice room, else those
        who can read this channel."""
        if voters:
            wanted = [str(v) for v in voters]
            if any(v.strip().casefold() in _EVERYONE for v in wanted):
                found = self._members(guild)
            else:
                found = []
                for v in wanted:
                    m = _RAW_MENTION.fullmatch(v.strip())
                    if m:
                        who = guild.get_member(int(m.group(1)))
                        if who is not None and not getattr(who, "bot", False):
                            found.append(who)
                        continue
                    hits = match_members(guild, v)
                    if not hits:
                        return {"error": "person_not_found", "person": v}
                    if len(hits) > 1:
                        return {
                            "error": "ambiguous",
                            "person": v,
                            "candidates": [h.display_name for h in hits[:10]],
                        }
                    found.append(hits[0])
        elif session is not None:
            found = self.meetings._humans(session.voice_channel)
        else:
            found = [
                m
                for m in self._members(guild)
                if getattr(here.permissions_for(m), "view_channel", False)
                or getattr(here.permissions_for(m), "read_messages", False)
            ]
        uniq = list({m.id: m for m in found}.values())
        if not uniq:
            return {"error": "no_voters", "note": "ask who should vote"}
        if len(uniq) > V.MAX_VOTERS:
            return {"error": "too_many_voters", "max": V.MAX_VOTERS}
        return uniq

    async def _call_vote(
        self, message, guild, member, here, question, options, minutes, anonymous, voters, lang
    ) -> dict:
        session = self._vote_session(guild, member, getattr(here, "id", None))
        vlang = session.lang if session is not None else lang
        clean = V.clean_vote(question, options, minutes, vlang)
        if isinstance(clean, dict):
            return clean
        q, opts, mins = clean
        if SENSITIVE_DM_RE.search(q):
            return {"error": "refused_sensitive", "reason": "money transfer or credential request"}
        people = self._resolve_voters(guild, member, here, session, voters)
        if isinstance(people, dict):
            return people
        return await self._open_vote(
            guild, member, session, here, q, opts, mins, bool(anonymous), people, vlang
        )

    async def _open_vote(
        self, guild, opener, session, here, q, opts, mins, anonymous, people, lang
    ) -> dict:
        if (
            len(await asyncio.to_thread(self.vote_repo.open_votes, guild.id))
            >= V.MAX_OPEN_PER_GUILD
        ):
            return {"error": "too_many_open_votes", "max": V.MAX_OPEN_PER_GUILD}
        text_ch = session.text_channel if session is not None else here
        now = self.s.now()
        vote = V.new_vote(
            guild_id=guild.id,
            channel_id=text_ch.id,
            room=getattr(session.voice_channel, "name", "") if session is not None else "",
            question=q,
            options=opts,
            voters=[m.id for m in people],
            opened_by=opener.id,
            opener_name=getattr(opener, "display_name", ""),
            minutes=mins,
            anonymous=anonymous,
            lang=lang,
            voice_channel_id=session.voice_channel.id if session is not None else 0,
            now=now,
        )
        await asyncio.to_thread(self.vote_repo.put, vote)
        self._open_votes[vote["id"]] = (guild.id, datetime.fromisoformat(vote["closes_at"]))
        self._vote_spawn(self._vote_deliver(guild, vote["id"], people, text_ch, session))
        return {
            "status": "opened",
            "vote": vote["id"],
            "voters": len(people),
            "closes_at": V.hhmm(vote["closes_at"], self._vote_tz()),
            "anonymous": anonymous,
            "say": t("vote_say", lang, minutes=mins),
        }

    # ------------------------------------------------------------- delivery
    async def _vote_deliver(self, guild, vote_id: str, people, text_ch, session) -> None:
        """Post the tally message, then one ballot per voter. Never raises."""
        try:
            vote = await asyncio.to_thread(self.vote_repo.get, vote_id)
            if vote is None:
                return
            tz = self._vote_tz()
            tally = await text_ch.send(V.tally_text(vote, vote["lang"], tz))
            unreached: list[str] = []
            delivered: dict[str, dict] = {}
            for m in people:
                ulang = self._person_lang(guild.id, m.id, vote["question"])
                key = "vote_ballot" if vote["room"] else "vote_ballot_chat"
                body = t(
                    key, ulang, question=vote["question"], room=vote["room"],
                    opener=vote["opener_name"], time=V.hhmm(vote["closes_at"], tz),
                )  # fmt: skip
                res = await send_dm(m, body, view=ballot_view(vote))
                sent = getattr(res, "message", None)
                if res.ok and sent is not None:
                    delivered[str(m.id)] = {
                        "channel_id": getattr(getattr(sent, "channel", None), "id", 0),
                        "message_id": getattr(sent, "id", 0),
                    }
                else:
                    unreached.append(getattr(m, "display_name", str(m.id)))
            async with self._vote_lock(vote_id):
                fresh = await asyncio.to_thread(self.vote_repo.get, vote_id) or vote
                fresh["tally_message_id"] = getattr(tally, "id", 0)
                fresh["delivered"] = delivered
                fresh["unreached"] = unreached
                await asyncio.to_thread(self.vote_repo.put, fresh)
            if unreached:
                with contextlib.suppress(Exception):
                    await text_ch.send(
                        t("vote_unreached", vote["lang"], names=", ".join(unreached[:10]))
                    )
        except Exception:
            logger.exception("vote delivery failed")

    async def _edit_tally(self, guild, vote: dict) -> None:
        ch = self._vote_channel(guild, vote)
        if ch is None or not vote.get("tally_message_id"):
            return
        with contextlib.suppress(Exception):
            await ch.get_partial_message(vote["tally_message_id"]).edit(
                content=V.tally_text(vote, vote["lang"], self._vote_tz())
            )

    # ------------------------------------------------------------- button presses
    async def handle_vote_press(self, interaction, vote_id: str, index: int) -> None:
        user = interaction.user
        vote = await asyncio.to_thread(self.vote_repo.get, vote_id)
        if vote is None:
            await interaction.response.send_message(t("vote_press_unknown", "th"), ephemeral=True)
            return
        gid = vote["guild_id"]
        lang = self._person_lang(gid, user.id)
        async with self._vote_lock(vote_id):
            vote = await asyncio.to_thread(self.vote_repo.get, vote_id) or vote
            res = V.cast(vote, user.id, index, self.s.now())
            if res == "ok":
                await asyncio.to_thread(self.vote_repo.put, vote)
        if res != "ok":
            key = {"closed": "vote_press_closed", "not_voter": "vote_press_not_voter"}.get(
                res, "vote_press_unknown"
            )
            await interaction.response.send_message(t(key, lang), ephemeral=True)
            return
        body = t(
            "vote_chosen", lang, question=vote["question"], option=vote["options"][index],
            time=V.hhmm(vote["closes_at"], self._vote_tz()),
        )  # fmt: skip
        await interaction.response.edit_message(content=body, view=ballot_view(vote))
        guild = self.get_guild(gid)
        if guild is None:
            return
        if V.all_voted(vote):
            await self._vote_close(guild, vote_id)
        else:
            await self._edit_tally(guild, vote)

    # ------------------------------------------------------------- closing
    async def _vote_close(self, guild, vote_id: str) -> dict | None:
        """Close once: result line in the meeting chat, in the minutes, spoken in the voice room;
        ballots lose their buttons. The closed vote, or None when it was not open."""
        async with self._vote_lock(vote_id):
            vote = await asyncio.to_thread(self.vote_repo.get, vote_id)
            if vote is None or not V.is_open(vote):
                return None
            vote["status"] = "closed"
            vote["closed_at"] = V.iso(self.s.now())
            await asyncio.to_thread(self.vote_repo.put, vote)
        self._open_votes.pop(vote_id, None)
        lang = vote["lang"]
        ch = self._vote_channel(guild, vote)
        text = V.result_text(vote, self._vote_names(guild, vote), lang)
        sent = None
        if ch is not None:
            with contextlib.suppress(Exception):
                sent = await ch.send(text)
        if sent is not None:
            vote["result_message_id"] = getattr(sent, "id", 0)
            await asyncio.to_thread(self.vote_repo.put, vote)
        await self._edit_tally(guild, vote)
        session = self.meetings.session_in(guild.id, vote["voice_channel_id"])
        if session is not None:
            self.meetings.record_vote(session, text.replace("\n", " · "))
            live = getattr(session, "live", None)
            if live is not None:
                cue = t("vote_cue", lang, line=V.result_line(vote, lang, emoji=False))
                self._vote_spawn(live.announce_cue(cue, f"vote {vote_id}"))
        await self._vote_close_ballots(vote)
        return vote

    async def _vote_close_ballots(self, vote: dict) -> None:
        for uid, ref in vote.get("delivered", {}).items():
            with contextlib.suppress(Exception):
                lang = self._person_lang(vote["guild_id"], int(uid))
                b = vote["ballots"].get(uid)
                choice = (
                    t("vote_you_chose", lang, option=vote["options"][b["option"]])
                    if b
                    else t("vote_you_skipped", lang)
                )
                ch = self.get_channel(ref["channel_id"]) or await self.fetch_channel(
                    ref["channel_id"]
                )
                await ch.get_partial_message(ref["message_id"]).edit(
                    content=t("vote_ballot_closed", lang, question=vote["question"], choice=choice),
                    view=None,
                )

    async def _votes_due(self) -> None:
        """Tick: close votes whose deadline passed. Open votes are found once after a start."""
        if not self.guilds or self._clock() - self._votes_last < VOTE_TICK_SECONDS:
            return
        self._votes_last = self._clock()
        if not self._votes_scanned:
            self._votes_scanned = True
            for g in list(self.guilds):
                with contextlib.suppress(Exception):
                    for v in await asyncio.to_thread(self.vote_repo.open_votes, g.id):
                        self._open_votes[v["id"]] = (g.id, datetime.fromisoformat(v["closes_at"]))
        now = self.s.now()
        for vid, (gid, closes) in list(self._open_votes.items()):
            if closes <= now:
                guild = self.get_guild(gid)
                if guild is None:
                    continue
                try:
                    await self._vote_close(guild, vid)
                except Exception:
                    logger.exception("closing vote %s failed", vid)

    # ------------------------------------------------------------- close / status tools
    def _vote_visible(self, member, is_exec: bool):
        return lambda r: is_exec or member.id == r["opened_by"] or member.id in r["voters"]

    async def _close_vote_tool(self, guild, member, is_exec: bool, ref) -> dict:
        rec, cands = await asyncio.to_thread(
            self.vote_repo.find, guild.id, ref, lambda r: V.is_open(r)
        )
        if rec is None:
            return {"error": "vote_not_found", "candidates": [c["question"] for c in cands]}
        if not (is_exec or member.id == rec["opened_by"]):
            return {"error": "not_allowed", "note": "only who opened the vote or an executive"}
        done = await self._vote_close(guild, rec["id"])
        if done is None:
            return {"error": "already_closed"}
        return {"status": "closed", "result": V.result_line(done, done["lang"], emoji=False)}

    async def _vote_status(self, guild, member, is_exec: bool, ref) -> dict:
        rec, cands = await asyncio.to_thread(
            self.vote_repo.find, guild.id, ref, self._vote_visible(member, is_exec)
        )
        if rec is None:
            return {"error": "vote_not_found", "candidates": [c["question"] for c in cands]}
        out = {
            "question": rec["question"],
            "status": rec["status"],
            "voted": len(rec["ballots"]),
            "voters": len(rec["voters"]),
            "anonymous": rec["anonymous"],
            "counts": dict(zip(rec["options"], V.counts(rec), strict=False)),
            "closes_at": V.hhmm(rec["closes_at"], self._vote_tz()),
        }
        if not V.is_open(rec):
            out["result"] = V.result_text(rec, self._vote_names(guild, rec), rec["lang"])
        return out

    # ------------------------------------------------------------- offers from untagged talk
    def _offer_allowed(self, guild_id: int, channel_id: int) -> bool:
        if not (self._votes_on(guild_id) and self.team.enabled(guild_id, "vote_offers")):
            return False
        now = self._clock()
        if now < self._offer_silent.get(channel_id, float("-inf")):
            return False
        return now - self._offer_last.get(channel_id, float("-inf")) >= V.OFFER_GAP.total_seconds()

    def vote_offer_soon(self, message) -> None:
        """Called for an untagged message in a listen / join channel. Background, never raises."""
        guild = getattr(message, "guild", None)
        if guild is None or getattr(message.author, "bot", False) or message.attachments:
            return
        if not V.looks_like_vote_call(message.content or ""):
            return
        if self._is_snoozed(message.channel.id) or not self._offer_allowed(
            guild.id, message.channel.id
        ):
            return
        self._vote_spawn(self._vote_offer(guild, message.channel, message.content, 0, None))

    async def vote_offer_from_room(self, s, name: str, text: str) -> None:
        """An untagged line of a voice meeting (classic transcript): the offer is posted as text in
        the meeting chat, never spoken."""
        try:
            guild = s.voice_channel.guild
            if s.text_channel is None or not V.looks_like_vote_call(text):
                return
            if not self._offer_allowed(guild.id, s.text_channel.id):
                return
            await self._vote_offer(guild, s.text_channel, text, s.voice_channel.id, s.lang)
        except Exception:
            logger.warning("room vote offer failed", exc_info=True)

    async def _vote_offer(self, guild, channel, text: str, voice_id: int, lang) -> None:
        cid = channel.id
        if cid in self._offer_busy:
            return
        self._offer_busy.add(cid)
        try:
            history = [x async for x in self._vote_history(channel)] if not voice_id else []
            got = await asyncio.to_thread(
                V.extract_offer, self.llm, self.s.gate_model, history, text
            )
            if got is None or not self._offer_allowed(guild.id, cid):
                return
            key = (cid, V.question_key(got["question"]))
            if key in self._offer_seen:
                return
            lang = lang or self._lang(channel)
            clean = V.clean_vote(got["question"], got["options"], V.DEFAULT_MINUTES, lang)
            if isinstance(clean, dict):  # explicit options unusable: fall back to agree / disagree
                clean = V.clean_vote(got["question"], [], V.DEFAULT_MINUTES, lang)
                if isinstance(clean, dict):
                    return
            q, opts, _ = clean
            offer = V.new_offer(
                guild_id=guild.id, channel_id=cid, question=q, options=opts,
                voice_channel_id=voice_id, lang=lang, now=self.s.now(),
            )  # fmt: skip
            await asyncio.to_thread(self.offer_repo.put, offer)
            self._offer_seen.add(key)
            self._offer_last[cid] = self._clock()
            await channel.send(
                t("vote_offer", lang, question=q, options=" / ".join(opts)),
                view=offer_view(offer["id"], lang),
            )
        except Exception:
            logger.warning("vote offer failed", exc_info=True)
        finally:
            self._offer_busy.discard(cid)

    async def _vote_history(self, channel):
        for snap in list(self.recent._chans.get(channel.id, ()))[-6:]:
            name = getattr(snap.author, "display_name", "?")
            yield f"{name}: {snap.content}"

    async def handle_vote_offer(self, interaction, act: str, offer_id: str) -> None:
        offer = await asyncio.to_thread(self.offer_repo.get, offer_id)
        lang = (offer or {}).get("lang", "th")
        if offer is None or offer["status"] != "pending":
            await interaction.response.send_message(t("vote_offer_gone", lang), ephemeral=True)
            return
        guild = self.get_guild(offer["guild_id"])
        if guild is None:
            await interaction.response.send_message(t("vote_offer_gone", lang), ephemeral=True)
            return
        cid = offer["channel_id"]
        if act == "no":
            offer["status"] = "declined"
            await asyncio.to_thread(self.offer_repo.put, offer)
            self._offer_silent[cid] = self._clock() + V.OFFER_SILENCE.total_seconds()
            await interaction.response.edit_message(
                content=t("vote_offer_declined", lang), view=None
            )
            return
        if not self._votes_on(guild.id):
            await interaction.response.send_message(t("vote_offer_gone", lang), ephemeral=True)
            return
        offer["status"] = "opened"
        await asyncio.to_thread(self.offer_repo.put, offer)
        await interaction.response.edit_message(
            content=t("vote_offer_opened", lang, question=offer["question"]), view=None
        )
        opener = guild.get_member(interaction.user.id) or interaction.user
        channel = guild.get_channel(cid) or interaction.channel
        session = (
            self.meetings.session_in(guild.id, offer["voice_channel_id"])
            if offer["voice_channel_id"]
            else None
        )
        if session is not None:
            people = self.meetings._humans(session.voice_channel)
        else:
            people = await self._offer_voters(guild, channel, opener)
        if not people:
            return
        await self._open_vote(
            guild, opener, session, channel, offer["question"], offer["options"],
            V.DEFAULT_MINUTES, False, people, lang,
        )  # fmt: skip

    async def _offer_voters(self, guild, channel, opener) -> list:
        """The thread's members, else the people who wrote in the channel in the last 24 h (newest
        first, at most OFFER_VOTERS_MAX), plus the one who pressed. Only people who can read it."""
        members: dict[int, object] = {}
        if isinstance(channel, discord.Thread):
            for m in channel.members:
                u = guild.get_member(m.id)
                if u is not None and not u.bot:
                    members[u.id] = u
        else:
            since = self.s.now() - timedelta(hours=OFFER_ACTIVE_HOURS)
            ids: list[int] = []
            try:
                days = sorted({since.date().isoformat(), self.s.now().date().isoformat()})
                msgs = await asyncio.to_thread(
                    self.memory_store.messages, guild.id, {channel.id}, days[0], days[-1],
                    channel_id=channel.id, limit=300,
                )  # fmt: skip
                for m in msgs:  # newest first
                    ca = m.created_at
                    if m.kind != "salid" and m.author_id and (ca is None or ca >= since):
                        ids.append(m.author_id)
            except Exception:
                logger.warning("could not read recent authors", exc_info=True)
            for snap in self.recent._chans.get(channel.id, ()):
                ids.append(getattr(snap.author, "id", 0))
            for uid in reversed(ids):
                u = guild.get_member(uid)
                if u is not None and not getattr(u, "bot", False):
                    members[u.id] = u
        members[opener.id] = opener
        readable = [
            m
            for m in members.values()
            if getattr(channel.permissions_for(m), "view_channel", True)
            and getattr(channel.permissions_for(m), "read_messages", True)
        ]
        # keep the presser, then the most recently active, up to the cap
        rest = [m for m in readable if m.id != opener.id][-(OFFER_VOTERS_MAX - 1) :]
        return [m for m in readable if m.id == opener.id] + rest
