# ruff: noqa: F811, F401
"""Meeting votes: validation, ballots, tally, closing, delivery, the minutes, the voice cue and
the offers from untagged talk. Offline with fake Discord objects."""

import asyncio
import itertools
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_coordinator import BOSS, INTERN, NS, PEER, STAFF, world
from test_modes import TMsg, bot  # noqa: F401  (fixture)

from salid import dm as dm_module
from salid import votes as V
from salid.bot import SecretaryBot
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.dm import DMResult
from salid.meeting import MeetingManager
from salid.votes_bot import VoteButton, VoteOfferButton

NOW = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)
TZ = UTC


# ------------------------------------------------------------------ pure
def test_clean_vote_defaults_and_limits():
    q, opts, m = V.clean_vote(" เลื่อนเปิดตัวไหม ", None, None, "th")
    assert q == "เลื่อนเปิดตัวไหม" and opts == ["เห็นด้วย", "ไม่เห็นด้วย", "งดออกเสียง"] and m == 5
    assert V.clean_vote("move it?", [], 5, "en")[1] == ["Agree", "Disagree", "Abstain"]
    assert V.clean_vote("q", ["A", "B", "a"], 0, "th")[1:] == (["A", "B"], 1)
    assert V.clean_vote("q", ["A", "B"], 999, "th")[2] == 60
    assert V.clean_vote("", ["A", "B"], 5, "th")["error"] == "need_question"
    assert V.clean_vote("q", ["A"], 5, "th")["error"] == "need_at_least_2_options"
    assert V.clean_vote("q", [str(i) for i in range(7)], 5, "th")["error"] == "too_many_options"
    assert V.clean_vote("q", ["A", "x" * 41], 5, "th")["error"] == "too_long"


def mk(voters=(1, 2, 3, 4, 5), options=None, anonymous=False, lang="th"):
    return V.new_vote(
        guild_id=GUILD, channel_id=70, room="ห้องประชุม", question="เลื่อนเปิดตัวไหม",
        options=options or list(V.DEFAULT_OPTIONS[lang]), voters=list(voters), opened_by=1,
        anonymous=anonymous, lang=lang, now=NOW,
    )  # fmt: skip


def test_one_ballot_per_person_and_changing_it():
    v = mk()
    assert V.cast(v, 1, 0, NOW) == "ok" and V.cast(v, 1, 1, NOW) == "ok"
    assert V.counts(v) == [0, 1, 0] and len(v["ballots"]) == 1
    assert V.cast(v, 99, 0, NOW) == "not_voter" and V.cast(v, 2, 7, NOW) == "bad_option"
    v["status"] = "closed"
    assert V.cast(v, 2, 0, NOW) == "closed"


def test_majority_tie_and_custom_options():
    v = mk()
    for u, i in [(1, 0), (2, 0), (3, 0), (4, 0), (5, 1)]:
        V.cast(v, u, i, NOW)
    assert V.outcome(v)["verdict"] == "pass"
    line = V.result_line(v, "th")
    assert line.startswith("✅ มติ:") and "เห็นด้วย 4 / ไม่เห็นด้วย 1 / งดออกเสียง 0" in line
    assert "ผ่าน" in line and "เสียงข้างมาก" in line  # the rule is part of the line
    V.cast(v, 4, 1, NOW)
    assert "3 / ไม่เห็นด้วย 2" in V.result_line(v, "th")
    t2 = mk(voters=(1, 2))
    V.cast(t2, 1, 0, NOW), V.cast(t2, 2, 1, NOW)
    assert V.outcome(t2)["verdict"] == "tie" and "ไม่ได้ข้อสรุป" in V.result_line(t2, "th")
    abst = mk(voters=(1, 2, 3))
    for u in (1, 2, 3):
        V.cast(abst, u, 2, NOW)  # only abstentions: nobody for or against
    assert V.outcome(abst)["verdict"] == "tie"
    assert V.outcome(mk())["verdict"] == "none"
    c = mk(options=["A", "B", "C"], voters=(1, 2, 3))
    V.cast(c, 1, 2, NOW), V.cast(c, 2, 2, NOW), V.cast(c, 3, 0, NOW)
    assert V.outcome(c)["winner"] == 2 and "C wins" in V.result_line(c, "en")
    assert V.result_line(mk(lang="en"), "en").startswith("🗳️ Vote:")


def test_named_vs_anonymous_results():
    names = {1: "Ann", 2: "Bob", 3: "Cat", 4: "Dan", 5: "Eve"}
    for anon in (False, True):
        v = mk(anonymous=anon)
        V.cast(v, 1, 0, NOW), V.cast(v, 2, 1, NOW)
        text = V.result_text(v, names, "th")
        assert ("Ann" in text) is (not anon) and ("Bob" in text) is (not anon)
        assert ("Cat" in text) is (not anon)  # named: also who did not vote
        assert ("ไม่เปิดเผยชื่อ" in text) is anon


def test_tally_text_counts_only():
    v = mk()
    V.cast(v, 1, 0, NOW), V.cast(v, 2, 0, NOW), V.cast(v, 3, 1, NOW)
    s = V.tally_text(v, "th", TZ)
    assert "โหวตแล้ว 3/5" in s and "เห็นด้วย 2 · ไม่เห็นด้วย 1 · งดออกเสียง 0" in s
    assert "09:05" in s and "1" in s


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    from fake_firestore import FakeFirestore

    return FirestoreDocStore(client=FakeFirestore())


def test_repo_both_backends(docs):
    repo = V.VoteRepo(docs)
    a, b = mk(), mk()
    b["question"] = "ไปกินข้าวร้านไหน"
    b["opened_at"] = V.iso(NOW + timedelta(minutes=1))
    repo.put(a), repo.put(b)
    assert repo.get(a["id"])["question"] == "เลื่อนเปิดตัวไหม" and repo.get("zzz") is None
    assert repo.find(GUILD, "last")[0]["id"] == b["id"]
    assert repo.find(GUILD, "เลื่อน")[0]["id"] == a["id"]
    assert repo.find(GUILD, a["id"])[0]["id"] == a["id"]
    b["status"] = "closed"
    repo.put(b)
    assert [r["id"] for r in repo.open_votes(GUILD)] == [a["id"]]
    assert repo.find(GUILD, "last")[0]["id"] == a["id"]  # an open one first
    off = V.OfferRepo(docs)
    o = V.new_offer(guild_id=GUILD, channel_id=1, question="q", options=[], now=NOW)
    off.put(o)
    assert off.get(o["id"])["status"] == "pending"


# ------------------------------------------------------------------ the detector
TRIGGER = [
    "โหวตกันไหม",
    "เอาไงดี ประชุมพรุ่งนี้ไหม",
    "ใครเห็นด้วยบ้าง",
    "ไปกินร้าน A หรือ ร้าน B ดี",
    "shall we vote on moving the launch?",
    "who's in favour of moving to Friday?",
]
QUIET = [
    "ประชุมกี่โมง",
    "เห็นด้วยครับ",
    "วันนี้อากาศดี",
    "I voted A yesterday",
    "ขอบคุณที่โหวตให้นะ",
    "what time is the meeting?",
]


@pytest.mark.parametrize("text", TRIGGER)
def test_detector_triggers(text):
    assert V.looks_like_vote_call(text)


@pytest.mark.parametrize("text", QUIET)
def test_detector_stays_quiet(text):
    assert not V.looks_like_vote_call(text)


# ------------------------------------------------------------------ fakes
class PM:
    def __init__(self, chan, mid):
        self.chan, self.id = chan, mid

    async def edit(self, **kw):
        self.chan.edits.append((self.id, kw))


class VChan:
    ids = itertools.count(5000)

    def __init__(self, id, name="room"):
        self.id, self.name, self.sent, self.edits = id, name, [], []
        self.mention = f"<#{id}>"

    async def send(self, content=None, view=None, **kw):
        m = SimpleNamespace(id=next(self.ids), channel=self, content=content, view=view)
        self.sent.append(m)
        return m

    def get_partial_message(self, mid):
        return PM(self, mid)

    def permissions_for(self, m):
        return SimpleNamespace(view_channel=True, read_messages=True, send_messages=True)


class Resp:
    def __init__(self):
        self.sent, self.edited = [], []

    async def send_message(self, text, **kw):
        self.sent.append((text, kw))

    async def edit_message(self, **kw):
        self.edited.append(kw)


def press(user, guild_id=GUILD):
    return SimpleNamespace(user=user, guild_id=guild_id, response=Resp(), channel=None)


class FakeLive:
    def __init__(self):
        self.cues = []

    async def announce_cue(self, cue, tag="cue"):
        self.cues.append(cue)


@pytest.fixture
def vw(world, bot, monkeypatch):
    monkeypatch.setattr(bot.team, "license_gate", None)  # other suites may leave a gate behind
    g = world.g
    text = VChan(70, "meeting-chat")
    text.guild = g  # a real channel knows its server, so Salid reads that server's language
    g.text_channels.append(text)
    room = SimpleNamespace(
        id=80, name="ห้องประชุม", guild=g, members=[world.staff, world.peer, world.intern]
    )
    for m in (world.staff, world.peer, world.intern):
        m.voice = SimpleNamespace(channel=room)
    private: dict[int, VChan] = {}

    async def router(member, body, files, view):
        ch = private.setdefault(member.id, VChan(9000 + member.id, f"private-{member.id}"))
        sent = await ch.send(body, view=view)
        return DMResult(True, message=sent)

    monkeypatch.setattr(dm_module, "ROUTER", router)
    monkeypatch.setattr(dm_module, "SENT_HOOK", None)
    monkeypatch.setattr(SecretaryBot, "get_guild", lambda s, i: g if i == GUILD else None)
    monkeypatch.setattr(
        SecretaryBot,
        "get_channel",
        lambda s, i: next((c for c in private.values() if c.id == i), None),
    )
    session = SimpleNamespace(
        guild_id=GUILD, voice_channel=room, text_channel=text, lang="th", votes_taken=[],
        live=FakeLive(), recording="recording", ending=False,
    )  # fmt: skip
    bot.meetings.sessions[(GUILD, 80)] = session
    return SimpleNamespace(
        g=g, w=world, text=text, room=room, private=private, session=session, bot=bot, last_id=""
    )


async def settle(bot):
    while bot._vote_tasks:
        await asyncio.gather(*list(bot._vote_tasks))


async def open_vote(vw, who=None, **kw):
    who = who or vw.w.staff
    msg = SimpleNamespace(guild=vw.g, channel=vw.text, author=who)
    args = dict(question="เลื่อนเปิดตัวไปเดือนหน้าไหม", options=None, minutes=5, anonymous=False,
                voters=None)  # fmt: skip
    args.update(kw)
    res = await vw.bot._call_vote(
        msg, vw.g, who, vw.text, args["question"], args["options"], args["minutes"],
        args["anonymous"], args["voters"], "th",
    )  # fmt: skip
    await settle(vw.bot)
    if "vote" in res:
        vw.last_id = res["vote"]
    return res


def the_vote(vw):
    return vw.bot.vote_repo.get(vw.last_id)


async def vote(vw, who, index):
    vid = the_vote(vw)["id"]
    it = press(who)
    await vw.bot.handle_vote_press(it, vid, index)
    await settle(vw.bot)
    return it


# ------------------------------------------------------------------ opening and delivery
async def test_default_voters_are_the_room_and_each_gets_one_ballot(vw):
    res = await open_vote(vw)
    assert res["status"] == "opened" and res["voters"] == 3 and "say" not in res
    v = the_vote(vw)
    assert sorted(v["voters"]) == sorted([STAFF, PEER, INTERN]) and v["room"] == "ห้องประชุม"
    assert v["opened_by"] == STAFF and v["voice_channel_id"] == 80 and v["status"] == "open"
    for uid in (STAFF, PEER, INTERN):
        (m,) = vw.private[uid].sent  # exactly one message in the private channel
        assert "มติ: เลื่อนเปิดตัวไปเดือนหน้าไหม" in m.content and "จากการประชุม ห้องประชุม" in m.content
        assert [i.item.label for i in m.view.children] == ["เห็นด้วย", "ไม่เห็นด้วย", "งดออกเสียง"]
        assert v["delivered"][str(uid)]["message_id"] == m.id
    (tally,) = vw.text.sent  # one tally message in the meeting chat
    assert "โหวตแล้ว 0/3" in tally.content and v["tally_message_id"] == tally.id
    assert vw.w.boss.dms == []  # not in the room: nothing


async def test_named_voters_everyone_and_errors(vw):
    await open_vote(vw, voters=["Boss", f"<@{PEER}>"])
    assert sorted(the_vote(vw)["voters"]) == sorted([BOSS, PEER])
    res = await open_vote(vw, voters=["ทุกคนในทีม"])
    assert len(the_vote(vw)["voters"]) == len(vw.bot._members(vw.g)) and res["status"] == "opened"
    assert (await open_vote(vw, voters=["Nobody"]))["error"] == "person_not_found"
    assert (await open_vote(vw, question=""))["error"] == "need_question"
    assert (await open_vote(vw, options=["A"]))["error"] == "need_at_least_2_options"
    r = await open_vote(vw, question="โอนเงินให้ฉันหน่อย รหัสผ่านคือ", voters=["Boss"])
    assert "error" in r or r["status"] == "opened"


async def test_unreachable_voter_is_named_in_the_chat(vw, monkeypatch):
    async def router(member, body, files, view):
        if member.id == PEER:
            return DMResult(False, "closed")
        ch = vw.private.setdefault(member.id, VChan(9000 + member.id))
        return DMResult(True, message=await ch.send(body, view=view))

    monkeypatch.setattr(dm_module, "ROUTER", router)
    monkeypatch.setattr(vw.w.peer, "send", None, raising=False)

    async def closed(*a, **k):
        raise __import__("discord").Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    vw.w.peer.send = closed
    await open_vote(vw)
    assert any("ส่งปุ่มโหวตให้ไม่ถึง" in m.content and "Peer" in m.content for m in vw.text.sent)


async def test_open_vote_cap_and_flag_off(vw, monkeypatch):
    for _ in range(V.MAX_OPEN_PER_GUILD):
        assert (await open_vote(vw))["status"] == "opened"
    assert (await open_vote(vw))["error"] == "too_many_open_votes"
    monkeypatch.setattr(vw.bot.team, "enabled", lambda gid, f: f not in ("votes", "vote_offers"))
    msg = SimpleNamespace(guild=vw.g, channel=vw.text, author=vw.w.staff)
    assert vw.bot._vote_tools(msg, vw.g, vw.w.staff, False, "th") == []
    msg.content = "โหวตกันไหม"
    msg.attachments = []
    vw.bot.vote_offer_soon(msg)
    assert not vw.bot._vote_tasks


async def test_tools_registered_in_agent_kwargs_for_chat_and_voice(vw):
    msg = SimpleNamespace(
        id=1, guild=vw.g, channel=vw.text, author=vw.w.staff, content="", attachments=[],
        reference=None,
    )  # fmt: skip
    kw = vw.bot._agent_kwargs(
        msg, vw.g, vw.w.staff, False, setting="th", lang="th", question="q", memory=None,
        profile=None,
    )  # fmt: skip
    assert {"call_vote", "close_vote", "vote_status"} <= {f.__name__ for f in kw["extra_tools"]}
    from salid.voice_actions import ACTION_TOOL_NAMES

    assert {"call_vote", "close_vote"} <= ACTION_TOOL_NAMES


# ------------------------------------------------------------------ pressing buttons
async def test_press_records_changes_and_edits_the_ballot(vw):
    await open_vote(vw)
    it = await vote(vw, vw.w.staff, 0)
    (edit,) = it.response.edited
    assert "✅ คุณเลือก: เห็นด้วย" in edit["content"] and "เปลี่ยนได้จนถึง" in edit["content"]
    assert the_vote(vw)["ballots"][str(STAFF)]["option"] == 0
    it = await vote(vw, vw.w.staff, 1)  # changing it is allowed until the close
    assert "ไม่เห็นด้วย" in it.response.edited[0]["content"]
    v = the_vote(vw)
    assert V.counts(v) == [0, 1, 0] and len(v["ballots"]) == 1
    # the tally message in the meeting chat was edited, counts only
    last = vw.text.edits[-1][1]["content"]
    assert "โหวตแล้ว 1/3" in last and "Somchai" not in last


async def test_press_by_a_non_voter_or_unknown_vote_is_ephemeral(vw):
    await open_vote(vw)
    it = press(vw.w.boss)
    await vw.bot.handle_vote_press(it, the_vote(vw)["id"], 0)
    assert it.response.sent[0][1]["ephemeral"] and "ไม่ได้อยู่ในรายชื่อ" in it.response.sent[0][0]
    assert the_vote(vw)["ballots"] == {}
    it = press(vw.w.staff)
    await vw.bot.handle_vote_press(it, "deadbeef0000", 0)
    assert it.response.sent[0][1]["ephemeral"]


def test_button_custom_id_round_trip():
    b = VoteButton("0123456789ab", 2, "งดออกเสียง")
    assert b.item.custom_id == "salid:vote:0123456789ab:2"
    assert VoteButton.__discord_ui_compiled_template__.fullmatch(b.item.custom_id)
    assert VoteOfferButton.__discord_ui_compiled_template__.fullmatch("salid:voff:go:0123456789ab")


# ------------------------------------------------------------------ closing
async def test_closes_when_everyone_voted_and_records_the_minutes(vw):
    await open_vote(vw)
    await vote(vw, vw.w.staff, 0)
    await vote(vw, vw.w.peer, 0)
    assert the_vote(vw)["status"] == "open"
    await vote(vw, vw.w.intern, 1)
    v = the_vote(vw)
    assert v["status"] == "closed" and v["result_message_id"]
    result = vw.text.sent[-1].content
    assert result.startswith("✅ มติ: เลื่อนเปิดตัวไปเดือนหน้าไหม → เห็นด้วย 2 / ไม่เห็นด้วย 1")
    assert "ผ่าน" in result and "Somchai" in result  # named vote: names after the close
    assert "ปิดแล้ว" in vw.text.edits[-1][1]["content"]  # the tally message was closed too
    assert vw.session.votes_taken and "เห็นด้วย 2" in vw.session.votes_taken[0]
    block = MeetingManager.votes_block(vw.session)
    assert "🗳️ มติที่ลงคะแนน" in block and "- ✅ มติ:" in block
    # ballots lose their buttons and say what the voter chose
    closed = [e for e in vw.private[STAFF].edits if e[1].get("view", 1) is None]
    assert closed and "คุณเลือก: เห็นด้วย" in closed[0][1]["content"]
    # a late press is refused
    it = press(vw.w.staff)
    await vw.bot.handle_vote_press(it, v["id"], 1)
    assert "ปิดแล้ว" in it.response.sent[0][0]
    # closing twice does nothing
    assert await vw.bot._vote_close(vw.g, v["id"]) is None


async def test_anonymous_result_has_no_names(vw):
    await open_vote(vw, anonymous=True)
    for u, i in [(vw.w.staff, 0), (vw.w.peer, 1), (vw.w.intern, 0)]:
        await vote(vw, u, i)
    result = vw.text.sent[-1].content
    assert "ไม่เปิดเผยชื่อ" in result
    assert not any(n in result for n in ("Somchai", "Peer", "Ploy"))
    st = await vw.bot._vote_status(vw.g, vw.w.staff, False, "last")
    assert "Somchai" not in str(st) and st["anonymous"] is True


async def test_closes_at_the_deadline(vw):
    await open_vote(vw)
    await vote(vw, vw.w.staff, 0)
    v = the_vote(vw)
    await vw.bot._votes_due()
    assert the_vote(vw)["status"] == "open"  # not due yet
    v["closes_at"] = V.iso(vw.bot.s.now() - timedelta(seconds=1))
    vw.bot.vote_repo.put(v)
    vw.bot._open_votes[v["id"]] = (GUILD, datetime.fromisoformat(v["closes_at"]))
    vw.bot._votes_last = float("-inf")
    await vw.bot._votes_due()
    await settle(vw.bot)
    assert the_vote(vw)["status"] == "closed"
    assert "เห็นด้วย 1" in vw.text.sent[-1].content and "ไม่ได้ลงคะแนน" in vw.text.sent[-1].content


async def test_open_votes_are_found_again_after_a_restart(vw):
    await open_vote(vw)
    v = the_vote(vw)
    v["closes_at"] = V.iso(vw.bot.s.now() - timedelta(minutes=1))
    vw.bot.vote_repo.put(v)
    vw.bot._open_votes.clear()
    vw.bot._votes_scanned = False
    await vw.bot._votes_due()
    assert the_vote(vw)["status"] == "closed"


async def test_manual_close_only_by_the_opener_or_an_executive(vw):
    await open_vote(vw)
    r = await vw.bot._close_vote_tool(vw.g, vw.w.peer, False, "last")
    assert r["error"] == "not_allowed" and the_vote(vw)["status"] == "open"
    r = await vw.bot._close_vote_tool(vw.g, vw.w.staff, False, "เลื่อน")
    assert r["status"] == "closed" and "ไม่มีผู้ลงคะแนน" in r["result"]
    await open_vote(vw)
    r = await vw.bot._close_vote_tool(vw.g, vw.w.boss, True, "last")  # an executive
    assert r["status"] == "closed"
    assert (await vw.bot._close_vote_tool(vw.g, vw.w.staff, False, "last"))[
        "error"
    ] == "vote_not_found"


async def test_vote_status_scope(vw):
    await open_vote(vw)
    await vote(vw, vw.w.staff, 0)
    st = await vw.bot._vote_status(vw.g, vw.w.peer, False, "last")
    assert st["voted"] == 1 and st["voters"] == 3 and "result" not in st
    assert "error" in await vw.bot._vote_status(vw.g, vw.w.boss, False, "last")  # not involved
    assert "voted" in await vw.bot._vote_status(vw.g, vw.w.boss, True, "last")  # executive


# ------------------------------------------------------------------ voice
async def test_voice_result_is_announced_and_chat_has_the_line(vw):
    res = await open_vote(vw)
    assert "say" not in res  # text only: the tally message tells everyone; returned at once
    from salid.live_voice import BACKGROUND_TOOLS

    assert "call_vote" not in BACKGROUND_TOOLS
    for u, i in [(vw.w.staff, 0), (vw.w.peer, 0), (vw.w.intern, 1)]:
        await vote(vw, u, i)
    (cue,) = vw.session.live.cues
    assert "เห็นด้วย 2 / ไม่เห็นด้วย 1" in cue and "ผ่าน" in cue
    assert "Somchai" not in cue and "✅" not in cue  # spoken: no names, no emoji
    assert vw.text.sent[-1].content.startswith("✅ มติ:")


async def test_announce_cue_waits_for_a_pause_and_sends_the_cue(lv_env):
    lv, live = lv_env
    await live.announce_cue("(cue text)", "vote x")
    assert any("cue text" in str(c) for c in lv.sent)


@pytest.fixture
def lv_env(monkeypatch):
    from salid import live_voice as lvm

    sent = []

    class FakeSession:
        async def send_client_content(self, turns, turn_complete):
            sent.append(turns.parts[0].text)

    obj = lvm.LiveVoice.__new__(lvm.LiveVoice)
    obj._live, obj._closing = FakeSession(), False
    obj.session = SimpleNamespace(ending=False)
    obj._announce_until = 0.0
    obj._last_engaged_at = 0.0

    async def pause():
        return True

    obj._wait_for_pause = pause
    return SimpleNamespace(sent=sent), obj


# ------------------------------------------------------------------ offers
def offer_llm(monkeypatch, question="ประชุมพรุ่งนี้ไหม", options=()):
    calls = []

    def fake(client, model, history, text):
        calls.append(text)
        return {"question": question, "options": list(options)}

    monkeypatch.setattr(V, "extract_offer", fake)
    return calls


def talk(vw, text, who=None, chan=None):
    return SimpleNamespace(
        guild=vw.g, channel=chan or vw.text, author=who or vw.w.staff, content=text,
        attachments=[], id=1,
    )  # fmt: skip


async def test_offer_once_and_rate_limited_per_channel(vw, monkeypatch):
    calls = offer_llm(monkeypatch)
    vw.bot.vote_offer_soon(talk(vw, "เอาไงดี ประชุมพรุ่งนี้ไหม"))
    await settle(vw.bot)
    (offer,) = vw.text.sent
    assert "ให้สลิดเปิดโหวตเรื่อง “ประชุมพรุ่งนี้ไหม” ไหมคะ" in offer.content
    assert [c.item.label for c in offer.view.children] == ["เปิดโหวต", "ไม่ต้อง"]
    for text in ("โหวตกันไหม", "ใครเห็นด้วยบ้าง"):  # within 10 minutes: no second offer, no LLM
        vw.bot.vote_offer_soon(talk(vw, text))
        await settle(vw.bot)
    assert len(vw.text.sent) == 1 and len(calls) == 1
    vw.bot.clock.advance(601)  # later, same question: never twice
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    assert len(vw.text.sent) == 1
    offer_llm(monkeypatch, question="กินอะไรดี")  # a different question: offered again
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    assert len(vw.text.sent) == 2


async def test_offer_needs_a_detector_hit_and_a_gate_yes_and_respects_snooze(vw, monkeypatch):
    calls = offer_llm(monkeypatch)
    vw.bot.vote_offer_soon(talk(vw, "ประชุมกี่โมง"))
    assert not vw.bot._vote_tasks and not calls  # no LLM spend without the free pre-filter
    monkeypatch.setattr(V, "extract_offer", lambda *a: None)  # the gate says no
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    assert vw.text.sent == []
    calls = offer_llm(monkeypatch)
    vw.bot._snoozed[vw.text.id] = vw.bot._clock() + 1000  # "@Salid เงียบก่อน"
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    assert not vw.bot._vote_tasks and not calls


async def test_route_hook_only_in_listen_and_join_untagged(vw, monkeypatch):
    seen = []
    monkeypatch.setattr(vw.bot, "vote_offer_soon", lambda m: seen.append(m.content))
    m = talk(vw, "โหวตกันไหม")

    async def nothing(*a, **k):
        return None

    monkeypatch.setattr(vw.bot, "_call_or_join", nothing)
    monkeypatch.setattr(vw.bot, "_answer", nothing)
    m.reference, m.mentions, m.reactions = None, [], []
    for mode in ("listen", "join", "call", "chat"):
        await vw.bot._route(m, mode, False, False, False, False)
    assert seen == ["โหวตกันไหม", "โหวตกันไหม"]


async def test_no_silences_offers_for_an_hour(vw, monkeypatch):
    offer_llm(monkeypatch)
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    oid = vw.bot.offer_repo.docs.find(V.OFFERS, GUILD)[0]["id"]
    it = press(vw.w.staff)
    await vw.bot.handle_vote_offer(it, "no", oid)
    assert "ไม่เปิดโหวต" in it.response.edited[0]["content"]
    assert vw.bot.offer_repo.get(oid)["status"] == "declined" and the_vote_count(vw) == 0
    offer_llm(monkeypatch, question="เรื่องอื่น")
    vw.bot.clock.advance(1200)  # past the 10 minute gap but inside the hour
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    assert not vw.bot._vote_tasks
    vw.bot.clock.advance(3600)
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    assert len(vw.text.sent) == 2
    it = press(vw.w.staff)
    await vw.bot.handle_vote_offer(it, "go", oid)  # an answered offer cannot be pressed again
    assert it.response.sent[0][1]["ephemeral"] and the_vote_count(vw) == 0


def the_vote_count(vw):
    return len(vw.bot.vote_repo.all(GUILD))


async def test_go_opens_the_vote_with_the_recent_authors_and_the_presser_as_opener(vw, monkeypatch):
    from test_coordinator import add_msg, at

    offer_llm(monkeypatch)
    vw.g.members[BOSS].display_name = "Boss"
    now = vw.bot.s.now()
    add_msg(vw.bot, 1, STAFF, "Somchai", "a", 70, when=now - timedelta(hours=2))
    add_msg(vw.bot, 2, PEER, "Peer", "b", 70, when=now - timedelta(hours=30))  # too old
    add_msg(vw.bot, 3, INTERN, "Ploy", "c", 70, when=now - timedelta(hours=1))
    add_msg(vw.bot, 4, 777, "Gone", "d", 70, when=now - timedelta(hours=1))  # left the server
    add_msg(vw.bot, 5, STAFF, "Somchai", "e", 71, when=now - timedelta(hours=1))  # other channel
    vw.bot.vote_offer_soon(talk(vw, "โหวตกันไหม"))
    await settle(vw.bot)
    oid = vw.bot.offer_repo.docs.find(V.OFFERS, GUILD)[0]["id"]
    it = press(vw.w.boss)
    await vw.bot.handle_vote_offer(it, "go", oid)
    await settle(vw.bot)
    assert "เปิดโหวตเรื่อง" in it.response.edited[0]["content"]
    vw.last_id = vw.bot.vote_repo.all(GUILD)[0]["id"]
    v = the_vote(vw)
    assert v["opened_by"] == BOSS and v["question"] == "ประชุมพรุ่งนี้ไหม"
    assert sorted(v["voters"]) == sorted([STAFF, INTERN, BOSS]) and PEER not in v["voters"]
    assert v["options"] == ["เห็นด้วย", "ไม่เห็นด้วย", "งดออกเสียง"] and not v["anonymous"]
    assert set(vw.private) == {STAFF, INTERN, BOSS}  # ballots in their private channels
    assert vw.bot.offer_repo.get(oid)["status"] == "opened"


async def test_voters_capped_at_thirty(vw):
    from test_coordinator import person

    now = vw.bot.s.now()
    for i in range(40):
        person(vw.g, 5000 + i, f"p{i}")
        from test_coordinator import add_msg

        add_msg(vw.bot, 100 + i, 5000 + i, f"p{i}", "x", 70, when=now - timedelta(minutes=i))
    people = await vw.bot._offer_voters(vw.g, vw.text, vw.w.boss)
    assert len(people) == 30 and people[0].id == BOSS
    assert 5000 in {p.id for p in people} and 5039 not in {p.id for p in people}


async def test_voice_room_offer_is_text_only_in_the_meeting_chat(vw, monkeypatch):
    offer_llm(monkeypatch, question="เลื่อนประชุมไหม", options=["เลื่อน", "ไม่เลื่อน"])
    await vw.bot.vote_offer_from_room(vw.session, "Somchai", "ใครเห็นด้วยบ้าง ว่าจะเลื่อน")
    (offer,) = vw.text.sent
    assert "เลื่อนประชุมไหม" in offer.content and "เลื่อน / ไม่เลื่อน" in offer.content
    assert vw.session.live.cues == []  # never spoken
    await vw.bot.vote_offer_from_room(vw.session, "Somchai", "โหวตกันไหม")
    assert len(vw.text.sent) == 1  # rate limited
    oid = vw.bot.offer_repo.docs.find(V.OFFERS, GUILD)[0]["id"]
    it = press(vw.w.peer)
    await vw.bot.handle_vote_offer(it, "go", oid)
    await settle(vw.bot)
    vw.last_id = vw.bot.vote_repo.all(GUILD)[0]["id"]
    v = the_vote(vw)
    assert (
        v["opened_by"] == PEER
        and v["voice_channel_id"] == 80
        and v["options"] == ["เลื่อน", "ไม่เลื่อน"]
    )
    assert sorted(v["voters"]) == sorted([STAFF, PEER, INTERN])  # the people in the room


async def test_voice_offer_flag_off_posts_nothing(vw, monkeypatch):
    offer_llm(monkeypatch)
    monkeypatch.setattr(vw.bot.team, "enabled", lambda gid, f: f != "vote_offers")
    await vw.bot.vote_offer_from_room(vw.session, "Somchai", "โหวตกันไหม")
    assert vw.text.sent == []


def test_meeting_hook_calls_the_offer_for_untagged_lines():
    import inspect

    src = inspect.getsource(MeetingManager.handle_segment)
    assert "vote_offer_from_room" in src
