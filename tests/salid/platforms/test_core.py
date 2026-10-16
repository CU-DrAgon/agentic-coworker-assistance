"""Addressing, tenancy, executives, replies, rate limit, acknowledgement: the platform-neutral core."""

import asyncio
import threading
from functools import partial

import pytest
from conftest import ALICE, BOB, FixedClockSettings, mk, scripted, text_resp
from platform_fakes import (
    NOW,
    Clock,
    FakeAsk,
    FakePort,
    gid_of,
    immediately,
    inc,
    make_assistant,
    rid_of,
)

from salid.guild_settings import GuildSettingsUpdate
from salid.platforms import ids
from salid.qa import ask


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


async def run(a, port, **kw):
    msg = inc(**kw)
    await a.handle(msg, port)
    await a.drain()
    return msg


# ------------------------------------------------------------------ addressing


async def test_direct_message_is_always_answered(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, is_direct=True, text="what is 2+2")
    assert a.ask_fn.calls[0][0] == "what is 2+2" and port.texts == ["ok"]


async def test_channel_chatter_is_recorded_but_never_answered(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    msg = await run(a, port, text="lunch at noon?")
    assert not a.ask_fn.calls and not port.delivered
    from platform_fakes import stored

    assert [m.text for m in stored(a, msg)] == ["lunch at noon?"]


@pytest.mark.parametrize(
    "kw",
    [
        {"mentions_bot": True, "text": "status?"},
        {"text": "สลิด ช่วยดูหน่อย"},
        {"text": "hey Salid what is the total"},
        {"text": "and then?", "thread": "99.9", "replies_to_bot": True},
    ],
)
async def test_channel_message_that_addresses_salid_is_answered(s, tmp_path, kw):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, **kw)
    assert len(a.ask_fn.calls) == 1 and port.texts == ["ok"]


async def test_tagging_someone_else_is_not_for_salid(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, text="@Mew can you check", mentions_others=True)
    assert not a.ask_fn.calls


async def test_conversation_window_follows_the_same_person_via_the_gate(s, tmp_path):
    clock = Clock()
    a, port = make_assistant(s, tmp_path, clock=clock), FakePort()
    await run(a, port, mentions_bot=True, text="total?", message="1.0")
    clock.advance(30)
    await run(a, port, text="and last month?", message="2.0")  # same user, gate says yes
    assert len(a.ask_fn.calls) == 2 and len(a.gate_calls) == 1
    await run(a, port, text="and bob's?", message="3.0", user="U2", user_name="Bob")
    assert len(a.ask_fn.calls) == 2  # another person: not in Alice's window, gate not asked
    assert len(a.gate_calls) == 1


async def test_window_closes_and_gate_can_say_no(s, tmp_path):
    clock = Clock()
    a, port = make_assistant(s, tmp_path, clock=clock, gate=lambda *x, **k: False), FakePort()
    await run(a, port, mentions_bot=True, text="total?", message="1.0")
    clock.advance(10)
    await run(a, port, text="lol ok", message="2.0")  # gate says it is not for Salid
    assert len(a.ask_fn.calls) == 1
    clock.advance(1000)
    await run(a, port, text="still there?", message="3.0")  # window expired: gate not asked
    assert len(a.ask_fn.calls) == 1


async def test_thanks_in_the_window_is_not_answered(s, tmp_path):
    clock = Clock()
    a, port = make_assistant(s, tmp_path, clock=clock), FakePort()
    await run(a, port, mentions_bot=True, text="total?", message="1.0")
    await run(a, port, text="thanks!", message="2.0")
    assert len(a.ask_fn.calls) == 1 and not a.gate_calls


async def test_salid_thread_stays_open_after_it_answered_there(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, mentions_bot=True, text="total?", message="5.0")  # answered in thread 5.0
    await run(a, port, text="more detail", message="6.0", thread="5.0", user="U2")
    assert len(a.ask_fn.calls) == 2
    await run(a, port, text="more detail", message="7.0", thread="other", user="U3")
    assert len(a.ask_fn.calls) == 2


async def test_channel_modes(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    probe = inc()
    gid, rid = gid_of(probe), rid_of(probe)
    from platform_fakes import stored

    a.team.store.save_settings(gid, GuildSettingsUpdate(channel_modes={str(rid): "listen"}), "t")
    a.team._cache.clear()
    await run(a, port, text="Salid hello", message="1.0")  # a bare name does not call in listen
    assert not a.ask_fn.calls
    await run(a, port, mentions_bot=True, text="hello", message="2.0")
    assert len(a.ask_fn.calls) == 1
    a.team.store.save_settings(gid, GuildSettingsUpdate(channel_modes={str(rid): "chat"}), "t")
    a.team._cache.clear()
    await run(a, port, text="no tag at all", message="3.0")
    assert len(a.ask_fn.calls) == 2
    a.team.store.save_settings(gid, GuildSettingsUpdate(channel_modes={str(rid): "off"}), "t")
    a.team._cache.clear()
    msg = await run(a, port, text="invisible", message="4.0", mentions_bot=True)
    assert len(a.ask_fn.calls) == 2
    assert "invisible" not in [m.text for m in stored(a, msg)]  # off: not even remembered


async def test_the_same_event_twice_is_handled_once(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    for _ in range(3):
        await run(a, port, mentions_bot=True, text="hi", message="9.9")
    assert len(a.ask_fn.calls) == 1


async def test_empty_mention_uses_the_previous_message_or_greets(s, tmp_path):
    (tmp_path / "b").mkdir()
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, text="what did we spend", message="1.0")
    await run(a, port, mentions_bot=True, text="", message="2.0")
    assert a.ask_fn.calls[0][0] == "what did we spend"
    a2, p2 = make_assistant(s, tmp_path / "b"), FakePort()
    await run(a2, p2, mentions_bot=True, text="", message="3.0", room="C9", is_direct=False)
    assert not a2.ask_fn.calls and p2.texts  # a greeting, not a model call


async def test_answers_per_room_are_rate_limited(s, tmp_path):
    a, port = make_assistant(s, tmp_path, platform_answers_per_minute=3), FakePort()
    for i in range(6):
        await run(a, port, mentions_bot=True, text=f"q{i}", message=f"{i}.0")
    assert len(a.ask_fn.calls) == 3
    await run(a, port, mentions_bot=True, text="q", message="7.0", room="C2")  # another room
    assert len(a.ask_fn.calls) == 4


# ------------------------------------------------------------------ tenancy + executives


async def test_every_call_uses_the_tenant_id_and_user_id(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    msg = await run(a, port, mentions_bot=True)
    _q, kw = a.ask_fn.calls[0]
    assert kw["guild_id"] == ids.tenant_id("slack", "T1") == gid_of(msg)
    assert kw["user_id"] == ids.user_id("slack", "U1")
    assert kw["guild_id"] >= ids.DISCORD_CEILING
    await run(a, port, mentions_bot=True, tenant="T2", message="2.0")
    assert a.ask_fn.calls[1][1]["guild_id"] != kw["guild_id"]


async def test_executives_come_from_env_lists_and_platform_admins(s, tmp_path):
    a = make_assistant(s, tmp_path, slack_exec_user_ids="U1", line_exec_user_ids="Uline")
    port = FakePort()
    await run(a, port, mentions_bot=True, message="1.0")
    await run(a, port, mentions_bot=True, message="2.0", user="U2")
    await run(a, FakePort(admin=True), mentions_bot=True, message="3.0", user="U3")
    await run(
        a,
        FakePort(platform="line"),
        mentions_bot=True,
        message="4.0",
        platform="line",
        user="Uline",
        tenant="Ubot",
    )
    await run(
        a,
        FakePort(platform="line"),
        mentions_bot=True,
        message="5.0",
        platform="line",
        user="U1",
        tenant="Ubot",
    )  # a Slack id on LINE: not an executive
    assert [c[1]["is_exec"] for c in a.ask_fn.calls] == [True, False, True, True, False]


async def test_admin_lookup_failure_is_not_an_executive(s, tmp_path):
    class Boom(FakePort):
        async def is_admin(self, msg):
            raise RuntimeError("slack down")

    a = make_assistant(s, tmp_path)
    await run(a, Boom(), mentions_bot=True)
    assert a.ask_fn.calls[0][1]["is_exec"] is False


async def test_non_executive_ledger_scope_end_to_end(s, tmp_path):
    """The real `ask`: a non-executive in a Slack workspace only sees their own receipts."""
    probe = inc()
    gid = gid_of(probe)
    me, other = ids.user_id("slack", "U1"), ids.user_id("slack", "U2")
    llm = scripted(
        _fc("spending_summary", start_date="2026-10-01", end_date="2026-10-31"),
        text_resp("done"),
    )
    a = make_assistant(s, tmp_path, ask=partial(ask, model=llm))
    mk(a.store, 1, me, "Alice", "Makro", "2026-10-01", 1000.0, guild=gid)
    mk(a.store, 2, other, "Bob", "Makro", "2026-10-02", 9000.0, guild=gid)
    mk(a.store, 3, ALICE, "Eve", "Makro", "2026-10-02", 500.0, guild=1000)  # a Discord guild
    await run(a, FakePort(), mentions_bot=True, text="how much did we spend")
    sent = _responses(llm)
    assert "1000.0" in sent and "9000" not in sent and "500.0" not in sent
    assert a.ask_fn is not None and BOB


def _responses(llm) -> str:
    """The tool results the model was shown."""
    return " ".join(
        str(p.function_response)
        for c in llm.requests[-1].contents
        for p in c.parts or []
        if p.function_response
    )


def _fc(name, **args):
    from conftest import fc_resp

    return fc_resp((name, args))


async def test_executive_sees_the_whole_workspace_but_not_other_tenants(s, tmp_path):
    probe = inc()
    gid = gid_of(probe)
    llm = scripted(
        _fc("spending_summary", start_date="2026-10-01", end_date="2026-10-31"),
        text_resp("done"),
    )
    a = make_assistant(s, tmp_path, ask=partial(ask, model=llm), slack_exec_user_ids="U1")
    mk(a.store, 1, ids.user_id("slack", "U1"), "A", "Makro", "2026-10-01", 1000.0, guild=gid)
    mk(a.store, 2, ids.user_id("slack", "U2"), "B", "Makro", "2026-10-02", 9000.0, guild=gid)
    mk(
        a.store,
        3,
        ids.user_id("slack", "U2"),
        "B",
        "Makro",
        "2026-10-02",
        7777.0,
        guild=ids.tenant_id("slack", "T2"),
    )
    await run(a, FakePort(), mentions_bot=True, text="how much did we spend")
    sent = _responses(llm)
    assert "10000.0" in sent and "7777" not in sent


# ------------------------------------------------------------------ features


async def test_qa_switch_off_answers_with_a_notice_only_when_addressed(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    msg = inc()
    a.team.store.save_settings(gid_of(msg), GuildSettingsUpdate(features={"qa": False}), "t")
    await run(a, port, mentions_bot=True, text="hi")
    assert not a.ask_fn.calls and len(port.texts) == 1
    assert "Q&A" in port.texts[0] or "ถามตอบ" in port.texts[0]


async def test_web_client_only_with_the_web_feature(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    a.llm = object()
    await run(a, port, mentions_bot=True, message="1.0")
    assert a.ask_fn.calls[0][1]["web_client"] is a.llm  # same behaviour as Discord's _web_kw
    msg = inc()
    a.team.store.save_settings(gid_of(msg), GuildSettingsUpdate(features={"web": False}), "t")
    a.team._cache.clear()
    await run(a, port, mentions_bot=True, message="2.0")
    assert "web_client" not in a.ask_fn.calls[1][1]


async def test_language_follows_the_message_for_an_unconfigured_tenant(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, mentions_bot=True, text="hello")
    assert a.ask_fn.calls[0][1]["language"] == "auto"


# ------------------------------------------------------------------ replies, failures, history


async def test_failed_model_call_apologises_without_leaking_details(s, tmp_path, caplog):
    a, port = make_assistant(s, tmp_path, ask=FakeAsk(fail=True)), FakePort()
    caplog.set_level("DEBUG")
    await run(a, port, mentions_bot=True, text="my card is 4111")
    assert len(port.texts) == 1 and "secret model failure" not in port.texts[0]
    assert "secret model failure" not in caplog.text and "4111" not in caplog.text


async def test_history_holds_both_people_and_salid(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    await run(a, port, text="we need 40 boxes", message="1.0", user="U2", user_name="Bob")
    await run(a, port, mentions_bot=True, text="how many?", message="2.0")
    hist = a.ask_fn.calls[0][1]["history"]
    assert "Bob: we need 40 boxes" in hist and "how many?" not in hist
    await run(a, port, mentions_bot=True, text="and now?", message="3.0")
    hist2 = a.ask_fn.calls[1][1]["history"]
    assert "สลิด: ok" in hist2 and "Alice: how many?" in hist2


async def test_history_is_read_from_the_platform_when_the_buffer_is_empty(s, tmp_path):
    from salid.platforms.core import HistLine

    a, port = make_assistant(s, tmp_path), FakePort()
    from datetime import timedelta

    port.hist = [HistLine(NOW - timedelta(minutes=5), "Mew", "launch is friday")]
    await run(a, port, mentions_bot=True, text="when is launch?")
    assert "Mew: launch is friday" in a.ask_fn.calls[0][1]["history"]


async def test_long_answers_are_capped_with_a_note(s, tmp_path):
    class Chunky(FakePort):
        max_messages = 3

        def format(self, text):
            return [f"part{i}" for i in range(8)]

    a, port = make_assistant(s, tmp_path), Chunky()
    await run(a, port, mentions_bot=True)
    assert len(port.delivered[0][1]) == 3 and port.delivered[0][1][-1].startswith("part2")
    assert len(port.delivered[0][1][-1]) > len("part2")


# ------------------------------------------------------------------ the slow acknowledgement


async def test_slow_answer_posts_one_moment_then_replaces_it(s, tmp_path):
    release = threading.Event()
    a = make_assistant(s, tmp_path, ask=FakeAsk(release=release), sleep=immediately)
    port = FakePort()
    task = asyncio.create_task(a.handle(inc(mentions_bot=True, text="hard one"), port))
    for _ in range(200):
        if port.acks:
            break
        await asyncio.sleep(0.01)
    assert len(port.acks) == 1  # posted because no answer was back yet
    release.set()
    await task
    msg, texts, ack = port.delivered[0]
    assert texts == ["ok"] and ack == ("ack", 1)  # handed to deliver, which replaces it


async def test_fast_answer_posts_no_acknowledgement(s, tmp_path):
    never = asyncio.Event()

    async def slow_sleep(_s):
        await never.wait()

    a, port = make_assistant(s, tmp_path, sleep=slow_sleep), FakePort()
    await run(a, port, mentions_bot=True)
    assert port.acks == [] and port.delivered[0][2] is None
    assert not any(not t.done() for t in a._bg)


async def test_tenant_id_owned_by_another_tenant_is_refused(s, tmp_path):
    """A tenant id is a hash: if its settings document already names a different owner
    (a collision), the message is dropped instead of sharing that tenant's ledger and memory."""
    a, port = make_assistant(s, tmp_path), FakePort()
    a.guilds_store = a.team.store
    gid = ids.tenant_id("slack", "T1")
    a.guilds_store.upsert_guild(gid, name="Slack TOTHER", icon_url="", channels=[])
    await run(a, port, is_direct=True, text="what did we spend?")
    assert not a.ask_fn.calls and not port.delivered


async def test_tenant_owner_is_recorded_and_accepted_next_time(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    a.guilds_store = a.team.store
    await run(a, port, is_direct=True, text="hi")
    await a.drain()
    gid = ids.tenant_id("slack", "T1")
    assert a.guilds_store.find(gid).guild_name == "Slack T1"
    b, port2 = make_assistant(s, tmp_path), FakePort()
    b.guilds_store = a.guilds_store
    await run(b, port2, is_direct=True, text="hi again")
    assert b.ask_fn.calls
