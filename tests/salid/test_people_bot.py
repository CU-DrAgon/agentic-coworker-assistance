# ruff: noqa: F811, F401
"""Company knowledge in the bot (`salid.people_bot`): the tools reach chat and voice, scoped by what
the asker can read; leaving (dossier left, open tasks, handover saved + wiki + PDF to the
executives); the nightly company map from public channels only; the welcome line; the flag.
Offline: fake Discord objects, SQLite, scripted models."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from conftest import GUILD, FakeClient, text_resp
from test_coordinator import (
    ADMIN_ID,
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    PUBLIC,
    SECRET,
    STAFF,
    at,
    person,
    world,
)
from test_modes import TMsg, bot  # noqa: F401  (fixture)
from test_private_chat import human, prep

from salid import company_brain as cb
from salid import people as P
from salid import qa
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.i18n import CATALOG, t
from salid.people_tools import KNOWLEDGE_PROMPT
from salid.tasks import Task

DAY = "2026-10-14"


def ev(cid=PUBLIC, mid=1):
    return P.Evidence(day=DAY, channel_id=cid, message_id=mid)


def learn(bot, uid, name, **kw):
    def apply(d):
        for k, (field, *args) in kw.items():
            getattr(P, k)(getattr(d, field), *args)
        return True

    bot.people.update(GUILD, uid, name, apply)


async def tools_for(bot, world, who, answering=None):
    member = world.g.members[who]
    where = SimpleNamespace(guild=world.g if answering else None, channel=answering)
    mem = await bot._memory_tools(where, world.g, member)
    is_exec = who in (OWNER_ID, ADMIN_ID, BOSS)
    return {f.__name__: f for f in bot._people_tools(None, world.g, member, is_exec, mem, "en")}


# ------------------------------------------------------------------ wiring
async def test_the_tools_reach_chat_and_voice_next_to_the_other_features(bot, world):
    member = world.g.members[STAFF]
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, member)
    msg = bot.room_message(world.g, member, world.pub)
    kw = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None,
    )  # fmt: skip
    names = {f.__name__ for f in kw["extra_tools"]}
    assert {"company_overview", "who_to_ask", "person_profile", "how_do_we",
            "onboarding_brief", "handover", "set_role", "person_left"} <= names  # fmt: skip
    assert {"teach_company", "wiki_search", "list_tasks"} <= names
    quiet = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None, interjection=True,
    )  # fmt: skip
    assert "who_to_ask" not in {f.__name__ for f in quiet.get("extra_tools", [])}
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"company_knowledge": False}), "x"
    )
    bot.team._cache.clear()
    off = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None,
    )  # fmt: skip
    assert "who_to_ask" not in {f.__name__ for f in off["extra_tools"]}


def test_the_prompt_section_is_keyed_on_who_to_ask_in_both_languages():
    th = qa.stable_prompt("th", False, knowledge=True)
    en = qa.stable_prompt("en", False, knowledge=True)
    assert KNOWLEDGE_PROMPT["th"].strip()[:30] in th and KNOWLEDGE_PROMPT["en"].strip()[:30] in en
    assert "institutional memory" in en and "ยังไม่รู้" in th
    assert "institutional memory" not in qa.stable_prompt("en", False)


def test_flag_default_on_and_strings_in_both_languages():
    assert "company_knowledge" in FEATURES and Features().company_knowledge is True
    keys = [k for k in CATALOG if k.startswith(("pe_", "ob_", "ho_"))]
    assert len(keys) >= 20
    for k in keys:
        assert CATALOG[k]["th"] and CATALOG[k]["en"], k


async def test_who_to_ask_is_scoped_by_what_the_asker_can_read(bot, world):
    learn(bot, STAFF, "Somchai", merge_topic=("expertise", "payroll", ev(SECRET), DAY, 5))
    learn(bot, PEER, "Peer", merge_topic=("expertise", "payroll", ev(PUBLIC), DAY, 1))
    peer_out = (await tools_for(bot, world, INTERN))["who_to_ask"]("payroll")
    assert [p["name"] for p in peer_out["people"]] == ["Peer"]  # the intern cannot read #ลับ
    boss_out = (await tools_for(bot, world, BOSS))["who_to_ask"]("payroll")  # a DM: reads all
    assert [p["name"] for p in boss_out["people"]] == ["Somchai", "Peer"]
    # the same executive answering in a public channel does not read the private evidence
    in_public = (await tools_for(bot, world, BOSS, answering=world.pub))["who_to_ask"]("payroll")
    assert [p["name"] for p in in_public["people"]] == ["Peer"]


async def test_the_summary_hook_reaches_the_people_listener_and_never_breaks_the_chain(bot, world):
    seen = []
    bot.people_listener.on_summary = lambda *a: seen.append(a[1:3])
    bot._on_summary_all(GUILD, PUBLIC, DAY, {}, [])
    assert seen == [(PUBLIC, DAY)]

    def boom(*a):
        raise RuntimeError("x")

    bot.people_listener.on_summary = boom
    bot._on_summary_all(GUILD, PUBLIC, DAY, {}, [])  # logged, not raised


# ------------------------------------------------------------------ the company map
async def test_the_nightly_map_uses_the_roster_and_only_what_everyone_can_read(bot, world):
    learn(bot, STAFF, "Somchai", merge_topic=("expertise", "invoices", ev(PUBLIC), DAY, 5))
    learn(bot, STAFF, "Somchai", merge_topic=("expertise", "layoffs", ev(SECRET), DAY, 9))
    learn(bot, STAFF, "Somchai", merge_contact=("contacts", P.HeardContact(who="ACME", about="supplier"), ev(PUBLIC), DAY))  # fmt: skip
    world.g.roles = [SimpleNamespace(name="Accountant")]
    world.staff.top_role = SimpleNamespace(name="Accountant", position=2)
    await asyncio.to_thread(bot._people_refresh, world.g)
    pb = bot.company.get(GUILD)
    names = {p.name for d in pb.org for pos in d.positions for p in pos.people}
    assert {"Somchai", "Boss", "Ploy"} <= names
    assert any(pos.title == "Accountant" for d in pb.org for pos in d.positions)
    assert [w.topic for w in pb.who_to_ask] == ["invoices"]  # not the private one
    assert [k.name for k in pb.key_contacts] == ["ACME"]
    assert bot.people.get(GUILD, STAFF).role_title == "Accountant"
    assert "invoices" in qa_block(bot, pb)


async def test_the_map_is_saved_when_one_channel_raises(bot, world, monkeypatch, caplog):
    learn(bot, STAFF, "Somchai", merge_topic=("expertise", "invoices", ev(PUBLIC), DAY, 5))
    real = bot._channel_viewers

    def flaky(guild, channel, **kw):
        if channel.id == SECRET:
            raise RuntimeError("boom")
        return real(guild, channel, **kw)

    monkeypatch.setattr(bot, "_channel_viewers", flaky)
    with caplog.at_level("INFO", logger="salid.people_bot"):
        await asyncio.to_thread(bot._people_refresh, world.g)
    pb = bot.company.get(GUILD)
    assert pb.org_built_at and [w.topic for w in pb.who_to_ask] == ["invoices"]
    line = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("people map: guild=")
    ]
    assert len(line) == 1 and f"guild={GUILD}" in line[0] and "topics=1" in line[0]
    assert any("channel skipped" in r.getMessage() for r in caplog.records)


async def test_the_map_is_saved_with_no_public_channels_and_an_empty_org(bot, world, monkeypatch):
    learn(bot, STAFF, "Somchai", merge_topic=("expertise", "invoices", ev(PUBLIC), DAY, 5))
    monkeypatch.setattr(bot, "_public_channel_ids", lambda guild: set())
    await asyncio.to_thread(bot._people_refresh, world.g)
    pb = bot.company.get(GUILD)
    assert pb.org_built_at and pb.who_to_ask == [] and pb.key_contacts == []
    monkeypatch.setattr(bot, "_people_roster", lambda *a, **k: [])
    bot.company.save(GUILD, cb.CompanyPlaybook())
    await asyncio.to_thread(bot.rebuild_company_map, world.g)
    assert bot.company.get(GUILD).org_built_at  # saved even with an empty org


async def test_a_failing_stage_never_stops_the_map(bot, world, monkeypatch, caplog):
    def boom(guild):
        raise RuntimeError("roles broke")

    monkeypatch.setattr(bot, "_people_stage_roles", boom)
    monkeypatch.setattr(bot, "_people_roster", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))  # fmt: skip
    with caplog.at_level("WARNING", logger="salid.people_bot"):
        await asyncio.to_thread(bot._people_refresh, world.g)
    assert bot.company.get(GUILD).org_built_at
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "stage=roles" in text and "stage=org" in text


async def test_a_partial_member_cache_needs_everyone_to_be_able_to_read(bot, world):
    world.g.member_count = 50  # only 6 are cached
    world.g.default_role = SimpleNamespace(name="@everyone")
    public = await asyncio.to_thread(bot._public_channel_ids, world.g)
    full = world.g
    full.member_count = len(world.g.members)
    whole = await asyncio.to_thread(bot._public_channel_ids, full)
    assert PUBLIC in whole and SECRET not in whole and public <= whole


async def test_the_assistant_itself_gets_no_dossier_and_is_not_in_the_roster(bot, world):
    me = bot.user
    bot.people.update(GUILD, me.id, "สลิด", lambda d: True)
    bot.people.update(GUILD, 424242, getattr(me, "display_name", "Salid"), lambda d: True)
    assert bot.people.list(GUILD, include_bots=True) == []
    bot.people.update(GUILD, 31337, "Salid", lambda d: True)  # a task handed to the bot by name
    assert bot.people.list(GUILD, include_bots=True) == []
    roster = await asyncio.to_thread(bot._people_roster, world.g, True)
    assert all(r.name not in ("Salid", "สลิด") for r in roster)
    world.g.members[31337] = SimpleNamespace(
        id=31337, display_name="สลิด", bot=False, top_role=None, guild_permissions=None
    )
    roster = await asyncio.to_thread(bot._people_roster, world.g, True)
    assert 31337 not in {r.user_id for r in roster}


def qa_block(bot, pb):
    from salid.company_brain import company_block

    return company_block(pb, "x")


async def test_nothing_is_built_when_the_flag_is_off(bot, world):
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"company_knowledge": False}), "x"
    )
    bot.team._cache.clear()
    await bot._people_due()
    assert bot.people.list(GUILD) == []


# ------------------------------------------------------------------ leaving
@pytest.fixture
def left_rig(bot, world, monkeypatch):
    g = prep(world.g)
    sent = []

    async def deliver(guild, m, text, files=None, view=None):
        sent.append((m.id, text, files))
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(bot, "deliver_to_person", deliver)
    import salid.people_bot as pb

    monkeypatch.setattr(pb, "markdown_pdf", lambda **kw: b"%PDF-" + kw["title"].encode())
    bot.llm = FakeClient(
        [text_resp("## Role\n- Accountant\n## Open items\n- Q3 report " + "x" * 60)]
    )
    learn(bot, STAFF, "Somchai", merge_fact=("responsibilities", "prepares payroll", ev(PUBLIC), DAY),
          merge_topic=("expertise", "payroll", ev(PUBLIC), DAY, 3))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, assignee_name="Somchai",
                        title="Q3 report", due="2026-10-30"))  # fmt: skip
    return SimpleNamespace(sent=sent, g=g)


async def test_a_member_who_leaves_gets_a_handover_and_the_executives_the_pdf(bot, world, left_rig):
    m = world.g.members[STAFF]
    await bot.on_member_remove(m)
    d = bot.people.get(GUILD, STAFF)
    assert d.status == "left" and d.left_at and "Accountant" in d.handover
    page = next(
        p for p in bot.wiki.list(GUILD) if p.title == t("pe_handover_page", "th", name="Somchai")
    )
    assert page.channel_ids == [PUBLIC] and page.body == d.handover
    assert left_rig.sent, "the executives were told"
    uids = {u for u, _, _ in left_rig.sent}
    assert uids <= {OWNER_ID, ADMIN_ID, BOSS} and STAFF not in uids
    _, text, files = left_rig.sent[0]
    assert "Somchai" in text and "Q3 report" in text and "(2026-10-30)" in text
    assert files and files[0].filename.startswith("handover-")
    prompt = bot.llm.models.calls[0][1]
    assert "prepares payroll" in prompt and "Q3 report" in prompt  # dossier + open tasks


async def test_the_handover_falls_back_to_the_dossier_without_a_model(bot, world, left_rig):
    bot.llm = None
    await bot.on_member_remove(world.g.members[STAFF])
    d = bot.people.get(GUILD, STAFF)
    assert "prepares payroll" in d.handover and "Q3 report" in d.handover


async def test_leaving_twice_or_without_any_knowledge_does_nothing_more(bot, world, left_rig):
    await bot.on_member_remove(world.g.members[STAFF])
    n = len(left_rig.sent)
    await bot.on_member_remove(world.g.members[STAFF])
    assert len(left_rig.sent) == n  # already left
    await bot.on_member_remove(world.g.members[INTERN])  # never spoke, no tasks
    assert bot.people.get(GUILD, INTERN) is None and len(left_rig.sent) == n


async def test_person_left_tool_marks_lists_tasks_and_writes_the_handover_later(
    bot, world, left_rig
):
    t_ = await tools_for(bot, world, BOSS)
    out = await asyncio.to_thread(t_["person_left"], "Somchai", "Peer")
    assert out["status"] == "recorded" and out["open_tasks"][0]["title"] == "Q3 report"
    for _ in range(50):
        await asyncio.sleep(0.02)
        if bot.people.get(GUILD, STAFF).handover:
            break
    d = bot.people.get(GUILD, STAFF)
    assert d.status == "left" and d.successor_id == PEER and d.handover and left_rig.sent
    ask = (await tools_for(bot, world, BOSS))["who_to_ask"]("payroll")
    p = ask["people"][0]
    assert p["left"] and p["successor"] == "Peer" and p["handover_available"]
    denied = (await tools_for(bot, world, INTERN))["person_left"]("Peer")
    assert denied["error"] == "not_allowed"


async def test_the_handover_tool_generates_on_demand_for_an_executive(bot, world, left_rig):
    t_ = await tools_for(bot, world, BOSS)
    out = await asyncio.to_thread(t_["handover"], "Somchai")
    assert out["written_now"] and "Accountant" in out["handover"]
    assert bot.people.get(GUILD, STAFF).status == "active"  # a planned leave: not marked yet


# ------------------------------------------------------------------ welcome
async def test_the_private_channel_welcome_adds_the_company_knowledge_line(bot, world):
    g = prep(world.g)
    m = human(g, 41, "Newbie")
    bot.people.update(GUILD, 41, "Newbie", lambda d: P.set_role(d, "", "stated", "finance") or True)
    ch, created, err = await bot.ensure_private_channel(g, m)
    assert created and not err
    welcome = ch.sent[0][0]
    assert t("pe_welcome", "th") in welcome
    assert "Finance" in welcome and "คู่มือต้อนรับ" in welcome  # the brief is offered


async def test_no_extra_line_without_a_department_or_when_the_flag_is_off(bot, world):
    g = prep(world.g)
    m = human(g, 42, "Nobody")
    ch, _, _ = await bot.ensure_private_channel(g, m)
    text = ch.sent[0][0]
    assert t("pe_welcome", "th") in text and "คู่มือต้อนรับ" not in text
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"company_knowledge": False}), "x"
    )
    bot.team._cache.clear()
    m2 = human(g, 43, "Other")
    ch2, _, _ = await bot.ensure_private_channel(g, m2)
    assert t("pe_welcome", "th") not in ch2.sent[0][0]
