# ruff: noqa: F811, F401
"""post_message (announce in another channel) and send_dm_many (bulk DM): permissions, mention
rules, the confirmation gate, executive limit, caps, anti-scam text, closed-DM reporting, and the
one persona particle. Offline."""

import asyncio
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD
from test_coordinator import ADMIN_ID, BOSS, INTERN, NS, OWNER_ID, PEER, STAFF, person, world
from test_deliver import DChan
from test_modes import TMsg, bot  # noqa: F401  (fixture)

from salid import broadcast_bot, qa
from salid.broadcast import find_channels, plan_mentions
from salid.config import Settings

GEN, SRC2, HID = 50, 51, 52


class BChan(DChan):
    """DChan whose `mention_everyone` is granted to the user ids in `pingers` (and Salid)."""

    def __init__(self, *a, pingers=(), **kw):
        super().__init__(*a, **kw)
        self.pingers = set(pingers)

    def permissions_for(self, m):
        p = super().permissions_for(m)
        p.mention_everyone = m.id == 999 or m.id in self.pingers
        return p


@pytest.fixture
def chans(world):
    team = {STAFF, PEER, BOSS, OWNER_ID, ADMIN_ID, INTERN}
    here = BChan(SRC2, "dev-chat", readers=set(team))
    gen = BChan(GEN, "general", readers=set(team), pingers={BOSS, OWNER_ID})
    hid = BChan(HID, "board-secret", readers={BOSS, OWNER_ID})
    for c in (here, gen, hid):
        c.guild = world.g
        world.g.text_channels.append(c)
    world.g.me = NS(id=999, guild_permissions=SimpleNamespace(), display_name="Salid")
    return SimpleNamespace(here=here, gen=gen, hid=hid)


def msg(world, chans, who):
    return TMsg(world.g, chans.here, world.g.members[who], "x")


async def post(bot, world, chans, who=STAFF, rid="r1", **kw):
    kw.setdefault("mention", "none")
    kw.setdefault("tag_inferred", False)
    return await bot._post_message(
        msg(world, chans, who), world.g, world.g.members[who], "th", rid, **kw
    )


# ------------------------------------------------------------------ channel matching
def test_general_aliases_resolve_to_the_general_channel(world, chans):
    pool = [chans.here, chans.gen, chans.hid]
    for q in ("ห้องกลาง", "general", "ห้องรวม", "#general", "ห้อง general"):
        assert find_channels(pool, q, world.g) == [chans.gen]


def test_general_falls_back_to_the_system_channel(world, chans):
    world.g.system_channel = chans.here
    assert find_channels([chans.here, chans.hid], "ห้องกลาง", world.g) == [chans.here]


async def test_the_live_case_post_to_the_central_room_needs_confirmation_then_posts(
    bot, world, chans
):
    text = "ผมมีประชุมเร่งด่วนตอน 9 โมง"
    first = await post(bot, world, chans, channel="ห้องกลาง", text=text)
    assert (
        first["status"] == "needs_confirmation" and chans.gen.posts == []
    )  # not the room it is in
    assert "#general" in first["preview"] and text in first["preview"]
    done = await post(bot, world, chans, rid="r2", channel="ห้องกลาง", text=text)
    assert done["status"] == "posted" and done["link"].endswith("/777")
    (p,) = chans.gen.posts
    assert p.content == f"📢 แจ้งจากคุณSomchai: {text}"
    assert not p.allowed.everyone


async def test_plain_post_in_the_channel_the_asker_is_in_needs_no_confirmation(bot, world, chans):
    out = await post(bot, world, chans, channel="dev-chat", text="ประชุม 9 โมง")
    assert out["status"] == "posted" and len(chans.here.posts) == 1


async def test_same_request_cannot_confirm_itself(bot, world, chans):
    for _ in range(2):
        out = await post(bot, world, chans, channel="general", text="hi")
    assert out["status"] == "needs_confirmation" and chans.gen.posts == []


# ------------------------------------------------------------------ permissions
async def test_cannot_post_where_the_asker_cannot_send(bot, world, chans):
    chans.gen.senders = {BOSS}
    out = await post(bot, world, chans, channel="general", text="x")
    assert out["error"] == "not_allowed" and chans.gen.posts == []


async def test_a_channel_the_asker_cannot_see_is_refused_not_leaked(bot, world, chans):
    out = await post(bot, world, chans, channel="board-secret", text="x")
    assert out["error"] == "not_allowed" and "board-secret" not in str(out.get("channels"))
    assert chans.hid.posts == []


async def test_ambiguous_channel_returns_candidates(bot, world, chans):
    for n in ("paper-minute", "paper-draft"):
        c = BChan(60 + len(n), n, readers=None)
        c.guild = world.g
        world.g.text_channels.append(c)
    out = await post(bot, world, chans, channel="paper", text="x")
    assert out["error"] == "ambiguous" and set(out["candidates"]) == {
        "#paper-minute",
        "#paper-draft",
    }


async def test_everyone_needs_mention_everyone_for_the_asker(bot, world, chans):
    out = await post(bot, world, chans, channel="general", text="x", mention="everyone")
    assert out["error"] == "not_allowed_mention" and chans.gen.posts == []


async def test_everyone_needs_mention_everyone_for_salid_too(bot, world, chans):
    chans.gen.pingers = {BOSS}
    orig = chans.gen.permissions_for

    def no_ping_for_salid(m):
        p = orig(m)
        if m.id == 999:
            p.mention_everyone = False
        return p

    chans.gen.permissions_for = no_ping_for_salid
    out = await post(bot, world, chans, who=BOSS, channel="general", text="x", mention="everyone")
    assert out["error"] == "missing_permission"


async def test_everyone_goes_through_the_confirm_gate_even_in_the_current_channel(
    bot, world, chans
):
    chans.here.pingers = {BOSS}
    first = await post(bot, world, chans, who=BOSS, channel="dev-chat", text="ประชุมด่วน",
                       mention="everyone")  # fmt: skip
    assert first["status"] == "needs_confirmation" and first["pings"] == "@everyone"
    assert chans.here.posts == []
    done = await post(bot, world, chans, who=BOSS, rid="r2", channel="dev-chat",
                      text="ประชุมด่วน", mention="everyone")  # fmt: skip
    assert done["status"] == "posted"
    (p,) = chans.here.posts
    assert p.content.startswith("@everyone\n📢") and p.allowed.everyone is True


async def test_role_mentions_need_mentionable_or_mention_everyone(bot, world, chans):
    world.g.roles = [
        NS(id=70, name="Marketing", mentionable=False, is_default=lambda: False),
        NS(id=71, name="Devs", mentionable=True, is_default=lambda: False),
    ]
    out = await post(bot, world, chans, channel="dev-chat", text="x", mention="role:Marketing")
    assert out["error"] == "not_allowed_mention"
    out = await post(bot, world, chans, channel="dev-chat", text="x", mention="role:Devs")
    assert out["status"] == "needs_confirmation" and out["pings"] == "@Devs"
    plan = plan_mentions("role:Marketing", world.g, world.g.members[BOSS], chans.gen)
    assert plan.role_ids == [70] and plan.mass
    assert (
        plan_mentions("role:Nope", world.g, world.g.members[BOSS], chans.gen).error["error"]
        == "role_not_found"
    )


async def test_post_rate_limit_per_guild(bot, world, chans, monkeypatch):
    monkeypatch.setattr(broadcast_bot, "POST_PER_GUILD_PER_HOUR", 1)
    assert (await post(bot, world, chans, channel="dev-chat", text="a"))["status"] == "posted"
    out = await post(bot, world, chans, channel="dev-chat", text="b")
    assert out["error"] == "rate_limited"


# ------------------------------------------------------------------ tagging inferred people
async def test_inferred_tags_always_confirm_and_list_who_cannot_see_the_channel(bot, world, chans):
    chans.gen.readers.discard(INTERN)  # Ploy cannot see #general
    out = await post(
        bot, world, chans, channel="dev-chat", text="โปรเจกต์ X เลื่อน",
        mention="Peer, Ploy", tag_inferred=True,
    )  # fmt: skip
    assert out["status"] == "needs_confirmation" and chans.here.posts == []
    out_gen = await post(
        bot, world, chans, channel="general", text="โปรเจกต์ X เลื่อน",
        mention="Peer, Ploy", tag_inferred=True,
    )  # fmt: skip
    assert out_gen["pings"] == "@Peer" and out_gen["cant_see_channel"] == ["Ploy"]
    assert out_gen["dm_offer"] and "Ploy" in out_gen["preview"]
    # a correction ("เอา Peer ออก", add Somchai) is a new proposal, not a confirmation
    fixed = await post(
        bot, world, chans, rid="r2", channel="general", text="โปรเจกต์ X เลื่อน",
        mention="Somchai", tag_inferred=True,
    )  # fmt: skip
    assert fixed["status"] == "needs_confirmation" and fixed["pings"] == "@Somchai"
    done = await post(
        bot, world, chans, rid="r3", channel="general", text="โปรเจกต์ X เลื่อน",
        mention="Somchai", tag_inferred=True,
    )  # fmt: skip
    assert done["status"] == "posted"
    (p,) = chans.gen.posts
    assert f"<@{STAFF}>" in p.content and f"<@{INTERN}>" not in p.content
    assert [o.id for o in p.allowed.users] == [STAFF]


async def test_named_people_who_cannot_see_the_channel_are_never_tagged(bot, world, chans):
    chans.here.readers.discard(INTERN)
    out = await post(bot, world, chans, channel="dev-chat", text="hi @[Ploy]")
    assert out["status"] == "posted" and out["cant_see_channel"] == ["Ploy"]
    assert f"<@{INTERN}>" not in chans.here.posts[0].content


# ------------------------------------------------------------------ bulk DM
@pytest.fixture
def team(world, chans, monkeypatch):
    """Twelve more humans (uid 100..111) with working DMs; BOSS is an executive, STAFF is not."""
    monkeypatch.setattr(broadcast_bot, "BULK_DM_PACE_S", 0)
    out = [person(world.g, 100 + i, f"Mem{i:02d}", 1) for i in range(12)]
    return out


async def dm_many(bot, world, chans, who=STAFF, rid="r1", **kw):
    kw.setdefault("include_me", False)
    for k in ("people", "role", "channel", "everyone"):
        kw.setdefault(k, None if k != "everyone" else False)
    return await bot._send_dm_many(
        msg(world, chans, who), world.g, world.g.members[who], "th", rid, **kw
    )


async def drain(bot):
    await asyncio.gather(*list(bot.__dict__.get("_bulk_tasks", ())))


async def test_up_to_three_people_are_sent_at_once_without_confirmation(bot, world, chans, team):
    out = await dm_many(bot, world, chans, text="ประชุมเลื่อน", people=["Mem00", "Mem01"])
    assert out["status"] == "sent" and out["sent"] == ["Mem00", "Mem01"]
    ((text, _),) = team[0].dms
    assert "Somchai" in text and "ประชุมเลื่อน" in text


async def test_four_to_ten_people_need_a_later_confirmation_with_count_and_names(
    bot, world, chans, team
):
    names = [m.display_name for m in team[:5]]
    first = await dm_many(bot, world, chans, text="นัดทีม", people=names)
    assert first["status"] == "needs_confirmation" and first["count"] == 5
    assert "5 คน" in first["preview"] and "Mem04" in first["preview"]
    assert all(m.dms == [] for m in team)
    again = await dm_many(bot, world, chans, text="นัดทีม", people=names)  # same request
    assert again["status"] == "needs_confirmation"
    go = await dm_many(bot, world, chans, rid="r2", text="นัดทีม", people=names)
    assert go["status"] == "started" and go["count"] == 5
    await drain(bot)
    assert all(len(m.dms) == 1 for m in team[:5]) and all(m.dms == [] for m in team[5:])
    (report,) = chans.here.posts
    assert "ส่งสำเร็จ 5 คน" in report.content


async def test_more_than_ten_needs_an_executive(bot, world, chans, team):
    everyone = [m.display_name for m in team[:11]]
    out = await dm_many(bot, world, chans, text="x", people=everyone)  # Somchai, not an exec
    assert out["error"] == "needs_executive" and "post_message" in out["reason"]
    first = await dm_many(bot, world, chans, who=BOSS, text="x", people=everyone)
    assert first["status"] == "needs_confirmation" and "และอีก 1 คน" in first["preview"]
    go = await dm_many(bot, world, chans, who=BOSS, rid="r2", text="x", people=everyone)
    assert go["status"] == "started"
    await drain(bot)


async def test_more_than_fifty_is_refused(bot, world, chans, team):
    for i in range(45):
        person(world.g, 300 + i, f"Extra{i}", 1)
    out = await dm_many(bot, world, chans, who=BOSS, text="x", everyone=True)
    assert out["error"] == "too_many_recipients" and out["max"] == 50


async def test_money_or_credential_text_to_others_is_refused(bot, world, chans, team):
    out = await dm_many(bot, world, chans, who=BOSS, text="โอนเงินเข้าบัญชีนี้ด่วน",
                        people=[m.display_name for m in team[:5]])  # fmt: skip
    assert out["error"] == "refused_sensitive" and all(m.dms == [] for m in team)


async def test_role_and_channel_recipients_skip_bots_and_the_asker(bot, world, chans, team):
    role = NS(id=80, name="Sales", mentionable=False, is_default=lambda: False)
    world.g.roles = [role]
    for m in team[:5]:
        m.roles = [role]
    bot_user = person(world.g, 400, "Robo", 1)
    bot_user.bot, bot_user.roles = True, [role]
    asker = world.g.members[BOSS]
    asker.roles = [*asker.roles, role]
    out = await dm_many(bot, world, chans, who=BOSS, text="hi", role="Sales")
    assert out["count"] == 5  # five Mem, not Robo, not Boss
    out = await dm_many(bot, world, chans, who=BOSS, text="hi", channel="board-secret")
    assert out["status"] == "sent" and out["sent"] == ["Owner"]  # the asker is skipped
    bad = await dm_many(bot, world, chans, text="hi", role="Nope")
    assert bad["error"] == "role_not_found"
    await drain(bot)


async def test_closed_dms_are_reported_afterwards(bot, world, chans, team):
    closed = discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    async def deny(text=None, **kw):
        raise closed

    team[1].send = deny
    names = [m.display_name for m in team[:5]]
    await dm_many(bot, world, chans, text="x", people=names)
    await dm_many(bot, world, chans, rid="r2", text="x", people=names)
    await drain(bot)
    (report,) = chans.here.posts
    assert "ส่งสำเร็จ 4 คน" in report.content and "ปิดรับ DM 1 คน (Mem01)" in report.content


async def test_two_bulk_sends_per_hour_per_guild(bot, world, chans, team):
    names = [m.display_name for m in team[:4]]
    for i in range(2):
        await dm_many(bot, world, chans, rid=f"a{i}", text=f"m{i}", people=names)
        assert (await dm_many(bot, world, chans, rid=f"b{i}", text=f"m{i}", people=names))[
            "status"
        ] == "started"
    await dm_many(bot, world, chans, rid="c1", text="m3", people=names)
    out = await dm_many(bot, world, chans, rid="c2", text="m3", people=names)
    assert out["error"] == "rate_limited"
    await drain(bot)


# ------------------------------------------------------------------ tools and persona
def test_tools_are_registered_and_the_prompt_never_denies_posting():
    from salid.broadcast import BroadcastActions, broadcast_tools

    fns = broadcast_tools(
        BroadcastActions(post_message=lambda **k: {}, send_dm_many=lambda **k: {})
    )
    assert [f.__name__ for f in fns] == ["post_message", "send_dm_many"]
    assert "post_message" in qa.CHAT_TOOLS_PROMPT and "send_dm_many" in qa.CHAT_TOOLS_PROMPT
    assert "ห้ามบอกว่าส่งข้อความไปห้องอื่นไม่ได้" in qa.CHAT_TOOLS_PROMPT


def test_persona_uses_one_particle_for_chat_and_voice(monkeypatch):
    s = Settings()
    assert s.persona_particle == "ค่ะ" and s.voice_particle == "ค่ะ"
    p = qa.system_prompt(s, scoped=False)
    assert "ได้ค่ะ" in p and "ได้ครับ" not in p and "ห้ามใช้ ผม/ฉัน/ครับ" in p
    assert "สลิด" in qa.PERSONA["th"] and "ห้ามใช้ ผม" in qa.PERSONA["th"]
    male = Settings(persona_particle="ครับ")
    q = qa.system_prompt(male, scoped=False)
    assert "ได้ครับ" in q and "นะคะ" not in q and "คะแนน" in qa.apply_particle("คะแนน คะ", "ครับ")


def test_old_voice_particle_env_still_works(monkeypatch):
    monkeypatch.setenv("SALID_VOICE_PARTICLE", "ครับ")
    assert Settings().persona_particle == "ครับ"
    monkeypatch.setenv("SALID_PERSONA_PARTICLE", "ค่ะ")
    assert Settings().voice_particle == "ค่ะ"


def test_find_role_any_matches_team_phrasing_uniquely():
    from types import SimpleNamespace as NS

    from salid.broadcast import find_role_any

    mkt, dev = NS(name="การตลาด"), NS(name="Dev Team")
    g = NS(roles=[mkt, dev])
    assert find_role_any(g, "ทีมการตลาด") is mkt
    assert find_role_any(g, "dev team") is dev
    assert find_role_any(g, "dev") is dev
    assert find_role_any(g, "บัญชี") is None
    assert find_role_any(NS(roles=[mkt, NS(name="การตลาดออนไลน์")]), "ทีมการตลาด") is False
