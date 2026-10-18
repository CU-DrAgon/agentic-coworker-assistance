# ruff: noqa: F401, F811
"""Which company a DM is about (salid.dm_guild): sticky choice, buttons, confirm-to-switch, reply
tags, left servers, footer, isolation between companies."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_bot_multitenant import (
    GUILD,
    OTHER_GUILD,
    STAFF,
    FakeChannel,
    FakeGuild,
    bot,
    member,
    world,
)  # noqa: F401

from salid import bot as botmod
from salid.bot import SecretaryBot
from salid.dm_guild import CompanyButton, SwitchButton, fuzzy_pick, parse_switch
from salid.i18n import t
from salid.memory_access import dm_scope

THIRD = 3000
_ids = iter(range(5000, 99999))


class DM:
    def __init__(self, text, uid=STAFF, reference=None):
        self.guild, self.channel = None, FakeChannel(1)
        self.author = SimpleNamespace(id=uid, bot=False)
        self.content, self.attachments, self.mentions = text, [], []
        self.reference = reference
        self.id = next(_ids)
        self.created_at = datetime.now(UTC)
        self.replies = []

    async def reply(self, text, **kw):
        self.replies.append((text, kw))


class Interaction:
    def __init__(self, uid=STAFF):
        self.user = SimpleNamespace(id=uid)
        self.locale = "th"
        self.edits, self.sent = [], []
        outer = self

        class R:
            async def edit_message(self, **kw):
                outer.edits.append(kw)

            async def send_message(self, text, **kw):
                outer.sent.append(text)

        self.response = R()


@pytest.fixture
def two(bot, world):  # noqa: F811
    """The staff member is in A and B; a third server C does not have them."""
    member(world.a, STAFF)
    member(world.b, STAFF)
    return bot


def active(bot):
    return bot.dm_guilds.active(STAFF)


def buttons(msg):
    return [i for i in msg.replies[0][1]["view"].children]


def test_parse_switch_and_fuzzy():
    assert parse_switch("เปลี่ยนเป็นบริษัท Zeta") == ("Zeta", True)
    assert parse_switch("switch to Zeta Corp") == ("Zeta Corp", False)
    assert parse_switch("ยอดขายเดือนนี้เท่าไร") is None
    names = {1: "Zeta Corp", 2: "Alpha Shop"}
    assert fuzzy_pick("zeta", names) == 1 and fuzzy_pick("alpa shop", names) == 2
    assert fuzzy_pick("nothing", names) is None


async def test_two_guilds_offer_buttons_and_record_nothing(two, world):
    stranger = FakeGuild(THIRD, "Secret Co")
    stranger.get_member = lambda uid: None
    m = DM("ยอดขายเท่าไร")
    await two.on_message(m)
    await two.drain_memory()
    assert two.calls == []  # not answered
    assert m.replies[0][0].startswith(t("dmg_ask", "th"))
    labels = [b.item.label for b in buttons(m)]
    assert labels == ["A", "B"] and all(isinstance(b, CompanyButton) for b in buttons(m))
    for gid in (GUILD, OTHER_GUILD):  # nothing recorded anywhere
        assert two.memory_store.messages(gid, {dm_scope(STAFF)}, "2000-01-01", "2100-01-01") == []


async def test_click_sets_the_company_and_answers_the_held_message(two, world):
    m = DM("ยอดขายเท่าไร")
    await two.on_message(m)
    click = Interaction()
    await two.handle_company_pick(click, OTHER_GUILD, STAFF)
    assert active(two) == OTHER_GUILD
    assert two.calls == [("answer", OTHER_GUILD)]
    await two.drain_memory()
    rows = lambda g: two.memory_store.messages(g, {dm_scope(STAFF)}, "2000-01-01", "2100-01-01")  # noqa: E731
    assert [r.text for r in rows(OTHER_GUILD)] == ["ยอดขายเท่าไร"] and rows(GUILD) == []
    # sticky: the next DM goes there with no question, and carries the footer
    m2 = DM("และเดือนก่อน")
    await two.on_message(m2)
    assert two.calls[-1] == ("answer", OTHER_GUILD) and m2.replies == []


async def test_someone_elses_button_is_refused(two):
    click = Interaction(uid=77)
    await two.handle_company_pick(click, GUILD, STAFF)
    assert active(two) is None and click.sent == [t("dmg_not_yours", "th")]


async def test_reply_to_a_tagged_salid_dm_uses_that_company_once(two, world):
    two.dm_guilds.set_active(STAFF, GUILD)
    two.dm_guilds.tag(555, OTHER_GUILD)
    m = DM("โอเคค่ะ", reference=SimpleNamespace(message_id=555))
    await two.on_message(m)
    assert two.calls == [("answer", OTHER_GUILD)] and m.replies == []
    assert active(two) == GUILD  # the sticky choice is untouched
    await two.on_message(DM("ต่อค่ะ"))
    assert two.calls[-1] == ("answer", GUILD)


async def test_first_switch_is_direct_later_ones_need_confirmation(two, world):
    m = DM("switch to B")
    await two.on_message(m)
    assert active(two) == OTHER_GUILD and "B" in m.replies[0][0]
    ask = DM("เปลี่ยนเป็นบริษัท A")
    await two.on_message(ask)
    assert active(two) == OTHER_GUILD  # not yet
    assert ask.replies[0][0] == t("dmg_confirm", "th", name="A")
    assert all(isinstance(b, SwitchButton) for b in buttons(ask))
    yes = DM("ใช่")
    await two.on_message(yes)
    assert active(two) == GUILD and two.calls == []  # a confirmation is not a question


async def test_switch_buttons_confirm_or_cancel(two, world):
    two.dm_guilds.set_active(STAFF, GUILD)
    await two.on_message(DM("switch to B"))
    cancel = Interaction()
    await two.handle_company_switch(cancel, "no", OTHER_GUILD, STAFF)
    assert active(two) == GUILD and cancel.edits[0]["content"] == t("dmg_kept", "th")
    ok = Interaction()
    await two.handle_company_switch(ok, "ok", OTHER_GUILD, STAFF)
    assert active(two) == OTHER_GUILD


async def test_leaving_the_chosen_server_clears_it_and_asks_again(bot, world, monkeypatch):
    member(world.a, STAFF)
    member(world.b, STAFF)
    third = FakeGuild(THIRD, "C")
    member(third, STAFF)
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [world.a, world.b, third]))
    bot.dm_guilds.set_active(STAFF, THIRD)
    del third.members[STAFF]  # left C
    m = DM("ยอด")
    await bot.on_message(m)
    assert (
        active(bot) is None and bot.calls == [] and m.replies[0][0].startswith(t("dmg_ask", "th"))
    )


async def test_single_guild_is_unchanged_and_has_no_footer(bot, world):
    member(world.a, STAFF)
    m = DM("ยอด")
    await bot.on_message(m)
    assert bot.calls == [("answer", GUILD)] and m.replies == [] and bot._dm_footers == {}


async def test_footer_only_with_two_or_more_companies(two, world):
    two.dm_guilds.set_active(STAFF, GUILD)
    m = DM("ยอด")
    await two.on_message(m)
    assert two._dm_footers[m.id] == "\n" + t("dmg_footer", "th", name="A")
    sent = []

    async def reply(text, **kw):
        sent.append(text)

    m.reply = reply
    await two._deliver(m, world.a, None, "คำตอบ")
    assert sent == ["คำตอบ\n— บริษัท: A"] and m.id not in two._dm_footers


async def test_meeting_reason_replies_only_count_in_the_chosen_company(two, world, monkeypatch):
    now = datetime.now(UTC)
    two.scheduler.store.put_meeting({
        "id": "m1", "guild_id": GUILD, "title": "Sync", "status": "scheduled", "kind": "text",
        "text_channel_id": 10, "attendee_ids": [STAFF], "attendee_names": ["x"],
        "dm_sent": {str(STAFF): now.isoformat(timespec="seconds")}, "attendance_notes": {},
        "start_utc": (now + timedelta(minutes=5)).isoformat(timespec="seconds"),
        "end_utc": (now + timedelta(minutes=35)).isoformat(timespec="seconds"),
    })  # fmt: skip
    monkeypatch.setattr(two.scheduler, "get_guild", lambda gid: world.a)
    monkeypatch.setattr(two.scheduler, "clock", lambda: now)

    async def post(*a, **k):
        pass

    monkeypatch.setattr(two.scheduler, "post", post)
    two.dm_guilds.set_active(STAFF, OTHER_GUILD)
    await two.on_message(DM("ไปช้า"))
    assert two.scheduler.store.get_meeting("m1")["attendance_notes"] == {}  # company B: not A's
    assert two.calls == [("answer", OTHER_GUILD)]
    two.dm_guilds.set_active(STAFF, GUILD)
    m = DM("ไปช้า")
    await two.on_message(m)
    assert two.scheduler.store.get_meeting("m1")["attendance_notes"][str(STAFF)]["note"]


async def test_disclosure_and_approval_dms_are_tagged_with_their_company(two, world):
    sent = SimpleNamespace(id=808)
    await two._dm_tag(sent, OTHER_GUILD)
    assert two.dm_guilds.tagged(808) == OTHER_GUILD
    await two._dm_tag(SimpleNamespace(id=None), GUILD)  # tolerated
    assert botmod.dm_module.ROUTER is not None
