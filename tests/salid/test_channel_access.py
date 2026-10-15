# ruff: noqa: F811, F401
"""Channels Salid cannot read: helper, /ตั้งค่า + /ห้อง messages, the 2000-char split, hub sync."""

from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_bot_multitenant import (
    ADMIN,
    MANAGER,
    ME,
    STAFF,
    FakeChannel,
    FakeGuild,
    Interaction,
    bot,
    member,
    world,
)

from salid.channel_access import (
    MAX_LISTED,
    hidden_channels,
    hidden_channels_messages,
    split_message,
)
from salid.i18n import t


class Chan(FakeChannel):
    def __init__(self, id, name, *, view=True, history=True, category=None, position=0):
        super().__init__(id, name)
        self.view, self.history, self.category, self.position = view, history, category, position

    def permissions_for(self, _):
        return SimpleNamespace(view_channel=self.view, read_message_history=self.history)


def cat(name, position):
    return SimpleNamespace(name=name, position=position)


class G(FakeGuild):
    def __init__(self, text=(), voice=(), forums=()):
        super().__init__(GUILD, "A", text)
        self.voice_channels, self.forums = list(voice), list(forums)


def test_hidden_channels_missing_view_history_and_visible():
    g = G(
        [Chan(1, "ok"), Chan(2, "secret", view=False), Chan(3, "old", history=False)],
        voice=[Chan(4, "voice", view=False)],
        forums=[Chan(5, "forum", history=False)],
    )
    got = {e["id"]: (e["missing"], e["reason"]) for e in hidden_channels(g, ME)}
    assert got == {
        2: ("view", "ไม่เห็นห้อง"),
        3: ("history", "อ่านย้อนหลังไม่ได้"),
        4: ("view", "ไม่เห็นห้อง"),
        5: ("history", "อ่านย้อนหลังไม่ได้"),
    }


def test_hidden_channels_sorted_by_category_then_position():
    a, b = cat("A", 2), cat("B", 1)
    g = G(
        [
            Chan(1, "a-late", view=False, category=a, position=5),
            Chan(2, "a-early", view=False, category=a, position=1),
            Chan(3, "b", view=False, category=b, position=9),
            Chan(4, "no-category", view=False, position=3),
        ]
    )
    es = hidden_channels(g, ME)
    assert [e["id"] for e in es] == [4, 3, 2, 1]  # uncategorised, B (pos 1), A (pos 2)
    assert [e["category"] for e in es] == ["", "B", "A", "A"]
    assert es[0]["name"] == "no-category"


def test_unknown_permissions_count_as_visible():
    assert hidden_channels(G([FakeChannel(1)]), ME) == []


def test_messages_none_few_and_many():
    assert hidden_channels_messages([], "th") == [t("hidden_none", "th")]
    assert hidden_channels_messages([], "en") == ["Salid can see every channel now 👍"]
    g = G([Chan(i, f"c{i}", view=False) for i in range(1, 4)])
    (msg,) = hidden_channels_messages(hidden_channels(g, ME), "th")
    assert "ห้องที่สลิดยังมองไม่เห็น (3 ห้อง)" in msg and "<#1>" in msg and "<#3>" in msg
    for step in ("1. กดชื่อห้องด้านบน แล้วกด ⚙️ (Edit Channel)", "4. ติ๊ก ✅ View Channel"):
        assert step in msg
    assert "ห้องลับที่ไม่อยากให้สลิดรู้ ไม่ต้องเพิ่มครับ" in msg and "และอีก" not in msg
    many = G([Chan(i, f"c{i}", view=False) for i in range(1, 21)])
    (msg,) = hidden_channels_messages(hidden_channels(many, ME), "th")
    assert f"<#{MAX_LISTED}>" in msg and f"<#{MAX_LISTED + 1}>" not in msg
    assert "และอีก 5 ห้อง (ดูทั้งหมดในหน้า Hub)" in msg
    en = hidden_channels_messages(hidden_channels(g, ME), "en")[0]
    assert "Channels Salid cannot see yet (3)" in en and "Read Message History" in en


def test_split_message_respects_the_limit():
    text = "\n".join(f"line {i} " + "x" * 90 for i in range(60))
    chunks = split_message(text)
    assert len(chunks) > 1 and all(len(c) <= 2000 for c in chunks)
    assert "\n".join(chunks) == text
    assert all(len(c) <= 50 for c in split_message("y" * 120, 50))
    assert split_message("short") == ["short"]


# ------------------------------------------------------------------ commands
@pytest.fixture
def hidden_world(world):
    world.a.text_channels += [Chan(30, "secret", view=False), Chan(31, "ok")]
    for c in world.a.text_channels[-2:]:
        c.guild = world.a
    return world


async def test_setup_adds_the_hidden_channels_section(bot, hidden_world):
    bot.s = bot.s.model_copy(update={"hub_url": "https://hub.example/"})
    it = Interaction(hidden_world.a, member(hidden_world.a, MANAGER, manage_guild=True))
    await bot.handle_setup(it)
    assert len(it.sent) == 2 and all(e for _, e in it.sent)
    assert "link=" in it.sent[0][0] and "<#30>" in it.sent[1][0] and "<#31>" not in it.sent[1][0]


async def test_channels_command_permissions_and_content(bot, hidden_world):
    g = hidden_world.a
    it = Interaction(g, member(g, MANAGER, manage_guild=True))
    await bot.handle_channels(it)
    assert it.sent[0][1] is True and "ห้องที่สลิดยังมองไม่เห็น (1 ห้อง)" in it.sent[0][0]
    ex = Interaction(g, member(g, 6, roles=["Executive"]))  # an executive without Manage Server
    await bot.handle_channels(ex)
    assert "<#30>" in ex.sent[0][0]
    staff = Interaction(g, member(g, STAFF, roles=["พนักงาน"]))
    await bot.handle_channels(staff)
    assert staff.sent == [(t("setup_denied", "th"), True)]
    dm = Interaction(None, SimpleNamespace(id=1))
    await bot.handle_channels(dm)
    assert dm.sent[0][1] is True


async def test_channels_command_splits_long_lists_and_answers_in_all_clear(bot, world):
    g = world.a
    assert await _run(bot, g) == [t("hidden_none", "th")]
    g.text_channels += [Chan(1000 + i, "n" * 80, view=False) for i in range(60)]
    chunks = await _run(bot, g)
    assert "และอีก 45 ห้อง" in "\n".join(chunks)
    assert all(len(c) <= 2000 for c in chunks)

    long = ["• <#1> " + "z" * 90] * 40
    assert len(split_message("\n".join(long))) > 1


async def _run(bot, g):
    it = Interaction(g, member(g, MANAGER, manage_guild=True))
    await bot.handle_channels(it)
    return [t for t, _ in it.sent]


# ------------------------------------------------------------------ hub sync
async def test_guild_sync_carries_visibility_flags(bot, hidden_world):
    await bot._sync_guild(hidden_world.a)
    chans = {c.id: c for c in bot.guilds_store.get(GUILD).channels}
    assert chans["30"].visible is False and chans["30"].readable_history is True
    assert chans["31"].visible is True and chans["31"].readable_history is True
    hidden_world.a.text_channels[-1].history = False
    await bot._sync_guild(hidden_world.a)
    assert {c.id: c.readable_history for c in bot.guilds_store.get(GUILD).channels}["31"] is False
