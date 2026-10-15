# ruff: noqa: F811, F401
"""Short-term context buffer: recent messages per channel kept from gateway events, so a reply
does not read Discord history every time. Offline: small Discord fakes."""

from datetime import UTC, datetime
from types import SimpleNamespace

from test_modes import bot, say, send, world  # noqa: F401  (fixtures)

from salid.recent import RecentMessages


def msg(mid, cid=10, text="t", at=None, author="a"):
    return SimpleNamespace(id=mid, channel=SimpleNamespace(id=cid), author=author, content=text,
                           created_at=at or datetime(2026, 10, 14, 5, 0, tzinfo=UTC),
                           attachments=[], role_mentions=[])  # fmt: skip


def test_unprimed_channels_return_none_then_buffer_after_prime():
    r = RecentMessages()
    r.add(msg(5))
    assert r.history(10, 10) is None
    r.prime(10, [msg(4), msg(3)])
    assert [s.id for s in r.history(10, 10)] == [5, 4, 3]
    assert [s.id for s in r.history(10, 10, before_id=5)] == [4, 3]


def test_edit_delete_and_bound():
    r = RecentMessages(per_channel=3)
    r.prime(10, [])
    for i in range(1, 6):
        r.add(msg(i))
    r.edit(10, 5, "new")
    r.delete(10, [4])
    assert [(s.id, s.content) for s in r.history(10, 10)] == [(5, "new"), (3, "t")]


def test_reset_and_sweep():
    r = RecentMessages(max_age_s=60)
    r.prime(10, [msg(1)])
    r.reset()
    assert r.history(10, 5) is None
    r.prime(11, [msg(2, cid=11)])
    r.sweep(datetime(2026, 10, 14, 6, 0, tzinfo=UTC))
    assert r.history(11, 5) is None


def test_max_channels_evicts_the_least_recent():
    r = RecentMessages(max_channels=2)
    for cid in (1, 2, 3):
        r.prime(cid, [msg(cid, cid=cid)])
    assert r.history(1, 5) is None and r.history(3, 5) is not None


async def test_bot_reads_history_once_then_from_the_buffer(bot, world):
    calls = []
    original = world.gen.history

    def counting(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    world.gen.history = counting
    bot._start_import = lambda: None  # on_resumed would start a history import
    await send(bot, world, world.gen, "เรื่องงบ")
    m = say(world, world.gen, "สลิด เมื่อกี้ว่าไง")
    await bot.on_message(m)
    a = await bot._recent_chat(m)
    b = await bot._recent_chat(m)
    assert len(calls) == 1 and a == b and "เรื่องงบ" in a
    await bot.on_resumed()
    await bot._recent_chat(m)
    assert len(calls) == 2
