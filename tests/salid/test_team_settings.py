"""Hub-managed team settings: validation, Firestore round trip, fail-open reads, bot gating."""

from types import SimpleNamespace

import pytest
from conftest import FixedClockSettings
from fake_firestore import FakeFirestore
from pydantic import ValidationError

from salid.bot import DEEP_OFF_MESSAGE, QA_OFF_MESSAGE, SecretaryBot
from salid.team_settings import (
    BotTeamSettings,
    TeamSettings,
    TeamSettingsStore,
    make_bot_team_settings,
)


@pytest.fixture
def tstore():
    return TeamSettingsStore(client=FakeFirestore())


def test_missing_document_means_defaults(tstore):
    t = tstore.get()
    assert t.features.model_dump() == dict(receipts=True, qa=True, digest=True, deep_tasks=True)
    assert t.receipt_channel_ids == [] and t.digest_time == "18:00"


def test_save_round_trip_and_layout(tstore):
    saved = tstore.save(
        TeamSettings(
            features={"qa": False}, ask_channel_ids=["12", "12", " 34 "], digest_time="07:30"
        ),
        updated_by="boss@example.com",
    )
    doc = tstore.client.collection("salid_teams").docs["default"]
    assert doc["features"]["qa"] is False and doc["features"]["receipts"] is True
    assert doc["ask_channel_ids"] == ["12", "34"] and doc["updated_by"] == "boss@example.com"
    assert doc["updated_at"] == saved.updated_at != ""
    assert tstore.get() == saved
    assert set(tstore.client.cols) == {"salid_teams"}


@pytest.mark.parametrize(
    "bad",
    [
        {"receipt_channel_ids": ["abc"]},
        {"ask_channel_ids": ["1; drop"]},
        {"digest_channel_id": "x1"},
        {"digest_time": "25:00"},
        {"digest_time": "9:00"},
    ],
)
def test_validation_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        TeamSettings(**bad)


class Boom:
    def get(self):
        raise RuntimeError("firestore down")


def test_bot_settings_fail_open_and_cache(tstore):
    now = [0.0]
    reads = []
    real_get = tstore.get
    tstore.get = lambda: reads.append(1) or real_get()
    team = BotTeamSettings(tstore, ttl=60, clock=lambda: now[0])
    assert team.enabled("qa") and len(reads) == 1
    tstore.save(TeamSettings(features={"qa": False}), "x")
    assert team.enabled("qa") and len(reads) == 1  # cached
    now[0] = 61
    assert not team.enabled("qa") and len(reads) == 2
    team._store = Boom()  # outage: keep the last known value rather than flipping features
    now[0] = 200
    assert not team.enabled("qa")
    assert BotTeamSettings(Boom()).enabled("receipts")  # no previous value: defaults, all on


def test_channel_and_digest_rules(tstore):
    team = BotTeamSettings(tstore)
    assert team.receipt_channel_ids({1}) == {1}
    assert team.digest_channel_id(5) == 5 and team.digest_time("19:00") == "19:00"
    tstore.save(
        TeamSettings(receipt_channel_ids=["2"], digest_channel_id="9", digest_time="06:15"), "x"
    )
    team = BotTeamSettings(tstore)
    assert team.receipt_channel_ids({1}) == {1, 2}  # extends the env list
    assert team.digest_channel_id(5) == 9 and team.digest_time("19:00") == "06:15"  # overrides


def test_sqlite_mode_never_touches_firestore():
    team = make_bot_team_settings(FixedClockSettings(store="sqlite"))
    assert team._store is None and all(team.enabled(f) for f in ("receipts", "qa", "digest"))
    fs = make_bot_team_settings(FixedClockSettings(store="firestore", team_id="t1"))
    assert fs._store.collection == "salid_teams" and fs._store.team_id == "t1"


# ------------------------------------------------------------------ bot gating
class Msg:
    def __init__(self, channel_id=10, attachments=(), content="hi", guild=True):
        self.author = SimpleNamespace(id=7, bot=False)
        self.channel = SimpleNamespace(id=channel_id)
        self.attachments, self.content, self.guild = list(attachments), content, guild
        self.mentions, self.reference, self.replies = [], None, []

    async def reply(self, text, **kw):
        self.replies.append(text)


def make_bot(tstore, seeded, **features):
    s = FixedClockSettings(receipt_channel_ids={10}, ask_channel_ids={20})
    tstore.save(TeamSettings(features=features), "x")
    bot = SecretaryBot(s, seeded, None, team=BotTeamSettings(tstore))
    bot.calls = []

    async def intake(m):
        bot.calls.append("intake")

    bot._intake = intake
    return bot


async def test_receipts_toggle(tstore, seeded):
    bot = make_bot(tstore, seeded, receipts=False)
    await bot.on_message(Msg(10, attachments=["img"]))
    assert bot.calls == []  # ignored silently
    bot = make_bot(tstore, seeded)
    await bot.on_message(Msg(10, attachments=["img"]))
    assert bot.calls == ["intake"]


async def test_qa_toggle_replies_briefly(tstore, seeded, monkeypatch):
    asked = []
    monkeypatch.setattr("salid.bot.ask", lambda *a, **k: asked.append(1))
    bot = make_bot(tstore, seeded, qa=False)
    m = Msg(20, content="ยอดวันนี้")
    await bot.on_message(m)
    assert m.replies == [QA_OFF_MESSAGE] and not asked


async def test_deep_toggle(tstore, seeded):
    bot = make_bot(tstore, seeded, deep_tasks=False)
    m = Msg(20, content="งานใหญ่: ทำรายงาน")
    await bot.on_message(m)
    assert m.replies == [DEEP_OFF_MESSAGE] and not bot._deep_tasks


async def test_hub_channel_ids_extend_env_lists(tstore, seeded):
    bot = make_bot(tstore, seeded)
    tstore.save(TeamSettings(receipt_channel_ids=["55"]), "x")
    bot.team = BotTeamSettings(tstore)
    await bot.on_message(Msg(55, attachments=["img"]))
    assert bot.calls == ["intake"]


def test_digest_time_comes_from_hub(tstore, seeded):
    from datetime import datetime

    bot = make_bot(tstore, seeded)
    tstore.save(TeamSettings(digest_time="07:30"), "x")
    bot.team = BotTeamSettings(tstore)
    assert bot.next_digest_at(datetime(2026, 10, 14, 6, 0, tzinfo=bot.s.zone)).hour == 7
