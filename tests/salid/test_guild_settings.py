"""Per-guild settings: validation, Firestore layout, fail-open cache, link codes."""

from datetime import timedelta

import pytest
from conftest import GUILD, OTHER_GUILD, FixedClockSettings
from pydantic import ValidationError

from salid import guild_settings as gs
from salid.guild_settings import (
    BotGuildSettings,
    GuildSettings,
    GuildSettingsUpdate,
    LinkError,
    MemoryGuildStore,
    make_guild_store,
)


@pytest.fixture
def gstore():
    return MemoryGuildStore()


def install(store, gid=GUILD, name="ร้านกาแฟ", **kw):
    store.upsert_guild(
        gid, name=name, icon_url="i.png", channels=[{"id": "10", "name": "ใบเสร็จ"}], **kw
    )


def test_missing_document_means_defaults(gstore):
    g = gstore.get(GUILD)
    assert g.features.model_dump() == dict(
        receipts=True, qa=True, digest=True, deep_tasks=True, voice_meetings=True
    )
    assert g.receipt_channel_ids == [] and g.digest_time == "18:00" and g.active
    assert g.exec_role_names == ["ผู้บริหาร", "Executive"] and g.admin_emails == []
    assert gstore.find(GUILD) is None


def test_bot_writes_and_hub_writes_do_not_clobber_each_other(gstore):
    install(gstore, installed=True)
    gstore.save_settings(
        GUILD, GuildSettingsUpdate(features={"qa": False}, digest_channel_id="10"), "a@x.com"
    )
    # the bot resyncs (rename + new channel list): hub fields survive
    gstore.upsert_guild(GUILD, name="ใหม่", icon_url="", channels=[{"id": "11", "name": "x"}])
    g = gstore.get(GUILD)
    assert g.guild_name == "ใหม่" and g.channels == [gs.Channel(id="11", name="x")]
    assert g.features.qa is False and g.digest_channel_id == "10" and g.updated_by == "a@x.com"
    assert g.installed_at  # set once, on install
    # and a hub save keeps what the bot wrote
    gstore.save_settings(GUILD, GuildSettingsUpdate(), "b@x.com")
    assert gstore.get(GUILD).guild_name == "ใหม่" and gstore.get(GUILD).channels
    gstore.set_active(GUILD, False)
    assert gstore.get(GUILD).active is False and gstore.get(GUILD).guild_name == "ใหม่"


def test_document_layout(gstore):
    install(gstore, installed=True)
    gstore.save_settings(
        GUILD, GuildSettingsUpdate(ask_channel_ids=["12", "12", " 34 "]), "a@x.com"
    )
    doc = gstore.client.collection("salid_guilds").docs[str(GUILD)]
    assert set(doc) == {
        "guild_name", "icon_url", "channels", "active", "installed_at", "features",
        "receipt_channel_ids", "ask_channel_ids", "digest_channel_id", "digest_time",
        "exec_role_names", "updated_by", "updated_at",
    }  # fmt: skip
    assert doc["ask_channel_ids"] == ["12", "34"] and doc["active"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {"receipt_channel_ids": ["abc"]},
        {"ask_channel_ids": ["1; drop"]},
        {"digest_channel_id": "x1"},
        {"digest_time": "25:00"},
        {"digest_time": "9:00"},
        {"exec_role_names": ["x" * 101]},
    ],
)
def test_validation_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        GuildSettingsUpdate(**bad)


def test_update_model_cannot_carry_admin_fields():
    u = GuildSettingsUpdate(**{"admin_emails": ["evil@x.com"], "active": False})
    assert "admin_emails" not in u.model_dump() and "active" not in u.model_dump()
    assert GuildSettings(admin_emails=["a@x.com"]).admin_emails == ["a@x.com"]


class Boom:
    def get(self, guild_id):
        raise RuntimeError("firestore down")


def test_bot_settings_cache_per_guild_and_fail_open(gstore):
    now = [0.0]
    reads = []
    real_get = gstore.get
    gstore.get = lambda gid: reads.append(gid) or real_get(gid)
    team = BotGuildSettings(gstore, ttl=60, clock=lambda: now[0])
    assert team.enabled(GUILD, "qa") and reads == [GUILD]
    gstore.save_settings(GUILD, GuildSettingsUpdate(features={"qa": False}), "x")
    reads[:] = [GUILD]  # (saving reads the doc back; only the bot's reads are counted)
    assert team.enabled(GUILD, "qa") and reads == [GUILD]  # cached
    assert team.enabled(OTHER_GUILD, "qa") and reads == [GUILD, OTHER_GUILD]  # own entry
    now[0] = 61
    assert not team.enabled(GUILD, "qa")
    team.store = Boom()  # outage: keep the last known value rather than flipping features
    now[0] = 200
    assert not team.enabled(GUILD, "qa")
    assert BotGuildSettings(Boom()).enabled(GUILD, "receipts")  # nothing known: defaults, all on


def test_channel_digest_and_role_rules(gstore):
    team = BotGuildSettings(gstore)
    assert team.receipt_channel_ids(GUILD, {1}) == {1}
    assert team.digest_channel_id(GUILD, 5) == 5 and team.digest_time(GUILD, "19:00") == "19:00"
    assert team.exec_role_names(GUILD, {"Boss"}) == {"Boss"}  # untouched doc: env wins
    gstore.save_settings(
        GUILD,
        GuildSettingsUpdate(
            receipt_channel_ids=["2"], digest_channel_id="9", digest_time="06:15",
            exec_role_names=["Owner"],
        ),
        "x",
    )  # fmt: skip
    team = BotGuildSettings(gstore)
    assert team.receipt_channel_ids(GUILD, {1}) == {1, 2}  # extends the env list
    assert team.digest_channel_id(GUILD, 5) == 9 and team.digest_time(GUILD, "19:00") == "06:15"
    assert team.exec_role_names(GUILD, {"Boss"}) == {"Owner"}  # overrides
    assert team.receipt_channel_ids(OTHER_GUILD, {1}) == {1}  # other guild unaffected


def test_make_guild_store_selects_backend():
    mem = make_guild_store(FixedClockSettings(store="sqlite", guild_collection="g1"))
    assert isinstance(mem, MemoryGuildStore) and mem.collection == "g1"
    fs = make_guild_store(FixedClockSettings(store="firestore"))
    assert type(fs) is gs.GuildStore and fs.collection == "salid_guilds" and fs._client is None


def test_list_guilds_by_admin_email(gstore):
    install(gstore, GUILD, "B shop")
    install(gstore, OTHER_GUILD, "A shop")
    code = gstore.create_link_code(GUILD, 1, "u")
    gstore.redeem_link_code(code, "Owner@Example.com")
    assert [(g, s.guild_name) for g, s in gstore.list_guilds("owner@example.com")] == [
        (GUILD, "B shop")
    ]
    assert gstore.list_guilds("stranger@example.com") == []
    assert [g for g, _ in gstore.list_guilds(None)] == [OTHER_GUILD, GUILD]  # super admin: all


# ------------------------------------------------------------------ link codes
def test_link_code_layout_and_redeem(gstore):
    install(gstore)
    code = gstore.create_link_code(GUILD, 42, "Boss#1")
    assert len(code) >= 43 and code != gstore.create_link_code(GUILD, 42, "Boss#1")  # 32 bytes
    doc = gstore.client.collection("salid_link_codes").docs[code]
    assert doc["guild_id"] == GUILD and doc["discord_user_id"] == 42 and doc["used"] is False
    assert doc["discord_user_name"] == "Boss#1" and doc["expires_at"] > doc["created_at"]
    gid, g = gstore.redeem_link_code(code, "Boss@Example.com")
    assert gid == GUILD and g.admin_emails == ["boss@example.com"] and g.guild_name == "ร้านกาแฟ"
    assert gstore.client.collection("salid_link_codes").docs[code]["used"] is True


def test_link_code_is_single_use(gstore):
    code = gstore.create_link_code(GUILD, 1, "u")
    gstore.redeem_link_code(code, "a@x.com")
    with pytest.raises(LinkError):
        gstore.redeem_link_code(code, "b@x.com")
    assert gstore.get(GUILD).admin_emails == ["a@x.com"]  # the loser did not get in


def test_link_code_expires_and_unknown_codes_fail(gstore, monkeypatch):
    code = gstore.create_link_code(GUILD, 1, "u")
    real = gs._now()
    monkeypatch.setattr(gs, "_now", lambda: real + gs.LINK_TTL + timedelta(seconds=1))
    with pytest.raises(LinkError):
        gstore.redeem_link_code(code, "a@x.com")
    with pytest.raises(LinkError):
        gstore.redeem_link_code("nope", "a@x.com")
    assert gstore.find(GUILD) is None  # failed redeems create nothing
    assert gstore.client.collection("salid_link_codes").docs[code]["used"] is False


def test_second_admin_is_added_not_replaced(gstore):
    install(gstore)
    for email in ("a@x.com", "b@x.com", "a@x.com"):
        gstore.redeem_link_code(gstore.create_link_code(GUILD, 1, "u"), email)
    assert gstore.get(GUILD).admin_emails == ["a@x.com", "b@x.com"]
