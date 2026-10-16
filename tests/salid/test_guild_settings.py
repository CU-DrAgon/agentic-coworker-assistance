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
        receipts=True,
        qa=True,
        digest=True,
        deep_tasks=True,
        voice_meetings=True,
        auto_join_meetings=True,
        memory=True,
        image_memory=True,
        media_memory=True,
        doc_memory=True,
        read_receipts=True,
        web=True,
        documents=True,
        scheduling=True,
        email=True,
        followups=True,
        reports=True,
        briefings=True,
        approvals=True,
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
        "receipt_channel_ids", "ask_channel_ids", "digest_channel_id", "receipt_home_channel_id", "receipt_forward", "digest_time",
        "exec_role_names", "updated_by", "updated_at", "channel_modes", "default_mode",
        "channel_languages", "default_language", "memory_retention_days",
        "followup_time", "report_time", "report_channel_id", "morning_brief_time", "evening_brief_time",
    }  # fmt: skip
    assert doc["ask_channel_ids"] == ["12", "34"] and doc["active"] is True


def test_channel_modes_are_validated_and_bounded():
    u = GuildSettingsUpdate(
        channel_modes={"12": "join", "x": "chat", "13": "bogus", " 14 ": "off"}, default_mode="zzz"
    )
    assert u.channel_modes == {"12": "join", "14": "off"} and u.default_mode == "call"
    big = GuildSettingsUpdate(channel_modes={str(i): "chat" for i in range(900)})
    assert len(big.channel_modes) == gs.MAX_CHANNEL_MODES
    assert GuildSettingsUpdate(default_mode="join").default_mode == "join"


def test_set_channel_mode_sets_and_removes_one_channel(gstore):
    gstore.set_channel_mode(GUILD, 5, "join")  # no document yet: created
    gstore.set_channel_mode(GUILD, 6, "off")
    assert gstore.get(GUILD).channel_modes == {"5": "join", "6": "off"}
    gstore.set_channel_mode(GUILD, 5, None)
    assert gstore.get(GUILD).channel_modes == {"6": "off"}
    with pytest.raises(ValueError):
        gstore.set_channel_mode(GUILD, 5, "bogus")


def test_save_settings_replaces_the_channel_mode_map(gstore):
    gstore.set_channel_mode(GUILD, 5, "join")
    gstore.save_settings(GUILD, GuildSettingsUpdate(channel_modes={"6": "chat"}), "a@x.com")
    assert gstore.get(GUILD).channel_modes == {"6": "chat"}


def test_bot_cache_sees_a_mode_change_at_once(gstore):
    team = gs.BotGuildSettings(gstore, ttl=3600)
    assert team.channel_mode(GUILD, 5, "x", set()) == "call"
    team.set_channel_mode(GUILD, 5, "listen")
    assert team.channel_mode(GUILD, 5, "x", set()) == "listen"
    assert team.channel_mode(GUILD, 6, "x", {6}) == "chat"  # env ask channel


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


def test_channel_languages_are_validated_and_bounded():
    u = GuildSettingsUpdate(
        channel_languages={"12": "en", "x": "th", "13": "bogus", " 14 ": "auto"},
        default_language="zzz",
    )
    assert u.channel_languages == {"12": "en", "14": "auto"} and u.default_language == "th"
    big = GuildSettingsUpdate(channel_languages={str(i): "en" for i in range(900)})
    assert len(big.channel_languages) == gs.MAX_CHANNEL_LANGUAGES
    assert GuildSettingsUpdate(default_language="auto").default_language == "auto"
    assert GuildSettingsUpdate().default_language == "th"  # today's behaviour


def test_set_channel_language_sets_and_removes_one_channel(gstore):
    gstore.set_channel_language(GUILD, 5, "en")  # no document yet: created
    gstore.set_channel_language(GUILD, 6, "auto")
    assert gstore.get(GUILD).channel_languages == {"5": "en", "6": "auto"}
    gstore.set_channel_language(GUILD, 5, None)
    assert gstore.get(GUILD).channel_languages == {"6": "auto"}
    with pytest.raises(ValueError):
        gstore.set_channel_language(GUILD, 5, "fr")


def test_save_settings_replaces_the_channel_language_map(gstore):
    gstore.set_channel_language(GUILD, 5, "en")
    gstore.save_settings(
        GUILD,
        GuildSettingsUpdate(channel_languages={"6": "th"}, default_language="en"),
        "a@x.com",
    )
    g = gstore.get(GUILD)
    assert g.channel_languages == {"6": "th"} and g.default_language == "en"


def test_bot_cache_sees_a_language_change_at_once_and_resolves_override_then_default(gstore):
    team = gs.BotGuildSettings(gstore, ttl=3600)
    assert team.channel_language(GUILD, 5) == "th"  # nothing stored: today's default
    gstore.save_settings(GUILD, GuildSettingsUpdate(default_language="en"), "a@x.com")
    team._cache.clear()
    assert team.channel_language(GUILD, 5) == "en"
    team.set_channel_language(GUILD, 5, "auto")
    assert team.channel_language(GUILD, 5) == "auto"  # override, no waiting for the TTL
    assert team.channel_language(GUILD, 6) == "en"  # other channels follow the default
    team.set_channel_language(GUILD, 5, None)
    assert team.channel_language(GUILD, 5) == "en"
