"""Guild settings persist in SQLite in local mode (SqliteGuildStore)."""

import sqlite3
import threading

import pytest
from conftest import GUILD, OTHER_GUILD, FixedClockSettings

from salid.guild_settings import (
    GuildSettingsUpdate,
    LinkError,
    MemoryGuildStore,
    SqliteGuildStore,
    make_guild_store,
)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "sub" / "ledger.db")


@pytest.fixture
def gs(path):
    return SqliteGuildStore(path)


def install(store, gid=GUILD):
    store.upsert_guild(gid, name="Shop", icon_url="i.png", channels=[{"id": "10", "name": "r"}])


def test_settings_round_trip_and_survive_new_instance(gs, path):
    install(gs)
    gs.save_settings(GUILD, GuildSettingsUpdate(digest_time="08:30"), "a@x.com")
    gs.set_channel_mode(GUILD, 5, "chat")
    gs.set_channel_language(GUILD, 5, "en")
    gs.set_channel_tone(GUILD, 5, "calm")
    gs.set_server_tone(GUILD, "friendly")
    gs.set_tone_note(GUILD, "short please")
    again = SqliteGuildStore(path).get(GUILD)
    assert again.guild_name == "Shop" and again.digest_time == "08:30"
    assert again.channel_modes == {"5": "chat"}
    assert again.channel_languages == {"5": "en"}
    assert again.channel_tones == {"5": "calm"}
    assert again.default_tone == "friendly" and again.tone_note == "short please"


def test_server_language_and_restore_rule(gs, path):
    install(gs)
    gs.set_channel_language(GUILD, 5, "en")
    gs.set_server_language(GUILD, "th")
    g = SqliteGuildStore(path).get(GUILD)
    assert g.default_language == "th" and g.channel_languages == {} and g.language_chosen
    gs.set_server_language(GUILD, "en")
    assert gs.get(GUILD).channel_languages == {}


def test_language_asked_and_chosen(gs, path):
    install(gs)
    assert not gs.get(GUILD).language_asked_at
    gs.mark_language_asked(GUILD)
    g = SqliteGuildStore(path).get(GUILD)
    assert g.language_asked_at and not g.language_chosen
    gs.set_server_language(GUILD, "en")
    assert SqliteGuildStore(path).get(GUILD).language_chosen


def test_link_code_create_and_consume(gs, path):
    install(gs)
    code = gs.create_link_code(GUILD, 42, "Boss#1")
    assert SqliteGuildStore(path).peek_link_code(code) == GUILD
    gid, g = gs.redeem_link_code(code, "Boss@Example.com")
    assert gid == GUILD and g.admin_emails == ["boss@example.com"]
    with pytest.raises(LinkError):
        SqliteGuildStore(path).redeem_link_code(code, "b@x.com")


def test_update_on_missing_doc_raises_like_memory(gs):
    with pytest.raises(KeyError):
        gs.client.collection("c").document("nope").update({"a": 1})
    with pytest.raises(KeyError):
        MemoryGuildStore().client.collection("c").document("nope").update({"a": 1})


def test_list_guilds_filters_by_admin_email(gs):
    install(gs, GUILD)
    install(gs, OTHER_GUILD)
    gs.redeem_link_code(gs.create_link_code(GUILD, 1, "u"), "a@x.com")
    assert [g for g, _ in gs.list_guilds("a@x.com")] == [GUILD]
    assert {g for g, _ in gs.list_guilds()} == {GUILD, OTHER_GUILD}


def test_concurrent_writes(gs):
    install(gs)
    errors = []

    def work(n):
        try:
            for i in range(20):
                gs.set_channel_mode(GUILD, n * 100 + i, "chat")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(n,)) for n in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(gs.get(GUILD).channel_modes) == 200


def test_shares_a_file_with_other_connections(gs, path):
    install(gs)
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE IF NOT EXISTS other (x)")
        n = c.execute("SELECT COUNT(*) FROM salid_guild_docs").fetchone()[0]
    assert n == 1


def test_make_guild_store_picks_backend(path):
    s = make_guild_store(FixedClockSettings(store="sqlite", db_path=path))
    assert isinstance(s, SqliteGuildStore)
    m = make_guild_store(FixedClockSettings(store="sqlite", db_path=":memory:"))
    assert isinstance(m, MemoryGuildStore)
