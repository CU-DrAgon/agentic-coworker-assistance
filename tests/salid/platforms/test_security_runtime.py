"""Secrets and message text never reach the logs; shared-store safety; wiring; smoke output."""

import logging
from datetime import timedelta

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, FakeAsk, FakePort, inc, make_assistant

from salid.guild_settings import MemoryGuildStore
from salid.memory import build_message
from salid.platforms import ids, smoke
from salid.platforms.config import PlatformSettings
from salid.platforms.line import LineAdapter, LinePort
from salid.platforms.runtime import PlatformScheduler, build_assistant
from salid.platforms.slack import SlackAdapter, SlackPort
from salid.store import SQLiteStore

SECRETS = ("xoxb-LEAK-bot", "xapp-LEAK-app", "line-LEAK-secret", "line-LEAK-token")
USER_TEXT = "my iban is TH99-SECRET-4242"


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


def all_secrets_ps(**kw):
    return PlatformSettings(
        _env_file=None,
        slack_bot_token=SECRETS[0],
        slack_app_token=SECRETS[1],
        line_channel_secret=SECRETS[2],
        line_channel_access_token=SECRETS[3],
        **kw,
    )


def assert_clean(text: str):
    for needle in (*SECRETS, "TH99-SECRET-4242", "boom-detail"):
        assert needle not in text


async def test_failures_log_types_not_messages_or_tokens(s, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    class Exploding:
        """Every API call fails with an error text that quotes the token and the user text."""

        def __getattr__(self, name):
            async def call(*a, **k):
                raise RuntimeError(f"boom-detail {SECRETS[0]} {USER_TEXT}")

            return call

    ps = all_secrets_ps()
    a = make_assistant(s, tmp_path, ps=ps, ask=FakeAsk(fail=True))
    port = SlackPort(Exploding(), "T1", "UBOT")
    adapter = SlackAdapter(a, port, lambda url: (_ for _ in ()).throw(OSError(SECRETS[0])))
    await adapter.on_message(
        {"channel": "C1", "channel_type": "channel", "user": "U1", "ts": "1.0",
         "text": f"<@UBOT> {USER_TEXT}", "files": [{"name": "a.png", "mimetype": "image/png",
         "size": 3, "url_private_download": "https://files.slack.com/a"}]},
        "T1",
    )  # fmt: skip
    await a.drain()

    class BadLine:
        def reply(self, *a):
            raise RuntimeError(f"boom-detail {SECRETS[3]} {USER_TEXT}")

        push = loading = profile = group_name = content = reply

    lport = LinePort(BadLine(), ps)
    ladapter = LineAdapter(a, lport, "Ubot")
    await ladapter.on_event(
        {"type": "message", "timestamp": 1, "replyToken": "t",
         "source": {"type": "user", "userId": "Ux"},
         "message": {"type": "text", "id": "1", "text": USER_TEXT}}, "Ubot")  # fmt: skip
    await a.drain()
    assert caplog.records  # something was logged...
    assert_clean(caplog.text)  # ...and none of it is a secret or user content


def test_settings_never_print_secrets():
    ps = all_secrets_ps()
    assert_clean(repr(ps) + str(ps) + repr(ps.model_fields_set))
    assert "slack_bot_token" not in repr(ps)


async def test_smoke_prints_no_secrets_and_reports_failures(capsys):
    ps = all_secrets_ps()
    lines = smoke.plan(ps)
    assert lines[0] == "slack: will start" and lines[1].startswith("line:  will start")

    async def bad(_ps):
        raise RuntimeError(f"401 {SECRETS[0]}")

    async def good(_ps):
        return {"team": "Acme", "bot_user_id": "UBOT", "empty": ""}

    out = await smoke.check(ps, slack_check=bad, line_check=good)
    assert out[0].startswith("slack: token check FAILED") and "RuntimeError" in out[0]
    assert out[1] == "line: token ok (team=Acme, bot_user_id=UBOT)"
    assert_clean("\n".join([*lines, *out]))
    none = PlatformSettings(_env_file=None, slack_bot_token="x")
    assert "missing SALID_SLACK_APP_TOKEN" in smoke.plan(none)[0]
    assert await smoke.check(none) == []


# ------------------------------------------------------------------ shared store safety


def _msg(gid, cid, mid, days_ago, s):
    when = NOW - timedelta(days=days_ago)
    return build_message(
        s, guild_id=gid, channel_id=cid, message_id=mid, author_id=1, author_name="x",
        created_at=when, text=f"m{mid}",
    )  # fmt: skip


def test_the_platform_purge_never_applies_the_default_retention_to_discord_guilds(s, tmp_path):
    a = make_assistant(s, tmp_path)
    store = a.memory_store
    discord_guild, tenant = 1000, ids.tenant_id("slack", "T1")
    store.add_message(_msg(discord_guild, 5, 1, 200, s))  # older than the 90-day default
    store.add_message(_msg(tenant, 6, 2, 200, s))
    store.add_message(_msg(tenant, 6, 3, 1, s))
    sched = PlatformScheduler(
        store, s, None, now=lambda: NOW, retention_of=lambda gid: 30, enabled=lambda gid: True
    )
    sched.tick([tenant])
    day = lambda d: (NOW - timedelta(days=d)).astimezone(s.zone).date().isoformat()  # noqa: E731
    kept_discord = store.messages(discord_guild, [5], day(400), day(0), limit=10)
    kept_tenant = store.messages(tenant, [6], day(400), day(0), limit=10)
    assert [m.text for m in kept_discord] == ["m1"]  # untouched: it is not ours to purge
    assert [m.text for m in kept_tenant] == ["m3"]  # the tenant's own 30-day retention applied


def test_build_assistant_wires_the_shared_stores(s, tmp_path):
    s2 = FixedClockSettings(exec_user_ids=set(), db_path=str(tmp_path / "x.db"))
    a = build_assistant(
        s2, all_secrets_ps(), store=SQLiteStore(s2.db_path), llm=object(),
        guilds=MemoryGuildStore(collection="g"),
    )  # fmt: skip
    assert a.memory_store.get_summary(1, 1, "2026-10-14") is None
    assert a.tenants == {} and isinstance(a.memory.scheduler, PlatformScheduler)
    assert a.exec_ids["slack"] == set()


async def test_tenant_registration_writes_a_guild_document_once(s, tmp_path):
    a, port = make_assistant(s, tmp_path), FakePort()
    a.guilds_store = a.team.store
    for i in range(3):
        await a.handle(inc(message=f"{i}.0"), port)
    await a.drain()
    gid = ids.tenant_id("slack", "T1")
    doc = a.team.store.get(gid)
    assert doc.guild_name == "Slack T1" and doc.installed_at and doc.active
    assert a.tenants[gid] == ("slack", "T1")
