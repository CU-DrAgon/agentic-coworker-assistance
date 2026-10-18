"""The real SDK objects, offline: request models are valid and the Bolt app is wired."""

from types import SimpleNamespace

import pytest
from conftest import FixedClockSettings
from platform_fakes import make_assistant

from salid.platforms import line, slack
from salid.platforms.config import PlatformSettings


class Spy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kw):
            self.calls.append((name, args, kw))
            if name in ("reply_message", "push_message"):
                return SimpleNamespace(sent_messages=[SimpleNamespace(id="m1")])
            if name == "get_message_content":
                return bytearray(b"data")
            return SimpleNamespace(
                display_name="Alice", group_name="Ops", user_id="Ubot", basic_id="@b"
            )

        return call


def test_line_requests_are_valid_sdk_models():
    api = line.SdkLineApi("token")
    spy = Spy()
    api.api, api.blob = spy, spy
    assert api.reply("tok", ["a", "b"]) == ["m1"]
    assert api.push("Cgroup", ["x" * 5000]) == ["m1"]
    api.loading("Ualice", 30)
    assert api.profile("group", "Cg", "Ua") == "Alice"
    assert api.profile("room", "Rr", "Ua") == "Alice"
    assert api.profile("user", "Ua", "Ua") == "Alice"
    assert api.group_name("group", "Cg") == "Ops" and api.group_name("room", "Rr") == ""
    assert api.content("123") == b"data"
    assert api.bot_info()["user_id"] == "Ubot"
    names = [c[0] for c in spy.calls]
    assert names[:2] == ["reply_message", "push_message"]
    reply = spy.calls[0][1][0]
    assert reply.reply_token == "tok" and [m.text for m in reply.messages] == ["a", "b"]
    assert spy.calls[2][1][0].chat_id == "Ualice"


def test_line_download_over_the_limit_is_refused():
    api = line.SdkLineApi("token")

    class Big(Spy):
        def get_message_content(self, mid):
            return bytearray(b"x" * (15 * 1024 * 1024 + 1))

    api.blob = Big()
    with pytest.raises(ValueError):
        api.content("1")


async def test_slack_app_is_wired_without_a_network(tmp_path, monkeypatch):
    from slack_sdk.web.async_client import AsyncWebClient

    async def auth_test(self, **kw):
        return {"team_id": "T9", "user_id": "UBOT", "team": "Acme"}

    monkeypatch.setattr(AsyncWebClient, "auth_test", auth_test)
    s = FixedClockSettings(exec_user_ids=set())
    a = make_assistant(s, tmp_path)
    ps = PlatformSettings(_env_file=None, slack_bot_token="xoxb-a", slack_app_token="xapp-b")
    handler, adapter, port = await slack.connect(ps, a)
    assert (port.team_id, port.bot_user_id) == ("T9", "UBOT")
    assert (
        len(handler.app._async_listeners) == 4
    )  # message, app_mention, /salid command, lifecycle events
    info = await slack.check(ps)
    assert info == {"team": "Acme", "team_id": "T9", "bot_user_id": "UBOT"}
