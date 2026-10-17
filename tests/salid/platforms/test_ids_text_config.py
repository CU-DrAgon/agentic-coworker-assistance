from datetime import UTC, datetime

import pytest

from salid.platforms import ids, textfmt
from salid.platforms.config import PlatformSettings


def snowflake(when: datetime) -> int:
    return (int(when.timestamp() * 1000) - 1420070400000) << 22 | 0x3FFFFF


def test_tenant_ids_are_stable_and_distinct():
    a = ids.tenant_id("slack", "T123")
    assert a == ids.tenant_id("slack", "T123")
    assert a != ids.tenant_id("slack", "T124")
    assert a != ids.tenant_id("line", "T123")  # same key on another platform is another tenant
    assert ids.platform_of(a) == "slack" and ids.platform_of(ids.tenant_id("line", "U1")) == "line"


def test_tenant_ids_never_collide_with_discord_and_fit_int64():
    far_future = snowflake(datetime(2045, 1, 1, tzinfo=UTC))
    assert far_future < ids.DISCORD_CEILING
    for platform in ("slack", "line"):
        for key in ("T1", "T2", "Ud5f0a", "x" * 40):
            t = ids.tenant_id(platform, key)
            assert ids.DISCORD_CEILING <= t < 2**63
            assert ids.is_platform_tenant(t)
    assert not ids.is_platform_tenant(far_future)
    assert ids.platform_of(far_future) is None


def test_user_room_message_ids_are_positive_and_scoped():
    u = ids.user_id("slack", "U1")
    assert 0 < u < ids.DISCORD_CEILING and u == ids.user_id("slack", "U1")
    assert u != ids.user_id("line", "U1")
    r = ids.room_id("slack", "T1", "C1")
    assert r > 0 and r != ids.room_id("slack", "T2", "C1")  # not DM-note scopes (negative)
    m = ids.message_id("slack", "T1", "C1", "1.2")
    assert m > 0 and m != ids.message_id("slack", "T1", "C2", "1.2")
    assert ids.bot_user_id("slack", "T1") != u


def test_settings_secrets_hidden_and_parsed(monkeypatch):
    ps = PlatformSettings(
        _env_file=None,
        slack_bot_token="xoxb-secret",
        slack_app_token="xapp-secret",
        slack_exec_user_ids="U1, U2 ,",
        line_channel_secret="line-secret",
        line_channel_access_token="line-token",
        line_exec_user_ids="Uabc",
    )
    text = repr(ps) + str(ps)
    for secret in ("xoxb-secret", "xapp-secret", "line-secret", "line-token"):
        assert secret not in text
    assert ps.slack_exec_user_ids == {"U1", "U2"} and ps.line_exec_user_ids == {"Uabc"}
    assert ps.enabled_platforms() == ["slack", "line"]
    assert ps.line_port == 8091


def test_a_platform_starts_only_with_all_its_secrets(monkeypatch):
    for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "LINE_CHANNEL_SECRET"):
        monkeypatch.delenv(f"SALID_{k}", raising=False)
    assert PlatformSettings(_env_file=None).enabled_platforms() == []
    assert PlatformSettings(_env_file=None, slack_bot_token="x").enabled_platforms() == []
    only_line = PlatformSettings(
        _env_file=None, line_channel_secret="s", line_channel_access_token="t"
    )
    assert only_line.enabled_platforms() == ["line"]


# ------------------------------------------------------------------ splitting


def test_split_short_text_and_empty():
    assert textfmt.split_text("hi", 10) == ["hi"]
    assert textfmt.split_text("  \n", 10) == []


def test_split_prefers_paragraphs_then_lines_then_words_and_respects_limit():
    para = ("word " * 50).strip()
    text = f"{para}\n\n{para}\n\n{para}"
    parts = textfmt.split_text(text, 300)
    assert len(parts) == 3 and all(len(p) <= 300 for p in parts)
    long = "x" * 1000
    pieces = textfmt.split_text(long, 300)
    assert all(len(p) <= 300 for p in pieces) and "".join(pieces) == long


def test_split_never_exceeds_limit_counted_in_utf16():
    text = "😀" * 3000  # 2 UTF-16 units each
    parts = textfmt.split_text(text, 4900, textfmt.utf16_len)
    assert len(parts) >= 2 and all(textfmt.utf16_len(p) <= 4900 for p in parts)
    assert "".join(parts) == text
    thai = "สวัสดีค่ะ " * 800
    assert all(
        textfmt.utf16_len(p) <= 4900 for p in textfmt.split_text(thai, 4900, textfmt.utf16_len)
    )


def test_split_keeps_code_fences_balanced():
    code = "```python\n" + "\n".join(f"line {i}" for i in range(200)) + "\n```"
    parts = textfmt.split_text(code, 400)
    assert len(parts) > 1
    assert all(p.count("```") % 2 == 0 for p in parts)
    assert all(len(p) <= 400 for p in parts)


# ------------------------------------------------------------------ Slack / LINE output


def test_to_slack_maps_markdown():
    out = textfmt.to_slack(
        "# Title\n**bold** and *it* and ~~gone~~\n- item\n[site](https://x.io/a)"
    )
    assert "*Title*" in out and "*bold*" in out and "_it_" in out and "~gone~" in out
    assert "• item" in out and "<https://x.io/a|site>" in out


def test_to_slack_escapes_control_sequences_and_keeps_code():
    out = textfmt.to_slack("hey <!channel> <@U123> a & b `<code> & x`")
    assert "<!channel>" not in out and "&lt;!channel&gt;" in out and "&lt;@U123&gt;" in out
    assert "a &amp; b" in out
    assert "`<code> & x`" in "".join(out.split("`<code> & x`")[:1]) or "&lt;code&gt;" in out
    fenced = textfmt.to_slack("```py\nx = '<a>' & 1\n```")
    assert fenced.startswith("```\n") and "&lt;a&gt;" in fenced


def test_to_slack_turns_tables_into_code_blocks():
    out = textfmt.to_slack("| a | b |\n|---|---|\n| 1 | 2 |\n\nafter")
    assert out.startswith("```\n| a | b |") and "---" not in out and "after" in out


def test_to_line_is_plain_text():
    out = textfmt.to_line("## Head\n**bold** `code` [site](https://x.io)\n- one\n\n\n\n- two")
    assert out == "Head\nbold code site (https://x.io)\n• one\n\n• two"


# ------------------------------------------------------------------ Slack input


def test_slack_to_plain():
    names = {"U9": "Mew"}
    out = textfmt.slack_to_plain(
        "<@UBOT> ask <@U9> about <#C1|general> <https://a.io|docs> &amp; <!here>",
        "UBOT",
        names.get,
    )
    assert out == "ask @Mew about #general docs (https://a.io) & @here"
    assert textfmt.slack_mentions("<@UBOT> hi <@U9|mew>") == ["UBOT", "U9"]


@pytest.mark.parametrize("text", ["", None])
def test_slack_to_plain_empty(text):
    assert textfmt.slack_to_plain(text) == ""
