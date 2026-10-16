"""Missing-permission replies: exact names, short human text, one-click re-authorize link."""

from types import SimpleNamespace

from salid.permissions import (
    SALID_PERMISSIONS,
    channel_missing,
    guild_missing,
    missing_permissions,
    permission_message,
    reauthorize_url,
)


def test_permission_value_matches_the_web_invite():
    from meeting_scribe.app import SALID_PERMISSIONS as WEB

    assert SALID_PERMISSIONS == WEB


def test_exact_missing_names():
    perms = SimpleNamespace(manage_events=False, create_events=True, connect=True)
    assert missing_permissions(perms, ("manage_events", "create_events", "connect")) == [
        "manage_events"
    ]
    assert missing_permissions(None, ("connect",)) == ["connect"]
    guild = SimpleNamespace(me=SimpleNamespace(guild_permissions=perms))
    assert guild_missing(guild, ("manage_events",)) == ["manage_events"]
    chan = SimpleNamespace(permissions_for=lambda me: SimpleNamespace(attach_files=False))
    assert channel_missing(chan, guild, ("attach_files",)) == ["attach_files"]


def test_reauthorize_link():
    url = reauthorize_url(123)
    assert url == (
        "https://discord.com/oauth2/authorize?client_id=123"
        f"&scope=bot+applications.commands&permissions={SALID_PERMISSIONS}"
    )
    assert reauthorize_url(None) == "" and reauthorize_url("x") == ""


def test_message_names_only_what_is_missing_in_both_languages():
    th = permission_message(["manage_events"], "th", 123)
    en = permission_message(["manage_channels", "attach_files"], "en", 123)
    assert "Manage Events" in th and "Create Events" not in th and "client_id=123" in th
    assert "Manage Channels and Attach Files" in en and "open this link once" in en
    assert "Server Settings" not in th + en
    assert not any(ord(ch) > 0x1F000 for ch in th + en)  # no emoji
    nolink = permission_message(["connect"], "en", None)
    assert "Connect" in nolink and "http" not in nolink
