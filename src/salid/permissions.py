"""Missing Discord permissions: which ones exactly, and the one-click re-authorize link.

Every place that cannot do its job for lack of a permission (scheduling, channel creation, voice,
attachments) builds its reply here: the exact missing names, one or two natural sentences, and
the OAuth link an admin opens once to re-grant the bot's full permission set.
"""

from __future__ import annotations

from collections.abc import Iterable

from .i18n import t

# Keep equal to meeting_scribe.app.SALID_PERMISSIONS (a test checks it): View Channels, Send
# Messages, Embed Links, Attach Files, Read Message History, Add Reactions, Manage Channels,
# Connect, Create Events and Manage Events.
SALID_PERMISSIONS = 1166416 | (1 << 44) | (1 << 33) | (1 << 49)  # + Send Polls

# Voice helpers ("Salid · 2"...) only listen and speak: View Channel, Connect, Speak, Use Voice
# Activity.
HELPER_PERMISSIONS = (1 << 10) | (1 << 20) | (1 << 21) | (1 << 25)

LABELS = {
    "view_channel": "View Channel",
    "send_messages": "Send Messages",
    "embed_links": "Embed Links",
    "attach_files": "Attach Files",
    "read_message_history": "Read Message History",
    "add_reactions": "Add Reactions",
    "manage_channels": "Manage Channels",
    "connect": "Connect",
    "speak": "Speak",
    "manage_events": "Manage Events",
    "create_events": "Create Events",
    "send_polls": "Send Polls",
}


def missing_permissions(perms, needed: Iterable[str]) -> list[str]:
    """The `needed` permission attributes that `perms` (discord Permissions) lacks. No
    permissions object at all means everything is missing."""
    return [n for n in needed if not getattr(perms, n, False)]


def guild_missing(guild, needed: Iterable[str]) -> list[str]:
    return missing_permissions(
        getattr(getattr(guild, "me", None), "guild_permissions", None), needed
    )


def channel_missing(channel, guild, needed: Iterable[str]) -> list[str]:
    me = getattr(guild, "me", None)
    try:
        perms = channel.permissions_for(me)
    except Exception:
        perms = None
    return missing_permissions(perms, needed)


def reauthorize_url(app_id) -> str:
    """Discord's "add bot" link with the full permission set; empty without an application id."""
    if not str(app_id or "").isdigit():
        return ""
    return (
        f"https://discord.com/oauth2/authorize?client_id={app_id}"
        f"&scope=bot+applications.commands&permissions={SALID_PERMISSIONS}"
    )


def helper_invite_url(app_id) -> str:
    """Invite link of a voice helper (voice-only permissions); empty without an application id."""
    if not str(app_id or "").isdigit():
        return ""
    return (
        f"https://discord.com/oauth2/authorize?client_id={app_id}"
        f"&scope=bot&permissions={HELPER_PERMISSIONS}"
    )


def permission_message(
    missing: Iterable[str], lang: str, app_id=None, *, helper: bool = False
) -> str:
    """One or two plain sentences naming only what is missing, plus the re-authorize link."""
    names = [LABELS.get(m, m.replace("_", " ").title()) for m in missing]
    joined = (" และ " if lang == "th" else " and ").join(
        [", ".join(names[:-1]), names[-1]] if len(names) > 1 else names
    )
    url = helper_invite_url(app_id) if helper else reauthorize_url(app_id)
    key = "perm_missing" if url else "perm_missing_nolink"
    return t(key, lang, what=joined or "-", url=url)
