"""A private Salid channel per person inside each company's server.

The server-first way to talk to Salid with clear context: a text channel only that member (and
Salid, and anyone with Administrator who can technically see every channel) can read. Created on
request, never for everyone. Pure rules and storage here; the Discord glue is `private_chat_bot`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import discord

from .coo_store import PRIVATE_CHATS, DocStore

LEAVE_GRACE = timedelta(hours=24)  # a private channel outlives its member by this much
GUILD_CHANNEL_CAP = 450  # Discord allows 500 per server: stop auto-creating well before
BACKFILL_PACE_S = 2.0  # seconds between channels made for existing members
CATEGORY_LIMIT = 50  # Discord: channels per category
CHANNEL_PREFIX = "สลิด"
PRIVATE_COMMANDS = ("ห้องส่วนตัว", "private-chat")
MAX_NAME = 90

_ASK = re.compile(
    r"(?:ขอ|สร้าง|เปิด|อยาก(?:ได้|มี)|ทำ)\s*ห้อง(?:คุย)?(?:ส่วนตัว)(?:\s*(?:กับ\s*สลิด|ให้))?"
    r"|ห้องคุยส่วนตัว|ห้องส่วนตัว(?:กับ\s*สลิด)?"
    r"|(?:make|create|open|give|set\s*up)\s+(?:me\s+)?(?:a\s+|my\s+)?private\s+(?:chat\s+)?(?:channel|room)"
    r"|private\s+(?:chat\s+)?(?:channel|room)\s+(?:with|for)",
    re.I,
)
_DELETE = re.compile(
    r"(?:ลบ|ปิด|เลิกใช้)\s*ห้อง(?:คุย)?ส่วนตัว|(?:delete|remove|close)\s+(?:my\s+)?private\s+"
    r"(?:chat\s+)?(?:channel|room)",
    re.I,
)
_CONFIRM = re.compile(r"^\s*(?:ใช่|ยืนยัน|ตกลง|y|yes|ok|okay|confirm)\s*[.!ค่ะครับนะคะ]*$", re.I)


def wants_private_channel(text: str) -> bool:
    s = text or ""
    return len(s) <= 200 and not _DELETE.search(s) and bool(_ASK.search(s))


def wants_delete(text: str) -> bool:
    return len(text or "") <= 200 and bool(_DELETE.search(text or ""))


def is_confirm(text: str) -> bool:
    return bool(_CONFIRM.match(text or ""))


def channel_name(nickname: str) -> str:
    """`สลิด-<nickname>`: lower case, spaces to dashes, Thai kept, other symbols dropped."""
    slug = re.sub(r"[^\w฀-๿-]+", "-", (nickname or "").strip().casefold())
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return f"{CHANNEL_PREFIX}-{slug}"[:MAX_NAME] if slug else CHANNEL_PREFIX


def category_name(base: str, n: int) -> str:
    return base if n <= 1 else f"{base} {n}"


def overwrites(guild, member) -> dict:
    """@everyone cannot see it; the member can read / write / attach; Salid can also manage the
    channel. Only permissions Salid holds guild-wide may be granted here (no Manage Messages)."""
    return {
        guild.default_role: discord.PermissionOverwrite(view_channel=False),
        member: discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            attach_files=True,
        ),
        guild.me: discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            attach_files=True,
            embed_links=True,
            manage_channels=True,
        ),
    }


def pick_category(categories, base: str):
    """(existing category with room or None, name for a new one) following `base`, `base 2`..."""
    n = 1
    while True:
        name = category_name(base, n)
        found = next((c for c in categories if c.name == name), None)
        if found is None:
            return None, name
        if len(found.channels) < CATEGORY_LIMIT:
            return found, name
        n += 1


class PrivateChatRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def get(self, guild_id: int, user_id: int) -> int | None:
        doc = self.docs.get(PRIVATE_CHATS, guild_id, str(user_id))
        return int(doc["channel_id"]) if doc else None

    def put(self, guild_id: int, user_id: int, channel_id: int) -> None:
        self.docs.put(
            PRIVATE_CHATS,
            guild_id,
            str(user_id),
            {
                "channel_id": channel_id,
                "user_id": user_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )

    def schedule_delete(self, guild_id: int, user_id: int, at: datetime) -> None:
        """The member left: the channel goes at `at` (persisted, so a restart keeps the promise)."""
        doc = self.docs.get(PRIVATE_CHATS, guild_id, str(user_id))
        if doc:
            self.docs.put(
                PRIVATE_CHATS, guild_id, str(user_id), {**doc, "delete_at": at.isoformat()}
            )

    def cancel_delete(self, guild_id: int, user_id: int) -> bool:
        doc = self.docs.get(PRIVATE_CHATS, guild_id, str(user_id))
        if not doc or not doc.get("delete_at"):
            return False
        doc.pop("delete_at")
        self.docs.put(PRIVATE_CHATS, guild_id, str(user_id), doc)
        return True

    def due_deletions(self, guild_id: int, now: datetime) -> list[int]:
        docs = self.docs.find(
            PRIVATE_CHATS,
            guild_id,
            lambda d: bool(d.get("delete_at")) and datetime.fromisoformat(d["delete_at"]) <= now,
        )
        return [int(d["user_id"]) for d in docs]

    def delete(self, guild_id: int, user_id: int) -> None:
        self.docs.delete(PRIVATE_CHATS, guild_id, str(user_id))

    def owner_of(self, guild_id: int, channel_id: int) -> int | None:
        for d in self.docs.find(
            PRIVATE_CHATS, guild_id, lambda d: d.get("channel_id") == channel_id
        ):
            return int(d["user_id"])
        return None
