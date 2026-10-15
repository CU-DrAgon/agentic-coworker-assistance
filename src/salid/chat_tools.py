"""Agent tools for everything a slash command can do, so people can just ask in words.

The bot binds the callables in `ChatActions` to the guild, channel and author of the message
(none of those is ever a tool argument) and applies the same permission rules as the slash
commands: changing settings needs a server manager or executive. Each tool returns a small dict
the model turns into one short confirmation sentence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChatActions:
    set_channel_mode: Callable[[str, str | None], dict] | None = None
    set_channel_language: Callable[[str, str | None], dict] | None = None
    set_memory_days: Callable[[int], dict] | None = None
    show_settings: Callable[[], dict] | None = None
    hidden_channels: Callable[[], dict] | None = None
    cancel_deep_task: Callable[[], dict] | None = None
    send_dm: Callable[[str, str, str | None], dict] | None = None


def build_chat_tools(actions: ChatActions, calls: list[tuple[str, dict]]) -> list:
    """The agent-facing functions (docstrings are the tool descriptions)."""

    def logged(name: str, args: dict, fn):
        calls.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except TimeoutError:
            return {"error": "timed out talking to Discord"}
        except Exception:
            logger.exception("chat tool %s failed", name)
            return {"error": "internal error"}

    def set_channel_mode(mode: str, channel: str | None = None) -> dict:
        """Change how much Salid takes part in a channel (admins / executives only). Use when
        someone asks in words, e.g. "ห้องนี้ไม่ต้องตอบนะ" (off), "ฟังเฉยๆ" (listen), "ตอบทุกข้อความ"
        (chat), "ช่วยตอบเองถ้ามีคำถาม" (join), "เรียกแล้วค่อยตอบ" (call).

        Args:
            mode: off (ignore and do not remember the channel), listen (remember silently, answer
                only when tagged), call (answer when tagged or called by name; the default), join
                (like call, and chip in with help unprompted), chat (answer every message).
            channel: the channel name or mention; omit for the channel being talked in.
        """
        return logged(
            "set_channel_mode",
            {"mode": mode, "channel": channel},
            lambda: actions.set_channel_mode(mode, channel),
        )

    def set_channel_language(lang: str, channel: str | None = None) -> dict:
        """Set the language Salid replies in for a channel (admins / executives only), e.g.
        "ช่วยพูดอังกฤษในห้องนี้" or "answer in Thai here".

        Args:
            lang: th, en, or auto (follow the language each person writes in).
            channel: the channel name or mention; omit for the channel being talked in.
        """
        return logged(
            "set_channel_language",
            {"lang": lang, "channel": channel},
            lambda: actions.set_channel_language(lang, channel),
        )

    def set_memory_days(days: int) -> dict:
        """Set how many days Salid keeps the server's chat memory (admins / executives only),
        e.g. "จำแค่ 30 วันพอ".

        Args:
            days: number of days to keep.
        """
        return logged("set_memory_days", {"days": days}, lambda: actions.set_memory_days(days))

    def show_settings() -> dict:
        """Show the current settings for this channel and server: mode, reply language, memory
        days. Anyone may ask ("ตั้งค่าตอนนี้เป็นยังไง")."""
        return logged("show_settings", {}, actions.show_settings)

    def hidden_channels() -> dict:
        """List the channels Salid cannot see or read back, and how to let it in (admins /
        executives only). Use for "ทำไมสลิดไม่เห็นห้อง X" or "which channels can't you see"."""
        return logged("hidden_channels", {}, actions.hidden_channels)

    def cancel_deep_task() -> dict:
        """Cancel the big background task that is running (whoever started it, a server manager
        or an executive), e.g. "ยกเลิกงานที่ทำอยู่"."""
        return logged("cancel_deep_task", {}, actions.cancel_deep_task)

    def send_dm(text: str, to: str = "me", attach: str | None = None) -> dict:
        """Send a private message (DM) from Salid to a member of this server. Salid CAN do this.
        Use for "ส่งสรุปเข้า DM ผม", "ส่ง PDF ให้ผมทาง DM", "DM รายงานให้หัวหน้า", "ฝากบอกพี่มิวทาง
        DM ว่า...". When to is someone else the DM says who it is from, so put only what the
        requester asked to pass on (never ledger figures, memory or other people's data that they
        have not shown the recipient). On success confirm in one short sentence; on dms_closed
        pass on the guidance given.

        Args:
            text: the message (short; for "me" it may be a summary or answer you prepared).
            to: "me" (the requester, the default) or the member's display name as listed in the
                people you may tag.
            attach: file_name of a PDF made earlier in this conversation (make_report_pdf) to
                attach.
        """
        return logged(
            "send_dm",
            {"to": to, "text": text, "attach": attach},
            lambda: actions.send_dm(text, to, attach),
        )

    tools = {
        "set_channel_mode": set_channel_mode,
        "set_channel_language": set_channel_language,
        "set_memory_days": set_memory_days,
        "show_settings": show_settings,
        "hidden_channels": hidden_channels,
        "cancel_deep_task": cancel_deep_task,
        "send_dm": send_dm,
    }
    return [fn for name, fn in tools.items() if getattr(actions, name) is not None]
