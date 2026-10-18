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
    set_my_notifications: Callable[..., dict] | None = None
    show_my_notifications: Callable[[], dict] | None = None
    deliver_file: Callable[..., dict] | None = None
    join_voice: Callable[[], dict] | None = None
    leave_voice: Callable[[], dict] | None = None
    deliver_minutes: Callable[..., dict] | None = None
    remember_my_preference: Callable[[str, str], dict] | None = None
    show_my_preferences: Callable[[], dict] | None = None
    forget_my_preferences: Callable[[str | None], dict] | None = None
    set_server_language: Callable[[str, bool], dict] | None = None


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
        """Per-channel reply languages no longer exist: language is set for the whole server
        (set_server_language) and per person (remember_my_preference language). This only returns
        that explanation and changes nothing.

        Args:
            lang: th, en, or another supported language.
            channel: the channel name or mention; omit for the channel being talked in.
        """
        return logged(
            "set_channel_language",
            {"lang": lang, "channel": channel},
            lambda: actions.set_channel_language(lang, channel),
        )

    def set_server_language(language: str, apply_to_all_channels: bool = True) -> dict:
        """Switch Salid's language for the WHOLE server (admins / executives only), everything she
        says and produces: replies, reminders, reports, PDFs, meeting summaries, voice. Use for
        "ใช้ภาษาอังกฤษทั้งเซิร์ฟเวอร์", "เปลี่ยนสลิดเป็นภาษาอังกฤษทั้งหมด", "switch the whole server to
        English", "speak English everywhere" and the same for Thai. There are no per-channel
        languages.

        Args:
            language: th or en.
            apply_to_all_channels: true (default) also clears the per-channel language settings,
                so every channel follows the new language (switching back to Thai restores the
                old per-channel settings); false only changes the server default.
        """
        return logged(
            "set_server_language",
            {"language": language, "apply_to_all_channels": apply_to_all_channels},
            lambda: actions.set_server_language(language, apply_to_all_channels),
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
        pass on the guidance given. One to three named people per request: for four or more people (a team, a
        role, everyone in a channel) use send_dm_many instead, or post_message to announce in a channel; never DM
        a whole server or role with this tool. Never say DMs are impossible or not allowed, and
        never relay a request for money or credentials.

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

    def set_my_notifications(
        morning: bool | None = None,
        evening: bool | None = None,
        morning_time: str | None = None,
        evening_time: str | None = None,
        weekends: bool | None = None,
        doc_alerts: str | None = None,
        ask_reminders: bool | None = None,
    ) -> dict:
        """Change the requester's OWN private notifications from Salid (anyone may): the morning
        brief, the evening brief, their times, whether they also come on weekends, alerts about
        important documents, and reminders about requests nobody answered. Pass only what they
        asked to change. Use for "ไม่ต้องส่งสรุปเย็น", "ส่งสรุปเช้า 7 โมง", "ส่งเสาร์อาทิตย์ด้วย",
        "แจ้งเอกสารสำคัญทุกฉบับ", "เลิกเตือนเรื่องที่ค้างตอบ". Confirm in one short sentence.

        Args:
            morning: true/false, the morning brief on or off.
            evening: true/false, the evening brief on or off.
            morning_time: local time like "07:30" ("default" = the server's default time).
            evening_time: local time like "18:00" ("default" = the server's default time).
            weekends: true to also get briefs on Saturday and Sunday, false for weekdays only.
            doc_alerts: all (every document they can read), important (default) or off.
            ask_reminders: true/false, the reminder when someone asked them something and they
                have not answered for a couple of hours.
        """
        a = dict(
            morning=morning, evening=evening, morning_time=morning_time,
            evening_time=evening_time, weekends=weekends, doc_alerts=doc_alerts,
            ask_reminders=ask_reminders,
        )  # fmt: skip
        return logged("set_my_notifications", a, lambda: actions.set_my_notifications(**a))

    def show_my_notifications() -> dict:
        """Show the requester's own notification settings (briefs, times, weekends, document
        alerts, reminders). Use for "ตอนนี้ได้รับแจ้งเตือนอะไรบ้าง" or "what do you send me"."""
        return logged("show_my_notifications", {}, actions.show_my_notifications)

    def deliver_file(
        destination: str,
        note: str = "",
        source: str = "this",
        file_index: int | None = None,
        mention: list[str] | None = None,
    ) -> dict:
        """Deliver a file / document for the requester: post it in a channel, DM it to a person,
        or attach it to an email draft. Salid CAN do this. Use for "ส่ง / เอาไปลง / ฝากให้ /
        forward / send this to ...", e.g. "นำเอกสารนี้ไปส่งที่ห้องเปเปอร์แล้วแท็ก jemm ด้วย".
        Write `note` yourself: a short, polite message for the recipients in the requester's
        language (do not paste the document). If the result says needs_confirmation, show the
        warning to the requester and call again with EXACTLY the same arguments only after they
        confirm in a later message. If it says ambiguous, ask which candidate. On success confirm
        in one short sentence.

        Args:
            destination: a channel (name in Thai or English, or a #mention), a person (display
                name) to DM, or "email:<address>" to draft an email with the file attached.
            note: the short polite message to post with the file (may tag people with @[name]).
            source: "this" = the file(s) on the current message or on the message it replies to;
                or a message link / id (e.g. from chat_search); or the file_name of a PDF made
                earlier in this conversation.
            file_index: which attachment (0 = first) when the source has several; omit for all.
            mention: display names of people to tag in the channel post.
        """
        a = dict(
            destination=destination, note=note, source=source, file_index=file_index,
            mention=mention or [],
        )  # fmt: skip
        return logged("deliver_file", a, lambda: actions.deliver_file(**a))

    def join_voice() -> dict:
        """Join the voice channel the requester is in right now and take meeting notes. Use for
        every request to come in NOW: "เข้ามาประชุมหน่อย", "มาฟังด้วย", "เข้าห้องเสียง", "join us",
        "hop in". NEVER use schedule_meeting for these: that is only for a future time the user
        names. If the result says not_in_voice, tell them to enter a voice channel first."""
        return logged("join_voice", {}, actions.join_voice)

    def leave_voice() -> dict:
        """Leave the voice meeting (the summary is posted). Use for "ออกไปก่อน", "ไปได้แล้ว",
        "ออกจากห้อง", "leave us", "you can go"."""
        return logged("leave_voice", {}, actions.leave_voice)

    def deliver_minutes(destination: str, meeting: str = "last") -> dict:
        """Send the stored summary / resolutions (มติ) of a finished voice meeting to a text
        channel. Use for "ส่งสรุปประชุมเมื่อกี้ไปห้อง X", "send the minutes to #x". Only a meeting the
        requester attended or can see, and only a channel they can post in. Confirm in one short
        sentence; if it is ambiguous ask which channel.

        Args:
            destination: the channel name (Thai or English) or a #mention.
            meeting: "last" (the latest meeting in this server) or the voice channel's name.
        """
        a = dict(destination=destination, meeting=meeting)
        return logged("deliver_minutes", a, lambda: actions.deliver_minutes(**a))

    def remember_my_preference(kind: str, value: str = "") -> dict:
        """Remember a LASTING preference of the requester about how Salid should treat them (only
        their own; anyone may). Use when they state it as a standing preference: "เรียกผมว่าพี่ปัส",
        "ตอบสั้นๆ นะ" (as a general habit), "สนใจเรื่องการตลาดเป็นหลัก", "สรุปให้ผมเป็นภาษาอังกฤษ",
        "ไม่ต้องใส่อีโมจิ", "ผมเป็นคนดูแลบัญชี". NOT for one-off requests ("ตอบสั้นๆ ข้อนี้"); if it
        is unclear whether it is lasting, ask briefly first. Confirm in one short sentence.

        Args:
            kind: nickname (what to call them), reply_length (value short / normal / detailed),
                language (th or en, for DMs, personal briefs and private replies), focus_add /
                focus_remove (a topic they care about), role_note (their job, e.g. "ดูแลบัญชี"),
                style_add / style_remove (a short instruction such as "no emoji"), delivery
                (value dm = send my briefings, reminders and notices by DM instead of my private
                Salid channel; channel = back to the channel), tone (value cheerful / friendly /
                professional / concise / calm: how Salid talks to them in THEIR private channel
                and DMs only; in a shared room the room's tone wins), task_confirm_mode (value
                approve = when I give someone a task, ask me to approve it after they accept;
                notify = just tell me; empty = follow the organisation).
            value: the text (or short / normal / detailed, th / en). Empty clears nickname,
                reply_length, language, tone or role_note.
        """
        return logged(
            "remember_my_preference",
            {"kind": kind, "value": value},
            lambda: actions.remember_my_preference(kind, value),
        )

    def show_my_preferences() -> dict:
        """Show what Salid remembers about how the requester likes to work (their own only).
        Use for "จำอะไรเกี่ยวกับผมไว้บ้าง" or "what do you know about my preferences"."""
        return logged("show_my_preferences", {}, actions.show_my_preferences)

    def forget_my_preferences(kind: str | None = None) -> dict:
        """Forget the requester's saved preferences: one kind (nickname, reply_length, language,
        focus, role_note, style) or, with no kind, everything. Use for "ลืมที่ผมเคยบอกไว้ทั้งหมด".

        Args:
            kind: which preference to forget; omit to forget all of them.
        """
        return logged(
            "forget_my_preferences", {"kind": kind}, lambda: actions.forget_my_preferences(kind)
        )

    tools = {
        "set_channel_mode": set_channel_mode,
        "set_channel_language": set_channel_language,
        "set_server_language": set_server_language,
        "set_memory_days": set_memory_days,
        "show_settings": show_settings,
        "hidden_channels": hidden_channels,
        "cancel_deep_task": cancel_deep_task,
        "send_dm": send_dm,
        "set_my_notifications": set_my_notifications,
        "show_my_notifications": show_my_notifications,
        "deliver_file": deliver_file,
        "join_voice": join_voice,
        "leave_voice": leave_voice,
        "deliver_minutes": deliver_minutes,
        "remember_my_preference": remember_my_preference,
        "show_my_preferences": show_my_preferences,
        "forget_my_preferences": forget_my_preferences,
    }
    return [fn for name, fn in tools.items() if getattr(actions, name) is not None]
