"""Deliver a file for a person: post it in a channel, DM it, or draft an email with it attached.

`match_channels` is the pure fuzzy channel finder (Thai or English names, with a phonetic
fallback across scripts). `DeliverMixin` (a mixin of `SecretaryBot`) is the Discord side: it finds
the source file(s), checks in code that the requester may read the source and may post at the
destination, asks for a confirmation in a LATER message when the destination audience cannot read
the source channel (or when the channel was only guessed phonetically), posts the file with a
short polite note, and logs every delivery.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher

import discord

from .coo_store import DELIVERIES
from .dm import closed_dm_message, send_dm
from .i18n import t
from .media import parse_message_ref
from .mentions import MENTION_RE, find_member, match_members, resolve_mentions
from .permissions import channel_missing, permission_message
from .secretary import SecretaryTools, fingerprint
from .secretary_store import new_id, utc_iso

logger = logging.getLogger(__name__)

MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_FILES = 10
MAX_NOTE_CHARS = 1500
DM_PER_HOUR = 20
PHONETIC_MIN = 0.75
_RAW_MENTION = re.compile(r"<@[!&]?\d+>")
_PREFIXES = ("ห้อง", "ช่อง", "แชนแนล", "แชแนล", "channel", "room")
SAME_WORDS = ("me", "myself", "ฉัน", "ผม", "เรา", "ตัวเอง")


# ------------------------------------------------------------------ channel matching
def _letters(s: str) -> str:
    """Casefolded letters, digits and Thai marks only (separators and punctuation removed)."""
    return "".join(c for c in s.casefold() if unicodedata.category(c)[0] in "LNM")


def _norm(query: str) -> str:
    q = query.strip().lstrip("#").strip().casefold()
    for p in _PREFIXES:
        if q.startswith(p):
            q = q[len(p) :].strip()
            break
    return _letters(q)


_THAI = {
    **dict.fromkeys("กขฃคฅฆ", "k"), "ง": "n", **dict.fromkeys("จฉชฌ", "c"),
    **dict.fromkeys("ซศษส", "s"), **dict.fromkeys("ญย", "y"), **dict.fromkeys("ดฎ", "d"),
    **dict.fromkeys("ตฏถฐทธฑฒ", "t"), **dict.fromkeys("นณ", "n"), "บ": "b",
    **dict.fromkeys("ปผพภ", "p"), **dict.fromkeys("ฝฟ", "f"), "ม": "m", "ร": "r",
    **dict.fromkeys("ลฬ", "l"), "ว": "w", **dict.fromkeys("หฮ", "h"),
}  # fmt: skip
_LATIN_DIGRAPHS = (("ph", "p"), ("th", "t"), ("kh", "k"), ("ch", "c"), ("sh", "s"), ("ck", "k"))
_LATIN = {**dict.fromkeys("cgq", "k"), "j": "c", "z": "s", "v": "w", "x": "k"}


def has_thai(s: str) -> bool:
    return any("฀" <= c <= "๿" for c in s)


def consonant_skeleton(text: str) -> str:
    """Rough sound skeleton shared by Thai and Latin spellings: consonants only, in classes
    ("เปเปอร์" and "paper" both give "pp"/"ppr", close enough to compare)."""
    out: list[str] = []
    if has_thai(text):
        for c in text:
            if c == "์" and out:  # thanthakhat: the consonant before it is silent
                out.pop()
            elif c in _THAI:
                out.append(_THAI[c])
        return "".join(out)
    s = text.casefold()
    for a, b in _LATIN_DIGRAPHS:
        s = s.replace(a, b)
    for c in s:
        if c.isalpha() and c not in "aeiouyh":
            out.append(_LATIN.get(c, c))
    return "".join(out)


def _phonetic_score(query: str, name: str) -> float:
    q = consonant_skeleton(query)
    if len(q) < 2:
        return 0.0
    best = 0.0
    for tok in [name, *re.split(r"[-_\s]+", name)]:
        if has_thai(tok) == has_thai(query):
            continue  # same script: the plain tiers handle it
        k = consonant_skeleton(tok)
        if k:
            best = max(best, SequenceMatcher(None, q, k).ratio())
    return best


def match_channels(channels, query: str) -> tuple[list, str]:
    """(best channels, how): "exact", "token" (a word of the name), "contains" or "phonetic" (a
    guess across Thai / Latin spelling); ([], "") when nothing fits. More than one channel in the
    returned list means the request is ambiguous."""
    q = _norm(query)
    if not q:
        return [], ""
    chans = list(channels)
    exact = [c for c in chans if _letters(c.name) == q]
    if exact:
        return exact, "exact"
    token = [c for c in chans if q in {_letters(x) for x in re.split(r"[-_\s]+", c.name)}]
    if token:
        return token, "token"
    contains = [
        c for c in chans if len(q) >= 3 and (q in _letters(c.name) or _letters(c.name) in q)
    ]
    if contains:
        return contains, "contains"
    scored = sorted(((_phonetic_score(q, c.name), c) for c in chans), key=lambda x: -x[0])
    scored = [(s, c) for s, c in scored if s >= PHONETIC_MIN]
    if not scored:
        return [], ""
    top = scored[0][0]
    return [c for s, c in scored if s >= top - 0.1], "phonetic"


# ------------------------------------------------------------------ source items
@dataclass
class _Item:
    name: str
    size: int
    read: object  # async () -> bytes


def _perm(channel, who, name: str, fallback: str = "") -> bool:
    try:
        perms = channel.permissions_for(who)
    except Exception:
        return False
    return bool(getattr(perms, name, getattr(perms, fallback, False) if fallback else False))


class DeliverMixin:
    # ------------------------------------------------------------- tool entry
    def _deliver_actions(self, message, guild, member, lang: str, loop, secretary, request_id: str):
        """The sync `deliver_file` callable bound to this message (runs in the agent's thread)."""

        def deliver(destination, note="", source="this", file_index=None, mention=None):
            fut = asyncio.run_coroutine_threadsafe(
                self._deliver_file(
                    message, guild, member, lang, secretary, request_id,
                    destination=destination, note=note, source=source,
                    file_index=file_index, mention=mention or [],
                ),
                loop,
            )  # fmt: skip
            return fut.result(timeout=180)

        return deliver

    # ------------------------------------------------------------- sources
    async def _items_of(self, msg, file_index) -> tuple[list[_Item], str]:
        atts = list(getattr(msg, "attachments", ()) or ())
        if not atts:
            return [], "that message has no attachments"
        if file_index is not None:
            if not 0 <= int(file_index) < len(atts):
                return [], f"file_index out of range (the message has {len(atts)} files)"
            atts = [atts[int(file_index)]]
        return [
            _Item(getattr(a, "filename", "") or "file", int(getattr(a, "size", 0) or 0), a.read)
            for a in atts[:MAX_FILES]
        ], ""

    def _channel_of_id(self, guild, member, message_id: int):
        """A channel the member can read whose log holds this message id (bare-id sources)."""
        now = self.s.now()
        start = (now - timedelta(days=self.team.memory_retention_days(guild.id))).date().isoformat()
        ids = {c.id for c in self._all_channels(guild) if self._can_read(c, member)}
        for rec in self.memory_store.messages(
            guild.id, ids, start, now.date().isoformat(), limit=5000
        ):
            if rec.message_id == message_id:
                return rec.thread_id or rec.channel_id
        return None

    async def _deliver_source(self, message, guild, member, source: str, file_index, secretary):
        """(items, source channel, source message, error dict | None)."""
        src = (source or "this").strip()
        if src.casefold() in ("this", "here", "current", "") or src in ("นี้", "อันนี้"):
            msg = message if getattr(message, "attachments", None) else None
            if msg is None:
                if getattr(message, "reference", None) is None:
                    return [], None, None, {"error": "no_file", "note": "no attachment on this "
                                            "message and it is not a reply to one; ask the user "
                                            "to attach the file or reply to it"}  # fmt: skip
                msg = await self._referenced_message(message)
                if msg is None:
                    return [], None, None, {
                        "error": "source_deleted",
                        "note": "the message that was replied to was deleted or cannot be read",
                    }  # fmt: skip
        elif (ref := parse_message_ref(src)) is not None:
            cid, mid = ref
            if cid is None:
                cid = await asyncio.to_thread(self._channel_of_id, guild, member, mid)
            getter = getattr(guild, "get_channel_or_thread", None) or guild.get_channel
            channel = getter(cid) if cid else None
            if channel is None:
                return [], None, None, {"error": "source_not_found", "note": "message not found "
                                        "in a channel the requester can read; use its link"}  # fmt: skip
            try:
                msg = await channel.fetch_message(mid)
            except discord.HTTPException:
                return [], None, None, {"error": "source_deleted",
                                        "note": "the source message was deleted"}  # fmt: skip
        else:  # the name of a PDF made earlier in this conversation
            data = self.secretary_state.recall_pdf(guild.id, member.id, src)
            if data is None and secretary is not None:
                data = next((d for n, d in secretary.attachments if n == src), None)
            if data is None:
                return [], None, None, {"error": "source_not_found", "note": f"no message or "
                                        f"PDF named {src!r} (made in this conversation)"}  # fmt: skip

            async def read(d=data):
                return d

            return [_Item(src, len(data), read)], None, None, None
        channel = getattr(msg, "channel", None)
        if channel is not None and not _perm(channel, member, "read_messages", "view_channel"):
            return [], None, None, {"error": "not_allowed", "note": "the requester cannot read "
                                    "the channel of that file"}  # fmt: skip
        items, err = await self._items_of(msg, file_index)
        if err:
            return [], None, None, {"error": "no_file", "note": err}
        return items, channel, msg, None

    # ------------------------------------------------------------- destination
    def _destination(self, guild, member, destination: str):
        """("channel", channel, how) | ("member", member, "") | ("email", address, "") | an error
        dict. Channels are only those the requester can see."""
        dest = (destination or "").strip()
        if dest.casefold().startswith("email:"):
            return "email", dest[6:].strip(), ""
        digits = re.fullmatch(r"<#(\d+)>", dest)
        visible = [
            c for c in guild.text_channels if _perm(c, member, "view_channel", "read_messages")
        ]
        if digits:
            hit = next((c for c in visible if c.id == int(digits[1])), None)
            return ("channel", hit, "exact") if hit else {"error": "channel_not_found"}
        channels, how = match_channels(visible, dest)
        wants_channel = dest.startswith("#") or dest.casefold().startswith(_PREFIXES)
        people = [] if wants_channel else match_members(guild, dest)
        if dest.casefold() in SAME_WORDS:
            return "member", member, ""
        if channels and people and how != "phonetic":
            return {
                "error": "ambiguous",
                "candidates": [f"channel #{c.name}" for c in channels[:5]]
                + [f"person {m.display_name}" for m in people[:5]],
                "note": "ask whether a channel or a person is meant",
            }
        if len(channels) == 1:
            return "channel", channels[0], how
        if len(channels) > 1:
            return {"error": "ambiguous", "candidates": [f"#{c.name}" for c in channels[:8]],
                    "note": "ask the user which channel"}  # fmt: skip
        if len(people) == 1:
            return "member", people[0], ""
        if len(people) > 1:
            return {"error": "ambiguous", "candidates": [m.display_name for m in people[:8]],
                    "note": "ask the user which person"}  # fmt: skip
        return {
            "error": "destination_not_found",
            "channels": [c.name for c in visible[:25]],
            "note": "if the user named a channel in Thai, pick the matching name from channels "
            "and call again; otherwise ask where to send it",
        }

    # ------------------------------------------------------------- the tool
    def _note_text(self, guild, note: str, mention: list[str]) -> tuple[str, list[str], set[int]]:
        """(text with real mentions, names that could not be tagged, ids that were tagged)."""
        text = _RAW_MENTION.sub("", note or "").strip()[:MAX_NOTE_CHARS]
        unresolved: list[str] = []
        for name in mention or []:
            if find_member(guild, name) is None:
                unresolved.append(name)
            elif f"@[{name}]" not in text:
                text = f"{text} @[{name}]".strip()
        for n in MENTION_RE.findall(text):
            if find_member(guild, n) is None and n not in unresolved:
                unresolved.append(n)
        out = resolve_mentions(text, guild)
        ids = {int(i) for i in re.findall(r"<@(\d+)>", out)}
        return out, unresolved, ids

    async def _deliver_file(
        self, message, guild, member, lang, secretary, request_id, *,
        destination, note, source, file_index, mention,
    ) -> dict:  # fmt: skip
        gid = guild.id
        items, src_channel, src_msg, err = await self._deliver_source(
            message, guild, member, source, file_index, secretary
        )
        if err:
            return err
        dest = await asyncio.to_thread(self._destination, guild, member, destination)
        if isinstance(dest, dict):
            return dest
        kind, target, how = dest
        # a thread counts as its parent channel for who may read it
        access = getattr(src_channel, "parent", None) or src_channel
        src_name = getattr(access, "name", "") if access is not None else ""
        if kind == "email":
            return await self._deliver_email(guild, member, secretary, items, src_msg, target, note)
        if kind == "channel":
            bad = channel_missing(target, guild, ("send_messages", "attach_files"))
            if not _perm(target, member, "send_messages"):
                return {"error": "not_allowed", "note": "the requester cannot send messages in "
                        f"#{target.name}"}  # fmt: skip
            if bad:
                return {"error": "missing_permission", "channel": target.name,
                        "how_to_grant": permission_message(bad, lang, self.application_id)}  # fmt: skip
        elif target.bot:
            return {"error": "person_not_found"}
        # who would see the file at the destination without being able to read its source
        warning = ""
        if access is not None:
            src_viewers = await asyncio.to_thread(self._channel_viewers, guild, access)
            if kind == "channel":
                audience = await asyncio.to_thread(self._channel_viewers, guild, target)
            else:
                audience = frozenset({target.id})
            missing = sorted(audience - src_viewers - {member.id})
            if missing:
                names = [getattr(guild.get_member(u), "display_name", str(u)) for u in missing]
                who = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
                warning = t("deliver_warn_audience", lang, source=src_name, who=who)
        # A unique phonetic match ("ห้องเปเปอร์" -> #paper-minute) posts straight away; the reply
        # names the channel, so a wrong guess is visible and can be deleted by reply.
        if warning:
            dkey = target.id if kind != "email" else target
            key = (
                "deliver", gid, member.id,
                fingerprint(
                    kind, dkey, getattr(src_msg, "id", source), file_index, note, sorted(mention)
                ),
            )  # fmt: skip
            if not self.secretary_state.confirm(key, request_id):
                where = f"#{target.name}" if kind == "channel" else target.display_name
                return {
                    "status": "needs_confirmation",
                    "destination": where,
                    "warning": warning or None,
                    "files": [i.name for i in items],
                    "instruction": (
                        "Tell the requester where it would go"
                        + (" and show the warning" if warning else "")
                        + ", ask them to confirm, then call deliver_file again with EXACTLY the "
                        "same arguments in a later message. Do not claim it was sent."
                    ),
                }
        text, unresolved, ids = self._note_text(guild, note, mention)
        url = getattr(src_msg, "jump_url", "") if src_msg is not None else ""
        total = sum(i.size for i in items)
        too_big = total > MAX_TOTAL_BYTES
        files: list[discord.File] = []
        if not too_big:
            for it in items:
                data = await it.read()
                files.append(discord.File(io.BytesIO(data), filename=it.name))
        requester = f"<@{member.id}>"
        if kind == "channel":
            footer = (
                t("deliver_footer", lang, who=requester, url=url)
                if url
                else t("deliver_footer_plain", lang, who=requester)
            )
            body = "\n\n".join(x for x in (text, footer) if x)
            if too_big and url:
                body += "\n" + t("deliver_oversize", lang, url=url)
            allowed = discord.AllowedMentions(
                users=[discord.Object(id=i) for i in sorted(ids)],
                roles=False, everyone=False, replied_user=False,
            )  # fmt: skip
            try:
                sent = await target.send(body[:2000], files=files or None, allowed_mentions=allowed)
            except discord.HTTPException:
                logger.warning("deliver_file: posting failed", exc_info=True)
                return {"error": "could_not_post", "channel": target.name}
            result = {"status": "posted", "channel": target.name, "mention": f"<#{target.id}>",
                      "link": getattr(sent, "jump_url", "")}  # fmt: skip
            dest_log = {"type": "channel", "id": target.id, "name": target.name}
        else:
            now = self._clock()
            recent = [x for x in self._dm_sent.get(gid, []) if now - x < 3600]
            if len(recent) >= DM_PER_HOUR:
                return {"error": "rate_limited", "per_hour": DM_PER_HOUR}
            body = "\n".join(
                x
                for x in (
                    t("deliver_dm_from", lang, who=getattr(member, "display_name", ""),
                      server=getattr(guild, "name", "")),
                    re.sub(r"<@\d+>", "", text),
                    t("deliver_original", lang, url=url) if url else "",
                    t("deliver_oversize", lang, url=url) if too_big and url else "",
                )
                if x
            )  # fmt: skip
            res = await send_dm(target, body[:2000], files)
            if not res.ok:
                if res.reason == "closed":
                    return {"error": "dms_closed", "person": target.display_name,
                            "guidance": closed_dm_message(target.display_name, guild.name, lang)}  # fmt: skip
                return {"error": "could_not_dm", "person": target.display_name}
            self._dm_sent[gid] = [*recent, now]
            result = {"status": "sent_dm", "to": target.display_name}
            dest_log = {"type": "dm", "id": target.id, "name": target.display_name}
        result["files"] = [i.name for i in items] if not too_big else []
        if too_big:
            result["note"] = "files over 25 MB in total: only a link to the original was sent"
        if unresolved:
            result["could_not_tag"] = unresolved
        await asyncio.to_thread(
            self._log_delivery, gid, member, src_channel, src_msg, source, items, dest_log
        )
        return result

    def _log_delivery(self, gid, member, src_channel, src_msg, source, items, dest) -> None:
        key = f"{utc_iso(datetime.now(UTC))}-{new_id()}"
        self.docs.put(
            DELIVERIES, gid, key,
            {
                "requester_id": member.id,
                "requester_name": getattr(member, "display_name", ""),
                "source": {
                    "channel_id": getattr(src_channel, "id", 0),
                    "message_id": getattr(src_msg, "id", 0),
                    "ref": source if src_msg is None else "",
                    "files": [i.name for i in items],
                },
                "destination": dest,
                "at": utc_iso(datetime.now(UTC)),
            },
        )  # fmt: skip

    async def _deliver_email(self, guild, member, secretary, items, src_msg, address, note) -> dict:
        """An email draft with the (first) file attached, through the existing confirm flow."""
        if secretary is None:
            return {"error": "email_disabled"}
        item = items[0]
        if item.size > 20 * 1024 * 1024:
            return {"error": "attachment too large for email"}
        data = await item.read()
        self.secretary_state.remember_pdf(guild.id, member.id, item.name, data)
        body = _RAW_MENTION.sub("", re.sub(r"@\[([^\]]+)\]", r"\1", note or "")).strip()
        body = body or t("deliver_email_body", self._lang_of_ctx(secretary), name=item.name)
        out = await asyncio.to_thread(
            SecretaryTools(secretary).draft_email,
            [address], item.name.rsplit(".", 1)[0] or item.name, body, None, item.name,
        )  # fmt: skip
        if len(items) > 1 and out.get("status") == "drafted":
            out["note"] = f"only the first file ({item.name}) is attached to the draft"
        if out.get("status") == "drafted":
            await asyncio.to_thread(
                self._log_delivery, guild.id, member, getattr(src_msg, "channel", None), src_msg,
                "", items[:1], {"type": "email", "id": 0, "name": address},
            )  # fmt: skip
        return out

    @staticmethod
    def _lang_of_ctx(secretary) -> str:
        return secretary.lang if secretary.lang in ("th", "en") else "th"
