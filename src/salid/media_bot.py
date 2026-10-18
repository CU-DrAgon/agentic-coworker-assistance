"""Bot glue for audio/video (salid.media) and documents (salid.docs): transcripts for addressed
messages, background memory notes for everything else, history backlog and the open_file tool.
A mixin of SecretaryBot; it uses the bot's `s`, `llm`, `team`, `memory`, `memory_store`,
`importer`, `_spawn`, `_apply_notes`, `_mode`."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from functools import partial

import discord

from . import data_studio, docs, media
from .briefings import DocInfo, split_names
from .image_memory import describe_image, format_note, image_mime
from .media import parse_message_ref  # noqa: F401 (re-export)
from .memory import describe_attachment

logger = logging.getLogger(__name__)

MAX_MEDIA_PER_MESSAGE = 4
MAX_DOCS_PER_MESSAGE = 4
MEDIA_CACHE_KEPT = 50
MAX_DOC_DOWNLOAD = 25 * 1024 * 1024
VOICE_PREFIX = "[ข้อความเสียงจากผู้ใช้] "


class MediaMixin:
    def _init_media(self) -> None:
        self.media_budget = media.MediaBudget(self.s.media_minutes_per_day)
        self.doc_budget = media.DailyBudget(self.s.doc_pages_per_day)
        self._media_sem = asyncio.Semaphore(2)
        self._doc_sem = asyncio.Semaphore(2)
        self._media_cache: dict[int, dict[int, media.Transcript]] = {}
        self._doc_infos: dict[tuple[int, int], DocInfo] = {}  # (message id, attachment) -> analysis
        self._agent_receipts: dict[int, list[str]] = {}  # message id -> receipt notes (agent path)
        self.media_stats = {"transcribed": 0, "failed": 0, "skipped": 0, "docs": 0}
        # originals of documents Salid read -> real files in the command center (bucket only)
        self.file_archive = None
        if getattr(self.s, "receipt_bucket", ""):
            try:
                from .coo_store import make_doc_store
                from .files import FileArchive
                from .library import make_blobs

                self.file_archive = FileArchive(
                    make_doc_store(self.s, self.store), make_blobs(self.s)
                )
            except Exception as e:
                logger.warning("file archive unavailable (%s)", type(e).__name__)

    def _archive_original(self, message, att, data: bytes, kind: str, note=None, dt=None) -> None:
        """Keep the original file as a document of the company (best effort, off the event loop)."""
        if self.file_archive is None or message is None or getattr(message, "guild", None) is None:
            return
        ch = getattr(message, "channel", None)
        mime = str(getattr(att, "content_type", "") or "") or "application/octet-stream"
        self.file_archive.save(
            message.guild.id,
            filename=getattr(att, "filename", "") or "document",
            data=data,
            mime=mime.split(";")[0],
            by=str(getattr(getattr(message, "author", None), "display_name", "") or ""),
            channel_id=getattr(ch, "id", None),
            channel_name=str(getattr(ch, "name", "") or ""),
            source_url=str(getattr(message, "jump_url", "") or ""),
            summary=str(getattr(note, "summary", "") or ""),
            tag=str(getattr(note, "doc_type", "") or kind),
            text=str(getattr(dt, "text", "") or ""),
        )

    # ------------------------------------------------------------- switches
    def _media_enabled(self, gid: int) -> bool:
        return self.team.enabled(gid, "memory") and self.team.enabled(gid, "media_memory")

    def _docs_enabled(self, gid: int) -> bool:
        return self.team.enabled(gid, "memory") and self.team.enabled(gid, "doc_memory")

    def _day(self) -> str:
        return self.s.today().isoformat()

    # ------------------------------------------------------------- transcripts
    def _cache_transcript(self, message_id: int, idx: int, tr: media.Transcript) -> None:
        self._media_cache.setdefault(message_id, {})[idx] = tr
        while len(self._media_cache) > MEDIA_CACHE_KEPT:
            self._media_cache.pop(next(iter(self._media_cache)))

    async def _transcribe_att(self, message, idx: int, att, *, gid: int) -> media.Transcript:
        """Download + transcribe one attachment (cached per message/attachment). Raises MediaError."""
        hit = self._media_cache.get(message.id, {}).get(idx)
        if hit is not None:
            return hit
        if (getattr(att, "size", 0) or 0) > media.MAX_BYTES:
            self.media_stats["skipped"] += 1
            raise media.MediaError("file too large (over 100 MB)")
        data = await att.read()
        try:
            async with self._media_sem:
                tr = await asyncio.to_thread(
                    media.transcribe_media,
                    self.llm,
                    self.s.gemini_model,
                    data,
                    getattr(att, "filename", "") or "",
                )
        except Exception as e:  # never log contents
            self.media_stats["failed"] += 1
            logger.warning("media transcription failed (%s)", type(e).__name__)
            raise media.MediaError(type(e).__name__) from e
        finally:
            del data
        self.media_stats["transcribed"] += 1
        self.media_budget.add(gid, self._day(), tr.duration)
        self._cache_transcript(message.id, idx, tr)
        return tr

    async def _fill_media_input(self, message, f, att) -> None:
        """Turn a placeholder FileInput into the transcript text part for the agent."""
        try:
            tr = await self._transcribe_att(message, f.index, att, gid=message.guild.id)
        except Exception:
            f.skip = "audio could not be transcribed"
            return
        f.mime = "text/plain"
        f.data = media.prompt_text(tr).encode()
        f.size = len(f.data)
        f.label = media.label_for(f.index, f.filename, tr)

    async def _fill_office_input(self, f, att, data_on: bool = False) -> None:
        """Office / text document sent to Salid: its text (about 30k chars) as a text part. With
        the `data` feature a spreadsheet / CSV / JSON also keeps its bytes (`f.raw`) for the data
        tools and gets one line telling the model to use them."""
        if (getattr(att, "size", 0) or 0) > MAX_DOC_DOWNLOAD:
            f.skip = "over 25 MB"
            return
        keep_note = None
        try:
            data = await att.read()
            keep = data_on and data_studio.is_data_file(f.filename)
            try:
                kind, text, pages = await asyncio.to_thread(
                    docs.extract_document, data, f.filename, getattr(att, "content_type", None)
                )
            except docs.DocUnreadable as e:
                if keep and data_studio.file_ext(f.filename) == "xls":  # no text preview possible
                    kind, text, pages, keep_note = "xls", "", 0, e
                else:
                    raise
        except docs.DocUnreadable as e:
            f.skip = str(e)
            return
        except Exception as e:
            logger.warning("document read failed (%s)", type(e).__name__)
            f.skip = "could not be read"
            return
        extra = ""
        if keep:
            try:
                sheets, rows = await asyncio.to_thread(data_studio.quick_counts, data, f.filename)
                f.raw = data
                extra = f"\n(use data_load for analysis: {sheets} sheets, {rows} rows)"
            except Exception:
                if keep_note is not None:
                    f.skip = str(keep_note)
                    return
        if not text.strip() and not extra:
            f.skip = "no readable text in the file"
            return
        f.mime = "text/plain"
        f.data = (docs.clip_for_agent(text) + extra).encode()
        f.size = len(f.data)
        f.label = f"[ไฟล์ #{f.index}: {f.filename}, {pages} {docs.unit_label(kind)}]"

    def _voice_question(self, message, q: str, files: list) -> tuple[str, list]:
        """A short voice message with no text addressed to Salid is the user's spoken question:
        its transcript becomes `q` and the transcript file part is dropped."""
        if (
            q
            or len(files) != 1
            or not (
                media.is_voice_message(message)
                or media.media_kind(_att_of(message, files[0])) == "voice"
            )
        ):
            return q, files
        f = files[0]
        tr = self._media_cache.get(message.id, {}).get(f.index)
        if tr is None or not tr.segments or tr.duration > media.VOICE_QUESTION_MAX_S:
            return q, files
        text = " ".join(s.text for s in tr.segments)
        return VOICE_PREFIX + text, []

    # ------------------------------------------------------------- memory: audio / video
    def _media_soon(self, message, logged: bool, react: bool = False) -> None:
        cands = media.media_candidates(message)[:MAX_MEDIA_PER_MESSAGE]
        gid = message.guild.id
        if not (cands and logged and self._media_enabled(gid)):
            return
        cached = self._media_cache.get(message.id, {})
        if not self.media_budget.available(gid, self._day()) and not cached:
            self.media_stats["skipped"] += 1
            return
        self._spawn(self._media_message(message, cands, react=react))

    async def _media_note(self, message, idx: int, att) -> str | None:
        kind = media.media_kind(att) or "audio"
        try:
            tr = await self._transcribe_att(message, idx, att, gid=message.guild.id)
        except Exception:
            return None
        summary = await asyncio.to_thread(media.summarize_note, self.llm, self.s.gate_model, tr)
        return media.memory_note(kind, tr, summary)

    async def _media_message(self, message, cands, react: bool = False) -> dict[int, str]:
        notes: dict[int, str] = {}
        for i, att in cands:
            if i not in self._media_cache.get(message.id, {}) and not self.media_budget.available(
                message.guild.id, self._day()
            ):
                self.media_stats["skipped"] += 1
                continue
            note = await self._media_note(message, i, att)
            if note:
                notes[i] = note
        if notes:
            self._apply_notes(message, notes)
            if react:
                await self._read_react(message)
        return notes

    # ------------------------------------------------------------- memory: documents
    def _doc_cands(self, message) -> list[tuple[int, object]]:
        found = [
            (i, a)
            for i, a in enumerate(getattr(message, "attachments", ()) or ())
            if docs.doc_kind(a) and not image_mime(a)
        ]
        return found[:MAX_DOCS_PER_MESSAGE]

    def _doc_soon(self, message, logged: bool, react: bool = False) -> None:
        cands = self._doc_cands(message)
        gid = message.guild.id
        if cands and logged and self._docs_enabled(gid):
            self._spawn(self._doc_message(message, cands, react=react))

    def _receipt_line(self, message, idx: int, order: int) -> str | None:
        """The receipt pipeline's note for this attachment, if it already read it."""
        note = self._image_notes.get(message.id, {}).get(idx)
        if note and note.startswith("[ใบเสร็จ]"):
            return note
        lines = self._agent_receipts.get(message.id, [])
        return lines[order] if order < len(lines) else None

    async def _doc_note(self, message, idx: int, att, order: int = 0, *, gid: int) -> str | None:
        """Note for one document, or None when it cannot be read. Never ingests a receipt: when
        the receipt pipeline already did, its note is reused (plus the free text layer)."""
        name = getattr(att, "filename", "") or "document"
        kind = docs.doc_kind(att)
        if (getattr(att, "size", 0) or 0) > MAX_DOC_DOWNLOAD:
            return None
        data = await att.read()
        try:
            try:
                dt = await asyncio.to_thread(docs.extract_text, data, kind)
            except Exception as e:
                logger.warning("document read failed (%s)", type(e).__name__)
                return None
            line = self._receipt_line(message, idx, order)
            if line:
                return docs.receipt_doc_note(name, kind, dt, line)
            scanned = docs.is_scanned(kind, dt)
            if not dt.text and not scanned:
                return None
            if not self.doc_budget.available(gid, self._day()):
                return None
            note = None
            if scanned or len(dt.text) > 300 or kind == "pdf":
                try:
                    async with self._doc_sem:
                        note = await asyncio.to_thread(
                            docs.analyze, self.llm, self.s.gate_model, data, kind, dt
                        )
                except Exception as e:
                    logger.warning("document analysis failed (%s)", type(e).__name__)
                    if scanned:
                        return None
            self.doc_budget.add(gid, self._day(), dt.pages)
            self.media_stats["docs"] += 1
            try:
                await asyncio.to_thread(self._archive_original, message, att, data, kind, note, dt)
            except Exception as e:
                logger.warning("file archive failed (%s)", type(e).__name__)
            if note is not None and message is not None:  # kept for the live path's alert
                self._doc_infos[(message.id, idx)] = DocInfo(
                    name, kind, note.doc_type, note.summary, note.amount_value,
                    split_names(note.addressed_to),
                )  # fmt: skip
                while len(self._doc_infos) > MEDIA_CACHE_KEPT:
                    self._doc_infos.pop(next(iter(self._doc_infos)))
            return docs.doc_note(name, kind, dt, note)
        finally:
            del data

    async def _doc_message(self, message, cands, react: bool = False) -> dict[int, str]:
        notes: dict[int, str] = {}
        for order, (i, att) in enumerate(cands):
            note = await self._doc_note(message, i, att, order, gid=message.guild.id)
            if (info := self._doc_infos.pop((message.id, i), None)) is not None:
                self._doc_alert_soon(
                    message, i, info
                )  # a live message: maybe an important document
            if note:
                notes[i] = note
        if notes:
            self._apply_notes(message, notes)
            if react:
                await self._read_react(message)
        return notes

    # ------------------------------------------------------------- history backlog
    async def _attachment_backlog(self, guild, channel_ids, keywords, handler) -> int:
        """Old records still holding a plain descriptor for `keywords` ("[audio:", ...), newest
        first. `handler(msg, idx, att, order)` -> (note or None, ok to continue); the budgets are
        enforced by the handler. Messages that are gone get "-unread" markers."""
        gid = guild.id
        now = self.s.now()
        start = (
            (now - timedelta(days=self.team.fresh_memory_retention_days(gid, fail_closed=True)))
            .date()
            .isoformat()
        )
        rows: dict[int, object] = {}
        for kw in keywords:
            for rec in await asyncio.to_thread(
                self.memory_store.messages, gid, channel_ids, start, now.date().isoformat(),
                keyword=kw, limit=5000,
            ):  # fmt: skip
                rows[rec.message_id] = rec
        done = 0
        for rec in sorted(rows.values(), key=lambda r: r.created_at, reverse=True):
            if rec.kind == "salid":
                continue
            new_text, stop = await self._backlog_one(guild, rec, handler)
            if new_text != rec.text:
                done += 1
                self._memory_enqueue(partial(self.memory.edit, gid, rec.message_id, new_text, None))
            if stop:
                break
            await asyncio.sleep(self.importer.page_sleep)
        return done

    async def _backlog_one(self, guild, rec, handler) -> tuple[str, bool]:
        getter = getattr(guild, "get_channel_or_thread", None) or guild.get_channel
        channel = getter(rec.thread_id or rec.channel_id)
        try:
            msg = await channel.fetch_message(rec.message_id) if channel else None
        except discord.HTTPException:
            msg = None
        text = rec.text
        found = 0
        for order, (i, att) in enumerate(self._backlog_atts(msg, rec.text, handler)):
            marker = describe_attachment(att)
            if marker not in text:
                continue
            found += 1
            note, go = await handler(msg, i, att, order)
            if not go:
                return text, True
            if note is None:
                note = marker.replace(":", "-unread:", 1)
            text = text.replace(marker, note, 1)
        if msg is None:  # gone: stop retrying these markers
            for kind in handler.kinds:
                text = re.sub(rf"\[{kind}: ", f"[{kind}-unread: ", text)
        return text, False

    @staticmethod
    def _backlog_atts(msg, text, handler):
        if msg is None:
            return []
        return [
            (i, a)
            for i, a in enumerate(msg.attachments)
            if handler.pick(a) and describe_attachment(a) in text
        ]

    async def _media_backlog(self, guild) -> None:
        gid = guild.id

        async def run(msg, idx, att, order):
            if not self.media_budget.available(gid, self._day()):
                return None, False
            return await self._media_note(msg, idx, att), True

        run.kinds, run.pick = ("audio", "video"), lambda a: media.media_kind(a) is not None
        n = await self._attachment_backlog(
            guild, self._backlog_channels, ("[audio:", "[video:"), run
        )
        if n:
            logger.info("memory backlog media guild=%s updated=%d", gid, n)

    async def _doc_backlog(self, guild) -> None:
        gid = guild.id

        async def run(msg, idx, att, order):
            if not self.doc_budget.available(gid, self._day()):
                return None, False
            return await self._doc_note(msg, idx, att, order, gid=gid), True

        run.kinds = ("file",)
        run.pick = lambda a: bool(docs.doc_kind(a)) and not image_mime(a)
        n = await self._attachment_backlog(guild, self._backlog_channels, ("[file:",), run)
        if n:
            logger.info("memory backlog documents guild=%s updated=%d", gid, n)

    async def _media_doc_backlog(self, guild, channel_ids: set[int]) -> None:
        self._backlog_channels = channel_ids
        gid = guild.id
        if self._media_enabled(gid):
            await self._media_backlog(guild)
        if self._docs_enabled(gid):
            await self._doc_backlog(guild)

    # ------------------------------------------------------------- open_file tool
    def _make_opener(self, guild, allowed: dict[int, str], loop):
        """Sync callable for the agent's open_file tool (called from the ask thread)."""

        def opener(channel_id: int | None, message_id: int, index: int) -> dict:
            fut = asyncio.run_coroutine_threadsafe(
                self._open_file(guild, allowed, channel_id, message_id, index), loop
            )
            return fut.result(timeout=300)

        return opener

    async def _open_file(
        self, guild, allowed: dict[int, str], channel_id: int | None, message_id: int, index: int
    ) -> dict:
        """Re-read one attachment of a remembered message and return its text (document text
        layer / OCR, transcript for audio and video). Only channels in `allowed` (the asker's
        memory scope); anything else is refused."""
        gid = guild.id
        if channel_id is None:  # bare message id: find it among the readable records
            now = self.s.now()
            start = (
                (now - timedelta(days=self.team.fresh_memory_retention_days(gid, fail_closed=True)))
                .date()
                .isoformat()
            )
            recs = await asyncio.to_thread(
                self.memory_store.messages, gid, set(allowed), start, now.date().isoformat(),
                limit=5000,
            )  # fmt: skip
            hit = next((r for r in recs if r.message_id == message_id), None)
            if hit is None:
                return {"error": "message not found in the channels you can read; use its link"}
            channel_id = hit.thread_id or hit.channel_id
        getter = getattr(guild, "get_channel_or_thread", None) or guild.get_channel
        channel = getter(channel_id)
        where = self._memory_channel(channel) if channel is not None else None
        if where is None or where[0] not in allowed:
            return {"error": "not allowed: that channel is not readable for this asker"}
        try:
            msg = await channel.fetch_message(message_id)
        except discord.HTTPException:
            return {"error": "message not found (deleted?)"}
        atts = list(msg.attachments)
        if not 0 <= index < len(atts):
            return {"error": f"file_index out of range (message has {len(atts)} files)"}
        att = atts[index]
        name = getattr(att, "filename", "") or "file"
        try:
            text = await self._read_attachment_text(msg, index, att, gid)
        except (media.MediaError, docs.DocUnreadable) as e:
            return {"error": f"could not read the file: {e}"}
        except Exception as e:
            logger.warning("open_file failed (%s)", type(e).__name__)
            return {"error": "could not read the file"}
        cut = len(text) > docs.OPEN_FILE_CHARS
        return {
            "file": name, "link": getattr(msg, "jump_url", ""), "truncated": cut,
            "text": text[: docs.OPEN_FILE_CHARS],
        }  # fmt: skip

    async def _read_attachment_text(self, msg, idx: int, att, gid: int) -> str:
        if media.media_kind(att) is not None:
            tr = await self._transcribe_att(msg, idx, att, gid=gid)
            return media.prompt_text(tr, docs.OPEN_FILE_CHARS)
        kind = docs.doc_kind(att)
        if kind and (getattr(att, "size", 0) or 0) <= MAX_DOC_DOWNLOAD:
            data = await att.read()
            dt = await asyncio.to_thread(docs.extract_text, data, kind)
            if docs.is_scanned(kind, dt):
                text = await asyncio.to_thread(docs.ocr_text, self.llm, self.s.gate_model, data)
                self.doc_budget.add(gid, self._day(), dt.pages)
                return text
            return dt.text
        if mime := image_mime(att):
            data = await att.read()
            note = await asyncio.to_thread(describe_image, self.llm, self.s.gate_model, data, mime)
            return format_note(note)
        raise media.MediaError("unsupported file type")


def _att_of(message, f):
    atts = list(getattr(message, "attachments", ()) or ())
    return atts[f.index] if 0 <= f.index < len(atts) else None
