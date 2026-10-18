"""Originals of the files Salid reads in Discord, kept as REAL documents of the company.

Until now a PDF, Word or spreadsheet posted in a channel became only a text note in the memory
log ("[PDF: name] summary"). `FileArchive.save` stores the original bytes next to the web
uploads (`library.make_blobs`: GCS under `salid-docs/discord-<guild>/` when `SALID_RECEIPT_BUCKET`
is set) and writes the same record shape as a web upload into `salid_org_docs`, keyed by the
guild id (= the Discord workspace's memory id), so the command center lists it under Documents
with a download button and the chat-note stays the searchable summary.

One record per distinct file content per guild (sha256), so a file posted twice is one document.
Best effort: any failure is logged by kind and the note path continues.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from .coo_store import DocStore
from .library import ORG_DOCS, STORED_TEXT

logger = logging.getLogger(__name__)

MAX_ARCHIVE_BYTES = 20 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(UTC)


class FileArchive:
    def __init__(self, docs: DocStore, blobs: Any, clock=_now):
        self.docs, self.blobs, self._clock = docs, blobs, clock

    @staticmethod
    def doc_id(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def save(
        self,
        guild_id: int,
        *,
        filename: str,
        data: bytes,
        mime: str,
        by: str = "",
        channel_id: int | None = None,
        channel_name: str = "",
        source_url: str = "",
        summary: str = "",
        tag: str = "",
        text: str = "",
    ) -> str | None:
        """Store the original and its record; returns the doc id, None when skipped or failed."""
        if not data or len(data) > MAX_ARCHIVE_BYTES:
            return None
        doc_id = self.doc_id(data)
        try:
            if self.docs.get(ORG_DOCS, guild_id, doc_id):
                return doc_id  # same content already archived for this guild
            uri = self.blobs.save(f"discord-{guild_id}", doc_id, filename, data, mime)
            rec = {
                "id": doc_id,
                "filename": filename,
                "title": filename,
                "mime": mime,
                "size": len(data),
                "uri": uri,
                "tag": tag,
                "summary": summary,
                "text": (text or "")[:STORED_TEXT],
                "created_at": self._clock().isoformat(timespec="seconds"),
                "by": by,
                "source": "discord",
                "channel_id": str(channel_id) if channel_id is not None else "",
                "channel_name": channel_name,
                "source_url": source_url,
            }
            self.docs.put(ORG_DOCS, guild_id, doc_id, rec)
            return doc_id
        except Exception as e:  # never the contents, only the kind of failure
            logger.warning("file archive failed (%s)", type(e).__name__)
            return None
