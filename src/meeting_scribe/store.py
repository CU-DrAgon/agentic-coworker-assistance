"""Meeting persistence: Firestore in production, in-memory for tests and local runs.

Layout: `{collection}/{meeting_id}` holds the meeting fields (no chunks);
`{collection}/{meeting_id}/chunks/{index:04d}` holds one ChunkTranscript each.
"""

import logging
import uuid
from typing import Any, Protocol

from .config import get_settings
from .models import ChunkTranscript, Meeting, MeetingStatus, Summary

logger = logging.getLogger(__name__)


class Store(Protocol):
    def create(self, title: str, participants: list[str] | None = None) -> Meeting: ...
    def get(self, meeting_id: str) -> Meeting | None: ...
    def add_chunk(self, meeting_id: str, chunk: ChunkTranscript) -> None: ...
    def set_status(
        self, meeting_id: str, status: MeetingStatus, error: str | None = None
    ) -> None: ...
    def set_summary(self, meeting_id: str, summary: Summary) -> None: ...
    def list_recent(self, limit: int = 20) -> list[Meeting]: ...


def _new_meeting(title: str, participants: list[str] | None) -> Meeting:
    return Meeting(
        id=uuid.uuid4().hex[:12],
        title=title or "Untitled meeting",
        participants=list(participants or []),
    )


class MeetingStore:
    """Firestore-backed store. The client is created lazily on first use."""

    def __init__(self, client: Any = None, collection: str | None = None) -> None:
        self._client = client
        self._collection = collection or get_settings().firestore_collection

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=get_settings().gcp_project)
        return self._client

    def _doc(self, meeting_id: str) -> Any:
        return self.client.collection(self._collection).document(meeting_id)

    @staticmethod
    def _to_meeting(data: dict[str, Any], chunks: list[ChunkTranscript] | None = None) -> Meeting:
        return Meeting.model_validate({**data, "chunks": chunks or []})

    def create(self, title: str, participants: list[str] | None = None) -> Meeting:
        meeting = _new_meeting(title, participants)
        self._doc(meeting.id).set(meeting.model_dump(mode="json", exclude={"chunks"}))
        logger.info("Created meeting %s", meeting.id)
        return meeting

    def get(self, meeting_id: str) -> Meeting | None:
        doc = self._doc(meeting_id)
        snap = doc.get()
        if not snap.exists:
            return None
        chunks = [ChunkTranscript.model_validate(s.to_dict()) for s in doc.collection("chunks").stream()]
        chunks.sort(key=lambda c: c.index)
        return self._to_meeting(snap.to_dict(), chunks)

    def add_chunk(self, meeting_id: str, chunk: ChunkTranscript) -> None:
        self._doc(meeting_id).collection("chunks").document(f"{chunk.index:04d}").set(
            chunk.model_dump(mode="json")
        )

    def set_status(
        self, meeting_id: str, status: MeetingStatus, error: str | None = None
    ) -> None:
        self._doc(meeting_id).update({"status": status.value, "error": error})

    def set_summary(self, meeting_id: str, summary: Summary) -> None:
        self._doc(meeting_id).update({"summary": summary.model_dump(mode="json")})

    def list_recent(self, limit: int = 20) -> list[Meeting]:
        query = (
            self.client.collection(self._collection)
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
        )
        return [self._to_meeting(s.to_dict()) for s in query.stream()]


class InMemoryMeetingStore:
    """Same interface as MeetingStore, for tests and `STORE=memory`."""

    def __init__(self) -> None:
        self._meetings: dict[str, Meeting] = {}

    def create(self, title: str, participants: list[str] | None = None) -> Meeting:
        meeting = _new_meeting(title, participants)
        self._meetings[meeting.id] = meeting
        return meeting.model_copy(deep=True)

    def get(self, meeting_id: str) -> Meeting | None:
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            return None
        copy = meeting.model_copy(deep=True)
        copy.chunks.sort(key=lambda c: c.index)
        return copy

    def add_chunk(self, meeting_id: str, chunk: ChunkTranscript) -> None:
        meeting = self._meetings[meeting_id]
        meeting.chunks = [c for c in meeting.chunks if c.index != chunk.index]
        meeting.chunks.append(chunk.model_copy(deep=True))

    def set_status(
        self, meeting_id: str, status: MeetingStatus, error: str | None = None
    ) -> None:
        meeting = self._meetings[meeting_id]
        meeting.status = status
        meeting.error = error

    def set_summary(self, meeting_id: str, summary: Summary) -> None:
        self._meetings[meeting_id].summary = summary.model_copy(deep=True)

    def list_recent(self, limit: int = 20) -> list[Meeting]:
        recent = sorted(self._meetings.values(), key=lambda m: m.created_at, reverse=True)
        return [m.model_copy(update={"chunks": []}, deep=True) for m in recent[:limit]]
