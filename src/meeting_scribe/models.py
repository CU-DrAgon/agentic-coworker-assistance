"""Shared data contract. Every module reads and writes these types."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Utterance(BaseModel):
    """One speaker turn. Times are seconds from the start of the whole meeting."""

    speaker: str  # stable label across chunks, e.g. "Speaker A" or a real name
    start_s: float
    end_s: float
    text: str


class ChunkTranscript(BaseModel):
    """Transcript for one uploaded audio chunk (~45-60 s)."""

    index: int  # 0-based order within the meeting
    gcs_uri: str
    offset_s: float  # where this chunk starts in the meeting timeline
    utterances: list[Utterance] = Field(default_factory=list)


class MeetingStatus(StrEnum):
    RECORDING = "recording"
    SUMMARIZING = "summarizing"
    DONE = "done"
    FAILED = "failed"


class ActionItem(BaseModel):
    owner: str  # speaker name/label, or "Unassigned"
    task: str
    due: str | None = None  # free text as spoken, e.g. "วันศุกร์"


class Summary(BaseModel):
    title: str
    overview: str  # 2-3 sentences
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Maps raw labels ("Speaker A") to names inferred from the conversation.
    speaker_names: dict[str, str] = Field(default_factory=dict)


class Meeting(BaseModel):
    id: str
    title: str = "Untitled meeting"
    participants: list[str] = Field(default_factory=list)  # optional names typed by the host
    status: MeetingStatus = MeetingStatus.RECORDING
    created_at: datetime = Field(default_factory=utcnow)
    chunks: list[ChunkTranscript] = Field(default_factory=list)
    summary: Summary | None = None
    error: str | None = None

    def utterances(self) -> list[Utterance]:
        """All utterances in meeting order."""
        return [u for c in sorted(self.chunks, key=lambda c: c.index) for u in c.utterances]
