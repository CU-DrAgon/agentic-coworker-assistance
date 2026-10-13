"""Store tests: InMemoryMeetingStore and MeetingStore against a tiny fake Firestore."""

from datetime import timedelta

import pytest

from meeting_scribe.models import (
    ActionItem,
    ChunkTranscript,
    MeetingStatus,
    Summary,
    Utterance,
    utcnow,
)
from meeting_scribe.store import InMemoryMeetingStore, MeetingStore

# --- tiny fake of the Firestore surface MeetingStore uses ---------------------------


class FakeSnap:
    def __init__(self, data: dict | None) -> None:
        self._data = data
        self.exists = data is not None

    def to_dict(self) -> dict | None:
        return None if self._data is None else dict(self._data)


class FakeQuery:
    def __init__(self, docs: list["FakeDoc"]) -> None:
        self._docs = docs

    def order_by(self, field: str, direction: str = "ASCENDING") -> "FakeQuery":
        docs = sorted(self._docs, key=lambda d: d.data[field], reverse=direction == "DESCENDING")
        return FakeQuery(docs)

    def limit(self, n: int) -> "FakeQuery":
        return FakeQuery(self._docs[:n])

    def stream(self) -> list[FakeSnap]:
        return [FakeSnap(d.data) for d in self._docs]


class FakeDoc:
    def __init__(self) -> None:
        self.data: dict | None = None
        self.subs: dict[str, FakeCollection] = {}

    def set(self, data: dict) -> None:
        self.data = dict(data)

    def update(self, fields: dict) -> None:
        if self.data is None:
            raise KeyError("NotFound")
        self.data.update(fields)

    def get(self) -> FakeSnap:
        return FakeSnap(self.data)

    def collection(self, name: str) -> "FakeCollection":
        return self.subs.setdefault(name, FakeCollection())


class FakeCollection:
    def __init__(self) -> None:
        self.docs: dict[str, FakeDoc] = {}

    def document(self, doc_id: str) -> FakeDoc:
        return self.docs.setdefault(doc_id, FakeDoc())

    def _query(self) -> FakeQuery:
        return FakeQuery([d for d in self.docs.values() if d.data is not None])

    def order_by(self, field: str, direction: str = "ASCENDING") -> FakeQuery:
        return self._query().order_by(field, direction)

    def stream(self) -> list[FakeSnap]:
        return self._query().stream()


class FakeFirestore:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def collection(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())


# --- fixtures ------------------------------------------------------------------------


def make_chunk(index: int) -> ChunkTranscript:
    return ChunkTranscript(
        index=index,
        gcs_uri=f"gs://b/meetings/m/chunk-{index:04d}.flac",
        offset_s=index * 45.0,
        utterances=[
            Utterance(
                speaker="Speaker A", start_s=index * 45.0, end_s=index * 45.0 + 4, text="สวัสดีครับ"
            )
        ],
    )


def make_summary() -> Summary:
    return Summary(
        title="ประชุมทีม",
        overview="คุยเรื่องแผนงาน",
        decisions=["เลื่อนส่งงาน"],
        action_items=[ActionItem(owner="สมชาย", task="ส่งรายงาน", due="วันศุกร์")],
        speaker_names={"Speaker A": "สมชาย"},
    )


@pytest.fixture(params=["memory", "firestore"])
def store(request):
    if request.param == "memory":
        return InMemoryMeetingStore()
    return MeetingStore(client=FakeFirestore(), collection="meetings")


# --- behaviour shared by both stores ---------------------------------------------------


def test_create_and_get(store):
    meeting = store.create("ประชุม", ["สมชาย", "สมหญิง"])
    got = store.get(meeting.id)
    assert got is not None
    assert got.title == "ประชุม"
    assert got.participants == ["สมชาย", "สมหญิง"]
    assert got.status == MeetingStatus.RECORDING
    assert got.chunks == []
    assert got.summary is None


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_add_chunk_returns_chunks_ordered_by_index(store):
    meeting = store.create("m", [])
    for i in (2, 0, 1):
        store.add_chunk(meeting.id, make_chunk(i))
    got = store.get(meeting.id)
    assert [c.index for c in got.chunks] == [0, 1, 2]
    assert got.chunks[0].utterances[0].text == "สวัสดีครับ"
    assert [u.start_s for u in got.utterances()] == [0.0, 45.0, 90.0]


def test_add_chunk_same_index_overwrites(store):
    meeting = store.create("m", [])
    store.add_chunk(meeting.id, make_chunk(0))
    replacement = make_chunk(0).model_copy(update={"utterances": []})
    store.add_chunk(meeting.id, replacement)
    got = store.get(meeting.id)
    assert len(got.chunks) == 1
    assert got.chunks[0].utterances == []


def test_set_status_and_error(store):
    meeting = store.create("m", [])
    store.set_status(meeting.id, MeetingStatus.FAILED, error="boom")
    got = store.get(meeting.id)
    assert got.status == MeetingStatus.FAILED
    assert got.error == "boom"
    store.set_status(meeting.id, MeetingStatus.DONE)
    got = store.get(meeting.id)
    assert got.status == MeetingStatus.DONE
    assert got.error is None


def test_set_summary_roundtrip(store):
    meeting = store.create("m", [])
    store.set_summary(meeting.id, make_summary())
    got = store.get(meeting.id)
    assert got.summary == make_summary()


def test_list_recent_newest_first_without_chunks(store):
    ids = []
    for i in range(3):
        meeting = store.create(f"m{i}", [])
        store.add_chunk(meeting.id, make_chunk(0))
        ids.append(meeting.id)
    # Make creation order unambiguous regardless of clock resolution.
    base = utcnow()
    for i, mid in enumerate(ids):
        _set_created_at(store, mid, base + timedelta(seconds=i))
    recent = store.list_recent(limit=2)
    assert [m.id for m in recent] == [ids[2], ids[1]]
    assert all(m.chunks == [] for m in recent)


def _set_created_at(store, meeting_id, when) -> None:
    if isinstance(store, InMemoryMeetingStore):
        store._meetings[meeting_id].created_at = when
    else:
        store.client.collection("meetings").document(meeting_id).update(
            {"created_at": when.isoformat()}
        )


def test_create_defaults_title(store):
    assert store.create("", None).title == "Untitled meeting"


def test_returned_meeting_is_a_copy():
    store = InMemoryMeetingStore()
    meeting = store.create("m", [])
    meeting.title = "mutated"
    assert store.get(meeting.id).title == "m"


# --- Firestore-specific ---------------------------------------------------------------


def test_firestore_layout_uses_chunks_subcollection():
    client = FakeFirestore()
    store = MeetingStore(client=client, collection="meetings")
    meeting = store.create("m", [])
    store.add_chunk(meeting.id, make_chunk(3))
    doc = client.collection("meetings").document(meeting.id)
    assert "chunks" not in doc.data
    assert "0003" in doc.collection("chunks").docs


def test_client_is_lazy():
    store = MeetingStore(collection="meetings")
    assert store._client is None
