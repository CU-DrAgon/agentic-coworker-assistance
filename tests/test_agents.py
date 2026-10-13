"""Offline tests for the ADK pipeline: structure, state wiring and Summary assembly."""

import json
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

from meeting_scribe import agents
from meeting_scribe.config import get_settings
from meeting_scribe.models import ChunkTranscript, Meeting, Summary, Utterance

FIXTURE = Path(__file__).parent / "fixtures" / "thai_meeting.json"


class FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[str, SimpleNamespace] = {}

    async def create_session(self, *, app_name, user_id, session_id, state=None):
        self.sessions[session_id] = SimpleNamespace(state=dict(state or {}))
        return self.sessions[session_id]

    async def get_session(self, *, app_name, user_id, session_id):
        return self.sessions.get(session_id)


class FakeRunner:
    """Stands in for InMemoryRunner: 'runs' the agents by writing canned outputs to state."""

    app_name = "test"

    def __init__(self, outputs: dict | None = None) -> None:
        self.session_service = FakeSessionService()
        self.outputs = outputs
        self.calls = 0

    async def run_async(self, *, user_id, session_id, new_message):
        self.calls += 1
        self.session_service.sessions[session_id].state.update(self.outputs or {})
        return
        yield  # makes this an async generator


def _meeting(utterances: list[Utterance]) -> Meeting:
    chunk = ChunkTranscript(index=0, gcs_uri="gs://b/c", offset_s=0, utterances=utterances)
    return Meeting(id="m1", title="Weekly sync", participants=["Somchai"], chunks=[chunk])


CANNED = {
    "decisions": {"decisions": ["Launch on 15 November"]},
    "action_items": {
        "action_items": [{"owner": "Wit", "task": "Finish notifications", "due": "Friday"}]
    },
    "speaker_names": {"speaker_names": [{"label": "Speaker A", "name": "Somchai"}]},
    "synthesis": {
        "title": "Launch planning",
        "overview": "The team set a launch date.",
        "key_points": ["Backend nearly done"],
        "open_questions": ["English support?"],
    },
}


def test_root_agent_structure() -> None:
    root = agents.root_agent
    assert isinstance(root, SequentialAgent)
    extractors, synthesizer = root.sub_agents
    assert isinstance(extractors, ParallelAgent)
    assert [a.output_key for a in extractors.sub_agents] == [
        "decisions",
        "action_items",
        "speaker_names",
    ]
    assert all(isinstance(a, LlmAgent) for a in extractors.sub_agents)
    s = get_settings()
    assert {a.model for a in extractors.sub_agents} == {s.extract_model}
    assert isinstance(synthesizer, LlmAgent)
    assert synthesizer.model == s.summary_model
    assert synthesizer.output_key == "synthesis"
    # Every {placeholder} the instructions read must be provided via session state.
    for a in (*extractors.sub_agents, synthesizer):
        assert "{transcript}" in a.instruction


def test_format_transcript() -> None:
    us = [
        Utterance(speaker="Speaker A", start_s=65.4, end_s=70, text=" สวัสดี "),
        Utterance(speaker="Speaker B", start_s=70, end_s=71, text="   "),
    ]
    assert agents.format_transcript(us) == "[01:05] Speaker A: สวัสดี"


def test_build_summary_from_state() -> None:
    summary = agents.build_summary(CANNED)
    assert summary.title == "Launch planning"
    assert summary.decisions == ["Launch on 15 November"]
    assert summary.action_items[0].owner == "Wit"
    assert summary.action_items[0].due == "Friday"
    assert summary.open_questions == ["English support?"]
    assert summary.speaker_names == {"Speaker A": "Somchai"}


def test_build_summary_accepts_json_strings_and_missing_lists() -> None:
    state = {
        "decisions": json.dumps({"decisions": []}),
        "action_items": {},
        "speaker_names": {"speaker_names": [{"label": "Speaker B", "name": "  "}]},
        "synthesis": json.dumps({"title": "T", "overview": "O"}),
    }
    summary = agents.build_summary(state)
    assert summary.decisions == [] and summary.action_items == []
    assert summary.speaker_names == {}  # blank names dropped


@pytest.mark.asyncio
async def test_summarize_meeting_with_mocked_runner() -> None:
    runner = FakeRunner(CANNED)
    fixture = [Utterance(**u) for u in json.loads(FIXTURE.read_text())]
    summary = await agents.summarize_meeting(_meeting(fixture), runner=runner)

    assert isinstance(summary, Summary)
    assert summary.title == "Launch planning"
    assert runner.calls == 1
    # The transcript and participants were handed to the agents through session state.
    state = next(iter(runner.session_service.sessions.values())).state
    assert state["transcript"].startswith("[00:00] Speaker A:")
    assert state["transcript"].count("\n") == len(fixture) - 1
    assert state["participants"] == "Somchai"


@pytest.mark.asyncio
async def test_empty_transcript_skips_llm() -> None:
    runner = FakeRunner(CANNED)
    summary = await agents.summarize_meeting(_meeting([]), runner=runner)
    assert runner.calls == 0
    assert summary.title == "Weekly sync"
    assert summary.decisions == [] and summary.action_items == []


@pytest.mark.asyncio
async def test_missing_synthesis_raises() -> None:
    """If the synthesizer produced nothing, fail loudly so the pipeline marks FAILED."""
    runner = FakeRunner({})
    us = [Utterance(speaker="Speaker A", start_s=0, end_s=1, text="hello")]
    with pytest.raises(pydantic.ValidationError):
        await agents.summarize_meeting(_meeting(us), runner=runner)


def test_fixture_shape() -> None:
    data = json.loads(FIXTURE.read_text())
    assert 20 <= len(data) <= 30
    assert {u["speaker"] for u in data} == {"Speaker A", "Speaker B", "Speaker C"}


# --- speaker-label resolution ---------------------------------------------------------


def _meet1(participants: list[str]) -> Meeting:
    """Shape of the first real meeting: names and generic labels mixed across chunks."""
    us = [
        Utterance(speaker="Toey", start_s=0, end_s=1, text="เริ่มประชุมนะคะ"),
        Utterance(speaker="Speaker B", start_s=1, end_s=2, text="ครับ"),
        Utterance(speaker="Speaker C", start_s=2, end_s=3, text="โอเค"),
        Utterance(speaker="Speaker D", start_s=3, end_s=4, text="ได้"),
        Utterance(speaker="Jems", start_s=4, end_s=5, text="ผมเจมส์ครับ"),
    ]
    m = _meeting(us)
    m.participants = participants
    return m


def test_unresolved_labels_with_participants() -> None:
    labels = agents.unresolved_labels(_meet1(["Peat", "Toey", "Jems"]))
    assert labels == ["Speaker B", "Speaker C", "Speaker D"]


def test_unresolved_labels_without_participants() -> None:
    assert agents.unresolved_labels(_meet1([])) == ["Speaker B", "Speaker C", "Speaker D"]
    m = _meet1([])
    m.chunks[0].utterances.append(Utterance(speaker="Unknown", start_s=5, end_s=6, text="x"))
    assert "Unknown" not in agents.unresolved_labels(m)


def test_resolve_speaker_names_maps_every_label() -> None:
    raw = {"Speaker B": "เตย", "Speaker C": "Somchai", "Toey": "Toey"}
    names = agents.resolve_speaker_names(
        raw, ["Speaker B", "Speaker C", "Speaker D"], ["Peat", "Toey", "Jems"], transcript=""
    )
    # Every unresolved label gets an entry; non-participant names are never invented.
    assert names == {"Speaker B": "Unknown", "Speaker C": "Unknown", "Speaker D": "Unknown"}
    raw = {"Speaker B": "toey", "Speaker C": "Peat"}
    names = agents.resolve_speaker_names(
        raw, ["Speaker B", "Speaker C", "Speaker D"], ["Peat", "Toey", "Jems"], transcript=""
    )
    assert names == {"Speaker B": "Toey", "Speaker C": "Peat", "Speaker D": "Unknown"}


def test_resolve_speaker_names_without_participants_requires_name_in_transcript() -> None:
    raw = {"Speaker A": "สมชาย", "Speaker B": "วิทย์"}
    names = agents.resolve_speaker_names(
        raw, ["Speaker A", "Speaker B", "Speaker C"], [], transcript="[00:00] Speaker A: ผมสมชาย"
    )
    assert names == {"Speaker A": "สมชาย", "Speaker B": "Unknown", "Speaker C": "Unknown"}


@pytest.mark.asyncio
async def test_summarize_resolves_labels_and_action_owners() -> None:
    outputs = dict(CANNED)
    outputs["speaker_names"] = {
        "speaker_names": [
            {"label": "Speaker B", "name": "เตย"},
            {"label": "Speaker C", "name": "Peat"},
        ]
    }
    outputs["action_items"] = {
        "action_items": [
            {"owner": "Speaker C", "task": "ส่งสเปกให้ทีม", "due": None},
            {"owner": "Speaker D", "task": "เตรียมสไลด์", "due": "วันศุกร์"},
            {"owner": "Jems", "task": "ตั้งค่า CI", "due": None},
        ]
    }
    runner = FakeRunner(outputs)
    summary = await agents.summarize_meeting(_meet1(["Peat", "Toey", "Jems"]), runner=runner)
    assert summary.speaker_names == {
        "Speaker B": "Unknown",
        "Speaker C": "Peat",
        "Speaker D": "Unknown",
    }
    assert [a.owner for a in summary.action_items] == ["Peat", "Unknown", "Jems"]
    state = next(iter(runner.session_service.sessions.values())).state
    assert state["unresolved_labels"] == "Speaker B, Speaker C, Speaker D"


def test_resolver_instruction_requires_full_coverage() -> None:
    extractors, _ = agents.root_agent.sub_agents
    resolver = extractors.sub_agents[2]
    assert "{unresolved_labels}" in resolver.instruction
    assert "EVERY" in resolver.instruction and "'Unknown'" in resolver.instruction


# --- action items -----------------------------------------------------------------------


def test_recorder_instructions_are_not_action_items() -> None:
    state = dict(CANNED)
    state["action_items"] = {
        "action_items": [
            {"owner": "Speaker B", "task": "หยุดบันทึกและสรุปข้อมูลการประชุม", "due": None},
            {"owner": "Toey", "task": "Stop recording and summarize", "due": None},
            {"owner": "Jems", "task": "ส่งสรุปการประชุมให้ลูกค้า", "due": "พรุ่งนี้"},
        ]
    }
    summary = agents.build_summary(state)
    assert [a.task for a in summary.action_items] == ["ส่งสรุปการประชุมให้ลูกค้า"]


def test_action_item_instruction_excludes_logistics() -> None:
    extractors, _ = agents.root_agent.sub_agents
    actions = extractors.sub_agents[1]
    text = actions.instruction
    assert "recorder" in text.lower() or "recording" in text.lower()
    assert "deliverable" in text.lower()


@pytest.mark.parametrize(
    ("task", "kept"),
    [
        ("หยุดบันทึกและสรุปข้อมูลการประชุม", False),
        ("หยุดบันทึก", False),
        ("Stop recording and summarize", False),
        ("stop the recording", False),
        ("Stop recording expenses by hand", True),
        ("เริ่มบันทึกข้อมูลลงระบบ", True),
        ("เริ่มอัดวิดีโอสอนใช้งาน", True),
        ("ส่งสรุปการประชุมให้ลูกค้า", True),
    ],
)
def test_recorder_regex_cases(task: str, kept: bool) -> None:
    state = dict(CANNED)
    state["action_items"] = {"action_items": [{"owner": "Wit", "task": task, "due": None}]}
    assert bool(agents.build_summary(state).action_items) is kept


def test_resolve_names_without_participants_checks_body_only() -> None:
    transcript = "[00:00] Speaker A: ผมชื่อ Somchai ครับ\n[00:05] Speaker B: โอเค"
    raw = {"Speaker A": "somchai", "Speaker B": "Speaker", "Speaker C": "A"}
    names = agents.resolve_speaker_names(
        raw, ["Speaker A", "Speaker B", "Speaker C"], [], transcript
    )
    assert names == {"Speaker A": "somchai", "Speaker B": "Unknown", "Speaker C": "Unknown"}


def test_action_owner_lookup_is_case_and_space_insensitive() -> None:
    state = dict(CANNED)
    state["speaker_names"] = {"speaker_names": [{"label": "Speaker C", "name": "Peat"}]}
    state["action_items"] = {
        "action_items": [
            {"owner": "speaker c", "task": "a", "due": None},
            {"owner": "Speaker C ", "task": "b", "due": None},
            {"owner": "toey", "task": "c", "due": None},
            {"owner": "Wit", "task": "d", "due": None},
        ]
    }
    s = agents.build_summary(state, labels=["Speaker C"], participants=["Peat", "Toey"])
    assert [a.owner for a in s.action_items] == ["Peat", "Peat", "Toey", "Wit"]


@pytest.mark.asyncio
async def test_summarize_with_no_unresolved_labels() -> None:
    us = [Utterance(speaker="Toey", start_s=0, end_s=1, text="สวัสดี")]
    m = _meeting(us)
    m.participants = ["Peat", "Toey"]
    runner = FakeRunner(dict(CANNED))
    summary = await agents.summarize_meeting(m, runner=runner)
    assert summary.speaker_names == {}
    state = next(iter(runner.session_service.sessions.values())).state
    assert state["unresolved_labels"] == "(none)"
