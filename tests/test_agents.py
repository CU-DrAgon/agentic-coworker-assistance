"""Offline tests for the ADK pipeline: structure, state wiring and Summary assembly."""

import json
from pathlib import Path
from types import SimpleNamespace

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
    with pytest.raises(Exception):
        await agents.summarize_meeting(_meeting(us), runner=runner)


def test_fixture_shape() -> None:
    data = json.loads(FIXTURE.read_text())
    assert 20 <= len(data) <= 30
    assert {u["speaker"] for u in data} == {"Speaker A", "Speaker B", "Speaker C"}
