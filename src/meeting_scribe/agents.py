"""ADK multi-agent summarization pipeline.

SequentialAgent[
    ParallelAgent[decisions extractor, action-item extractor, speaker-name resolver],
    synthesizer,
]

The transcript is placed in session state (`transcript`); each extractor writes its
structured result to its own `output_key`, and the synthesizer reads all three.
"""

import logging
import os
import time
import uuid
from typing import Any

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field

from .config import get_settings
from .models import ActionItem, Meeting, Summary, Utterance

logger = logging.getLogger(__name__)

APP_NAME = "meeting_scribe"
USER_ID = "scribe"

_settings = get_settings()
# ADK reads these to route Gemini calls through Vertex AI.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _settings.gcp_project)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", _settings.gemini_location)


# --- structured outputs -------------------------------------------------------------


class DecisionsOut(BaseModel):
    decisions: list[str] = Field(default_factory=list)


class ActionItemsOut(BaseModel):
    action_items: list[ActionItem] = Field(default_factory=list)


class SpeakerName(BaseModel):
    label: str  # raw label as it appears in the transcript, e.g. "Speaker A"
    name: str  # name inferred from the conversation


class SpeakerNamesOut(BaseModel):
    speaker_names: list[SpeakerName] = Field(default_factory=list)


class SynthesisOut(BaseModel):
    """Everything in Summary except decisions/action items/speaker names, which come from
    the extractors (kept out of the schema so the synthesizer cannot alter them)."""

    title: str
    overview: str
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# --- agents -------------------------------------------------------------------------

_GROUNDING = (
    "Use ONLY information stated in the transcript. Never invent facts, names, dates or "
    "tasks. Write all text in {language}, matching how the speakers talk. Keep speaker "
    "labels (for example 'Speaker A') exactly as they appear in the transcript unless "
    "told otherwise.\n\nTranscript, one turn per line as '[mm:ss] speaker: text':\n"
    "{transcript}"
)


def _build_agents(language: str) -> SequentialAgent:
    s = get_settings()
    grounding = _GROUNDING.replace("{language}", language)

    decisions = LlmAgent(
        name="decisions_extractor",
        model=s.extract_model,
        description="Extracts decisions made in the meeting.",
        instruction=(
            "You extract DECISIONS from a meeting transcript. A decision is something the "
            "group explicitly agreed or settled on (a choice made, a plan approved). "
            "Do not include tasks, proposals that were not agreed, or open questions. "
            "Each decision is one short self-contained sentence. Return an empty list if "
            "there are none.\n\n" + grounding
        ),
        output_schema=DecisionsOut,
        output_key="decisions",
    )
    actions = LlmAgent(
        name="action_item_extractor",
        model=s.extract_model,
        description="Extracts action items with owner and due date.",
        instruction=(
            "You extract ACTION ITEMS from a meeting transcript: concrete tasks someone "
            "committed to or was asked to do. For each give 'owner' (the person who will "
            "do it: use the real name if the speakers state or address it, otherwise the "
            "speaker label; 'Unassigned' only if truly nobody is responsible), 'task' "
            "(short imperative sentence) and 'due' (the deadline exactly as spoken, e.g. "
            "'Friday'; null if none was mentioned). Return an empty list if there are "
            "none.\n\n" + grounding
        ),
        output_schema=ActionItemsOut,
        output_key="action_items",
    )
    speakers = LlmAgent(
        name="speaker_name_resolver",
        model=s.extract_model,
        description="Maps speaker labels to real names.",
        instruction=(
            "You resolve SPEAKER NAMES. Map each generic speaker label (for example "
            "'Speaker A') to the person's real name, but only when the transcript makes it "
            "clear: the speaker introduces themselves, or is directly addressed by name "
            "and then replies. Use the name as spoken (first name or nickname is fine). "
            "Omit labels you cannot resolve with confidence; never guess. "
            "Participants the host listed: {participants}\n\n" + grounding
        ),
        output_schema=SpeakerNamesOut,
        output_key="speaker_names",
    )
    extractors = ParallelAgent(
        name="extractors",
        description="Runs the extractors concurrently.",
        sub_agents=[decisions, actions, speakers],
    )
    synthesizer = LlmAgent(
        name="synthesizer",
        model=s.summary_model,
        description="Writes the title, overview, key points and open questions.",
        instruction=(
            "You write the final meeting summary. Produce: 'title' (short, specific), "
            "'overview' (2-3 sentences), 'key_points' (3-7 most important points "
            "discussed, one sentence each, not repeating decisions word for word), and "
            "'open_questions' (questions raised but left unanswered; empty if none). "
            "Where a speaker's real name is known from the mapping below, use the name "
            "instead of the label.\n\n"
            "Already extracted (do not repeat them in your output):\n"
            "Decisions: {decisions}\nAction items: {action_items}\n"
            "Speaker names: {speaker_names}\n\n" + grounding
        ),
        output_schema=SynthesisOut,
        output_key="synthesis",
    )
    return SequentialAgent(
        name="meeting_summarizer",
        description="Extracts decisions, action items and speaker names, then synthesizes.",
        sub_agents=[extractors, synthesizer],
    )


root_agent = _build_agents(_settings.meeting_language)


# --- running ------------------------------------------------------------------------


def format_transcript(utterances: list[Utterance]) -> str:
    """One line per turn: '[mm:ss] speaker: text'."""
    lines = []
    for u in utterances:
        text = u.text.strip()
        if not text:
            continue
        m, s = divmod(int(u.start_s), 60)
        lines.append(f"[{m:02d}:{s:02d}] {u.speaker}: {text}")
    return "\n".join(lines)


def _parse(model: type[BaseModel], value: Any) -> Any:
    """Output-schema results land in state as a dict (or occasionally a JSON string)."""
    if isinstance(value, BaseModel):
        return model.model_validate(value.model_dump())
    if isinstance(value, str):
        return model.model_validate_json(value)
    return model.model_validate(value or {})


def build_summary(state: dict[str, Any]) -> Summary:
    """Assemble the Summary from the agents' session-state outputs."""
    synthesis = _parse(SynthesisOut, state.get("synthesis"))
    decisions = _parse(DecisionsOut, state.get("decisions"))
    actions = _parse(ActionItemsOut, state.get("action_items"))
    names = _parse(SpeakerNamesOut, state.get("speaker_names"))
    return Summary(
        title=synthesis.title,
        overview=synthesis.overview,
        key_points=synthesis.key_points,
        decisions=decisions.decisions,
        action_items=actions.action_items,
        open_questions=synthesis.open_questions,
        speaker_names={n.label: n.name for n in names.speaker_names if n.name.strip()},
    )


async def summarize_meeting(meeting: Meeting, runner: InMemoryRunner | None = None) -> Summary:
    """Run the ADK pipeline over the meeting transcript and return a Summary."""
    transcript = format_transcript(meeting.utterances())
    if not transcript:
        return Summary(
            title=meeting.title,
            overview="No speech was captured in this recording.",
        )

    runner = runner or InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session_id = uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=session_id,
        state={
            "transcript": transcript,
            "participants": ", ".join(meeting.participants) or "(none given)",
        },
    )
    message = types.Content(
        role="user", parts=[types.Part(text="Summarize this meeting.")]
    )
    started = time.monotonic()
    async for _ in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=message
    ):
        pass

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=session_id
    )
    summary = build_summary(session.state if session else {})
    logger.info(
        "Summarized meeting %s: %d utterances in %.1fs",
        meeting.id,
        len(meeting.utterances()),
        time.monotonic() - started,
    )
    return summary
