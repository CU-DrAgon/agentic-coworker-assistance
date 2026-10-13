"""Chunk transcription with Gemini on Vertex AI: Thai speech -> speaker-labelled utterances."""

import logging
import re
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from meeting_scribe.config import get_settings
from meeting_scribe.models import ChunkTranscript, Utterance

logger = logging.getLogger(__name__)

PRIOR_LIMIT = 12
TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
RETRY_DELAY_S = 5.0
NOISE_RE = re.compile(r"^[\s\W_]*$")  # empty or punctuation only


class _Turn(BaseModel):
    speaker: str
    start_s: float  # seconds from the start of THIS audio clip
    end_s: float
    text: str


class _Response(BaseModel):
    utterances: list[_Turn]


class _TransientError(Exception):
    """Retryable failure (API hiccup or unparseable model output)."""


@lru_cache
def _client() -> genai.Client:
    s = get_settings()
    return genai.Client(vertexai=True, project=s.gcp_project, location=s.gemini_location)


def _prompt(prior: list[Utterance], participants: list[str] | None, language: str) -> str:
    lines = [
        f"Transcribe this {language} meeting audio clip with speaker diarization.",
        "",
        "Rules:",
        f"- Write each utterance verbatim in {language}, in its original script. Do NOT translate.",
        "- One entry per continuous speaker turn. Start a new entry whenever the speaker changes.",
        "- start_s and end_s are seconds from the start of THIS clip (0 = first sample of the clip).",
        "- Label voices consistently within the clip as 'Speaker A', 'Speaker B', 'Speaker C', ... "
        "in order of first appearance. Distinguish speakers by voice, not by what they talk about.",
        "- Skip silence, background noise, music and filler sounds. Never invent speech. "
        "If there is no speech, return an empty list.",
    ]
    if participants:
        lines.append(
            f"- Known participants: {', '.join(participants)}. Use a participant's real name as the "
            "label ONLY when the speaker clearly introduces themselves or is directly addressed by "
            "name in this clip; otherwise keep the 'Speaker X' label."
        )
    if prior:
        context = "\n".join(f"{u.speaker}: {u.text}" for u in prior[-PRIOR_LIMIT:])
        lines += [
            "",
            "This clip continues an ongoing meeting. Previous utterances (for label continuity only; "
            "do NOT repeat them in the output):",
            context,
            "",
            "Reuse exactly the same speaker labels as above for the same voices. "
            "Use a new letter only for a voice that clearly has not spoken yet.",
        ]
    return "\n".join(lines)


def _call(model: str, gcs_uri: str, mime_type: str, prompt: str) -> _Response:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_Response,
        temperature=0.0,
        # Keep the model's default thinking: ThinkingLevel.LOW halves latency but cost
        # diarization accuracy in smoke tests (13/18 vs 18/18 turns correct).
    )
    contents = [types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type), prompt]
    try:
        resp = _client().models.generate_content(model=model, contents=contents, config=config)
    except errors.APIError as e:
        if e.code in TRANSIENT_CODES:
            raise _TransientError(f"API error {e.code}") from e
        raise
    parsed = resp.parsed
    if isinstance(parsed, _Response):
        return parsed
    try:
        return _Response.model_validate_json(resp.text or "")
    except ValidationError as e:
        raise _TransientError("unparseable model output") from e


def _clean(turns: list[_Turn], offset_s: float) -> list[Utterance]:
    out: list[Utterance] = []
    for t in turns:
        text = t.text.strip()
        if NOISE_RE.match(text):
            continue
        start = max(t.start_s, 0.0)
        end = max(t.end_s, start)
        out.append(
            Utterance(
                speaker=t.speaker.strip() or "Speaker A",
                start_s=round(offset_s + start, 2),
                end_s=round(offset_s + end, 2),
                text=text,
            )
        )
    return sorted(out, key=lambda u: u.start_s)


def transcribe_chunk(
    gcs_uri: str,
    index: int,
    offset_s: float,
    prior: list[Utterance],
    participants: list[str] | None = None,
    mime_type: str = "audio/flac",
    model: str | None = None,
) -> ChunkTranscript:
    """Transcribe one audio chunk. Times in the result are meeting-relative (chunk time + offset_s)."""
    settings = get_settings()
    model = model or settings.transcribe_model
    prompt = _prompt(prior, participants, settings.meeting_language)

    started = time.monotonic()
    try:
        result = _call(model, gcs_uri, mime_type, prompt)
    except _TransientError as e:
        logger.warning("chunk %d: transient failure (%s), retrying once", index, e)
        time.sleep(RETRY_DELAY_S)
        try:
            result = _call(model, gcs_uri, mime_type, prompt)
        except _TransientError as e2:
            raise RuntimeError(f"transcription failed for chunk {index}: {e2}") from e2

    utterances = _clean(result.utterances, offset_s)
    logger.info(
        "chunk %d transcribed with %s: %d utterances in %.1fs",
        index, model, len(utterances), time.monotonic() - started,
    )  # fmt: skip
    return ChunkTranscript(index=index, gcs_uri=gcs_uri, offset_s=offset_s, utterances=utterances)
