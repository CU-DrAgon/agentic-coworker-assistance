from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from meeting_scribe import transcribe as t
from meeting_scribe.models import Utterance


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t.time, "sleep", lambda _s: None)


def _fake_client(monkeypatch: pytest.MonkeyPatch, *responses: object) -> MagicMock:
    """Client whose generate_content yields each response (exceptions are raised)."""
    client = MagicMock()
    client.models.generate_content.side_effect = list(responses)
    monkeypatch.setattr(t, "_client", lambda: client)
    return client


def _resp(*turns: tuple[str, float, float, str]) -> SimpleNamespace:
    parsed = t._Response(
        utterances=[t._Turn(speaker=s, start_s=a, end_s=b, text=x) for s, a, b, x in turns]
    )
    return SimpleNamespace(parsed=parsed, text=parsed.model_dump_json())


def test_offsets_and_cleaning(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_client(
        monkeypatch,
        _resp(
            ("Speaker B", 5.0, 8.0, " สบายดีครับ "),
            ("Speaker A", 1.0, 4.0, "สวัสดีครับ"),
            ("Speaker A", 9.0, 9.5, "..."),  # noise
            ("Speaker A", 10.0, 11.0, "   "),  # empty
            ("Speaker C", -1.0, -2.0, "ผิดเวลา"),  # bad times get clamped
        ),
    )
    chunk = t.transcribe_chunk("gs://b/c.flac", index=2, offset_s=90.0, prior=[])
    assert chunk.index == 2 and chunk.offset_s == 90.0 and chunk.gcs_uri == "gs://b/c.flac"
    assert [(u.speaker, u.start_s, u.end_s) for u in chunk.utterances] == [
        ("Speaker C", 90.0, 90.0),
        ("Speaker A", 91.0, 94.0),
        ("Speaker B", 95.0, 98.0),
    ]
    assert chunk.utterances[2].text == "สบายดีครับ"


def test_prompt_includes_prior_and_participants(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(monkeypatch, _resp())
    prior = [
        Utterance(speaker=f"Speaker {c}", start_s=i, end_s=i + 1, text=f"ข้อความ{i}")
        for i, c in enumerate("ABCABCABCABCABC")
    ]
    chunk = t.transcribe_chunk("gs://b/c.flac", 1, 45.0, prior, participants=["Somchai", "Malee"])
    assert chunk.utterances == []
    contents = client.models.generate_content.call_args.kwargs["contents"]
    prompt = contents[1]
    assert "Somchai, Malee" in prompt
    assert "ข้อความ14" in prompt and "ข้อความ3" in prompt  # last 12 of 15
    assert "ข้อความ2\n" not in prompt  # 13th from the end is dropped
    assert "Thai" in prompt and "Do NOT translate or transliterate" in prompt
    assert "preserve English code-switching in English" in prompt


def test_first_chunk_prompt_has_no_prior_section(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(monkeypatch, _resp())
    t.transcribe_chunk("gs://b/c.flac", 0, 0.0, [])
    prompt = client.models.generate_content.call_args.kwargs["contents"][1]
    assert "Previous utterances" not in prompt
    assert "Known participants" not in prompt


def test_retries_once_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(
        monkeypatch,
        errors.ServerError(503, {"error": {"message": "unavailable"}}),
        _resp(("Speaker A", 0.0, 1.0, "สวัสดี")),
    )
    chunk = t.transcribe_chunk("gs://b/c.flac", 0, 0.0, [])
    assert len(chunk.utterances) == 1
    assert client.models.generate_content.call_count == 2


def test_retries_on_unparseable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = SimpleNamespace(parsed=None, text="not json")
    client = _fake_client(monkeypatch, bad, _resp(("Speaker A", 0.0, 1.0, "ครับ")))
    assert len(t.transcribe_chunk("gs://b/c.flac", 0, 0.0, []).utterances) == 1
    assert client.models.generate_content.call_count == 2


def test_gives_up_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    err = errors.ServerError(503, {"error": {"message": "unavailable"}})
    client = _fake_client(monkeypatch, err, err, err)
    with pytest.raises(RuntimeError, match="chunk 3"):
        t.transcribe_chunk("gs://b/c.flac", 3, 0.0, [])
    assert client.models.generate_content.call_count == 2


def test_non_transient_error_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(monkeypatch, errors.ClientError(400, {"error": {"message": "bad"}}))
    with pytest.raises(errors.ClientError):
        t.transcribe_chunk("gs://b/c.flac", 0, 0.0, [])
    assert client.models.generate_content.call_count == 1


def test_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(monkeypatch, _resp())
    t.transcribe_chunk("gs://b/c.flac", 0, 0.0, [], model="gemini-3.1-pro-preview")
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-3.1-pro-preview"
