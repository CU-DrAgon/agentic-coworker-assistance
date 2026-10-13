"""API tests: in-memory store, every external call (ffmpeg, GCS, Gemini, ADK, Discord) faked."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from meeting_scribe import pipeline
from meeting_scribe.config import get_settings
from meeting_scribe.models import ActionItem, ChunkTranscript, Summary, Utterance


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("USE_PUBSUB", "false")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.invalid/webhook")
    monkeypatch.setattr(pipeline, "SETTLE_S", 0.0)
    monkeypatch.setattr(pipeline, "POLL_S", 0.0)
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()


@pytest.fixture
def fakes(env):
    """Patch the lazily imported collaborators and record how they were called."""
    calls = {"transcribe": [], "posted": [], "summarized": 0}

    env.setattr("meeting_scribe.audio.to_flac", lambda data: b"FLAC" + data)

    def upload_chunk(meeting_id, index, data, content_type="audio/flac"):
        return f"gs://bucket/meetings/{meeting_id}/chunk-{index:04d}.flac"

    env.setattr("meeting_scribe.storage.upload_chunk", upload_chunk)

    def transcribe_chunk(gcs_uri, index, offset_s, prior, participants=None, mime_type="audio/flac"):
        calls["transcribe"].append(
            {"index": index, "offset_s": offset_s, "prior": list(prior), "participants": participants}
        )
        utt = Utterance(speaker="Speaker A", start_s=offset_s, end_s=offset_s + 2, text=f"chunk {index}")
        return ChunkTranscript(index=index, gcs_uri=gcs_uri, offset_s=offset_s, utterances=[utt])

    env.setattr("meeting_scribe.transcribe.transcribe_chunk", transcribe_chunk)

    async def summarize_meeting(meeting):
        calls["summarized"] += 1
        return Summary(
            title="T",
            overview="O",
            action_items=[ActionItem(owner="Speaker A", task="do it")],
            speaker_names={"Speaker A": "Somchai"},
        )

    env.setattr("meeting_scribe.agents.summarize_meeting", summarize_meeting)
    env.setattr(
        "meeting_scribe.discord_post.post_summary",
        lambda meeting, url: calls["posted"].append((meeting.id, url)),
    )
    return calls


@pytest.fixture
def client(env):
    from meeting_scribe.app import app

    with TestClient(app) as c:
        yield c


def create(client, **body) -> dict:
    r = client.post("/api/meetings", json=body or {"title": "Sync", "participants": ["Somchai", " "]})
    assert r.status_code == 200
    return r.json()


def upload(client, meeting_id, index, offset_s=0.0, data=b"audio"):
    return client.post(
        f"/api/meetings/{meeting_id}/chunks",
        data={"index": index, "offset_s": offset_s},
        files={"file": ("c.webm", data, "audio/webm")},
    )


def pubsub_body(payload) -> dict:
    return {"message": {"data": base64.b64encode(json.dumps(payload).encode()).decode()}}


def test_index_and_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/")
    assert r.status_code == 200 and "Meeting Scribe" in r.text


def test_create_and_get(client):
    m = create(client)
    assert m["title"] == "Sync" and m["participants"] == ["Somchai"] and m["status"] == "recording"
    assert client.get(f"/api/meetings/{m['id']}").json()["id"] == m["id"]
    assert [x["id"] for x in client.get("/api/meetings").json()] == [m["id"]]
    assert client.get("/api/meetings/nope").status_code == 404


def test_chunk_upload_unknown_meeting_and_bad_input(client, fakes):
    assert upload(client, "nope", 0).status_code == 404
    m = create(client)
    assert upload(client, m["id"], 0, data=b"").status_code == 400
    assert upload(client, m["id"], -1).status_code == 422


def test_chunk_ordering_uses_only_lower_index_prior(client, fakes):
    m = create(client)
    # Upload out of order, as concurrent browser uploads can.
    for index in (1, 0, 2):
        r = upload(client, m["id"], index, offset_s=index * 45.0)
        assert r.status_code == 200
        assert r.json()["gcs_uri"].endswith(f"chunk-{index:04d}.flac")

    by_index = {c["index"]: c for c in fakes["transcribe"]}
    assert by_index[1]["prior"] == []  # chunk 0 had not landed yet
    assert by_index[0]["participants"] == ["Somchai"]
    assert [u.text for u in by_index[2]["prior"]] == ["chunk 0", "chunk 1"]  # sorted by index
    assert by_index[2]["offset_s"] == 90.0

    stored = client.get(f"/api/meetings/{m['id']}").json()
    assert [c["index"] for c in stored["chunks"]] == [0, 1, 2]


def test_prior_context_is_capped(client, fakes):
    m = create(client)
    store = pipeline.get_store()
    utts = [Utterance(speaker="Speaker A", start_s=i, end_s=i + 1, text=str(i)) for i in range(30)]
    store.add_chunk(m["id"], ChunkTranscript(index=0, gcs_uri="gs://x", offset_s=0, utterances=utts))
    assert upload(client, m["id"], 1, 45).status_code == 200
    assert [u.text for u in fakes["transcribe"][0]["prior"]] == [str(i) for i in range(18, 30)]


def test_undecodable_audio_is_422(client, fakes, env):
    def bad(data):
        raise RuntimeError("ffmpeg failed")

    env.setattr("meeting_scribe.audio.to_flac", bad)
    m = create(client)
    assert upload(client, m["id"], 0).status_code == 422
    assert client.get(f"/api/meetings/{m['id']}").json()["chunks"] == []


def test_finish_in_process_summarizes_and_posts(client, fakes):
    m = create(client)
    upload(client, m["id"], 0)
    upload(client, m["id"], 1, 45)

    r = client.post(f"/api/meetings/{m['id']}/finish")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert body["summary"]["speaker_names"] == {"Speaker A": "Somchai"}
    assert fakes["summarized"] == 1
    assert fakes["posted"] == [(m["id"], "https://discord.invalid/webhook")]

    # Chunks are rejected once the meeting is finished.
    assert upload(client, m["id"], 2).status_code == 409
    # Finishing again is a no-op.
    assert client.post(f"/api/meetings/{m['id']}/finish").json()["status"] == "done"
    assert fakes["summarized"] == 1 and len(fakes["posted"]) == 1


def test_finish_without_speech_marks_failed(client, fakes):
    m = create(client)
    body = client.post(f"/api/meetings/{m['id']}/finish").json()
    assert body["status"] == "failed" and "No speech" in body["error"]
    assert fakes["posted"] == []


def test_finish_unknown_meeting(client, fakes):
    assert client.post("/api/meetings/nope/finish").status_code == 404


def test_finish_skips_discord_without_webhook(client, fakes, env):
    env.setenv("DISCORD_WEBHOOK_URL", "")
    get_settings.cache_clear()
    m = create(client)
    upload(client, m["id"], 0)
    assert client.post(f"/api/meetings/{m['id']}/finish").json()["status"] == "done"
    assert fakes["posted"] == []


def test_finish_with_pubsub_publishes_instead_of_summarizing(client, fakes, env):
    env.setenv("USE_PUBSUB", "true")
    get_settings.cache_clear()
    published = []
    env.setattr(pipeline, "publish_finalize", published.append)
    m = create(client)
    upload(client, m["id"], 0)

    body = client.post(f"/api/meetings/{m['id']}/finish").json()
    assert body["status"] == "summarizing"
    assert published == [m["id"]]
    assert fakes["summarized"] == 0


def test_pubsub_finalize_decodes_and_is_idempotent(client, fakes):
    m = create(client)
    upload(client, m["id"], 0)

    for _ in range(2):  # redelivery of the same message
        r = client.post("/pubsub/finalize", json=pubsub_body({"meeting_id": m["id"]}))
        assert r.status_code == 204

    assert client.get(f"/api/meetings/{m['id']}").json()["status"] == "done"
    assert fakes["summarized"] == 1
    assert len(fakes["posted"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": {}},
        {"message": {"data": "!!!not-base64!!!"}},
        {"message": {"data": base64.b64encode(b"not json").decode()}},
        pubsub_body({"other": 1}),
        pubsub_body({"meeting_id": 5}),
    ],
)
def test_pubsub_malformed_payload_is_acked(client, fakes, payload):
    assert client.post("/pubsub/finalize", json=payload).status_code == 204
    assert fakes["summarized"] == 0


def test_pubsub_unknown_meeting_is_acked(client, fakes):
    assert client.post("/pubsub/finalize", json=pubsub_body({"meeting_id": "ghost"})).status_code == 204


def test_pubsub_failure_returns_5xx_then_retry_succeeds(fakes, env):
    from meeting_scribe.app import app

    state = {"fail": True}

    async def flaky(meeting):
        if state["fail"]:
            raise RuntimeError("model down")
        return Summary(title="T", overview="O")

    env.setattr("meeting_scribe.agents.summarize_meeting", flaky)
    with TestClient(app, raise_server_exceptions=False) as client:
        m = create(client)
        upload(client, m["id"], 0)
        body = pubsub_body({"meeting_id": m["id"]})

        assert client.post("/pubsub/finalize", json=body).status_code == 500
        failed = client.get(f"/api/meetings/{m['id']}").json()
        assert failed["status"] == "failed" and "model down" in failed["error"]

        state["fail"] = False
        assert client.post("/pubsub/finalize", json=body).status_code == 204
        assert client.get(f"/api/meetings/{m['id']}").json()["status"] == "done"


def test_static_index_ships_in_package():
    from importlib.resources import files

    assert (files("meeting_scribe") / "static" / "index.html").is_file()


def test_pubsub_oidc_rejects_and_accepts(client, fakes, env):
    env.setenv("PUBSUB_VERIFY_AUDIENCE", "https://svc.example/pubsub/finalize")
    env.setenv("PUSHER_SA_EMAIL", "pusher@proj.iam.gserviceaccount.com")
    claims = {"email": "pusher@proj.iam.gserviceaccount.com", "email_verified": True}
    seen = []

    def verifier(token, request, audience=None):
        seen.append((token, audience))
        if token == "bad":
            raise ValueError("bad signature")
        return claims if token == "good" else {"email": "attacker@evil.example"}

    env.setattr("google.oauth2.id_token.verify_oauth2_token", verifier)
    m = create(client)
    upload(client, m["id"], 0)
    body = pubsub_body({"meeting_id": m["id"]})

    assert client.post("/pubsub/finalize", json=body).status_code == 401  # no header
    assert client.post("/pubsub/finalize", json=body, headers={"Authorization": "Bearer bad"}).status_code == 401
    assert client.post("/pubsub/finalize", json=body, headers={"Authorization": "Bearer other"}).status_code == 403
    assert fakes["summarized"] == 0
    assert client.post("/pubsub/finalize", json=body, headers={"Authorization": "Bearer good"}).status_code == 204
    assert fakes["summarized"] == 1
    assert seen[-1] == ("good", "https://svc.example/pubsub/finalize")


def test_pubsub_oidc_disabled_when_audience_unset(client, fakes, env):
    env.delenv("PUBSUB_VERIFY_AUDIENCE", raising=False)
    m = create(client)
    upload(client, m["id"], 0)
    assert client.post("/pubsub/finalize", json=pubsub_body({"meeting_id": m["id"]})).status_code == 204


def test_pubsub_oidc_requires_configured_verified_identity(client, fakes, env):
    env.setenv("PUBSUB_VERIFY_AUDIENCE", "https://svc.example/pubsub/finalize")
    env.delenv("PUSHER_SA_EMAIL", raising=False)
    env.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *args, **kwargs: {"email": "pusher@proj.iam.gserviceaccount.com", "email_verified": True},
    )
    body = pubsub_body({"meeting_id": "unknown"})
    assert client.post("/pubsub/finalize", json=body, headers={"Authorization": "Bearer good"}).status_code == 503

    env.setenv("PUSHER_SA_EMAIL", "pusher@proj.iam.gserviceaccount.com")
    env.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *args, **kwargs: {"email": "pusher@proj.iam.gserviceaccount.com", "email_verified": False},
    )
    assert client.post("/pubsub/finalize", json=body, headers={"Authorization": "Bearer good"}).status_code == 403


def test_static_index_has_live_recording_safeguards():
    from importlib.resources import files

    html = (files("meeting_scribe") / "static" / "index.html").read_text(encoding="utf-8")
    assert "Use headphones" in html
    assert 'display.getTracks().forEach((t) => t.addEventListener("ended"' in html
    assert 'if (st.ctx.state === "suspended") await st.ctx.resume()' in html
    stop_block = html[html.index("async function stopRecording()") : html.index("async function finishMeeting()")]
    assert "await Promise.allSettled(st.uploads)" in stop_block
    assert "if (st.counts.failed)" in stop_block
    assert stop_block.index("if (st.counts.failed)") < stop_block.index("await finishMeeting()")
