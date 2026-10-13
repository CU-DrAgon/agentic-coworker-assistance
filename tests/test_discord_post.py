"""Discord posting tests using httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from meeting_scribe import discord_post
from meeting_scribe.discord_post import (
    DiscordError,
    build_embed,
    bullet_list,
    format_transcript,
    post_summary,
)
from meeting_scribe.models import ActionItem, ChunkTranscript, Meeting, Summary, Utterance

URL = "https://discord.test/api/webhooks/1/SECRET-TOKEN"


def make_meeting(summary: Summary | None = None) -> Meeting:
    return Meeting(
        id="abc123",
        title="ประชุมทีม",
        chunks=[
            ChunkTranscript(
                index=1,
                gcs_uri="gs://b/x",
                offset_s=45,
                utterances=[Utterance(speaker="Speaker B", start_s=65, end_s=70, text="ตกลงครับ")],
            ),
            ChunkTranscript(
                index=0,
                gcs_uri="gs://b/y",
                offset_s=0,
                utterances=[Utterance(speaker="Speaker A", start_s=3, end_s=9, text="สวัสดีครับ")],
            ),
        ],
        summary=summary,
    )


def make_summary(**kw) -> Summary:
    base = dict(
        title="ประชุมวางแผน",
        overview="ทีมคุยเรื่องกำหนดส่งงาน",
        key_points=["จุดที่ 1", "จุดที่ 2"],
        decisions=["เลื่อนส่งงานเป็นวันศุกร์"],
        action_items=[
            ActionItem(owner="สมชาย", task="ส่งรายงาน", due="วันศุกร์"),
            ActionItem(owner="Unassigned", task="จองห้อง"),
        ],
        open_questions=["งบพอไหม"],
        speaker_names={"Speaker A": "สมชาย"},
    )
    base.update(kw)
    return Summary(**base)


def embed_size(embed: dict) -> int:
    total = len(embed.get("title", "")) + len(embed.get("description", ""))
    total += len(embed.get("footer", {}).get("text", ""))
    total += sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    return total


# --- formatting ----------------------------------------------------------------------


def test_transcript_ordered_with_names_applied():
    text = format_transcript(make_meeting(make_summary()))
    assert text.splitlines() == ["[00:03] สมชาย: สวัสดีครับ", "[01:05] Speaker B: ตกลงครับ"]


def test_transcript_without_summary_keeps_labels():
    assert format_transcript(make_meeting()).startswith("[00:03] Speaker A:")


def test_embed_contents():
    embed = build_embed(make_meeting(make_summary()))
    assert embed["title"] == "ประชุมวางแผน"
    assert embed["description"] == "ทีมคุยเรื่องกำหนดส่งงาน"
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    assert list(fields) == ["Key points", "Decisions", "Action items", "Open questions"]
    assert "• **สมชาย** — ส่งรายงาน (วันศุกร์)" in fields["Action items"]
    assert "• **Unassigned** — จองห้อง" in fields["Action items"]
    assert "(" not in fields["Action items"].splitlines()[1]
    assert embed["footer"]["text"] == "Duration 1m 10s · 2 speakers"


def test_empty_sections_are_omitted():
    embed = build_embed(make_meeting(make_summary(decisions=[], open_questions=[])))
    assert [f["name"] for f in embed["fields"]] == ["Key points", "Action items"]


def test_no_summary_still_builds():
    embed = build_embed(make_meeting())
    assert embed["title"] == "ประชุมทีม"
    assert embed["fields"] == []


def test_bullet_list_truncates_with_more_note():
    lines = [f"item number {i} " + "x" * 40 for i in range(50)]
    out = bullet_list(lines, limit=300)
    assert len(out) <= 300
    assert out.splitlines()[-1].startswith("… +")


def test_bullet_list_fits_when_small():
    assert bullet_list(["a", "b"]) == "• a\n• b"


def test_embed_respects_discord_limits():
    long = "ก" * 5000
    summary = make_summary(
        title=long,
        overview=long,
        key_points=[long] * 30,
        decisions=[long] * 30,
        action_items=[ActionItem(owner="o" * 300, task=long, due=long)] * 30,
        open_questions=[long] * 30,
    )
    embed = build_embed(make_meeting(summary))
    assert len(embed["title"]) <= 256
    assert len(embed["description"]) <= 4096
    assert all(len(f["name"]) <= 256 and 0 < len(f["value"]) <= 1024 for f in embed["fields"])
    assert len(embed["fields"]) <= 25
    assert embed_size(embed) <= 6000


# --- HTTP behaviour --------------------------------------------------------------------


def capture_client(responses: list[httpx.Response]):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses[min(len(requests), len(responses)) - 1]

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_post_sends_multipart_with_embed_and_transcript():
    client, requests = capture_client([httpx.Response(204)])
    post_summary(make_meeting(make_summary()), URL, client=client)
    (request,) = requests
    assert str(request.url) == URL
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content.decode("utf-8")
    assert 'name="payload_json"' in body
    assert 'name="files[0]"; filename="transcript-abc123.txt"' in body
    assert "[00:03] สมชาย: สวัสดีครับ" in body
    payload = json.loads(body.split("\r\n\r\n", 2)[1].split("\r\n--", 1)[0])
    assert payload["embeds"][0]["title"] == "ประชุมวางแผน"
    assert payload["attachments"] == [{"id": 0, "filename": "transcript-abc123.txt"}]


def test_429_retries_once_using_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(discord_post.time, "sleep", sleeps.append)
    client, requests = capture_client(
        [httpx.Response(429, json={"retry_after": 0.5}), httpx.Response(204)]
    )
    post_summary(make_meeting(make_summary()), URL, client=client)
    assert len(requests) == 2
    assert sleeps == [0.5]
    # The retry re-sends the full multipart body, including the file.
    assert b"transcript-abc123.txt" in requests[1].content


def test_429_twice_raises(monkeypatch):
    monkeypatch.setattr(discord_post.time, "sleep", lambda s: None)
    client, requests = capture_client([httpx.Response(429, json={"retry_after": 0})])
    with pytest.raises(DiscordError, match="429"):
        post_summary(make_meeting(make_summary()), URL, client=client)
    assert len(requests) == 2


def test_retry_after_is_capped(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(discord_post.time, "sleep", sleeps.append)
    client, _ = capture_client([httpx.Response(429, json={"retry_after": 999}), httpx.Response(204)])
    post_summary(make_meeting(make_summary()), URL, client=client)
    assert sleeps == [discord_post.MAX_RETRY_WAIT_S]


def test_non_2xx_raises_without_leaking_webhook_url():
    client, requests = capture_client([httpx.Response(400, json={"message": "bad"})])
    with pytest.raises(DiscordError) as exc:
        post_summary(make_meeting(make_summary()), URL, client=client)
    assert "400" in str(exc.value)
    assert "SECRET-TOKEN" not in str(exc.value)
    assert len(requests) == 1  # no retry on non-429
