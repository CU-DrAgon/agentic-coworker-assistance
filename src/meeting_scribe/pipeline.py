"""Orchestration: chunk ingestion and meeting finalization.

Sibling modules (audio, storage, transcribe, agents, discord_post, store) are imported
lazily inside functions so this module stays cheap to import and easy to monkeypatch.
"""

import asyncio
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any

from starlette.concurrency import run_in_threadpool

from meeting_scribe.config import get_settings
from meeting_scribe.models import ChunkTranscript, Meeting, MeetingStatus

logger = logging.getLogger(__name__)

PRIOR_UTTERANCES = 12  # trailing utterances passed to the transcriber for label consistency

# finalize waits for in-flight chunk uploads: chunk count must stay unchanged this long.
SETTLE_S = 3.0
POLL_S = 1.0
SETTLE_TIMEOUT_S = 90.0

_finalize_locks: dict[str, asyncio.Lock] = {}


class AudioDecodeError(Exception):
    """The uploaded bytes could not be converted to FLAC."""


@lru_cache
def get_store() -> Any:
    """Process-wide meeting store: in-memory when STORE=memory, else Firestore."""
    if os.environ.get("STORE", "").lower() == "memory":
        from meeting_scribe.store import InMemoryMeetingStore

        return InMemoryMeetingStore()
    from meeting_scribe.store import MeetingStore

    return MeetingStore()


def process_chunk(meeting_id: str, index: int, offset_s: float, data: bytes) -> ChunkTranscript:
    """Convert, upload, and transcribe one chunk, then persist it. Blocking; call in a thread."""
    from meeting_scribe.audio import to_flac
    from meeting_scribe.storage import upload_chunk
    from meeting_scribe.transcribe import transcribe_chunk

    store = get_store()
    meeting = store.get(meeting_id)
    if meeting is None:
        raise KeyError(meeting_id)

    try:
        flac = to_flac(data)
    except Exception as exc:
        raise AudioDecodeError(str(exc)) from exc

    gcs_uri = upload_chunk(meeting_id, index, flac)
    prior = [u for c in sorted(meeting.chunks, key=lambda c: c.index) if c.index < index for u in c.utterances]
    chunk = transcribe_chunk(
        gcs_uri,
        index,
        offset_s,
        prior[-PRIOR_UTTERANCES:],
        meeting.participants or None,
    )
    store.add_chunk(meeting_id, chunk)
    logger.info("meeting=%s chunk=%d utterances=%d", meeting_id, index, len(chunk.utterances))
    return chunk


async def _wait_for_chunks(meeting_id: str) -> Meeting:
    """Return the meeting once its chunk count has been stable for SETTLE_S."""
    store = get_store()
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    last_count = -1
    stable_since = time.monotonic()
    while True:
        meeting = await run_in_threadpool(store.get, meeting_id)
        if meeting is None:
            raise KeyError(meeting_id)
        now = time.monotonic()
        if len(meeting.chunks) != last_count:
            last_count, stable_since = len(meeting.chunks), now
        if now - stable_since >= SETTLE_S or now >= deadline:
            return meeting
        await asyncio.sleep(POLL_S)


async def finalize(meeting_id: str) -> Meeting:
    """Summarize a meeting and post it to Discord. Idempotent: a DONE meeting is a no-op."""
    lock = _finalize_locks.setdefault(meeting_id, asyncio.Lock())
    async with lock:
        return await _finalize(meeting_id)


async def _finalize(meeting_id: str) -> Meeting:
    from meeting_scribe.agents import summarize_meeting
    from meeting_scribe.discord_post import post_summary

    store = get_store()
    settings = get_settings()

    meeting = await run_in_threadpool(store.get, meeting_id)
    if meeting is None:
        raise KeyError(meeting_id)
    if meeting.status == MeetingStatus.DONE:
        logger.info("meeting=%s already done; skipping", meeting_id)
        return meeting

    try:
        await run_in_threadpool(store.set_status, meeting_id, MeetingStatus.SUMMARIZING)
        meeting = await _wait_for_chunks(meeting_id)
        if not meeting.utterances():
            raise ValueError("No speech was transcribed, so there is nothing to summarize.")

        if meeting.summary is None:  # a retry after a Discord failure reuses the saved summary
            summary = await summarize_meeting(meeting)
            await run_in_threadpool(store.set_summary, meeting_id, summary)
            meeting = meeting.model_copy(update={"summary": summary})

        if settings.discord_webhook_url:
            await run_in_threadpool(post_summary, meeting, settings.discord_webhook_url)
        else:
            logger.info("meeting=%s no Discord webhook configured; skipping post", meeting_id)

        await run_in_threadpool(store.set_status, meeting_id, MeetingStatus.DONE)
    except Exception as exc:
        logger.exception("meeting=%s finalize failed", meeting_id)
        await run_in_threadpool(store.set_status, meeting_id, MeetingStatus.FAILED, str(exc))
        raise
    finally:
        _finalize_locks.pop(meeting_id, None)

    return await run_in_threadpool(store.get, meeting_id) or meeting


def publish_finalize(meeting_id: str) -> None:
    """Publish {meeting_id} to the finalize topic. Blocking; call in a thread."""
    from google.cloud import pubsub_v1

    settings = get_settings()
    publisher = pubsub_v1.PublisherClient()
    topic = publisher.topic_path(settings.gcp_project, settings.pubsub_topic)
    publisher.publish(topic, json.dumps({"meeting_id": meeting_id}).encode()).result(timeout=30)
