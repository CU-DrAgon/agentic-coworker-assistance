"""FastAPI app: static UI, meeting/chunk API, and the Pub/Sub finalize push endpoint."""

import base64
import binascii
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from meeting_scribe import pipeline
from meeting_scribe.config import get_settings
from meeting_scribe.models import ChunkTranscript, Meeting, MeetingStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
MAX_CHUNK_BYTES = 32 * 1024 * 1024  # Cloud Run's HTTP/1 request limit

app = FastAPI(title="Meeting Scribe")


class CreateMeeting(BaseModel):
    title: str = Field(default="Untitled meeting", max_length=200)
    participants: list[str] = Field(default_factory=list, max_length=20)


def _get_or_404(meeting_id: str) -> Meeting:
    meeting = pipeline.get_store().get(meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found")
    return meeting


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/meetings")
def create_meeting(body: CreateMeeting) -> Meeting:
    title = body.title.strip() or "Untitled meeting"
    participants = [p.strip() for p in body.participants if p.strip()]
    return pipeline.get_store().create(title, participants)


@app.get("/api/meetings")
def list_meetings() -> list[Meeting]:
    return pipeline.get_store().list_recent(20)


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: str) -> Meeting:
    return _get_or_404(meeting_id)


@app.post("/api/meetings/{meeting_id}/chunks")
async def upload_chunk(
    meeting_id: str,
    file: UploadFile = File(...),
    index: int = Form(..., ge=0),
    offset_s: float = Form(0.0, ge=0),
) -> ChunkTranscript:
    meeting = await run_in_threadpool(_get_or_404, meeting_id)
    if meeting.status != MeetingStatus.RECORDING:
        raise HTTPException(409, f"Meeting is {meeting.status.value}; chunks are no longer accepted")
    data = await file.read(MAX_CHUNK_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty audio chunk")
    if len(data) > MAX_CHUNK_BYTES:
        raise HTTPException(413, "Audio chunk is too large (32 MB max)")
    try:
        return await run_in_threadpool(pipeline.process_chunk, meeting_id, index, offset_s, data)
    except pipeline.AudioDecodeError as exc:
        logger.warning("meeting=%s chunk=%d undecodable audio: %s", meeting_id, index, exc)
        raise HTTPException(422, "Could not decode the audio chunk") from exc


@app.post("/api/meetings/{meeting_id}/finish")
async def finish_meeting(meeting_id: str) -> Meeting:
    meeting = await run_in_threadpool(_get_or_404, meeting_id)
    if meeting.status == MeetingStatus.DONE:
        return meeting
    store = pipeline.get_store()
    await run_in_threadpool(store.set_status, meeting_id, MeetingStatus.SUMMARIZING)
    if get_settings().use_pubsub:
        await run_in_threadpool(pipeline.publish_finalize, meeting_id)
    else:
        try:
            await pipeline.finalize(meeting_id)
        except Exception:  # finalize already recorded FAILED + error on the meeting
            logger.warning("meeting=%s in-process finalize failed", meeting_id)
    return await run_in_threadpool(_get_or_404, meeting_id)


@app.post("/pubsub/finalize", status_code=204)
async def pubsub_finalize(request: Request) -> Response:
    """Pub/Sub push endpoint. Bad payloads are acked (204) so they are not redelivered forever;
    finalize failures bubble up as 5xx so Pub/Sub retries."""
    try:
        envelope = await request.json()
        meeting_id = json.loads(base64.b64decode(envelope["message"]["data"]))["meeting_id"]
        if not isinstance(meeting_id, str) or not meeting_id:
            raise ValueError("meeting_id must be a non-empty string")
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        logger.warning("Ignoring malformed Pub/Sub message: %s", exc)
        return Response(status_code=204)
    try:
        await pipeline.finalize(meeting_id)
    except KeyError:
        logger.warning("Pub/Sub message for unknown meeting %s; dropping", meeting_id)
    return Response(status_code=204)
