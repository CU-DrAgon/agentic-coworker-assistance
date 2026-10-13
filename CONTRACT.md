# Build contract (internal, for the build agents)

Meeting Scribe — Google AI Builder Cup prototype. Code, comments, docs, UI in **English**; meeting audio content is **Thai**.

## Product
Host opens our web app next to a Google Meet, clicks Record, shares the Meet tab with "Share tab audio". Browser mixes tab audio + host mic and uploads a standalone audio chunk every ~45 s. Server transcribes each chunk with Gemini on Vertex AI (Thai, speaker diarization, ~3 speakers); live transcript appears on the page. On Stop, a Google ADK multi-agent pipeline summarizes (key points, decisions, action items with owner/due, open questions, speaker names) and posts to Discord via webhook with the full transcript attached. Firestore for data, Cloud Storage for audio, finalize via Pub/Sub push (in-process when USE_PUBSUB=false). Cloud Run, asia-southeast1.

## Already exists — read, don't recreate
- `src/meeting_scribe/config.py` — `get_settings()`.
- `src/meeting_scribe/models.py` — Utterance, ChunkTranscript, MeetingStatus, ActionItem, Summary, Meeting. Don't change; backward-compatible additions only, and report them.
- GCP project `meeting-scribe-hackathon`, ADC logged in (`source ~/google-cloud-sdk/path.bash.inc` for gcloud). Bucket `gs://meeting-scribe-hackathon-audio`, Firestore (default) native DB, Pub/Sub topic `meeting-finalize`. Gemini works at Vertex location `global` for `gemini-3.5-flash` and `gemini-3.1-pro-preview` (`genai.Client(vertexai=True, project=..., location="global")`).
- `.env` has `DISCORD_WEBHOOK_URL`. Never print, log, or commit its value.

## Module contract
1. `audio.py` (transcribe): `to_flac(data: bytes) -> bytes` — ffmpeg via stdin/stdout, 16 kHz mono FLAC.
2. `transcribe.py` (transcribe): `transcribe_chunk(gcs_uri, index, offset_s, prior: list[Utterance], participants: list[str] | None = None, mime_type="audio/flac") -> ChunkTranscript`. Structured output. Chunk-relative times + offset_s. Consistent labels across chunks: pass last ~12 prior utterances, reuse "Speaker A/B/C" (participant name only when clearly identifiable). No translation. Drop empty/noise. Retry once on transient errors.
3. `storage.py` (infra): `upload_chunk(meeting_id, index, data: bytes, content_type="audio/flac") -> str` → `gs://<bucket>/meetings/{meeting_id}/chunk-{index:04d}.flac`.
4. `store.py` (infra): `MeetingStore` — `create(title, participants) -> Meeting`, `get(id) -> Meeting | None`, `add_chunk(id, chunk)` (subcollection `chunks/{index:04d}`), `set_status(id, status, error=None)`, `set_summary(id, summary)`, `list_recent(limit=20)` (no chunks). Lazy Firestore client. Plus `InMemoryMeetingStore` with the same interface.
5. `discord_post.py` (infra): `post_summary(meeting, webhook_url) -> None` — one embed (title, overview, Key points, Decisions, Action items "• **Owner** — task (due)", Open questions, footer duration/speaker count), Discord limits (field 1024, description 4096, total 6000, truncate gracefully). Apply summary.speaker_names to transcript. Attach `transcript-{id}.txt` ("[mm:ss] Name: text") via multipart `payload_json` + `files[0]`. httpx; raise on non-2xx; one retry on 429 using retry_after.
6. `agents.py` (agents): `async summarize_meeting(meeting) -> Summary`. ADK on Vertex (env GOOGLE_GENAI_USE_VERTEXAI=TRUE, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION=global). SequentialAgent[ ParallelAgent[decisions extractor, action-item extractor, speaker-name resolver] (extract_model, output_key) → synthesizer (summary_model) → Summary ]. InMemoryRunner. Grounded only in transcript. Content in meeting language. Module-level `root_agent`.
7. `pipeline.py`, `app.py`, `static/index.html` (web): FastAPI.
   - `GET /`, `GET /healthz`
   - `POST /api/meetings` {title, participants[]} → Meeting
   - `POST /api/meetings/{id}/chunks` multipart file, index, offset_s → to_flac → upload_chunk → transcribe_chunk (prior = chunks with lower index) → add_chunk → ChunkTranscript. `run_in_threadpool`. Synchronous (no fire-and-forget).
   - `GET /api/meetings/{id}`, `GET /api/meetings`
   - `POST /api/meetings/{id}/finish` → SUMMARIZING; publish {meeting_id} if use_pubsub else await `pipeline.finalize`.
   - `POST /pubsub/finalize` → decode base64 JSON → finalize → 204; idempotent.
   - `pipeline.finalize(id)`: wait until chunk count stable, summarize, set_summary, post_summary (if webhook), DONE; on error FAILED + re-raise.
   - Env `STORE=memory` → InMemoryMeetingStore, else Firestore.
   - index.html: self-contained, clean, accessible. Title + participants form. Record: getDisplayMedia({video:true,audio:true}) (warn if no audio track: "Tick 'Share tab audio'") + getUserMedia mic, mixed via Web Audio. New MediaRecorder per ~45 s segment so each blob is standalone; upload with index/offset_s; concurrent uploads, render ordered; retry 3x. Poll transcript every 5 s, speaker-colored, timestamps. "Upload a recording" file option (chunk 0 + finish). Stop: flush last segment, await uploads, /finish, poll until done, show summary. Timer + upload status.
8. Offline pytest tests per module in `tests/test_<module>.py` (mock Gemini/Firestore/GCS/Discord). `uv run pytest -q` passes.

## Rules
Minimal readable code, type hints, `logging` (no secrets/full transcripts at INFO). Stay in your files. Code against this contract if other modules don't exist yet. **Don't read library source in `.venv` except small, targeted greps** — it blows the context window.
