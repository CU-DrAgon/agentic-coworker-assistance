# Meeting Scribe

Record a Google Meet from your browser, get a live Thai transcript with speaker labels, and have a
structured summary (key points, decisions, action items with owners, open questions) posted to Discord
when the meeting ends. Built for the Google AI Builder Cup.

The code, UI and docs are English; the meeting audio and the transcript/summary are Thai.

## Architecture

```
 Browser (Chrome)                       Cloud Run: meeting-scribe (FastAPI)
 ┌──────────────────────┐   chunk/45s   ┌──────────────────────────────────────────┐
 │ Meet tab audio + mic │ ────────────▶ │ POST /api/meetings/{id}/chunks           │
 │ mixed (Web Audio),   │  (sequential) │   ffmpeg → 16 kHz FLAC                   │
 │ MediaRecorder/45 s   │               │   ├─▶ Cloud Storage (audio)              │
 │                      │ ◀──────────── │   └─▶ Gemini on Vertex AI (transcribe,   │
 │ live transcript +    │  poll 5 s     │        diarize) ─▶ Firestore             │
 │ final summary        │               │                                          │
 └──────────────────────┘               │ POST /api/meetings/{id}/finish           │
                                        │   └─▶ Pub/Sub topic meeting-finalize     │
                                        │                  │ push (OIDC-signed)    │
                                        │ POST /pubsub/finalize ◀──────────────────┘
                                        │   ADK agents on Vertex AI:               │
                                        │   [decisions ‖ action items ‖ speaker    │
                                        │    names] → synthesizer → Summary        │
                                        │   ─▶ Firestore  ─▶ Discord webhook       │
                                        └──────────────────────────────────────────┘
```

## Google Cloud services

- **Cloud Run** (`asia-southeast1`): hosts the web UI and API; scales to zero.
- **Vertex AI / Gemini** (`gemini-3.5-flash` for transcription and extraction, `gemini-3.1-pro-preview`
  for the final summary): Thai speech-to-text with speaker diarization and the summarization.
- **Agent Development Kit (ADK)**: multi-agent summarizer. Three extractors run in parallel, then a
  synthesizer merges their output into the final summary.
- **Firestore** (native mode): meetings and per-chunk transcripts.
- **Cloud Storage**: converted FLAC audio per chunk.
- **Pub/Sub**: decouples "Stop" from the slow summarization step. The push subscription has a 600 s
  ack deadline and retries on failure; the endpoint is idempotent.
- **Secret Manager**: holds the Discord webhook URL.
- **IAM**: a runtime service account with least-privilege roles, and a separate pusher service account
  whose OIDC token the app verifies on `/pubsub/finalize`.

## Local development

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), ffmpeg, and Application Default
Credentials for the project (`gcloud auth application-default login`).

```bash
uv sync
uv run pytest -q                      # offline tests, everything external is mocked
uvx ruff check . && uvx ruff format --check .   # lint + format

# Run against real GCP (Firestore, GCS, Vertex AI); /finish summarizes in-process.
cp .env.example .env                  # then set DISCORD_WEBHOOK_URL; without it the Discord post is skipped
uv run uvicorn meeting_scribe.app:app --port 8080
# Fully in memory (still calls Vertex AI and GCS): STORE=memory uv run uvicorn ...
```

Settings (see `src/meeting_scribe/config.py`) come from environment variables or `.env`:
`GCP_PROJECT`, `AUDIO_BUCKET`, `USE_PUBSUB`, `DISCORD_WEBHOOK_URL`, model names, `MEETING_LANGUAGE`.

`scripts/make_synthetic_sample.sh` (macOS) builds a scripted Thai test meeting in `samples/`.
`.env` is git-ignored; never commit the webhook URL.

## Deploy

```bash
gcloud auth login && gcloud config set project meeting-scribe-hackathon
scripts/deploy.sh
```

The script is idempotent. It enables the needed APIs, creates the `meeting-scribe-run` and
`meeting-scribe-pusher` service accounts with their roles, stores `DISCORD_WEBHOOK_URL` from `.env`
in Secret Manager (via stdin, never echoed), deploys the service with `gcloud run deploy --source .`,
and creates or updates the `meeting-finalize-push` subscription. It sets `PUBSUB_VERIFY_AUDIENCE` and
`PUSHER_SA_EMAIL` so only the pusher identity can call `/pubsub/finalize`. It prints the service URL.

Prerequisites it does not create: the `meeting-finalize` topic, the audio bucket and the Firestore
database.

Note: Cloud Run's frontend reserves `/healthz`, so use `/health` for deployed health checks.

## Using it with Google Meet

1. Open the Meeting Scribe URL in **Chrome** (or Edge) on desktop, in a separate window from the Meet tab.
2. Enter a title and, optionally, participant names (they help the model label speakers), then click **Record**.
3. In the browser's sharing dialog choose the **Chrome Tab** option, pick the **Google Meet tab**, and
   tick **Share tab audio**. The app warns if no audio track was shared.
4. Allow microphone access so your own voice is included. **Wear headphones**; otherwise the microphone
   can capture remote voices already present in tab audio, causing echo, duplicate words and worse diarization.
5. **Keep the Meet tab shared** for the whole meeting. Stopping the share ends the capture.
6. Click **Stop** when done. The app uploads the last part, then shows the summary and posts it to Discord.

You can also use **Upload a recording** to process an existing audio file instead.

## Limitations

- Desktop Chrome/Edge only; tab-audio capture is not available on mobile or Firefox/Safari.
- Audio is uploaded in ~45 s segments, one at a time, so the live transcript lags by roughly a segment
  plus transcription time.
- Speaker labels ("Speaker A/B/C") come from the model and can drift across segments; real names are
  only assigned when the transcript clearly identifies them.
- A single upload is capped at 32 MB (Cloud Run request limit) and finalization at one Pub/Sub
  delivery per 600 s; very long meetings may need a larger limit.
- No authentication on the web UI or meeting API: anyone with the URL can create meetings. Add IAP or
  another front door before sharing it publicly.
- Transcription quality on overlapping speech, noisy audio or a lot of English code-switching is limited
  by the model.
- Tests are offline and mock every external service; the end-to-end paths were verified manually.
