# Meeting Scribe

Record a Google Meet from your browser, get a live Thai transcript with speaker labels, and have a
structured summary (key points, decisions, action items with owners, open questions) posted to Discord
when the meeting ends. Built for the Google AI Builder Cup.

The code, UI and docs are English; the meeting audio and the transcript/summary are Thai.

## Architecture

Diagrams (open in a browser): [architecture map](docs/diagrams/architecture.html) (source:
`docs/diagrams/architecture.archify.json`) and [meeting sequence](docs/diagrams/meeting-sequence.html).

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

## สลิด / Salid — receipts & expense Q&A bot

A second package, `src/salid/`, is a Discord bot. Staff post receipt photos or PDFs in a receipt
channel; Gemini (Vertex AI) extracts vendor, date, totals and a category into a ledger, flags
low-confidence or duplicate receipts, and replies in Thai. Executives ask questions in Thai
("how much did we spend on ingredients this week?") and get answers from ledger tool calls, plus an
optional daily digest. Non-executives only see their own receipts.

**Multi-tenant.** Any Discord server (guild) can add Salid with the public invite link and gets its own
isolated ledger and settings with no technical setup. The tenant key is the guild id: every receipt,
dedupe check, Q&A tool, digest, deep-task export and hub view is scoped to one guild.

Customer flow: (1) open the invite link (the hub's "เพิ่ม Salid เข้า Discord" button) and add Salid to the
server; Salid posts a short Thai welcome. (2) Post receipt photos in a channel whose name contains
`ใบเสร็จ` or `receipt`, or @mention Salid with an image/PDF in any channel: it works immediately.
(3) Optional: a server owner / Administrator / Manage Server member types `/ตั้งค่า` (or `/setup`) and
opens the private one-time link (valid 15 minutes) to manage the server in the hub after signing in with
Google. Executives are the owner, members with Administrator or Manage Server, members with a role in the
server's executive role names (default `ผู้บริหาร`, `Executive`) and `SALID_EXEC_USER_IDS`.

- **Channel modes** (`/โหมด` or `/mode`, or the hub's "โหมดของแต่ละห้อง"): per channel, `🔇 off` (ignore the
  channel), `👂 listen` (receipts get an emoji reaction only; an @mention gets 👂), `🙋 call` (default:
  answers when @mentioned or called by name, `สลิด` and misspellings but not `สลิป`; for 3 minutes
  afterwards a cheap gate model, `SALID_GATE_MODEL`, decides whether each message is meant for Salid, then a
  short sign-off), `🎧 join` (call, plus Salid may chime in on questions after the channel is quiet for 30 s,
  at most 3 times per hour per channel) and `💬 chat` (answers every message; the `คุยกับน้องสลิด` channel
  and ask channels). Only server managers or executives may change a mode. `@Salid เงียบก่อน` makes the
  channel `listen` for an hour; `@Salid คุยได้แล้ว` ends it. The old "any channel named salid answers
  everything" rule is gone.
- **Thai and English commands**: every typed command (an @mention followed by the words, matched only at the
  start of the message, case-insensitive) has both forms: `เข้าประชุม` / `join`, `join the meeting`, `come to the
  meeting`; `ออกประชุม` / `leave`, `leave the meeting`, `end meeting`, `stop meeting`; `เงียบก่อน` / `be quiet`,
  `quiet`, `shh`, `shush`, `mute`, `hush`, `snooze`; `คุยได้แล้ว` / `you can talk`, `talk again`, `unmute`,
  `unsnooze`, `wake up`, `come back`. Slash commands have Thai names with English aliases (`/โหมด` `/mode`,
  `/ภาษา` `/language`, `/เข้าประชุม` `/join`, `/ออกประชุม` `/leave`, `/ตั้งค่า` `/setup`). The name is recognised in
  Latin letters too ("hey salid, what did we spend?").
- **Reply language** (`/ภาษา` or `/language`, `@Salid พูดภาษาอังกฤษ` / `speak English` / `English please`,
  `พูดไทย` / `speak Thai`, `ภาษาอัตโนมัติ` / `auto language`, or the hub's per-channel language select and the
  server default): `th` (default, the old behaviour), `en`, or `auto` (answer in the language of the message:
  any Thai character means Thai, otherwise English; with no text, e.g. a slash command, the user's Discord
  locale). Resolution: channel override, else the guild's `default_language`; DMs use `auto`. It applies to
  everything Salid says: fixed replies (catalog in `src/salid/i18n.py`), Q&A and voice answers (system prompt),
  meeting summaries, the daily digest, receipt replies and deep-task reports. Only managers/executives may
  change it; others get a short refusal. A voice meeting uses the language of the voice channel's text chat as
  it was when the meeting started.
- **Q&A runs on Google ADK**: an `LlmAgent` (Gemini on Vertex) with four ledger `FunctionTool`s that
  are built per request around the asker's scope, so the model cannot widen it. At most 4 tool rounds.
- **Storage**: `SALID_STORE=sqlite` (default, local dev) or `firestore` (collection
  `SALID_FIRESTORE_COLLECTION`, default `salid_receipts`, doc id `{guild_id}_{sha256}`). Each query filters
  `guild_id ==` server-side (automatic single-field index only, no composite index); the date range and
  the rest are done in Python over that guild's receipts, which is fine at shop scale. Existing SQLite
  ledgers are migrated on open (old rows become guild 0). Optionally archive the original receipt files to
  Cloud Storage with `SALID_RECEIPT_BUCKET` (object prefix `salid-receipts/{guild_id}/`, the `gs://` URI
  is saved on the record).
- **Per-guild settings**: Firestore `salid_guilds/{guild_id}` (`SALID_GUILD_COLLECTION`): feature flags,
  extra receipt/ask channels, digest channel and time, executive role names, `admin_emails`, and what the
  bot syncs (name, icon, text channels, `installed_at`, `active`). The bot caches it ~60 s per guild;
  a missing document means defaults (every feature on). With `SALID_STORE=sqlite` an in-memory copy is
  used so local dev needs no Firestore. Env channel lists and `SALID_EXEC_ROLE_NAMES` still apply as
  defaults. `/ตั้งค่า` writes a one-time doc in `salid_link_codes/{code}`.
- **Deep tasks (executives only)**: send `งานใหญ่: <task>` or `/deep <task>`. Salid acknowledges at
  once, exports the receipts the caller may see into a throwaway workspace and runs **Gemini CLI** headless
  there (isolated HOME, minimal environment, file-only tools, timeout `SALID_DEEP_TASK_TIMEOUT`), then posts
  the Thai report (attached as `report.md` when long) with seconds, tokens and the models actually used.
  Gemini CLI is an external Node binary that `uv` does not install: set `SALID_GEMINI_CLI_PATH`, otherwise
  deep tasks are disabled. Model pinning caveat: with Vertex, the CLI rewrites `gemini-3.5-flash` to
  `gemini-3.8-flash`, so the default is `SALID_DEEP_TASK_MODEL=gemini-3-flash-preview`.

The bot is prepared for a Cloud Run worker pool (see below) but not deployed.

**Voice meetings (`/เข้าประชุม`, alias `/join`).** A member in a voice channel types `/เข้าประชุม` (or
`@Salid เข้าประชุม`); Salid joins, announces itself in the channel's text chat, and transcribes every speaker by
Discord name (per-user audio, so no diarization). Calling "สลิด ..." out loud gets a text answer in that chat, using
the last ~5 minutes of transcript as context; the same transcript is context for text messages to Salid in that
chat ("เมื่อกี้ตกลงอะไรกัน"). `/ออกประชุม` (`/leave`), everyone leaving, or 3 hours ends it with a Thai summary
(decisions, action items with owners, open questions). Audio and transcript are kept in memory only, never stored.
Needs the **Connect** permission, the voice-states intent (non-privileged), `libopus` + `ffmpeg` (in
`Dockerfile.salid`) and `discord-ext-voice-recv` + `davey` (Discord's DAVE end-to-end encryption; see
`src/salid/meeting.py`). Toggle: hub feature `voice_meetings`.

**Spoken replies.** When called by name in a voice meeting, Salid also says a 1-2 sentence version of
the answer out loud (Gemini TTS `SALID_TTS_MODEL`, default `gemini-2.5-flash-tts`, voice `SALID_TTS_VOICE`,
default `Puck`); the full answer still goes to the chat. While Salid is speaking and for 1.5 s after,
wake words are ignored so its own voice coming back through someone's speakers cannot re-trigger it.
Turn it off with `SALID_VOICE_REPLY=false`.

**Live voice mode (`SALID_VOICE_MODE=live`, the default).** Instead of the wake-word pipeline, a Gemini Live
native-audio session (`SALID_LIVE_MODEL`, `SALID_LIVE_LOCATION=us-central1`, voice `SALID_LIVE_VOICE`) hears the
whole voice channel and answers about 1.5 s after someone stops talking, like a voice assistant. Everyone's audio is
mixed into one 16 kHz stream (`src/salid/live_voice.py`); with proactive audio on, Salid stays silent unless it is
addressed by name or clearly asked something, and anyone can talk over it (barge-in stops it at once). The session is
opened when Salid joins, resumes after the server's periodic disconnects, and what Salid says is also posted to the
channel's text chat as `🔊 ...`. Money questions use the same ledger tools as the text Q&A, but because the mixed
stream does not say who is speaking they only work when **everyone in the voice channel is an executive**; otherwise
Salid says to ask in the text chat. The per-speaker transcription keeps running for the transcript and summary but no
longer answers wake words in this mode. If the Live session cannot connect (4 tries), that meeting falls back to the
classic wake-word mode. `SALID_VOICE_MODE=classic` restores the old behaviour; `SALID_VOICE_REPLY=false` turns off
speech in both modes. Audio is streamed to Vertex AI for the duration of the meeting (never stored by Salid), and a Live
session is billed by audio time while people are talking.

Discord setup (once, by the operator):

1. Create an application and bot at the Discord Developer Portal (name it **Salid**); copy the bot token.
2. Enable the **Message Content** and **Server Members** privileged intents. The public invite link
   (`salid_invite_url`: scopes `bot` + `applications.commands`, permissions 1166416) is built from
   `SALID_DISCORD_CLIENT_ID` and shown on the home page and in the hub.
3. Put `SALID_DISCORD_TOKEN` in `.env`. The slash commands are registered globally at startup (Discord can
   take up to an hour to show new global commands).
4. Optional local defaults, applied to every guild: `SALID_RECEIPT_CHANNEL_IDS`, `SALID_ASK_CHANNEL_IDS`,
   `SALID_DIGEST_CHANNEL_ID`, `SALID_EXEC_USER_IDS`, `SALID_EXEC_ROLE_NAMES`. `SALID_HUB_URL` is the web
   service the `/ตั้งค่า` link points to (defaults to the Cloud Run URL).

Other keys: `SALID_DIGEST_TIME`, `SALID_STORE`, `SALID_DB_PATH`, `SALID_FIRESTORE_COLLECTION`,
`SALID_RECEIPT_BUCKET`, `SALID_GEMINI_CLI_PATH`, `SALID_DEEP_TASK_TIMEOUT`, `SALID_DEEP_TASK_MODEL`,
`SALID_GCP_PROJECT`, `SALID_GEMINI_LOCATION`, `SALID_GEMINI_MODEL`, `SALID_TZ` (see `.env.example`). All Salid variables carry the `SALID_` prefix so they
never collide with Meeting Scribe's.

```bash
uv run python -m salid                                   # run the bot (needs ADC for Vertex AI)
uv run python scripts/salid_seed_demo.py ./data/demo.db  # seed a demo ledger
SALID_DB_PATH=./data/demo.db uv run python -m salid.cli ask "สรุปค่าใช้จ่ายสัปดาห์นี้" --as-exec
uv run python -m salid.cli extract receipt.png           # run extraction on a local file
SALID_GEMINI_CLI_PATH=/path/to/gemini SALID_DB_PATH=./data/demo.db \
  uv run python -m salid.cli deep "วิเคราะห์ค่าใช้จ่ายเดือนนี้ แล้วเขียนรายงาน" --as-exec
uv run pytest -m live -q                                 # one real Vertex call (skipped by default)
```

## Deploying Salid

Salid runs as a Cloud Run **worker pool** (no HTTP port; it holds the Discord gateway connection).

```bash
export SALID_DISCORD_TOKEN=...            # from your shell, never from a command line or git
scripts/deploy_salid.sh                   # dry run (default): prints every command, changes nothing
scripts/deploy_salid.sh --apply           # really deploy
scripts/deploy_salid.sh --configure-hub   # optional, separate: set hub env vars on meeting-scribe
```

The script is idempotent: it enables missing APIs, creates the `salid-bot` service account with least
privilege (`datastore.user`, `aiplatform.user`, `logging.logWriter`, `secretmanager.secretAccessor` on
the one secret, `storage.objectAdmin` on `SALID_RECEIPT_BUCKET` only if set), stores the token in the
`salid-discord-token` secret via stdin (new version only if none exists, or with `--rotate-token`),
builds `Dockerfile.salid` with Cloud Build (`cloudbuild.salid.yaml`) and deploys the pool with
`SALID_STORE=firestore`. Optional defaults (`SALID_RECEIPT_CHANNEL_IDS`, `SALID_GUILD_COLLECTION`,
`SALID_HUB_URL`, ...) are passed through from your environment. It never touches the existing Meeting Scribe resources, except the optional
`--configure-hub` step, which only runs `gcloud run services update --update-env-vars` (needs
`HUB_GOOGLE_CLIENT_ID`, `SALID_DISCORD_CLIENT_ID` and optionally `HUB_ADMIN_EMAILS` in your environment).

**Cost.** One always-on instance (`--instances 1`: two would answer every Discord message twice),
1 vCPU / 2 GiB. 2 GiB because the Gemini CLI (Node) runs beside the Python bot during deep tasks;
1 vCPU because the bot is I/O bound. It bills around the clock (roughly tens of USD per month; check
the Cloud Run pricing page for asia-southeast1).

**Stop it** (keeps config and secret, stops billing for instances):
`gcloud run worker-pools update salid-bot --region asia-southeast1 --instances 0`; restart with
`--instances 1`. Remove it completely with `gcloud run worker-pools delete salid-bot --region asia-southeast1`.

## Salid command center (`/hub`)

A Thai web page in the same app and Cloud Run service, multi-guild. Anyone signs in with Google (any
verified account); a guild switcher at the top lists the servers that account administers. Per server:
the latest receipts, Salid expense Q&A as an executive, feature switches (receipts, Q&A, daily digest,
deep tasks), receipt/ask/digest channels picked from the channel list the bot synced, the digest time and
the executive role names, and per-channel modes plus reply languages (with a server default for each). A user with no server sees a big "เพิ่ม Salid เข้า Discord" button and the two
steps: 1. add Salid to the server, 2. type `/ตั้งค่า` in Discord and open the link.

How an account becomes a guild admin, with no OAuth secret and no bot token in the web service:
`/ตั้งค่า` (owner / Administrator / Manage Server only) creates `salid_link_codes/{random 32 bytes}`
(`guild_id`, Discord user, `expires_at` = now + 15 min, `used=false`) and replies *ephemerally* with
`{SALID_HUB_URL}/hub?link=<code>`. After Google sign-in the page calls `POST /api/hub/link`; a Firestore
transaction checks the code is unused and unexpired, marks it used and adds the verified email to the
guild's `admin_emails`. The page removes `?link=` from the address bar immediately.

API (Bearer Google ID token, verified against `HUB_GOOGLE_CLIENT_ID`; 503 when that is empty):
`GET /api/hub/me`, `GET /api/hub/guilds`, `POST /api/hub/link`, and per guild
`GET|PUT /api/hub/guilds/{id}/settings`, `GET .../activity`, `POST .../ask`, all 403 unless the email is in
that guild's `admin_emails` (or is a super admin). `HUB_ADMIN_EMAILS` is now an optional comma-separated
list of super admins who see every guild and the meeting items in activity (meetings are not per guild
yet, so guild admins never see them). Saved channel ids must exist in the guild's synced channel list.

One manual setup step: in Google Cloud Console (APIs & Services, Credentials) create an
**OAuth 2.0 Client ID** of type *Web application*. Authorized JavaScript origins: the Cloud Run URL
and `http://localhost:8080`. No redirect URI or client secret is needed. Then set
`HUB_GOOGLE_CLIENT_ID` (the public client id). Optional: `SALID_DISCORD_CLIENT_ID` for the
Add-to-Discord button, `HUB_ADMIN_EMAILS` for super admins. `SALID_GUILD_COLLECTION` (default
`salid_guilds`) must match between the web service and the bot. The web service reads the same
`SALID_STORE` settings as the bot, so set `SALID_STORE=firestore` there to see the real ledger.
Deep tasks are Discord-only.

## Limitations

- Desktop Chrome/Edge only; tab-audio capture is not available on mobile or Firefox/Safari.
- Audio is uploaded in ~45 s segments, one at a time, so the live transcript lags by roughly a segment
  plus transcription time.
- Speaker labels come from the model and can drift across segments. When the host lists participants,
  every label is one of those names or "Unknown"; otherwise generic "Speaker A/B/C" labels are used and
  the summarizer maps each one to a name found in the transcript, or to "Unknown".
- A single upload is capped at 32 MB (Cloud Run request limit) and finalization at one Pub/Sub
  delivery per 600 s; very long meetings may need a larger limit.
- No authentication on the web UI or meeting API: anyone with the URL can create meetings. Add IAP or
  another front door before sharing it publicly.
- Transcription quality on overlapping speech, noisy audio or a lot of English code-switching is limited
  by the model.
- Tests are offline and mock every external service; the end-to-end paths were verified manually.

## License

Copyright (c) 2026 The Frontier Company Limited.

This project is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may read, run and modify it for noncommercial purposes. Any commercial use requires a separate license from The Frontier Company Limited.
