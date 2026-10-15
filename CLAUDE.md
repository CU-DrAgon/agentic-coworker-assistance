# Meeting Scribe — project context for people and AI agents

Read this first, then `README.md` (product, architecture, how to use it with Google Meet).

## What this is

A Google AI Builder Cup prototype (theme: Future of Work & Enterprise Productivity).
A host opens the web app next to a Google Meet, clicks Record and shares the Meet tab with
"Share tab audio". The browser mixes tab audio + mic and uploads ~45 s audio segments.
The server transcribes each segment in Thai with speaker labels (Gemini on Vertex AI), and on
Stop an ADK multi-agent summarizer produces decisions, action items and speaker names, saves
them to Firestore and posts a Discord embed with the transcript attached.

- Code, comments, docs and UI are **English**. Meeting audio, transcripts and summaries are **Thai**.
- Submission deadline: **18 Oct 2026**.
- Judging: Gen AI technical merit 40%, problem fit 25%, innovation 25%, UX 10%.
- Repo: https://github.com/CU-DrAgon/agentic-coworker-assistance

## Pipeline in one line

Browser chunk → `POST /api/meetings/{id}/chunks` → ffmpeg → 16 kHz FLAC → Cloud Storage →
Gemini (`gemini-3.5-flash`, transcribe + diarize) → Firestore. On Stop: `/finish` → Pub/Sub →
`/pubsub/finalize` (OIDC-verified) → ADK `[decisions ‖ action items ‖ speaker names]` on
`gemini-3.5-flash` → synthesizer on `gemini-3.1-pro-preview` → Firestore → Discord webhook.

## Code map (`src/meeting_scribe/`)

| File | Role |
| --- | --- |
| `config.py` | `Settings` from env / `.env` (unknown keys ignored) |
| `models.py` | Shared contract: `Utterance`, `ChunkTranscript`, `Meeting`, `Summary`, `ActionItem`. **Do not break it.** |
| `audio.py` | `to_flac` (ffmpeg) |
| `transcribe.py` | `transcribe_chunk`: prompt, label rules, retry. With a participant list every label is a listed name or `Unknown`; otherwise `Speaker A/B/C`. |
| `agents.py` | ADK pipeline `summarize_meeting`, `root_agent`; `unresolved_labels`, `resolve_speaker_names` (never invents names), `build_summary` (applies names to action-item owners, drops recorder instructions) |
| `storage.py` / `store.py` | GCS upload; Firestore `MeetingStore` and `InMemoryMeetingStore` |
| `discord_post.py` | Embed + transcript attachment; never logs the webhook URL |
| `pipeline.py` | `process_chunk`, `finalize` (idempotent, per-process lock) |
| `app.py` | FastAPI routes; health check is `/health` (Cloud Run reserves `/healthz`) |
| `static/index.html` | Recorder UI, live transcript, summary |
| `hub.py` / `static/hub.html` | Salid command center: `/hub` (Thai) + `/api/hub/{me,settings,activity,ask}`. Google ID token as Bearer, verified against `HUB_GOOGLE_CLIENT_ID`, email must be in `HUB_ADMIN_EMAILS`; empty config = 503. No server secret. |
| `scripts/deploy.sh` | Idempotent Cloud Run deploy + IAM + Secret Manager + Pub/Sub push subscription |
| `scripts/resummarize_dry_run.py` | Re-run the summarizer on a stored meeting and print before/after. Read-only, but calls Vertex AI. |

## Salid (`src/salid/`) — second package

Discord bot "สลิด": receipt photos to an expense ledger, Thai executive Q&A. Env vars are all
`SALID_*` (see `.env.example`). `SALID_STORE=sqlite` (default) or `firestore`. The Cloud Run worker pool
is still **not built** and Salid is **not deployed**. Bot-facing replies and prompts are Thai.

| File | Role |
| --- | --- |
| `config.py` | `Settings` (pydantic-settings, `SALID_` prefix, `.env`) |
| `models.py` | `Receipt`, `ReceiptExtraction`, categories |
| `store.py` | `Store` protocol, `SQLiteStore`, `make_store` (`SALID_STORE`) |
| `firestore_store.py` | `FirestoreStore`: doc id = sha256, transactional int id counter in `<collection>_meta`, ISO `effective_date` range query server-side, everything else in Python (no composite index; shop scale only) |
| `archive.py` | Optional GCS archive of original receipts (`SALID_RECEIPT_BUCKET`, prefix `salid-receipts/`), best effort |
| `llm.py` | Vertex Gemini client + retry on transient errors |
| `extract.py` | Receipt extraction, dedupe, ingest, corrections |
| `qa.py` | Expense Q&A as a Google ADK `LlmAgent` + `InMemoryRunner`; `build_tools(LedgerTools)` makes per-request FunctionTools bound to the caller (`forced_submitter_id`), `ask`/`ask_async` return `Answer`; max 4 tool rounds |
| `deep_task.py` | Executive "งานใหญ่:" / `/deep` tasks: caller-scoped CSV export + TASK.md in a temp dir, Gemini CLI headless with isolated HOME and a pinned model, collects `report.md`, always deletes the temp dir |
| `digest.py` | Daily digest text |
| `team_settings.py` | Hub-managed Firestore doc `salid_teams/default` (feature toggles, channel ids, digest time); the bot reads it cached 60 s, fail-open, only with `SALID_STORE=firestore` |
| `bot.py` / `__main__.py` | Discord client / `python -m salid` |
| `cli.py` | `python -m salid.cli extract|ask|deep|digest` for offline testing |
| `scripts/salid_seed_demo.py` | Seed a demo SQLite ledger |

Salid tests live in `tests/salid/` (a scripted ADK `BaseLlm` and a fake Firestore; the `store` fixture runs
every store test on SQLite and Firestore); `tests/salid/test_salid_live.py` is marked `live` and hits real
Vertex, real Firestore (throwaway `salid_receipts_test_<random>` collection, deleted afterwards) and GCS
(`salid-receipts/test-<random>/`).

Salid env vars beyond the basics: `SALID_STORE`, `SALID_FIRESTORE_COLLECTION`, `SALID_RECEIPT_BUCKET`
(empty = no archive), `SALID_GEMINI_CLI_PATH` (empty = deep tasks disabled), `SALID_DEEP_TASK_TIMEOUT`
(600), `SALID_DEEP_TASK_MODEL` (`gemini-3-flash-preview`), `SALID_TEAM_COLLECTION` (`salid_teams`),
`SALID_TEAM_ID` (`default`). Hub (web service) env: `HUB_GOOGLE_CLIENT_ID`, `HUB_ADMIN_EMAILS`,
`SALID_DISCORD_CLIENT_ID`; setup in README "Salid command center".

**Gemini CLI is an external Node binary, not installed by uv** (tested 0.64.0). Never run it in a directory
with untrusted `.gemini/` or `.env`; `deep_task.py` uses an empty temp workspace and an isolated
`HOME`/`GEMINI_CLI_HOME`. Pinning findings: `-m gemini-3.5-flash` is silently rewritten to
`gemini-3.8-flash` on Vertex (hard-coded, not configurable), so we pin `gemini-3-flash-preview` and also
set `experimental.dynamicModelConfiguration` + `modelConfigs.modelIdResolutions` for every alias,
`disableLoopDetection`, `topicUpdateNarration=false` and `tools.core` (file tools only). `DeepResult.pinned_ok`
reports whether only the pinned model ran.

Tests in `tests/` are fully offline (every external service is mocked). Fixture:
`tests/fixtures/thai_meeting.json`.

## Infrastructure that already exists — never recreate

- GCP project `meeting-scribe-hackathon`, region `asia-southeast1`, Vertex Gemini location `global`
- Bucket `gs://meeting-scribe-hackathon-audio`, Firestore `(default)` native DB
- Pub/Sub topic `meeting-finalize` + push subscription `meeting-finalize-push`
- Cloud Run service `meeting-scribe` (**max-instances=1 on purpose**: the finalize lock is per-process)
- Service accounts `meeting-scribe-run` (runtime) and `meeting-scribe-pusher` (Pub/Sub push)
- Secret Manager secret `discord-webhook-url`
- Deployed URL: `gcloud run services describe meeting-scribe --region asia-southeast1 --format='value(status.url)'`
  (keep it private: there is no auth on the app)

## Credentials and safety rules

- `.env` holds `DISCORD_WEBHOOK_URL`, a GitHub token (`GH_TOKEN`) and GCP settings. **Never** print,
  log, echo, paste into chat, commit, or put any of these values on a command line. Refer to them by
  key name only.
- Before any commit: `git status` and `git diff --cached`. `.env`, `samples/` and `*.json` key files
  must never be staged.
- Do not create service-account key files. Use `gcloud auth login` + `gcloud auth application-default login`.
- `gcloud run deploy --source .` uploads the directory. There is no `.gcloudignore`, so `.gitignore`
  applies and excludes `.env`. Confirm before deploying.
- `scripts/deploy.sh` pushes the webhook from `.env` into Secret Manager as a **new version**. Run it
  only intentionally.
- **Never set `USE_PUBSUB=true` locally**: it routes finalize to the production push endpoint.
  Local runs use `STORE=memory` and the default `USE_PUBSUB=false`.
- Re-transcribing chunks costs Vertex quota. 3–4 chunks is fine; ask before a whole meeting.
- Ask a human before anything destructive, costly or production-affecting: deploys, new secret
  versions, deleting data, posting to Discord, changing Firestore records.

## Local development

```bash
uv sync
uv run pytest -q                                  # offline, all mocked (live tests are skipped; `-m live` runs them)
uvx ruff check . && uvx ruff format --check .     # lint + format (line length 100)
STORE=memory uv run uvicorn meeting_scribe.app:app --port 8080
curl -s localhost:8080/health
```

Requires Python 3.12 (uv installs it), ffmpeg, and ADC for anything that touches GCP.

## Known limitations — do not "fix" without asking

- No auth on the web app or API; keep the URL private.
- About 0.1 s of audio is lost at each 45 s segment boundary.
- No retention policy on audio or Firestore (deleting data needs approval).
- Old Cloud Run logs may contain the webhook URL; rotate it after the hackathon.
- Desktop Chrome/Edge only (tab-audio capture).

## Reference data

Firestore meeting `e6379be86dfd` ("Meet1", 25 min, 34 chunks, 309 utterances, participants
Peat / Toey / Jems) is the first real recording. It showed the two bugs fixed on 13 Oct 2026:
speaker labels drifting between real names and `Speaker B/C/D`, and a spoken instruction to the
recorder ("หยุดบันทึกและสรุปข้อมูลการประชุม") extracted as an action item. Use
`uv run python scripts/resummarize_dry_run.py e6379be86dfd` to check regressions without writing
anything.

## How to work here

Small surgical changes. Write a failing offline test first, keep the suite green, match the existing
style. State what you verified and what you could not. Work on a feature branch and open a PR;
redeploy with `scripts/deploy.sh` only after a human approves, then smoke `/health`.
