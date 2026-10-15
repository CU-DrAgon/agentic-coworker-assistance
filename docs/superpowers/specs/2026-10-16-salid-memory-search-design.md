# Salid memory search: faster, cheaper, meaning-based (design)

Date: 2026-10-16. Status: approved in conversation, awaiting written-spec review.
Branch: `feature/salid-memory-search`. Target: merged before the 2026-10-18 submission.

## Goal

Salid finds the right team memory faster and cheaper, matches by meaning across Thai and English,
and logs what each answer costs. It also strengthens the "Gen AI technical merit" story (40% of
judging) with hybrid retrieval.

## Scope

In scope (chosen: "small, safe, before 18 Oct", approach A):

1. Hybrid (keyword + embedding) search over day summaries, with an in-process per-guild index.
2. Firestore read-path fixes for `chat_search`, `chat_days`, `chat_messages`.
3. Voice wake-word answers get memory tools.
4. In-process short-term context buffer instead of Discord history calls on every reply.
5. Cache-friendly system prompt order.
6. Per-answer cost and latency metrics.

Out of scope (later): durable "remember/forget" facts memory, per-user memory, deleting memory when
Salid leaves a server, privacy page, Firestore vector indexes, embedding raw messages, connection
reuse for the ADK model client (measure first with item 6).

## Constraints

- Additive storage only: no data migration, no new Firestore index, no infrastructure change.
- Access rules (`memory_access.py`), retention purge and deletion behave exactly as today.
- Production bot: `SALID_STORE=firestore`, one fixed Cloud Run worker instance (1 vCPU, 2 GiB,
  shared with Gemini CLI deep tasks). The hub reads memory import progress only; it never writes
  summaries.
- Offline tests for everything; live calls (Vertex, Firestore) only with human approval.
- No deploy and no Discord posts without human approval.

## Success criteria

1. `chat_search` on Firestore with 90-day retention issues at most 10 queries (today: about 180
   sequential queries, each returning every message of the guild for one day). Checked by a
   query-counting test on the fake Firestore.
2. An English question finds a Thai-only discussion, and the reverse: offline with fixed vectors,
   once live.
3. Every answer logs a latency breakdown and token counts including cached tokens.
4. Voice wake-word answers can use team memory under the same access rules.
5. Full suite green; existing tool output keeps its shape (new fields only).

## Current state (findings that motivate this)

- `qa.MemoryTools.chat_search` loads up to 5,000 raw messages per call (`MAX_ROWS_SEARCH`).
  `FirestoreMemoryStore.messages()` and `.summaries()` run one query per day in sequence; each
  per-day message query returns every message of the guild that day. Join mode runs `chat_search`
  for up to 5 candidates per gate (`bot._interjection_hints`).
- Retrieval is case-insensitive substring only; Thai has no word breaks, so recall depends on the
  model guessing exact substrings.
- `meeting.MeetingManager.answer` calls `ask` without `memory=`.
- `_recent_chat`, `_previous_text` and `_gate_lines` each read Discord channel history over REST.
- `system_prompt` puts the current time near the start, so the long stable part never forms an
  identical prefix for Gemini's implicit cache; it is re-sent on every model call (up to 5 per
  answer).
- Only "answered in X s (N rounds)" is logged.

## Design

### 1. Hybrid memory search

**`salid/embed.py` (new).** `Embedder` protocol: `embed(texts: list[str], task: str) ->
list[array]` where `task` is `"document"` or `"query"` (mapped to `RETRIEVAL_DOCUMENT` /
`RETRIEVAL_QUERY`). Vectors are L2-normalised `array("f")`.

- `VertexEmbedder(client, model, dim)`: `client.models.embed_content(model, contents,
  EmbedContentConfig(task_type, output_dimensionality=dim))`, split into batches of the API limit
  (confirmed by the probe), with the same transient-error retry as `llm.generate` (shared helper in
  `llm.py`).
- `NullEmbedder`: returns no vectors; callers treat that as "keyword only".
- `make_embedder(settings, client)`: `NullEmbedder` when `SALID_EMBED_MODEL` is empty.
- New settings: `embed_model` (default `gemini-embedding-001`, subject to the probe), `embed_dim`
  (256), `embed_location` (empty = `gemini_location`; a separate genai client is created only when
  it differs).

**Summary items.** `MemoryTools._summary_lines` moves to `memory_summary.summary_items(summary)
-> list[str]` (overview, topics, decisions, action items, numbers, open questions), the single
definition of a searchable item for keyword matching and embedding.

**Vectors on `DaySummary`** (all optional, defaults keep old records valid):

- `vec_model: str = ""` (model and dim, e.g. `gemini-embedding-001@256`),
- `vec_hashes: list[str] = []` (12-hex-char SHA-1 of each item text, same order as the items),
- `vecs: bytes = b""` (packed little-endian float32, `len(items) * dim * 4` bytes).

A summary has usable vectors when `vec_model` matches the configured model and the hash count
matches its items. SQLite: guarded `ALTER TABLE salid_day_summaries ADD COLUMN` for the three
fields on open (hashes as JSON text, vecs as BLOB). Firestore: no change (bytes field). Vectors live
in the summary record, so the retention purge, dirty regeneration and `delete_guild` already cover
them.

**Writing vectors.** `memory.vectorize(summary_dict, previous: DaySummary | None, embedder, dim)
-> (vec_model, hashes, vecs)` reuses vectors for items whose hash exists in `previous` and embeds
only new items in one batched call. `MemoryScheduler._update` calls it after the summary call.
`MemoryImporter`'s past-day summaries are written without vectors and picked up by the backfill
(no embedding code in the importer). Any embedding error is logged (counts only) and the summary is
saved without vectors: summarising never depends on embedding.

**Backfill.** Inside `MemoryScheduler.tick`, after due updates: one guild per tick
(round-robin over the tick's guilds), up to `MAX_VECTORIZE_PER_TICK` (10) of its summaries without
usable vectors, newest first, found with `store.all_summaries(guild_id)` (served from the index when
the store is wrapped, so no extra query). A guild with nothing missing is skipped until one of its
summaries is written again. Vectors are written with a new store method
`put_summary_vectors(guild_id, channel_id, local_date, vec_model, hashes, vecs)` that updates only
those fields and is a no-op when the summary no longer exists (purged meanwhile). Runs in the same
worker thread as the scheduler, so no race with summary updates. The scheduler depends only on the
`MemoryStore` protocol, never on the wrapper.

**`salid/memory_index.py` (new): `IndexedMemoryStore`.** Wraps any `MemoryStore`, delegates every
protocol method, and keeps a per-guild index of summaries:

- Entry per channel-day: `channel_id`, `local_date`, `message_count`, `final`, the item texts, the
  vectors (one `array("f")` per item, or none), and the `DaySummary` itself for `chat_days`.
- Miss: one new store call `all_summaries(guild_id)` (Firestore: single `guild_id ==` query;
  SQLite: one SELECT). The wrapper's own `all_summaries` is served from the index.
- Coherence: `put_summary` and `put_summary_vectors` update the entry; `delete_messages` marks
  affected entries dirty (as the store does); `purge` and `delete_guild` drop the affected guilds.
  The bot wraps its store once, so the scheduler, importer, deletes and purge all go through it.
  Correct only with one writer process; stated in the module docstring and CLAUDE.md.
- Bounds: LRU over guilds by approximate bytes, `INDEX_MAX_BYTES` 64 MB; size logged on load and
  eviction.
- Read API: `summary_entries(guild_id, allowed_channels, start, end) -> list[IndexEntry]`.

**Hybrid `chat_search`.** New optional tool argument `query: str` (the question in plain words, any
language). Steps:

1. Start the query embedding (`query` or the joined keywords; LRU cache of 512 query vectors) while
   the guild's entries load.
2. Keyword score per (day, channel) group from summary items, as today (`SUMMARY_WEIGHT`).
3. Similarity score per group = best cosine (`math.sumprod` of normalised vectors) over its items,
   kept only above `MIN_SIMILARITY` (provisional 0.55; recalibrated by the probe).
4. Fuse the two rankings with reciprocal rank fusion (`k = 60`); keep the top `limit`.
5. Raw messages only for the top `RAW_GROUPS` (5) groups (per channel-day query) and for the last
   `RAW_RECENT_DAYS` (3) days (one query per day, in parallel): keyword matches, up to 3 per group,
   with links.
6. Output keeps the current keys and adds per hit `match: "keyword" | "meaning" | "both"`, plus a
   `note` when older raw messages were not scanned ("for an exact code or number older than 3
   days use chat_messages with a keyword and date range").

`MemoryTools.chat_search(..., semantic: bool = True)`: `semantic` is not in the tool schema; join
mode hints call it with `semantic=False` (no embedding calls). Embedding failure means keyword-only
for that call, never an error.

Both backends run the same algorithm (simpler to test and reason about).

### 2. Firestore read path

- `chat_days`: summaries from `summary_entries`; the "newer than summary" tail for today and
  yesterday reads each day once via `messages()` and groups by channel in Python (today: 2 x (1 +
  channels) queries, each returning the whole guild day).
- `FirestoreMemoryStore.messages()`: per-day queries in parallel (thread pool, at most 8), with a
  `channel_id ==` filter when one channel is requested.
- `FirestoreMemoryStore.messages_after()`: adds `channel_id ==`. Equality-only filters are served
  from automatic single-field indexes like today's two-filter query; confirmed by the live
  Firestore test before merge.

### 3. Voice answers get memory

`bot._memory_tools(message, guild, member)` delegates to a new
`_memory_tools_for(guild, member, answering_channel_id)`. `MeetingManager.answer` passes the voice
channel as the answering channel, so everyone who can read the voice channel's chat must be able to
read each source channel. Voice channel `off` or memory disabled means no memory tools, as today.

### 4. Short-term context buffer

`salid/recent.py` (new): `RecentMessages` keeps per channel a bounded deque (40) of `Snapshot`
(id, author id and display name, is Salid, is bot, created_at, content, attachment filenames,
role-mention ids, referenced message id).

- Fed by `on_message` before any filtering (Salid's own replies included), and by the existing raw
  edit / delete / bulk-delete handlers.
- A channel is primed only after one REST history read; until then readers fall back to REST and
  prime it with the result. `on_ready` and `on_resumed` reset every channel to unprimed.
- `_recent_chat`, `_previous_text` and `_gate_lines` read the buffer first and format with the
  same code, so model input is unchanged.
- Bounds: channels idle more than `HISTORY_SECONDS` (3 h) evicted on the tick; at most 1,000
  channels (about 25 MB worst case).
- Last in the task order; can be dropped without affecting the rest.

### 5. Cache-friendly system prompt

`system_prompt` order: persona, general rules, capabilities, memory guidance (all stable per
language) first; then the volatile tail: date and time, non-executive scope note, interjection
note, language line. Wording unchanged except `MEMORY_PROMPT`: pass the question as `query`; 2-5
keywords are enough because search also matches meaning.

### 6. Observability

- `Answer` gains fields with defaults: `model_calls`, `tokens_in`, `tokens_cached`, `tokens_out`,
  `tokens_thought`, `tool_ms: dict[str, float]`. Tokens are summed from `event.usage_metadata`
  (`prompt_token_count`, `cached_content_token_count`, `candidates_token_count`,
  `thoughts_token_count`); tool time from timing `build_tools.safe`.
- `_answer` logs one line per reply: guild, channel, exec, rounds, `ms_total`, `ms_context`,
  `ms_agent`, per-tool ms, the four token counts. IDs and numbers only, never content.
- Memory logs: per summary update `embedded` / `reused`; per search `ms_embed`, `index_hit`,
  `raw_queries`; index size on load and eviction.

## Testing

Offline, test-first, fakes only (`ScriptedLlm`, fake genai client, fake Firestore which records
queries):

- Embedder: batching, task types, dim, normalisation, retry on 429; `NullEmbedder`.
- `vectorize`: hash reuse embeds only changed items; error path.
- Scheduler: vectors written on update; summary saved when embedding fails; backfill cap and
  ordering; importer summaries get vectors through the backfill.
- `IndexedMemoryStore`: one `all_summaries` per guild on miss; coherence after `put_summary`,
  `put_summary_vectors`, `delete_messages`, `purge`, `delete_guild`; byte-bounded LRU eviction.
- Hybrid `chat_search` with a deterministic fake embedder: English query finds a Thai-only day and
  the reverse; keyword-only fallback; RRF order; `match` field; output shape; never returns a
  channel outside `allowed`; `semantic=False` makes no embedding call.
- Query budget on the fake Firestore with a 90-day guild: `chat_search` at most 10 queries;
  warm `chat_days` at most 3.
- Recent buffer: priming, REST fallback, edit, delete, bulk delete, reconnect reset, eviction,
  formatted output identical to the REST path.
- Prompt: stable prefix byte-identical across two times per language; volatile parts at the end.
- Voice: `answer` passes memory tools scoped to the voice channel; none when memory is off.
- Metrics: `Answer` token fields summed from scripted events carrying `usage_metadata`.
- Live (`-m live`, human approval, well under 1 US cent): embedding probe (model and location,
  batch limit, dim, Thai/English similarity, `MIN_SIMILARITY` calibration); live Firestore test
  extended with the three-equality `messages_after` query.

Plus `uvx ruff check . && uvx ruff format --check .`. Baseline before this work: 794 passed, 3
failed only because `libopus` is not installed on the dev machine (`OpusNotLoaded` in the voice sink
tests).

## Risks

| Risk | Mitigation |
| --- | --- |
| Embedding model unavailable in `global` or different limits | Probe first; `SALID_EMBED_LOCATION`; worst case ship keyword-only (read-path fixes still land) |
| Exact codes in older raw messages found less often | Last 3 days scanned fully; summaries keep numbers and IDs; tool note points to `chat_messages` |
| Similarity threshold wrong | Calibrated in the probe; fused with keyword rank; join mode keyword-only |
| Index coherence needs one writer | True today (one fixed instance; hub never writes summaries); documented |
| Memory on the 2 GiB worker | Index 64 MB cap, buffer about 25 MB; sizes logged |
| Buffer misses gateway events | Primed flag, reset on reconnect, REST fallback; last and droppable |
| Deadline | Ordered by value and risk; one commit per task with a green suite; ship what is complete by 17 Oct |

## Rollout

- One PR from `feature/salid-memory-search`, one commit per task.
- Deploy only with human approval (`scripts/deploy_salid.sh --apply`); afterwards the backfill
  adds vectors to existing summaries at about 10 per tick.
- Kill switch: `SALID_EMBED_MODEL=` (empty) gives keyword-only search and no embedding calls.
- Docs: README team memory section, `.env.example`, CLAUDE.md code map (add `memory.py`,
  `memory_store.py`, `firestore_memory.py`, `memory_summary.py`, `memory_access.py`,
  `memory_import.py`, `image_memory.py`, and the new `embed.py`, `memory_index.py`, `recent.py`).

## Task order

1. Metrics.
2. Prompt order.
3. Voice memory.
4. Firestore read-path fixes.
5. Embedder and live probe (approval needed for the probe).
6. Vectors on summaries, scheduler, importer, backfill.
7. `IndexedMemoryStore` and hybrid `chat_search`, prompt update.
8. Short-term context buffer (droppable).
9. Docs.
