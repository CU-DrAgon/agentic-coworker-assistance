# Salid Memory Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Salid's team-memory search faster, cheaper and meaning-based (Thai and English), give voice answers memory, and log per-answer cost and latency.

**Architecture:** Day-summary items get embeddings at write time (stored on the summary record); an `IndexedMemoryStore` wrapper keeps a bounded per-guild cache of summaries and vectors; `chat_search` fuses keyword and similarity ranks (RRF) and reads raw messages only for the top groups and the last 3 days. Smaller changes: Firestore read-path fixes, a cache-friendly prompt order, voice memory tools, an in-process recent-messages buffer, and token/latency metrics.

**Tech Stack:** Python 3.12, google-adk 2.11, google-genai 2.29 (`embed_content`), Firestore / SQLite, discord.py, pytest (offline fakes).

**Spec:** `docs/superpowers/specs/2026-10-16-salid-memory-search-design.md`

## Global Constraints

- No new dependencies; Python 3.12 stdlib only for vector math (`array("f")`, `math.sumprod`).
- `uvx ruff check . && uvx ruff format --check .` clean (line length 100).
- Code, comments, docs in English; Thai user-facing and prompt text stays Thai (new prompt copy is given verbatim below).
- Logs carry ids, counts and timings only, never message or summary content.
- Every memory read stays scoped by `guild_id` and the allowed-channel set; nothing may widen it.
- No Firestore composite index: only equality filters on several fields, or one single-field range.
- Additive storage only: new optional fields and columns; old records must load unchanged.
- Backward compatible: `Answer`, `DaySummary`, `MemoryTools` outputs gain fields, never lose or rename them.
- Tests offline; live tests only `-m live`, run only after a human approves.
- Never print `.env`, never stage `.env`, `samples/`, `*.json` key files; no deploy and no Discord posts without approval.
- Baseline: `uv run pytest -q` = 794 passed, 3 failed (`OpusNotLoaded` in `tests/salid/test_meeting.py` sink tests: `libopus` missing locally). "Suite green" below means no failures other than those 3.

## Review Focus

1. A summary with zero items (an empty regenerated day) must count as vectorised, or the backfill re-selects it forever. Test: Task 6 `test_empty_summary_counts_as_vectorised`.
2. Changing `SALID_EMBED_MODEL` or `SALID_EMBED_DIM` leaves vectors of another size; `math.sumprod` raises on unequal lengths, so stale vectors must be ignored in search and re-made by the backfill. Tests: Task 6 `test_backfill_redoes_vectors_of_another_model`, Task 7 `test_stale_vectors_are_ignored_not_crashed_on`.
3. The embedding API returning fewer vectors than texts must raise, so the summary is saved without vectors instead of with misaligned ones. Test: Task 5 `test_vertex_embedder_rejects_a_short_response`.
4. Tool threads read the index while the scheduler thread writes it; readers must get a snapshot list. Test: Task 7 `test_summary_entries_returns_a_snapshot`.
5. The model may call `chat_search` with only `query` and no keywords; that must work when embeddings are on and return the existing error only when both are empty. Test: Task 7 `test_query_only_search_works`.

## File Structure

| File | Responsibility | Tasks |
| --- | --- | --- |
| `src/salid/qa.py` | `Answer` metrics, prompt split, tool timing, hybrid `chat_search`, prompt copy | 1, 2, 4, 7 |
| `src/salid/bot.py` | answer log line, `_memory_tools_for`, wiring of embedder / index / buffer | 1, 3, 6, 7, 8 |
| `src/salid/meeting.py` | voice answers pass memory tools | 3 |
| `src/salid/firestore_memory.py` | parallel day reads, channel filter, new store methods | 4, 6 |
| `src/salid/llm.py` | shared retry helper, `embed()` | 5 |
| `src/salid/embed.py` (new) | embedders, vector packing, cosine | 5 |
| `src/salid/config.py`, `.env.example` | `embed_model`, `embed_dim`, `embed_location` | 5, 9 |
| `src/salid/memory_summary.py` | `summary_items()` | 6 |
| `src/salid/memory_store.py` | `DaySummary` vector fields, protocol + SQLite methods, migration | 6 |
| `src/salid/memory.py` | `vectorize`, `vectors_usable`, scheduler vectors and backfill | 6 |
| `src/salid/memory_index.py` (new) | `IndexedMemoryStore`, `IndexEntry`, `entries_for` | 7 |
| `src/salid/recent.py` (new) | `RecentMessages` short-term buffer | 8 |
| `tests/salid/conftest.py`, `tests/salid/fake_firestore.py` | usage metadata in `ScriptedLlm`, `FakeEmbedder`, executed-query log | 1, 4, 6 |
| `tests/salid/test_embed.py`, `test_memory_index.py`, `test_recent.py` (new) | unit tests | 5, 7, 8 |
| `README.md`, `CLAUDE.md` | docs | 9 |

---

### Task 1: Per-answer metrics

**Files:**
- Modify: `src/salid/qa.py` (`Answer`, `build_tools`, `ask_async`)
- Modify: `src/salid/bot.py` (`_answer`)
- Modify: `tests/salid/conftest.py` (`ScriptedLlm`, new `usage()` helper)
- Test: `tests/salid/test_salid_qa.py`, `tests/salid/test_salid_features.py`

**Interfaces:**
- Produces: `Answer(text, rounds, seconds, tool_calls, model_calls=0, tokens_in=0, tokens_cached=0, tokens_out=0, tokens_thought=0, tool_ms: dict[str, float] = {})`; `build_tools(ledger, calls=None, memory=None, timings: dict[str, float] | None = None) -> list` (adds elapsed ms per tool name into `timings`).

- [ ] **Step 1: Let `ScriptedLlm` carry usage metadata**

In `conftest.py`, `ScriptedLlm.generate_content_async` yields `LlmResponse(content=..., usage_metadata=getattr(item, "usage_metadata", None))`. Add:

```python
def usage(resp, prompt=0, cached=0, out=0, thought=0):
    resp.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        cached_content_token_count=cached,
        candidates_token_count=out,
        thoughts_token_count=thought,
    )
    return resp
```

- [ ] **Step 2: Write the failing tests**

`tests/salid/test_salid_qa.py`:

```python
def test_answer_reports_tokens_and_tool_time(settings, seeded):
    llm = scripted(
        usage(
            fc_resp(("spending_summary", {"start_date": "2026-10-01", "end_date": "2026-10-14"})),
            prompt=100,
            cached=40,
            out=10,
            thought=5,
        ),
        usage(text_resp("ยอด ฿10,370.50"), prompt=120, cached=60, out=20),
    )
    a = ask(
        settings, seeded, "เดือนนี้เท่าไร", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm
    )
    assert (a.model_calls, a.tokens_in, a.tokens_cached, a.tokens_out, a.tokens_thought) == (
        2,
        220,
        100,
        30,
        5,
    )
    assert set(a.tool_ms) == {"spending_summary"} and a.tool_ms["spending_summary"] >= 0


def test_answer_without_usage_metadata_reports_zero_tokens(settings, seeded):
    a = ask(
        settings,
        seeded,
        "hi",
        guild_id=GUILD,
        is_exec=True,
        user_id=EXEC_ID,
        model=scripted(text_resp("hi")),
    )
    assert (a.model_calls, a.tokens_in, a.tool_ms) == (1, 0, {})
```

`tests/salid/test_salid_features.py` (reuse `_real_answer`; the stub has no token fields, the log must still work):

```python
async def test_answer_logs_one_metrics_line_without_content(bot, world, monkeypatch, caplog):
    _real_answer(bot, monkeypatch, "ลับมาก")
    m = say(world, world.gen, "สลิด ยอดเท่าไร")
    with caplog.at_level("INFO", logger="salid.bot"):
        await bot._answer(m, world.g, m.author)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("answer "))
    for key in (
        "guild=",
        "channel=",
        "ms_total=",
        "ms_context=",
        "ms_agent=",
        "tok_in=0",
        "tok_cached=0",
    ):
        assert key in line
    assert "ลับมาก" not in line and "ยอด" not in line
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/salid/test_salid_qa.py -k "tokens" tests/salid/test_salid_features.py -k metrics -q`
Expected: FAIL (`Answer` has no `model_calls`; no `answer ` log line).

- [ ] **Step 4: Implement**

- `Answer`: add the fields above with defaults (`tool_ms` via `field(default_factory=dict)`).
- `build_tools`: `safe()` times `fn()` with `time.monotonic()` and adds ms to `timings[name]` (sum when a tool runs twice).
- `ask_async`: count model calls via `build_agent(on_model_call=...)`; sum `event.usage_metadata` fields (treat `None` as 0) over all events; fill the new `Answer` fields.
- `bot._answer`: take `t0 = time.monotonic()` at entry, `t_ctx` after `_memory_tools`, `t_agent` after `ask`; replace the `"answered in ..."` log with one line:
  `answer guild=%s channel=%s exec=%s rounds=%d calls=%d ms_total=%d ms_context=%d ms_agent=%d tools=%s tok_in=%d tok_cached=%d tok_out=%d tok_thought=%d`, where `tools` is `name:ms,...` (or `-`). Read every new field with `getattr(ans, name, default)`: stubs return only `text`, `seconds`, `rounds`.

- [ ] **Step 5: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_salid_qa.py tests/salid/test_salid_features.py -q && uv run pytest -q`
Expected: PASS; suite green.

- [ ] **Step 6: Commit**

```bash
git add src/salid/qa.py src/salid/bot.py tests/salid/conftest.py tests/salid/test_salid_qa.py tests/salid/test_salid_features.py
git commit -m "Salid: log tokens, cached tokens and per-tool latency for every answer"
```

---

### Task 2: Cache-friendly system prompt

**Files:**
- Modify: `src/salid/qa.py` (`system_prompt`)
- Test: `tests/salid/test_salid_qa.py`

**Interfaces:**
- Produces: `stable_prompt(language: str, memory: bool) -> str` (persona, general rules, `CAPABILITIES_PROMPT`, `MEMORY_PROMPT` when `memory`); `volatile_prompt(settings, now, *, scoped: bool, interjection: bool, language: str) -> str` (date/time line, scope note, `INTERJECTION_PROMPT`, language line). `system_prompt(...)` keeps its signature and returns `stable_prompt(...) + volatile_prompt(...)`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("lang", ["th", "en", "auto"])
def test_prompt_prefix_is_identical_across_times_and_askers(settings, lang):
    t1 = datetime(2026, 10, 14, 9, 1, tzinfo=ZoneInfo("Asia/Bangkok"))
    t2 = datetime(2026, 10, 16, 18, 47, tzinfo=ZoneInfo("Asia/Bangkok"))
    a = system_prompt(settings, scoped=False, now=t1, language=lang, memory=True)
    b = system_prompt(settings, scoped=True, now=t2, language=lang, memory=True, interjection=True)
    stable = stable_prompt(lang, memory=True)
    assert a.startswith(stable) and b.startswith(stable)
    assert stable.endswith(MEMORY_PROMPT) and CAPABILITIES_PROMPT in stable
    assert "2026-10-14" not in stable and a.index("2026-10-14") > len(stable)
    assert "ผู้ถามไม่ใช่ผู้บริหาร" in b[len(stable) :] and NO_REPLY in b[len(stable) :]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/salid/test_salid_qa.py -k prefix -q`
Expected: FAIL (`ImportError: stable_prompt`).

- [ ] **Step 3: Implement** `stable_prompt` / `volatile_prompt` by moving the existing strings; wording unchanged, only order. The rules sentence that contains the date ("วันนี้คือ ... ใช้วันที่นี้แปลง ...") moves whole into `volatile_prompt`; the rest of the general rules stay in `stable_prompt`.

- [ ] **Step 4: Run to verify it passes, then the suite**

Run: `uv run pytest -q`
Expected: suite green (existing prompt tests check substrings, not order).

- [ ] **Step 5: Commit**

```bash
git add src/salid/qa.py tests/salid/test_salid_qa.py
git commit -m "Salid: stable system prompt first so Gemini can cache the prefix"
```

---

### Task 3: Voice answers get team memory

**Files:**
- Modify: `src/salid/bot.py` (`_memory_tools`, new `_memory_tools_for`)
- Modify: `src/salid/meeting.py` (`MeetingManager.answer`)
- Test: `tests/salid/test_meeting.py`

**Interfaces:**
- Produces: `async SecretaryBot._memory_tools_for(guild, member, answering_channel_id: int | None) -> MemoryTools | None` (`None` answering = DM rule). `_memory_tools(message, guild, member)` computes the answering channel as today and delegates.

- [ ] **Step 1: Write the failing tests**

```python
async def test_wake_word_answers_get_memory_scoped_to_the_voice_channel(env, monkeypatch):
    s = await start(env)
    scoped, asked = [], {}

    async def tools_for(guild, member, answering):
        scoped.append((guild.id, member.id, answering))
        return "MEMORY"

    monkeypatch.setattr(env.bot, "_memory_tools_for", tools_for)
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.update(kw) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด เมื่อวานตกลงงบไว้เท่าไร")
    await env.mgr.handle_segment(s, segment(11))
    assert scoped == [(1000, 11, env.chan.id)] and asked["memory"] == "MEMORY"
    await env.mgr.end(1000)


async def test_wake_word_without_the_memory_feature_passes_no_memory(env, monkeypatch):
    env.bot.guilds_store.save_settings(1000, GuildSettingsUpdate(features=dict(memory=False)), "x")
    s = await start(env)
    asked = {}
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.update(kw) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปเมื่อวานหน่อย")
    await env.mgr.handle_segment(s, segment(11))
    assert asked["memory"] is None
    await env.mgr.end(1000)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_meeting.py -k "memory" -q`
Expected: FAIL (`KeyError: 'memory'`).

- [ ] **Step 3: Implement**: split `_memory_tools`; in `MeetingManager.answer` call `await bot._memory_tools_for(member.guild, member, s.voice_channel.id)` before `ask` and pass `memory=`.

- [ ] **Step 4: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_meeting.py tests/salid/test_memory.py -q && uv run pytest -q`
Expected: PASS (except the 3 known sink failures); suite green.

- [ ] **Step 5: Commit**

```bash
git add src/salid/bot.py src/salid/meeting.py tests/salid/test_meeting.py
git commit -m "Salid: voice wake-word answers can search team memory"
```

---

### Task 4: Firestore read-path fixes

**Files:**
- Modify: `tests/salid/fake_firestore.py` (executed-query log)
- Modify: `src/salid/firestore_memory.py` (`_day`, `messages`, `messages_after`)
- Modify: `src/salid/qa.py` (`MemoryTools.chat_days` tail)
- Test: `tests/salid/test_memory.py`

**Interfaces:**
- Produces: `FakeFirestore.streams: list[tuple[str, tuple[tuple[str, str, object], ...]]]` (collection name and `(field, op, value)` filters, one entry per executed `stream()`); `FirestoreMemoryStore._day(guild_id, day, channel_id: int | None = None)`.

- [ ] **Step 1: Add the executed-query log to the fake**: `FakeFirestore.__init__` adds `self.streams = []`; `FakeQuery.stream()` appends `(self.col.name, tuple((f.field_path, f.op_string, f.value) for f in self.filters))`. The old `queries` list stays.

- [ ] **Step 2: Write the failing tests**

```python
def test_firestore_messages_after_filters_the_channel_server_side(settings):
    st = offline_memory_store()
    st.add_message(mk(settings, 1, bkk(14, 9).astimezone(UTC), cid=10))
    st.add_message(mk(settings, 2, bkk(14, 9).astimezone(UTC), cid=11))
    st.client.streams.clear()
    assert [m.message_id for m in st.messages_after(GUILD, 10, "2026-10-14")] == [1]
    assert ("channel_id", "==", 10) in st.client.streams[-1][1]


def test_chat_days_reads_each_tail_day_once(settings):
    st = offline_memory_store()
    for i, cid in enumerate((10, 11, 12) * 2):
        day = 13 + i // 3
        st.add_message(mk(settings, i + 1, bkk(day, 9, i).astimezone(UTC), "x", cid=cid))
    tools = MemoryTools(st, GUILD, {10: "a", 11: "b", 12: "c"}, settings)
    st.client.streams.clear()
    r = tools.chat_days("2026-10-13", "2026-10-14")
    msg_reads = [f for name, f in st.client.streams if name == "salid_messages"]
    assert len(msg_reads) == 2 and len(r["newer_than_summary"]) == 6
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/salid/test_memory.py -k "server_side or tail_day_once" -q`
Expected: FAIL (no channel filter; 8 message reads).

- [ ] **Step 4: Implement**

- `_day` adds `("channel_id", "==", channel_id)` when given; `messages_after` passes it; `messages` passes `channel_id` and runs the per-day reads through `concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(days)))`, preserving day order before `select_messages`.
- `chat_days` tail: for each of yesterday and today inside the range, one `self.store.messages(gid, self.allowed, d, d, limit=MAX_ROWS, newest_first=False)`; group by `channel_id`; skip channels whose summary is final; keep messages with `m.key > summary.cursor`; first `TAIL_PER_CHANNEL` per channel; then the existing `tail[-TAIL_TOTAL:]`. Import `MAX_ROWS` from `memory_store`.

- [ ] **Step 5: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_memory.py -q && uv run pytest -q`
Expected: PASS (existing `test_chat_days_returns_summaries_and_todays_raw_tail` unchanged); suite green.

- [ ] **Step 6: Commit**

```bash
git add tests/salid/fake_firestore.py src/salid/firestore_memory.py src/salid/qa.py tests/salid/test_memory.py
git commit -m "Salid: fewer Firestore reads for memory tools (channel filter, parallel days, one read per tail day)"
```

---

### Task 5: Embedder and live probe

**Files:**
- Create: `src/salid/embed.py`
- Modify: `src/salid/llm.py`, `src/salid/config.py`
- Test: `tests/salid/test_embed.py`, `tests/salid/test_salid_live.py`

**Interfaces:**
- Produces (`salid.embed`): `class Embedder(Protocol)` with `model_id: str`, `dim: int`, `embed(texts: list[str], task: str) -> list[array]` (`task` is `"document"` or `"query"`); `VertexEmbedder(client, model: str, dim: int, batch: int = EMBED_BATCH)` with `model_id = f"{model}@{dim}"`; `NullEmbedder` (`model_id = ""`, `dim = 0`, `embed` returns `[]`); `make_embedder(settings, client) -> Embedder`; `normalize(values) -> array`; `pack(vecs: list[array]) -> bytes`; `unpack(data: bytes, dim: int) -> list[array]`; `cosine(a: array, b: array) -> float`.
- Produces (`salid.llm`): `with_retry(fn, delays=None)` and `embed(client, model, contents, config, delays=None)`, where `None` means the module's `RETRY_DELAYS` read at call time (`generate` keeps its current signature).
- Produces (`Settings`): `embed_model: str = "gemini-embedding-001"`, `embed_dim: int = 256`, `embed_location: str = ""`.

- [ ] **Step 1: Write the failing tests** (`tests/salid/test_embed.py`)

```python
class FakeEmbedClient:
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config))
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def test_vertex_embedder_batches_and_sets_task_and_dim():
    c = FakeEmbedClient()
    vecs = VertexEmbedder(c, "m", 2, batch=2).embed(["a", "b", "c", "d", "e"], "document")
    assert [len(x[1]) for x in c.calls] == [2, 2, 1] and len(vecs) == 5
    cfg = c.calls[0][2]
    assert cfg.task_type == "RETRIEVAL_DOCUMENT" and cfg.output_dimensionality == 2
    assert vecs[0].typecode == "f" and abs(cosine(vecs[0], vecs[0]) - 1.0) < 1e-6  # normalised


def test_query_task_and_model_id():
    c = FakeEmbedClient()
    e = VertexEmbedder(c, "gemini-embedding-001", 2)
    e.embed(["q"], "query")
    assert c.calls[0][2].task_type == "RETRIEVAL_QUERY" and e.model_id == "gemini-embedding-001@2"


def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("salid.llm.RETRY_DELAYS", (0.0,))
    c = FakeEmbedClient(errors.ClientError(429, {"error": {"message": "quota"}}))
    assert len(VertexEmbedder(c, "m", 2).embed(["a"], "document")) == 1 and len(c.calls) == 2


def test_vertex_embedder_rejects_a_short_response():
    c = FakeEmbedClient()
    c.embed_content = lambda model, contents, config: SimpleNamespace(embeddings=[])
    with pytest.raises(ValueError):
        VertexEmbedder(c, "m", 2).embed(["a"], "document")


def test_kill_switch_and_missing_client_give_the_null_embedder():
    assert isinstance(make_embedder(Settings(embed_model=""), object()), NullEmbedder)
    assert isinstance(make_embedder(Settings(), None), NullEmbedder)
    assert NullEmbedder().embed(["a"], "query") == [] and NullEmbedder().model_id == ""


def test_pack_roundtrip():
    vs = [normalize([1.0, 2.0]), normalize([0.0, 1.0])]
    back = unpack(pack(vs), 2)
    assert len(pack(vs)) == 16 and [list(v) for v in back] == [list(v) for v in vs]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_embed.py -q`
Expected: FAIL (`ModuleNotFoundError: salid.embed`).

- [ ] **Step 3: Implement**

- `llm.py`: extract the existing retry loop into `with_retry(fn, delays)` (reads `RETRY_DELAYS` at call time so the test can patch it); `generate` and new `embed` use it.
- `embed.py`: `EMBED_BATCH = 1` (raised in Step 6); `VertexEmbedder.embed` maps `task` to `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`, splits into batches of `batch`, calls `llm.embed(client, model, batch_texts, EmbedContentConfig(task_type=..., output_dimensionality=dim))`, raises `ValueError` when a response has a different number of embeddings than inputs, returns `normalize(values)` for each. `normalize` returns `array("f")` scaled to unit length (a zero vector stays zero). `pack` / `unpack` use little-endian float32 (`byteswap()` on big-endian hosts). `cosine` is `math.sumprod(a, b)` (inputs are normalised). `make_embedder`: `NullEmbedder` when `settings.embed_model` is empty or `client` is `None`; when `settings.embed_location` is set and differs from `settings.gemini_location`, build `genai.Client(vertexai=True, project=settings.gcp_project, location=settings.embed_location)`.
- `config.py`: the three settings, with a comment pointing to `salid.embed`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/salid/test_embed.py -q`
Expected: PASS.

- [ ] **Step 5: Add the live probe** to `tests/salid/test_salid_live.py` (marked `live`): embed `["งบโฆษณาเดือนนี้ 50,000 บาท", "marketing budget for this month", "ซื้อน้ำแข็ง 3 ถุง"]` as documents with `VertexEmbedder(make_client(Settings()), Settings().embed_model, 256, batch=3)`; assert `len(v) == 256` for each and `cosine(th, en) > cosine(th, ice)`; print the three similarities and whether the 3-text batch succeeded (catch the error, retry with `batch=1`, print which worked).

- [ ] **Step 6: HUMAN GATE, then calibrate.** Ask the human to approve one live run (cost well under 1 US cent). Run: `uv run pytest tests/salid/test_salid_live.py -m live -k probe -s`. Record the output in the commit message. Set `EMBED_BATCH` to the largest batch that worked (cap 100) and note the measured Thai/English and unrelated similarities next to `MIN_SIMILARITY` in Task 7. If the model fails in `global`, retry once with `SALID_EMBED_LOCATION=us-central1` and make that the documented default; if no model works, set the `embed_model` default to `""` and continue (keyword-only path).

- [ ] **Step 7: Commit**

```bash
git add src/salid/embed.py src/salid/llm.py src/salid/config.py tests/salid/test_embed.py tests/salid/test_salid_live.py
git commit -m "Salid: Vertex embedder with retry, packing and kill switch (probe: <results>)"
```

---

### Task 6: Vectors on day summaries, with backfill

**Files:**
- Modify: `src/salid/memory_summary.py` (`summary_items`)
- Modify: `src/salid/memory_store.py` (`DaySummary`, protocol, SQLite), `src/salid/firestore_memory.py`
- Modify: `src/salid/memory.py` (`vectorize`, `vectors_usable`, scheduler)
- Modify: `src/salid/qa.py` (`MemoryTools._summary_lines` becomes a call to `summary_items`)
- Modify: `src/salid/bot.py` (create `self.embedder = make_embedder(settings, llm)`, pass to the scheduler)
- Modify: `tests/salid/conftest.py` (`FakeEmbedder`)
- Test: `tests/salid/test_memory.py`

**Interfaces:**
- Consumes: Task 5 `Embedder`, `pack`, `unpack`.
- Produces:
  - `summary_items(summary: dict) -> list[str]` (same lines and order as today's `_summary_lines`).
  - `DaySummary.vec_model: str = ""`, `vec_hashes: list[str] = []`, `vecs: bytes = b""`.
  - `MemoryStore.all_summaries(guild_id) -> list[DaySummary]`; `MemoryStore.put_summary_vectors(guild_id, channel_id, local_date, vec_model, vec_hashes, vecs) -> bool` (False when the summary does not exist; touches only the three fields).
  - `memory.item_hash(text: str) -> str` (first 12 hex chars of SHA-1 of the UTF-8 text); `memory.Vectors(NamedTuple)`: `model_id: str, hashes: list[str], data: bytes, embedded: int, reused: int`; `memory.vectorize(summary: dict, previous: DaySummary | None, embedder) -> Vectors`; `memory.vectors_usable(s: DaySummary, model_id: str) -> bool`.
  - `MemoryScheduler(..., embedder=None)` (default `NullEmbedder()`); `MAX_VECTORIZE_PER_TICK = 10`.

- [ ] **Step 1: Add `FakeEmbedder` to `conftest.py`**

```python
CONCEPTS = {
    "ads": ("งบโฆษณา", "โฆษณา", "ads", "marketing", "advertis"),
    "ice": ("น้ำแข็ง", "ice"),
    "rent": ("ค่าเช่า", "rent"),
}


class FakeEmbedder:
    """Deterministic embedder: one axis per concept a text mentions (Thai or English) plus a small
    constant axis, normalised. Same concept -> cosine 1.0; unrelated -> below 0.3."""

    model_id, dim = "fake@4", 4

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def embed(self, texts, task):
        self.calls.append((list(texts), task))
        if self.fail:
            raise RuntimeError("embedding down")
        low = [t.casefold() for t in texts]
        return [
            normalize([float(any(w in t for w in ws)) for ws in CONCEPTS.values()] + [0.3])
            for t in low
        ]
```

- [ ] **Step 2: Write the failing tests** (`tests/salid/test_memory.py`, new section "vectors")

```python
def test_summary_items_lists_every_part():
    s = {
        "overview": "o",
        "topics": [{"title": "t", "details": "d", "people": ["แอน"]}],
        "decisions": ["x"],
        "action_items": [{"who": "บอย", "what": "w", "due": ""}],
        "numbers": ["5"],
        "open_questions": ["q"],
    }
    assert summary_items(s) == [
        "o",
        "Topic: t d แอน",
        "Decision: x",
        "Action: บอย w",
        "Number: 5",
        "Open: q",
    ]


def test_update_writes_vectors_and_reuses_unchanged_items(mstore, settings):
    first = json_resp({"overview": "งบโฆษณา", "decisions": ["d1"]})
    second = json_resp({"overview": "งบโฆษณา", "decisions": ["d1", "ซื้อน้ำแข็ง"]})
    clock, llm, emb = Clock(bkk(14, 10)), FakeClient([first, second]), FakeEmbedder()
    s = sched(mstore, settings, clock, llm, embedder=emb)
    svc = MemoryService(mstore, settings, s)
    for i in range(30):
        log(svc, settings, i + 1, bkk(14, 9, i))
    s.tick([GUILD])
    for i in range(30, 60):
        log(svc, settings, i + 1, bkk(14, 10, i - 30))  # after the first update's cursor
    s.tick([GUILD])
    assert emb.calls[1] == (["Decision: ซื้อน้ำแข็ง"], "document")
    got = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert vectors_usable(got, "fake@4") and len(unpack(got.vecs, 4)) == 3


def test_summary_is_saved_when_embedding_fails(mstore, settings):
    clock, llm = Clock(bkk(14, 10)), FakeClient([summary_json("a")])
    s = sched(mstore, settings, clock, llm, embedder=FakeEmbedder(fail=True))
    svc = MemoryService(mstore, settings, s)
    for i in range(30):
        log(svc, settings, i + 1, bkk(14, 9, i))
    s.tick([GUILD])
    got = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert got.summary["overview"] == "overview a" and got.vec_model == ""


def test_backfill_vectorises_old_summaries_newest_first_with_a_cap(mstore, settings):
    for d in range(1, 13):
        mstore.put_summary(
            DaySummary(
                guild_id=GUILD,
                channel_id=10,
                local_date=f"2026-10-{d:02d}",
                final=True,
                message_count=3,
                summary={"overview": f"day {d}"},
            )
        )
    emb = FakeEmbedder()
    s = sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=emb)
    s.tick([GUILD])
    assert [c[0] for c in emb.calls][0] == ["day 12"] and len(emb.calls) == 10
    s.tick([GUILD])
    assert len(emb.calls) == 12
    s.tick([GUILD])
    assert len(emb.calls) == 12  # nothing left: the guild is skipped
    assert all(vectors_usable(x, "fake@4") for x in mstore.all_summaries(GUILD))


def test_backfill_redoes_vectors_of_another_model(mstore, settings):
    old = vectorize({"overview": "งบโฆษณา"}, None, FakeEmbedder())
    mstore.put_summary(
        DaySummary(
            guild_id=GUILD,
            channel_id=10,
            local_date="2026-10-01",
            final=True,
            summary={"overview": "งบโฆษณา"},
            vec_model="other@8",
            vec_hashes=old.hashes,
            vecs=old.data,
        )
    )
    emb = FakeEmbedder()
    sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=emb).tick([GUILD])
    assert mstore.get_summary(GUILD, 10, "2026-10-01").vec_model == "fake@4"


def test_empty_summary_counts_as_vectorised():
    s = DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-01", summary={})
    assert vectors_usable(s, "fake@4") and vectorize({}, None, FakeEmbedder()).embedded == 0


def test_put_summary_vectors_and_all_summaries_parity(mstore):
    mstore.put_summary(
        DaySummary(
            guild_id=GUILD,
            channel_id=10,
            local_date="2026-10-01",
            summary={"overview": "a"},
            message_count=4,
        )
    )
    mstore.put_summary(
        DaySummary(guild_id=GUILD + 1, channel_id=10, local_date="2026-10-01", summary={})
    )
    assert mstore.put_summary_vectors(GUILD, 10, "2026-10-01", "fake@4", ["h"], b"\0" * 16)
    assert not mstore.put_summary_vectors(GUILD, 99, "2026-10-01", "fake@4", ["h"], b"")
    got = mstore.all_summaries(GUILD)
    assert [(x.channel_id, x.vec_model, x.vecs, x.message_count) for x in got] == [
        (10, "fake@4", b"\0" * 16, 4)
    ]


def test_old_sqlite_summary_table_gets_vector_columns(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as c:
        c.execute(
            "CREATE TABLE salid_day_summaries (guild_id INTEGER, channel_id INTEGER, local_date TEXT, "
            "summary TEXT, last_message_at TEXT, last_message_id INTEGER DEFAULT 0, message_count INTEGER DEFAULT 0, "
            "updated_at TEXT, final INTEGER DEFAULT 0, dirty INTEGER DEFAULT 0, PRIMARY KEY (guild_id, channel_id, local_date))"
        )
        c.execute(
            "INSERT INTO salid_day_summaries (guild_id, channel_id, local_date, summary) VALUES (1000, 10, '2026-10-01', '{}')"
        )
    st = SQLiteMemoryStore(path)
    assert st.get_summary(GUILD, 10, "2026-10-01").vec_model == ""
    assert st.put_summary_vectors(GUILD, 10, "2026-10-01", "fake@4", ["h"], b"\1" * 16)
    assert st.get_summary(GUILD, 10, "2026-10-01").vecs == b"\1" * 16
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/salid/test_memory.py -k "vector or summary_items or all_summaries or sqlite_summary_table" -q`
Expected: FAIL (`ImportError: summary_items`).

- [ ] **Step 4: Implement**

- `summary_items` moves the body of `MemoryTools._summary_lines`; `MemoryTools` calls it.
- `DaySummary`: three fields. SQLite: `SCHEMA` gains `vec_model TEXT DEFAULT ''`, `vec_hashes TEXT DEFAULT '[]'`, `vecs BLOB`; on open, read `PRAGMA table_info(salid_day_summaries)` and `ALTER TABLE ... ADD COLUMN` each missing one; `_sum` decodes `vec_hashes` JSON and `vecs` (`None` becomes `b""`); `put_summary` writes all 13 columns. `all_summaries` is one SELECT by `guild_id`; `put_summary_vectors` is an UPDATE returning `rowcount > 0`.
- Firestore: `all_summaries` is one `guild_id ==` query on the summaries collection; `put_summary_vectors` does `get()`; returns False when missing, else `update()` of the three fields.
- `memory.vectorize`: items via `summary_items`; reuse a previous vector when `previous` has `vec_model == embedder.model_id` and the same item hash; embed the rest in one `embedder.embed(missing, "document")` call (no call when nothing is missing); `Vectors(model_id="", hashes=[], data=b"", embedded=0, reused=0)` for a `NullEmbedder`. `vectors_usable(s, model_id)` is `bool(model_id) and (not items or (s.vec_model == model_id and len(s.vec_hashes) == len(items)))` with `items = summary_items(s.summary)`.
- Scheduler: `_update` calls `vectorize(new, summary, self.embedder)` after the summary is computed, inside its own try/except (warning with ids only; empty vectors on failure); puts the three fields on `out`; logs `embedded` / `reused`; discards `gid` from `self._vectors_done`. `tick` then calls `self._backfill(guild_ids)` when `self.embedder.model_id`: next enabled guild round-robin not in `self._vectors_done`; `[s for s in store.all_summaries(gid) if not vectors_usable(s, model_id)]` sorted by `local_date` descending; vectorise up to `MAX_VECTORIZE_PER_TICK` and `put_summary_vectors`; on any exception log and stop for this tick; when none were missing add `gid` to `self._vectors_done`.
- Bot: `self.embedder = make_embedder(settings, llm)` before the scheduler; pass `embedder=self.embedder`.

- [ ] **Step 5: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_memory.py tests/salid/test_memory_import.py -q && uv run pytest -q`
Expected: PASS; suite green (the bot fixture passes `llm=None`, so it gets a `NullEmbedder`).

- [ ] **Step 6: Commit**

```bash
git add src/salid/memory_summary.py src/salid/memory_store.py src/salid/firestore_memory.py src/salid/memory.py src/salid/qa.py src/salid/bot.py tests/salid/conftest.py tests/salid/test_memory.py
git commit -m "Salid: embed day-summary items at write time, reuse unchanged vectors, backfill old summaries"
```

---

### Task 7: Index wrapper and hybrid `chat_search`

**Files:**
- Create: `src/salid/memory_index.py`
- Modify: `src/salid/qa.py` (`MemoryTools`, `build_tools.chat_search`, `MEMORY_PROMPT`)
- Modify: `src/salid/bot.py` (wrap the store; pass `embedder` to `MemoryTools`; `semantic=False` in `_interjection_hints`)
- Test: `tests/salid/test_memory_index.py`, `tests/salid/test_memory.py`

**Interfaces:**
- Consumes: Task 5 `Embedder`, `unpack`, `cosine`; Task 6 `all_summaries`, `put_summary_vectors`, `summary_items`, `vectors_usable`.
- Produces:
  - `IndexEntry` (dataclass): `summary: DaySummary`, `items: list[str]`, `vectors: list[array]` (empty unless `vectors_usable`), properties `channel_id`, `local_date`.
  - `IndexedMemoryStore(base: MemoryStore, embedder=None, max_bytes: int = INDEX_MAX_BYTES)`; `INDEX_MAX_BYTES = 64 * 1024 * 1024`; `summary_entries(guild_id, allowed_channels, start_date, end_date) -> list[IndexEntry]` (a new list each call, sorted by `(local_date, channel_id)`); overrides `all_summaries`, `summaries`, `put_summary`, `put_summary_vectors`, `delete_messages`, `purge`, `delete_guild`; delegates everything else via `__getattr__`; attribute `base`.
  - `entries_for(store, guild_id, allowed, start, end, embedder) -> list[IndexEntry]`: `store.summary_entries(...)` when the store has it, else built from `store.summaries(...)`.
  - `MemoryTools(store, guild_id, allowed, settings, retention_days=None, embedder=None)`; `chat_search(keywords: list[str] | None = None, start_date=None, end_date=None, limit=20, query: str | None = None, semantic: bool = True) -> dict`; constants `RAW_GROUPS = 5`, `RAW_RECENT_DAYS = 3`, `MIN_SIMILARITY = 0.55`, `RRF_K = 60`, `QUERY_CACHE_SIZE = 512`.

- [ ] **Step 1: Write the failing index tests** (`tests/salid/test_memory_index.py`)

```python
def summ(day, cid=10, gid=GUILD, text="งบโฆษณา"):
    return DaySummary(
        guild_id=gid,
        channel_id=cid,
        local_date=day,
        message_count=3,
        final=True,
        summary={"overview": text},
    )


def wrapped(max_bytes=INDEX_MAX_BYTES):
    """Firestore fake store whose `all_summaries` calls are counted in `base.loads`."""
    base = offline_memory_store()
    base.loads, real = [], base.all_summaries
    base.all_summaries = lambda gid: base.loads.append(gid) or real(gid)
    return base, IndexedMemoryStore(base, FakeEmbedder(), max_bytes=max_bytes)


def loads(base):
    return base.loads


def test_one_load_query_per_guild_then_served_from_memory():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    base.client.streams.clear()
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")
    st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31")
    assert len(loads(base)) == 1 and len(base.client.streams) == 1


def test_writes_keep_the_index_fresh(settings):
    base, st = wrapped()
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")  # load
    st.put_summary(summ("2026-10-02", text="ค่าเช่า"))
    v = vectorize({"overview": "ค่าเช่า"}, None, FakeEmbedder())
    st.put_summary_vectors(GUILD, 10, "2026-10-02", v.model_id, v.hashes, v.data)
    [e] = st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31")
    assert e.items == ["ค่าเช่า"] and len(e.vectors) == 1
    st.add_message(mk(settings, 1, bkk(2, 9).astimezone(UTC)))
    st.delete_messages(GUILD, [1])
    assert st.summary_entries(GUILD, {10}, "2026-10-02", "2026-10-02")[0].summary.dirty
    st.delete_guild(GUILD)
    assert st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31") == []
    assert len(loads(base)) == 2  # reloaded once after delete_guild


def test_purge_drops_the_cache():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")
    st.purge({GUILD: datetime(2026, 10, 5, tzinfo=UTC)}, datetime(2026, 1, 1, tzinfo=UTC))
    assert st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31") == [] and len(loads(base)) == 2


def test_lru_evicts_by_bytes():
    base, st = wrapped(max_bytes=1)
    base.put_summary(summ("2026-10-01", gid=GUILD))
    base.put_summary(summ("2026-10-01", gid=GUILD + 1))
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-01")
    st.summaries(GUILD + 1, {10}, "2026-10-01", "2026-10-01")
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-01")
    assert len(loads(base)) == 3  # GUILD was evicted by GUILD + 1 and reloaded


def test_summaries_match_the_base_store():
    base, st = wrapped()
    for d, cid in (("2026-10-01", 10), ("2026-10-02", 11), ("2026-10-03", 10)):
        base.put_summary(summ(d, cid))
    for allowed, a, b in (
        ({10}, "2026-10-01", "2026-10-03"),
        ({10, 11}, "2026-10-02", "2026-10-02"),
        (set(), "2026-10-01", "2026-10-03"),
    ):
        assert st.summaries(GUILD, allowed, a, b) == base.summaries(GUILD, allowed, a, b)


def test_summary_entries_returns_a_snapshot():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    got = st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-01")
    got.clear()
    assert len(st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-01")) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_memory_index.py -q`
Expected: FAIL (`ModuleNotFoundError: salid.memory_index`).

- [ ] **Step 3: Implement `memory_index.py`**: per guild an `OrderedDict` of `{(channel_id, local_date): IndexEntry}` plus an approximate byte size (item text bytes + `len(vecs)` + 200 per entry); one `threading.RLock` around all cache access; loading calls `base.all_summaries`; after inserting a guild, evict least recently used guilds while the total exceeds `max_bytes`, never the guild just loaded; `put_summary` / `put_summary_vectors` call the base then update a loaded guild's entry; `delete_messages` marks loaded entries for the returned `(channel_id, local_date)` dirty (`model_copy(update={"dirty": True})`); `purge` clears all guilds; `delete_guild` drops that guild. Vectors are unpacked once at load or write (only when `vectors_usable`). Log `memory index load guild=%s summaries=%d bytes=%d ms=%d` and `memory index evict guild=%s`. Module docstring states the single-writer assumption.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/salid/test_memory_index.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing hybrid-search tests** (`tests/salid/test_memory.py`, section "hybrid search")

```python
def topic_day(st, day, cid, title, details=""):
    """A final day summary with one topic, vectorised with the fake embedder."""
    summary = {"overview": "", "topics": [{"title": title, "details": details, "people": []}]}
    v = vectorize(summary, None, FakeEmbedder())
    st.put_summary(
        DaySummary(
            guild_id=GUILD,
            channel_id=cid,
            local_date=day,
            message_count=3,
            final=True,
            summary=summary,
            vec_model=v.model_id,
            vec_hashes=v.hashes,
            vecs=v.data,
        )
    )


def hybrid_tools(mstore, settings, allowed=None, emb=None):
    st = IndexedMemoryStore(mstore, emb or FakeEmbedder())
    return st, MemoryTools(
        st,
        GUILD,
        allowed or {10: "general", 12: "ads"},
        settings,
        retention_days=30,
        embedder=emb or FakeEmbedder(),
    )


def test_english_query_finds_a_thai_only_day(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา", "ค่าโฆษณาเฟซบุ๊ก 20,000")
    topic_day(st, "2026-10-06", 10, "ซื้อน้ำแข็ง")
    r = tools.chat_search(["marketing budget"], query="what is our marketing budget?")
    assert [(h["date"], h["channel"], h["match"]) for h in r["hits"]] == [
        ("2026-10-05", "ads", "meaning")
    ]
    assert "งบโฆษณา" in r["hits"][0]["summary_hits"][0]


def test_thai_query_finds_an_english_only_day(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 10, "Facebook ads plan")
    assert tools.chat_search(["งบโฆษณา"], query="งบโฆษณาเท่าไร")["hits"][0]["date"] == "2026-10-05"


def test_keyword_and_meaning_together_rank_first(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-03", 10, "Q4 ads plan")  # keyword "ads" and meaning
    topic_day(st, "2026-10-04", 10, "โฆษณา")  # meaning only (newer, so it wins the meaning tie)
    r = tools.chat_search(["ads"], query="ads")
    # RRF: 10-03 = 1/61 + 1/62 beats 10-04 = 1/61
    assert [(h["date"], h["match"]) for h in r["hits"]] == [
        ("2026-10-03", "both"),
        ("2026-10-04", "meaning"),
    ]


def test_embedding_failure_falls_back_to_keywords(mstore, settings):
    st, tools = hybrid_tools(mstore, settings, emb=FakeEmbedder(fail=True))
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    r = tools.chat_search(["งบโฆษณา"], query="marketing budget")
    assert [h["match"] for h in r["hits"]] == ["keyword"] and "error" not in r


def test_semantic_false_makes_no_embedding_call(mstore, settings):
    emb = FakeEmbedder()
    st, tools = hybrid_tools(mstore, settings, emb=emb)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    tools.chat_search(["งบโฆษณา"], semantic=False)
    assert not [c for c in emb.calls if c[1] == "query"]


def test_meaning_hits_never_leave_the_allowed_channels(mstore, settings):
    st, tools = hybrid_tools(mstore, settings, allowed={10: "general"})
    topic_day(st, "2026-10-05", 11, "งบโฆษณาลับ")
    assert tools.chat_search(["marketing"], query="marketing budget")["hits"] == []


def test_stale_vectors_are_ignored_not_crashed_on(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    st.put_summary_vectors(GUILD, 12, "2026-10-05", "other@8", ["x"], b"\0" * 32)
    assert tools.chat_search(["marketing"], query="marketing")["hits"] == []


def test_query_only_search_works(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    assert tools.chat_search([], query="marketing budget")["hits"][0]["date"] == "2026-10-05"
    assert "error" in tools.chat_search([], query=" ")


def test_chat_search_query_budget_on_firestore(settings):
    base = offline_memory_store()
    st, tools = hybrid_tools(base, settings, allowed={10: "general", 11: "b"})
    start = date(2026, 10, 14)
    for n in range(90):
        day = (start - timedelta(days=n)).isoformat()
        for cid in (10, 11):
            base.put_summary(
                DaySummary(
                    guild_id=GUILD,
                    channel_id=cid,
                    local_date=day,
                    message_count=2,
                    final=n > 1,
                    summary={"overview": "ads" if n % 7 == 0 else "x"},
                )
            )
            at = datetime.fromisoformat(day).replace(hour=3, tzinfo=UTC)
            for i in range(2):
                base.add_message(
                    mk(settings, n * 100 + cid * 2 + i, at + timedelta(minutes=i), "ads", cid=cid)
                )
    base.client.streams.clear()
    assert tools.chat_search(["ads"], query="ads")["hits"]
    assert len(base.client.streams) <= 10
    base.client.streams.clear()
    tools.chat_days("2026-10-08", "2026-10-14")
    assert len(base.client.streams) <= 3
```

- [ ] **Step 6: Update the two keyword tests whose fixtures relied on scanning old raw messages** (behaviour change from the spec: raw messages older than `RAW_RECENT_DAYS` are reached through summaries):
  - `test_chat_search_ties_go_newest_first_and_limits_groups`: messages on days 12, 13, 14 instead of 5, 6, 7; expected dates `["2026-10-14", "2026-10-13", "2026-10-12"]`.
  - `test_chat_search_output_is_capped`: 14 final summaries (days 1 to 14) whose overview is `"ads " + "ก" * 450` repeated in six decisions, instead of raw messages; same assertions (`<= 4500` characters, `"truncated"` in `note`, `<= 3` messages per hit).
  - Add `test_old_raw_only_days_point_to_chat_messages`: a message "ads" on day 5 and no summary; `chat_search(["ads"])` has no hits and its `note` contains `"chat_messages"`.

- [ ] **Step 7: Run to verify they fail**

Run: `uv run pytest tests/salid/test_memory.py -k "query or meaning or hybrid or semantic or stale or budget or ties or capped or raw_only" -q`
Expected: FAIL (`MemoryTools` has no `embedder`; no `match` field).

- [ ] **Step 8: Implement hybrid `chat_search`** in `MemoryTools`:

1. Keywords as today (empty list allowed); `query = (query or "").strip()`; error `"give at least one keyword or a query"` when both are empty, or when only a query is given and `semantic` is False or the embedder is a `NullEmbedder`.
2. Range as today. When `semantic` and `embedder.model_id` and (`query` or keywords): submit `embedder.embed([query or " ".join(kws)], "query")` to a one-worker `ThreadPoolExecutor`, with a module-level `OrderedDict` LRU of `QUERY_CACHE_SIZE` keyed `(model_id, text)`.
3. `entries = entries_for(self.store, gid, self.allowed, a, b, self.embedder)`.
4. Per `(local_date, channel_id)` group: keyword score from items (`hits * SUMMARY_WEIGHT`, lines kept as today); similarity = max `cosine(qv, v)` over vectors whose length equals `len(qv)` (skip others); a meaning hit needs `>= MIN_SIMILARITY`. If the embedding call raises, log a warning (ids only) and continue keyword-only.
5. Raw scan of the last `RAW_RECENT_DAYS` days inside the range: one `self.store.messages(gid, self.allowed, max(a, d0), b, limit=MAX_ROWS_SEARCH, newest_first=True)`; keyword hits add to the group's score and messages, as today.
6. Fuse: keyword ranking = groups with score > 0 by `(score, date)` descending; meaning ranking = meaning-hit groups by `(similarity, date)` descending; `rrf = sum(1 / (RRF_K + rank))` over the rankings a group is in (rank starts at 1); order by `(rrf, date)` descending; keep `limit`.
7. For the top `RAW_GROUPS` kept groups older than the raw window: `self.store.messages_after(gid, cid, day, None, MAX_FETCH)` and collect up to `SEARCH_MSGS_PER_GROUP` keyword-matching messages (no keywords: none).
8. Output as today plus per hit `"match"` (`"both"`, `"keyword"` or `"meaning"`); meaning-only hits get their two most similar items as `summary_hits`. `note` parts joined with `" | "`: the existing truncated / no-match notes, plus `"raw messages older than 3 days are searched through summaries only; for an exact code, number or phrase use chat_messages with a keyword and dates"` whenever the range starts before the raw window.
9. `call()` drops `semantic` from model-supplied args.
10. Log one line per call: `memory search guild=%s ms=%d ms_embed=%d ms_index=%d raw_queries=%d hits=%d keyword=%d meaning=%d both=%d` (`raw_queries` = raw-window days read + `messages_after` calls).

`build_tools.chat_search` gains `query: str | None = None` (docstring: "the user's question in plain words, any language; improves meaning-based matching") and its `keywords` docstring becomes "2-5 search words: names, numbers, codes and key terms (Thai or English)". `MEMORY_PROMPT`: replace the sentence that starts "ถ้าคำถามไม่มีวันที่ชัดเจน" up to "แล้วค่อยเปิดวันที่เจอด้วย chat_days หรือ chat_messages." with exactly:

`ถ้าคำถามไม่มีวันที่ชัดเจน ให้เรียก chat_search ก่อน โดยใส่ query เป็นคำถามของผู้ใช้ และ keywords 2-5 คำ (ชื่อคน ตัวเลข รหัส คำสำคัญ) เพราะ chat_search ค้นตามความหมายได้ทั้งไทยและอังกฤษ แล้วค่อยเปิดวันที่เจอด้วย chat_days หรือ chat_messages. ถ้าหารหัส ตัวเลข หรือคำพูดเป๊ะๆ ที่เก่ากว่า 3 วัน ให้ใช้ chat_messages พร้อม keyword และช่วงวันที่.`

Bot: `self.memory_store = IndexedMemoryStore(memory or self._default_memory_store(settings, store), self.embedder)` (create `self.embedder` first); `_memory_tools_for` passes `embedder=self.embedder`; `_interjection_hints` calls `tools.chat_search(kws, limit=..., semantic=False)`.

- [ ] **Step 9: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_memory.py tests/salid/test_memory_index.py tests/salid/test_salid_features.py -q && uv run pytest -q`
Expected: PASS; suite green. Also update `test_ask_wires_the_memory_tools_and_prompt` only if it asserts removed wording (it checks `"chat_search"` and `"ความจำของทีม"`, both kept).

- [ ] **Step 10: Commit**

```bash
git add src/salid/memory_index.py src/salid/qa.py src/salid/bot.py tests/salid/test_memory_index.py tests/salid/test_memory.py
git commit -m "Salid: hybrid keyword + meaning memory search with a bounded per-guild index"
```

---

### Task 8: Short-term context buffer (droppable)

**Files:**
- Create: `src/salid/recent.py`
- Modify: `src/salid/bot.py` (`on_message`, raw edit/delete handlers, `on_ready`, `on_resumed`, `tick`, `_recent_chat`, `_previous_text`, `_gate_lines`, new `_history`)
- Test: `tests/salid/test_recent.py`

**Interfaces:**
- Produces: `Snapshot` (frozen dataclass): `id: int`, `channel_id: int`, `author`, `content: str`, `created_at: datetime`, `attachments: tuple`, `role_mentions: tuple`; `Snapshot.of(message) -> Snapshot`. `RecentMessages(per_channel: int = 40, max_channels: int = 1000, max_age_s: float = 3 * 3600)` with `add(message)`, `edit(channel_id, message_id, content)`, `delete(channel_id, message_ids)`, `prime(channel_id, messages)`, `history(channel_id, limit, before_id=None) -> list[Snapshot] | None` (newest first; `None` when the channel is not primed), `reset()`, `sweep(now: datetime)`. `async SecretaryBot._history(channel, limit, before=None) -> list` (buffer, else Discord history which then primes the buffer).

- [ ] **Step 1: Write the failing tests** (`tests/salid/test_recent.py`)

```python
def msg(mid, cid=10, text="t", at=None, author="a"):
    return SimpleNamespace(
        id=mid,
        channel=SimpleNamespace(id=cid),
        author=author,
        content=text,
        created_at=at or datetime(2026, 10, 14, 5, 0, tzinfo=UTC),
        attachments=[],
        role_mentions=[],
    )


def test_unprimed_channels_return_none_then_buffer_after_prime():
    r = RecentMessages()
    r.add(msg(5))
    assert r.history(10, 10) is None
    r.prime(10, [msg(4), msg(3)])
    assert [s.id for s in r.history(10, 10)] == [5, 4, 3]
    assert [s.id for s in r.history(10, 10, before_id=5)] == [4, 3]


def test_edit_delete_and_bound():
    r = RecentMessages(per_channel=3)
    r.prime(10, [])
    for i in range(1, 6):
        r.add(msg(i))
    r.edit(10, 5, "new")
    r.delete(10, [4])
    assert [(s.id, s.content) for s in r.history(10, 10)] == [(5, "new"), (3, "t")]


def test_reset_and_sweep():
    r = RecentMessages(max_age_s=60)
    r.prime(10, [msg(1)])
    r.reset()
    assert r.history(10, 5) is None
    r.prime(11, [msg(2, cid=11)])
    r.sweep(datetime(2026, 10, 14, 6, 0, tzinfo=UTC))
    assert r.history(11, 5) is None


def test_max_channels_evicts_the_least_recent():
    r = RecentMessages(max_channels=2)
    for cid in (1, 2, 3):
        r.prime(cid, [msg(cid, cid=cid)])
    assert r.history(1, 5) is None and r.history(3, 5) is not None


async def test_bot_reads_history_once_then_from_the_buffer(bot, world):
    calls = []
    original = world.gen.history

    def counting(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    world.gen.history = counting
    bot._start_import = lambda: None  # on_resumed would start a history import
    await send(bot, world, world.gen, "เรื่องงบ")
    m = say(world, world.gen, "สลิด เมื่อกี้ว่าไง")
    await bot.on_message(m)
    a = await bot._recent_chat(m)
    b = await bot._recent_chat(m)
    assert len(calls) == 1 and a == b and "เรื่องงบ" in a
    await bot.on_resumed()
    await bot._recent_chat(m)
    assert len(calls) == 2
```

(`bot`, `world`, `send`, `say` come from `test_modes`, as in `test_memory.py`.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_recent.py -q`
Expected: FAIL (`ModuleNotFoundError: salid.recent`).

- [ ] **Step 3: Implement**

- `recent.py`: an `OrderedDict` of channel id to `deque(maxlen=per_channel)` plus a primed set and last-activity time; `add` ignores objects without `channel.id` (some test doubles have `channel=None`); `prime` merges with what is buffered (dedupe by id, sort by id, keep the newest `per_channel`); `add` and `prime` move the channel to the end and evict the oldest channel beyond `max_channels`; `sweep` drops channels whose newest message is older than `max_age_s`.
- Bot: `self.recent = RecentMessages(max_age_s=HISTORY_SECONDS)`; first line of `on_message` is `self.recent.add(message)`; raw edit / delete / bulk delete also call `edit` / `delete` (with `payload.channel_id`) before the guild check; `on_ready` and `on_resumed` call `self.recent.reset()`; `tick` calls `self.recent.sweep(datetime.now(UTC))` at most once a minute.
- `_history(channel, limit, before=None)`: `self.recent.history(channel.id, limit, before.id if before else None)`; on `None` read `channel.history(limit=max(limit, 40), before=before)` (catch `discord.HTTPException` as today, return `[]`), prime with the result when `before` is `None` or is the newest buffered message, and return the first `limit`. `_recent_chat`, `_previous_text`, `_gate_lines` iterate `await self._history(...)` with their current limits and unchanged formatting.

- [ ] **Step 4: Run to verify they pass, then the suite**

Run: `uv run pytest tests/salid/test_recent.py tests/salid/test_bot_multitenant.py tests/salid/test_modes.py -q && uv run pytest -q`
Expected: PASS (`test_recent_chat_follows_the_whole_room` unchanged); suite green.

- [ ] **Step 5: Commit**

```bash
git add src/salid/recent.py src/salid/bot.py tests/salid/test_recent.py
git commit -m "Salid: in-process recent-messages buffer instead of a Discord history call per reply"
```

---

### Task 9: Docs, final verification, PR

**Files:**
- Modify: `README.md` (Team memory section), `CLAUDE.md` (Salid code map, env vars), `.env.example`

- [ ] **Step 1: Docs**
  - README, Team memory layer 3: hybrid search (keyword + embeddings on summary items, RRF, raw scan of the last 3 days and the top 5 matches, `query` argument), the per-guild index (64 MB cap, one bot process), backfill, kill switch `SALID_EMBED_MODEL=`; voice answers use memory; the per-answer log line.
  - `.env.example`: `# SALID_EMBED_MODEL=gemini-embedding-001   # empty = keyword-only memory search`, `# SALID_EMBED_DIM=256`, `# SALID_EMBED_LOCATION=   # empty = SALID_GEMINI_LOCATION` (values as calibrated in Task 5).
  - CLAUDE.md Salid table: rows for `memory.py`, `memory_store.py`, `firestore_memory.py`, `memory_summary.py`, `memory_access.py`, `memory_import.py`, `image_memory.py`, `embed.py`, `memory_index.py`, `recent.py` (one line each, matching the table's style); env var list adds `SALID_EMBED_*`; note that the memory index and recent buffer assume one bot process.

- [ ] **Step 2: Full verification**

Run: `uv run pytest -q && uvx ruff check . && uvx ruff format --check .`
Expected: suite green (only the 3 known `OpusNotLoaded` failures); ruff clean.

- [ ] **Step 3: HUMAN GATE: live Firestore check.** Ask approval, then run `uv run pytest tests/salid/test_salid_live.py -m live -q` with a live test added for `messages_after` on the throwaway collection (three equality filters return the channel's messages without an index error).

- [ ] **Step 4: Commit and open the PR**

```bash
git status && git diff --cached
git add README.md CLAUDE.md .env.example tests/salid/test_salid_live.py
git commit -m "Docs: Salid hybrid memory search, index, metrics and new settings"
git push -u origin feature/salid-memory-search
gh pr create --title "Salid: faster, cheaper, meaning-based memory search" --body "<summary, test results, live probe numbers, risks>"
```

Pushing and opening the PR are outward-facing: confirm with the human first.
