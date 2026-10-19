# Salid Workflow Monitor (Phases 0-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every `/center` viewer a calm, live, permission-filtered view of Salid's delivery queues (change outbox, web → Discord actions) and background jobs (video ads, deep tasks, deep research), behind `MONITOR_ENABLED`.

**Architecture:** Records stay the source of truth. Pure adapters turn them into `MonitorItem`s that `monitor_api.py` filters per viewer. Producers append content-free events to a per-company ring log (`salid_monitor_log`, transactional `emit_many`, batched once per bot tick). The bot and the platform bridge write a 30 s heartbeat. The `/center` Monitor section polls `changes?since=seq` every 5 s and refetches the changed items in one `ids=` batch.

**Tech Stack:** Python 3.12, FastAPI, pydantic, `DocStore` (SQLite / Firestore), discord.py mixins, vanilla JS in `center.html`, pytest, Node (pure-JS tests), Playwright for Python with pytest-playwright 0.10.0 and axe-playwright-python 0.1.8 (bundles axe-core 4.12.1).

**Spec:** `docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md`

## Global Constraints

- Flags: the web setting `MONITOR_ENABLED` (`meeting_scribe.config.Settings.monitor_enabled`, default false) gates every monitor route (404 when off) and the nav item. The bot setting `SALID_MONITOR_ENABLED` (`salid.config.Settings.monitor_enabled`, default false) gates emit and heartbeat writes. The outbox re-key and action pruning are NOT gated, so there is only one storage layout.
- Change log: DocStore kind `salid_monitor_log`, guild 0, key = org id (an unlinked workspace uses `guild:<memory id>`). The doc is `{seq, floor, events}`; `RING = 200`. Each event is exactly `{v: 1, seq, at, kind, id, status, attention, audience}`. `floor` = the highest seq no longer in the ring.
- `emit_many` runs in ONE `DocStore.transact`, so `seq` only increases. It is best effort: on failure it logs, does `EMIT_STATS["failed"] += 1` and returns `None`, never raising. The bot flushes its buffer once per tick (`TICK_SECONDS = 5.0`), one `emit_many` per log key.
- Heartbeat: kind `salid_monitor_heartbeat`, guild 0, key `bot` or `platforms`. The doc is `{process, version, tick_at, queues: {actions, outbox_pending, memory}}`, written every `HEARTBEAT_EVERY_S = 30`. States: online when age ≤ 90 s, delayed when 90 s < age ≤ 300 s, offline when age > 300 s or the doc is missing. A future `tick_at` counts as age 0.
- `stuck` thresholds: video step 600 s, deep task `deep_task_timeout` + 120 s (720 s by default), research 900 s, action still pending after 60 s while the bot is online. While the bot is offline, every active item of a bot-run kind (`video_ad`, `deep_task`, `research`, `action`) gets `bot_offline`.
- API: base `/api/center/orgs/{org_id}/monitor`; operations view under `/api/center/admin/monitor`. Lists are `{items, next_cursor}` with the documents offset cursor (`"50"`); `limit` defaults to 50, max 100. `ids=` takes ≤ 50 comma-separated `kind:id` (422 above). Errors are 402 (licence, org routes only), 403, 404 and 429 with `Retry-After`. `changes` is limited to 1 per 2 s per person (token bucket, burst 3) and returns `ETag: "<seq>"`, or 304 on `If-None-Match`.
- Privacy: no API string may contain a prompt, brief, research question, deep-task text, transcript, DM note, message text, action `args`, raw error or stderr, token or person name. Errors use codes from `monitor.ERROR_TEXT`. `audience` never leaves the server. Items carry account keys only (`discord:<id>`, emails for web actions).
- Client: poll every 5 s ± 10 % jitter while visible. Backoff is `[5000, 10000, 20000, 40000, 60000]` ms (cap 60 s) with ±10 % jitter. Show "Reconnecting…" after 2 failures; "offline" after 4 failures or when `navigator.onLine === false`. Heal every 60 s; at most one toast per 10 s; refresh the summary and nav badge every 60 s. A 429 waits `Retry-After` and does not count as a failure.
- `center.html`: exactly one inline `<script>`, no new external hosts. The monitor JS is one marked pure block plus one view. Keys are `mon_*` in both `I.th` and `I.en`. Text uses `--muted`, never `--faint`. Targets are at least 24 px. The row highlight is off under `prefers-reduced-motion`.
- English code, comments and docs; Thai and English UI strings.
- `monitor.py`, `monitor_sources.py` and `jobs.py` never import `discord`.
- Tooling: every `uv` on PATH is old and rewrites `uv.lock`. Locally, always use `uvx --from 'uv>=0.10' uv run --no-sync …` and `uvx --from 'uv>=0.10' uv add --dev …`; CI keeps its own uv. Lint with `uvx ruff check . && uvx ruff format --check .` (line length 100). Two failures in `tests/salid/test_media.py` (no libvorbis) happen only locally: ignore them.
- Node v26.5.0 is installed locally; Node tests skip when `node` is missing (as in `tests/test_hub.py`).
- Offline tests only. The new `e2e` marker is deselected by default: `addopts = "-m 'not live and not e2e'"`.
- Never print secrets. Never run `scripts/salid_outbox_migrate.py --apply`, deploy, or touch production Firestore without human approval.

## Review Focus

1. **A viewer's role changes mid-session.** An admin demoted to member, or removed (403), must lose admin-only items and events on the very next `items?ids=` / `changes` call, and the client stops on 403 (Task 4: `test_demoted_admin_loses_admin_items_and_events_next_poll`; Task 5: `monOnError` cases).
2. **Org with no linked chat workspace, or workspace with no org.** A web-only org gets 200 with empty items, no relevant heartbeat and no `bot_offline`. Events of an unlinked guild go to `guild:<id>` and appear only in the operations view (Task 4: `test_web_only_org_is_empty_and_never_bot_offline`; Task 3: `test_flush_resolves_guild_to_org_or_unlinked_key`).
3. **Firestore emit transaction conflict or store failure.** Concurrent emits keep `seq` strictly increasing with no loss, and a raising `transact` never breaks the producer (Task 2: `test_concurrent_emits_never_lose_or_repeat_seq`, `test_emit_failure_is_counted_and_never_raises`; Task 9: `test_a_failing_log_never_breaks_enqueue_or_the_tick`).
4. **Cursor older than the ring floor after a burst (> 200 events between polls), or ahead of `seq` (log recreated).** The server returns `reset: true`, the client reloads, and duplicates or out-of-order events are ignored (Task 2: `test_changes_since_floor_or_future_resets`; Task 5: Node reducer cases).
5. **Clock skew between bot, web and browser.** A heartbeat `tick_at` in the future counts as online, and durations run on the server's `now` plus local elapsed time, never on the browser clock (Task 1: `test_heartbeat_thresholds_and_clock_skew`; Task 5: `monElapsed` case).

## Spec ambiguities resolved in this plan

1. **Two flags.** The web uses `MONITOR_ENABLED`, as the spec says. The bot and the platforms read `SALID_MONITOR_ENABLED`, because every Salid variable is `SALID_*`.
2. **The outbox re-key is not flag-gated.** One storage layout only. The migration is a dry-run-by-default script that a human approves at deploy.
3. **Per-org outbox key.** It is `org_guild(org_id) = workspace_memory_id("web", org_id)`, because DocStore guild ids are ints. The log and heartbeat stay on guild 0, keyed by string.
4. **Rate limit for several open tabs.** "1 `changes` per 2 s per person" is a token bucket (rate 1 per 2 s, burst 3), so a few tabs polling independently are not starved.
5. **"Admin who is an executive in that workspace".** The web cannot see Discord roles, so this uses the org's mirrored position level (`is_exec_level(Member.level)`, levels 1-3). It fails closed.
6. **Licence gate.** It applies to everyone on org routes, super admins included (the spec text is literal). Only `/api/center/admin/monitor/*` is exempt.
7. **Gaps.** Filtering per viewer makes `seq` gaps normal, so a gap means `reset` from the server (`since < floor` or `since > seq`) or `resp.floor > st.seq` on the client.
8. **`bot_offline`.** It covers every active item of a bot-run kind, queued actions included, not just running ones.
9. **`admin_only` beats `people`.** A demoted admin stops seeing the actions they queued.
10. **Titles.** They are English safe labels; the client shows a localized label by `kind` (`mon_k_*`).
11. **Steps.** Deep tasks and research have no `steps[]`: there is no progress callback today. Video step timings are measured in the bot's progress callback, so `run_video_ad` is unchanged.
12. **Change status precedence.** failed > pending (queued) > sent (waiting) > all acked (done). A change with no targets counts as done.

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `src/salid/monitor.py` | Create | `MonitorItem`/`Step`/`Audience`/`SafeError`, `Viewer`, `visible`/`controls`/`allowed`, attention, error codes, filters/summary/paging, change log (`emit_many`, `changes_since`), `Emitter`, heartbeat |
| `src/salid/monitor_sources.py` | Create | Adapters (record → item) for change, action, video_ad, deep_task, research; source registry `collect`/`find` |
| `src/salid/monitor_bot.py` | Create | `MonitorMixin`: emitter + heartbeat on the bot tick; producer hooks for actions/changes/video/jobs |
| `src/salid/jobs.py` | Create | `JobBook`: `salid_jobs` records + `_active` index + restart sweep |
| `src/salid/jobs_bot.py` | Create | `JobsMixin`: deep-task / research job records, sweep on first `on_ready` |
| `src/meeting_scribe/monitor_api.py` | Create | Monitor routes, operations view, rate limiter, ETag/304 |
| `scripts/salid_outbox_migrate.py` | Create | One-time outbox move guild 0 → per-org guild (dry run by default) |
| `src/salid/change_outbox.py` | Modify | Per-org key `org_guild`, `due/to_alert/prune(org_ids)`, `on_change`, `migrate_legacy` |
| `src/salid/change_outbox_bot.py`, `src/salid/platforms/change_outbox_platform.py`, `src/salid/platforms/runtime.py` | Modify | Pass org ids; platform emitter + heartbeat loop |
| `src/salid/actions.py`, `src/salid/actions_bot.py` | Modify | `on_change` hook, 30-day `prune` |
| `src/salid/videoad.py`, `src/salid/videoad_bot.py` | Modify | Pure step helpers; `steps[]` in job records; emit |
| `src/salid/bot.py`, `src/salid/search_bot.py`, `src/salid/memory.py`, `src/salid/config.py` | Modify | Mixins wired into init/tick/on_ready; job ids; `backlog()`; flag |
| `src/meeting_scribe/config.py`, `app.py`, `center.py`, `static/center.html` | Modify | Flag, routers, emit hooks, `__MONITOR__` placeholder, Monitor section |
| `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml` | Modify | Dev deps, `e2e` marker, e2e CI job |
| `tests/salid/test_monitor.py`, `test_monitor_sources.py`, `test_monitor_bot.py`, `test_jobs.py`, `tests/test_monitor_api.py`, `tests/test_monitor_page.py`, `tests/e2e/*` | Create | Tests per task |
| `CLAUDE.md` | Modify | Code-map rows for the new modules and flags |

---

### Task 1: Monitor model, access rules, attention and error codes

**Files:** Create `src/salid/monitor.py`; create `tests/salid/test_monitor.py`.

**Interfaces:**
- Consumes: `salid.orgs.Org`/`Member.all_identities()` (orgs.py:140-180), `salid.positions.is_exec_level` (positions.py:153).
- Produces:
  - `KINDS`, `STATUSES`, `ACTIVE = ("queued","running","waiting","paused")`, `FINAL = ("done","failed","cancelled")`, `BOT_KINDS`, `ERROR_TEXT: dict[str, str]`.
  - Models: `class Step(BaseModel)` (`name, state="pending", started_at="", ms: int|None=None, attempts=0, error_code=""`); `class Audience(BaseModel)` (`admin_only=False, exec_only=False, dm=False, people: list[str]=[], channel_ids: list[str]=[]`); `class SafeError(BaseModel)` (`code, message`); `class MonitorItem(BaseModel)` with every spec §3.1 field plus `.key -> "kind:id"`.
  - `@dataclass(frozen=True) class Viewer(email: str, role: str, ids: frozenset[str], executive: bool = False)`.
  - Functions: `person_key(platform: str, account_id: str | int) -> str`; `viewer_for(org: Org, email: str, role: str) -> Viewer`; `allowed(kind: str, audience: Audience, viewer: Viewer) -> bool`; `visible(item, viewer) -> bool`; `mine(item, viewer) -> bool`; `controls(item, viewer) -> list[str]`; `safe_error(code: str) -> SafeError`; `heartbeat_state(tick_at: str | None, now: datetime) -> str`; `stuck_after_s(kind: str, deep_timeout_s: int = 600) -> int | None`; `apply_attention(item, now, bot: str = "online", *, deep_timeout_s: int = 600) -> MonitorItem`.

Rules the tests pin:
- `allowed`:
  - `dm` → only when `viewer.ids ∩ people` (even the owner is refused).
  - `owner` and `super` → True.
  - `admin` → True, except that `exec_only` without `executive` falls to the member rule.
  - `admin_only` → only owner, admin or super.
  - `member` → only when `viewer.ids ∩ people`.
- `viewer_for`: `ids = {email.lower()} ∪ {person_key(p, i) …}`. `executive` = owner, or admin with `is_exec_level(member.level)`. A non-member super gets role `super` and `executive` False.
- `controls`:
  - `super` → `[]`.
  - `cancel`: `video_ad`/`deep_task`/`research` in `queued`/`running`, for owner, admin, or a member who is the requester (for `deep_task` the member must also be `executive`).
  - `retry`: a failed `action`, owner or admin only.
  - `resend`: a failed `change`, owner or admin only.
- `apply_attention`, in order:
  1. failed → `failed`.
  2. Bot offline, kind in `BOT_KINDS`, status active → `bot_offline`.
  3. Running and `updated_at` older than `stuck_after_s` → `stuck`.
  4. A queued `action` older than 60 s while the bot is `online` → `stuck`.

  Otherwise keep what the adapter set (changes set their own `stuck`).
- `ERROR_TEXT` keys: `unknown restarted failed error month_limit storyboard render upload_refused not_confirmed delivery_failed missing_permission not_found bad_request discord_error expired`. Each value is one short English sentence.

- [ ] **Step 1: Write the failing tests** in `tests/salid/test_monitor.py`:
```python
NOW = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
V = lambda role, ids=(), ex=False: Viewer("x@y.com", role, frozenset({"x@y.com", *ids}), ex)
@pytest.mark.parametrize("role,ex,kind,aud,ids,ok", [
  ("owner", False, "video_ad", Audience(people=["discord:7"]), (), True),
  ("owner", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False),
  ("admin", False, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), False),
  ("admin", True, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), True),
  ("member", False, "change", Audience(admin_only=True), (), False),
  ("member", False, "action", Audience(admin_only=True, people=["x@y.com"]), (), False),
  ("member", False, "video_ad", Audience(people=["discord:7"]), ("discord:7",), True),
  ("member", False, "video_ad", Audience(people=["discord:8"]), ("discord:7",), False),
  ("super", False, "deep_task", Audience(exec_only=True), (), True),
  ("super", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False)])
def test_visible_roles_by_kinds_matrix(role, ex, kind, aud, ids, ok): ...
def test_controls_matrix():  # owner running video ["cancel"]; owner failed action ["retry"];
    # owner failed change ["resend"]; owner done video []; member own running video ["cancel"];
    # member own running deep_task (not executive) []; member failed action []; super running video []
def test_viewer_for_collects_verified_ids_and_executive_level():
    # Member(discord_user_id="7", identities={"slack": "U1"}, level=3, role="admin") ->
    # ids == {"a@x.com", "discord:7", "slack:U1"} and executive is True; level 5 -> False
def test_heartbeat_thresholds_and_clock_skew():
    ages = {90: "online", 91: "delayed", 300: "delayed", 301: "offline", -120: "online"}
    for age, want in ages.items():
        assert heartbeat_state((NOW - timedelta(seconds=age)).isoformat(), NOW) == want
    assert heartbeat_state(None, NOW) == heartbeat_state("garbage", NOW) == "offline"
def test_attention_rules():  # video running updated 601 s ago -> stuck, 599 -> none; deep_task
    # (deep_timeout_s=600) 721 -> stuck, 719 -> none; research 901 -> stuck; queued action 61 s +
    # bot "online" -> stuck, + "delayed" -> none; running video + bot "offline" -> bot_offline;
    # running change + bot "offline" -> none; failed anything -> ("failed", True)
def test_safe_error_never_echoes_input():
    assert safe_error("discord: SECRET").model_dump() == {"code": "unknown", "message": ERROR_TEXT["unknown"]}
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_monitor.py`. Expected: FAIL (`ModuleNotFoundError: salid.monitor`).
- [ ] **Step 3: Implement** the interfaces above in `src/salid/monitor.py`, with a module docstring that cites the spec. `heartbeat_state`: `age = max(0, (now - parse(tick_at)).total_seconds())`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/monitor.py tests/salid/test_monitor.py && git commit -m "Monitor: item model, access matrix, attention and error codes"`

### Task 2: Change log, emit, viewer-filtered changes, Emitter, heartbeat store, list helpers

**Files:** Modify `src/salid/monitor.py`; modify `tests/salid/test_monitor.py`; modify `CLAUDE.md` (Salid code map: one `monitor.py` row).

**Interfaces:**
- Consumes: `DocStore.transact/get/put/find_items` (coo_store.py:41-73), `OrgStore.org_of_memory_id` (orgs.py:549), `tests/salid/fake_firestore.FakeFirestore`.
- Produces:
  - Constants and stats: `LOG = "salid_monitor_log"`, `HEARTBEAT = "salid_monitor_heartbeat"`, `RING = 200`, `EVENT_V = 1`, `MAX_EVENT_PEOPLE = 20`, `HEARTBEAT_EVERY_S = 30.0`, `EMIT_STATS: collections.Counter`.
  - Log: `log_key(org_id: str | None, memory_id: int) -> str`; `event_for(item: MonitorItem) -> dict`; `emit_many(docs, key: str, events: list[dict], *, now: datetime | None = None) -> int | None`; `emit(docs, key, event, *, now=None) -> int | None`; `emit_items(docs, key: str, items: list[MonitorItem]) -> None` (never raises); `read_log(docs, key) -> dict`.
  - Reading changes: `changes_since(log: dict, since: int, viewer: Viewer) -> dict` (`{seq, floor, events, reset}`); `item_events(log: dict, kind: str, item_id: str) -> list[dict]` (adds `severity`: `error` if failed, `warning` if attention, else `info`).
  - Buffering: `org_key_resolver(docs, ttl_s: float = 60.0, clock=time.monotonic) -> Callable[[int], str]`; `class Emitter(docs, *, enabled: bool, resolve: Callable[[int], str] | None = None, now=_utcnow)` with `add(key, event)`, `add_for_guild(memory_id, event)` and `flush() -> int` (thread-safe via `threading.Lock`; never raises).
  - Heartbeat: `write_heartbeat(docs, process: str, *, queues: dict[str, int], now: datetime) -> bool`; `read_heartbeats(docs) -> dict[str, dict]`; `app_version() -> str` (`K_REVISION` env, else `importlib.metadata.version("meeting-scribe")`, else `"dev"`).
  - List helpers: `filter_items(items, *, kinds=None, statuses=None, attention=None, owner=None, workspace=None, start=None, end=None, q=None) -> list[MonitorItem]`; `paginate(items, cursor: str | None, limit: int) -> tuple[list, str | None]`; `summarize(items, now: datetime, zone: ZoneInfo) -> dict` (`counts`, `by_kind`, `attention`, `failed_24h`, `done_today` by Bangkok date).
- Behaviour:
  - `changes_since` sets `reset = since < floor or since > seq`; events with `seq > since`, sorted by seq, filtered by `allowed(e["kind"], Audience(**e["audience"]), viewer)`.
  - `event_for` trims `people` and `channel_ids` to 20 each.

- [ ] **Step 1: Write the failing tests** (`docs` fixture parametrized over `SQLiteDocStore(":memory:")` and `FirestoreDocStore(client=FakeFirestore())`):
```python
def test_emit_assigns_increasing_seq_and_trims_ring_to_200(docs):
    assert emit_many(docs, "org_t", [ev(i) for i in range(205)], now=NOW) == 205
    log = read_log(docs, "org_t")
    assert (log["seq"], log["floor"], len(log["events"]), log["events"][0]["seq"]) == (205, 5, 200, 6)
    assert set(log["events"][0]) == {"v", "seq", "at", "kind", "id", "status", "attention", "audience"}
def test_changes_since_floor_or_future_resets(docs):  # after 205 events:
    # since=4 -> reset True; since=5 -> reset False and 200 events; since=205 -> []; since=206 -> reset True
def test_concurrent_emits_never_lose_or_repeat_seq(docs):  # 2 threads x 20 emits
    assert sorted(e["seq"] for e in read_log(docs, "org_t")["events"]) == list(range(1, 41))
def test_emit_failure_is_counted_and_never_raises():
    class Broken: transact = staticmethod(lambda fn: (_ for _ in ()).throw(RuntimeError("down")))
    n = EMIT_STATS["failed"]; assert emit(Broken(), "org_t", ev(1)) is None
    assert EMIT_STATS["failed"] == n + 1
def test_changes_are_filtered_per_viewer(docs):  # admin_only hidden from member; people event
    # shown to member with matching id; dm event hidden from owner
def test_event_carries_no_content():  # event_for(item with title/error/22 people) has no title/error, 20 people
def test_emitter_batches_one_transaction_per_key_per_flush():  # 5 adds "org_a" + 3 add_for_guild(77)
    # with resolve=lambda m: "guild:77" -> flush() == 8, transact called twice; enabled=False -> 0 writes
def test_heartbeat_round_trip(docs): ...
def test_filter_paginate_and_summarize():  # 120 items -> paginate(None, 50) -> 50, "50";
    # paginate("100", 50) -> 20, None; summarize counts done_today on the Bangkok date only
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_monitor.py`. Expected: FAIL (`ImportError: emit_many`).
- [ ] **Step 3: Implement.** `emit_many` runs one `docs.transact(fn)` where `fn` reads `LOG/0/key`, appends events with `seq = last + 1` and `at = now iso`, trims to `RING`, sets `floor = events[0]["seq"] - 1`, and puts the doc. On an exception it logs the error type only, increments the counter and returns None. Add the CLAUDE.md row.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/monitor.py tests/salid/test_monitor.py CLAUDE.md && git commit -m "Monitor: change log ring, emit, per-viewer changes, emitter, heartbeat"`

### Task 3: Bot and platform heartbeat + emitter flush (flag `SALID_MONITOR_ENABLED`)

**Files:**
- Create: `src/salid/monitor_bot.py`, `tests/salid/test_monitor_bot.py`.
- Modify:
  - `src/salid/config.py`: add `monitor_enabled: bool = False` near `video_max_jobs` (~line 248).
  - `src/salid/memory.py`: add `MemoryScheduler.backlog()` after `mark_dirty` (~line 211).
  - `src/salid/bot.py`: add `MonitorMixin` to the `SecretaryBot` bases (lines 382-425); call `_monitor_init()` just before `_init_actions()` (~line 501); in `tick` (lines 3500-3520), record `_actions_due()`'s return and end with `await self._monitor_due()`.
  - `src/salid/change_outbox_bot.py`: in `_change_outbox_run`, call `self._monitor_note_queue("outbox_pending", len(mine))`.
  - `src/salid/platforms/runtime.py`: in `run()` (lines 329-368), add `PlatformMonitor` and `monitor_loop`.
  - `CLAUDE.md`: `monitor_bot.py` row; flag in the env list.

**Interfaces:**
- Consumes: `Emitter`, `write_heartbeat`, `org_key_resolver`, `HEARTBEAT_EVERY_S` (Task 2); `self._clock`, `self.s.now`, `self.docs`, `self.memory.scheduler`.
- Produces:
  - `MemoryScheduler.backlog() -> int` (`len(self._entries)` under the lock).
  - `MonitorMixin`:
    - `_monitor_init(self) -> None`: `self.monitor = Emitter(self.docs, enabled=self.s.monitor_enabled, resolve=org_key_resolver(self.docs))`, `self._monitor_beat = float("-inf")`, `self._monitor_queues = {"actions": 0, "outbox_pending": 0, "memory": 0}`.
    - `_monitor_note_queue(self, name: str, n: int) -> None`.
    - `async _monitor_due(self) -> None`: `await asyncio.to_thread(self.monitor.flush)` every tick; the heartbeat `process="bot"` when `self._clock() - self._monitor_beat >= HEARTBEAT_EVERY_S`; never raises; does nothing at all while disabled.
  - `platforms/runtime.py`:
    - `@dataclass class PlatformMonitor(emitter: Emitter, docs, queues: Callable[[], dict[str, int]], clock=time.monotonic, last_beat: float = float("-inf"))`.
    - `async def monitor_tick(m: PlatformMonitor) -> None` (flush, plus the heartbeat `process="platforms"` every 30 s).
    - `async def monitor_loop(m) -> None` (every `TICK_SECONDS`).
    - Started in `run()` only when `s.monitor_enabled and a.identity is not None`, with docs `a.identity.orgs.docs` and `queues = {"actions": 0, "outbox_pending": 0, "memory": a.memory.scheduler.backlog()}`.

- [ ] **Step 1: Write the failing tests** (the `bot` fixture from `test_modes`; `link` from `test_positions_bot`):
```python
async def test_heartbeat_every_30_s_with_queues(bot):
    bot.monitor.enabled = True
    await bot._monitor_due()
    hb = bot.docs.get("salid_monitor_heartbeat", 0, "bot")
    assert hb["process"] == "bot" and set(hb["queues"]) == {"actions", "outbox_pending", "memory"}
    writes = spy(monitor, "write_heartbeat")
    await bot._monitor_due()
    assert writes.n == 0
    bot.clock.advance(30)
    await bot._monitor_due()
    assert writes.n == 1


async def test_disabled_writes_nothing(bot):
    await bot._monitor_due()
    assert bot.docs.get("salid_monitor_heartbeat", 0, "bot") is None


async def test_flush_resolves_guild_to_org_or_unlinked_key(bot):
    bot.monitor.enabled = True
    bot.monitor.add_for_guild(GUILD, ev)
    await bot._monitor_due()
    assert read_log(bot.docs, f"guild:{GUILD}")["seq"] == 1
    link(bot)
    bot._monitor_init()
    bot.monitor.enabled = True  # fresh resolver cache
    bot.monitor.add_for_guild(GUILD, ev)
    await bot._monitor_due()
    assert read_log(bot.docs, "o1")["seq"] == 1


async def test_platform_monitor_tick_writes_platforms_heartbeat_and_flushes(): ...
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_monitor_bot.py`. Expected: FAIL (`AttributeError: 'SecretaryBot' object has no attribute 'monitor'`).
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the same command, then `… pytest -q tests/salid/test_modes.py tests/salid/test_change_outbox.py tests/salid/platforms`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/monitor_bot.py src/salid/config.py src/salid/memory.py src/salid/bot.py src/salid/change_outbox_bot.py src/salid/platforms/runtime.py tests/salid/test_monitor_bot.py CLAUDE.md && git commit -m "Monitor: bot and platform heartbeat, emitter flushed once per tick"`

### Task 4: `monitor_api.py`: summary, items, detail, changes, operations view (flag `MONITOR_ENABLED`)

**Files:**
- Create: `src/salid/monitor_sources.py` (registry only), `src/meeting_scribe/monitor_api.py`, `tests/test_monitor_api.py`.
- Modify:
  - `src/meeting_scribe/config.py`: add `monitor_enabled: bool = False` after `legal_contact_email` (~line 67).
  - `src/meeting_scribe/app.py`: `include_router` after `pm_api.router` (line 57).
  - `CLAUDE.md`: `monitor_api.py` row.

**Interfaces:**
- Consumes:
  - `center.Ctx`, `member_ctx`, `super_user`, `get_orgs` (center.py:385-453).
  - `licensing.license_ok`; `hub.get_salid_runtime` (deep timeout).
  - From monitor: `viewer_for`, `apply_attention`, `filter_items`, `paginate`, `summarize`, `changes_since`, `item_events`, `read_log`, `read_heartbeats`, `heartbeat_state`.
- Produces in `monitor_sources.py`:
  - `Source = Callable[[DocStore, Org, datetime], list[MonitorItem]]`, `Finder = Callable[[DocStore, Org, str], MonitorItem | None]`, `SOURCES: dict[str, tuple[Source, Finder]] = {}`, `HISTORY_MAX = 500`.
  - `register(kind, source, finder) -> None`; `collect(docs, org, now, kinds=None) -> list[MonitorItem]` (a raising source is logged and skipped); `find(docs, org, kind, item_id) -> MonitorItem | None`.
  - `discord_workspaces(org) -> list[Workspace]` (connected Discord workspaces only).
- Produces in `monitor_api.py`:
  - Routers and limits: `router = APIRouter(prefix="/api/center/orgs/{org_id}/monitor")`, `ops_router = APIRouter(prefix="/api/center/admin/monitor")`, `PAGE, PAGE_MAX, IDS_MAX = 50, 100, 50`.
  - `def now() -> datetime` (UTC; tests replace it); `def get_deep_timeout() -> int` (600 on any error); `def monitor_enabled() -> None` (404 when off).
  - `class TokenBucket(rate_s=2.0, burst=3, clock=time.monotonic)` with `take(key: str) -> float` (0.0 = allowed, else seconds to wait); module `LIMITER`.
  - `@dataclass class Mon(ctx, docs, viewer, when, bot: str)`; `async def mon_ctx(ctx=Depends(member_ctx), orgs=Depends(get_orgs), _=Depends(monitor_enabled)) -> Mon` (402 when `license_ok(org.license, now())` is false); `relevant_heartbeats(org, beats, when) -> dict` (`bot` when the org has a Discord workspace, `platforms` when it has slack/line/telegram/whatsapp).
  - `def item_view(item, viewer) -> dict` (`model_dump(exclude={"audience"})` + `mine`).
  - Routes:
    - `GET /summary` → `{counts, by_kind, attention, failed_24h, done_today, heartbeat, now, seq}`.
    - `GET /items` (params `kind: list[str]`, `status: list[str]`, `attention`, `owner`, `workspace`, `from` (alias), `to`, `priority` (accepted; no effect until phase 3), `q`, `ids`, `cursor`, `limit=Query(50, ge=1, le=100)`) → `{items, next_cursor}`.
    - `GET /items/{kind}/{item_id}` → `{**item_view, steps, events, controls}`.
    - `GET /changes?since=` → `{seq, floor, now, events, reset}` + `ETag`; 304; 429 + `Retry-After`.
  - Ops routes: `GET /summary` → `{orgs: [{id, name, counts, attention, seq}], heartbeats, unlinked: [{key, seq, last_at}], now}`; `GET /items` → the same filters across orgs, rows add `org_id`; read-only, no licence gate.

- [ ] **Step 1: Write the failing tests** in `tests/test_monitor_api.py`. Copy the `tests/test_pm_api.py:49-100` pattern: `sys.path` gets `tests/salid`; `docs` is parametrized SQLite / FakeFirestore. Seed org `org_t` with `boss` (owner), `adm` (admin), `mem` (member, `discord_user_id="7"`) and Discord workspace `discord-1000`. Set `monkeypatch.setenv("MONITOR_ENABLED", "true")`. Register a fake source with `monkeypatch.setitem(monitor_sources.SOURCES, "video_ad", (src, finder))`. Add the module guard helper `def strings(obj)` (yields every string value and key, recursively) and `SENTINEL = "SENSITIVE-7f3a"`.
```python
def test_flag_off_is_404(w): ...   # every monitor and ops route
def test_roles_see_their_items(w):  # owner/adm all; mem only people=["discord:7"]; super all, controls []
def test_unlicensed_org_is_402_for_everyone_but_ops_view_works(w): ...
def test_paging_and_ids_batch(w):  # 120 items: 50 + "50"; limit=101 -> 422; 51 ids -> 422;
    # ids with an unknown and an invisible key -> both omitted
def test_detail_404_steps_events_controls(w): ...
def test_changes_etag_304_and_rate_limit(w, monkeypatch):
    r = get("/changes?since=0"); etag = r.headers["etag"]
    assert get("/changes?since=0", headers={"If-None-Match": etag}).status_code == 304
    # fake clock: 3 calls inside 2 s pass (burst 3), the 4th -> 429 with int(Retry-After) >= 1;
    # another person is unaffected
def test_demoted_admin_loses_admin_items_and_events_next_poll(w):
    # adm sees an admin_only item and its event -> owner demotes adm -> next items?ids= omits it and
    # changes?since=0 drops the event; after removal from the org -> 403
def test_web_only_org_is_empty_and_never_bot_offline(w):
    # summary 200, counts {}, heartbeat == {}, items {"items": [], "next_cursor": None}
def test_ops_view_lists_orgs_and_unlinked_logs(w):  # emit to "guild:999" -> unlinked has it; mem -> 403
def test_audience_and_sentinels_never_returned(w):  # item audience.people=["discord:"+SENTINEL]
    assert SENTINEL not in "".join(strings(all_responses)) and "audience" not in item_keys
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/test_monitor_api.py`. Expected: FAIL (`ModuleNotFoundError: meeting_scribe.monitor_api`).
- [ ] **Step 3: Implement.** Do all the store work through `run_in_threadpool`. `items_for(m)` = `collect` → `apply_attention(…, m.bot, deep_timeout_s=get_deep_timeout())` → `visible` → sort by (`updated_at`, `id`) descending. The 304 response carries the same ETag and no body.
- [ ] **Step 4: Run** the same command plus `tests/test_center_smoke.py tests/salid/test_center_api.py`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/monitor_sources.py src/meeting_scribe/monitor_api.py src/meeting_scribe/config.py src/meeting_scribe/app.py tests/test_monitor_api.py CLAUDE.md && git commit -m "Monitor API: summary, items, detail, changes with ETag and rate limit, operations view"`

### Task 5: Client feed: pure functions + Node tests

**Files:** Modify `src/meeting_scribe/static/center.html` (insert a block before `/* ---------- state & routing ---------- */`, ~line 1050); create `tests/test_monitor_page.py`.

**Interfaces** (JS between the exact markers `/* ---------- monitor: pure functions (Node-tested, no DOM) ---------- */` and `/* ---------- monitor: end pure functions ---------- */`):
- Constants: `MON_KINDS` (spec §3.1 list); `MON_BACKOFF_MS = [5000, 10000, 20000, 40000, 60000]`; `MON_JITTER = 0.1`; `MON_HEAL_MS = 60000`; `MON_TOAST_GAP_MS = 10000`; `MON_SUMMARY_MS = 60000`.
- Feed functions:
  - `monBackoff(failures, rnd) -> ms`: `MON_BACKOFF_MS[min(failures, 4)] * (1 - MON_JITTER + 2 * MON_JITTER * rnd)`, rounded.
  - `monFeedState({failures, online, paused, hidden}) -> "paused"|"hidden"|"offline"|"delayed"|"live"`, checked in that order (offline at ≥ 4 failures or `online === false`, delayed at ≥ 2).
  - `monReduce(st, resp) -> {st, refetch, reload}`:
    - a stale response (`resp.seq < st.seq`) → unchanged;
    - `resp.reset || resp.floor > st.seq` → reload;
    - otherwise refetch keys of events with `seq > st.seq` and a known kind, de-duplicated, in seq order.
- Row functions:
  - `monNewer(a, b) -> bool`: `updated_at`, then `_req`.
  - `monMerge(rows, fresh, frozen) -> {rows, added}`: `frozen` = updates in place, no reorder, new keys counted in `added`; otherwise merge and sort by `updated_at` descending, keeping the newer row by `monNewer`.
- Helpers:
  - `monElapsed(startIso, endIso, serverNowIso, fetchedAtMs, nowMs) -> ms`: end, else server now + (nowMs − fetchedAtMs), minus start.
  - `monToastGate(lastAt, nowMs) -> bool`: `nowMs - lastAt >= MON_TOAST_GAP_MS`.
  - `monNotifyGate({optedIn, permission, hidden, mine, isAdmin, status, attention}) -> bool`: true only when `optedIn && permission === "granted" && hidden` and the item concerns the viewer (`mine`, or `isAdmin && status === "failed"`) and is `failed` or `attention`.
  - `monOnError(status, retryAfterS) -> {action, delayMs, failure}`:
    - 401 → `"auth"`;
    - 402 → `"licence"` (stop);
    - 403 → `"forbidden"` (stop);
    - 429 → `"wait"`, `delayMs = retryAfterS * 1000` (default 2000), `failure: false`;
    - anything else, including network errors, → `"retry"` with `failure: true`.
  - `monParseHash(hash) -> state` and `monHash(state) -> "#/monitor?…"`: keys `tab kind status owner workspace from to priority q`, lists comma-separated.

- [ ] **Step 1: Write the failing tests** in `tests/test_monitor_page.py`. Use the `tests/test_hub.py:654-694` pattern: slice the page between the markers, append a harness that prints JSON, and run `node`, skipped when `shutil.which("node") is None`:
```python
assert out["backoff"] == [5000, 10000, 20000, 40000, 60000, 60000, 60000]  # rnd 0.5, n = 0..6
assert out["jitter"] == [4500, 5500]  # n=0, rnd 0 and 1
assert out["feed"] == {
    "f1": "live",
    "f2": "delayed",
    "f4": "offline",
    "net": "offline",
    "paused5": "paused",
    "hidden": "hidden",
}
assert out["reduce_ok"] == {
    "seq": 12,
    "refetch": ["change:c", "video_ad:j"],
    "reload": False,
}  # events 12, 11, 9(dup), zzz(unknown)
assert out["reduce_stale"] == {"seq": 10, "refetch": [], "reload": False}  # resp.seq 8
assert out["reduce_reset"]["reload"] and out["reduce_floor"]["reload"]  # reset; floor 15 > seq 10
assert out["merge_frozen"] == {"order": ["a", "b"], "b_rev": 2, "added": 1}
assert out["merge_free"]["order"] == ["c", "b", "a"] and out["older_ignored"] is True
assert out["elapsed"] == 303000  # 5 min + 3 s, skewed Date
assert out["toast"] == [False, True] and out["hash_roundtrip"] is True
assert out["notify"] == {
    "mine_failed_hidden": True,
    "visible_tab": False,
    "not_opted_in": False,
    "permission_default": False,
    "admin_other_failed": True,
    "member_other_failed": False,
    "mine_done": False,
}
assert out["errors"] == {
    "401": "auth",
    "402": "licence",
    "403": "forbidden",
    "500": "retry",
    "429": ["wait", 7000, False],
}  # Retry-After: 7
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/test_monitor_page.py`. Expected: FAIL (`ValueError: substring not found`, missing marker).
- [ ] **Step 3: Implement** the pure block (no DOM access, no `t()` calls).
- [ ] **Step 4: Run** the same command plus `tests/test_center_page.py`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/meeting_scribe/static/center.html tests/test_monitor_page.py && git commit -m "Monitor page: pure feed functions (backoff, states, reducer, merge) with Node tests"`

### Task 6: Monitor section: view, states, i18n, mock data, a11y wiring

**Files:**
- `src/meeting_scribe/static/center.html`:
  - CSS block `/* monitor:css */` … `/* monitor:css:end */` before line 485; add `.mon-flash` to the reduced-motion rule at line 485.
  - i18n keys in `I.th` (755-846) and `I.en` (847-937).
  - `NAV_SECTIONS` (1053): add `["monitor", "n_monitor", "◉"]` after `overview`, shown only when `MONITOR`.
  - `route()` (1081-1100): `c.focus({ preventScroll: true })` after the heading renders.
  - `VIEWS.monitor` after `VIEWS.mycal` (~2752).
  - `mockApi` monitor routes before `if (sub === "/actions")` (~2938).
  - `const MONITOR = "__MONITOR__" === "1" || MOCK;` after `INVITE_URL` (line 733).
- `src/meeting_scribe/center.py`: `center_page()` (460-478) replaces `__MONITOR__` with `"1"` when `get_settings().monitor_enabled`, else `""`.
- Tests: `tests/test_center_page.py`, `tests/test_center_smoke.py`.

**Interfaces:**
- Consumes: Task 4 routes; Task 5 functions; the existing `el`, `t`, `api`, `toast`, `loading`, `errBox`, `stale`, `handleAuth`, `AuthError`, `ApiError`, `fmtTime`, `.drawer`, `.tabs`, `.tbl.stack`, `.stat`, `.sr`.
- Produces:
  - Fetching: `async function monFetch(path, etag) -> {status, body, etag, retryAfter}`. It calls `fetch` with `Authorization` and `If-None-Match` (the existing `api()` drops custom headers); under `MOCK` it calls `mockApi`; 401 → `throw new AuthError("expired")`.
  - The view: `VIEWS.monitor = async (body, my) => {…}` builds:
    - `loading()` skeleton first;
    - a health strip (feed state text, `title` with the exact Bangkok time, bot state, Pause live updates, notifications toggle);
    - five tiles as buttons: Running, Queued, Needs attention, Failed (24 h), Done today; a click applies that tile's filter;
    - tabs (Needs attention is the default when non-empty, then Active, History);
    - filters (select/inputs) mirrored to the hash via `history.replaceState`, as removable chips; under 640 px they collapse into a `<details>` sheet;
    - rows (status icon + text, title, owner, `done/total` steps or "waiting on … · 2 d", live duration, last update);
    - "Load more"; an "N new" pill;
    - `<div id="mon-live" class="sr" aria-live="polite">` (one summary per batch);
    - for super admins, an "Operations" tab calling `/api/center/admin/monitor/*`;
    - nav badge: `renderNav` shows `summary.attention` on the Monitor link (`aria-label` includes the count), refreshed every `MON_SUMMARY_MS` while the tab is visible.
  - Behaviour:
    - `monDrawer(item, opener)` is `role="dialog"` with `aria-modal`; a Tab/Shift+Tab trap; Escape closes and focus returns to `opener`. It shows the step timeline (duration, attempts), the safe error + `mon_do_*` hint, cost, links (`safeUrl`), and the event list with search, severity filter and Copy. `controls` are not rendered (release 2).
    - Poller: a `setTimeout` loop using `monBackoff`/`monFeedState`/`monReduce`/`monMerge`. It pauses when `document.hidden` (fetches at once on `visibilitychange`), stops when `stale(my)` or the body is detached, resumes on `online`, and heals every 60 s. Errors go through `monOnError`: 403 → member explanation and stop; 402 → licence notice and stop; 429 → wait `Retry-After`; 401 → `handleAuth`. It keeps the last data with a delayed/offline banner and shows a bot-offline banner.
    - Dismiss: `localStorage["salid-monitor-dismissed"] = {key: updated_at}`; the item returns when `updated_at` changes.
    - Toasts only for `mine` items (failures for admins), grouped, gated by `monToastGate`. OS notifications go through `monNotifyGate`. `Notification.requestPermission()` is called only from the notifications-toggle click (never on load), the opt-in is stored in `localStorage["salid-monitor-notify"]`, and there is one `new Notification(...)` per batch with a safe title only; clicking it focuses the tab and opens the item's drawer.
  - i18n keys (EN shown; Thai written for each key in the page's tone, `ค่ะ`):
    - `n_monitor` ("Monitor" / "ติดตามงาน");
    - feed: `mon_live mon_reconnecting mon_offline mon_paused mon_updated mon_pause mon_resume mon_notify`;
    - bot: `mon_bot_online mon_bot_delayed mon_bot_offline mon_banner_bot mon_banner_delayed`;
    - tiles: `mon_t_running mon_t_queued mon_t_attention mon_t_failed24 mon_t_done_today`;
    - tabs: `mon_tab_attention mon_tab_active mon_tab_history`;
    - filters: `mon_f_kind mon_f_status mon_f_owner mon_f_workspace mon_f_from mon_f_to mon_f_priority mon_f_q mon_chip_remove`;
    - list: `mon_new mon_load_more mon_empty mon_member_note mon_steps mon_waiting_on`;
    - per value: `mon_k_<11 kinds>`, `mon_s_<7 statuses>`, `mon_a_<5 reasons>`, `mon_err_<ERROR_TEXT codes>`, `mon_do_default`, `mon_do_restarted`, `mon_do_missing_permission`;
    - drawer: `mon_d_steps mon_d_events mon_d_copy mon_d_search mon_d_sev_all mon_d_sev_info mon_d_sev_warning mon_d_sev_error mon_d_cost`;
    - other: `mon_dismiss mon_licence mon_forbidden mon_view mon_toast mon_announce mon_ops mon_ops_unlinked`.
  - Mock data: `/monitor/summary`; `/monitor/items` (a failed change with `not_confirmed` and attention; a running video ad with 2 done, 1 running and 2 pending steps; a queued action with `stuck`; a done research; a running deep task); `/monitor/items/{kind}/{id}`; `/monitor/changes` → `{seq: 7, floor: 0, now, events: [], reset: false}`.

- [ ] **Step 1: Write the failing tests:**
```python
# tests/test_center_page.py
def test_monitor_nav_view_and_placeholder():
    s = html(); assert '["monitor", "n_monitor"' in s and "VIEWS.monitor = async" in s and "__MONITOR__" in s
def test_monitor_keys_in_both_languages():
    th, en = html().split("\nen: {", 1)
    for k in set(re.findall(r"\b(mon_[a-z0-9_]+):", html())): assert f"{k}:" in th and f"{k}:" in en, k
def test_monitor_css_is_accessible():
    css = between("/* monitor:css */", "/* monitor:css:end */")
    assert "--faint" not in css and "min-height: 24px" in css
    assert re.search(r"prefers-reduced-motion[^}]*\.mon-flash", html())
def test_monitor_live_region_and_dialog():
    s = html(); assert 'id: "mon-live"' in s and '"aria-live": "polite"' in s and 'role: "dialog"' in s
# tests/test_center_smoke.py
def test_center_page_sets_the_monitor_flag(client, monkeypatch):  # MONITOR_ENABLED=true -> 'MONITOR = "1"'; unset -> '"" === "1"'
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/test_center_page.py tests/test_center_smoke.py`. Expected: FAIL (assertions).
- [ ] **Step 3: Implement** the interfaces above. Keep `src.count("<script>") == 1`.
- [ ] **Step 4: Run** the same command plus `tests/test_monitor_page.py`, then open `/center?mock=1#/monitor` locally (`STORE=memory uvx --from 'uv>=0.10' uv run --no-sync uvicorn meeting_scribe.app:app --port 8080`) and check the keyboard path by hand. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/meeting_scribe/static/center.html src/meeting_scribe/center.py tests/test_center_page.py tests/test_center_smoke.py && git commit -m "Monitor page: live section with health strip, tiles, filters, drawer, i18n and mock data"`

### Task 7: Change outbox per-org re-key + one-time migration

**Files:**
- Modify:
  - `src/salid/change_outbox.py`: `GUILD` (line 39) and the `ChangeOutbox` class (194-401).
  - `src/salid/change_outbox_bot.py`: `_change_outbox_run` (128-161) and `_change_alerts` (206-227).
  - `src/salid/platforms/change_outbox_platform.py`: `_run` (67-105) and `_alerts` (107-131).
  - `tests/salid/test_change_outbox.py`: every `ob.due()` / `to_alert(p)` / `prune()` call gets `["o1"]`.
  - `tests/salid/platforms/test_change_outbox_platform.py`.
- Create: `scripts/salid_outbox_migrate.py`.

**Interfaces:**
- Consumes: `orgs.workspace_memory_id` (orgs.py:99), `OrgStore.org_of_guild` / `org_of_workspace` (orgs.py:537-547), `DocStore.find_any` / `transact`.
- Produces in `change_outbox.py`:
  - Keys: `LEGACY_GUILD = 0`; `def org_guild(org_id: str) -> int` (`workspace_memory_id("web", org_id)`).
  - Reading: `ChangeOutbox._org_changes(org_id: str) -> list[Change]` (replaces `_all`); `get(change_id, org_id: str = "")` (empty → `find_any(KIND, change_id)`).
  - Updates: `_update(change_id, fn, org_id="")`; `ack(…, org_id="")`, `mark_attempt(…, org_id="")`, `mark_alerted(change_id, platform, org_id="")`.
  - Delivery: `open_for(workspace_id, person)` keeps its signature and resolves the org through `OrgStore(self.docs).org_of_workspace(platform, external_id)` (None → `[]`); `due(org_ids: Iterable[str])`; `to_alert(platform, org_ids)`; `prune(org_ids, force=False)`.
  - `record(changes, org)` dedupes and stores under `org_guild(org.id)`.
  - Migration: `def migrate_legacy(docs, *, apply: bool) -> dict[str, int]` → `{"found", "movable", "moved", "skipped"}`. Each move is one `transact` (put under `org_guild`, delete guild 0); a doc without `org_id` is skipped.
- Consumers:
  - Discord: `ChangeOutboxMixin._change_org_ids(self) -> set[str]` (via `to_thread`, `org_of_guild` per guild).
  - Platforms: `PlatformChanges._org_ids(self) -> set[str]` (`org_of_workspace` per pusher).
  - Both pass `org_id=change.org_id` to `ack`/`mark_attempt`/`mark_alerted`.
- Script: `scripts/salid_outbox_migrate.py [--apply]`, docstring in the `salid_people_backfill.py` style. It prints the counts only.

- [ ] **Step 1: Write the failing tests** (in `tests/salid/test_change_outbox.py`):
```python
def test_changes_are_stored_per_org_and_due_reads_only_those_orgs():
    docs = FirestoreDocStore(client=FakeFirestore())
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    assert docs.get(co.KIND, co.org_guild("o1"), c.id) and docs.get(co.KIND, 0, c.id) is None
    docs.client.streams.clear()
    ob.due(["o1"])
    assert {s for s in docs.client.streams if s[0] == co.KIND} == {
        (co.KIND, (("guild_id", "==", co.org_guild("o1")),))
    }


def test_get_without_org_uses_the_change_id_alone(docs): ...  # the button click path
def test_open_for_resolves_the_org_of_the_workspace(docs): ...  # unlinked workspace -> []
def test_migrate_legacy_moves_once_and_skips_orphans(docs):
    # 2 legacy docs (orgs o1, o2) + 1 without org_id at guild 0
    assert migrate_legacy(docs, apply=False) == {"found": 3, "movable": 2, "moved": 0, "skipped": 1}
    assert migrate_legacy(docs, apply=True)["moved"] == 2
    assert migrate_legacy(docs, apply=True) == {"found": 1, "movable": 0, "moved": 0, "skipped": 1}
    assert ChangeOutbox(docs).list_changes("o2")  # readable after the move
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_change_outbox.py tests/salid/platforms/test_change_outbox_platform.py`. Expected: FAIL (`AttributeError: module 'salid.change_outbox' has no attribute 'org_guild'`).
- [ ] **Step 3: Implement** the interfaces above. `center.list_changes` (center.py:1176-1185) is unchanged: `list_changes(org_id)` now reads one org.
- [ ] **Step 4: Run** the same command plus `tests/salid/test_center_api.py`. Expected: PASS (existing drig and platform tests are green with org ids passed).
- [ ] **Step 5: Commit:** `git add src/salid/change_outbox.py src/salid/change_outbox_bot.py src/salid/platforms/change_outbox_platform.py scripts/salid_outbox_migrate.py tests/salid/test_change_outbox.py tests/salid/platforms/test_change_outbox_platform.py && git commit -m "Change outbox: one key per org, consumers read only their orgs, legacy migration script"`

Deploy note for the PR body: run `scripts/salid_outbox_migrate.py` (dry run), then `--apply` with human approval right after deploy. In-flight legacy changes resend every 15 min and fail after ~2 h 15 min, so the window is safe.

### Task 8: Delivery adapters: change and action sources

**Files:** Modify `src/salid/monitor_sources.py`; create `tests/salid/test_monitor_sources.py`; modify `tests/test_monitor_api.py`.

**Interfaces:**
- Consumes: `Change`/`Target`, `ChangeOutbox.list_changes`, `RESEND_MINUTES`/`MAX_ATTEMPTS` (change_outbox.py:39-45); `ActionQueue.recent` (actions.py:73-75); `monitor.safe_error`/`person_key`; `register`/`discord_workspaces` (Task 4).
- Produces:
  - `def action_error_code(error: str) -> str`:
    - `"missing permission"` → `missing_permission`;
    - `expired…` → `expired`;
    - `channel not found…` → `not_found`;
    - `discord:…` → `discord_error`;
    - `"bad request"` → `bad_request`;
    - any other non-empty text → `error`;
    - empty → `""`.
  - `def adapt_change(change: Change, *, now: datetime, resend_minutes: int = RESEND_MINUTES, max_attempts: int = MAX_ATTEMPTS) -> MonitorItem`:
    - status precedence: any `failed` → failed, else any `pending` → queued, else any `sent` → waiting, else done (no targets = done);
    - steps = one per target, `name=f"{platform} {kind}"`, state `pending→pending`, `sent→running`, `acked→done`, `failed→failed`, `attempts`, `started_at=sent_at`, `error_code` = `not_confirmed` when `last_error == "not confirmed in time"`, else `delivery_failed`;
    - `stuck` when still open after `at + resend*(max_attempts+1)`;
    - `title=f"Change · {kind}"`; `raw_status=f"{acked}/{n} acked"`;
    - `audience=Audience(admin_only=True)`; `workspace="web"`;
    - no subject email or name.
  - `def adapt_action(rec: dict) -> MonitorItem`:
    - status `pending→queued`, `done→done`, `failed→failed`;
    - `title=f"Discord action · {kind}"`;
    - `requester_id=rec["by"].lower()`; `audience=Audience(admin_only=True, people=[by])`;
    - `workspace=f"discord-{guild_id}"`;
    - `links={"discord": url}` only for `https://discord.com/` URLs;
    - never `args` or the raw `error`.
  - Registered sources `changes_source`/`changes_finder` and `actions_source`/`actions_finder`. The finders find an action by id across `discord_workspaces(org)` and a change with `get(id, org_id=org.id)`.

- [ ] **Step 1: Write the failing tests:**
```python
def test_change_status_precedence_and_steps():  # targets (acked, failed) -> failed + attention;
    # (pending, sent) -> queued; (sent, acked) -> waiting; () -> done; steps[1].error_code == "not_confirmed"
def test_change_open_past_deadline_is_stuck():  # at=T0, now=T0+15*(8+1) min -> attention_reason "stuck"
def test_action_mapping_and_error_codes():
    assert [action_error_code(e) for e in ("missing permission", "expired before the bot could do it",
        "channel not found: x", "discord: 50013", "bad request", "KeyError", "")] == [
        "missing_permission", "expired", "not_found", "discord_error", "bad_request", "error", ""]
# tests/test_monitor_api.py
def test_delivery_items_through_the_api_hide_raw_text(w):
    # action args {"text": SENTINEL}, error "discord: " + SENTINEL; change target last_error SENTINEL,
    # subject_email SENTINEL+"@x.com" -> owner sees both kinds, mem sees neither, SENTINEL in no string
def test_pending_action_is_stuck_after_60_s_only_while_the_bot_is_online(w): ...
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_monitor_sources.py tests/test_monitor_api.py`. Expected: FAIL (`ImportError: adapt_change`).
- [ ] **Step 3: Implement** the adapters and register the sources when the module is imported.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/monitor_sources.py tests/salid/test_monitor_sources.py tests/test_monitor_api.py && git commit -m "Monitor: change outbox and action queue adapters"`

### Task 9: Emit from delivery producers + action pruning (30 d)

**Files:**
- `src/salid/actions.py`: the whole file (104 lines).
- `src/salid/actions_bot.py`: `_init_actions` (39-40), `_actions_due` (42-66).
- `src/salid/change_outbox.py`: `ChangeOutbox.__init__`/`from_settings`/`record`/`_update`, `record_org_edit` (405-418).
- `src/salid/change_outbox_bot.py`: `_change_outbox_init` (72-75).
- `src/salid/monitor_bot.py`; `src/salid/platforms/runtime.py`: `run()` (342-346).
- `src/meeting_scribe/center.py`: the 6 `record_org_edit` calls (1057, 1104, 1138, 1156, 1585, 1615), `discord_action_tools` (2728-2790), `queue_action` (2806-2823).
- Tests: `tests/salid/test_monitor_bot.py`, `tests/test_monitor_api.py`.

**Interfaces:**
- Consumes: `adapt_action`, `adapt_change` (Task 8); `Emitter`, `event_for`, `emit_items` (Task 2).
- Produces in `actions.py`:
  - `KEEP_DAYS = 30`, `PRUNE_EVERY_S = 3600.0`.
  - `ActionQueue.__init__(docs, clock=None, on_change: Callable[[dict], None] | None = None)`: called after `enqueue` and `finish`; a hook error is logged and swallowed.
  - `ActionQueue.prune(guild_id: int, *, force: bool = False) -> int`: deletes non-pending records whose `created_at` is older than 30 days; at most once per hour per guild per instance.
  - `_actions_due` calls `self.actions.prune(guild.id)` for each guild.
- Produces in `change_outbox.py`:
  - `ChangeOutbox(…, on_change: Callable[[list[Change]], None] | None = None)`: called with the stored list after `record` and `[c]` after a successful `_update` (outside the transaction function); swallowed on error.
  - `from_settings(docs, settings, now=None, on_change=None)`; `record_org_edit(docs, before, after, by="salid", settings=None, on_change=None)`.
- `MonitorMixin` hooks:
  - `_monitor_action(self, rec: dict) -> None` → `add_for_guild(rec["guild_id"], event_for(adapt_action(rec)))`.
  - `_monitor_changes(self, changes: list[Change]) -> None` → `add(c.org_id, event_for(adapt_change(c, now=…)))`.
  - Wired in `_init_actions` and `_change_outbox_init`.
- `center.py`:
  - `def _monitor_hook(docs, org_id: str, adapt) -> Callable | None` returns `None` when `get_settings().monitor_enabled` is false, else a callable doing `monitor.emit_items(docs, org_id, [adapt(x) for x in xs])`.
  - `def _record_edit(orgs, before, after, by) -> int` replaces the 6 calls.
  - `queue_action` and `discord_action_tools` build `ActionQueue(docs, on_change=…)`.
- `runtime.run()`: the platform `ChangeOutbox.from_settings(…, on_change=…)` adds to the Task 3 platform emitter.

- [ ] **Step 1: Write the failing tests:**
```python
async def test_action_lifecycle_emits_queued_then_done(bot):  # enqueue + _actions_due with a fake guild
    # (test_center_api.py:1935 pattern), then _monitor_due -> statuses in seq order == ["queued", "done"]
async def test_outbox_updates_emit_change_events(drig):  # web_edit + tick + _monitor_due -> log "o1"
    # has kind "change" events, statuses end with "waiting" after the notice is sent
def test_prune_keeps_30_days_and_pending():  # done 31 d old -> gone; done 29 d -> kept; pending 31 d -> kept;
    # second prune() within 3600 s -> 0 without reading; force=True reads again
async def test_a_failing_log_never_breaks_enqueue_or_the_tick(bot, monkeypatch):  # transact raises
    # -> enqueue returns the record, _actions_due finishes it, _monitor_due does not raise
# tests/test_monitor_api.py
def test_web_enqueue_and_member_edit_emit_to_the_org_log(w):  # POST /actions -> log org_t has
    # an action "queued" event; PUT member role -> a change event; MONITOR_ENABLED unset -> no log doc
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_monitor_bot.py tests/test_monitor_api.py`. Expected: FAIL (`TypeError: unexpected keyword argument 'on_change'`).
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the same command plus `tests/salid/test_change_outbox.py tests/salid/test_center_api.py tests/test_hub.py tests/salid/platforms`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/actions.py src/salid/actions_bot.py src/salid/change_outbox.py src/salid/change_outbox_bot.py src/salid/monitor_bot.py src/salid/platforms/runtime.py src/meeting_scribe/center.py tests/salid/test_monitor_bot.py tests/test_monitor_api.py && git commit -m "Monitor: delivery queues emit their changes; action records pruned after 30 days"`

### Task 10: Video jobs persist `steps[]` and emit

**Files:**
- Modify:
  - `src/salid/videoad.py`: pure helpers after `_report` (~line 1207).
  - `src/salid/videoad_bot.py`: `VideoJob` (97-112), the `progress` closure in `_video_steps` (331-336), `_video_finish` (433-456), `_video_save` (509-533), `_sweep_video_jobs` (595-650).
  - `src/salid/monitor_sources.py`.
  - `tests/salid/test_videoad_bot.py`, `tests/salid/test_monitor_sources.py`, `tests/test_monitor_api.py`.

**Interfaces:**
- Consumes: `videoad.STEPS` (line 920), `RETURNED`; the `runner` fixture (test_videoad_bot.py:110-142); `MonitorMixin.monitor` (Task 3).
- Produces in `videoad.py`:
  - `def step_started(steps: list[dict], step: str, *, at: str, elapsed_ms: int | None) -> bool`: closes the running step as done with `elapsed_ms` and appends `{name, state: "running", started_at: at, ms: None, attempts: 1, error_code: ""}`. It returns False (no change) for a step not in `STEPS` or equal to the running one (a repeated `hero`, `RETURNED`).
  - `def steps_closed(steps: list[dict], state: str, *, elapsed_ms: int | None, error_code: str = "") -> None`: the running step becomes done, failed or cancelled.
- `videoad_bot.py`:
  - `VideoJob.steps: list[dict]` and `VideoJob.step_t0: float`.
  - The progress closure calls `step_started(job.steps, step, at=self.s.now().isoformat(timespec="seconds"), elapsed_ms=…)` before `_video_save`.
  - `_video_finish` calls `steps_closed(job.steps, state, …, error_code=reason)` before the final record.
  - `_video_save` writes `"steps": job.steps` and, after a successful put, `self.monitor.add_for_guild(job.guild_id, event_for(adapt_video(job.job_id, rec)))`.
  - The sweep closes the record's steps with `("failed", "restarted")` and emits.
- `monitor_sources.py`:
  - `VIDEO_JOBS = "salid_video_jobs"`.
  - `def adapt_video(job_id: str, rec: dict) -> MonitorItem`:
    - state `queued→queued`, `STEPS→running`, `done→done`, `failed→failed`, `cancelled→cancelled`, `returned→cancelled` if `reason == "cancelled"` else failed;
    - every `STEPS` name present (missing ones `pending` while active, `skipped` once final);
    - `requester_id=person_key("discord", requester_id)`; `audience.people=[that]`; `cost_usd`; `error=safe_error(reason)` when failed;
    - `links={"discord": f"https://discord.com/channels/{gid}/{channel_id}"}`; `title="Video ad"`.
  - Source/finder over `discord_workspaces(org)` via `find_items(VIDEO_JOBS, ws.memory_id)`.

- [ ] **Step 1: Write the failing tests:**
```python
async def test_a_job_records_its_steps_with_timings(on, world, runner):  # delivered job ->
    rec = records(on)[0]
    assert [s["name"] for s in rec["steps"]] == ["storyboard", "stills", "audio", "render"]
    assert {s["state"] for s in rec["steps"]} == {"done"} and all(isinstance(s["ms"], int) for s in rec["steps"])
async def test_a_failed_render_marks_the_running_step_failed(on, world, runner):  # runner.error = RuntimeError("x")
    assert records(on)[0]["steps"][-1] == {**ANY_STEP, "name": "render", "state": "failed", "error_code": "render"}
async def test_video_progress_emits_in_order(on, world, runner):  # on.monitor.enabled = True -> after the
    # job + _monitor_due: statuses of kind "video_ad" in seq order start "queued", contain "running", end "done"
async def test_the_sweep_closes_steps_as_restarted_and_emits(bot, world): ...
def test_video_constant_matches_the_bot(): assert monitor_sources.VIDEO_JOBS == videoad_bot.JOBS
def test_adapt_video_mapping():  # "returned"+reason "" -> failed; "stills" -> running with 5 steps;
    # done -> hero "skipped"; failed "upload_refused" -> error.code == "upload_refused"
def test_running_video_is_stuck_after_10_min(w): ...   # API, updated_at 601 s before now()
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_videoad_bot.py tests/salid/test_monitor_sources.py tests/test_monitor_api.py`. Expected: FAIL (`KeyError: 'steps'`).
- [ ] **Step 3: Implement** the interfaces above (`videoad.py` stays free of Discord and of the monitor).
- [ ] **Step 4: Run** the same command plus `tests/salid/test_videoad.py`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/videoad.py src/salid/videoad_bot.py src/salid/monitor_sources.py tests/salid/test_videoad_bot.py tests/salid/test_monitor_sources.py tests/test_monitor_api.py && git commit -m "Monitor: video jobs persist steps[] and emit their progress"`

### Task 11: `salid_jobs` for deep tasks and deep research, with restart sweep

**Files:**
- Create: `src/salid/jobs.py`, `src/salid/jobs_bot.py`, `tests/salid/test_jobs.py`.
- Modify:
  - `src/salid/bot.py`: `_DeepJob` (326-331), the bases (382-425), init (~line 600 `_jobs_init()` after `_change_outbox_init()`), `on_ready` (628-639), `_launch_deep` (2748-2762), `_run_deep` (2863-2904).
  - `src/salid/search_bot.py`: `_ResearchJob` (42-46), `_launch_research` (193-208), `_run_research` (210-266).
  - `src/salid/monitor_sources.py`, `src/salid/monitor_bot.py`, `CLAUDE.md` (rows).

**Interfaces:**
- Consumes: `DocStore.transact/get/put/find_items`; `MonitorMixin.monitor`; `person_key`, `safe_error`, `register`.
- Produces in `jobs.py`:
  - `KIND = "salid_jobs"`, `INDEX_GUILD, INDEX_KEY = 0, "_active"`, `KINDS = ("deep_task", "research")`, `FINAL = frozenset({"done", "failed", "cancelled", "restarted"})`.
  - `def new_id() -> str` (12 hex chars).
  - `class JobBook(docs, now: Callable[[], datetime], on_change: Callable[[dict], None] | None = None)` with:
    - `start(kind, *, job_id, guild_id, channel_id, requester_id, dm=False) -> bool`: index entry first (one `transact`), then the record `{job_id, kind, guild_id, channel_id, requester_id, dm, state: "running", created_at, updated_at, finished_at: "", reason: ""}`.
    - `finish(guild_id, job_id, state, reason="") -> bool`: record, then out of the index.
    - `active() -> dict[str, int]`.
    - `sweep(is_live: Callable[[str], bool]) -> list[dict]`: non-live, non-final records become `restarted` with `reason="restarted"`; ended or missing ones just leave the index.

    It never stores task text or the research question and never raises (False / `[]` on store errors).
- `jobs_bot.py`, `class JobsMixin` with:
  - `_jobs_init(self)` (`self.jobs = JobBook(self.docs, self.s.now, on_change=self._monitor_job)`, `self._jobs_swept = False`);
  - `async _job_start(self, kind, job_id, message, guild_id) -> None`;
  - `async _job_end(self, guild_id, job_id, state, reason="") -> None`;
  - `_job_live(self, job_id) -> bool` (any unfinished `_deep_jobs` / `_research_jobs` entry with that `job_id`);
  - `async _sweep_jobs(self) -> None` (once per process).
- Bot changes:
  - `_DeepJob.job_id: str = ""` and `_ResearchJob.job_id: str = ""`, set in `_launch_*`.
  - `_run_deep(…, job_id="")` and `_run_research(…, job_id="")` start the record first and end it in `finally`:
    - deep task: success → done; `CancelledError`/`DeepCancelled` → cancelled; `DeepTaskError` → failed `"failed"`; `Exception` → failed `"error"`;
    - research: `ResearchCancelled` → cancelled; `Exception` → failed `"error"`.
  - `on_ready` awaits `self._sweep_jobs()`.
- `MonitorMixin._monitor_job(self, rec: dict) -> None` → `add_for_guild(rec["guild_id"], event_for(adapt_job(rec)))`.
- `monitor_sources.adapt_job(rec) -> MonitorItem`:
  - state `running→running`, `done→done`, `failed`/`restarted→failed` (error `reason`), `cancelled→cancelled`;
  - title `"Deep task"` / `"Deep research"`; `audience=Audience(exec_only=kind == "deep_task", dm=rec["dm"], people=[person_key("discord", requester)])`;
  - sources for `deep_task` and `research` over `discord_workspaces(org)`.

- [ ] **Step 1: Write the failing tests:**
```python
def test_job_lifecycle_and_index(docs):  # start -> active() == {"j1": 1000}, record "running";
    # finish(done) -> active() == {}, finished_at set
def test_sweep_marks_cut_off_jobs_restarted_and_skips_live(docs):
    # j1 running (not live) -> "restarted"; j2 running + live -> untouched; j3 done -> leaves the index
def test_records_never_hold_the_task_text(bot, world, monkeypatch):  # deep task text SENTINEL ->
    assert SENTINEL not in json.dumps(bot.docs.find("salid_jobs", GUILD))
async def test_research_job_is_recorded_and_ends_done(bot, world):  # test_search.py:527 pattern
    assert [r["state"] for r in bot.docs.find("salid_jobs", GUILD)] == ["done"]
async def test_cancelled_research_ends_cancelled(bot, world): ...  # test_search.py:593 pattern
def test_deep_tasks_visible_to_executives_only(w):  # API: adm (level 0) does not see the deep task,
    # adm with Member.level=3 sees it, boss sees it, mem who requested it sees it; a dm job is hidden from boss
def test_job_stuck_thresholds(w):  # deep task updated 721 s ago -> stuck; research 901 s -> stuck
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q tests/salid/test_jobs.py tests/test_monitor_api.py`. Expected: FAIL (`ModuleNotFoundError: salid.jobs`).
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the same command plus `tests/salid/test_search.py tests/salid/test_message_control.py tests/salid/test_salid_files.py`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add src/salid/jobs.py src/salid/jobs_bot.py src/salid/bot.py src/salid/search_bot.py src/salid/monitor_sources.py src/salid/monitor_bot.py tests/salid/test_jobs.py tests/test_monitor_api.py CLAUDE.md && git commit -m "Monitor: salid_jobs records for deep tasks and research, restart sweep"`

### Task 12: Playwright smoke + accessibility scan + CI job

**Files:**
- Create: `tests/e2e/__init__.py` (empty), `tests/e2e/conftest.py`, `tests/e2e/test_monitor_smoke.py`.
- Modify:
  - `pyproject.toml`: `[dependency-groups].dev` and `[tool.pytest.ini_options]` (lines 46-54).
  - `uv.lock`.
  - `.github/workflows/ci.yml`: a new `e2e` job after `test` (~line 49).
- Verified packages:
  - `pytest-playwright==0.10.0` (Oct 2026; requires `playwright>=1.18`; supplies the session `base_url` fixture itself when pytest-base-url is absent, plus `page` and `browser_context_args`);
  - `axe-playwright-python==0.1.8` (Jul 2026; `from axe_playwright_python.sync_playwright import Axe`, `Axe().run(page, context=None, options={"resultTypes": ["violations"]}) -> AxeResults` with `.violations_count`, `.generate_report()`, `.response`; bundles axe-core 4.12.1);
  - resolves `playwright` 1.64.0.

**Interfaces:**
- Consumes: Task 6 page, mock routes and markup (`#mon-live`, dialog, tiles as buttons).
- Produces:
  - `tests/e2e/conftest.py`:
    - `@pytest.fixture(scope="session") def live_server() -> Iterator[str]`: `os.environ.setdefault("STORE", "memory")`, a free port, `uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))` in a daemon thread, wait up to 10 s for `server.started`; teardown `should_exit = True` and `join(5)`.
    - `@pytest.fixture(scope="session") def base_url(live_server) -> str` (overrides pytest-playwright's).
    - An autouse `block_external(page)` that aborts `https://accounts.google.com/**` and `https://fonts.*/**`.
  - `pytestmark = pytest.mark.e2e`.

- [ ] **Step 1: Write the failing tests** (and `tests/e2e/conftest.py` per Interfaces) in `tests/e2e/test_monitor_smoke.py`:
```python
def test_keyboard_path_filters_rows_and_drawer(page):
    page.goto("/center?mock=1#/monitor")
    expect(page.get_by_role("heading", level=1, name="Monitor")).to_be_visible()
    expect(page.locator("#content")).to_be_focused()  # route focus
    page.get_by_role("combobox", name="Kind").focus()
    page.keyboard.press("ArrowDown")
    expect(page.get_by_role("button", name=re.compile("Remove filter"))).to_be_visible()  # chip
    row = page.get_by_role("button", name=re.compile("Video ad"))
    row.focus()
    page.keyboard.press("Enter")
    dialog = page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    for _ in range(12):
        page.keyboard.press("Tab")
    assert page.evaluate("document.querySelector('[role=dialog]').contains(document.activeElement)")
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    expect(row).to_be_focused()


def test_tiles_and_live_region_have_names(page):
    page.goto("/center?mock=1#/monitor")
    expect(page.get_by_role("button", name=re.compile("Needs attention"))).to_be_visible()
    expect(page.locator("#mon-live")).to_have_attribute("aria-live", "polite")


WCAG = {
    "runOnly": {"type": "tag", "values": ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]},
    "resultTypes": ["violations"],
}


def test_axe_finds_no_wcag_aa_violations_in_the_monitor(page):
    page.goto("/center?mock=1#/monitor")
    page.get_by_role("button", name=re.compile("Video ad")).click()
    for ctx in ("#content", "[role=dialog]"):
        res = Axe().run(page, context=ctx, options=WCAG)
        assert res.violations_count == 0, res.generate_report()
```
- [ ] **Step 2: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q -m e2e tests/e2e`. Expected: FAIL (`fixture 'page' not found`: pytest-playwright is not installed yet).
- [ ] **Step 3: Add the dependencies and marker.** Run `uvx --from 'uv>=0.10' uv add --dev pytest-playwright==0.10.0 axe-playwright-python==0.1.8`, then `uvx --from 'uv>=0.10' uv sync` and `uvx --from 'uv>=0.10' uv run --no-sync playwright install chromium`. In `pyproject.toml`, set `markers = ["live: …(unchanged)", "e2e: Playwright browser tests against /center?mock=1 (deselected by default; run with -m e2e)"]` and `addopts = "-m 'not live and not e2e'"`.
- [ ] **Step 4: Run** `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q -m e2e tests/e2e`. Expected: PASS. If axe reports a violation, fix the markup in `center.html` (label, accessible name, contrast) until it passes. Then run `uvx --from 'uv>=0.10' uv run --no-sync env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q` and confirm the 3 e2e tests are deselected.
- [ ] **Step 5: Add the CI job** to `.github/workflows/ci.yml`. It reuses the pinned `actions/checkout@3d3c42e…` and `astral-sh/setup-uv@1c37ad0…` (with `enable-cache: true`), then runs `uv sync --locked`, `uv run playwright install --with-deps chromium` and `uv run pytest -m e2e -q`:
```yaml
  e2e:
    name: E2E smoke + a11y (Playwright, chromium)
    runs-on: ubuntu-24.04
    timeout-minutes: 20
```
- [ ] **Step 6: Commit:** `git add pyproject.toml uv.lock .github/workflows/ci.yml tests/e2e && git commit -m "E2E: Playwright smoke and axe WCAG 2.2 AA scan of the monitor (-m e2e, own CI job)"`

---

## Deferred within phases 0-2 (deliberate)

- Event visibility by `channel_ids` for members: carried in `audience` and in events, but checked only from phase 3 (tasks follow the existing center channel rule).
- The `priority` filter is accepted with no effect until phase 3 adds task items.
- The axe scan covers only `#content` and the drawer. Widening it to the whole page (nav labels still use `--faint`) is a follow-up.
- `controls` are computed and returned by the detail route, but no buttons are rendered until release 2.

## Later plans

- Phase 3, team work: adapters for tasks, approvals, votes, forms and reminders, plus a `history[]` of the last 20 status changes.
- Phase 4, meetings: `org_id`, timestamps, attempts, chunk failures, ADK steps, and the Pub/Sub redelivery fix; meetings API auth only with the owner's approval.
- Release 2, controls: cancel / retry / resend through the action queue with `Idempotency-Key` and `202 {action_id}`.
- Release 3: owner-only pause/resume for the outbox and the memory import.

## Implementation rulings

Decisions taken while implementing this plan (from the run's ledger), one line each:

- **R1** Monitor rows are keyboard-focusable buttons whose accessible name holds the localized kind label and the title; filters have visible labels with the `mon_f_*` text ("Kind", ...); the view heading is an h1 "Monitor" in English (what the e2e asserts).
- **R2** The e2e makes the UI language deterministic with an init script that sets `localStorage` `salid-language` to `en`.
- **R3** Ruff formats the Python blocks of this Markdown file, so the plan is formatted with `uvx ruff format` in its own commit to keep `ruff format --check .` green.
- **R4** `viewer_for` treats members at position level 1-3 as executives too (executive = owner, or admin / member with `is_exec_level`); visibility is unchanged.
- **R5** The bot's monitor flush (every `TICK_SECONDS`) and heartbeat (every 30 s) run in their own asyncio loop next to the tick loop, so a slow or failing tick job cannot fake "bot offline".
- **R6** Tenant guard: `monitor_api._seen` drops every item whose `org_id` is not the requested org; adapters must set `org_id`.
- **R7** `monReduce` checks `reset` (and `floor`) before the stale-response test, and a reload moves the cursor to `resp.seq`, so a recreated log reloads.
- **R8** `monOnError(304)` is `{action: "unchanged", delayMs: 0, failure: false}`: a quiet feed is never a failure.
- **R9** The e2e filter step uses `select_option` on the keyboard-focused "Kind" combobox (ArrowDown opens the native picker on macOS).
- **R10** With OS notifications opted in and granted, a hidden tab keeps polling every `MON_HEAL_MS` (60 s) instead of stopping; notifications fire when the tab is hidden or unfocused.
- **R11** Row, drawer, toast and notification labels are the localized kind label plus the title without its leading English kind label.
- **R12** `ChangeOutbox._update` falls back to the `LEGACY_GUILD` doc inside the same transaction, so acks and button clicks on pre-migration notices work until `--apply`.
- **R13** Task 10 follow-ups: Task 11 updates the CLAUDE.md `videoad_bot.py` row and the `Step.state` comment (adds `skipped`); Task 12 adds the `skipped` step label (th + en) and icon.
- **R14** Video jobs requested in a DM or a private Salid channel are DM-scope (`dm=True`, `Audience(dm=True, people=[requester])`, no Discord link); the RETURNED save closes the running step; the emit wrapper is `MonitorMixin._monitor_video`.
- **R15** A `CancelledError` without the user-cancel flag is a shutdown: the job record stays `running` for the restart sweep; a research report that could not be posted is recorded failed / `error`.
- **R16** Task 12 also carries the `skipped` step label and icon and the `route()` focus fix (focus `#content` when the route changes or focus was inside it before the re-render).
- **R17** Final fix wave: CLAUDE.md monitor deploy runbook, code-map rows and env flags; the headless web bot keeps its emitter off; the `?mock=1` monitor data matches the real adapters; doc drift (spec status, this list, the `filter_items` and `_action_queue` docstrings). The other review minors are deferred.
