# Salid workflow monitor: design

Date: 2026-10-18. Status: design approved in conversation, awaiting spec review.
Scope: a real-time monitoring view in `/center` for the work Salid and the web app run in the background.

## 1. Problem and goal

Salid runs long, multi-step work that people cannot see: video-ad jobs, deep tasks and deep research, the change
outbox (web edits delivered to every platform), the web → Discord action queue, task handoffs, approvals, votes,
forms, reminders and meeting recordings. Today the web app shows almost none of it live. The only exceptions are
the actions card (5 s poll while pending) and the recorder's own meeting poll. Failures surface late, or only in
Discord.

Goal: one calm, accessible, live view that tells each person what is running, queued, waiting on someone, failed or
needs attention. It also shows how far each item has got, how long it has taken, and what to do next. It must work
within today's constraints:
- one Cloud Run instance with 20 request slots and a 900 s timeout, and CPU only while a request is open;
- the bot and the web app share data only through the store;
- a vanilla-JS single-file page.

### Decisions made with the owner

| Topic | Decision |
| --- | --- |
| Workflows | All four families: delivery queues, Salid jobs, team work, meeting recordings (phased, section 9) |
| Audience | Org owners/admins see the whole company; members see only their own items; super admins get a cross-company operations view |
| Controls | Release 1 read-only; release 2 cancel jobs + retry/resend failed deliveries; release 3 pause/resume (company outbox, memory import) |
| Freshness | About 5 s for active items while someone is watching; history and counts about 30-60 s |
| Alerts | In-app toasts plus opt-in browser (OS) notifications |
| Real-time approach | Per-company change log + browser delta polling (approach A). SSE is a later upgrade on the same cursor; Firestore listeners rejected (duplicated access rules, no SQLite mode) |
| End-to-end tests | Add a small Playwright smoke + accessibility scan (new dev dependency) |

## 2. Architecture overview

```
 producers (bot, platform bridges, web routes)
   │ write their own records as today (source of truth)
   │ + monitor.emit(org, event)  ──►  salid_monitor_log/{org}   (seq + ring of 200 events)
   │ + heartbeat every 30 s      ──►  salid_monitor_heartbeat/{process}
   ▼
 web: monitor_api.py
   GET summary / items / items/{kind}/{id}   ← adapters read the source records
   GET changes?since=seq                     ← one log read, ETag = seq, 304 when unchanged
   ▼
 /center "Monitor" section (inline JS)
   poll changes every 5 s while visible → refetch changed items (ids=) → render
```

## 3. Data model (`src/salid/monitor.py`, pure: no `discord` import)

### 3.1 `MonitorItem`

| Field | Notes |
| --- | --- |
| `kind` | `video_ad`, `deep_task`, `research`, `change`, `action`, `meeting`, `task`, `approval`, `vote`, `form`, `reminder` |
| `id`, `org_id`, `workspace` | workspace = platform + workspace memory id (from `orgs.workspace_memory_id`) |
| `title` | Safe label built by the adapter (for example "Video ad · 5 scenes", "Deep research", "Change → Slack"). Never briefs, prompts, research questions, transcripts, DM notes or message text |
| `status` | `queued`, `running`, `waiting`, `paused`, `done`, `failed`, `cancelled` |
| `attention` | bool plus `attention_reason` (`failed`, `stuck`, `waiting_too_long`, `overdue`, `bot_offline`) |
| `raw_status` | The workflow's own value, shown in the detail drawer |
| `requester_id`, `assignee_id`, `waiting_on` | Account ids where they exist |
| `steps[]` | `name`, `state`, `started_at`, `ms`, `attempts`, `error_code` |
| `started_at`, `updated_at`, `finished_at` | ISO UTC |
| `cost_usd` | Optional; video jobs today |
| `error` | `{code, message}`; the message is a short safe text from a code table, never raw stderr or exception text |
| `links` | Related page (project, task, document) and Discord jump URL when known |
| `audience` | `{admin_only, people:[ids], channel_ids:[...]}`, used by `visible()` and copied into events |

### 3.2 Status mapping (adapters)

Existing records stay the source of truth; each kind gets one pure adapter (record → `MonitorItem`).

| Kind | Source | Mapping |
| --- | --- | --- |
| `video_ad` | `salid_video_jobs` | `queued` → queued; `storyboard/stills/hero/audio/render` → running (current step); `done` → done; `failed` → failed (attention); `cancelled` → cancelled; `returned` → failed or cancelled by `reason` |
| `deep_task`, `research` | new `salid_jobs` (section 3.5) | `running` → running; `done` → done; `failed`/`restarted` → failed (attention); `cancelled` → cancelled |
| `change` | `salid_change_outbox` | Item = one change, steps = its targets. Any target `pending` → queued; any `sent` (awaiting ack) → waiting; all `acked` → done; any `failed` → failed (attention) |
| `action` | `salid_actions` | `pending` → queued (attention `stuck` after 60 s while the bot heartbeat is fresh); `done` → done; `failed`/expired → failed (attention) |
| `meeting` | meeting store | `recording`/`summarizing` → running; `done` → done; `failed` → failed (attention) |
| `task` | `salid_tasks` | `confirm` in `pending/awaiting_approval/owner_asked` → waiting (on that person); open + accepted → running ("in progress"); overdue → attention `overdue`; `done` → done; declined → cancelled |
| `approval` | `salid_approvals` | `pending` → waiting (attention after 1 day); `approved` → done; `declined/cancelled` → cancelled |
| `vote` | `salid_votes` | `open` → running; `closed` → done |
| `form` | `salid_forms` | open → running (steps = recipients by state); closed → done |
| `reminder` | `salid_reminders` | active → queued (`next_at`); fired once → done; failures → failed; cancelled → cancelled |

`stuck`: a running item whose `updated_at` is older than its kind's threshold (video step 10 min, deep task
timeout + 2 min, research 15 min), or any running item while the bot heartbeat is offline.

### 3.3 Change log (`salid_monitor_log`, key = org id)

- One record per company: `{seq, floor, events: [...]}`, a ring of the last 200 events.
- Event: `{v: 1, seq, at, kind, id, status, attention, audience}`. It carries no content, no prompts and no names.
- `monitor.emit(docs, org_id, event)` / `emit_many(...)` is a transactional read-modify-write, so `seq` only increases.
  It is **best effort**: a failure logs a counter and never breaks the workflow.
- Bursts (one change delivered to many targets) are batched into one write per bot tick (`emit_many`), to stay within
  Firestore's guidance against sustained rapid writes to a single document.
- A workspace not linked to any org logs under `guild:<memory id>` and is visible only in the operations view.

### 3.4 Heartbeat (`salid_monitor_heartbeat`, key = process name)

The bot and each platform bridge write `{process, version, tick_at, queues: {actions, outbox_pending, memory}}`
every 30 s. The UI shows the bot as online when the heartbeat is no older than 90 s, delayed when it is older than
90 s, and offline when it is older than 5 min. While the bot is offline, running items get attention `bot_offline`
("may be stuck") rather than being marked failed.

### 3.5 Backend additions by phase

- **Phase 1:**
  - The change outbox is re-keyed per org; today every change sits under global guild 0 and `_all()` reads every org's
    changes. A one-time migration moves the existing records.
  - Action-queue records are pruned after 30 days.
- **Phase 2:**
  - Video jobs persist `steps[]` (the timings are computed today but only logged).
  - Deep tasks and deep research save a `salid_jobs` record plus an `_active` index (copying the video pattern), with
    a restart sweep that marks cut-off jobs `restarted`.
- **Phase 3:** each team-work record keeps a `history[]` of its last 20 status changes (`at, by, from, to`).
- **Phase 4:**
  - Meetings get an `org_id`, `finished_at`/`failed_at`, an attempt counter, chunk failures recorded, and ADK
    `run_async` events (author = agent name) as steps.
  - Fix: `pubsub_finalize` stops endless redelivery of permanent failures (max attempts / dead-letter).
  - Auth on `/api/meetings*` only with the owner's separate approval (a listed known limitation).

## 4. API (`src/meeting_scribe/monitor_api.py`, mounted like `pm_api.py`)

Base: `/api/center/orgs/{org}/monitor`. Responses are always wrapped objects. Lists use the documents cursor
pattern `{items, next_cursor}`, limit 50, max 100.

| Route | Purpose | Release |
| --- | --- | --- |
| `GET /summary` | Counts by status and kind, attention count, heartbeat, `now`, `seq` | 1 |
| `GET /items` | Filters: `kind` (multi), `status` (multi), `attention`, `owner`, `workspace`, `from`, `to`, `priority` (tasks), `q`; `ids=` batch (≤ 50) for refetching changed items; `cursor`, `limit` | 1 |
| `GET /items/{kind}/{id}` | Detail: steps, history/events, sanitized error, links, `controls` allowed for this viewer | 1 |
| `GET /changes?since=<seq>` | `{seq, now, events, reset}`. ETag = seq; `304` when unchanged; `reset: true` when `since < floor`; events filtered per viewer by `audience` | 1 |
| `POST /items/{kind}/{id}/cancel`, `/retry` | Enqueues `cancel_job` / `resend_change` / `retry_action` in the existing action queue; `Idempotency-Key` header required; returns `202 {action_id}` | 2 |
| `POST /pause`, `/resume` | Body `{scope: "outbox" or "memory_import"}`; owner only | 3 |
| `GET /api/center/admin/monitor/summary`, `/items` | Super-admin operations view: all orgs, heartbeats, backlogs, unlinked workspaces | 1 |

- **Errors:** the existing FastAPI conventions; `402` licence, `403` not allowed, `404` unknown item, `429` rate
  limit with `Retry-After`.
- **Rate limit:** one `changes` call per 2 s per person. An in-process limiter is enough with `max-instances=1`.
- **Feature gate:** `MONITOR_ENABLED` (default false) until phase 1 ships.

### 4.1 Permissions (pure `monitor.visible(item, viewer)` and `monitor.controls(item, viewer)`)

| Viewer | Sees | Controls |
| --- | --- | --- |
| Org owner | Everything in the org (DM-scope items never) | cancel, retry/resend, pause/resume |
| Org admin | Everything in the org, except deep tasks if not an executive in that workspace | cancel, retry/resend |
| Member | Items they requested, are assigned, must approve or vote on; tasks per the existing center rule (no tasks from channels they cannot read) | cancel their own jobs (deep tasks: executives only) |
| Super admin | Operations view across orgs; org items read-only unless they also hold an org role | none by default |

The licence gate (`license_ok`) applies to every route except the super-admin view.

## 5. UI/UX (a new "Monitor" section in `center.html`, Thai: ติดตามงาน)

It reuses center's existing patterns: `.drawer`, `.tabs`, `.tbl.stack`, `.stat`, `loading()`, `errBox()`,
`toast()`, `confirmDlg`.

- **Navigation:** a nav item for everyone, with an attention-count badge. Super admins also get an Operations tab.
- **Health strip:**
  - live / reconnecting / offline, with the time of the last update;
  - the bot heartbeat;
  - **Pause live updates** (covers WCAG 2.2.2);
  - an opt-in notifications toggle.
- **Count tiles:** Running, Queued, Needs attention, Failed (24 h), Done today. Clicking a tile applies its filter.
- **Tabs:** Needs attention (default when it has items), Active, History.
- **Filters:** kind, status, owner, workspace/platform, date range, priority, search. Filters live in the URL hash and
  show as removable chips.
- **Rows:**
  - a status icon plus text (never colour alone), title, owner;
  - progress: steps done (e.g. 3/5), or "waiting on <name> · 2 d";
  - a live duration counter for running items, and the last update time;
  - 50 per page with "Load more"; on mobile, rows become cards and filters move into a sheet.
- **Detail drawer:**
  - a step timeline with durations and attempts;
  - the sanitized error with a "what you can do" hint, cost, and links;
  - a searchable, copyable event list with a severity filter (structured events, not raw logs);
  - controls behind a confirmation dialog (release 2).
- **Dismiss:** a person may dismiss an attention item for themselves (stored in localStorage in v1).
- **Calm live updates:**
  - rows update in place;
  - no reordering while the pointer or keyboard focus is in the list (changes are applied when interaction ends);
  - new items show as a "3 new" pill;
  - a short highlight on changed rows, off under `prefers-reduced-motion`.
- **Announcements and notifications:**
  - Screen readers get one polite `aria-live` summary per batch.
  - Toasts are only for items that concern the viewer (failures for admins), grouped, at most one every 10 s, with
    a "View" button.
  - Browser notifications are opt-in (permission asked on the toggle click), and only fire while the tab is hidden
    for failures or attention items that concern the viewer.
- **States:** skeleton loading; a helpful empty state; a member-view explanation; an offline/delayed banner that keeps
  the last data; a bot-offline banner; error with Retry; the licence notice.
- **Accessibility (WCAG 2.2 AA):**
  - full keyboard use; focus moves into the drawer and back on close; route changes focus `#content`;
  - targets of at least 24 px;
  - text uses `--muted` rather than `--faint`, which fails AA.
- **Language and time:** Thai and English strings in center's `I` catalog; relative times with the exact Bangkok
  time in `title`.

## 6. Live behaviour and failure handling

**Client feed** (inline in `center.html`; the poller and reducer are pure functions, testable with Node):

- **States:** `live`, `delayed`, `offline`, `paused` (by the user), `hidden` (tab in the background).
- **Polling:** every 5 s ± 0.5 s while visible. When the tab becomes visible again it fetches at once and refreshes
  the full list.
- **Backoff:** 5 → 10 → 20 → 40 → 60 s cap, with jitter. "Reconnecting…" after 2 failures. "Offline" after 4
  failures or when `navigator.onLine` is false; the `online` event resumes.
- **Error codes:** `401` → the existing sign-in flow; `403` → member-view explanation and stop; `402` → licence
  notice; `429` → honour `Retry-After`.
- **Applying changes:** events are hints; changed items are refetched in one `items?ids=` batch, and the newest
  server state wins (`updated_at` plus the request counter, as in center's `navSeq`).
  - `seq` ≤ last → ignore (duplicate).
  - Out-of-order events in a batch → sort by `seq`.
  - `reset` or a gap → reload the view.
  - A 60 s heal refreshes the visible page.
- **Clock:** durations are computed against the server's `now`.
- **Tabs:** several open tabs poll independently (no coordination in v1).

**Server:**

- **`changes`:** one log read; ETag/304; per-viewer filtering from `audience` without reading items.
- **Producers:** an `emit` failure logs and continues; the client heal covers the gap.
- **Versioning:** events carry `v`; clients ignore unknown kinds and fields; fields are only ever added.

**Controls (release 2):**

- No optimistic success: the row shows "Cancelling…" until the result arrives.
- `Idempotency-Key` prevents a double click or resubmit from acting twice.
- The result arrives through the change log (action done or failed).
- After 60 s with no result: "Still waiting for Salid", together with the heartbeat state.

## 7. Security and privacy

- Titles, errors and events never contain prompts, briefs, research questions, transcripts, DM notes, message text,
  tokens or raw stderr. A guard test checks every string field the API returns.
- Access is decided only server-side by `visible`/`controls`. The client merely hides controls.
- The super-admin view is read-only by default (least privilege).
- Controls go through the existing action queue, so the bot applies its own rules again (for example, requester or
  admin for video cancel; executives for deep tasks).
- Signed-in pages keep `Cache-Control: no-store` and `Referrer-Policy: no-referrer`.

## 8. Performance and cost (estimates from the code, not measured)

- **Per open, visible monitor tab:** one log read every 5 s, plus refetches of changed items and a 60 s page refresh.
  Hidden tabs cost nothing.
- **Heartbeats:** 2 writes per minute per process.
- **`emit`:** one transactional write per state change (batched per tick for bursts).
- **No long-lived connections,** so the instance's 20 request slots stay available.
- The outbox re-key removes a whole-collection read that today grows with the number of orgs.

## 9. Phases and releases

Each phase is a small PR behind `MONITOR_ENABLED`.

| Phase | Contents |
| --- | --- |
| 0. Foundation | `monitor.py` (model, mapping, `visible`/`controls`, `emit`, change log, heartbeat writer); `monitor_api.py` (summary, items, detail, changes, operations view); the Monitor section (feed, health strip, states, i18n, `?mock=1` data) |
| 1. Delivery queues | Outbox and action adapters, `emit` calls, outbox per-org re-key + migration, action pruning |
| 2. Salid jobs | Video `steps[]` + `emit`; `salid_jobs` for deep tasks and research + restart sweep |
| 3. Team work | Adapters for tasks, approvals, votes, forms, reminders; `history[]` |
| 4. Meetings | Org ownership, timestamps, step events, attempts, the Pub/Sub redelivery fix; meetings API auth only if approved |
| Release 2 | Cancel / retry / resend through the action queue with idempotency (after phase 2) |
| Release 3 | Owner-only pause/resume for the outbox and the memory import |

## 10. Testing

- **Unit (pytest, offline, test-first):**
  - each adapter;
  - the status-mapping table;
  - the roles × kinds matrix for `visible`/`controls`;
  - `emit` keeps `seq` increasing under concurrent writes (fake Firestore and SQLite);
  - ring overflow sets `reset`;
  - per-viewer filtering of `changes`;
  - ETag/304;
  - the sensitive-field guard;
  - rate limit;
  - heartbeat thresholds.
- **API (TestClient with dependency overrides):** owner, admin, member, super admin, unlicensed org; paging; `ids=`
  batch; the operations view.
- **Bot side:** existing tests extended to assert `emit` calls (video progress, outbox updates, actions).
- **Browser code (Node, like the hub retention test):**
  - reducer: duplicates, out-of-order events, gap → reset;
  - backoff schedule;
  - feed states;
  - no reorder while focused;
  - Thai/English key parity;
  - static page checks (one `<script>`, no new external hosts).
- **End to end:** a small Playwright smoke against `/center?mock=1`: keyboard path through filters, rows and the drawer;
  drawer focus trap and return; an automated accessibility scan. This adds a dev dependency and browser installation in
  CI.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| A code path forgets `emit` | The item is the truth; 60 s heal; bot tests assert `emit` |
| Write rate on the per-company log document | `emit_many` once per tick; one log per company |
| Wrong `audience` hints leak events | Permission-matrix tests; events carry no content anyway |
| Fast-moving `main` | Small PRs, rebased often |
| `center.html` keeps growing (359 KB) | The monitor code stays one clearly marked block of pure functions plus a view; extracting shared helpers is a separate decision |

## 12. Out of scope

SSE (a later upgrade on the same cursor); Firestore listeners; raw log streaming (operators use Cloud Logging);
coordination between browser tabs; auth on the meetings API unless approved separately; changes to `hub.html`
and `pm.html`.
