# Salid slide decks (design)

Date: 2026-10-16. Status: approved in conversation, awaiting written-spec review.
Branch: `feature/salid-decks` (worktree `.claude/worktrees/salid-decks`). Target: demo-ready before
the 2026-10-18 submission.

## Goal

A Salid user asks for a presentation in Discord, in Thai or English, and gets a professional
`.pptx` attached to Salid's reply in about 15-30 seconds. Decks can cover the team's own data
(spending for a period, a meeting recap, "what happened this week" from team memory) or any free
topic ("a 5-slide pitch for our new menu").

This is the first of several content sub-projects (later, each with its own spec: images and
posters, standalone charts and infographics, audio briefings, video).

## Scope

In scope: a `create_deck` tool on the existing chat agent, data-bound native charts from the
ledger, one in-code theme, Thai and English, the file attached to the text reply, a per-guild
feature flag, docs.

Out of scope (later): images in slides, several themes or brand kits, editing an existing deck by
chat, slide previews as images, Google Slides export, decks from voice-meeting answers, decks in
the hub, storing decks.

## Constraints

- Numbers in charts never come from the model: code fetches them from the ledger.
- Access is unchanged: a deck contains only what the agent's tools return for this asker in this
  channel, plus text the model writes. Chart data uses the same scoped ledger (non-executives:
  their own receipts only).
- Nothing is stored: the deck is built in memory, posted and discarded.
- Logs carry counts and timings only, never content.
- Offline tests; no new infrastructure; no deploy without human approval.
- One new dependency, `python-pptx`, added with `uv add` (needs a current uv, see Rollout).

## Success criteria

1. "@Salid ทำสไลด์สรุปค่าใช้จ่ายเดือนนี้" returns a `.pptx` whose chart categories and values equal
   `spending_summary` for the same range and scope.
2. A meeting-recap deck works from a meeting summary Salid posted earlier (team memory or the
   room's recent messages).
3. Thai text is readable in PowerPoint, Keynote and Google Slides (manual check of a sample deck).
4. The generated file opens with `python-pptx` in tests, with the expected slides, titles, notes
   and chart data.
5. Full suite green; ruff clean.

## Design

### Flow

1. The user asks for a deck. The chat agent runs as today and gathers data with its existing tools
   (`spending_summary`, `chat_search`, `chat_days`, ...) when the deck is about team data.
2. The agent calls `create_deck(title, slides, subtitle="")` with a structured outline.
3. Code validates the outline, fetches chart data through the scoped ledger, renders the `.pptx`
   in memory and returns `{"status": "created", "slides": N, "file": "<name>"}` to the model, or
   `{"error": "..."}` naming what to fix.
4. The model writes a short text reply (what the deck contains). The bot attaches the file to
   that reply.

### Components

**`salid/deck.py` (new).**

- Pydantic models, also used as the tool's parameter types so the model sees the full schema:
  - `ChartQuery`: `group_by` (`category` | `vendor` | `submitter` | `day`), `start_date`,
    `end_date` (ISO dates, same checks as the ledger tools), `style` (`bar` | `pie` | `line`,
    default `bar`), `top` (1-10, default 8; the rest summed as "Other" / "อื่นๆ"). For
    `group_by="day"` there is no "Other" bucket: days stay in date order and the range may span at
    most 62 days.
  - `Slide`: `kind` (`section` | `bullets` | `two_column` | `chart`), `title` (max 80 chars),
    `bullets` (max 6, each max 140 chars; the left column for `two_column`), `right` (same limits,
    `two_column` only), `chart` (`ChartQuery`, `chart` only), `notes` (speaker notes, max 1000).
  - `DeckSpec`: `title` (max 80), `subtitle` (max 120, optional), `slides` (1-12).
  - Validation also checks kind-specific fields (a `chart` slide needs `chart`; `bullets` needs at
    least one bullet).
- `ChartData`: categories and one series per currency (from `store.build_summary` output:
  `groups[].key`, `groups[].totals_by_currency`), already reduced to `top` plus "Other".
- `render_deck(spec: DeckSpec, charts: dict[int, ChartData], *, lang: str, today: date) -> bytes`:
  a pure function (no network, no model), `charts` keyed by slide index.
- `deck_filename(title: str, today: date) -> str`: ASCII slug of the title plus
  `-YYYYMMDD.pptx`; `salid-deck-YYYYMMDD.pptx` when the slug is empty (e.g. an all-Thai title).

**`create_deck` tool (in `qa.build_tools`).**

- Offered only when a `DeckSink` is passed: normal text answers with the guild feature `decks` on.
  Not offered for voice wake-word answers, join-mode interjections or the hub.
- Its parameters are annotated with the pydantic models. ADK builds the schema from them but
  passes plain dicts, so the tool validates with `DeckSpec.model_validate` and returns validation
  errors (field path + message) as data so the model can fix the outline in its next round.
- For each chart slide it calls the scoped ledger (`LedgerTools`, which already injects
  `guild_id` and `forced_submitter_id`) for `summarize(start, end, group_by)`, turns the result
  into `ChartData`, renders, and appends the file to the sink.
- Output over `MAX_DECK_BYTES` (2 MB) or a render exception: `{"error": "could not build the
  deck"}`, logged with the exception type only.
- At most one deck per answer (a second call replaces the first).

**`Answer.files`**: new field `list[tuple[str, bytes]]`, default empty. `ask` / `ask_async` take
`deck: bool = False`; when true they create the sink and the tool and return the files.

**`bot._answer`**: passes `deck=self.team.enabled(gid, "decks") and not interjection`, and attaches
`ans.files` to the reply (first message part) as `discord.File`.

**Prompt.** `DECK_PROMPT` in the stable (cacheable) part of the system prompt when the tool is
offered: make a deck only when the user asks for slides, a presentation, a deck, สไลด์ or พรีเซนต์;
gather data in at most 2 tool rounds, then call `create_deck`; title slide is automatic; 4-8
content slides, one idea per slide, at most 5 short bullets, details in speaker notes; use `chart`
slides for spending (code fills the numbers; never type numbers into a chart); write slide text in
the reply language; afterwards reply briefly with what the deck covers (the file is attached).
`CAPABILITIES_PROMPT` gains one line for slides.

**Feature flag.** `decks` in `guild_settings.Features`, default on, like the other features.

**i18n.** New `CATALOG` entries in Thai and English: the title-slide footer ("สร้างโดยสลิด" /
"Made with Salid") and date line, "อื่นๆ" / "Other", the no-data placeholder.

### Look

- 16:9 (13.333 x 7.5 in). Title slide: dark background, large title, subtitle, date and footer.
  Content slides: white, a coloured accent bar, title at top, body below, page number bottom
  right. Section slides: accent background, large centred title.
- Font sizes step down with text length (python-pptx has no reliable auto-fit): title 36-28 pt,
  bullets 24-16 pt.
- One colour palette for accents and chart series, defined as constants.

### Thai text

Every run sets the Latin font and the complex-script font (`a:cs`, the slot PowerPoint uses for
Thai). python-pptx does not expose `a:cs`, so the run XML is edited directly. Fonts are constants:
Latin `Arial`, complex script `Leelawadee UI`; apps without it substitute a system Thai font. A
sample deck is checked by hand in PowerPoint, Keynote and Google Slides before merge.

### Charts

Native PowerPoint charts (editable). Bar by default, line for `group_by="day"` when asked, pie for
a few categories. Values in baht with thousands separators; one series per currency when several
are present. A period with no data shows a placeholder text instead of an empty chart.

### Limits

12 content slides, 6 bullets, 2 MB per file. Rendering under a second. The outline costs about
1-3k output tokens; a typical deck takes 1-2 extra tool rounds within the existing limit of 4.

## Errors

- Invalid outline: validation errors returned to the model; it fixes and calls again.
- Bad chart dates: the same `ValueError` checks as the ledger tools, returned as data.
- Render failure or oversize file: error returned to the model, which tells the user; logged with
  the exception type only.
- The model never calls `create_deck`: a normal text reply.

## Security and privacy

- No deck is written to disk or the database.
- Content comes from scoped tools and model text; python-pptx escapes text, so content from chat
  memory cannot break the file.
- Logs: counts and timings only (slides, charts, bytes, ms).
- Salid's reply is recorded in team memory with the usual `[file: <name>.pptx]` descriptor.

## Observability

`create_deck` time appears in the per-answer log line (`tools=create_deck:<ms>`). One log line per
deck: `deck guild=%s slides=%d charts=%d bytes=%d ms=%d`.

## Testing

Offline, test-first:

- Render: read back with python-pptx; slide count and order (title slide first); titles, bullets,
  notes; Thai text preserved; `a:cs` set on runs; chart categories and values equal the ledger
  summary, including "Other" and several currencies; empty data shows the placeholder; long text
  gets a smaller font; filename slug and fallback.
- Validation: limits, kind-specific fields, error messages name the field.
- Tool (scripted ADK LLM): a `create_deck` call puts a `.pptx` in `Answer.files`; a
  non-executive's chart covers only their own receipts; an invalid outline returns an error and a
  corrected second call succeeds; the tool is absent without `deck=True`; the declaration exposes
  the nested schema (if ADK cannot express it for Gemini, fall back to `list[dict]` with the
  schema described in the docstring, validated the same way).
- Bot: the file is attached to the reply; no deck tool for interjections or when the flag is off.

Plus `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .`.

## Rollout

- Work in the worktree on `feature/salid-decks`; one PR. No deploy without approval.
- Before merge: render a sample Thai deck for the human to open (visual check).
- Prerequisite: `python-pptx` must be added with a current uv. The uv on PATH is 0.6.5 (from
  another project's virtualenv) and would rewrite `uv.lock` to an older format. Ask the human to
  install a current uv first (e.g. `brew install uv`); otherwise keep the lock churn out of the PR.
- Docs: README Salid section, CLAUDE.md code map (`deck.py`), capabilities text.

## Risks

| Risk | Mitigation |
| --- | --- |
| Thai font looks wrong in one app | Font names are constants; swap after the visual check |
| Weak outlines from the model | Prompt guidance, strict limits, error-and-retry |
| A rich deck needs more than 4 tool rounds | Prompt: gather data in 1-2 rounds, then build |
| Nested tool schema not accepted by Gemini | Fallback to `list[dict]` + docstring schema, same validation |
| Deadline | Small MVP; one theme; no images in slides |
