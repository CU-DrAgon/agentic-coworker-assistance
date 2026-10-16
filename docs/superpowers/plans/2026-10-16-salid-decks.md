# Salid Slide Decks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Salid's chat agent can build a professional `.pptx` deck (Thai or English, native charts with ledger numbers) and attach it to its Discord reply.

**Architecture:** A pure renderer (`salid/deck.py`: pydantic outline models, `chart_data`, `render_deck`, `deck_filename`) plus a `create_deck` tool added by `qa.build_tools` when the caller passes a `DeckSink`. The tool validates the outline, fetches chart numbers through the asker's scoped `LedgerTools`, renders in memory and leaves the file in the sink; `ask` returns it as `Answer.files` and `bot._answer` attaches it to the reply.

**Tech Stack:** Python 3.12, python-pptx (new), pydantic, google-adk 2.11 (FunctionTool), discord.py, pytest (offline).

**Spec:** `docs/superpowers/specs/2026-10-16-salid-decks-design.md`

## Global Constraints

- Work only in the worktree `/Users/thanatpon.ap/Documents/agentic-coworker-assistance/.claude/worktrees/salid-decks` on branch `feature/salid-decks`; never `cd` to the main checkout (another session works there).
- Python through uv only. Until python-pptx is added (Task 1), run commands as `uv run --frozen ...` so uv never rewrites `uv.lock`. The `uv` on PATH may be 0.6.5 (another project's venv): it must not rewrite the lock format (see Task 1).
- Never use bare `git stash` (the stash is shared with other sessions).
- Chart numbers never come from the model: code fetches them via `LedgerTools.spending_summary`.
- Access unchanged: the deck tool is bound to the request's `LedgerTools` (guild + `forced_submitter_id`); no new model-visible scope parameter.
- No deck is written to disk or the database; logs carry counts and timings only.
- Limits (from the spec): title max 80 chars, subtitle max 120, 1-12 slides, 6 bullets per list, 140 chars per bullet, notes max 1000, chart `top` 1-10 (default 8), `group_by="day"` range max 62 days, file max 2 MB (`MAX_DECK_BYTES = 2_000_000`).
- Fonts are constants: Latin `Arial`, complex script (Thai) `Leelawadee UI`.
- Every new user-facing string goes into `src/salid/i18n.py` `CATALOG` in Thai and English.
- Offline tests; `uv run pytest -q` green and `uv run ruff check . && uv run ruff format --check .` clean after every task. Live tests run only with human approval.
- No deploy, no Discord posts without approval. Commits without attribution lines.

## Review Focus

1. Text from chat memory with control characters (`\x00`, `\x0b`) or newlines inside a bullet must not crash rendering (python-pptx rejects XML-invalid characters). Test: Task 2 `test_control_characters_and_newlines_are_cleaned`.
2. The bot lacks Discord's Attach Files permission (or the upload fails): the text answer must still be posted, with a short note. Test: Task 5 `test_text_is_still_posted_when_the_file_cannot_be_attached`.
3. The model calls `create_deck` twice in one answer (e.g. after a success): exactly one file is attached, the last one. Test: Task 4 `test_a_second_create_deck_replaces_the_first`.
4. An all-Thai or emoji title must still give a safe filename. Test: Task 1 `test_deck_filename_slug_and_fallback`.
5. Reply language `auto` with an English question gives English fixed strings in the deck (footer, date, "Other"). Test: Task 4 `test_deck_fixed_strings_follow_the_reply_language`.

## File Structure

| File | Responsibility | Tasks |
| --- | --- | --- |
| `src/salid/deck.py` (new) | outline models, validation, filename, chart data, rendering | 1, 2, 3 |
| `src/salid/qa.py` | `DeckSink`, `create_deck` tool in `build_tools`, `Answer.files`, `deck` param, `DECK_PROMPT`, capabilities line | 4 |
| `src/salid/i18n.py` | `deck_footer`, `deck_other`, `deck_no_data`, `deck_attach_failed` | 2, 3, 5 |
| `src/salid/guild_settings.py`, `src/meeting_scribe/static/hub.html` | `decks` feature flag and hub toggle | 5 |
| `src/salid/bot.py` | pass `deck=`, attach `ans.files`, fallback when attaching fails | 5 |
| `scripts/salid_sample_deck.py` (new) | render a sample Thai deck offline for the visual check | 3 |
| `tests/salid/test_deck.py` (new) | renderer and model tests | 1, 2, 3 |
| `tests/salid/test_salid_qa.py`, `test_salid_features.py`, `test_guild_settings.py`, `tests/test_hub.py`, `test_salid_live.py` | tool, bot, flag, live tests | 4, 5, 6 |
| `pyproject.toml`, `uv.lock`, `README.md`, `CLAUDE.md` | dependency, docs | 1, 6 |

---

### Task 1: Dependency, outline models, filename

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `src/salid/deck.py`
- Test: `tests/salid/test_deck.py`

**Interfaces:**
- Produces (`salid.deck`): `ChartQuery`, `Slide`, `DeckSpec` (pydantic `BaseModel`s, fields below), `deck_filename(title: str, today: date) -> str`, constants `MAX_SLIDES = 12`, `MAX_BULLETS = 6`, `MAX_BULLET_CHARS = 140`, `MAX_DAY_RANGE = 62`, `MAX_DECK_BYTES = 2_000_000`.
  - `ChartQuery`: `group_by: Literal["category", "vendor", "submitter", "day"]`, `start_date: str`, `end_date: str`, `style: Literal["bar", "pie", "line"] = "bar"`, `top: int = 8` (1-10).
  - `Slide`: `kind: Literal["section", "bullets", "two_column", "chart"]`, `title: str` (max 80), `bullets: list[str] = []`, `right: list[str] = []`, `chart: ChartQuery | None = None`, `notes: str = ""` (max 1000).
  - `DeckSpec`: `title: str` (max 80), `subtitle: str = ""` (max 120), `slides: list[Slide]` (1-12).

- [ ] **Step 1: Check uv, then add the dependency**

Run: `uv --version`. Then `uv add python-pptx` and `git diff --stat uv.lock` plus `git diff uv.lock | grep '^[-+]revision'`.
Expected: the lock diff only adds `python-pptx` and its dependencies (lxml, XlsxWriter, possibly others already present), and no `revision` line changes. If `revision` changes or hundreds of unrelated lines move: `git checkout -- pyproject.toml uv.lock`, stop, and ask the human to install a current uv (`brew install uv`) before continuing.

- [ ] **Step 2: Write the failing tests** (`tests/salid/test_deck.py`)

```python
def bullets(n=2, text="ข้อ"):
    return {"kind": "bullets", "title": "หัวข้อ", "bullets": [f"{text} {i}" for i in range(n)]}


def test_valid_outline_parses():
    spec = DeckSpec.model_validate({"title": "สรุปเดือนตุลาคม", "slides": [bullets()]})
    assert spec.slides[0].kind == "bullets" and spec.subtitle == ""


@pytest.mark.parametrize("bad, field", [
    ({"title": "x" * 81, "slides": [bullets()]}, "title"),
    ({"title": "t", "slides": []}, "slides"),
    ({"title": "t", "slides": [bullets()] * 13}, "slides"),
    ({"title": "t", "slides": [bullets(7)]}, "bullets"),
    ({"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": ["x" * 141]}]}, "bullets"),
    ({"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": []}]}, "bullets"),
    ({"title": "t", "slides": [{"kind": "chart", "title": "t"}]}, "chart"),
    ({"title": "t", "slides": [{"kind": "video", "title": "t"}]}, "kind"),
])
def test_invalid_outlines_name_the_field(bad, field):
    with pytest.raises(ValidationError) as e:
        DeckSpec.model_validate(bad)
    assert field in str(e.value)


@pytest.mark.parametrize("q, ok", [
    ({"group_by": "category", "start_date": "2026-10-01", "end_date": "2026-10-31"}, True),
    ({"group_by": "day", "start_date": "2026-09-01", "end_date": "2026-10-31"}, True),   # 61 days
    ({"group_by": "day", "start_date": "2026-08-01", "end_date": "2026-10-31"}, False),  # 92 days
    ({"group_by": "category", "start_date": "2026-10-31", "end_date": "2026-10-01"}, False),
    ({"group_by": "category", "start_date": "31/10/2026", "end_date": "2026-10-31"}, False),
    ({"group_by": "category", "start_date": "2026-10-01", "end_date": "2026-10-31", "top": 11}, False),
])
def test_chart_query_dates_and_limits(q, ok):
    if ok:
        ChartQuery.model_validate(q)
    else:
        with pytest.raises(ValidationError):
            ChartQuery.model_validate(q)


def test_deck_filename_slug_and_fallback():
    d = date(2026, 10, 16)
    assert deck_filename("Spending report: Oct 2026!", d) == "spending-report-oct-2026-20261016.pptx"
    assert deck_filename("สรุปค่าใช้จ่าย", d) == "salid-deck-20261016.pptx"
    assert deck_filename("🎉 Q4 / ../plan", d) == "q4-plan-20261016.pptx"
    assert len(deck_filename("word " * 40, d)) <= 64
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/salid/test_deck.py -q`
Expected: FAIL (`ModuleNotFoundError: salid.deck`).

- [ ] **Step 4: Implement the models and `deck_filename` in `src/salid/deck.py`**

Field limits via `Field(max_length=...)`, `min_length/max_length` on lists, and per-item length with `Annotated[str, StringConstraints(max_length=MAX_BULLET_CHARS)]`. A `model_validator(mode="after")` on `Slide` requires `chart` for `kind="chart"` and at least one bullet for `bullets` / `two_column` (error text names the field: "chart is required for a chart slide", "bullets needs at least one item"). `ChartQuery` validates ISO dates (`date.fromisoformat`), `end >= start`, and `(end - start).days + 1 <= MAX_DAY_RANGE` for `day`. `deck_filename`: lowercase ASCII words (`[a-z0-9]+` after NFKD and dropping non-ASCII) joined by `-`, cut to 40 chars at a word boundary, then `-YYYYMMDD.pptx`; empty slug gives `salid-deck-YYYYMMDD.pptx`. Module docstring: what the module is (outline + renderer, pure, no network).

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/salid/test_deck.py -q && uv run pytest -q`
Expected: PASS; suite green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/salid/deck.py tests/salid/test_deck.py
git commit -m "Salid: deck outline models, validation and filenames (adds python-pptx)"
```

---

### Task 2: Render text slides (theme, Thai fonts, notes)

**Files:**
- Modify: `src/salid/deck.py`, `src/salid/i18n.py`
- Test: `tests/salid/test_deck.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: `render_deck(spec: DeckSpec, charts: dict[int, ChartData], *, lang: str, today: date) -> bytes` (charts used in Task 3; Task 2 renders a chart slide's title only); `clean_text(s: str) -> str`; constants `LATIN_FONT = "Arial"`, `THAI_FONT = "Leelawadee UI"`; i18n keys `deck_footer` (`{"th": "สร้างโดยสลิด · {date}", "en": "Made with Salid · {date}"}`); `format_deck_date(d: date, lang: str) -> str` returning `"16 October 2026"` (en) or `"16 ตุลาคม 2569"` (th, Buddhist Era = year + 543).

- [ ] **Step 1: Write the failing tests** (read back with `pptx.Presentation(io.BytesIO(data))`)

```python
def render(spec_dict, lang="th", charts=None):
    return Presentation(io.BytesIO(render_deck(DeckSpec.model_validate(spec_dict), charts or {},
                                               lang=lang, today=date(2026, 10, 16))))


def texts(slide):
    return [s.text_frame.text for s in slide.shapes if s.has_text_frame]


def test_title_slide_then_one_slide_per_outline_slide():
    prs = render({"title": "สรุปประชุม", "subtitle": "ทีมขาย", "slides": [
        {"kind": "section", "title": "ภาพรวม"},
        {"kind": "bullets", "title": "สิ่งที่ตกลง", "bullets": ["เลื่อนเปิดร้านเป็นวันศุกร์"], "notes": "รายละเอียด"},
        {"kind": "two_column", "title": "ก่อน / หลัง", "bullets": ["เดิม"], "right": ["ใหม่"]},
    ]})
    assert len(prs.slides) == 4 and prs.slide_width == Inches(13.333)
    cover = " ".join(texts(prs.slides[0]))
    assert "สรุปประชุม" in cover and "ทีมขาย" in cover and "16 ตุลาคม 2569" in cover and "สลิด" in cover
    assert "เลื่อนเปิดร้านเป็นวันศุกร์" in " ".join(texts(prs.slides[2]))
    assert prs.slides[2].notes_slide.notes_text_frame.text == "รายละเอียด"
    assert {"เดิม", "ใหม่"} <= set(" ".join(texts(prs.slides[3])).split())


def test_every_run_sets_the_thai_font_after_latin():
    prs = render({"title": "ทดสอบ", "slides": [{"kind": "bullets", "title": "หัวข้อ", "bullets": ["ข้อความ"]}]})
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    rpr = r._r.find(qn("a:rPr"))
                    tags = [c.tag for c in rpr]
                    assert rpr.find(qn("a:latin")).get("typeface") == LATIN_FONT
                    assert rpr.find(qn("a:cs")).get("typeface") == THAI_FONT
                    assert tags.index(qn("a:cs")) == tags.index(qn("a:latin")) + 1


def test_long_text_gets_a_smaller_font():
    short = render({"title": "t", "slides": [{"kind": "bullets", "title": "t", "bullets": ["สั้น"]}]})
    long_ = render({"title": "t", "slides": [{"kind": "bullets", "title": "t",
                                               "bullets": ["ยาวมาก" * 23] * 6}]})
    size = lambda prs: max(r.font.size for s in prs.slides[1].shapes if s.has_text_frame
                           for p in s.text_frame.paragraphs for r in p.runs if "t" != p.text)
    assert size(long_) < size(short) and size(long_) >= Pt(16)


def test_control_characters_and_newlines_are_cleaned():
    prs = render({"title": "a\x00b", "slides": [{"kind": "bullets", "title": "t",
                                                  "bullets": ["line1\nline2\x0b end"]}]})
    body = " ".join(texts(prs.slides[1]))
    assert "line1 line2 end" in body and "\x0b" not in body
    assert "ab" in " ".join(texts(prs.slides[0]))


def test_english_fixed_strings():
    cover = " ".join(texts(render({"title": "Q4", "slides": [bullets()]}, lang="en").slides[0]))
    assert "Made with Salid" in cover and "16 October 2026" in cover


def test_page_numbers_on_content_slides():
    prs = render({"title": "t", "slides": [bullets(), bullets()]})
    assert "2" in texts(prs.slides[1]) and "3" in texts(prs.slides[2])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_deck.py -q`
Expected: FAIL (`ImportError: render_deck`).

- [ ] **Step 3: Implement in `src/salid/deck.py` and add the i18n key**

- 16:9 (`Inches(13.333)` x `Inches(7.5)`), blank layout (`prs.slide_layouts[6]`), all shapes drawn in code. Palette constants: `NAVY = RGBColor(0x1F, 0x2A, 0x44)`, `ACCENT = RGBColor(0x2E, 0x86, 0xDE)`, `INK = RGBColor(0x22, 0x22, 0x22)`, `MUTED = RGBColor(0x6B, 0x72, 0x80)`, `CHART_COLORS` (6 colours starting with ACCENT).
- Title slide: NAVY background, title (white), subtitle, `format_deck_date` line and `t("deck_footer", lang, date=...)`. Section slide: ACCENT background, centred title. Content slides: white, ACCENT bar on the left edge, title top, body below, page number (slide number as text) bottom right.
- Font sizing: title 36 pt, down to 28 pt when over 40 chars; bullet size from the slide's total bullet chars: <=200 chars 24 pt, <=400 20 pt, else 16 pt. `word_wrap=True`, `auto_size=MSO_AUTO_SIZE.NONE`.
- Every run goes through one helper that sets text, size, colour, `font.name = LATIN_FONT` and inserts `a:cs typeface=THAI_FONT` directly after `a:latin` in `a:rPr` (schema order: latin, ea, cs, sym, hlinkClick; python-pptx does not expose `a:cs`).
- `clean_text`: replace `\r`, `\n`, `\t` with spaces, drop other control characters (`unicodedata.category(ch) == "Cc"`), collapse spaces, strip. Applied to every text and note.
- Speaker notes via `slide.notes_slide.notes_text_frame.text`.
- A `chart` slide renders its title only in this task (Task 3 adds the chart).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/salid/test_deck.py -q && uv run pytest -q`
Expected: PASS; suite green.

- [ ] **Step 5: Commit**

```bash
git add src/salid/deck.py src/salid/i18n.py tests/salid/test_deck.py
git commit -m "Salid: render deck text slides (16:9 theme, Thai font slot, notes, cleaned text)"
```

---

### Task 3: Data-bound charts and the sample deck

**Files:**
- Modify: `src/salid/deck.py`, `src/salid/i18n.py`
- Create: `scripts/salid_sample_deck.py`
- Test: `tests/salid/test_deck.py`

**Interfaces:**
- Consumes: Task 2 `render_deck`; `store.build_summary` output shape (`groups[].key`, `groups[].totals_by_currency`; groups biggest first, or by date for `day`).
- Produces: `ChartData` (dataclass: `categories: list[str]`, `series: dict[str, list[float]]` currency -> values aligned with categories; property `empty: bool`), `chart_data(summary: dict, query: ChartQuery, lang: str) -> ChartData`; i18n keys `deck_other` (`{"th": "อื่นๆ", "en": "Other"}`) and `deck_no_data` (`{"th": "ไม่มีข้อมูลในช่วงนี้", "en": "No data for this period"}`).

- [ ] **Step 1: Write the failing tests**

```python
def summary(groups):
    return {"groups": [{"key": k, "totals_by_currency": by} for k, by in groups]}


Q = {"group_by": "category", "start_date": "2026-10-01", "end_date": "2026-10-31"}


def test_top_groups_plus_other_per_currency():
    s = summary([("วัตถุดิบ", {"THB": 9000}), ("ค่าเดินทาง", {"THB": 250.5}),
                 ("สาธารณูปโภค", {"THB": 120}), ("อุปกรณ์", {"USD": 10})])
    cd = chart_data(s, ChartQuery.model_validate({**Q, "top": 2}), "th")
    assert cd.categories == ["วัตถุดิบ", "ค่าเดินทาง", "อื่นๆ"]
    assert cd.series == {"THB": [9000, 250.5, 120], "USD": [0, 0, 10]}


def test_day_groups_keep_date_order_and_no_other():
    s = summary([("2026-10-01", {"THB": 5}), ("2026-10-02", {"THB": 7})])
    cd = chart_data(s, ChartQuery.model_validate({**Q, "group_by": "day", "top": 1}), "en")
    assert cd.categories == ["10-01", "10-02"] and cd.series == {"THB": [5, 7]}


def test_missing_keys_and_empty_data():
    assert chart_data(summary([(None, {"THB": 1})]), ChartQuery.model_validate(Q), "th").categories == ["—"]
    assert chart_data(summary([]), ChartQuery.model_validate(Q), "th").empty


def test_chart_slide_draws_a_native_chart_with_the_given_numbers():
    spec = {"title": "t", "slides": [{"kind": "chart", "title": "ค่าใช้จ่ายตามหมวด", "chart": Q}]}
    cd = ChartData(categories=["วัตถุดิบ", "อื่นๆ"], series={"THB": [9000.0, 370.5]})
    prs = render(spec, charts={0: cd})
    chart = next(s.chart for s in prs.slides[1].shapes if s.has_chart)
    assert list(chart.plots[0].categories) == ["วัตถุดิบ", "อื่นๆ"]
    assert list(chart.plots[0].series[0].values) == [9000.0, 370.5]


@pytest.mark.parametrize("style, expected", [("bar", XL_CHART_TYPE.COLUMN_CLUSTERED),
                                             ("line", XL_CHART_TYPE.LINE_MARKERS),
                                             ("pie", XL_CHART_TYPE.PIE)])
def test_chart_styles(style, expected):
    spec = {"title": "t", "slides": [{"kind": "chart", "title": "c", "chart": {**Q, "style": style}}]}
    prs = render(spec, charts={0: ChartData(["a", "b"], {"THB": [1.0, 2.0]})})
    assert next(s.chart for s in prs.slides[1].shapes if s.has_chart).chart_type == expected


def test_empty_chart_shows_the_placeholder():
    spec = {"title": "t", "slides": [{"kind": "chart", "title": "c", "chart": Q}]}
    prs = render(spec, charts={0: ChartData([], {})})
    assert not any(s.has_chart for s in prs.slides[1].shapes)
    assert "ไม่มีข้อมูลในช่วงนี้" in " ".join(texts(prs.slides[1]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_deck.py -q`
Expected: FAIL (`ImportError: chart_data`).

- [ ] **Step 3: Implement `ChartData`, `chart_data` and chart rendering in `src/salid/deck.py`**

- `chart_data`: keep `summary["groups"]` order; non-day: first `top` groups, the rest summed per currency into one `t("deck_other", lang)` category (only if any remain); day: all groups, labels `MM-DD`; `None`/empty keys become `"—"`; currencies sorted with `THB` first; missing currency values are 0.
- Chart slide: `CategoryChartData` with one series per currency (series name = currency code), `slide.shapes.add_chart(...)` in the body area; types: bar -> `COLUMN_CLUSTERED`, line -> `LINE_MARKERS`, pie -> `PIE` (first currency only). Number format `#,##0` and data labels on; legend only with 2+ series or for pie; series colours from `CHART_COLORS`; chart text fonts set via the same Latin/Thai helper where python-pptx exposes text properties (`chart.font`). Empty data: a centred `t("deck_no_data", lang)` text instead of a chart. Missing `charts[i]` for a chart slide is treated as empty.

- [ ] **Step 4: Create `scripts/salid_sample_deck.py`**

Renders a fixed sample Thai deck (title, section, bullets with notes, two-column, bar chart with sample `ChartData`, pie chart) to the path given as argv[1] (default `/tmp/salid-sample-deck.pptx`) and prints the path and byte size. Offline: no model, no store. Run: `uv run python scripts/salid_sample_deck.py /tmp/salid-sample-deck.pptx`; expected: prints the path and a size under 2 MB.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/salid/test_deck.py -q && uv run pytest -q && uv run python scripts/salid_sample_deck.py /tmp/salid-sample-deck.pptx`
Expected: PASS; suite green; sample written.

- [ ] **Step 6: Commit**

```bash
git add src/salid/deck.py src/salid/i18n.py scripts/salid_sample_deck.py tests/salid/test_deck.py
git commit -m "Salid: native deck charts from ledger summaries, sample deck script"
```

---

### Task 4: `create_deck` tool, `Answer.files`, prompt

**Files:**
- Modify: `src/salid/qa.py`
- Test: `tests/salid/test_salid_qa.py`

**Interfaces:**
- Consumes: `deck.DeckSpec`, `deck.Slide`, `deck.chart_data`, `deck.render_deck`, `deck.deck_filename`, `deck.MAX_DECK_BYTES`; `LedgerTools.spending_summary(start_date, end_date, group_by)`.
- Produces: `DeckSink` (dataclass: `files: list[tuple[str, bytes]]`, `lang: str`, `today: date`); `build_tools(ledger, calls=None, memory=None, timings=None, deck: DeckSink | None = None)` adds `create_deck(title: str, slides: list[Slide], subtitle: str = "") -> dict` when `deck` is given; `Answer.files: list[tuple[str, bytes]] = []`; `ask(..., deck: bool = False)` and `ask_async(..., deck: bool = False)`; `stable_prompt(language, memory, deck=False)`; `DECK_PROMPT`.

- [ ] **Step 1: Write the failing tests** (`tests/salid/test_salid_qa.py`, `seeded` ledger from conftest: Alice 1000 + 250.5, Bob 9000 + 3000 + 120)

```python
DECK = {"title": "สรุปค่าใช้จ่าย", "slides": [
    {"kind": "chart", "title": "ตามหมวด",
     "chart": {"group_by": "category", "start_date": "2026-10-01", "end_date": "2026-10-31"}}]}


def deck_ask(settings, store, *script, is_exec=True, user_id=EXEC_ID, question="ทำสไลด์สรุปค่าใช้จ่ายเดือนนี้", language="th"):
    llm = scripted(*script)
    return llm, ask(settings, store, question, guild_id=GUILD, is_exec=is_exec, user_id=user_id,
                    model=llm, deck=True, language=language)


def chart_values(data):
    prs = Presentation(io.BytesIO(data))
    chart = next(s.chart for sl in prs.slides for s in sl.shapes if s.has_chart)
    return dict(zip(chart.plots[0].categories, chart.plots[0].series[0].values))


def test_create_deck_attaches_a_pptx_with_ledger_numbers(settings, seeded):
    _, a = deck_ask(settings, seeded, fc_resp(("create_deck", DECK)), text_resp("ทำสไลด์ให้แล้วครับ"))
    [(name, data)] = a.files
    assert name.endswith(".pptx") and data[:2] == b"PK"
    assert chart_values(data) == {"วัตถุดิบ": 10120.0, "ค่าเดินทาง": 250.5}


def test_a_non_executive_chart_covers_only_their_own_receipts(settings, seeded):
    _, a = deck_ask(settings, seeded, fc_resp(("create_deck", DECK)), text_resp("ok"),
                    is_exec=False, user_id=ALICE)
    assert chart_values(a.files[0][1]) == {"วัตถุดิบ": 1000.0, "ค่าเดินทาง": 250.5}


def test_an_invalid_outline_returns_an_error_and_a_fixed_call_succeeds(settings, seeded):
    bad = {"title": "t", "slides": [{"kind": "chart", "title": "c"}]}
    llm, a = deck_ask(settings, seeded, fc_resp(("create_deck", bad)), fc_resp(("create_deck", DECK)),
                      text_resp("ok"))
    first = [p.function_response.response for c in llm.requests[1].contents for p in c.parts or []
             if p.function_response][0]
    assert "chart" in json.dumps(first, ensure_ascii=False) and "error" in json.dumps(first)
    assert len(a.files) == 1


def test_a_second_create_deck_replaces_the_first(settings, seeded):
    other = {"title": "Second", "slides": [{"kind": "bullets", "title": "b", "bullets": ["x"]}]}
    _, a = deck_ask(settings, seeded, fc_resp(("create_deck", DECK)), fc_resp(("create_deck", other)),
                    text_resp("ok"))
    assert [n.split("-")[0] for n, _ in a.files] == ["second"]


def test_deck_fixed_strings_follow_the_reply_language(settings, seeded):
    _, a = deck_ask(settings, seeded, fc_resp(("create_deck", DECK)), text_resp("done"),
                    question="make slides of this month's spending", language="auto")
    cover = Presentation(io.BytesIO(a.files[0][1])).slides[0]
    assert "Made with Salid" in " ".join(s.text_frame.text for s in cover.shapes if s.has_text_frame)


def test_no_deck_tool_without_deck_and_prompt_only_with_it(settings, seeded):
    llm = scripted(text_resp("hi"))
    ask(settings, seeded, "hi", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    names = {d.name for t_ in llm.requests[0].config.tools for d in t_.function_declarations}
    assert "create_deck" not in names and "create_deck" not in llm.requests[0].config.system_instruction
    llm2, _ = deck_ask(settings, seeded, text_resp("hi"))
    decl = {d.name: d for t_ in llm2.requests[0].config.tools for d in t_.function_declarations}["create_deck"]
    assert '"kind"' in json.dumps(decl.parameters_json_schema)  # the nested slide schema is visible
    assert "create_deck" in llm2.requests[0].config.system_instruction
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_salid_qa.py -k deck -q`
Expected: FAIL (`TypeError: ask() got an unexpected keyword argument 'deck'`).

- [ ] **Step 3: Implement in `src/salid/qa.py`**

- `DeckSink` dataclass; `Answer.files` field (default empty list).
- In `build_tools`, when `deck` is given, add `create_deck` on both return paths (today the function returns early when `memory is None`; the deck tool must not depend on memory being on). Its docstring states purpose, the four slide kinds and that chart numbers are filled by the system). Body through the existing `safe(...)` wrapper (so validation errors, which are `ValueError`s, come back as `{"error": ...}` and timing is recorded): `spec = DeckSpec.model_validate({"title": title, "subtitle": subtitle, "slides": slides})` (slides may arrive as dicts or `Slide` objects; dump objects with `model_dump()` first); for each chart slide `i`, `chart_data(ledger.spending_summary(q.start_date, q.end_date, q.group_by), q, deck.lang)`; `render_deck(spec, charts, lang=deck.lang, today=deck.today)` inside `try/except Exception` returning `{"error": "could not build the deck"}` (log exception type only); over `MAX_DECK_BYTES` returns the same error; on success `deck.files[:] = [(deck_filename(spec.title, deck.today), data)]` and return `{"status": "created", "slides": len(spec.slides) + 1, "file": name}`; log `deck guild=%s slides=%d charts=%d bytes=%d ms=%d`. Truncate validation error text to 1500 chars.
- If the declaration test shows ADK cannot express `list[Slide]`, annotate `slides: list[dict]` and put the slide schema in the docstring; validation stays the same. Record the choice in the commit message.
- `ask_async(..., deck: bool = False)`: when true, `sink = DeckSink([], resolve_lang(language, question), (now or settings.now()).date())`, pass to `build_tools`, pass `deck=True` to `build_agent` -> `system_prompt` -> `stable_prompt(language, memory, deck)`, return `files=sink.files`. `ask` forwards `deck`.
- `DECK_PROMPT` (appended in `stable_prompt` after `MEMORY_PROMPT` when `deck`), exactly:

```python
DECK_PROMPT = (
    "\nสไลด์นำเสนอ: เรียก create_deck เฉพาะเมื่อผู้ใช้ขอสไลด์ งานนำเสนอ พรีเซนต์ หรือ slides / deck / presentation. "
    "ถ้าเป็นเรื่องข้อมูลของทีม ให้เก็บข้อมูลด้วยเครื่องมือไม่เกิน 2 รอบ แล้วเรียก create_deck. "
    "หน้าปกระบบสร้างให้เอง: ใส่สไลด์เนื้อหา 4-8 หน้า หน้าละหนึ่งประเด็น bullet ไม่เกิน 5 ข้อ ข้อละสั้นๆ รายละเอียดใส่ใน notes. "
    "เรื่องค่าใช้จ่ายให้ใช้สไลด์ชนิด chart (ระบบเติมตัวเลขจากบัญชีเอง) ห้ามพิมพ์ตัวเลขของร้านลงในกราฟเอง. "
    "เขียนข้อความในสไลด์เป็นภาษาเดียวกับคำตอบ. ถ้าได้ error ให้แก้ตามที่บอกแล้วเรียกใหม่. "
    "เมื่อสร้างเสร็จ ตอบสั้นๆ ว่าสไลด์มีอะไรบ้าง (ไฟล์แนบไปกับคำตอบแล้ว)."
)
```

- `CAPABILITIES_PROMPT` gains, before the "ข้อจำกัด" sentence: `"9) สไลด์นำเสนอ: ขอให้ทำสไลด์ได้ เช่น “ทำสไลด์สรุปค่าใช้จ่ายเดือนนี้” หรือ “ทำสไลด์สรุปประชุมเมื่อวาน” สลิดส่งไฟล์ PowerPoint (.pptx) ที่แก้ต่อได้ กราฟใช้ตัวเลขจริงจากบัญชี\n"`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/salid/test_salid_qa.py -q && uv run pytest -q`
Expected: PASS; suite green (the existing stable-prompt test still holds: the deck prompt is part of the stable prefix).

- [ ] **Step 5: Commit**

```bash
git add src/salid/qa.py tests/salid/test_salid_qa.py
git commit -m "Salid: create_deck tool with data-bound charts; Answer.files"
```

---

### Task 5: Feature flag, hub toggle, bot attachment

**Files:**
- Modify: `src/salid/guild_settings.py`, `src/meeting_scribe/static/hub.html`, `src/salid/bot.py`, `src/salid/i18n.py`
- Test: `tests/salid/test_salid_features.py`, `tests/salid/test_guild_settings.py`, `tests/test_hub.py`

**Interfaces:**
- Consumes: Task 4 `ask(..., deck=)` and `Answer.files`.
- Produces: `Features.decks: bool = True`; `"decks"` in `FEATURES`; i18n `deck_attach_failed` (`{"th": "(แนบไฟล์สไลด์ไม่ได้ อาจเพราะสลิดไม่มีสิทธิ์แนบไฟล์ในห้องนี้)", "en": "(I couldn't attach the slide file. I may not have permission to attach files here.)"}`).

- [ ] **Step 1: Write the failing tests** (`tests/salid/test_salid_features.py`; `_real_answer` stubs `botmod.ask`)

```python
def _deck_answer(bot, monkeypatch, files=(("deck-20261016.pptx", b"PK\x03\x04data"),)):
    seen = []
    monkeypatch.setattr(botmod, "ask", lambda s, store, q, **kw: seen.append(kw) or SimpleNamespace(
        text="ทำสไลด์ให้แล้วครับ", seconds=0, rounds=0, files=list(files)))
    bot.__dict__.pop("_answer", None)
    return seen


async def test_the_deck_file_is_attached_to_the_reply(bot, world, monkeypatch):
    seen = _deck_answer(bot, monkeypatch)
    m = say(world, world.gen, "สลิด ทำสไลด์สรุปค่าใช้จ่าย")
    sent = []
    async def reply(text, **kw):
        sent.append((text, kw.get("files")))
        return Sent()
    m.reply = reply
    await bot._answer(m, world.g, m.author)
    assert seen[0]["deck"] is True
    [(text, files)] = sent
    assert text == "ทำสไลด์ให้แล้วครับ" and [f.filename for f in files] == ["deck-20261016.pptx"]


async def test_text_is_still_posted_when_the_file_cannot_be_attached(bot, world, monkeypatch):
    _deck_answer(bot, monkeypatch)
    m = say(world, world.gen, "สลิด ทำสไลด์")
    sent = []
    async def reply(text, **kw):
        if kw.get("files"):
            raise discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions")
        sent.append(text)
        return Sent()
    m.reply = reply
    await bot._answer(m, world.g, m.author)
    assert sent and sent[0].startswith("ทำสไลด์ให้แล้วครับ") and "แนบไฟล์สไลด์ไม่ได้" in sent[0]


async def test_no_deck_for_interjections_or_when_the_flag_is_off(bot, world, monkeypatch):
    seen = _deck_answer(bot, monkeypatch, files=())
    m = say(world, world.gen, "ยอดเท่าไร")
    await bot._answer(m, world.g, m.author, interjection=True)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(decks=False)), "x")
    bot.team._cache.clear()
    await bot._answer(say(world, world.gen, "สลิด ทำสไลด์"), world.g, m.author)
    assert [kw["deck"] for kw in seen] == [False, False]
```

Also update the two full-dict assertions to include `decks=True`: `tests/salid/test_guild_settings.py` (around line 40) and `tests/test_hub.py` (around line 267).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/salid/test_salid_features.py -k deck -q tests/salid/test_guild_settings.py tests/test_hub.py`
Expected: FAIL (`KeyError: 'deck'`; feature dicts missing `decks`).

- [ ] **Step 3: Implement**

- `guild_settings.py`: `decks: bool = True` in `Features`, `"decks"` in `FEATURES`, docstring line listing features.
- `hub.html`: `{ id: "decks" }` in `FEATURES` (after `deep_tasks`), `decks: true` in the defaults object, and text keys `f_decks_n` (th "สไลด์นำเสนอ", en "Slide decks"), `f_decks_d` (th "ขอให้สลิดทำสไลด์ PowerPoint จากข้อมูลของทีมหรือหัวข้อใดก็ได้ กราฟใช้ตัวเลขจริงจากบัญชี ไฟล์ไม่ถูกเก็บไว้", en "Ask Salid for a PowerPoint deck from team data or any topic. Charts use real ledger numbers. Files are not stored."), `f_decks_w` (th "แท็กสลิดแล้วพิมพ์ “ทำสไลด์…”", en "Tag Salid and ask “make slides…”").
- `bot._answer`: pass `deck=self.team.enabled(gid, "decks") and not interjection` to `ask`. Build `files = [discord.File(io.BytesIO(data), filename=name) for name, data in getattr(ans, "files", ())]`; the first `message.reply` gets `files=files` when present. On `discord.HTTPException` for that call, reply again with the text plus `"\n" + t("deck_attach_failed", lang)` and no files; log the HTTP status only.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q`
Expected: suite green.

- [ ] **Step 5: Commit**

```bash
git add src/salid/guild_settings.py src/meeting_scribe/static/hub.html src/salid/bot.py src/salid/i18n.py tests/salid/test_salid_features.py tests/salid/test_guild_settings.py tests/test_hub.py
git commit -m "Salid: decks feature flag and hub toggle; attach deck files to replies"
```

---

### Task 6: Docs, live test, verification, PR

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `tests/salid/test_salid_live.py`

- [ ] **Step 1: Docs**
  - README Salid section: a "Slide decks" paragraph (how to ask, what is built, data-bound charts, scope, not stored, feature flag `decks`, sample script).
  - CLAUDE.md Salid table: a row for `deck.py` (outline models, `chart_data`, `render_deck`, Thai font slot, pure); the `qa.py` row mentions `create_deck` / `Answer.files`; the guild-settings feature list includes `decks`.

- [ ] **Step 2: Live test** (marked `live`, deselected by default): seed a temp SQLite ledger with two receipts, call real `ask(..., deck=True)` with "ทำสไลด์สรุปค่าใช้จ่ายเดือนตุลาคม 2026 แยกตามหมวด"; assert one `.pptx` in `ans.files`, it opens with python-pptx, and it contains a chart. This also proves Gemini accepts the `create_deck` schema.

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run python scripts/salid_sample_deck.py /tmp/salid-sample-deck.pptx`
Expected: suite green, ruff clean, sample written.

- [ ] **Step 4: HUMAN GATES** (ask before each)
  - Visual check: the human opens `/tmp/salid-sample-deck.pptx` in PowerPoint, Keynote and Google Slides (Thai text, layout, charts).
  - Live test: `uv run pytest tests/salid/test_salid_live.py -m live -k deck -s` (a few Vertex calls).

- [ ] **Step 5: Commit, push, PR** (push and PR only after the human agrees)

```bash
git status && git diff --cached
git add README.md CLAUDE.md tests/salid/test_salid_live.py
git commit -m "Docs: Salid slide decks; live deck test"
git push -u origin feature/salid-decks
gh pr create --base main --title "Salid: slide decks (.pptx) from chat" --body "<summary, tests, visual check, live test result>"
```
