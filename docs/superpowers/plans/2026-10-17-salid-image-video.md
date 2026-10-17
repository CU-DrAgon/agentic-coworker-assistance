# Salid image generation and hybrid video ads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Salid gains a `create_image` tool (text-to-image with Thai text typeset by Salid) and a `create_video_ad` background job that delivers finished 15/20/30 s 1080x1920 ads (scene stills or shop photos with motion, Thai narration and captions, optional Omni hero clips and music).

**Architecture:** `imagegen.py` holds the one Nano Banana call, safety/block handling, the English-only guard, the decode guard and the `create_image` tool; promo kits reuse it. `videoad.py` is the pure video pipeline (storyboard → stills → motion clips → captions → narration → ffmpeg assembly → job runner); `videogen.py` wraps Omni/Lyria behind empty-by-default settings. `imagegen_bot.py` / `videoad_bot.py` are the Discord mixins (flags, limits, progress message, Cancel button, delivery, DocStore records).

**Tech Stack:** Python 3.12, google-genai 2.29.0 (`generate_content`, `client.interactions`), Pillow 12.3 (RAQM), ffmpeg/ffprobe, discord.py, DocStore (SQLite/Firestore), pytest (offline).

**Spec:** `docs/superpowers/specs/2026-10-17-salid-image-video-design.md`

## Global Constraints

- Models never get Thai text: every prompt sent to an image/video/music model passes `imagegen.english_only`; Thai is typeset with `salid.cards` (RAQM).
- Image model `settings.image_model = "gemini-nano-banana-2.1"`; `ImageConfig(aspect_ratio, image_size, person_generation="ALLOW_ADULT", prominent_people="BLOCK_PROMINENT_PEOPLE")`; no temperature/top_p/top_k/seed/logprobs; one attempt per call (`delays=()`), 60 s timeout.
- Omni (`video_hero_model`) and music (`video_music_model`) default to `""` = off; with both empty every video request still produces a finished ad.
- Video output: 1080x1920, 30 fps, H.264 High yuv420p, AAC 128 kbps 48 kHz, `+faststart`; duration = requested seconds ± 0.2 s (scene totals within ± 1 s); size ≤ `min(19, guild_limit_mib - 1)` MiB.
- Caption safe area for 1080x1920: y ∈ [270, 1250], x ≥ 65 from each side.
- Limits: images 20 per guild per day and 5 per user per day (in memory); video 10 ads per guild per month, 60 hero seconds per month (DocStore, survives restarts); one video job per guild, `video_max_jobs = 2` across the bot.
- Flags: `images` default True; `video_ads` default **False** (first opt-in flag: hub default `false`).
- Logs: guild/step/ms/cost/outcome only — never prompts, captions, narration, images or video.
- Fixed user-facing strings go in `src/salid/i18n.py` (`t(key, lang)`, Thai and English).
- `src/salid/qa.py` is NOT changed by this plan (PR #26 is still open on that code); moving `qa._gif_first_frame` onto `imagegen.open_capped` is a follow-up after #26 merges.
- No live calls in tasks; live tests are marked `live` and run only with human approval by someone with GCP project access.
- Test command: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib SALID_PDF_FONT=/System/Library/Fonts/Supplemental/Ayuthaya.ttf pytest <paths> -q -p no:warnings`; lint `uvx ruff check . && uvx ruff format --check .`.
- Commits: `git status` and `git diff --cached` first; never stage `.env`, `samples/`, key `*.json`.

## Review Focus

1. **Narration longer than its scene**: the audio never runs past the video; the final duration stays within ± 0.2 s (test in Task 5).
2. **Shop photo with an extreme aspect** (panorama, very tall screenshot): framed into 9:16 without distortion, on a blurred fill (test in Task 4).
3. **A guild whose upload limit is below the default**: the ad is encoded under `guild_limit - 1` MiB (test in Task 5 with a 6 MiB budget).
4. **Cancel while a paid step is running**: no further paid calls start, usage counts only started steps, the record ends `cancelled` (test in Task 6).
5. **Bot restart mid-job**: the stored record is marked `failed` with reason `restarted`, the requester is told, and the monthly count is neither refunded nor double-counted (test in Task 6).

---

### Task 1: Shared image generation core (`imagegen.py`) and promo on top of it

**Files:**
- Create: `src/salid/imagegen.py`
- Modify: `src/salid/promo.py` (`generate_scene` delegates; `prepare_photo` uses `open_capped`; `PhotoTooLarge = imagegen.ImageTooLarge`)
- Modify: `src/salid/config.py` (`image_model: str = "gemini-nano-banana-2.1"`)
- Test: `tests/salid/test_imagegen.py`; existing `tests/salid/test_promo.py` must stay green

**Interfaces:**
- Produces:
  - `MAX_DECODE_PIXELS = 50_000_000`; `class ImageTooLarge(ValueError)`
  - `open_capped(data: bytes, *, formats: list[str] | None = None, max_pixels: int | None = None) -> Image.Image` (opened, not loaded; raises `ImageTooLarge` from the header or on Pillow's `DecompressionBombError`; other decode errors raise `ValueError("unreadable image")`)
  - `english_only(text: str) -> bool` (False if any U+0E00–U+0E7F character)
  - `BLOCKED: set[types.FinishReason]` (IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION, IMAGE_OTHER, NO_IMAGE, SAFETY, PROHIBITED_CONTENT)
  - `generate_images(client, model: str, prompt: str, *, refs: Sequence[bytes] = (), aspect: str = "1:1", size: str = "1K", count: int = 1) -> tuple[list[Image.Image], str]` (reason `""` if ≥ 1 image, else `"blocked" | "timeout" | "error"`; `count` calls in parallel via a ThreadPoolExecutor of `min(count, 4)`)
  - `IMAGE_TIMEOUT_MS = 60_000`

- [ ] **Step 1: Write the failing tests** (`tests/salid/test_imagegen.py`, reuse `promo_helpers.fake_client`, `image_response`, `text_response`, `png_bytes`)

```python
def test_request_config_safety_and_no_sampling_params():
    c = fake_client(image_response(png_bytes((768, 1376))))
    imgs, reason = imagegen.generate_images(
        c, "gemini-nano-banana-2.1", "a cup", aspect="9:16", size="2K"
    )
    assert reason == "" and imgs[0].size == (768, 1376)
    model, contents, cfg = c.models.calls[0]
    ic = cfg.image_config
    assert (ic.aspect_ratio, ic.image_size, ic.person_generation, ic.prominent_people) == (
        "9:16",
        "2K",
        "ALLOW_ADULT",
        "BLOCK_PROMINENT_PEOPLE",
    )
    assert cfg.response_modalities == ["TEXT", "IMAGE"] and cfg.http_options.timeout == 60_000
    assert all(
        getattr(cfg, k) is None for k in ("temperature", "top_p", "top_k", "seed", "logprobs")
    )


def test_count_makes_parallel_calls_and_keeps_what_came_back():
    ok = image_response(png_bytes((64, 64)))
    c = fake_client(ok, RuntimeError("x"), ok)
    imgs, reason = imagegen.generate_images(c, "m", "p", count=3)
    assert len(c.models.calls) == 3 and len(imgs) == 2 and reason == ""


def test_refs_go_before_the_prompt():
    c = fake_client(image_response(png_bytes((64, 64))))
    imagegen.generate_images(c, "m", "p", refs=[png_bytes((10, 10))])
    contents = c.models.calls[0][1]
    assert contents[0].inline_data is not None and contents[-1] == "p"


@pytest.mark.parametrize(
    "finish",
    [
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "NO_IMAGE",
        "SAFETY",
        "PROHIBITED_CONTENT",
    ],
)
def test_blocked_finish_reasons(finish):
    r = image_response(png_bytes((9, 9)), finish=getattr(types.FinishReason, finish))
    assert imagegen.generate_images(fake_client(r), "m", "p") == ([], "blocked")


def test_prompt_block_text_only_timeout_and_error():
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    assert imagegen.generate_images(fake_client(blocked), "m", "p") == ([], "blocked")
    assert imagegen.generate_images(fake_client(text_response("no")), "m", "p") == ([], "blocked")
    assert imagegen.generate_images(fake_client(httpx.ReadTimeout("t")), "m", "p") == (
        [],
        "timeout",
    )
    assert imagegen.generate_images(fake_client(RuntimeError("x")), "m", "p") == ([], "error")
    assert imagegen.generate_images(None, "m", "p") == ([], "error")


def test_thought_images_are_skipped():
    thought = types.Part.from_bytes(data=png_bytes((8, 8)), mime_type="image/png")
    thought.thought = True
    final = types.Part.from_bytes(data=png_bytes((32, 32)), mime_type="image/png")
    r = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[thought, final]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    assert imagegen.generate_images(fake_client(r), "m", "p")[0][0].size == (32, 32)


def test_english_only():
    assert imagegen.english_only("a latte on a table") and not imagegen.english_only(
        "ลาเต้ on a table"
    )


def test_open_capped_refuses_big_canvases_before_decoding(monkeypatch):
    data = png_bytes((1200, 1200))
    monkeypatch.setattr(Image.Image, "load", lambda self: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(imagegen.ImageTooLarge, match="megapixels"):
        imagegen.open_capped(data, max_pixels=1_000_000)


def test_open_capped_maps_pillow_bomb_errors_and_formats(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100_000)
    with pytest.raises(imagegen.ImageTooLarge):
        imagegen.open_capped(png_bytes((1200, 1200)))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="unreadable"):
        imagegen.open_capped(png_bytes((10, 10)), formats=["GIF"])


def test_promo_scene_goes_through_imagegen(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        promo.imagegen,
        "generate_images",
        lambda c, m, p, **k: seen.update(k) or ([Image.new("RGB", (9, 16))], ""),
    )
    img, reason = promo.generate_scene(object(), "m", promo_photo(), "cafe")
    assert (
        reason == ""
        and seen["aspect"] == "9:16"
        and seen["size"] == "2K"
        and len(seen["refs"]) == 1
    )
```
(`promo_photo()` = `promo.prepare_photo(png_bytes((800, 800)))`; imports `httpx`, `pytest`, `types`, `Image`, `imagegen`, `promo`.)

- [ ] **Step 2: Run to verify they fail**
Run: `… pytest tests/salid/test_imagegen.py -q -p no:warnings` — Expected: `ModuleNotFoundError: No module named 'salid.imagegen'`.

- [ ] **Step 3: Implement `imagegen.py`; refactor `promo.py`**
`generate_images` builds `GenerateContentConfig(response_modalities=["TEXT","IMAGE"], image_config=…, http_options=HttpOptions(timeout=IMAGE_TIMEOUT_MS))` and calls `llm.generate(..., delays=())` once per image; picks the last non-thought `inline_data` image part; maps exceptions (`httpx.TimeoutException` → timeout, others → error, WARNING log with the type name only). `promo.generate_scene` becomes `imgs, reason = imagegen.generate_images(client, model, scene_prompt(scene), refs=[photo.jpeg], aspect="9:16", size="2K"); return (imgs[0] if imgs else None, reason)` keeping `PROMO_RETRY_DELAYS` unused-free (remove it). `promo.prepare_photo` calls `imagegen.open_capped(data, max_pixels=MAX_PIXELS)` (read at call time so the existing monkeypatch test works); `PhotoTooLarge = imagegen.ImageTooLarge`.

- [ ] **Step 4: Run the tests**
Run: `… pytest tests/salid/test_imagegen.py tests/salid/test_promo.py tests/salid/test_promo_bot.py -q -p no:warnings` — Expected: all passed. If a promo test pinned the old retry delay or call shape, update it to the new contract and record the change in the commit message.

- [ ] **Step 5: Commit**
```bash
git add src/salid/imagegen.py src/salid/promo.py src/salid/config.py tests/salid/test_imagegen.py tests/salid/test_promo.py
git commit -m "Salid: shared image generation core (imagegen) with promo kits on top"
```

---

### Task 2: `create_image` tool, overlay, memory, limits and bot wiring

**Files:**
- Modify: `src/salid/imagegen.py` (tool, memory, overlay)
- Create: `src/salid/imagegen_bot.py` (`ImageMixin`)
- Modify: `src/salid/bot.py` (mixin in `SecretaryBot` bases before `PromoMixin`; `self._init_images()` after `self._init_promo()`; `*([] if interjection else self._image_tools(message, guild, asker, files)),` right after the `_promo_tools` entry)
- Modify: `src/salid/guild_settings.py` (`images` flag, default True), `src/meeting_scribe/static/hub.html` (toggle + `f_images_n/_d/_w`, in the 3 feature literals), `src/salid/config.py` (`images_per_day: int = 20`, `images_per_user_day: int = 5`), `src/salid/qa.py` is NOT touched — the capability line goes in the tool docstring only
- Modify: `README.md`, `CLAUDE.md` (rows for `imagegen.py` / `imagegen_bot.py`; settings), `tests/salid/test_guild_settings.py`, `tests/test_hub.py` (defaults gain `images=True`)
- Test: `tests/salid/test_imagegen.py` (tool), `tests/salid/test_imagegen_bot.py`

**Interfaces:**
- Consumes: Task 1 (`generate_images`, `english_only`, `open_capped`); `cards` (`fit_text`, `draw_lines`, `scrim`, `measure_badge`, `draw_badge`, `drawable`, `card_fonts`, `thai_ready`); `promo.ACCENTS`; `media.DailyBudget`.
- Produces:
  - `ASPECTS = ("1:1", "4:5", "9:16", "16:9")`; `SIZES = ("1K", "2K")`; `MAX_IMAGES = 4`; `PROMPT_MAX = 600`
  - `overlay(img: Image.Image, headline: str, subtitle: str, accent: str, fonts: CardFonts, *, top: bool) -> tuple[Image.Image, list[tuple[str, tuple[int, int, int, int]]]]` (sizes relative to width: margin `0.06·w`, headline sizes `(round(0.085·w), round(0.04·w))` ≤ 2 lines, subtitle `(round(0.05·w), round(0.03·w))` in an accent pill; scrim covers the band; returns placed boxes)
  - `class ImageMemory(ttl_s=1800.0, max_entries=50, clock=time.monotonic)`: `put(key, images: list[bytes]) -> None`, `get(key) -> list[bytes] | None` (PNG bytes)
  - `@dataclass class ImageCtx(guild_id: int, user_id: int, day: str, photos: list[tuple[int, bytes]], memory: ImageMemory, key: tuple[int, int, int], client: Any, model: str, guild_budget: DailyBudget, user_budget: DailyBudget, fonts: CardFonts, outbox: list = field(default_factory=list))` with `make(prompt, aspect, count, size, headline, subtitle, accent, file_index, revise) -> dict`
  - `image_tools(ctx: ImageCtx) -> list[Callable]` → `[create_image]` with `.outbox`
  - `ImageMixin._init_images()`, `ImageMixin._image_tools(message, guild, asker, files) -> list` (never raises; [] when `files is None`, flag off, `not cards.thai_ready()`)

Tool docstring (model-facing; exact copy):
```
Create images from a description (illustrations, posters, product scenes, social posts). Use for
"วาดรูป…", "สร้างภาพ…", "ทำโปสเตอร์…", "make an image of…". Write `prompt` IN ENGLISH, never
describing text; put Thai words the user wants on the image in `headline` / `subtitle` (Salid
typesets them). For a product photo the user attached, pass its number as `file_index`. To change
one of the last results ("เอาแบบที่ 2 แต่โทนอุ่นขึ้น") pass `revise` = its number (1-4) with the new
prompt; the last results are remembered for 30 minutes. Never depict real public figures or children.

Args:
    prompt: English description, max 600 characters.
    aspect: "1:1" (default), "4:5", "9:16" or "16:9".
    count: 1-4 variations (default 1).
    size: "1K" (default) or "2K".
    headline: optional text on the image, max 40 characters, in the user's language.
    subtitle: optional second line, max 60 characters.
    accent: subtitle pill colour: red, orange, green, blue, gold or black.
    file_index: attached photo to use as a reference; -1 = none.
    revise: number of a remembered result to edit; 0 = new images.
```
Result `{"status": "ok", "files": [...], "count_made": n, "reason": reason, "images_left_today": n, "note": NOTE}` with `NOTE = "Reply briefly in the user's language: the images are attached and were made with AI. Do not describe people in them as real."` plus `" Only {n} of {asked} could be made today (daily limit)."` when trimmed, plus the blocked/timeout sentence (`" The image service refused or failed for some of them; suggest a different description."`) when `count_made < count`. Errors: `{"status": "error", "error": ...}` for Thai in prompt (`"prompt must be English; put Thai text in headline/subtitle"`), bad aspect/size/count/accent, over-length fields, `file_index` not a usable photo, `revise` out of range or nothing remembered; `{"status": "limit", "note": "Today's image limit for this server (or this person) is used up."}` when nothing is left.

- [ ] **Step 1: Write the failing tests** (in `test_imagegen.py` and new `test_imagegen_bot.py`; fixtures like `test_promo_bot.py`)

```python
def ictx(**kw):
    img = image_response(png_bytes((768, 1376)))
    base = dict(
        guild_id=1,
        user_id=7,
        day="2026-10-17",
        photos=[],
        memory=imagegen.ImageMemory(),
        key=(1, 10, 7),
        client=fake_client(img, img, img, img),
        model="m",
        guild_budget=DailyBudget(20),
        user_budget=DailyBudget(5),
        fonts=cards.card_fonts(),
    )
    return imagegen.ImageCtx(**{**base, **kw})


def test_makes_count_files_and_counts_both_budgets():
    c = ictx()
    [tool] = imagegen.image_tools(c)
    out = tool(prompt="a cat in a cafe, watercolour", aspect="9:16", count=3)
    assert out["status"] == "ok" and out["files"] == ["image-1.png", "image-2.png", "image-3.png"]
    assert len(tool.outbox) == 3 and out["images_left_today"] == 2  # user cap 5 → 2 left


def test_thai_prompt_is_refused_before_any_call():
    c = ictx()
    out = imagegen.image_tools(c)[0](prompt="แมวในร้านกาแฟ")
    assert out["status"] == "error" and "English" in out["error"] and c.client.models.calls == []


def test_request_is_trimmed_to_what_is_left():
    ub = DailyBudget(5)
    ub.add(7, "2026-10-17", 4)
    out = imagegen.image_tools(ictx(user_budget=ub))[0](prompt="a cup", count=3)
    assert out["count_made"] == 1 and "Only 1 of 3" in out["note"]


def test_overlay_band_and_text_inside(monkeypatch):
    img = Image.new("RGB", (768, 1376), "white")
    out, placed = imagegen.overlay(
        img, "ลดพิเศษ 20%", "วันนี้เท่านั้น", "red", cards.card_fonts(), top=True
    )
    m = round(0.06 * 768)
    assert all(b[0] >= m and b[2] <= 768 - m and b[3] <= 1376 // 2 for _, b in placed)  # top half
    _, placed = imagegen.overlay(
        Image.new("RGB", (1024, 1024)), "h", "", "red", cards.card_fonts(), top=False
    )
    assert placed[0][1][1] >= 1024 // 2  # bottom half


def test_revise_sends_the_remembered_image_as_reference():
    c = ictx()
    tool = imagegen.image_tools(c)[0]
    tool(prompt="a cup", count=2)
    c2 = ictx(memory=c.memory)
    out = imagegen.image_tools(c2)[0](prompt="warmer light", revise=2)
    assert out["status"] == "ok" and c2.client.models.calls[0][1][0].inline_data is not None


def test_revise_without_memory_and_bad_fields_are_errors():
    t = imagegen.image_tools(ictx())[0]
    assert t(prompt="x", revise=1)["status"] == "error"
    assert t(prompt="x", aspect="3:2")["status"] == "error"
    assert t(prompt="x", count=5)["status"] == "error"
    assert t(prompt="x", headline="h" * 41)["status"] == "error"


def test_tool_never_raises_and_logs_no_prompt(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="salid.imagegen")
    imagegen.image_tools(ictx())[0](prompt="secret garden")
    assert "secret garden" not in caplog.text and "image guild=1" in caplog.text
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: 1 / 0)
    assert imagegen.image_tools(ictx())[0](prompt="x")["status"] == "error"
```
Bot tests (`test_imagegen_bot.py`, same imports/fixtures as `test_promo_bot.py`): offered with `files=[]` (no photo needed) for chat, not with `files=None`, not for interjections (via `_agent_kwargs`), not with the flag off, not when `thai_ready` is False; budgets keyed by guild and by user; `qa.tool_files([tool]) == tool.outbox`; `"images" in FEATURES and Features().images is True`.

- [ ] **Step 2: Run to verify they fail** — Expected: `AttributeError: module 'salid.imagegen' has no attribute 'ImageCtx'` / missing `_image_tools`.

- [ ] **Step 3: Implement** the tool (validate → budget take per image before the calls, user budget keyed by `user_id` in its own `DailyBudget`; refs = the chosen attached photo via `open_capped` re-encoded PNG, or the remembered image for `revise`; `generate_images`; optional `overlay` with `top = aspect == "9:16"`; PNG outbox replaced on each call; memory updated; one INFO line `image guild=%s made=%d asked=%d ms=%d`), the mixin (budgets from settings; key `(guild.id, message.channel.id, asker.id)`; default `files=None` check), flag, hub copy (th: name "สร้างภาพจากคำบรรยาย", desc "พิมพ์บรรยายภาพที่อยากได้ สลิดสร้างภาพให้ 1-4 แบบ (1:1, 4:5, 9:16, 16:9) ใส่ข้อความภาษาไทยบนภาพได้ แก้ต่อได้ภายใน 30 นาที", warn "ไม่เกิน 20 ภาพต่อวันต่อเซิร์ฟเวอร์ และ 5 ภาพต่อคน"; en: "Images from a description", "Describe the image you want: Salid makes 1-4 versions (1:1, 4:5, 9:16, 16:9), can put Thai text on them, and revises them for 30 minutes", "Up to 20 images per server and 5 per person per day"), docs.

- [ ] **Step 4: Run** `… pytest tests/salid -q -p no:warnings` then `… pytest tests/test_hub.py -q -p no:warnings` — Expected: all passed.

- [ ] **Step 5: Commit**
```bash
git add src/salid/imagegen.py src/salid/imagegen_bot.py src/salid/bot.py src/salid/guild_settings.py src/salid/config.py src/meeting_scribe/static/hub.html README.md CLAUDE.md tests/salid/test_imagegen.py tests/salid/test_imagegen_bot.py tests/salid/test_guild_settings.py tests/test_hub.py
git commit -m "Salid: create_image tool (Nano Banana 2.1, Thai overlay, revisions, daily limits)"
```

---

### Task 3: Storyboard model and planning (`videoad.py` part 1)

**Files:**
- Create: `src/salid/videoad.py`
- Test: `tests/salid/test_videoad.py`

**Interfaces:**
- Consumes: `imagegen.english_only`; `llm.generate`; `cards.break_ok`.
- Produces:
  - `LENGTHS = (15, 20, 30)`; `CPS = 12.0` (Thai narration chars per second, starting value)
  - `class Scene(BaseModel)`: `image_prompt_en: str` (≤ 300), `caption_th: str` (≤ 40), `voice_th: str` (≤ 120), `seconds: float` (2–8), `source: str` (`"ai"`, `"photo:N"`, `"hero"`)
  - `class Storyboard(BaseModel)`: `title: str` (≤ 60), `scenes: list[Scene]` (4–6)
  - `class StoryboardError(ValueError)`
  - `check_storyboard(sb: Storyboard, *, seconds: int, photos: int, hero_shots: int) -> list[str]` (problems: total outside `seconds ± 1`, Thai in a prompt, `photo:N` with N ≥ photos, more hero scenes than `hero_shots`)
  - `fit_narration(text: str, seconds: float, cps: float = CPS) -> str` (≤ `int(seconds·cps)` chars; cut at the last space, else at the last `cards.break_ok` boundary; never mid-syllable)
  - `plan_storyboard(client, model: str, *, brief: str, seconds: int, photos: int, hero_shots: int, lang: str) -> Storyboard` (JSON-schema output via `response_mime_type="application/json"` + `response_schema=Storyboard`; one re-ask with the problems listed; then `StoryboardError`; narration fitted per scene)
  - `STORYBOARD_PROMPT`: instructions including the safety rules from the spec (no celebrities/real public figures, no children, no health/cure claims for food/cosmetics/supplements, use the shop photos as `photo:N` when given, English image prompts that never describe text and contain no quotes, captions and voice in `lang`)

- [ ] **Step 1: Write the failing tests**

```python
def sb(*secs, src="ai", prompt="a latte on a table"):
    return videoad.Storyboard(
        title="t",
        scenes=[
            videoad.Scene(
                image_prompt_en=prompt, caption_th="ลาเต้", voice_th="หอมกรุ่น", seconds=s, source=src
            )
            for s in secs
        ],
    )


def test_check_storyboard_totals_thai_photos_and_hero_cap():
    assert videoad.check_storyboard(sb(5, 5, 5, 5), seconds=20, photos=0, hero_shots=0) == []
    assert videoad.check_storyboard(sb(5, 5, 5, 8), seconds=20, photos=0, hero_shots=0)  # 23 s
    assert videoad.check_storyboard(
        sb(5, 5, 5, 5, prompt="ลาเต้"), seconds=20, photos=0, hero_shots=0
    )
    assert videoad.check_storyboard(
        sb(5, 5, 5, 5, src="photo:2"), seconds=20, photos=1, hero_shots=0
    )
    assert videoad.check_storyboard(sb(5, 5, 5, 5, src="hero"), seconds=20, photos=0, hero_shots=1)


def test_scene_and_storyboard_limits():
    with pytest.raises(ValidationError):
        videoad.Scene(image_prompt_en="x", caption_th="ก" * 41, voice_th="", seconds=4, source="ai")
    with pytest.raises(ValidationError):
        videoad.Storyboard(title="t", scenes=sb(5, 5, 5).scenes)  # 3 scenes


def test_fit_narration_never_cuts_a_syllable():
    out = videoad.fit_narration("ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือน", 2.0, cps=12.0)
    assert len(out) <= 24 and cards.break_ok(out[-1], "ก")
    assert videoad.fit_narration("short", 5.0) == "short"


def test_plan_storyboard_reasks_once_then_fails():
    bad = json_response(sb(5, 5, 5, 8).model_dump())
    good = json_response(sb(5, 5, 5, 5).model_dump())
    c = fake_client(bad, good)
    out = videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=0, hero_shots=0, lang="th"
    )
    assert sum(s.seconds for s in out.scenes) == 20 and len(c.models.calls) == 2
    assert "total" in str(c.models.calls[1][1])  # problems were sent back
    with pytest.raises(videoad.StoryboardError):
        videoad.plan_storyboard(
            fake_client(bad, bad), "m", brief="x", seconds=20, photos=0, hero_shots=0, lang="th"
        )


def test_storyboard_request_asks_for_json_and_carries_the_rules():
    c = fake_client(json_response(sb(5, 5, 5, 5).model_dump()))
    videoad.plan_storyboard(c, "m", brief="latte", seconds=20, photos=2, hero_shots=1, lang="th")
    cfg = c.models.calls[0][2]
    assert cfg.response_mime_type == "application/json"
    assert "children" in cfg.system_instruction and "photo:" in cfg.system_instruction
```
(`json_response(obj)` = `text_response(json.dumps(obj, ensure_ascii=False))`; add it to `promo_helpers.py`.)

- [ ] **Step 2: Run to verify they fail** — Expected: `ModuleNotFoundError: No module named 'salid.videoad'`.
- [ ] **Step 3: Implement** per the Interfaces block (system instruction = `STORYBOARD_PROMPT` formatted with seconds, photos, hero_shots, lang; user content = the brief).
- [ ] **Step 4: Run** `… pytest tests/salid/test_videoad.py -q -p no:warnings` — Expected: all passed.
- [ ] **Step 5: Commit** `git add src/salid/videoad.py tests/salid/test_videoad.py tests/salid/promo_helpers.py && git commit -m "Salid: video ad storyboard model, checks and planning"`

---

### Task 4: Stills, framing, fallback card, motion clips and caption overlays

**Files:**
- Modify: `src/salid/videoad.py`
- Test: `tests/salid/test_videoad.py`

**Interfaces:**
- Consumes: Task 1 (`generate_images`, `open_capped`), `cards`, `promo.ACCENTS`, `promo.fallback_scene`-style blur fill.
- Produces:
  - `W, H, FPS = 1080, 1920, 30`; `SAFE = (65, 270, 1080 - 65, 1250)`; `AI_LABEL = "สร้างด้วย AI"`
  - `frame_photo(img: Image.Image) -> Image.Image` (exactly 1080x1920 RGB; cover crop when the source aspect is within 25 % of 9:16, else the photo contained on a blurred, darkened cover of itself)
  - `fallback_card(caption: str, accent: str, fonts: CardFonts) -> Image.Image` (1080x1920 accent background, caption centred in the safe area)
  - `scene_stills(client, model, scenes: list[Scene], photos: list[bytes], accent: str, fonts) -> tuple[list[Image.Image], int]` (AI scenes via `generate_images(..., aspect="9:16", size="2K")`, max 3 in parallel, one retry, then `fallback_card`; `photo:N` via `open_capped` + `frame_photo`; returns stills and the number of images billed)
  - `caption_png(caption: str, fonts, *, label: bool = True) -> tuple[Image.Image, tuple[int, int, int, int]]` (transparent 1080x1920 RGBA; caption with shadow inside `SAFE`; small `AI_LABEL` top-left inside `SAFE`; returns the caption box)
  - `zoompan_args(seconds: float, index: int) -> list[str]` (ffmpeg filter for a slow zoom-in on even, zoom-out on odd scenes, `fps=30`, `s=1080x1920`, `d = round(seconds·30)`)
  - `motion_clip(still: Path, seconds: float, index: int, out: Path) -> None` (runs ffmpeg with `zoompan_args`; raises `RuntimeError` with ffmpeg's last stderr line on failure)

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("src", [(3000, 1000), (800, 4000), (1080, 1920), (900, 1500)])
def test_frame_photo_is_exact_and_undistorted(src):  # Review Focus 2
    out = videoad.frame_photo(Image.new("RGB", src, (200, 30, 30)))
    assert out.size == (1080, 1920)


def test_wide_photo_sits_on_a_blurred_fill():
    img = Image.new("RGB", (3000, 1000), (220, 20, 20))
    out = videoad.frame_photo(img)
    assert out.getpixel((540, 960))[0] > 180 and out.getpixel((540, 40)) != out.getpixel((540, 960))


def test_failed_ai_still_retries_once_then_uses_a_card(monkeypatch):
    calls = []
    monkeypatch.setattr(
        videoad.imagegen, "generate_images", lambda *a, **k: calls.append(1) or ([], "error")
    )
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5).scenes[:1], [], "red", FONTS
    )
    assert len(calls) == 2 and stills[0].size == (1080, 1920) and billed == 2


def test_photo_scenes_use_the_shop_photo_without_a_model_call(monkeypatch):
    monkeypatch.setattr(videoad.imagegen, "generate_images", lambda *a, **k: 1 / 0)
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, src="photo:0").scenes, [png_bytes((900, 900))], "red", FONTS
    )
    assert stills[0].size == (1080, 1920) and billed == 0


def test_caption_stays_in_the_safe_area_with_the_label():
    png, box = videoad.caption_png("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", FONTS)
    x0, y0, x1, y1 = box
    assert png.mode == "RGBA" and png.size == (1080, 1920)
    assert x0 >= 65 and x1 <= 1015 and y0 >= 270 and y1 <= 1250
    assert png.getpixel((5, 1900))[3] == 0  # outside: transparent


def test_zoompan_args_frames_and_direction():
    a, b = videoad.zoompan_args(4.0, 0), videoad.zoompan_args(4.0, 1)
    assert "d=120" in " ".join(a) and "s=1080x1920" in " ".join(a) and a != b


def test_motion_clip_renders_with_ffmpeg(tmp_path):
    still = tmp_path / "s.png"
    Image.new("RGB", (1080, 1920), "gray").save(still)
    videoad.motion_clip(still, 1.0, 0, tmp_path / "c.mp4")
    assert probe(tmp_path / "c.mp4")["duration"] == pytest.approx(1.0, abs=0.1)
```
(`probe(path)` test helper runs `ffprobe -v error -show_entries format=duration:stream=codec_name,width,height -of json` and returns `{"duration": float, "codecs": [...], "size": (w, h)}`; put it in `promo_helpers.py`.)

- [ ] **Step 2: Run to verify they fail** — Expected: `AttributeError: module 'salid.videoad' has no attribute 'frame_photo'`.
- [ ] **Step 3: Implement** per the Interfaces block (ffmpeg invoked with `subprocess.run([... ], check=False, capture_output=True, timeout=120)`; still → clip uses `-loop 1 -i still -vf <zoompan> -t seconds -pix_fmt yuv420p -c:v libx264 -preset veryfast -crf 18`).
- [ ] **Step 4: Run** `… pytest tests/salid/test_videoad.py -q -p no:warnings` — Expected: all passed.
- [ ] **Step 5: Commit** `git commit -am "Salid: video ad stills, framing, fallback card, motion clips and captions"` (after `git add` of the test helper).

---

### Task 5: Narration and final assembly with the size budget

**Files:**
- Modify: `src/salid/videoad.py`
- Test: `tests/salid/test_videoad.py`

**Interfaces:**
- Consumes: `speech.say(settings, client, text, lang) -> bytes` (24 kHz mono s16le PCM), `speech.pcm_duration_s`; Task 4 clips/captions.
- Produces:
  - `narration_wav(say: Callable[[str], bytes], lines: list[tuple[float, float, str]], total_s: float, out: Path) -> bool` (`lines` = (start, scene_seconds, text); each PCM line is cut to its scene length; silence-padded to `total_s`; written as 48 kHz WAV via ffmpeg; returns False if every line failed)
  - `video_kbps(budget_mib: float, seconds: float, audio_kbps: int = 128) -> int` = `int(budget_mib * 8388.6 / seconds - audio_kbps)`
  - `assemble(clips: list[Path], seconds: list[float], captions: list[Path], audio: Path | None, out: Path, *, budget_mib: float, music: Path | None = None) -> Path` (`xfade` 0.4 s between clips with offsets from cumulative seconds; caption PNGs overlaid per scene with 0.2 s fades; audio mixed (music at -18 dB under voice) with `loudnorm=I=-16`; H.264 High yuv420p 30 fps, AAC 128k 48 kHz, `+faststart`, `-b:v kbps -maxrate 1.1x -bufsize 2x`, `-t` total; if the file exceeds the budget: one re-encode at 80 % bitrate, then at 720x1280; raises `RuntimeError` if still over)
  - `budget_mib(guild_limit_bytes: int | None) -> float` = `min(19.0, guild_limit_bytes / 2**20 - 1)` (default 25 MiB limit when None)

- [ ] **Step 1: Write the failing tests**

```python
def test_bitrate_formula():
    assert videoad.video_kbps(19, 30) == 5184 and videoad.budget_mib(None) == 19.0
    assert videoad.budget_mib(10 * 2**20) == 9.0


def test_narration_never_runs_past_its_scene(tmp_path):  # Review Focus 1
    long_pcm = b"\x00\x01" * 24_000 * 6  # 6 s of audio
    ok = videoad.narration_wav(
        lambda text: long_pcm, [(0.0, 2.0, "a"), (2.0, 2.0, "b")], 4.0, tmp_path / "v.wav"
    )
    assert ok and probe(tmp_path / "v.wav")["duration"] == pytest.approx(4.0, abs=0.05)


def test_narration_failures_degrade(tmp_path):
    assert (
        videoad.narration_wav(lambda t: 1 / 0, [(0.0, 2.0, "a")], 2.0, tmp_path / "v.wav") is False
    )


def test_assembled_ad_meets_the_delivery_spec(tmp_path):
    clips, caps = tiny_scene_assets(tmp_path, n=2, seconds=2.0)  # helper: 2 motion clips + captions
    out = videoad.assemble(
        clips, [2.0, 2.0], caps, silent_wav(tmp_path, 4.0), tmp_path / "ad.mp4", budget_mib=19
    )
    p = probe(out)
    assert p["size"] == (1080, 1920) and set(p["codecs"]) == {"h264", "aac"}
    assert p["duration"] == pytest.approx(4.0, abs=0.2) and out.stat().st_size < 19 * 2**20


def test_small_guild_budget_is_respected(tmp_path):  # Review Focus 3
    clips, caps = tiny_scene_assets(tmp_path, n=2, seconds=2.0, noisy=True)
    out = videoad.assemble(clips, [2.0, 2.0], caps, None, tmp_path / "ad.mp4", budget_mib=0.5)
    assert out.stat().st_size <= 0.5 * 2**20
```
(`tiny_scene_assets` and `silent_wav` helpers in `promo_helpers.py`, built with Task 4's `motion_clip` / `caption_png` and ffmpeg `anullsrc`; `noisy=True` uses noise stills so the encoder must cut bitrate.)

- [ ] **Step 2: Run to verify they fail** — Expected: `AttributeError: ... 'video_kbps'`.
- [ ] **Step 3: Implement** per the Interfaces block (PCM → WAV: `ffmpeg -f s16le -ar 24000 -ac 1 -i pipe:` → `-ar 48000`).
- [ ] **Step 4: Run** `… pytest tests/salid/test_videoad.py -q -p no:warnings` — Expected: all passed.
- [ ] **Step 5: Commit** `git commit -am "Salid: video ad narration and ffmpeg assembly under the upload budget"` (with the helper file added).

---

### Task 6: Job runner, records, monthly budgets and the Discord side

**Files:**
- Modify: `src/salid/videoad.py` (runner, handle, usage)
- Create: `src/salid/videoad_bot.py` (`VideoAdMixin`)
- Modify: `src/salid/bot.py` (mixin; `self._init_video_ads()`; tool entry after `_image_tools`; startup sweep in `on_ready` via `self._sweep_video_jobs()`), `src/salid/guild_settings.py` (`video_ads: bool = False`), `src/meeting_scribe/static/hub.html` (toggle, `video_ads: false` default before `...s.features`, save literal, `f_video_ads_n/_d/_w`), `src/salid/config.py` (`video_ads_per_month: int = 10`, `video_hero_seconds_per_month: int = 60`, `video_max_jobs: int = 2`, `video_hero_model: str = ""`, `video_music_model: str = ""`), `src/salid/i18n.py`, `README.md`, `CLAUDE.md`, `tests/salid/test_guild_settings.py`, `tests/test_hub.py`
- Test: `tests/salid/test_videoad.py` (runner), `tests/salid/test_videoad_bot.py`

**Interfaces:**
- Consumes: Tasks 3–5; `coo_store.DocStore`; `speech.say`; `cards.card_fonts`; bot `_reply_with_uploads` (on main) for delivery.
- Produces:
  - `STEPS = ("storyboard", "stills", "hero", "audio", "render")`
  - `class Cancelled(Exception)`; `@dataclass class VideoHandle(cancelled: threading.Event = Event(), progress: Callable[[str, str], None] = noop)`; `check(handle)` raises `Cancelled`
  - `class MediaUsage(docs: DocStore)`: `take(guild_id: int, month: str, field: str, amount: float, cap: float) -> bool` (DocStore kind `salid_media_usage`, key = month; refuses without writing when `used + amount > cap`)
  - `@dataclass class VideoRequest(guild_id, user_id, brief, seconds, photos: list[bytes], hero_shots, voice, music, lang, accent="red")`
  - `@dataclass class VideoResult(path: Path, storyboard: Storyboard, cost_usd: float, steps_ms: dict[str, int])`
  - `run_video_ad(req: VideoRequest, *, client, settings, say: Callable[[str], bytes], workdir: Path, budget_mib: float, handle: VideoHandle, usage: MediaUsage, month: str) -> VideoResult` (`check(handle)` before each step; the ad counter is taken once at start; hero seconds taken before each hero call; raises `Cancelled` / `StoryboardError` / `RuntimeError`)
  - `_hero_clip(client, settings, still: Path, scene: Scene, handle: VideoHandle, out: Path) -> bool` — module-level hook the runner calls for each `hero` scene only when `settings.video_hero_model` is set; in this task it returns False (the scene then uses its motion clip); Task 7 implements it. `fake_pipeline` patches it to count `hero_calls`.
  - `VideoAdMixin._init_video_ads()`, `_video_tools(message, guild, asker, files) -> list` (`[create_video_ad]`; [] for voice/flag off/no ffmpeg/`thai_ready` False), `_launch_video(...)`, `_run_video(...)`, `_sweep_video_jobs()`, `CancelView(job_id)` (a `discord.ui.View(timeout=900)` with one button; only the requester or an admin (`_may_configure`) can press it)
  - DocStore kind `salid_video_jobs`, key = job id, doc `{guild_id, channel_id, requester_id, state, created_at, updated_at, cost_usd, reason}`
  - i18n keys (th/en): `video_started`, `video_step_storyboard`, `video_step_stills`, `video_step_hero`, `video_step_audio`, `video_step_render`, `video_done`, `video_failed`, `video_cancelled`, `video_restarted`, `video_busy`, `video_month_limit`, `video_cancel_button`, `video_not_yours`

`create_video_ad` docstring (exact copy):
```
Make a finished vertical video ad (15, 20 or 30 seconds) for the user's product or promotion:
Salid writes a storyboard, makes the scenes (or uses the photos the user attached), adds Thai
captions and Thai narration, and posts the MP4 here when it is ready (a few minutes). Use for
"ทำวิดีโอโฆษณา…", "ทำคลิปโปรโมท…", "make a video ad…". Pass the user's brief as is.

Args:
    brief: what to advertise and the offer, in the user's words.
    seconds: 15, 20 (default) or 30.
    use_photos: use the photos attached to this message as scenes (default true).
    hero_shots: 0-2 generated motion shots (only when enabled for this server).
    voice: add Thai narration (default true).
    music: add background music (only when enabled for this server).
```
Tool results: `{"status": "started", "note": "Reply in one short line that the video is being made and will be posted here."}`; `{"status": "busy", ...}` when this guild has a running job or the bot is at `video_max_jobs`; `{"status": "limit", ...}` when the month's ads are used up; `{"status": "error", ...}` for bad `seconds`/`hero_shots`.

- [ ] **Step 1: Write the failing tests**

```python
def test_month_usage_refuses_without_writing(docs):
    u = videoad.MediaUsage(docs)
    assert u.take(1, "2026-10", "ads", 1, 2) and u.take(1, "2026-10", "ads", 1, 2)
    assert (
        not u.take(1, "2026-10", "ads", 1, 2)
        and docs.get("salid_media_usage", 1, "2026-10")["ads"] == 2
    )
    assert u.take(1, "2026-11", "ads", 1, 2) and u.take(2, "2026-10", "ads", 1, 2)


def test_runner_happy_path_reports_each_step(tmp_path, fake_pipeline):
    seen = []
    res = videoad.run_video_ad(
        req(),
        client=None,
        settings=S,
        say=lambda t: SILENCE,
        workdir=tmp_path,
        budget_mib=19,
        handle=videoad.VideoHandle(progress=lambda s, d: seen.append(s)),
        usage=videoad.MediaUsage(MemDocs()),
        month="2026-10",
    )
    assert [s for s in seen] == ["storyboard", "stills", "audio", "render"] and res.path.exists()


def test_cancel_during_stills_stops_paid_steps(tmp_path, fake_pipeline):  # Review Focus 4
    h = videoad.VideoHandle()
    fake_pipeline.on_stills = h.cancelled.set
    with pytest.raises(videoad.Cancelled):
        videoad.run_video_ad(
            req(hero_shots=1),
            client=None,
            settings=S_HERO,
            say=lambda t: SILENCE,
            workdir=tmp_path,
            budget_mib=19,
            handle=h,
            usage=(u := videoad.MediaUsage(d := MemDocs())),
            month="2026-10",
        )
    assert fake_pipeline.hero_calls == 0 and d.get("salid_media_usage", 1, "2026-10") == {"ads": 1}


def test_hero_is_skipped_while_the_model_setting_is_empty(tmp_path, fake_pipeline):
    videoad.run_video_ad(
        req(hero_shots=2),
        client=None,
        settings=S,
        say=lambda t: SILENCE,
        workdir=tmp_path,
        budget_mib=19,
        handle=videoad.VideoHandle(),
        usage=videoad.MediaUsage(MemDocs()),
        month="2026-10",
    )
    assert fake_pipeline.hero_calls == 0
```
(`fake_pipeline` fixture monkeypatches `plan_storyboard`, `scene_stills`, `motion_clip`, `narration_wav`, `assemble` with recorders; `MemDocs` = the existing in-memory DocStore used by other tests.)

Bot tests (`test_videoad_bot.py`): tool not offered with the flag off (default) / in voice / in interjections; offered when an admin turns `video_ads` on; second request in the same guild → `busy`; month limit → `limit` and nothing launched; Cancel button pressed by a non-requester non-admin → ephemeral `video_not_yours` and the job keeps running; by the requester → handle cancelled; restart sweep:

```python
async def test_restart_marks_running_jobs_failed_and_tells_the_requester(
    bot, world
):  # Review Focus 5
    bot.docs.put(
        "salid_video_jobs",
        GUILD,
        "j1",
        {
            "guild_id": GUILD,
            "channel_id": world.pub.id,
            "requester_id": world.staff.id,
            "state": "stills",
        },
    )
    bot.docs.put("salid_media_usage", GUILD, "2026-10", {"ads": 1})
    await bot._sweep_video_jobs()
    assert bot.docs.get("salid_video_jobs", GUILD, "j1")["state"] == "failed"
    assert bot.docs.get("salid_video_jobs", GUILD, "j1")["reason"] == "restarted"
    assert bot.docs.get("salid_media_usage", GUILD, "2026-10") == {"ads": 1}
    assert any("video_restarted" in m or t("video_restarted", "th") in m for m in world.pub.sent)
```
Plus: `"video_ads" in FEATURES and Features().video_ads is False`; hub default `video_ads: false` before the spread.

- [ ] **Step 2: Run to verify they fail** — Expected: missing `MediaUsage` / `_video_tools`.
- [ ] **Step 3: Implement** the runner per Interfaces (steps call Tasks 3–5; hero step calls `videogen` only when `settings.video_hero_model` is set — Task 7 provides it; until then the hero branch is skipped), the mixin (pattern of `search_bot._launch_research`: `asyncio.create_task` + `asyncio.to_thread(run_video_ad, ...)`, `self._video_jobs[guild_id]`, a `tempfile.TemporaryDirectory` always removed; progress message edited at most every 5 s with the step label; delivery reply with the MP4, the storyboard as a short list and a suggested caption ending in `#สร้างด้วยAI`; records updated on every state change; one INFO line per step `video guild=%s step=%s ms=%d cost=%.2f`), flag/hub/config/i18n/docs.
- [ ] **Step 4: Run** `… pytest tests/salid -q -p no:warnings` and `… pytest tests/test_hub.py -q -p no:warnings` — Expected: all passed.
- [ ] **Step 5: Commit** `git add …` (all files above) `&& git commit -m "Salid: create_video_ad background job (records, monthly budget, progress, cancel, delivery)"`

---

### Task 7: Omni hero clips, optional music, probe script and live tests

**Files:**
- Create: `src/salid/videogen.py`; `scripts/salid_media_probe.py`
- Modify: `src/salid/videoad.py` (hero + music steps call `videogen`), `tests/salid/test_salid_live.py`, `README.md`, `CLAUDE.md`
- Test: `tests/salid/test_videogen.py`

**Interfaces:**
- Consumes: google-genai `client.interactions.create/get/cancel` (`VideoResponseFormat(type="video", delivery="inline", aspect_ratio="9:16", resolution="720p", duration="Ns")`, `generation_config={"video_config": {"task": "image_to_video" | "text_to_video"}}`, `background=True`).
- Produces:
  - `HERO_SUFFIX = " Single continuous shot, no on-screen text, no dialogue."`
  - `start_hero(client, model: str, still_png: bytes, prompt_en: str, seconds: int) -> str` (interaction id; refuses non-English prompts or quotes with `ValueError`; `seconds` clamped to 4–6)
  - `wait_hero(client, interaction_id: str, *, cancelled: threading.Event, poll_s: float = 10.0, timeout_s: float = 480.0, sleep=time.sleep) -> tuple[bytes | None, str]` (MP4 bytes or reason `blocked` / `timeout` / `cancelled` / `error`; calls `interactions.cancel` on cancel/timeout)
  - `music_clip(client, model: str, prompt_en: str, seconds: int) -> bytes | None` (instrumental; None on any failure)
  - `scripts/salid_media_probe.py [--image] [--video]`: one 1K image and/or one 4 s 360p Omni clip, inline delivery, files written to `/tmp/salid-media-probe/` only; prints latency, status, usage; never prints credentials; refuses to run if `GOOGLE_APPLICATION_CREDENTIALS` points to a key whose project differs from `SALID_GCP_PROJECT` (prints a hint to run `gcloud auth application-default login`).

- [ ] **Step 1: Write the failing tests** (fake interactions client: `create` returns an object with `id`; `get` returns statuses from a script; `cancel` records calls)

```python
def test_hero_request_shape_and_english_guard():
    c = FakeInteractions(statuses=["completed"], video=b"MP4")
    iid = videogen.start_hero(
        c, "gemini-omni-1.1-flash-preview", png_bytes((768, 1376)), "a latte, slow push-in", 8
    )
    kw = c.created[0]
    assert kw["background"] is True and kw["response_format"][0]["delivery"] == "inline"
    assert kw["response_format"][0]["duration"] == "6s" and kw["input"][-1]["text"].endswith(
        videogen.HERO_SUFFIX
    )
    with pytest.raises(ValueError):
        videogen.start_hero(c, "m", b"x", "ลาเต้", 4)


def test_wait_returns_bytes_or_reasons_and_cancels():
    assert videogen.wait_hero(
        FakeInteractions(["in_progress", "completed"], b"MP4"),
        "i",
        cancelled=Event(),
        sleep=lambda s: None,
    ) == (b"MP4", "")
    c = FakeInteractions(["in_progress"] * 100, None)
    assert videogen.wait_hero(c, "i", cancelled=Event(), timeout_s=0.0, sleep=lambda s: None) == (
        None,
        "timeout",
    )
    assert c.cancelled == ["i"]
    ev = Event()
    ev.set()
    assert (
        videogen.wait_hero(
            FakeInteractions(["in_progress"], None), "i", cancelled=ev, sleep=lambda s: None
        )[1]
        == "cancelled"
    )
    assert (
        videogen.wait_hero(
            FakeInteractions(["failed"], None), "i", cancelled=Event(), sleep=lambda s: None
        )[1]
        == "blocked"
    )


def test_runner_uses_a_hero_clip_and_falls_back_to_motion(tmp_path, fake_pipeline, monkeypatch):
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (None, "blocked"))
    videoad.run_video_ad(
        req(hero_shots=1),
        client=object(),
        settings=S_HERO,
        say=lambda t: SILENCE,
        workdir=tmp_path,
        budget_mib=19,
        handle=videoad.VideoHandle(),
        usage=videoad.MediaUsage(MemDocs()),
        month="2026-10",
    )
    assert fake_pipeline.motion_calls == 4  # the hero scene fell back to motion
```
Live tests (marked `live`): `test_live_create_image` (one 1K image), `test_live_omni_probe` (4 s 360p, skip with a clear message on 403/PERMISSION_DENIED), `test_live_slideshow_ad` (full 15 s ad, no hero, saved under a temp dir and the path printed).

- [ ] **Step 2: Run to verify they fail** — Expected: `ModuleNotFoundError: No module named 'salid.videogen'`.
- [ ] **Step 3: Implement** per Interfaces; `videoad._hero_clip` (the Task 6 hook): take `seconds` from the hero budget before `start_hero`, `wait_hero`, transcode the returned clip to 1080x1920 30 fps (scale+pad) as that scene's clip; on failure use the motion clip; music step: `music_clip` when `video_music_model` is set and `req.music`, else none.
- [ ] **Step 4: Run** `… pytest tests/salid -q -p no:warnings` and lint — Expected: all passed, live deselected; ruff clean.
- [ ] **Step 5: Commit** `git add src/salid/videogen.py src/salid/videoad.py scripts/salid_media_probe.py tests/salid/test_videogen.py tests/salid/test_salid_live.py README.md CLAUDE.md && git commit -m "Salid: optional Omni hero clips and music behind settings; media probe and live tests"`

**Human gates after Task 7 (never without approval and GCP project access):** `scripts/salid_media_probe.py --image --video`; live tests `-m live -k "create_image or omni or slideshow"`; visual check of a real ad; push + PR; deploy; setting `SALID_VIDEO_HERO_MODEL` / `SALID_VIDEO_MUSIC_MODEL`.
