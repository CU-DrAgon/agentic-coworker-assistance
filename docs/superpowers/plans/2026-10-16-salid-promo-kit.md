# Salid promo post kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A product photo sent to Salid with a request comes back as three ready-to-post JPEGs (1:1, 4:5, 9:16) with code-typeset Thai text and an AI-restyled scene, plus a caption in the reply.

**Architecture:** `cards.py` is a reusable Pillow card renderer (RAQM Thai shaping, text fitting, scrims, badges, crops, JPEG). `promo.py` is pure feature logic: validation, one Nano Banana edit call, fallback scene, three layouts, revision memory and the `make_promo_kit` tool with an `outbox`. `promo_bot.py` is the Discord mixin that binds guild / channel / author, the flag and the daily cap, and `bot._agent_kwargs` appends the tool only when a photo is available.

**Tech Stack:** Python 3.12, Pillow 12.3 (RAQM + system libfribidi), google-genai 2.29 (`gemini-nano-banana-2.1`, location `global`), Google ADK tools, pytest (offline), uv.

**Spec:** `docs/superpowers/specs/2026-10-16-salid-promo-kit-design.md`

## Global Constraints

- Thai text is typeset by our code only; the image model never receives the user's Thai text.
- Pillow text is always drawn with `ImageFont.Layout.RAQM`; never BASIC.
- Image call: model `settings.promo_image_model` (default `gemini-nano-banana-2.1`), `response_modalities=["TEXT", "IMAGE"]`, `ImageConfig(aspect_ratio="9:16", image_size="2K")`, `HttpOptions(timeout=60_000)`; **no** `temperature`, `top_p`, `top_k`, `seed`, `logprobs` (API error on this model).
- Exactly one image-model call per kit; the daily unit is taken **before** the call.
- Outputs in this order: `promo-1x1.jpg` 1080×1080, `promo-4x5.jpg` 1080×1350, `promo-9x16.jpg` 1080×1920; RGB JPEG quality 90 → 85 → 80, each < 1,500,000 bytes.
- Field limits: headline 40, scene 300 (min 3), price 20, call_to_action 24, shop_name 30; accents `red orange green blue gold black`, default `red`.
- Settings: `promo_image_model: str = "gemini-nano-banana-2.1"`, `promo_kits_per_day: int = 20` (env `SALID_PROMO_IMAGE_MODEL`, `SALID_PROMO_KITS_PER_DAY`); 0 = AI off.
- Revision memory: key `(guild_id, channel_id, user_id)`, TTL 1800 s, max 50 entries, process memory only.
- Logs never contain user text, prompts or image bytes.
- Code, comments and docs in English; fixed user-facing strings would go in `i18n.py` (this feature adds none: the agent writes the reply; hub strings live in `hub.html`).
- Run tests as `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest <paths> -q -p no:warnings` (the uv on PATH is an old stray one; DYLD must be set at the last hop for libopus and libfribidi). Lint: `uvx ruff check . && uvx ruff format --check .`.
- Dependency changes only through `uvx uv@latest` (never pip); `uv.lock` revision line must not change.
- `.env.example` is blocked by the user's permission settings: document the two settings in README and CLAUDE.md instead; tell the user.
- Before every commit: `git status` and `git diff --cached`; never stage `.env`, `samples/` or key `*.json` files.

## Review Focus

1. **Phone photos with EXIF rotation** (portrait shots stored landscape + Orientation tag): the product must appear upright → `prepare_photo` applies `ImageOps.exif_transpose` (test in Task 3).
2. **Transparent PNG product cut-outs:** transparent areas must become white, not black, before the model / fallback see them (test in Task 3).
3. **Emoji or symbols the font cannot draw** in overlay fields ("ลด 20% 🔥"): no tofu boxes; such characters are dropped from overlay text (test in Task 1, used in Task 4).
4. **The model returns an unexpected size or aspect** (square, small, landscape): every format still comes out at its exact size via cover crop (test in Task 4).
5. **The agent calls `make_promo_kit` twice in one turn** (a retry or a self-correction): the reply carries only the latest kit's three files, not six (test in Task 5).

---

### Task 1: Dependencies, bundled fonts and the RAQM gate

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (Pillow from the dev group to runtime)
- Modify: `Dockerfile.salid` (apt line)
- Create: `src/salid/fonts/Kanit-Bold.ttf`, `src/salid/fonts/Kanit-Regular.ttf`, `src/salid/fonts/OFL.txt`
- Create: `src/salid/cards.py`
- Test: `tests/salid/test_cards.py`

**Interfaces:**
- Produces: `cards.FONTS_DIR: Path` (`src/salid/fonts`); `@dataclass(frozen=True) class CardFonts(bold: str, regular: str)`; `card_fonts() -> CardFonts` (cached); `thai_ready() -> bool`; `font(path: str, size: int) -> ImageFont.FreeTypeFont` (cached, RAQM); `drawable(text: str, path: str) -> str`.

- [ ] **Step 1: Move Pillow to runtime and add libfribidi**

Run: `uvx uv@latest remove --dev pillow && uvx uv@latest add 'pillow>=12.3.0'`
Expected: `pyproject.toml` lists `pillow>=12.3.0` under `[project] dependencies` and no longer under `[dependency-groups] dev`; `git diff uv.lock` touches only Pillow's dependency-group lines, and `head -3 uv.lock` shows the same `revision` as before.

In `Dockerfile.salid` change the apt line to `libopus0 ffmpeg fonts-thai-tlwg libfribidi0` and add to the comment above it: `# Promo kits: libfribidi0 lets Pillow's bundled raqm shape Thai (tone marks), see salid.cards.`

- [ ] **Step 2: Bundle Kanit (SIL OFL 1.1)**

Run:
```bash
for f in Kanit-Bold.ttf Kanit-Regular.ttf OFL.txt; do
  curl -sfL -o "src/salid/fonts/$f" "https://github.com/google/fonts/raw/main/ofl/kanit/$f"; done
ls -l src/salid/fonts
```
Expected: three files; `Kanit-Bold.ttf` ≈ 176 KB; `OFL.txt` starts with the SIL Open Font License text.

- [ ] **Step 3: Write the failing tests**

```python
# tests/salid/test_cards.py
from salid import cards


def test_raqm_is_available_for_thai():
    assert cards.thai_ready(), (
        "Pillow has no raqm: install libfribidi (brew install fribidi / apt libfribidi0) "
        "and run tests with DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib; report it, do not work around it"
    )


def test_bundled_kanit_is_used():
    f = cards.card_fonts()
    assert f.bold.endswith("Kanit-Bold.ttf") and f.regular.endswith("Kanit-Regular.ttf")


def test_falls_back_to_find_fonts_when_bundle_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cards, "FONTS_DIR", tmp_path)
    cards.card_fonts.cache_clear()
    monkeypatch.setattr(cards, "find_fonts", lambda configured="": FontSet("/r.ttf", None, True))
    assert cards.card_fonts() == cards.CardFonts(bold="/r.ttf", regular="/r.ttf")
    cards.card_fonts.cache_clear()


def test_fonts_use_raqm_layout():
    assert cards.font(cards.card_fonts().bold, 40).layout_engine == ImageFont.Layout.RAQM


def test_thai_is_shaped_not_drawn_glyph_by_glyph():
    # RAQM applies the font's mark positioning (stacked tone marks); BASIC does not
    def render(f):
        img = Image.new("L", (900, 140), 0)
        ImageDraw.Draw(img).text((10, 20), "ผู้ใหญ่ น้ำแข็ง ปั้น", font=f, fill=255)
        return img

    bold = cards.card_fonts().bold
    basic = ImageFont.truetype(bold, 60, layout_engine=ImageFont.Layout.BASIC)
    assert ImageChops.difference(render(cards.font(bold, 60)), render(basic)).getbbox() is not None


def test_drawable_drops_characters_without_glyphs():
    assert cards.drawable("ลด 20% 🔥", cards.card_fonts().bold) == "ลด 20%"
    assert cards.drawable("Latte เย็น 59.-", cards.card_fonts().bold) == "Latte เย็น 59.-"
```
(Imports: `from PIL import Image, ImageChops, ImageDraw, ImageFont`, `from salid.pdf_report import FontSet`.)

- [ ] **Step 4: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_cards.py -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'salid.cards'` (or ImportError).

- [ ] **Step 5: Implement in `src/salid/cards.py`**

Module docstring: what cards are for (Thai-safe image cards; reused by promo kits and later visual features), that RAQM is required and why. `card_fonts()` is `functools.lru_cache`d: bundled Kanit when both files exist in `FONTS_DIR`, else `find_fonts()` (import it as a module name `find_fonts` so tests can patch it) with `bold = fs.bold or fs.regular`. `thai_ready()` = `PIL.features.check("raqm")`. `font()` is `lru_cache(maxsize=64)` around `ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)`. `drawable()` keeps whitespace and every code point present in the font's cmap (fontTools `TTFont(path, lazy=True).getBestCmap()`, cached per path), then collapses repeated spaces and strips.

- [ ] **Step 6: Run tests to verify they pass, and that the wheel carries the fonts**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_cards.py -q -p no:warnings`
Expected: 6 passed.
Run: `uvx uv@latest build --wheel -o "$CLAUDE_JOB_DIR/tmp/whl" -q && unzip -l "$CLAUDE_JOB_DIR"/tmp/whl/*.whl | grep -c 'salid/fonts/'`
Expected: `3`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock Dockerfile.salid src/salid/fonts src/salid/cards.py tests/salid/test_cards.py
git commit -m "Salid: card renderer base (Kanit fonts, RAQM gate), Pillow at runtime, libfribidi0 in the image"
```

---

### Task 2: Card primitives (text fitting, scrims, badges, crops, JPEG)

**Files:**
- Modify: `src/salid/cards.py`
- Test: `tests/salid/test_cards.py`

**Interfaces:**
- Consumes: `font`, `card_fonts` (Task 1).
- Produces:
  - `break_ok(prev: str, nxt: str) -> bool`
  - `@dataclass class Fitted(size: int, lines: list[str], width: int, height: int, line_height: int)`
  - `line_height(f: ImageFont.FreeTypeFont) -> int` = `round((ascent + descent) * 1.15)`
  - `fit_text(text: str, path: str, max_width: int, max_lines: int, sizes: tuple[int, int]) -> Fitted` (sizes = (max, min))
  - `draw_lines(img: Image.Image, x: int, y: int, fitted: Fitted, path: str, *, fill: str, align: str = "left", width: int = 0) -> tuple[int, int, int, int]` (returns the block box `(x0, y0, x1, y1)`; `align` "left" | "right" | "center" within `width`)
  - `scrim(img: Image.Image, y0: int, y1: int, alpha0: int, alpha1: int) -> None` (in place, black, linear alpha from y0 to y1)
  - `@dataclass class Badge(w: int, h: int, fitted: Fitted)`; `measure_badge(text: str, path: str, sizes: tuple[int, int], max_width: int, pad: tuple[int, int] = (32, 16)) -> Badge`; `draw_badge(img, badge: Badge, x: int, y: int, path: str, *, fill: str, text_fill: str, radius: int | None = None) -> tuple[int, int, int, int]` (`radius=None` → pill, `h // 2`)
  - `cover(img: Image.Image, size: tuple[int, int]) -> Image.Image` (scale to cover, centre crop, exact size, RGB)
  - `to_jpeg(img: Image.Image, max_bytes: int = 1_500_000) -> bytes` (quality 90, 85, 80; last one returned even if over)
  - `contrast(a: str, b: str) -> float` (WCAG 2 ratio for `#RRGGBB`)

- [ ] **Step 1: Write the failing tests**

```python
BOLD = cards.card_fonts().bold


def test_never_breaks_inside_a_thai_syllable():
    assert not cards.break_ok("น", "้")  # tone mark (Mn)
    assert not cards.break_ok("้", "ำ")  # SARA AM U+0E33
    assert not cards.break_ok("ก", "า")  # following vowel U+0E30-0E3A
    assert not cards.break_ok("เ", "ก")  # after a leading vowel U+0E40-0E44
    assert not cards.break_ok("ก", "ๆ")  # U+0E45-0E4E
    assert cards.break_ok("า", "ก") and cards.break_ok("a", "b")


def test_fit_prefers_shrinking_to_splitting():
    f = cards.fit_text("โปรพิเศษ", BOLD, 952, 2, (96, 48))
    assert f.lines == ["โปรพิเศษ"] and f.size == 96


def test_fit_breaks_at_spaces_and_keeps_every_line_inside():
    f = cards.fit_text("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", BOLD, 600, 2, (96, 48))
    assert 1 <= len(f.lines) <= 2 and f.width <= 600
    assert all(cards.font(BOLD, f.size).getlength(l) <= 600 for l in f.lines)


def test_fit_splits_a_long_spaceless_thai_run_only_at_syllable_boundaries():
    text = "ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือนนี้เท่านั้น"  # 40+ chars, no spaces
    f = cards.fit_text(text, BOLD, 500, 2, (96, 48))
    assert "".join(f.lines) == text and len(f.lines) == 2 and f.width <= 500
    for a, b in zip(f.lines, f.lines[1:]):
        assert cards.break_ok(a[-1], b[0])


def test_line_height_leaves_room_for_marks():
    fnt = cards.font(BOLD, 80)
    asc, desc = fnt.getmetrics()
    assert cards.line_height(fnt) == round((asc + desc) * 1.15)


def test_drawn_block_stays_in_its_box():
    img = Image.new("RGB", (1080, 600), "white")
    f = cards.fit_text("ผู้ใหญ่ญี่ปุ่น น้ำแข็งปั้น", BOLD, 952, 2, (96, 48))
    x0, y0, x1, y1 = cards.draw_lines(img, 64, 100, f, BOLD, fill="black")
    assert 64 <= x0 and x1 <= 64 + 952 and y0 == 100 and y1 == 100 + f.height
    ink = ImageOps.invert(img.convert("L")).getbbox()
    assert ink[1] >= 100 and ink[3] <= y1


def test_scrim_darkens_towards_alpha1():
    img = Image.new("RGB", (100, 200), "white")
    cards.scrim(img, 0, 200, 0, 200)
    assert img.getpixel((50, 2))[0] > img.getpixel((50, 197))[0]


def test_badge_and_pill_geometry():
    b = cards.measure_badge("59.-", BOLD, (72, 40), 952)
    assert b.w == b.fitted.width + 64 and b.h == b.fitted.height + 32
    img = Image.new("RGB", (1080, 1080))
    assert cards.draw_badge(img, b, 100, 200, BOLD, fill="#E53935", text_fill="#FFFFFF") == (
        100,
        200,
        100 + b.w,
        200 + b.h,
    )


def test_cover_gives_the_exact_size_from_any_aspect():
    for src in ((512, 512), (2000, 800), (1536, 2752)):
        assert cards.cover(Image.new("RGB", src), (1080, 1350)).size == (1080, 1350)


def test_jpeg_under_the_cap():
    noisy = Image.effect_noise((1080, 1920), 120).convert("RGB")
    data = cards.to_jpeg(noisy)
    assert data[:2] == b"\xff\xd8" and len(data) < 1_500_000


def test_contrast_ratio():
    assert round(cards.contrast("#FFFFFF", "#000000"), 1) == 21.0
    assert round(cards.contrast("#E53935", "#FFFFFF"), 2) == 4.23
```
(Imports: `from PIL import Image, ImageOps`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_cards.py -q -p no:warnings`
Expected: the new tests FAIL with `AttributeError: module 'salid.cards' has no attribute ...`.

- [ ] **Step 3: Implement the primitives in `src/salid/cards.py`**

`break_ok(prev, nxt)` is False when `unicodedata.category(nxt)` is `Mn`/`Mc`/`Me`, or `nxt` is in U+0E30–U+0E3A or U+0E45–U+0E4E, or `prev` is in U+0E40–U+0E44; else True.

`fit_text` algorithm (the only non-obvious body):
```
for size in range(max, min - 1, -4) (always ending exactly at min):
    f = font(path, size)
    lines = greedy word wrap at spaces into max_width using f.getlength
    if every line fits and len(lines) <= max_lines: return Fitted(...)
at min size: also split any word wider than max_width at the last position where
    break_ok(word[i-1], word[i]) and the prefix fits; re-wrap; return what results
    (the field limits guarantee <= max_lines at min size; never truncate)
```
`Fitted.width` = widest line, `height` = `len(lines) * line_height`. `draw_lines` draws each line with `anchor="ls"` at baseline `y + i * lh + (lh - ascent - descent) // 2 + ascent` (the extra leading split above and below, so tall marks stay inside), so the block's box is `(x, y, x + block width, y + height)` (for `align="right"`/`"center"`, offset each line within `width`). `scrim` alpha-composites a black `L`-mask gradient. `draw_badge` uses `ImageDraw.rounded_rectangle` and centres the text block inside. `contrast` uses the WCAG relative-luminance formula.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_cards.py -q -p no:warnings`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/salid/cards.py tests/salid/test_cards.py
git commit -m "Salid: card primitives (Thai-safe fitting, scrims, badges, cover crop, JPEG ladder)"
```

---

### Task 3: Kit request, photo preparation and the AI scene call

**Files:**
- Create: `src/salid/promo.py`
- Modify: `src/salid/config.py` (two settings next to `doc_pages_per_day`)
- Create: `tests/salid/promo_helpers.py` (shared image / fake-client helpers, plain module)
- Test: `tests/salid/test_promo.py`

**Interfaces:**
- Consumes: `llm.generate(client, model, contents, config, delays)`; `media.DailyBudget`.
- Produces:
  - `ACCENTS: dict[str, tuple[str, str]]` = `{"red": ("#E53935", "#FFFFFF"), "orange": ("#E65100", "#FFFFFF"), "green": ("#2E7D32", "#FFFFFF"), "blue": ("#1E5EC8", "#FFFFFF"), "gold": ("#F2C14E", "#1A1A1A"), "black": ("#111111", "#FFFFFF")}`
  - `LIMITS = {"headline": 40, "scene": 300, "price": 20, "call_to_action": 24, "shop_name": 30}`; `PHOTO_MIMES = {"image/png", "image/jpeg", "image/webp"}`
  - `class PromoSpec(BaseModel)`: `headline: str = ""`, `scene: str = ""`, `price: str = ""`, `call_to_action: str = ""`, `shop_name: str = ""`, `accent: str = "red"`
  - `clean(text: str) -> str` (drops control / XML-invalid characters, collapses whitespace, strips)
  - `merge_spec(prev: PromoSpec | None, *, headline: str = "", scene: str = "", price: str = "", call_to_action: str = "", shop_name: str = "", accent: str = "", default_shop: str = "") -> PromoSpec` — raises `ValueError` with a message for the agent
  - `@dataclass class Photo(jpeg: bytes, short_side: int)`; `prepare_photo(data: bytes) -> Photo | None`
  - `scene_prompt(scene: str) -> str`
  - `PROMO_TIMEOUT_MS = 60_000`; `PROMO_RETRY_DELAYS = (3.0,)` (read at call time)
  - `generate_scene(client, model: str, photo: Photo, scene: str) -> tuple[Image.Image | None, str]` (reason `""` on success, else `"blocked"`, `"timeout"`, `"error"`)
  - `take_unit(budget: DailyBudget, guild_id: int, day: str) -> str` (`""` = unit taken, `"disabled"` when `budget.cap <= 0`, `"daily_limit"` when used up)
  - `Settings.promo_image_model: str = "gemini-nano-banana-2.1"`, `Settings.promo_kits_per_day: int = 20`

- [ ] **Step 1: Write the failing tests**

```python
def test_spec_limits_cleaning_and_accents():
    s = promo.merge_spec(None, headline="  ลาเต้\x00เย็น  ", scene="cafe table", default_shop="ร้าน A")
    assert (s.headline, s.shop_name, s.accent) == ("ลาเต้เย็น", "ร้าน A", "red")
    for field, n in promo.LIMITS.items():
        with pytest.raises(ValueError, match=field):
            promo.merge_spec(None, **{"headline": "h", "scene": "abc", field: "x" * (n + 1)})
    with pytest.raises(ValueError, match="accent"):
        promo.merge_spec(None, headline="h", scene="abc", accent="pink")
    with pytest.raises(ValueError, match="headline"):
        promo.merge_spec(None, scene="abc")
    with pytest.raises(ValueError, match="scene"):
        promo.merge_spec(None, headline="h", scene="ab")
    assert (
        promo.merge_spec(None, headline="h", scene="abc", default_shop="x" * 40).shop_name
        == "x" * 30
    )


def test_revision_keeps_unchanged_fields_and_dash_removes():
    prev = promo.merge_spec(
        None, headline="h", scene="abc", price="59.-", call_to_action="สั่งเลย", default_shop="A"
    )
    s = promo.merge_spec(prev, scene="warmer light", price="-")
    assert (s.headline, s.scene, s.price, s.call_to_action, s.shop_name) == (
        "h",
        "warmer light",
        "",
        "สั่งเลย",
        "A",
    )


def test_prepare_photo_downscales_and_re_encodes():
    p = promo.prepare_photo(png_bytes((4000, 3000)))
    im = Image.open(io.BytesIO(p.jpeg))
    assert im.format == "JPEG" and max(im.size) == 2048 and p.short_side == 1536
    assert promo.prepare_photo(b"not an image") is None


def test_prepare_photo_applies_exif_rotation():  # Review Focus 1
    p = promo.prepare_photo(jpeg_with_orientation((400, 300), orientation=6))
    assert Image.open(io.BytesIO(p.jpeg)).size == (300, 400)


def test_prepare_photo_puts_transparency_on_white():  # Review Focus 2
    rgba = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
    p = promo.prepare_photo(to_png(rgba))
    assert Image.open(io.BytesIO(p.jpeg)).getpixel((10, 10))[0] > 240


def test_prompt_keeps_the_product_and_has_no_thai():
    p = promo.scene_prompt("wooden cafe table, morning light")
    assert "wooden cafe table, morning light" in p
    for rule in (
        "Keep the product exactly as it is",
        "No text, letters, numbers",
        "No people",
        "9:16",
    ):
        assert rule in p
    assert not re.search("[฀-๿]", p)


def test_scene_request_config_and_no_sampling_params():
    client = fake_client(image_response(png_bytes((768, 1376))))
    img, reason = promo.generate_scene(client, "gemini-nano-banana-2.1", photo(), "cafe")
    assert reason == "" and img.size == (768, 1376)
    model, contents, cfg = client.models.calls[0]
    assert model == "gemini-nano-banana-2.1" and contents[1] == promo.scene_prompt("cafe")
    assert cfg.response_modalities == ["TEXT", "IMAGE"]
    assert (cfg.image_config.aspect_ratio, cfg.image_config.image_size) == ("9:16", "2K")
    assert cfg.http_options.timeout == 60_000
    assert all(
        getattr(cfg, k) is None
        for k in ("temperature", "top_p", "top_k", "seed", "response_logprobs", "logprobs")
    )


@pytest.mark.parametrize(
    "finish",
    [
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "NO_IMAGE",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "PROHIBITED_CONTENT",
        "SAFETY",
    ],
)
def test_blocked_finish_reasons(finish):
    resp = image_response(png_bytes((10, 10)), finish=getattr(types.FinishReason, finish))
    assert promo.generate_scene(fake_client(resp), "m", photo(), "cafe") == (None, "blocked")


def test_text_only_answer_is_blocked_and_errors_are_classified(monkeypatch):
    monkeypatch.setattr(promo, "PROMO_RETRY_DELAYS", ())
    assert promo.generate_scene(fake_client(text_response("I can't")), "m", photo(), "c") == (
        None,
        "blocked",
    )
    assert promo.generate_scene(fake_client(httpx.ReadTimeout("t")), "m", photo(), "c") == (
        None,
        "timeout",
    )
    assert promo.generate_scene(fake_client(RuntimeError("x")), "m", photo(), "c") == (
        None,
        "error",
    )
    assert promo.generate_scene(None, "m", photo(), "c") == (None, "error")


def test_take_unit_counts_before_the_call_and_respects_the_cap():
    b = DailyBudget(2)
    assert [promo.take_unit(b, 1, "d") for _ in range(3)] == ["", "", "daily_limit"]
    assert promo.take_unit(b, 2, "d") == ""  # per guild
    assert promo.take_unit(b, 1, "d2") == ""  # per day
    assert promo.take_unit(DailyBudget(0), 1, "d") == "disabled"


def test_settings_defaults():
    s = Settings()
    assert (s.promo_image_model, s.promo_kits_per_day) == ("gemini-nano-banana-2.1", 20)
```
`tests/salid/promo_helpers.py` (imported by this file and Task 6's): `png_bytes(size)`, `to_png(img)`, `jpeg_with_orientation(size, orientation)` (Pillow `Image.Exif()` with tag `0x0112`), `photo()` = `promo.prepare_photo(png_bytes((800, 800)))`, `image_response(png, finish=types.FinishReason.STOP)` / `text_response(text)` building `types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[...]), finish_reason=finish)])`, and `fake_client(*results)` = `conftest.FakeClient([...])` with each exception result wrapped in a callable that raises it (`FakeModels` records `(model, contents, config)` in `.calls`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'salid.promo'`.

- [ ] **Step 3: Implement in `src/salid/promo.py` and `config.py`**

Module docstring: what a promo kit is, that the image model only restyles the scene and never draws text, the one-call cost rule, and that the photo / spec memory is process-only. `merge_spec`: clean every argument; `"-"` → empty; empty → `prev` value (when `prev`), else default (`shop_name` → `clean(default_shop)[:30]`, `accent` → `"red"`); then check `accent in ACCENTS`, the `LIMITS`, non-empty headline and `len(scene) >= 3`, raising `ValueError(f"{field} ...")` naming the field. `prepare_photo`: `Image.open` → `ImageOps.exif_transpose` → composite RGBA/LA/P-with-transparency on white → RGB → `thumbnail((2048, 2048))` → JPEG quality 90; any exception → `None`.

`scene_prompt` returns exactly this text with `{scene}` filled (copy from the spec):
```
Edit this product photo into a professional advertising photo.
Keep the product exactly as it is: same shape, colours, label, logo, packaging, text printed on the product and quantity. Do not add, remove or redraw any part of the product.
Change only the background, the surface it stands on, the lighting and small props.
New scene: {scene}.
Composition: vertical 9:16, product centred and fully visible in the middle half of the height, the top 20% and bottom 25% of the frame plain, calm background with no objects.
No text, letters, numbers, logos, watermarks, price tags or signs anywhere in the new scene.
No people, hands or faces.
```
`generate_scene`: `None` client → `(None, "error")`; call `llm.generate(client, model, [types.Part.from_bytes(data=photo.jpeg, mime_type="image/jpeg"), scene_prompt(scene)], config, delays=PROMO_RETRY_DELAYS)`; a finish reason in `{IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, NO_IMAGE, IMAGE_RECITATION, IMAGE_OTHER, PROHIBITED_CONTENT, SAFETY}` or no `inline_data` image part → `"blocked"`; `httpx.TimeoutException` → `"timeout"`; any other exception → `"error"` (log a WARNING with `type(e).__name__` only); success → `Image.open(BytesIO(data)).convert("RGB")`. Settings: add the two fields with comments after `doc_pages_per_day` in `config.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: all passed (17 with the parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/salid/promo.py src/salid/config.py tests/salid/test_promo.py tests/salid/promo_helpers.py
git commit -m "Salid: promo kit request, photo preparation and the Nano Banana scene call"
```

---

### Task 4: Fallback scene and the three layouts

**Files:**
- Modify: `src/salid/promo.py`
- Test: `tests/salid/test_promo.py`

**Interfaces:**
- Consumes: Task 2 primitives; `PromoSpec`, `Photo`, `ACCENTS` (Task 3); `CardFonts`, `drawable` (Task 1).
- Produces:
  - `fallback_scene(photo: Photo) -> Image.Image` (9:16, 1536×2752)
  - `@dataclass(frozen=True) class Format(name: str, size: tuple[int, int], safe_top: int, safe_bottom: int, headline: tuple[int, int], price: tuple[int, int], cta: tuple[int, int], shop: tuple[int, int])`
  - `FORMATS: tuple[Format, ...]` in order `promo-1x1.jpg`, `promo-4x5.jpg`, `promo-9x16.jpg`
  - `@dataclass class Placed(kind: str, box: tuple[int, int, int, int])` (`kind` ∈ `shop`, `headline`, `price`, `cta`)
  - `MARGIN = 64`
  - `render_format(scene: Image.Image, spec: PromoSpec, fmt: Format, fonts: CardFonts) -> tuple[Image.Image, list[Placed]]`
  - `render_kit(scene: Image.Image, spec: PromoSpec, fonts: CardFonts) -> list[tuple[str, bytes]]`

`FORMATS` values (sizes are (max, min) px):

| name | size | safe_top | safe_bottom | headline | price | cta | shop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `promo-1x1.jpg` | 1080×1080 | 48 | 48 | (80, 40) | (64, 36) | (40, 32) | (32, 24) |
| `promo-4x5.jpg` | 1080×1350 | 48 | 48 | (88, 44) | (64, 36) | (40, 32) | (32, 24) |
| `promo-9x16.jpg` | 1080×1920 | 270 | 384 | (96, 48) | (72, 40) | (44, 32) | (36, 28) |

- [ ] **Step 1: Write the failing tests**

```python
FULL = promo.merge_spec(
    None,
    headline="ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20%",
    scene="cafe",
    price="59.-",
    call_to_action="สั่งเลยวันนี้",
    default_shop="ร้านบ้านกาแฟ",
)
FONTS = cards.card_fonts()


def test_kit_names_sizes_order_and_bytes():
    files = promo.render_kit(Image.new("RGB", (1536, 2752), "gray"), FULL, FONTS)
    assert [n for n, _ in files] == ["promo-1x1.jpg", "promo-4x5.jpg", "promo-9x16.jpg"]
    sizes = [Image.open(io.BytesIO(d)).size for _, d in files]
    assert sizes == [(1080, 1080), (1080, 1350), (1080, 1920)]
    assert all(len(d) < 1_500_000 for _, d in files)


def test_any_scene_aspect_gives_exact_sizes():  # Review Focus 4
    for src in ((512, 512), (1600, 900), (300, 500)):
        files = promo.render_kit(Image.new("RGB", src, "gray"), FULL, FONTS)
        assert [Image.open(io.BytesIO(d)).size for _, d in files] == [f.size for f in promo.FORMATS]


@pytest.mark.parametrize("fmt", promo.FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize(
    "spec",
    [
        FULL,
        promo.merge_spec(None, headline="ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือนนี้", scene="cafe"),
        promo.merge_spec(None, headline="New menu", scene="cafe", default_shop=""),
    ],
)
def test_text_stays_inside_margins_and_out_of_safe_zones(fmt, spec):
    _, placed = promo.render_format(Image.new("RGB", (1536, 2752)), spec, fmt, FONTS)
    w, h = fmt.size
    for p in placed:
        x0, y0, x1, y1 = p.box
        assert promo.MARGIN <= x0 and x1 <= w - promo.MARGIN, p
        assert fmt.safe_top <= y0 and y1 <= h - fmt.safe_bottom, p
    for a, b in itertools.combinations(placed, 2):
        assert not overlap(a.box, b.box), (a, b)
    present = {"price": spec.price, "cta": spec.call_to_action, "shop": spec.shop_name}
    assert {p.kind for p in placed} == {"headline"} | {k for k, v in present.items() if v}


def test_9x16_layout_order():
    _, placed = promo.render_format(Image.new("RGB", (1536, 2752)), FULL, promo.FORMATS[2], FONTS)
    by = {p.kind: p.box for p in placed}
    assert by["shop"][3] <= by["headline"][1] < by["headline"][3] <= by["price"][1]
    assert by["price"][2] == 1080 - promo.MARGIN  # right-aligned
    assert by["cta"][3] == 1920 - 384 - 40  # 40 px above the safe zone
    assert abs((by["cta"][0] + by["cta"][2]) / 2 - 540) <= 1  # centred


def test_feed_layout_order():
    _, placed = promo.render_format(Image.new("RGB", (1536, 2752)), FULL, promo.FORMATS[1], FONTS)
    by = {p.kind: p.box for p in placed}
    assert by["price"][1] == 48 and by["price"][2] == 1080 - promo.MARGIN  # top-right
    assert by["shop"][0] == promo.MARGIN and by["cta"][2] == 1080 - promo.MARGIN
    assert by["headline"][3] <= min(by["shop"][1], by["cta"][1])  # headline above the last row
    assert max(by["shop"][3], by["cta"][3]) == 1350 - 48


def test_every_accent_pair_is_readable_large_text():
    assert all(cards.contrast(fill, text) >= 3.0 for fill, text in promo.ACCENTS.values())


def test_emoji_never_reaches_the_image(monkeypatch):  # Review Focus 3
    fitted = []
    real = cards.fit_text
    monkeypatch.setattr(
        cards, "fit_text", lambda text, *a, **k: fitted.append(text) or real(text, *a, **k)
    )
    spec = promo.merge_spec(None, headline="ลด 20% 🔥", scene="cafe", call_to_action="สั่งเลย 👉")
    promo.render_format(Image.new("RGB", (1536, 2752)), spec, promo.FORMATS[0], FONTS)
    assert "ลด 20%" in fitted and "สั่งเลย" in fitted
    assert not any("🔥" in t or "👉" in t for t in fitted)


def test_fallback_scene_contains_the_original_photo():
    red = Image.new("RGB", (800, 600), (220, 20, 20))
    p = promo.prepare_photo(to_png(red))
    s = promo.fallback_scene(p)
    assert s.size == (1536, 2752)
    r, g, b = s.getpixel((768, 1376))  # centre shows the sharp photo
    assert r > 180 and g < 60 and b < 60
```
Add an `overlap(a, b)` helper (`a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]`) and the `itertools` import.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: new tests FAIL with `AttributeError: module 'salid.promo' has no attribute 'render_kit'`.

- [ ] **Step 3: Implement in `src/salid/promo.py`**

`fallback_scene`: canvas 1536×2752 = `cover(photo, size)` → `GaussianBlur(40)` → darken ~35% (`ImageEnhance.Brightness(0.65)`), then paste the photo scaled to fit inside x ∈ [96, 1440], y ∈ [20%, 75%] of the height, centred.

`render_format` (per spec §7; content box x ∈ [64, 1016], width 952):
- Call card primitives through the module (`from . import cards`, then `cards.fit_text(...)`), so tests can patch them.
- `img = cards.cover(scene, fmt.size)`; every overlay field goes through `cards.drawable(text, font path)` before fitting (empty result = field not drawn); accent `fill, text_fill = ACCENTS[spec.accent]`.
- **9:16** (`fmt.safe_top == 270`): `scrim(img, 0, int(0.40 * h), 170, 0)` and `scrim(img, int(0.60 * h), h, 0, 190)`; `y = 270 + 24`; shop (Regular, one line) at `y`, then `y += height + 12`; headline (Bold, ≤ 2 lines) at `y`, then `y += height + 24`; price badge right-aligned at `x = 1016 - w`, top `y`; CTA pill centred, bottom at `h - 384 - 40`.
- **Feed** (4:5, 1:1): `scrim(img, int(0.55 * h), h, 0, 190)`; price badge top-right at `(1016 - w, 48)`; last row bottom at `h - 48`: CTA pill right-aligned at `x = 1016 - w`, shop name left at `x = 64`, fitted into `952 - cta width - 24` and vertically centred on the row; headline bottom at `row top - 24` (or `h - 48` when there is no row).
- Badge pads `(32, 16)`, price radius 24, CTA `radius=None` (pill). Returns the image and the `Placed` boxes from `draw_lines` / `draw_badge`.

`render_kit` = `[(f.name, to_jpeg(render_format(scene, spec, f, fonts)[0])) for f in FORMATS]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: all passed.

- [ ] **Step 5: Look at one kit by eye**

Run a throwaway snippet that renders `FULL` over `fallback_scene` of any local photo into `$CLAUDE_JOB_DIR/tmp/kit/` and open the three JPEGs with the Read tool. Expected: Thai marks stacked correctly, text readable over the scrims, nothing touching the edges. Fix layout constants if not (the geometry tests must stay green).

- [ ] **Step 6: Commit**

```bash
git add src/salid/promo.py tests/salid/test_promo.py
git commit -m "Salid: promo kit layouts (1:1, 4:5, 9:16) and the original-photo fallback scene"
```

---

### Task 5: Revision memory and the `make_promo_kit` tool

**Files:**
- Modify: `src/salid/promo.py`
- Test: `tests/salid/test_promo.py`

**Interfaces:**
- Consumes: everything from Tasks 3–4; `media.DailyBudget`.
- Produces:
  - `class PromoMemory(ttl_s: float = 1800.0, max_entries: int = 50, clock: Callable[[], float] = time.monotonic)` with `put(key: tuple[int, int, int], photo: Photo, spec: PromoSpec) -> None`, `get(key) -> tuple[Photo, PromoSpec] | None` (expired → None and dropped; LRU beyond `max_entries`)
  - `@dataclass class PromoCtx`: `guild_id: int`, `day: str`, `default_shop: str`, `photos: list[tuple[int, bytes]]` (attachment index, data of this message's usable photos), `memory: PromoMemory`, `key: tuple[int, int, int]`, `client: Any`, `model: str`, `budget: DailyBudget`, `fonts: CardFonts`, `outbox: list[tuple[str, bytes]] = field(default_factory=list)`; method `make_kit(headline: str = "", scene: str = "", price: str = "", call_to_action: str = "", shop_name: str = "", accent: str = "", file_index: int = -1) -> dict`
  - `promo_tools(ctx: PromoCtx) -> list[Callable]` → `[make_promo_kit]` with `make_promo_kit.outbox = ctx.outbox`

`make_kit` rules (spec §3, §5, §9): photo = `photos` entry with `file_index` (≥ 0; missing → `{"status": "error", "error": "file #N is not a usable photo (PNG, JPEG or WebP)"}`), else the first photo, else the remembered one, else `{"status": "no_photo", "note": "Ask the user to send the product photo with the request."}`. A new photo starts a fresh kit (`prev=None`); a remembered photo is a revision (`prev` = remembered spec). `merge_spec` errors → `{"status": "error", "error": str(e)}`; unreadable photo → `{"status": "error", "error": "the photo could not be read"}`. Then `reason = take_unit(...)`; when `""` → `generate_scene(...)`; no image → `fallback_scene`. `ctx.outbox[:] = render_kit(...)` (replace, never append), `memory.put(key, photo, spec)`, one INFO log `promo kit guild=%s outcome=%s model_ms=%d render_ms=%d bytes=%d` (outcome `ai` or `fallback:<reason>`).

Result `{"status": "ok", "files": [names], "ai_scene": bool, "reason": reason, "kits_left_today": max(0, int(cap - used)), "note": note}`; `note` is built from these exact sentences:
- always: `"Reply in the user's language: say the three images are attached (1:1 for feed and LINE, 4:5 for Facebook/Instagram feed, 9:16 for Stories, Reels and TikTok), then give a ready-to-paste caption with 3-5 hashtags. Do not list the overlay text again."`
- `ai_scene`: `" Mention briefly that the background was made with AI and the product is their own photo."`
- `blocked`: `" The AI scene was blocked by safety filters, so the kit uses the original photo; say so and suggest a different scene."`
- `error` / `timeout`: `" The AI scene service failed, so the kit uses the original photo; say so and that they can try again."`
- `daily_limit`: `" Today's AI scene limit for this server is used up, so the kit uses the original photo; say so."`
- `disabled`: `" AI scenes are turned off here, so the kit uses the original photo."`
- `short_side < 400`: `" The photo is small (under 400 px); suggest sending a larger photo for sharper results."`

The tool function's docstring (the model-facing description; exact copy):
```
Make a promo post kit from the product photo the user sent: the product stays exactly as
photographed, AI creates a new scene around it, and Salid typesets the text, giving three images
(1:1, 4:5, 9:16). Use for requests like "ทำโพสต์โปรจากรูปนี้", "ทำภาพโฆษณาลาเต้ 59 บาท",
"make an ad for this". For a change to the last kit ("ขอโทนอุ่นกว่านี้", "เปลี่ยนราคาเป็น 49")
pass only what changes; the photo and the other fields are remembered for 30 minutes.

Args:
    headline: main line on the image, max 40 characters, short and catchy, in the user's language.
    scene: the setting to create around the product, IN ENGLISH, max 300 characters, e.g.
        "wooden cafe table, soft morning window light, a few coffee beans". Never describe text.
    price: optional, max 20 characters, e.g. "59.-" or "ลด 20%". "-" removes it.
    call_to_action: optional, max 24 characters, e.g. "สั่งเลยวันนี้". "-" removes it.
    shop_name: optional, max 30 characters; default is the server name. "-" removes it.
    accent: badge colour: red (default), orange, green, blue, gold or black.
    file_index: which attached file (the number in its label); -1 = the first photo.
```
The tool wrapper catches every exception (`logger.exception("make_promo_kit failed")`) and returns `{"status": "error", "error": "the promo kit failed; try again"}`.

- [ ] **Step 1: Write the failing tests**

```python
def ctx(**kw):
    scene = image_response(png_bytes((768, 1376)))
    base = dict(
        guild_id=1,
        day="2026-10-16",
        default_shop="ร้าน A",
        photos=[(0, png_bytes((900, 900)))],
        memory=promo.PromoMemory(),
        key=(1, 10, 7),
        client=fake_client(scene, scene),
        model="m",
        budget=DailyBudget(20),
        fonts=cards.card_fonts(),
    )
    return promo.PromoCtx(**{**base, **kw})


def test_memory_ttl_lru_and_key_isolation():
    now = [0.0]
    m = promo.PromoMemory(ttl_s=1800, max_entries=2, clock=lambda: now[0])
    m.put((1, 1, 1), photo(), FULL)
    m.put((1, 1, 2), photo(), FULL)
    assert m.get((1, 1, 1)) and m.get((2, 1, 1)) is None and m.get((1, 2, 1)) is None
    m.put((1, 1, 3), photo(), FULL)  # evicts the least recently used: (1, 1, 2)
    assert m.get((1, 1, 2)) is None and m.get((1, 1, 1))
    now[0] = 1801
    assert m.get((1, 1, 1)) is None


def test_first_kit_ai_scene_and_outbox():
    c = ctx()
    [tool] = promo.promo_tools(c)
    out = tool(headline="ลาเต้เย็น", scene="cafe table", price="59.-")
    assert out["status"] == "ok" and out["ai_scene"] and out["reason"] == ""
    assert out["files"] == ["promo-1x1.jpg", "promo-4x5.jpg", "promo-9x16.jpg"]
    assert [n for n, _ in tool.outbox] == out["files"] and tool.outbox is c.outbox
    assert out["kits_left_today"] == 19 and "made with AI" in out["note"]
    assert len(c.client.models.calls) == 1


def test_second_call_in_one_turn_replaces_the_files():  # Review Focus 5
    c = ctx()
    [tool] = promo.promo_tools(c)
    tool(headline="a", scene="cafe")
    tool(headline="b", scene="beach")
    assert len(tool.outbox) == 3


def test_revision_uses_the_remembered_photo_and_fields():
    c = ctx()
    promo.promo_tools(c)[0](headline="ลาเต้เย็น", scene="cafe", price="59.-")
    c2 = ctx(photos=[], memory=c.memory)
    out = promo.promo_tools(c2)[0](scene="warmer light")
    assert out["status"] == "ok"
    _, spec = c.memory.get((1, 10, 7))
    assert (spec.headline, spec.scene, spec.price) == ("ลาเต้เย็น", "warmer light", "59.-")


def test_new_photo_starts_a_fresh_kit():
    c = ctx()
    promo.promo_tools(c)[0](headline="old", scene="cafe", price="59.-")
    c2 = ctx(memory=c.memory)  # a new photo in this message
    assert promo.promo_tools(c2)[0](scene="beach")["status"] == "error"  # headline required again


@pytest.mark.parametrize("reason", ["daily_limit", "disabled"])
def test_limits_fall_back_without_calling_the_model(reason):
    budget = DailyBudget(1 if reason == "daily_limit" else 0)
    budget.add(1, "2026-10-16", 1)  # the only unit is already used
    c = ctx(budget=budget, client=fake_client(RuntimeError("must not be called")))
    out = promo.promo_tools(c)[0](headline="h", scene="cafe")
    assert (out["status"], out["ai_scene"], out["reason"]) == ("ok", False, reason)
    assert c.client.models.calls == [] and len(c.outbox) == 3


def test_blocked_scene_falls_back_and_counts():
    c = ctx(
        client=fake_client(
            image_response(png_bytes((9, 9)), finish=types.FinishReason.IMAGE_SAFETY)
        )
    )
    out = promo.promo_tools(c)[0](headline="h", scene="cafe")
    assert (out["ai_scene"], out["reason"], out["kits_left_today"]) == (False, "blocked", 19)
    assert "blocked by safety filters" in out["note"]


def test_photo_choice_errors_and_small_photo_note():
    assert (
        promo.promo_tools(ctx())[0](headline="h", scene="cafe", file_index=3)["status"] == "error"
    )
    assert promo.promo_tools(ctx(photos=[]))[0](headline="h", scene="cafe")["status"] == "no_photo"
    assert (
        promo.promo_tools(ctx(photos=[(0, b"junk")]))[0](headline="h", scene="cafe")["error"]
        == "the photo could not be read"
    )
    small = promo.promo_tools(ctx(photos=[(0, png_bytes((300, 300)))]))[0](
        headline="h", scene="cafe"
    )
    assert "under 400 px" in small["note"]


def test_tool_never_raises_and_logs_no_user_text(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="salid.promo")
    c = ctx()
    promo.promo_tools(c)[0](headline="ความลับร้าน", scene="secret garden")
    assert "ความลับร้าน" not in caplog.text and "secret garden" not in caplog.text
    assert "promo kit guild=1 outcome=ai" in caplog.text
    monkeypatch.setattr(promo, "render_kit", lambda *a: 1 / 0)
    assert promo.promo_tools(ctx())[0](headline="h", scene="cafe") == {
        "status": "error",
        "error": "the promo kit failed; try again",
    }


def test_tool_schema_and_description():
    [tool] = promo.promo_tools(ctx())
    assert tool.__name__ == "make_promo_kit"
    assert "IN ENGLISH" in tool.__doc__ and "30 minutes" in tool.__doc__
    assert list(inspect.signature(tool).parameters) == [
        "headline",
        "scene",
        "price",
        "call_to_action",
        "shop_name",
        "accent",
        "file_index",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: new tests FAIL with `AttributeError: module 'salid.promo' has no attribute 'PromoMemory'`.

- [ ] **Step 3: Implement `PromoMemory`, `PromoCtx.make_kit` and `promo_tools` in `src/salid/promo.py`**

`PromoMemory` uses an `OrderedDict` (move to end on `get`/`put`, pop first beyond `max_entries`) and a `threading.Lock` (tools run in worker threads). Everything else is fixed by the rules above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -q -p no:warnings`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/salid/promo.py tests/salid/test_promo.py
git commit -m "Salid: make_promo_kit tool with revision memory, daily cap and fallback notes"
```

---

### Task 6: Discord wiring, feature flag, hub toggle, capability line

**Files:**
- Create: `src/salid/promo_bot.py`
- Modify: `src/salid/bot.py` (import + `PromoMixin` in `SecretaryBot` bases before `MediaMixin`; `self._init_promo()` right after `self._init_media()` in `__init__`; `_agent_kwargs` gains `files: list | None = None` and the `extra_tools` entry; `_answer` passes `files=files` to `_agent_kwargs`)
- Modify: `src/salid/guild_settings.py` (`"promo"` after `"meeting_transcript_doc"` in `FEATURES`; `promo: bool = True  # promo post kits from product photos (AI scene + typeset text)` in `Features`; the module docstring's flag list)
- Modify: `src/meeting_scribe/static/hub.html` (`{ id: "promo" }` after `{ id: "bizfin" }`; `promo: true` in the two feature literals; `promo: features.promo` in the save literal; i18n keys `f_promo_n`, `f_promo_d`, `f_promo_w` after `f_bizfin_w`)
- Modify: `src/salid/qa.py` (`CAPABILITIES_PROMPT`: one numbered line after the last item, before "ข้อจำกัด:")
- Test: `tests/salid/test_promo_bot.py`; update `tests/salid/test_guild_settings.py` and `tests/test_hub.py` default-feature expectations (`promo=True`)

**Interfaces:**
- Consumes: `promo.PromoCtx`, `promo.PromoMemory`, `promo.promo_tools`, `promo.PHOTO_MIMES` (Tasks 3, 5); `cards.thai_ready`, `cards.card_fonts`; `media.DailyBudget`; `qa.FileInput`.
- Produces: `PromoMixin._init_promo(self) -> None` (sets `self.promo_budget = DailyBudget(self.s.promo_kits_per_day)`, `self.promo_memory = PromoMemory()`); `PromoMixin._promo_tools(self, message, guild, asker, files: list[FileInput] | None) -> list` (never raises).

`_promo_tools` returns `[]` unless: `files is not None`; `self.team.enabled(guild.id, "promo")`; `cards.thai_ready()` (else log ERROR once per process: `"libfribidi missing: promo kits off"`); and the message has a usable photo (`f.mime in PHOTO_MIMES and f.data and not f.skip`) or `self.promo_memory.get((guild.id, message.channel.id, asker.id))` is not None. The context: `guild_id=guild.id`, `day=self.s.today().isoformat()`, `default_shop=guild.name or ""`, `photos=[(f.index, f.data) for usable f]`, `key=(guild.id, message.channel.id, asker.id)`, `client=self.llm`, `model=self.s.promo_image_model`, `budget=self.promo_budget`, `fonts=cards.card_fonts()`. In `_agent_kwargs` add `*([] if interjection else self._promo_tools(message, guild, asker, files)),` to `extra_tools` after the `_bizfin_tools` entry.

Hub copy (exact):
- `f_promo_n`: th `"โพสต์โปรโมทจากรูปสินค้า"`, en `"Promo posts from product photos"`
- `f_promo_d`: th `"ส่งรูปสินค้าพร้อมบอกโปร สลิดทำภาพโพสต์ 3 ขนาด (1:1, 4:5, 9:16) ตัวสินค้าคงเดิม พื้นหลังสร้างด้วย AI ข้อความภาษาไทยจัดวางโดยสลิด พร้อมแคปชัน แก้ต่อได้ภายใน 30 นาที"`, en `"Send a product photo with your offer: Salid makes post images in 3 sizes (1:1, 4:5, 9:16); the product stays as photographed, the background is made with AI, Salid typesets the text and writes a caption; changes work for 30 minutes"`
- `f_promo_w`: th `"ไม่เกิน 20 ชุดต่อวันต่อเซิร์ฟเวอร์ เกินแล้วใช้รูปเดิมแทนพื้นหลัง AI"`, en `"Up to 20 kits per server per day; beyond that the original photo is used instead of an AI background"`

Capability line (exact, number = last item + 1, currently `18)`):
`"18) โพสต์โปรโมทจากรูปสินค้า: ส่งรูปสินค้าให้สลิดพร้อมบอกโปร (ราคา ข้อความ ชื่อร้าน) สลิดทำภาพโพสต์ 3 ขนาด (1:1 ฟีด/LINE, 4:5 Facebook/Instagram, 9:16 Story/Reels/TikTok) พร้อมแคปชัน ตัวสินค้าคงเดิม พื้นหลังสร้างด้วย AI; ขอแก้ได้ภายใน 30 นาทีโดยไม่ต้องส่งรูปใหม่\n"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/salid/test_promo_bot.py
from conftest import GUILD
from promo_helpers import fake_client, image_response, png_bytes
from test_coordinator import world  # noqa: F401  (fixture: world.g, .pub, .secret, .staff, .boss)
from test_modes import TMsg, bot  # noqa: F401  (fixture)


def img_input(i=0, mime="image/png"):
    return FileInput(i, "p.png", mime, data=png_bytes((900, 900)))


def names(tools):
    return {f.__name__ for f in tools}


async def test_offered_only_with_a_usable_photo(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "ทำโพสต์")
    assert names(bot._promo_tools(msg, world.g, m, [img_input()])) == {"make_promo_kit"}
    assert bot._promo_tools(msg, world.g, m, []) == []
    assert bot._promo_tools(msg, world.g, m, None) == []  # voice path
    assert bot._promo_tools(msg, world.g, m, [img_input(mime="image/gif")]) == []
    skipped = img_input()
    skipped.skip = "over 15 MB"
    assert bot._promo_tools(msg, world.g, m, [skipped]) == []


async def test_remembered_photo_offers_the_tool_to_the_same_person_in_the_same_channel(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "ขอโทนอุ่นกว่านี้")
    bot.promo_memory.put(
        (GUILD, world.pub.id, m.id),
        promo.prepare_photo(png_bytes((900, 900))),
        promo.merge_spec(None, headline="h", scene="cafe"),
    )
    assert names(bot._promo_tools(msg, world.g, m, [])) == {"make_promo_kit"}
    other = world.boss
    assert bot._promo_tools(TMsg(world.g, world.pub, other, "x"), world.g, other, []) == []
    assert bot._promo_tools(TMsg(world.g, world.secret, m, "x"), world.g, m, []) == []


async def test_flag_off_or_no_raqm_means_no_tool(bot, world, monkeypatch):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"promo": False}), "x")
    bot.team._cache.clear()
    assert bot._promo_tools(msg, world.g, m, [img_input()]) == []
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"promo": True}), "x")
    bot.team._cache.clear()
    monkeypatch.setattr("salid.promo_bot.cards.thai_ready", lambda: False)
    assert bot._promo_tools(msg, world.g, m, [img_input()]) == []


async def test_agent_kwargs_add_the_tool_for_chat_not_interjections(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert "make_promo_kit" in names(
        bot._agent_kwargs(msg, world.g, m, False, files=[img_input()], **kw)["extra_tools"]
    )
    assert "make_promo_kit" not in names(
        bot._agent_kwargs(msg, world.g, m, False, **kw).get("extra_tools", [])
    )
    inter = bot._agent_kwargs(msg, world.g, m, False, files=[img_input()], interjection=True, **kw)
    assert "make_promo_kit" not in names(inter.get("extra_tools", []))


async def test_kit_is_bound_to_guild_channel_author_and_budget(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    bot.llm = fake_client(image_response(png_bytes((768, 1376))))
    [tool] = bot._promo_tools(msg, world.g, m, [img_input()])
    out = tool(headline="ลาเต้", scene="cafe")
    assert out["status"] == "ok" and len(tool.outbox) == 3
    assert qa.tool_files([tool]) == tool.outbox  # what Answer.attachments carries
    assert bot.promo_memory.get((GUILD, world.pub.id, m.id)) is not None
    assert bot.promo_budget.used(GUILD, bot.s.today().isoformat()) == 1
    assert bot.llm.models.calls[0][0] == "gemini-nano-banana-2.1"


def test_promo_flag_defaults_on_and_capability_line():
    assert "promo" in FEATURES and Features().promo is True
    assert "โพสต์โปรโมทจากรูปสินค้า" in qa.CAPABILITIES_PROMPT
```
Other imports: `FileInput` and `qa` (`salid.qa`), `promo`, `FEATURES` / `Features` / `GuildSettingsUpdate` (`salid.guild_settings`). Update the two existing default-feature assertions (`tests/salid/test_guild_settings.py` ~line 68, `tests/test_hub.py` ~line 295) with `promo=True`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo_bot.py tests/salid/test_guild_settings.py -q -p no:warnings`
Expected: FAIL (`AttributeError: 'SecretaryBot' object has no attribute '_promo_tools'`, `promo` missing from `Features`).

- [ ] **Step 3: Implement `promo_bot.py` and the edits listed under Files**

Module docstring for `promo_bot.py`: Discord glue of promo kits (logic in `promo.py`); the tool is offered only for chat answers with a photo or a remembered kit; guild, channel and author are bound here, never model input.

- [ ] **Step 4: Run the full suite**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q -p no:warnings`
Expected: all passed (live tests deselected), including `test_hub_page_lists_every_guild_feature_flag` and `test_answer_asks_with_the_message_guild` (unchanged: `files` goes to `_agent_kwargs`, not to `ask`).
Run: `uvx ruff check . && uvx ruff format --check .`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/salid/promo_bot.py src/salid/bot.py src/salid/guild_settings.py src/salid/qa.py \
  src/meeting_scribe/static/hub.html tests/salid/test_promo_bot.py tests/salid/promo_helpers.py \
  tests/salid/test_promo.py tests/salid/test_guild_settings.py tests/test_hub.py
git commit -m "Salid: promo kits in chat (flag, hub toggle, daily cap, capability line)"
```

---

### Task 7: Sample script, live test and docs

**Files:**
- Create: `scripts/salid_sample_promo.py`
- Modify: `tests/salid/test_salid_live.py` (one live test)
- Modify: `README.md` (a `- **Promo post kit**` bullet after the Documents bullet, ~line 401; env vars), `CLAUDE.md` (code-map rows for `cards.py` and `promo.py` / `promo_bot.py`; the two env vars in the Salid env-var paragraph)
- Test: `tests/salid/test_promo.py` (sample script smoke test)

**Interfaces:**
- Consumes: `promo.prepare_photo`, `promo.fallback_scene`, `promo.generate_scene`, `promo.merge_spec`, `promo.render_kit`, `cards.card_fonts`, `llm.make_client`, `Settings`.
- Produces: `scripts/salid_sample_promo.py` with `main(argv: list[str] | None = None) -> int`; usage `uv run python scripts/salid_sample_promo.py <photo> [--live] [--out DIR] [--headline TEXT] [--scene TEXT]` (defaults: out `/tmp/salid-promo`, headline `"ลาเต้เย็น หอมกรุ่น ลด 20%"`, price `"59.-"`, CTA `"สั่งเลยวันนี้"`, shop `"ร้านบ้านกาแฟ"`, scene `"wooden cafe table, soft morning window light, a few coffee beans"`); offline renders the fallback kit; `--live` makes one real `generate_scene` call (and says so before calling); prints each written path and size.

- [ ] **Step 1: Write the failing smoke test**

```python
def test_sample_script_renders_offline(tmp_path):
    src = tmp_path / "p.png"
    src.write_bytes(png_bytes((900, 900)))
    spec = importlib.util.spec_from_file_location("sample", Path("scripts/salid_sample_promo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(src), "--out", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "promo-1x1.jpg",
        "promo-4x5.jpg",
        "promo-9x16.jpg",
    ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest tests/salid/test_promo.py -k sample -q -p no:warnings`
Expected: FAIL (`FileNotFoundError` for the script).

- [ ] **Step 3: Write the script, the live test and the docs**

Live test (marked by the module's `pytestmark`), exact name `test_live_promo_kit`: builds a 900×900 synthetic product (a white mug shape on grey drawn with `ImageDraw`), `generate_scene(make_client(Settings()), Settings().promo_image_model, prepare_photo(png), "wooden cafe table, soft morning light")`, prints the reason, renders the kit over the scene (or the fallback), asserts three JPEGs with the `FORMATS` sizes, writes them to a temp dir and prints the path for a visual check. README bullet: what the kit does, the flag `promo`, `SALID_PROMO_IMAGE_MODEL`, `SALID_PROMO_KITS_PER_DAY` (20; 0 = AI off), the fallback, the 30-minute revisions, that Thai text is typeset by Salid (Pillow RAQM, needs `libfribidi0`), the sample script and the live test. CLAUDE.md rows in the Salid code-map table after `digest.py`.

- [ ] **Step 4: Run the full suite and lint**

Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib pytest -q -p no:warnings`
Expected: all passed, live deselected.
Run: `uvx ruff check . && uvx ruff format --check .`
Expected: clean.
Run: `uvx uv@latest run --frozen env DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python scripts/salid_sample_promo.py <any local photo> --out "$CLAUDE_JOB_DIR/tmp/sample"` and open the three images with the Read tool.
Expected: correct Thai, readable text, product untouched (offline fallback scene).

- [ ] **Step 5: Commit**

```bash
git add scripts/salid_sample_promo.py tests/salid/test_salid_live.py tests/salid/test_promo.py README.md CLAUDE.md
git commit -m "Salid: promo kit sample script, live test and docs"
```

**Human gates after Task 7 (do not run without approval):** the live test (`pytest tests/salid/test_salid_live.py -m live -k promo -s`, about $0.06), `scripts/salid_sample_promo.py <real product photo> --live` + visual check, push + PR, any deploy.
