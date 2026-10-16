# Salid promo post kit: product photo to ready-to-post images (design)

Date: 2026-10-16. Status: approved in conversation, awaiting written-spec review.
Branch: `feature/salid-promo-kit` (from `origin/main` 3f002dc). Target: demo-ready before the
2026-10-18 submission.

## Goal

A shop owner or staff member drops a real product photo into Discord with a request such as
"@Salid ทำโพสต์โปร ลาเต้เย็น 59 บาท ลด 20% ถึงสิ้นเดือน ร้านบ้านกาแฟ" and gets, in under a
minute, three ready-to-post images (1:1, 4:5, 9:16) with perfect Thai text plus a ready-to-paste
caption. The product stays exactly as photographed; AI restyles only the scene around it. This is
Salid's first image *output* and the base (a Thai image-card renderer) that later visual features
(chart images, digest cards, narrated recap videos) reuse.

## Scope

In scope (chosen: "promo post kit" first, "restyle the scene", "shop name only", approach A):

1. `cards.py`: reusable Thai-safe image-card renderer (fonts, text fitting, scrims, badges, pills,
   cover crops, JPEG output).
2. `promo.py`: validated kit request, the scene prompt, one Nano Banana edit call, the three
   layouts, the fallback kit, the revision memory and the `make_promo_kit` agent tool.
3. `promo_bot.py`: Discord glue (feature flag, daily cap, photo choice, binding guild / channel /
   author).
4. Wiring: `bot.py` (`_agent_kwargs`), `guild_settings.FEATURES` + hub toggle, `config.py`,
   one capability line in `qa.py`, `pyproject.toml` (Pillow to runtime), `Dockerfile.salid`
   (`libfribidi0`), bundled Kanit fonts, docs.

Out of scope (later sub-projects, each with its own spec): brand kit (logo + colours in the hub),
chart / infographic images, visual weekly digest card, narrated recap video, SOP explainers, menu
boards / LINE rich menus, Veo video clips, posting the kit to other channels or social networks,
AI-generated people or food, any Thai text drawn by the image model.

## Constraints

- Thai text is typeset by our code, never by the image model: Thai is not on Google's
  "best performance" language list for image generation (ai.google.dev image-generation,
  Limitations, checked 2026-10-16), and a Thai marketing test rated Nano Banana weak on Thai text.
- Pillow draws Thai correctly only with `ImageFont.Layout.RAQM`; Pillow 12.3 wheels bundle raqm and
  HarfBuzz but need the system `libfribidi` at runtime (probe 2026-10-16: BASIC layout leaves dotted
  circles on tone marks, RAQM is correct). The bot image must install `libfribidi0`.
- Model: `gemini-nano-banana-2.1` (GA 2026-10-06, location `global` only, which is the bot's
  existing `gemini_location`). Setting `temperature`, `topP`, `topK`, `seed` or `logprobs` on it
  returns an API error (model page, checked 2026-10-16).
- Discord-only output; attachments go through the existing `outbox` -> `Answer.attachments` path
  (max 10 files per reply).
- One bot process (same assumption as `memory_index.py` and `recent.py`): the daily cap and the
  revision memory are in process memory.
- Live Vertex calls (the live test, the sample script's `--live`) need human approval (CLAUDE.md).
- Offline tests only in the default suite; no network, no real model.

## Success criteria

- A request with a product photo returns `promo-1x1.jpg` (1080×1080), `promo-4x5.jpg`
  (1080×1350) and `promo-9x16.jpg` (1080×1920), each JPEG under 1.5 MB, plus a caption in the
  reply, in one agent turn.
- Thai headline, price, call to action and shop name render with correct tone-mark stacking
  (RAQM) and never leave their boxes or the format's safe zones.
- Exactly one image-model call per kit; cost per kit about $0.05–0.06 at 2K (Nano Banana 2.1 list
  price $0.05 per 2K output image plus a ~1,120-token input image, from the research report,
  2026-10-16).
- When the model blocks, fails, times out or the daily cap is reached, the user still gets a kit
  built from the original photo and the reply says why.
- Revisions in the same channel within 30 minutes need no re-upload and only the changed fields.
- Ordinary chats pay nothing: the tool (and its description tokens) is offered only when a photo is
  available.

## Current state (findings that shape the design)

- Attachments of an addressed message become `qa.FileInput` objects (bytes already downloaded by
  `SecretaryBot._file_inputs`, limits in `qa.select_files`: 4 files, 15 MB each). Receipt recording
  already uses the "model picks an attachment number, code binds everything else" pattern
  (`FileActions`, `build_file_tools`).
- Feature modules expose agent tools as functions with an `outbox` attribute (`meeting_doc.py`,
  `ideas.py`, `bizfin_tools.py`); `qa.tool_files` collects them into `Answer.attachments` and
  `bot._answer` uploads them. All extra tools are assembled in ONE list in
  `SecretaryBot._agent_kwargs`, each behind a `*_bot.py` mixin method that checks its feature flag.
- `media.DailyBudget(cap)` is an in-memory per-guild, per-local-day counter (used for media minutes
  and document pages).
- `pdf_report.find_fonts()` finds a Thai font (bundled dir `src/salid/fonts/`, empty today, then
  TLWG paths in the Docker image). Pillow is only in the dev dependency group today, so the
  production image (`uv sync --no-dev`) does not have it.
- google-genai 2.29.0 (locked) has `ImageConfig(aspect_ratio, image_size, output_mime_type)`,
  `GenerateContentConfig.http_options.timeout` and finish reasons `IMAGE_SAFETY`,
  `IMAGE_PROHIBITED_CONTENT`, `NO_IMAGE`, `IMAGE_RECITATION`, `IMAGE_OTHER`, `PROHIBITED_CONTENT`
  (checked in the project venv).
- No image or video generation exists anywhere in the repo.

## Design

### 1. User experience

- **Trigger:** any message addressed to Salid (mention, name, reply, DM, chat-mode channel) with a
  PNG / JPEG / WebP photo and a request to make a post, ad, promotion or similar. No new command;
  the agent decides to call `make_promo_kit`.
- **Reply:** the three JPEGs attached in the order 1:1, 4:5, 9:16, and reply text written by the
  agent: one line saying which file is for what (1:1 feed / LINE, 4:5 Facebook / Instagram feed,
  9:16 Stories / Reels / TikTok), a ready-to-paste caption with 3–5 hashtags in the reply language,
  and (when the scene is AI-made) a short note that the background was made with AI. No extra model
  call writes the caption.
- **Revisions:** for 30 minutes the same person in the same channel can ask for changes without
  re-uploading ("ขอพื้นหลังโทนอบอุ่นกว่านี้", "เปลี่ยนราคาเป็น 49", "ไม่ต้องมีราคา"). Only changed
  fields are passed; the rest come from the remembered kit. A new photo in the message replaces the
  remembered one.
- **Language:** follows the existing reply-language rules. Overlay text is whatever the agent writes
  (Thai or English); the bundled font covers both.
- **Chat only:** not offered in voice answers and not in interjections (join mode).

### 2. Components

| Unit | Kind | Responsibility | Depends on |
| --- | --- | --- | --- |
| `src/salid/cards.py` | new, pure | Fonts (`card_fonts()`), `thai_ready()` (RAQM available), `fit_text` (font size + lines for a box), `draw_text_block`, `scrim` (vertical gradient), `badge` / `pill` (rounded shapes with centred text), `cover` (scale + centre crop to a size), `to_jpeg` (quality ladder under a byte cap) | Pillow, `pdf_report.find_fonts` (fallback) |
| `src/salid/promo.py` | new, pure (no Discord) | `PromoSpec` (validation), `merge_spec` (revision defaults), `scene_prompt`, `generate_scene` (one model call, returns image or a reason), `fallback_scene`, `render_kit` (three layouts), `PromoMemory` (30 min, 50 entries), `PromoCtx` + `promo_tools(ctx)` -> `make_promo_kit` with `outbox` | `cards`, `llm.generate`, google-genai types |
| `src/salid/promo_bot.py` | new mixin `PromoMixin` | `_promo_tools(message, guild, asker, files, lang)`: flag + `thai_ready` check, photo choice, budget, memory key, builds `PromoCtx` | `promo`, `media.DailyBudget` |
| `src/salid/fonts/` | new data | `Kanit-Bold.ttf`, `Kanit-Regular.ttf`, `OFL.txt` (google/fonts `ofl/kanit`, SIL OFL 1.1) | — |
| `bot.py` | edit | `PromoMixin` in `SecretaryBot` bases; `_agent_kwargs(..., files=None)`; chat `_answer` passes `files`; promo tools appended to `extra_tools` when not an interjection | — |
| `guild_settings.py` + `hub.html` | edit | `promo` in `FEATURES` and `Features` (default `True`); hub toggle with Thai/English name, description and warning keys (`f_promo_n/_d/_w`), in the three feature literals | — |
| `config.py` / `.env.example` | edit | `promo_image_model = "gemini-nano-banana-2.1"`, `promo_kits_per_day = 20` (`SALID_PROMO_*`) | — |
| `qa.py` | edit | One numbered line in `CAPABILITIES_PROMPT` (next free number) describing promo kits | — |
| `pyproject.toml` / `uv.lock` | edit | `pillow>=12.3.0` moves from the dev group to runtime dependencies (via `uvx uv@latest`, lock revision unchanged) | — |
| `Dockerfile.salid` | edit | `libfribidi0` added to the apt line, with a comment | — |
| `scripts/salid_sample_promo.py` | new | `<photo> [--live] [--out DIR]`: offline renders the fallback kit; `--live` makes one real call | `promo` |

Data flow: photo bytes (`FileInput.data`) -> `PromoCtx.photo()` -> `prepare_photo` (EXIF
rotation, RGB, long side ≤ 2048 px, JPEG q90) -> `generate_scene` (Nano Banana, `global`) -> PNG ->
`render_kit` -> three JPEGs -> `outbox` -> `Answer.attachments` -> Discord upload.

### 3. The agent tool

```
make_promo_kit(headline="", scene="", price="", call_to_action="", shop_name="",
               accent="", file_index=-1) -> dict
```

- `headline`: the main line on the image (≤ 40 characters). Required for a first kit.
- `scene`: the setting to create around the product, **in English** (≤ 300 characters), for example
  "wooden café table, soft morning window light, a few coffee beans". Required for a first kit.
- `price` (≤ 20), `call_to_action` (≤ 24), `shop_name` (≤ 30): optional. `"-"` removes a field
  that the remembered kit had. Empty shop name on a first kit = the server name, clipped to 30.
- `accent`: one of `red` (default), `orange`, `green`, `blue`, `gold`, `black`.
- `file_index`: which attachment of this message (the number in the file label); `-1` = the first
  usable photo of this message, else the remembered photo.
- On a revision every empty argument means "same as the remembered kit".

The docstring (the tool description the model sees) says when to use it, that the scene must be
English and must not describe text, that overlay text is typeset by Salid, and what to write in the
reply (files, caption with hashtags, AI-background note, limit / fallback explanation). No separate
prompt section is added to `qa.py`; the guidance travels with the tool, so it costs nothing when
the tool is absent.

Results (never raises):

- `{"status": "ok", "files": [...3 names], "ai_scene": true|false, "reason": "" | "blocked" |
  "error" | "timeout" | "daily_limit" | "disabled", "kits_left_today": n, "note": "..."}`
- `{"status": "error", "error": "..."}` for validation problems (field too long, unknown accent,
  missing headline / scene on a first kit, `file_index` not a usable photo) so the agent can retry.
- `{"status": "no_photo", "note": "..."}` when neither this message nor the memory has a photo
  (defensive; the tool is normally not offered then).

### 4. When the tool is offered

`PromoMixin._promo_tools` returns `[make_promo_kit]` only when all hold, else `[]` (never raises):

1. `files is not None` (chat path; voice never passes files) and not an interjection;
2. `self.team.enabled(guild.id, "promo")`;
3. `cards.thai_ready()` (else one ERROR log per process: "libfribidi missing: promo kits off");
4. this message has a usable photo (`FileInput` with mime PNG / JPEG / WebP, data present, no
   `skip`) **or** `PromoMemory` holds an unexpired entry for `(guild_id, channel_id, user_id)`.

GIF and HEIC photos are not used for kits (the model input and Pillow decode paths are not covered);
the agent can still read them as usual.

### 5. The AI scene

One call through `llm.generate` (existing retry on transient errors):

```
model    = settings.promo_image_model                      # gemini-nano-banana-2.1
contents = [Part.from_bytes(prepared_photo, "image/jpeg"), prompt]
config   = GenerateContentConfig(
    response_modalities=["TEXT", "IMAGE"],
    image_config=ImageConfig(aspect_ratio="9:16", image_size="2K"),
    http_options=HttpOptions(timeout=60_000),
)   # no temperature / top_p / top_k / seed / logprobs: an API error on this model
```

Prompt (built in code; `{scene}` is the agent's English text after cleaning):

```
Edit this product photo into a professional advertising photo.
Keep the product exactly as it is: same shape, colours, label, logo, packaging, text printed on
the product and quantity. Do not add, remove or redraw any part of the product.
Change only the background, the surface it stands on, the lighting and small props.
New scene: {scene}.
Composition: vertical 9:16, product centred and fully visible in the middle half of the height,
the top 20% and bottom 25% of the frame plain, calm background with no objects.
No text, letters, numbers, logos, watermarks, price tags or signs anywhere in the new scene.
No people, hands or faces.
```

The user's Thai text never goes to the image model. The result is the first `inline_data` image
part. `ai_scene=false` with a reason when: finish reason is one of the image/prohibited reasons or
no image part comes back (`blocked`); an exception other than timeout (`error`); the HTTP timeout
(`timeout`). The cap is checked and one unit is added **before** the call (blocked calls are billed
too); cap reached -> `daily_limit`; cap 0 -> `disabled`.

### 6. Fallback scene

`fallback_scene(photo)`: a 9:16 canvas filled with the photo scaled to cover, Gaussian-blurred
(radius ~40) and darkened ~35%, with the original photo scaled to fit inside the middle band
(within the same safe zones as the AI prompt asks for) on top. The same layouts then apply, so the
user always gets three usable images.

### 7. Layouts

All formats: 1080 px wide, side margins 64 px, text white unless on an accent shape. Text never
enters the format's safe zones; geometry is checked by tests.

| Format | Size | Safe zones (no text) | Headline | Price badge | Call to action | Shop name |
| --- | --- | --- | --- | --- | --- | --- |
| 9:16 | 1080×1920 | top 270 px, bottom 384 px | top area below the safe zone, over a top scrim (dark → clear over 40% of the height), ≤ 2 lines, 96 → 48 px | rounded accent rectangle under the headline, right-aligned, ≤ 72 px text | accent pill centred, bottom edge 40 px above the bottom safe zone, over a bottom scrim | small (36 px) line above the headline |
| 4:5 | 1080×1350 | 48 px edges | bottom band over a bottom scrim (clear → dark over 45% of the height), ≤ 2 lines, 88 → 44 px | rounded accent rectangle top-right | accent pill bottom-right, on the last row under the headline | small (32 px), bottom-left on the call-to-action row |
| 1:1 | 1080×1080 | 48 px edges | as 4:5, 80 → 40 px | as 4:5 | as 4:5 | as 4:5 |

- Crops: each format is `cover(master, size)` (scale to cover, centre crop). The 1:1 keeps the
  middle ~56% of a 9:16 master's height, which contains the "middle half" the prompt asks for.
- Fonts: Kanit Bold for headline, price and call to action; Kanit Regular for the shop name;
  fallback to `find_fonts()` bold / regular (TLWG in the Docker image) when the bundled files are
  missing.
- Text fitting: try sizes from max to min in 4 px steps; break lines greedily at spaces; a token
  wider than the box is split only at grapheme-cluster boundaries (never before a combining mark,
  Unicode categories Mn / Mc, so Thai vowels and tone marks stay on their base letter). Text is
  never truncated silently; the field limits guarantee a fit at the minimum size.
- Line height: from the font's ascent + descent (×1.15), so above / below marks are never clipped.
- Accent colours (fill / text): red `#E53935`/white, orange `#E65100`/white, green `#2E7D32`/white,
  blue `#1E5EC8`/white, gold `#F2C14E`/`#1A1A1A`, black `#111111`/white. All badge and pill text is
  ≥ 32 px bold (WCAG "large text"), so tests check contrast ≥ 3:1 for every pair (colours may be
  adjusted to pass).
- Output: RGB JPEG, quality 90 (then 85, 80 if needed) to stay under 1.5 MB.

### 8. Fields and validation

`PromoSpec` (pydantic): `headline`, `scene`, `price`, `call_to_action`, `shop_name`, `accent`.
Strings are cleaned (control characters and XML-invalid characters removed, whitespace collapsed),
then checked against the limits above. Empty `headline` or `scene` after merging with the
remembered kit -> validation error. `scene` must be at least 3 characters and is passed only to the
image prompt; overlay fields are passed only to the renderer.

### 9. Revision memory

`PromoMemory`: key `(guild_id, channel_id, user_id)`; value = prepared photo (≤ 2048 px JPEG, so
roughly ≤ 1 MB), its spec and a timestamp. TTL 30 minutes from the last kit, at most 50 entries
(least recently used dropped). Process memory only: never written to disk, Firestore or logs.
Updated after every delivered kit (AI or fallback). A new photo in the message always wins over the
remembered one.

### 10. Limits, privacy and safety

- Daily cap per guild: `SALID_PROMO_KITS_PER_DAY` (20, about $1.20 per guild per day at most);
  local day as used by the media budget; resets on restart (same as the other daily budgets).
- Photos: the existing attachment limits apply; photos under 400 px on the short side still work but
  the result `note` tells the agent to mention low quality.
- Kits are posted only as the reply in the channel where they were asked for.
- Google states its generated images carry an invisible SynthID watermark; the reply discloses the
  AI background. Prompt forbids people and text; the model's own safety filters apply.
- Logs never contain the user's text, the prompt, or image bytes.

### 11. Observability

One INFO line per kit: `promo kit guild=<id> outcome=<ai|fallback:<reason>> model_ms=<n>
render_ms=<n> bytes=<total>`; one WARNING with the exception type on model errors. The existing
per-answer metrics (`Answer.model_calls`, tokens) are unchanged: the image call goes through
`llm.generate`, not the ADK agent.

## Testing

Offline (default suite), TDD per task:

- `tests/salid/test_cards.py`: `thai_ready()` is true in the dev environment (fails with a clear
  "install libfribidi" message otherwise: report it, do not work around it); text drawn with RAQM
  (a Thai string's rendering differs from BASIC and has no dotted-circle glyph); `fit_text` keeps
  every line inside the box, respects max lines, shrinks before splitting, splits Thai only at
  cluster boundaries; `cover` output sizes; `to_jpeg` under the cap; badge / pill text centred;
  every accent pair passes contrast ≥ 3:1 (large text); bundled fonts found and fallback used when
  absent.
- `tests/salid/test_promo.py`: `PromoSpec` limits, cleaning, accent values, `"-"` removal;
  `merge_spec` revision defaults; `scene_prompt` contains the scene and the keep-the-product rules
  and no Thai characters; `generate_scene` with a fake client: request config (model, 9:16, 2K,
  timeout, response modalities) and **no** sampling parameters; image returned on success; each
  block reason, no image part, exception and timeout -> fallback with the right reason; cap
  reached / cap 0 -> no model call; budget counted before the call; `render_kit` returns the three
  names and sizes in order, text boxes inside canvas and outside safe zones, under 1.5 MB each;
  fallback scene size and that it contains the original photo; `PromoMemory` TTL, LRU bound, key
  isolation; the tool result shapes (`ok`, `error`, `no_photo`) and that the tool never raises;
  logs contain no user text.
- `tests/salid/test_promo_bot.py` (or in the multitenant bot tests): tool offered only with a usable
  photo or a remembered entry, never with `files=None` (voice) or for interjections, never with the
  flag off or without RAQM; guild / channel / user binding of the memory key; files from the outbox
  reach `Answer.attachments`.
- Existing parity tests: `FEATURES` / `Features` / hub toggles and i18n keys
  (`test_guild_settings.py`, `tests/test_hub.py`), the pinned `ask()` kwargs in
  `test_bot_multitenant.py` if they change.

Live (marked `live`, run only with human approval, about $0.06): `test_live_promo_kit` makes one
kit from a small synthetic product image, asserts three JPEGs of the right sizes and logs whether the
scene was AI-made; images saved under a temp dir for a visual check. `scripts/salid_sample_promo.py
<photo> --live` is the human visual gate with a real product photo.

## Risks

| Risk | Mitigation |
| --- | --- |
| The model alters the product (label, shape) | Strict keep-the-product prompt; live visual check before the demo; fallback kit always available; reply discloses AI background |
| The model adds text or signs to the scene | Prompt forbids it; live check; overlays sit on scrims |
| Product cut off in the 1:1 crop | Prompt asks for the middle half and plain top / bottom; live check; if it fails, approach B (one call per format) is the documented alternative |
| `libfribidi0` missing in the image -> broken Thai | `thai_ready()` gates the tool and logs an error; Dockerfile change; test fails loudly locally |
| Model ID / availability changes (preview churn) | `SALID_PROMO_IMAGE_MODEL` setting; errors fall back to the original-photo kit |
| Cost runaway | Per-guild daily cap (20), counted before each call; feature switch per guild |
| `qa.py` / `bot.py` conflicts with the fast-moving `main` | Edits limited to one capability line, one `_agent_kwargs` parameter and one `extra_tools` entry; rebase before the PR |
| In-memory cap and memory reset on restart | Accepted (same as the other daily budgets); one bot process |

## Rollout

1. Merge to `main` after review; the `promo` switch defaults on (licensed guilds only, through the
   existing license gate), capped at 20 kits per guild per day; admins can turn it off in the hub.
2. Before deploying the bot: human-approved live test + visual check with a real product photo;
   rebuild the image (adds `libfribidi0` and Pillow); deploy only with approval
   (`scripts/deploy_salid.sh --apply` is never run without it).
3. Demo script: photo + one request, then one revision ("ขอโทนอบอุ่นกว่านี้").

## Task order

1. Dependencies and fonts: Pillow to runtime, `libfribidi0` in the Dockerfile, bundled Kanit +
   OFL, `cards.thai_ready()` and font loading (+ tests).
2. `cards.py`: text fitting, text blocks, scrims, badges / pills, cover, JPEG ladder (+ tests).
3. `promo.py` part 1: `PromoSpec`, `merge_spec`, `scene_prompt`, `generate_scene` with the budget
   and fallback reasons (+ tests with a fake client).
4. `promo.py` part 2: `fallback_scene`, the three layouts, `render_kit` (+ geometry tests).
5. `promo.py` part 3: `PromoMemory`, `PromoCtx`, `promo_tools` / `make_promo_kit` (+ tests).
6. `promo_bot.py` + `bot.py` wiring + feature flag, hub toggle, config, capability line (+ tests).
7. Sample script, live test, README / CLAUDE.md / `.env.example` docs.
