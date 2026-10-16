# Salid image generation and hybrid video ads (design)

Date: 2026-10-17. Status: approved in conversation, awaiting written-spec review.
Branch: `feature/salid-image-video` (from `origin/main` 7def407).

## Goal

Salid creates images from text descriptions and finished, professional-looking vertical video
ads (15-30 s) for Thai small businesses and teams, inside Discord, with correct Thai text, safe
output and costs a server cannot run away with. Salid's edge over generic generators is the
workflow: the shop's own photos, Thai typography done right, team context and delivery in chat.

## Scope

In scope (chosen: "both in one spec", video = "15-30 s hybrid ad", approach A "agent tools +
background job"):

1. `create_image` agent tool: text-to-image (1-4 variations), optional reference photo, optional
   Thai overlay text typeset by Salid, 30-minute revisions.
2. `create_video_ad` agent tool: a background job that builds a 15/20/30 s 1080x1920 MP4 from a
   generated storyboard: scene stills (AI or the shop's photos) with motion, optional generated
   hero clips (Gemini Omni), Thai narration, Thai captions, optional music, an "AI" label.
3. Shared image-generation module used by `create_image`, video stills and the existing promo kit
   (removes the duplicated size/safety guard noted in the PR #26 review).
4. Feature flags, hub toggles, per-guild limits (persistent monthly budgets for video), docs.

Build order inside the one plan: images first (shippable alone), then the slideshow video ad,
then the Omni hero clip and music add-ons.

Out of scope: avatars / synthetic presenters, voice cloning, editing real people's photos,
posting to social networks, longer than 30 s videos, resuming jobs after a restart, a web UI.

## Constraints

- Thai text is never drawn by an image or video model: prompts to models are English; Thai is
  typeset with `salid.cards` (Pillow RAQM, system `libfribidi`). Thai is not on Google's image
  "best performance" language list; Omni evaluates English only.
- Models (verified 2026-10-16/17, see memory `salid-media-generation-constraints`):
  - images `gemini-nano-banana-2.1` (GA, `global`; 1K $0.0336, 2K $0.0504; rejects
    seed/temperature/topP/topK/logprobs; `candidateCount` 1);
  - video `gemini-omni-1.1-flash-preview` (Preview, `global`, Interactions API, 3-10 s,
    360p-4K, 9:16, up to 10 reference images, ~$0.10/s at 720p). Veo 3.1 is being retired
    (Gemini API previews shut down 2026-10-22; Agent Platform GA "Nov 17, 2026 or later") and is
    not used.
  - music (optional) `lyria-3-clip-preview` (Preview, instrumental, ~$0.04 per 30 s clip).
- **Omni and Lyria are unverified on this project**: nobody with project access has run a live
  call (the local shell's credentials belong to another organisation and were refused; the user
  has no Vertex access). Omni's model page also says pay-as-you-go is "not supported". Both are
  therefore optional add-ons, switched off by an empty model setting until a project owner runs
  the live probe. The slideshow ad must be complete and good without them.
- Discord: 20 MiB per attachment by default (larger with boosts, `guild.filesize_limit`); MP4
  H.264/AAC plays inline. Interaction replies must be deferred; results are posted later.
- One bot process (same as `memory_index.py`, `recent.py`): job concurrency limits are
  in-process; monthly budgets are persisted so a restart cannot reset them.
- Live calls (tests, probe) need human approval and someone with access to the GCP project.

## Success criteria

- "@Salid วาดรูป ..." returns 1-4 images in the requested aspect ratio within ~30 s, with any
  Thai overlay text correct; blocked requests get a clear Thai/English explanation.
- "@Salid ทำวิดีโอโฆษณา ... 20 วิ" posts a progress message at once and, within ~3 minutes
  (slideshow) or ~6 minutes (with a hero clip), a 1080x1920 MP4 of 20 s ± 1 s, H.264/AAC,
  under the guild's upload limit (default ≤ 19 MiB), with Thai narration, Thai captions inside
  the TikTok/Reels safe area and an "สร้างด้วย AI" label.
- Typical cost: image ~$0.04 (1K) / ~$0.06 (2K); 20 s slideshow ad ~$0.30; with one 6 s hero
  clip ~$0.90. No guild can exceed its daily (images) or monthly (video) limit, restarts included.
- With the Omni/Lyria settings empty, every video request still produces a finished ad.

## Current state (what we build on)

- `promo.py` (Nano Banana edit call, `prepare_photo` with a 50 MP header check, `PromoMemory`),
  `cards.py` (Thai text, scrims, shadows, cover crops, JPEG), `speech.say(settings, client, text,
  lang) -> bytes` and `pcm_duration_s` (Thai TTS), ffmpeg in the bot image, the `outbox` ->
  `Answer.attachments` -> `_deliver` path (with the text fallback when an upload is refused).
- `research.py` / `research_bot.py`: the background-job pattern (pure pipeline vs Discord glue:
  ack, background task, delivery, cancel, daily cap).
- `media.DailyBudget` (in-memory per guild/day); `coo_store.DocStore` (`put(kind, guild_id, key,
  doc)`, `find(...)`, SQLite or Firestore) for persistent records.
- google-genai 2.29.0 (pinned): `generate_content` with `ImageConfig(aspect_ratio, image_size,
  person_generation, prominent_people)`; `client.interactions.create/get/cancel` with
  `VideoResponseFormat(delivery="inline"|"uri", aspect_ratio, resolution, duration)` and
  `video_config.task`.

## Design

### 1. Shared image generation (`src/salid/imagegen.py`)

- `generate_images(client, model, prompt, *, refs: list[bytes] = (), aspect: str, size: str,
  count: int = 1) -> tuple[list[Image.Image], str]`: `count` parallel `generate_content` calls
  (the model returns one image per call), `response_modalities=["TEXT","IMAGE"]`,
  `ImageConfig(aspect_ratio, image_size, person_generation="ALLOW_ADULT",
  prominent_people="BLOCK_PROMINENT_PEOPLE")`, no sampling parameters, minimal thinking where
  accepted. Returns the images (the last non-thought image part per call) and a reason (`""`,
  `blocked`, `timeout`, `error`) when none came back. One attempt per call (no retries, so one
  bill per image).
- Block detection: `prompt_feedback.block_reason`, finish reasons IMAGE_SAFETY /
  IMAGE_PROHIBITED_CONTENT / IMAGE_RECITATION / IMAGE_OTHER / NO_IMAGE / SAFETY /
  PROHIBITED_CONTENT, or a text-only answer.
- `english_only(prompt)`: rejects Thai characters (the model would draw them).
- `open_capped(data, formats=None) -> Image`: the one decode guard, and only that: opens the image
  (optionally restricted to `formats`), refuses canvases over `MAX_DECODE_PIXELS` (50 MP) from the
  header and maps Pillow's `DecompressionBombError` to the same `ImageTooLarge` reason, before any
  pixel is decoded. Callers keep their own preparation: `promo.prepare_photo` (EXIF, transparency
  on white, ≤ 2048 px), `qa._gif_first_frame` (`formats=["GIF"]`, RGBA first frame ≤ 1536 px) and
  the video stills; the two existing constants/exceptions become this one.
- `promo.generate_scene` becomes a thin call into `generate_images` (same prompt, same result
  contract; its tests keep passing).

### 2. `create_image` tool (`imagegen.py` tool + `imagegen_bot.py` mixin)

```
create_image(prompt="", aspect="1:1", count=1, size="1K", headline="", subtitle="",
             accent="", file_index=-1, revise=0) -> dict
```

- `prompt`: English description written by the agent (≤ 600 chars, no Thai, never describing
  text); `aspect` one of 1:1, 4:5, 9:16, 16:9; `count` 1-4; `size` 1K or 2K; `headline` (≤ 40),
  `subtitle` (≤ 60), `accent` (promo palette): optional Thai/English overlay typeset by Salid on a
  scrim band (top for 9:16, bottom otherwise) with `cards`; `file_index`: an attached photo used
  as a reference; `revise` (1-4): edit result #N of the remembered last batch with the new prompt.
- Output: PNG (≤ 2K) files `image-1.png` … in the outbox; result dict with `files`,
  `count_made`, `reason`, `images_left_today`, and a `note` telling the agent to reply briefly,
  say the images were made with AI, and never describe people as real.
- Memory: per (guild, channel, user), 30 minutes, 50 entries (like `PromoMemory`), holding the
  last batch for `revise`.
- Offered when the `images` flag is on and Thai shaping works (`cards.thai_ready()`); not in voice
  answers or interjections.
- Limits: `SALID_IMAGES_PER_DAY` (20 per guild) and `SALID_IMAGES_PER_USER_DAY` (5), counted per
  image before the calls (`DailyBudget`); a request for more than what is left is trimmed and the
  note says so.

### 3. `create_video_ad` tool and the job (`videoad.py`, `videogen.py`, `videoad_bot.py`)

```
create_video_ad(brief="", seconds=20, use_photos=True, hero_shots=0, voice=True, music=False)
    -> dict
```

The tool validates and starts the job (one per guild); it returns `{"status": "started"}` and the
agent replies briefly. `hero_shots` (0-2) and `music` are ignored (with a note) while their model
settings are empty. Offered only when the `video_ads` flag is on (default OFF) for chat answers.

Job steps (`videoad.py`, pure; the bot runs them in a worker thread and edits one progress
message, at most every 5 s, with a Cancel button for the requester or an admin):

1. **Storyboard**: one call to the main model with a JSON schema. `Storyboard(title, scenes[4-6])`,
   each `Scene(image_prompt_en ≤ 300, caption_th ≤ 40, voice_th ≤ 120, seconds 2-8,
   source "ai" | "photo:N" | "hero")`; total = `seconds` ± 1; prompts English-only; at most
   `hero_shots` hero scenes; narration length fitted to the scene (≈ 12 Thai chars/s, trimmed by
   code). Safety rules in the instructions: no celebrities or real public figures, no children,
   no health/cure claims for food, cosmetics or supplements, products from the shop's photos when
   given. One re-ask on validation failure, then fail with a clear message.
2. **Stills**: `generate_images(... aspect="9:16", size="2K")` per AI scene (in parallel, max 3);
   shop photos via `open_capped` and a 9:16 framing (cover crop, or the promo fallback framing on a
   blurred fill when the aspect differs a lot). A failed still is retried once, then replaced by a
   plain accent-colour card with the caption.
3. **Motion**: each still becomes a 30 fps 1080x1920 clip of its scene length with a slow zoom or
   pan (`zoompan`), direction alternating.
4. **Hero clip** (only when `SALID_VIDEO_HERO_MODEL` is set): `videogen.start(client, model,
   still, prompt_en, seconds 4-6)` -> Omni Interactions `image_to_video`, 720p, 9:16,
   `delivery="inline"`, `background=True`; `poll` every 10 s up to 8 min; `cancel` on job cancel.
   Prompts add "single continuous shot, no on-screen text, no dialogue" and never contain quotes.
   Blocked/failed/timeout -> that scene uses its still with motion.
5. **Captions and label**: per scene a transparent 1080x1920 PNG from `cards` (Thai caption with
   shadow inside the safe area: y 270-1250 px, ≥ 65 px from the sides) and a small "สร้างด้วย AI"
   label; overlaid with a short fade.
6. **Audio**: narration per scene from `speech.say` (WAV via ffmpeg), placed at each scene start;
   optional music (only when `SALID_VIDEO_MUSIC_MODEL` is set) ducked under the voice; `loudnorm`
   to about -16 LUFS. Narration failure -> captions only.
7. **Assemble**: ffmpeg concat with 0.4 s `xfade`, overlays, audio mix; H.264 High, yuv420p,
   30 fps, AAC 128 kbps 48 kHz, `+faststart`; video bitrate = `budget_MiB × 8388.6 / seconds −
   128` kbps with `maxrate`/`bufsize`; budget = min(19 MiB, guild limit − 1 MiB). Over budget ->
   one re-encode at 80 % bitrate, then 720p. Output `ad-YYYYMMDD-HHMM.mp4`.
8. **Deliver**: reply to the request with the MP4, the storyboard as a short list and a suggested
   Thai caption with "#สร้างด้วยAI"; upload refused -> text with the reason (existing fallback).

### 4. Jobs, limits and settings

- `VideoJob` records in DocStore kind `salid_video_jobs` (id, guild, channel, requester, state:
  queued / storyboard / stills / hero / audio / render / done / failed / cancelled, timestamps,
  estimated and actual cost, error reason). The brief, storyboard text and media are never stored.
  On startup, jobs left running are marked failed and the requester is told to try again.
- Monthly usage in DocStore kind `salid_media_usage`, key `YYYY-MM`, per guild: `ads`,
  `hero_seconds`, `music_clips`; checked and incremented before each paid step.
- Settings: `SALID_IMAGE_MODEL` (`gemini-nano-banana-2.1`), `SALID_IMAGES_PER_DAY` (20),
  `SALID_IMAGES_PER_USER_DAY` (5), `SALID_VIDEO_ADS_PER_MONTH` (10),
  `SALID_VIDEO_HERO_MODEL` ("" = off; `gemini-omni-1.1-flash-preview` once verified),
  `SALID_VIDEO_HERO_SECONDS_PER_MONTH` (60), `SALID_VIDEO_MUSIC_MODEL` ("" = off),
  `SALID_VIDEO_MAX_JOBS` (2 across the bot). Promo keeps `SALID_PROMO_*`.
- Flags: `images` (default on), `video_ads` (default off); hub toggles with Thai/English copy.

### 5. Safety, privacy, observability

- Model safety settings above; our English-only guard; storyboard rules; regulated-category rule
  (food / cosmetics / supplements: no health or cure claims in captions or narration).
- No editing of real people's photos; reference photos are products or logos (the tool
  description says so; Omni refuses some recognisable people).
- The "สร้างด้วย AI" label on every video and the AI note in replies; Google's SynthID and C2PA
  are on model outputs (our edits break C2PA, which is documented).
- Logs: one line per image call and per job step (guild, step, ms, cost estimate, outcome); never
  prompts, captions, narration, images or video.

### 6. Error handling summary

Tools never raise; every result has a status. Video jobs end in `done`, `failed` (with a reason
the requester sees) or `cancelled`. Partial failures degrade (still card, still instead of hero,
captions without voice) instead of failing the ad.

## Testing

Offline (default suite):
- `imagegen`: request config (no sampling params, safety strings, aspect/size), parallel count,
  block reasons, English-only guard, `open_capped` (50 MP, bomb error, EXIF, transparency), promo
  still passes through it.
- `create_image`: limits, trimming, overlay placement and Thai rendering, `revise` memory, file
  names, tool never raises, no user text in logs.
- `videoad`: storyboard validation (scene counts, totals, English-only, hero cap, narration fit),
  still fallback card, caption PNG geometry inside the safe area, a real ffmpeg render of a tiny
  2-scene 4 s ad (duration ± 0.2 s, H.264/AAC, size under budget), the re-encode ladder.
- `videogen`: start/poll/cancel/blocked/timeout against a fake Interactions client.
- Job runner: state sequence, cancel during each step, restart marks running jobs failed,
  monthly budget persistence and refusal at the limit, one job per guild, global max jobs.
- Bot wiring: flags, video off by default, admin-only cancel, tool offered only in chat answers.

Live (marked `live`, need approval and GCP project access): one image call; one Omni 4 s 360p
probe (`scripts/salid_media_probe.py`, inline delivery, nothing stored); one full slideshow ad.

## Risks

| Risk | Mitigation |
| --- | --- |
| Omni unavailable on PayGo / Preview changes | Optional, off by default; slideshow ad is complete without it; model in settings |
| Cost runaway | Daily image caps, persistent monthly video budgets, one job per guild, global max jobs, video off by default |
| Garbled Thai, uncanny people or food | Thai typeset in code; shop photos preferred; no synthetic presenters; safety settings |
| Thai FDA / likeness / AI-labelling rules | Regulated-category rule, no real people, AI label and reply note |
| Long jobs vs Discord limits | Deferred reply, progress message, result posted later, size budget re-encode |
| ffmpeg render time / memory in the shared bot | 1080x1920 at 30 fps for ≤ 30 s, max 2 jobs, worker threads |
| Model churn | All model IDs are settings; probe script to re-check |

## Rollout

1. Images (`images` on) — can ship first.
2. Slideshow video ads (`video_ads` off; admins enable per guild).
3. After a project owner runs the live probe: set `SALID_VIDEO_HERO_MODEL` (and optionally
   `SALID_VIDEO_MUSIC_MODEL`).
Deploys and live tests only with human approval.

## Task order

1. `imagegen.py` core (`generate_images`, block detection, English-only, `open_capped`) and promo
   refactor onto it.
2. `create_image` tool + memory + overlay + limits; `imagegen_bot.py`, `images` flag, hub, docs.
3. `videoad.py` storyboard model and validation.
4. Stills, still fallback card, motion clips, caption PNGs.
5. Audio (narration) and ffmpeg assembly with the size budget.
6. Job runner, DocStore records and monthly budgets; `videoad_bot.py` (progress, cancel,
   delivery), `video_ads` flag, hub, docs.
7. `videogen.py` (Omni add-on) and the optional music step behind their settings; probe script;
   live tests.
