"""Generated motion and music for video ads, through the google-genai Interactions API
(`client.interactions.create / get / cancel`). Both are optional add-ons, used only while their
model settings are set (`SALID_VIDEO_HERO_MODEL`, `SALID_VIDEO_MUSIC_MODEL`; empty = off).

- `start_hero` / `wait_hero`: one Gemini Omni clip of a hero scene from its still
  (`image_to_video`; `text_to_video` without a still), 9:16, 720p, 4-6 s, inline delivery, run
  in the background and polled every 10 s for up to 8 minutes; a job cancel or the timeout
  cancels the interaction (`cancel_hero`, also used by the bot for one a restart left running).
  The prompt must be English without quote characters (Omni evaluates English only, and quotes
  invite on-screen text); HERO_SUFFIX is appended.
- `music_clip`: one Lyria clip (instrumental), synchronous. UNVERIFIED: the call shape follows
  the Gemini API docs and the installed SDK (which maps the Lyria models' `outputs` onto
  `output_audio`); nobody with project access has run it on Vertex yet, so any failure is None.

Paid creates (`start_hero`, `music_clip`) should get `llm.make_media_client` (no SDK retry on
5xx, so one request never starts two paid jobs); polls and cancel the normal client.
`error_status` / `permission_denied` read both SDK error families (the Interactions API raises
compat errors such as PermissionDeniedError with `.status_code`, not `.code`). Model ids are
settings; `scripts/salid_media_probe.py` checks them live (human approval only).
Never logs prompts or media: only reasons, statuses and exception type names.
"""

from __future__ import annotations

import base64
import logging
import math
import re
import time

from . import imagegen

logger = logging.getLogger(__name__)

HERO_SUFFIX = " Single continuous shot, no on-screen text, no dialogue."
HERO_MIN_S, HERO_MAX_S = 4, 6  # requested clip length; videoad holds or trims it to the scene
HERO_ASPECT = "9:16"
HERO_RESOLUTION = "720p"
POLL_S = 10.0
TIMEOUT_S = 480.0  # 8 minutes, then the interaction is cancelled
PENDING = frozenset({"queued", "in_progress"})
MUSIC_SUFFIX = " Instrumental only, no vocals, about {seconds} seconds long."
MUSIC_TIMEOUT_S = 120.0  # one synchronous Lyria call
QUOTES = "\"'`“”‘’‚„‟‹›«»「」『』"
_NO_QUOTES = str.maketrans("", "", QUOTES)
_QUOTE = re.compile(f"[{re.escape(QUOTES)}]")


def has_quotes(text: str) -> bool:
    """True if `text` has any quote character (straight, curly, guillemets, CJK brackets)."""
    return _QUOTE.search(text or "") is not None


def strip_quotes(text: str) -> str:
    """`text` without quote characters, spaces collapsed ("a barista's" -> "a baristas")."""
    return " ".join((text or "").translate(_NO_QUOTES).split())


def plain_english(text: str) -> bool:
    """A prompt the media models may get: not empty, English only, no quote characters."""
    return bool((text or "").strip()) and imagegen.english_only(text) and not has_quotes(text)


def hero_seconds(seconds: float) -> int:
    """The clip length requested for a scene of `seconds`: rounded up, clamped to 4-6 s (what
    the month's hero budget is charged)."""
    return min(HERO_MAX_S, max(HERO_MIN_S, math.ceil(seconds)))


def start_hero(
    client,
    model: str,
    still_png: bytes,
    prompt_en: str,
    seconds: int,
    *,
    resolution: str = HERO_RESOLUTION,
) -> str:
    """Start one background Omni clip and return its interaction id: `still_png` (a PNG) as the
    first frame's reference (`image_to_video`; empty = `text_to_video`), `prompt_en` +
    HERO_SUFFIX, 9:16, `resolution`, `hero_seconds(seconds)` long, inline delivery. Raises
    ValueError, before any call, when the prompt is empty, not English or has quote characters;
    the SDK's errors otherwise."""
    if not plain_english(prompt_en):
        raise ValueError("a hero prompt must be English without quote characters")
    content: list[dict] = []
    if still_png:
        data = base64.b64encode(still_png).decode("ascii")
        content.append({"type": "image", "data": data, "mime_type": "image/png"})
    content.append({"type": "text", "text": prompt_en.strip() + HERO_SUFFIX})
    interaction = client.interactions.create(
        model=model,
        input=content,
        response_format=[
            {
                "type": "video",
                "delivery": "inline",
                "aspect_ratio": HERO_ASPECT,
                "resolution": resolution,
                "duration": f"{hero_seconds(seconds)}s",
            }
        ],
        generation_config={
            "video_config": {"task": "image_to_video" if still_png else "text_to_video"}
        },
        background=True,
    )
    iid = getattr(interaction, "id", None)
    if not iid:
        raise RuntimeError("the video service returned no interaction id")
    return str(iid)


def _status(interaction) -> str:
    """The interaction's status in lower case ("" when there is none)."""
    return str(getattr(interaction, "status", "") or "").lower()


def error_status(e: BaseException) -> tuple[int | None, str]:
    """(HTTP status code, status text) of an SDK error, (None, "") for anything else. Covers
    both families: `google.genai.errors.APIError` (`.code`, `.status`) and the Interactions API's
    compat errors (`.status_code`, the JSON body's `error.status`, e.g. PermissionDeniedError)."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    status = getattr(e, "status", None)
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        code = code or err.get("code")
        status = status or err.get("status")
    return (code if isinstance(code, int) else None), str(status or "")


def permission_denied(e: BaseException) -> bool:
    """True if `e` is the service refusing these credentials (HTTP 403 / PERMISSION_DENIED)."""
    code, status = error_status(e)
    denied = type(e).__name__ == "PermissionDeniedError"
    return denied or code == 403 or status == "PERMISSION_DENIED"


def cancel_hero(client, interaction_id: str) -> None:
    """Cancel a hero clip's interaction (a job cancel, the timeout, or one a restart left
    running). Best effort: never raises, logs the error type only."""
    try:
        client.interactions.cancel(interaction_id)
    except Exception as e:  # best effort: it may have finished already
        logger.warning("hero clip cancel failed: %s", type(e).__name__)


def _inline(content) -> bytes | None:
    """The bytes of an SDK content block delivered inline: `data` is base64 text, standard or
    URL-safe, padded or not (the JSON mapping of protobuf bytes accepts both). Raises
    binascii.Error / ValueError when it is not base64."""
    data = getattr(content, "data", None)
    if not data:
        return None
    if isinstance(data, bytes | bytearray):
        return bytes(data)
    text = str(data).strip()
    text += "=" * (-len(text) % 4)
    if "-" in text or "_" in text:
        return base64.urlsafe_b64decode(text.encode("ascii"))
    return base64.b64decode(text, validate=True)


def wait_hero(
    client,
    interaction_id: str,
    *,
    cancelled,
    poll_s: float = POLL_S,
    timeout_s: float = TIMEOUT_S,
    sleep=time.sleep,
) -> tuple[bytes | None, str]:
    """Poll the clip until it ends: (MP4 bytes, "") or (None, reason). Reasons: "blocked" (the
    interaction failed, or completed without a video: a refusal), "timeout" (still running after
    `timeout_s`), "cancelled" (`cancelled` was set, or the service reports it cancelled) and
    "error" (any other ending, a failed poll or an unreadable video). A set `cancelled`, the
    timeout and a failed poll cancel the interaction (best effort). `sleep(seconds)` waits
    between polls (pass `cancelled.wait` to wake up at once on a cancel)."""
    deadline = time.monotonic() + timeout_s
    while True:
        if cancelled.is_set():
            cancel_hero(client, interaction_id)
            return None, "cancelled"
        try:
            interaction = client.interactions.get(interaction_id)
        except Exception as e:
            logger.warning("hero clip poll failed: %s", type(e).__name__)
            cancel_hero(client, interaction_id)
            return None, "error"
        status = _status(interaction)
        if status == "completed":
            try:
                data = _inline(getattr(interaction, "output_video", None))
            except Exception as e:
                logger.warning("hero clip unreadable: %s", type(e).__name__)
                return None, "error"
            return (data, "") if data else (None, "blocked")
        if status == "failed":
            return None, "blocked"
        if status == "cancelled":
            return None, "cancelled"
        if status not in PENDING:
            logger.warning("hero clip ended with status %s", status[:40])
            return None, "error"
        if time.monotonic() >= deadline:
            cancel_hero(client, interaction_id)
            return None, "timeout"
        sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


def music_clip(client, model: str, prompt_en: str, seconds: int) -> bytes | None:
    """One instrumental Lyria clip for `prompt_en` (MUSIC_SUFFIX appended: instrumental, no
    vocals, about `seconds` long; `lyria-3-clip-preview` makes about 30 s whatever is asked, so
    the caller trims it). Returns the audio bytes as delivered (MP3 per the docs) or None on any
    failure: a prompt that is not plain English, an API error, a status other than completed or
    no audio in the answer.

    UNVERIFIED on this project: `client.interactions.create(model=..., input=<text>)` with the
    audio in `interaction.output_audio.data` (base64) follows the Gemini API docs and the
    installed SDK; run `scripts/salid_media_probe.py --music` before setting
    SALID_VIDEO_MUSIC_MODEL."""
    if not plain_english(prompt_en):
        logger.warning("music prompt refused: not plain English")
        return None
    try:
        interaction = client.interactions.create(
            model=model,
            input=prompt_en.strip() + MUSIC_SUFFIX.format(seconds=int(seconds)),
            timeout=MUSIC_TIMEOUT_S,
        )
        status = _status(interaction)
        if status != "completed":
            logger.warning("music clip ended with status %s", status[:40])
            return None
        data = _inline(getattr(interaction, "output_audio", None))
    except Exception as e:  # unverified API: never fail the ad
        logger.warning("music clip failed: %s", type(e).__name__)
        return None
    if not data:
        logger.warning("music clip had no audio")
    return data or None
