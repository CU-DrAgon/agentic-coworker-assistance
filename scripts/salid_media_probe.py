"""Probe the media models once each on Vertex AI (live, billed: run only with human approval).

    uv run python scripts/salid_media_probe.py [--image] [--video] [--music]
        [--video-model ID] [--music-model ID]

--image  one 1K 9:16 Nano Banana image (SALID_IMAGE_MODEL), about $0.03.
--video  one 4 s 360p Gemini Omni image-to-video clip, inline delivery (--video-model, else
         SALID_VIDEO_HERO_MODEL, else gemini-omni-1.1-flash-preview), well under $1.
--music  one Lyria clip (--music-model, else SALID_VIDEO_MUSIC_MODEL, else
         lyria-3-clip-preview), about $0.04; this call shape is unverified, the probe checks it.
Paid creates use `llm.make_media_client` (no SDK retry on 5xx), as the bot does.

Needs ADC for SALID_GCP_PROJECT (`gcloud auth application-default login`). Refuses to run when
GOOGLE_APPLICATION_CREDENTIALS points to a key of another project. Files go to
/tmp/salid-media-probe/ only; nothing is stored anywhere else. Prints latency, status, usage and
the file written for each call; never credentials, and of an error only its type, HTTP status code
and status text (messages may carry account details).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from salid import imagegen, videogen
from salid.config import Settings
from salid.llm import make_client, make_media_client
from salid.media_samples import cup_png

OUT_DIR = Path("/tmp/salid-media-probe")
OMNI_DEFAULT = "gemini-omni-1.1-flash-preview"
LYRIA_DEFAULT = "lyria-3-clip-preview"
POLL_S = 5.0
IMAGE_PROMPT = "A glass of iced latte on a wooden cafe table, soft morning light, vertical framing."
VIDEO_PROMPT = "A white ceramic cup on a wooden table, steam rising slowly, gentle camera push-in."
MUSIC_PROMPT = "Warm, upbeat background music for a short video ad of a small coffee shop."
LOGIN = "gcloud auth application-default login"
KEY_MISMATCH = (
    "GOOGLE_APPLICATION_CREDENTIALS points to a key whose project differs from SALID_GCP_PROJECT. "
    f"Not running. Unset it and run: {LOGIN}"
)
KEY_UNREADABLE = (
    "GOOGLE_APPLICATION_CREDENTIALS is set but its key file cannot be checked. "
    f"Not running. Unset it and run: {LOGIN}"
)
AUDIO_EXT = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/l16": ".pcm"}


def credential_problem(key_path: str | None, project: str) -> str | None:
    """Why the probe must not run with the key at `key_path` (GOOGLE_APPLICATION_CREDENTIALS),
    or None. Only the key's `project_id` is looked at; the message never carries the key's
    contents, its project id or its path. No path, or a file without `project_id` (an
    authorized-user file from `gcloud auth application-default login`), is fine."""
    if not key_path:
        return None
    try:
        with open(key_path, encoding="utf-8") as fh:
            key_project = json.load(fh).get("project_id")
    except Exception:
        return KEY_UNREADABLE
    if key_project is not None and key_project != project:
        return KEY_MISMATCH
    return None


class _Recorded:
    """Calls `names` on `target`, keeping the last result and the last error for the report."""

    def __init__(self, target, names: tuple[str, ...]):
        self._target, self._names = target, names
        self.last = None
        self.error: BaseException | None = None

    def __getattr__(self, name):
        fn = getattr(self._target, name)
        if name not in self._names:
            return fn

        def call(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                self.error = e
                raise
            self.last = out
            return out

        return call


class RecordingClient:
    """The real client with `models.generate_content` and `interactions.create/get/cancel`
    recorded (the probe prints the last status, usage and error type)."""

    def __init__(self, client):
        self.models = _Recorded(client.models, ("generate_content",))
        self.interactions = _Recorded(client.interactions, ("create", "get", "cancel"))


def _error(e: BaseException | None) -> str:
    """An error as type, HTTP status code and status text only (messages may carry account
    details); both SDK error families (`videogen.error_status`)."""
    if e is None:
        return "none"
    code, status = videogen.error_status(e)
    return f"{type(e).__name__} status_code={code} status={status or None}"


def _usage(obj) -> str:
    if obj is None:
        return "none"
    try:
        return json.dumps(obj.model_dump(exclude_none=True, mode="json"), sort_keys=True)
    except Exception:
        return type(obj).__name__


def probe_image(client: RecordingClient, model: str) -> bool:
    t = time.monotonic()
    imgs, reason = imagegen.generate_images(client, model, IMAGE_PROMPT, aspect="9:16", size="1K")
    ms = int((time.monotonic() - t) * 1000)
    calls = client.models
    usage = getattr(calls.last, "usage_metadata", None)
    print(f"image model={model} ms={ms} status={'ok' if imgs else reason} usage={_usage(usage)}")
    if not imgs:
        print(f"  error={_error(calls.error)}")
        return False
    path = OUT_DIR / "image.png"
    imgs[0].save(path)
    print(f"  {path} {imgs[0].width}x{imgs[0].height}")
    return True


def probe_video(main: RecordingClient, media: RecordingClient, model: str) -> bool:
    """The create on `media` (as the bot does), polls and cancel on `main`."""
    calls = main.interactions
    t = time.monotonic()
    try:
        iid = videogen.start_hero(media, model, cup_png(), VIDEO_PROMPT, 4, resolution="360p")
    except Exception as e:
        ms = int((time.monotonic() - t) * 1000)
        print(f"video model={model} ms={ms} status=not-started usage=none error={_error(e)}")
        return False
    data, reason = videogen.wait_hero(
        main, iid, cancelled=threading.Event(), poll_s=POLL_S, timeout_s=videogen.TIMEOUT_S
    )
    ms = int((time.monotonic() - t) * 1000)
    last = calls.last
    status = getattr(last, "status", None)
    usage = getattr(last, "usage", None)
    print(
        f"video model={model} interaction={iid} ms={ms} status={status} "
        f"result={'ok' if data else reason} usage={_usage(usage)}"
    )
    if data is None:
        print(f"  error={_error(calls.error)}")
        return False
    path = OUT_DIR / "omni.mp4"
    path.write_bytes(data)
    print(f"  {path} {len(data):,} bytes")
    return True


def probe_music(media: RecordingClient, model: str) -> bool:
    calls = media.interactions
    t = time.monotonic()
    data = videogen.music_clip(media, model, MUSIC_PROMPT, 30)
    ms = int((time.monotonic() - t) * 1000)
    last = calls.last
    usage = getattr(last, "usage", None)
    print(
        f"music model={model} ms={ms} status={getattr(last, 'status', None)} "
        f"result={'ok' if data else 'none'} usage={_usage(usage)}"
    )
    if data is None:
        print(f"  error={_error(calls.error)}")
        return False
    mime = str(getattr(getattr(last, "output_audio", None), "mime_type", "") or "")
    path = OUT_DIR / f"music{AUDIO_EXT.get(mime, '.audio')}"
    path.write_bytes(data)
    print(f"  {path} {len(data):,} bytes mime={mime or 'unknown'}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", action="store_true", help="one 1K image (billed)")
    ap.add_argument("--video", action="store_true", help="one 4 s 360p Omni clip (billed)")
    ap.add_argument("--music", action="store_true", help="one Lyria clip (billed)")
    ap.add_argument("--video-model", default="", help=f"default: setting, else {OMNI_DEFAULT}")
    ap.add_argument("--music-model", default="", help=f"default: setting, else {LYRIA_DEFAULT}")
    args = ap.parse_args(argv)
    if not (args.image or args.video or args.music):
        ap.print_usage()
        print("nothing to probe: pass --image, --video and/or --music")
        return 2
    s = Settings()
    problem = credential_problem(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), s.gcp_project)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main, media = RecordingClient(make_client(s)), RecordingClient(make_media_client(s))
    print(f"project={s.gcp_project} location={s.gemini_location} out={OUT_DIR}")
    ok = True
    if args.image:
        ok = probe_image(main, s.image_model) and ok
    if args.video:
        model = args.video_model or s.video_hero_model or OMNI_DEFAULT
        ok = probe_video(main, media, model) and ok
    if args.music:
        ok = probe_music(media, args.music_model or s.video_music_model or LYRIA_DEFAULT) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
