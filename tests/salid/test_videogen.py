import base64
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from threading import Event

import httpx
import pytest
from conftest import FixedClockSettings
from google import genai
from google.genai import interactions as gi
from google.genai import types
from PIL import Image
from promo_helpers import png_bytes, probe, video_frame
from test_videoad import (
    S_HERO,
    S_MUSIC,
    SILENCE,
    MemDocs,
    fake_pipeline,  # noqa: F401 (registers the fixture in this module, see `pipeline`)
    req,
    sb,
    sources,
)

from salid import llm, videoad, videogen

REAL_HERO = videoad._hero_clip  # fake_pipeline replaces the hook; these tests put it back
REAL_MUSIC = videoad._music_track
REAL_START = videogen.start_hero
S_OFF = FixedClockSettings(video_hero_model="", video_music_model="")
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg"
)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class FakeInteractions:
    """A fake google-genai client for the Interactions API (it is also its own `interactions`).
    `create` records its keyword arguments: a background call returns an in-progress interaction
    with an id, a synchronous one (music) returns the last scripted status at once; `get`
    returns the scripted statuses in order (the last one repeats); `cancel` records the id. The
    answers are real SDK `Interaction` objects with the media inline as base64 in a model_output
    step, the shape the SDK parses into `output_video` / `output_audio`."""

    def __init__(self, statuses=("completed",), video=None, *, audio=None, on_get=None):
        self.statuses = list(statuses)
        self.video, self.audio, self.on_get = video, audio, on_get
        self.created, self.cancelled, self.got = [], [], []
        self.interactions = self

    def _interaction(self, status: str, model: str | None = None) -> gi.Interaction:
        content = []
        done = status.lower() == "completed"
        if done and self.video is not None:
            content.append({"type": "video", "data": b64(self.video), "mime_type": "video/mp4"})
        if done and self.audio is not None:
            content.append({"type": "audio", "data": b64(self.audio), "mime_type": "audio/mpeg"})
        steps = [{"type": "model_output", "content": content}] if content else []
        return gi.Interaction.model_validate(
            {"id": "i", "status": status, "model": model, "steps": steps}
        )

    def create(self, **kw):
        self.created.append(kw)
        if not kw.get("background"):
            return self._interaction(self.statuses[-1], kw.get("model"))
        return gi.Interaction(id="i", status="in_progress")

    def get(self, id, **kw):
        self.got.append(id)
        if self.on_get is not None:
            self.on_get(len(self.got))
        return self._interaction(self.statuses[min(len(self.got), len(self.statuses)) - 1])

    def cancel(self, id, **kw):
        self.cancelled.append(id)
        return gi.Interaction(id=id, status="cancelled")


def no_sleep(seconds):
    return None


# ------------------------------------------------------------------ start / wait
def test_hero_request_shape_and_english_guard():
    c = FakeInteractions(statuses=["completed"], video=b"MP4")
    still = png_bytes((768, 1376))
    iid = videogen.start_hero(c, "gemini-omni-1.1-flash-preview", still, "a latte, slow push-in", 8)
    kw = c.created[0]
    assert iid == "i" and kw["model"] == "gemini-omni-1.1-flash-preview"
    assert kw["background"] is True and kw["response_format"][0]["delivery"] == "inline"
    assert kw["response_format"][0]["duration"] == "6s" and kw["input"][-1]["text"].endswith(
        videogen.HERO_SUFFIX
    )
    assert kw["response_format"] == [
        {
            "type": "video",
            "delivery": "inline",
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "duration": "6s",
        }
    ]
    assert kw["generation_config"] == {"video_config": {"task": "image_to_video"}}
    image, text = kw["input"]
    assert image == {"type": "image", "data": b64(still), "mime_type": "image/png"}
    assert text == {"type": "text", "text": "a latte, slow push-in" + videogen.HERO_SUFFIX}
    with pytest.raises(ValueError):
        videogen.start_hero(c, "m", b"x", "ลาเต้", 4)
    with pytest.raises(ValueError):
        videogen.start_hero(c, "m", b"x", 'a latte with "SALE" on it', 4)
    with pytest.raises(ValueError):
        videogen.start_hero(c, "m", b"x", "  ", 4)
    assert len(c.created) == 1  # refused prompts never reach the service


@pytest.mark.parametrize("asked, sent", [(1, "4s"), (2.2, "4s"), (4.5, "5s"), (6, "6s"), (9, "6s")])
def test_hero_seconds_are_clamped_to_4_to_6(asked, sent):
    c = FakeInteractions()
    videogen.start_hero(c, "m", b"png", "a latte", asked)
    assert c.created[0]["response_format"][0]["duration"] == sent
    assert videogen.hero_seconds(asked) == int(sent[:-1])


def test_without_a_still_the_request_is_text_to_video():
    c = FakeInteractions()
    videogen.start_hero(c, "m", b"", "a latte", 4, resolution="360p")
    kw = c.created[0]
    assert kw["generation_config"] == {"video_config": {"task": "text_to_video"}}
    assert [block["type"] for block in kw["input"]] == ["text"]
    assert kw["response_format"][0]["resolution"] == "360p"


def test_strip_quotes_keeps_the_words():
    assert videogen.strip_quotes("a barista’s “fresh” 'latte'") == "a baristas fresh latte"
    assert not videogen.has_quotes(videogen.strip_quotes('say "hi" `x` «no»'))


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
    pre = FakeInteractions(["in_progress"], None)
    assert videogen.wait_hero(pre, "i", cancelled=ev, sleep=lambda s: None)[1] == "cancelled"
    assert pre.cancelled == ["i"]
    assert (
        videogen.wait_hero(
            FakeInteractions(["failed"], None), "i", cancelled=Event(), sleep=lambda s: None
        )[1]
        == "blocked"
    )


def test_wait_polls_every_poll_s():
    slept = []
    c = FakeInteractions(["queued", "in_progress", "completed"], b"MP4")
    out = videogen.wait_hero(c, "i", cancelled=Event(), poll_s=10.0, sleep=slept.append)
    assert out == (b"MP4", "") and slept == [10.0, 10.0] and c.got == ["i"] * 3


def test_a_cancel_during_polling_cancels_the_interaction():
    ev = Event()
    c = FakeInteractions(["in_progress"] * 10, None, on_get=lambda n: n == 2 and ev.set())
    assert videogen.wait_hero(c, "i", cancelled=ev, sleep=no_sleep) == (None, "cancelled")
    assert c.got == ["i", "i"] and c.cancelled == ["i"]


@pytest.mark.parametrize(
    "status, reason",
    [("completed", "blocked"), ("budget_exceeded", "error"), ("incomplete", "error"),
     ("cancelled", "cancelled"), ("something_new", "error")],
)  # fmt: skip
def test_other_endings(status, reason):
    c = FakeInteractions([status], None)  # completed without a video = refused
    assert videogen.wait_hero(c, "i", cancelled=Event(), sleep=no_sleep) == (None, reason)
    assert c.cancelled == []


def test_statuses_are_compared_in_any_case():
    c = FakeInteractions(["IN_PROGRESS", "Completed"], b"MP4")
    assert videogen.wait_hero(c, "i", cancelled=Event(), sleep=no_sleep) == (b"MP4", "")
    assert videogen.wait_hero(FakeInteractions(["FAILED"]), "i", cancelled=Event(),
                              sleep=no_sleep) == (None, "blocked")  # fmt: skip
    assert videogen.music_clip(FakeInteractions(["COMPLETED"], audio=b"MP3"), "m", "a", 9) == b"MP3"


def test_a_failing_poll_is_an_error_and_cancels(caplog):
    class Broken(FakeInteractions):
        def get(self, id, **kw):
            raise httpx.ConnectError("down")

    c = Broken()
    assert videogen.wait_hero(c, "i", cancelled=Event(), sleep=no_sleep) == (None, "error")
    assert c.cancelled == ["i"]


def test_a_failing_cancel_is_swallowed():
    class NoCancel(FakeInteractions):
        def cancel(self, id, **kw):
            raise RuntimeError("cannot cancel")

    c = NoCancel(["in_progress"])
    assert videogen.wait_hero(c, "i", cancelled=Event(), timeout_s=0.0, sleep=no_sleep) == (
        None,
        "timeout",
    )


@pytest.mark.parametrize(
    "encode, reason",
    [(base64.b64encode, ""), (base64.urlsafe_b64encode, ""),
     (lambda d: base64.urlsafe_b64encode(d).rstrip(b"="), ""), (lambda d: b"%%%", "error")],
    ids=["standard", "url-safe", "url-safe unpadded", "garbage"],
)  # fmt: skip
def test_inline_video_in_any_base64_flavour(encode, reason):
    data = b"\xfb\xff\xfe\x00video"  # encodes with + and / (- and _ when url-safe)

    class Answer(FakeInteractions):
        def get(self, id, **kw):
            return gi.Interaction.model_validate({"id": id, "status": "completed", "steps": [
                {"type": "model_output", "content": [
                    {"type": "video", "data": encode(data).decode(), "mime_type": "video/mp4"}
                ]}
            ]})  # fmt: skip

    out = videogen.wait_hero(Answer(), "i", cancelled=Event(), sleep=no_sleep)
    assert out == ((data, "") if not reason else (None, reason))


# ------------------------------------------------------------------ music
def test_music_clip_is_instrumental_and_returns_the_audio():
    c = FakeInteractions(["completed"], audio=b"MP3")
    assert videogen.music_clip(c, "lyria-3-clip-preview", "Warm upbeat shop music.", 20) == b"MP3"
    kw = c.created[0]
    assert kw["model"] == "lyria-3-clip-preview" and not kw.get("background")
    assert kw["input"].startswith("Warm upbeat shop music.") and "Instrumental only" in kw["input"]
    assert "no vocals" in kw["input"] and "20 seconds" in kw["input"]


def test_music_clip_is_none_on_any_failure():
    assert videogen.music_clip(FakeInteractions(["failed"], audio=b"x"), "m", "music", 20) is None
    assert videogen.music_clip(FakeInteractions(["completed"]), "m", "music", 20) is None
    assert videogen.music_clip(None, "m", "music", 20) is None

    class Broken(FakeInteractions):
        def create(self, **kw):
            raise RuntimeError("403 PERMISSION_DENIED")

    assert videogen.music_clip(Broken(), "m", "music", 20) is None
    thai = FakeInteractions(["completed"], audio=b"x")
    assert videogen.music_clip(thai, "m", "เพลงสนุก", 20) is None and thai.created == []
    quoted = FakeInteractions(["completed"], audio=b"x")
    assert videogen.music_clip(quoted, "m", 'a "jingle"', 20) is None and quoted.created == []


def test_logs_never_carry_prompts(caplog):
    caplog.set_level("INFO", logger="salid.videogen")
    secret = "a secret recipe latte"

    class Broken(FakeInteractions):
        def create(self, **kw):
            raise RuntimeError(kw.get("input") if isinstance(kw.get("input"), str) else "x")

    videogen.music_clip(Broken(), "m", secret, 20)
    videogen.wait_hero(FakeInteractions(["failed"]), "i", cancelled=Event(), sleep=no_sleep)
    assert secret not in caplog.text


# ------------------------------------------------------------------ the real SDK, offline
def sdk_client(monkeypatch, tmp_path, answer, options=types.HttpOptions):
    """A real google-genai client on the Gemini API whose HTTP goes to `answer(request)` (no
    network, no credentials: an API key and no Vertex/ADC lookup). `options(httpx_client=...)`
    makes its HttpOptions (`llm.media_http_options` for the paid-media client)."""
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "none.json"))
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        return answer(request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = genai.Client(
        vertexai=False, api_key="offline-test", http_options=options(httpx_client=http)
    )
    return client, seen


def test_requests_and_answers_match_the_installed_sdk(monkeypatch, tmp_path):
    states = iter(["in_progress", "completed"])

    def answer(request):
        path = request.url.path
        if request.method == "POST" and path.endswith("/interactions"):
            return httpx.Response(200, json={"id": "abc", "status": "in_progress"})
        if path.endswith("/cancel"):
            return httpx.Response(200, json={"id": "abc", "status": "cancelled"})
        status = next(states)
        steps = [
            {"type": "model_output",
             "content": [{"type": "video", "data": b64(b"MP4"), "mime_type": "video/mp4"}]}
        ] if status == "completed" else []  # fmt: skip
        return httpx.Response(200, json={"id": "abc", "status": status, "steps": steps})

    client, seen = sdk_client(monkeypatch, tmp_path, answer)
    iid = videogen.start_hero(client, "omni", b"PNG", "a latte", 5)
    assert videogen.wait_hero(client, iid, cancelled=Event(), sleep=no_sleep) == (b"MP4", "")
    (m1, p1, body), (m2, p2, _), (m3, p3, _) = seen
    assert (m1, m2, m3) == ("POST", "GET", "GET") and p2 == p3 == f"{p1}/abc"
    assert body == {
        "model": "omni",
        "background": True,
        "generation_config": {"video_config": {"task": "image_to_video"}},
        "input": [
            {
                "type": "user_input",
                "content": [
                    {"type": "image", "data": b64(b"PNG"), "mime_type": "image/png"},
                    {"type": "text", "text": "a latte" + videogen.HERO_SUFFIX},
                ],
            }
        ],
        "response_format": [
            {
                "type": "video",
                "delivery": "inline",
                "aspect_ratio": "9:16",
                "resolution": "720p",
                "duration": "5s",
            }
        ],
    }
    ev = Event()
    ev.set()
    assert videogen.wait_hero(client, iid, cancelled=ev, sleep=no_sleep) == (None, "cancelled")
    assert seen[-1][:2] == ("POST", f"{p1}/abc/cancel")


@pytest.mark.parametrize("legacy", [True, False], ids=["outputs", "steps"])
def test_music_answer_matches_the_installed_sdk(monkeypatch, tmp_path, legacy):
    audio = [{"type": "audio", "data": b64(b"MP3"), "mime_type": "audio/mpeg"}]
    shape = (
        {"outputs": audio} if legacy else {"steps": [{"type": "model_output", "content": audio}]}
    )

    def answer(request):
        return httpx.Response(
            200, json={"id": "m", "status": "completed", "model": "lyria-3-clip-preview", **shape}
        )

    client, seen = sdk_client(monkeypatch, tmp_path, answer)
    assert videogen.music_clip(client, "lyria-3-clip-preview", "Warm shop music.", 20) == b"MP3"
    [(method, _, body)] = seen
    assert method == "POST" and body["model"] == "lyria-3-clip-preview"
    assert "background" not in body and "Instrumental only" in str(body["input"])


UNAVAILABLE = {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}
DENIED = {"error": {"code": 403, "message": "detail-of-the-account", "status": "PERMISSION_DENIED"}}


def test_paid_creates_are_never_retried_on_a_5xx(monkeypatch, tmp_path):
    """A 503 after Omni / Lyria accepted the job would start (and bill) a second one: the media
    client sends exactly one POST per paid call."""

    def answer(request):
        return httpx.Response(503, json=UNAVAILABLE)

    client, seen = sdk_client(monkeypatch, tmp_path, answer, llm.media_http_options)
    with pytest.raises(Exception) as err:
        videogen.start_hero(client, "omni", b"PNG", "a latte", 5)
    assert videogen.error_status(err.value) == (503, "UNAVAILABLE")
    assert [m for m, _, _ in seen] == ["POST"]
    seen.clear()
    assert videogen.music_clip(client, "lyria-3-clip-preview", "Warm shop music.", 20) is None
    assert [m for m, _, _ in seen] == ["POST"]


def test_the_media_client_has_no_retries_but_429():
    s = FixedClockSettings(gcp_project="p", gemini_location="global")
    options = llm.make_media_client(s)._api_client._http_options
    assert options.timeout == llm.REQUEST_TIMEOUT_MS
    assert options.retry_options.attempts == 1 and options.retry_options.http_status_codes == [429]
    assert llm.make_client(s)._api_client._http_options.retry_options is None  # unchanged


def test_a_403_is_recognised_as_permission_denied(monkeypatch, tmp_path):
    def answer(request):
        return httpx.Response(403, json=DENIED)

    client, _ = sdk_client(monkeypatch, tmp_path, answer, llm.media_http_options)
    with pytest.raises(Exception) as err:
        videogen.start_hero(client, "omni", b"PNG", "a latte", 4, resolution="360p")
    e = err.value
    assert type(e).__name__ == "PermissionDeniedError" and getattr(e, "code", None) is None
    assert videogen.permission_denied(e) and videogen.error_status(e) == (403, "PERMISSION_DENIED")
    printed = load_probe()._error(e)
    assert "PermissionDeniedError" in printed and "status_code=403" in printed
    assert "PERMISSION_DENIED" in printed and "detail-of-the-account" not in printed


def test_permission_denied_in_both_error_families():
    from google.genai import errors

    assert videogen.permission_denied(errors.ClientError(403, DENIED))
    assert videogen.error_status(errors.ClientError(403, DENIED)) == (403, "PERMISSION_DENIED")
    assert not videogen.permission_denied(errors.ServerError(503, UNAVAILABLE))
    assert not videogen.permission_denied(RuntimeError("403 PERMISSION_DENIED"))  # text only
    assert videogen.error_status(RuntimeError("x")) == (None, "")


# ------------------------------------------------------------------ the runner
def tiny_mp4(tmp_path, size="360x640", seconds=2.0) -> bytes:
    """A real H.264 MP4 of a test pattern (with a sine tone), like a model's inline clip."""
    out = tmp_path / "omni-src.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    data = out.read_bytes()
    out.unlink()
    return data


def frame_rate(path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    return out.stdout.decode().strip()


NO_CLIENT = object()


@pytest.fixture
def pipeline(request):
    """test_videoad's `fake_pipeline` (the job's paid and slow parts replaced by recorders)."""
    return request.getfixturevalue("fake_pipeline")


def run_ad(
    tmp_path, *, settings=S_HERO, handle=None, usage=None, client=NO_CLIENT, media_client=None, **kw
):
    return videoad.run_video_ad(
        req(**kw),
        client=client,
        media_client=media_client,
        settings=settings,
        say=lambda t: SILENCE,
        workdir=tmp_path,
        budget_mib=19,
        handle=handle or videoad.VideoHandle(),
        usage=usage or videoad.MediaUsage(MemDocs()),
        month="2026-10",
    )


@pytest.fixture
def real_hero(monkeypatch, pipeline):
    """pipeline with the real `_hero_clip`, and `videogen.start_hero` recorded."""
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    starts = []

    def start(client, model, still_png, prompt_en, seconds, **kw):
        starts.append(dict(model=model, still=still_png, prompt=prompt_en, seconds=seconds))
        return f"i{len(starts)}"

    monkeypatch.setattr(videoad.videogen, "start_hero", start)
    pipeline.starts = starts
    return pipeline


def test_runner_uses_a_hero_clip_and_falls_back_to_motion(tmp_path, real_hero, monkeypatch):
    waits = []

    def wait(client, iid, **kw):
        waits.append(iid)
        return None, "blocked"

    monkeypatch.setattr(videoad.videogen, "wait_hero", wait)
    usage = videoad.MediaUsage(d := MemDocs())
    run_ad(tmp_path, hero_shots=1, usage=usage)
    [start] = real_hero.starts
    assert start["model"] == S_HERO.video_hero_model and start["seconds"] == 5
    assert start["still"] == (tmp_path / "still0.png").read_bytes() and waits == ["i1"]
    clips = real_hero.assembled[0]["clips"]
    assert clips[0] == tmp_path / "clip0.mp4" and clips[0].read_bytes() == b"clip"  # motion
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 5  # no refund


def test_the_pending_hero_id_is_reported_until_the_clip_ends(tmp_path, real_hero, monkeypatch):
    """A restart while Omni works would leave the paid job running: the job record gets the
    interaction id (via the progress callback) while it is pending, and loses it afterwards."""
    seen, reports = [], []
    h = videoad.VideoHandle(progress=lambda step, _d: reports.append((step, h.hero_id)))

    def wait(client, iid, **kw):
        seen.append(h.hero_id)
        return None, "blocked"

    monkeypatch.setattr(videoad.videogen, "wait_hero", wait)
    run_ad(tmp_path, hero_shots=1, handle=h)
    assert seen == ["i1"] and h.hero_id == ""
    at = reports.index(("hero", "i1"))
    assert reports[at + 1] == ("hero", "") and reports[-1] == ("render", "")


def test_the_hero_id_is_cleared_when_the_job_is_cancelled(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    h = videoad.VideoHandle()
    c = FakeInteractions(["in_progress"] * 10, None, on_get=lambda n: h.cancelled.set())
    with pytest.raises(videoad.Cancelled):
        run_ad(tmp_path, client=c, handle=h, hero_shots=1)
    assert h.hero_id == "" and c.cancelled == ["i"]


def test_cancel_hero_is_best_effort():
    c = FakeInteractions()
    videogen.cancel_hero(c, "abc")
    assert c.cancelled == ["abc"]

    class Broken(FakeInteractions):
        def cancel(self, id, **kw):
            raise RuntimeError("detail-of-the-account")

    videogen.cancel_hero(Broken(), "abc")  # never raises


@needs_ffmpeg
def test_a_returned_clip_is_transcoded_into_the_hero_scene(tmp_path, real_hero, monkeypatch):
    mp4 = tiny_mp4(tmp_path)
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (mp4, ""))
    run_ad(tmp_path, hero_shots=1)
    clips = real_hero.assembled[0]["clips"]
    assert clips[0] == tmp_path / "hero0.mp4" and clips[1] == tmp_path / "clip1.mp4"
    info = probe(clips[0])
    assert info["size"] == (1080, 1920) and info["codecs"] == ["h264"]  # no audio kept
    assert info["duration"] == pytest.approx(5.0, abs=0.1)  # held to the whole scene
    assert frame_rate(clips[0]) == "30/1"
    assert not (tmp_path / "hero0-omni.mp4").exists()  # the raw download is removed


@needs_ffmpeg
def test_a_wide_clip_is_cover_cropped_never_stretched(tmp_path):
    src = tmp_path / "wide.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=red:s=214x360:d=1",
         "-f", "lavfi", "-i", "color=c=green:s=212x360:d=1",
         "-f", "lavfi", "-i", "color=c=blue:s=214x360:d=1",
         "-filter_complex", "[0][1][2]hstack=inputs=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(src)],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    out = tmp_path / "hero.mp4"
    videoad.conform_clip(src, 2.5, out)
    assert probe(out)["size"] == (1080, 1920)
    assert probe(out)["duration"] == pytest.approx(2.5, abs=0.1)
    frame = video_frame(out, 2.0, tmp_path / "f.png")  # inside the held part
    for x in (5, 540, 1074):  # a stretch would show red and blue at the sides
        r, g, b = frame.getpixel((x, 960))
        assert g > 100 and r < 60 and b < 60


def test_a_bad_clip_falls_back_to_motion(tmp_path, real_hero, monkeypatch):
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (b"not a video", ""))
    run_ad(tmp_path, hero_shots=1)
    clips = real_hero.assembled[0]["clips"]
    assert clips[0] == tmp_path / "clip0.mp4" and not (tmp_path / "hero0.mp4").exists()


def test_a_cancel_during_wait_hero_cancels_the_interaction_and_the_job(
    tmp_path, pipeline, monkeypatch
):
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    h = videoad.VideoHandle()
    c = FakeInteractions(["in_progress"] * 10, None, on_get=lambda n: h.cancelled.set())
    with pytest.raises(videoad.Cancelled):
        run_ad(tmp_path, client=c, handle=h, hero_shots=2)
    assert len(c.created) == 1 and c.cancelled == ["i"]  # the second hero never started
    assert pipeline.assembled == []
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD + 5 * videoad.HERO_USD_PER_S)


def test_a_cancel_reported_by_the_service_alone_only_drops_the_hero(
    tmp_path, real_hero, monkeypatch
):
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (None, "cancelled"))
    res = run_ad(tmp_path, hero_shots=1)
    assert res.path.exists() and real_hero.assembled[0]["clips"][0] == tmp_path / "clip0.mp4"


def test_the_runner_counts_the_seconds_actually_requested(tmp_path, real_hero, monkeypatch):
    def plan(client, model, **kw):
        return sources(sb(8, 3, 5, 4), "hero", "hero")  # ceil would charge 8 + 3

    monkeypatch.setattr(videoad, "plan_storyboard", plan)
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (None, "blocked"))
    usage = videoad.MediaUsage(d := MemDocs())
    h = videoad.VideoHandle()
    run_ad(tmp_path, hero_shots=2, usage=usage, handle=h)
    assert [s["seconds"] for s in real_hero.starts] == [6, 4]  # clamp(ceil(scene), 4, 6)
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 10
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD + 10 * videoad.HERO_USD_PER_S)


def test_hero_prompts_lose_their_quotes_and_thai_never_starts(tmp_path, real_hero, monkeypatch):
    def plan(client, model, **kw):
        board = sources(sb(5, 5, 5, 5), "hero", "hero")
        board.scenes[0].image_prompt_en = "a barista's hands pour “art”"
        board.scenes[1].image_prompt_en = "ลาเต้"
        return board

    monkeypatch.setattr(videoad, "plan_storyboard", plan)
    monkeypatch.setattr(videoad.videogen, "start_hero", REAL_START)
    c = FakeInteractions(["failed"])
    usage = videoad.MediaUsage(d := MemDocs())
    run_ad(tmp_path, client=c, hero_shots=2, usage=usage)
    [created] = c.created  # the Thai prompt was refused before any call
    assert created["input"][-1]["text"] == (
        "a baristas hands pour art. Slow cinematic camera push-in." + videogen.HERO_SUFFIX
    )
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 5  # none for the Thai one


def test_the_hero_prompt_carries_a_motion_cue():
    scene = sb(5, 5, 5, 5, src="hero", prompt="a latte on a table.").scenes[0]
    assert videoad.hero_prompt(scene) == f"a latte on a table. {videoad.HERO_MOTION}"
    assert videoad.HERO_MOTION == "Slow cinematic camera push-in."
    assert videogen.plain_english(videoad.HERO_MOTION + videogen.HERO_SUFFIX)
    assert videoad.hero_prompt(sb(5, 5, 5, 5, prompt="“”").scenes[0]) == ""  # nothing left


def test_a_scene_left_with_the_fallback_card_gets_no_hero_call(tmp_path, real_hero, monkeypatch):
    def stills(client, model, scenes, photos, accent, fonts, *, cancelled=None):
        imgs = [Image.new("RGB", (108, 192), (90, 90, 90)) for _ in scenes]
        imgs[0] = videoad.fallback_card(accent)  # scene 0's picture failed
        return imgs, 2

    monkeypatch.setattr(videoad, "scene_stills", stills)
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (None, "blocked"))
    usage = videoad.MediaUsage(d := MemDocs())
    run_ad(tmp_path, hero_shots=2, usage=usage)
    assert [s["still"] for s in real_hero.starts] == [(tmp_path / "still1.png").read_bytes()]
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 5
    assert videoad.is_fallback_card(videoad.fallback_card("red"))
    assert not videoad.is_fallback_card(Image.new("RGB", (4, 4)))


def test_a_refused_start_skips_the_other_heroes_without_their_seconds(
    tmp_path, pipeline, monkeypatch, caplog
):
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    starts = []

    def refuse(client, model, still_png, prompt_en, seconds, **kw):
        starts.append(seconds)
        raise RuntimeError("403 PERMISSION_DENIED for this project")

    monkeypatch.setattr(videoad.videogen, "start_hero", refuse)
    usage = videoad.MediaUsage(d := MemDocs())
    h = videoad.VideoHandle()
    res = run_ad(tmp_path, hero_shots=2, usage=usage, handle=h)
    assert starts == [5] and res.path.exists()  # one refusal, the second hero never asked
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 5
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD + 5 * videoad.HERO_USD_PER_S)
    clips = pipeline.assembled[0]["clips"]
    assert clips[:2] == [tmp_path / "clip0.mp4", tmp_path / "clip1.mp4"]
    assert "PERMISSION_DENIED" not in caplog.text and "video hero refused" in caplog.text


def test_paid_creates_go_to_the_media_client_and_polls_to_the_main_one(
    tmp_path, pipeline, monkeypatch
):
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    monkeypatch.setattr(videoad, "_music_track", REAL_MUSIC)
    main, media = FakeInteractions(["failed"]), FakeInteractions(["failed"])
    both = FixedClockSettings(video_hero_model="omni-test", video_music_model="music-test")
    run_ad(tmp_path, settings=both, client=main, media_client=media, hero_shots=1, music=True)
    assert [kw["model"] for kw in media.created] == ["omni-test", "music-test"]
    assert main.created == [] and main.got == ["i"] and media.got == []


def test_hero_logs_carry_no_prompt(tmp_path, real_hero, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(videoad.videogen, "wait_hero", lambda *a, **k: (None, "timeout"))
    run_ad(tmp_path, hero_shots=1)
    assert "a latte on a table" not in caplog.text and "video hero clip failed" in caplog.text


def test_hero_clip_is_false_without_a_readable_still():
    scene = sb(5, 5, 5, 5).scenes[0]
    assert REAL_HERO(object(), S_HERO, Path("missing.png"), scene, None, Path("o.mp4")) is False


# ------------------------------------------------------------------ music in the runner
@pytest.fixture
def real_music(monkeypatch, pipeline):
    monkeypatch.setattr(videoad, "_music_track", REAL_MUSIC)
    return pipeline


def tiny_audio(tmp_path, seconds=3.0) -> bytes:
    out = tmp_path / "lyria.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=330:duration={seconds}", "-c:a", "aac", str(out)],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    data = out.read_bytes()
    out.unlink()
    return data


@needs_ffmpeg
def test_music_track_is_a_48k_stereo_wav_as_long_as_the_ad(tmp_path, real_music, monkeypatch):
    calls = []
    audio = tiny_audio(tmp_path, 30.0)

    def clip(client, model, prompt_en, seconds):
        calls.append((model, prompt_en, seconds))
        return audio

    monkeypatch.setattr(videoad.videogen, "music_clip", clip)
    run_ad(tmp_path, settings=S_MUSIC, music=True)
    [(model, prompt, seconds)] = calls
    assert model == S_MUSIC.video_music_model and seconds == 20
    assert videogen.plain_english(prompt)
    music = real_music.assembled[0]["music"]
    assert music == tmp_path / "music.wav"
    info = probe(music)
    assert info["codecs"] == ["pcm_s16le"] and info["duration"] == pytest.approx(20.0, abs=0.05)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
         "-of", "json", str(music)],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip
    [stream] = json.loads(out.stdout)["streams"]
    assert stream["sample_rate"] == "48000" and stream["channels"] == 2
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("music")] == ["music.wav"]


def test_no_music_clip_means_no_music(tmp_path, real_music, monkeypatch):
    monkeypatch.setattr(videoad.videogen, "music_clip", lambda *a: None)
    res = run_ad(tmp_path, settings=S_MUSIC, music=True)
    assert res.path.exists() and real_music.assembled[0]["music"] is None


def test_unreadable_music_means_no_music(tmp_path, real_music, monkeypatch):
    monkeypatch.setattr(videoad.videogen, "music_clip", lambda *a: b"not audio")
    res = run_ad(tmp_path, settings=S_MUSIC, music=True)
    assert res.path.exists() and real_music.assembled[0]["music"] is None


def test_a_cancel_while_the_music_is_made_stops_the_job(tmp_path, real_music, monkeypatch):
    h = videoad.VideoHandle()

    def clip(*a):
        h.cancelled.set()
        return b"audio"

    monkeypatch.setattr(videoad.videogen, "music_clip", clip)
    with pytest.raises(videoad.Cancelled):
        run_ad(tmp_path, settings=S_MUSIC, music=True, handle=h)
    assert real_music.assembled == []


def test_one_music_clip_per_ad_is_taken_before_the_call(tmp_path, pipeline):
    pipeline.music_ok = True
    usage = videoad.MediaUsage(d := MemDocs())
    h = videoad.VideoHandle()
    run_ad(tmp_path, settings=S_MUSIC, music=True, usage=usage, handle=h)
    assert pipeline.music_calls == 1
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 1, "music_clips": 1}
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD + videoad.MUSIC_USD)


def test_music_month_budget_used_up_means_no_music_call(tmp_path, pipeline, caplog):
    caplog.set_level("INFO", logger="salid.videoad")
    pipeline.music_ok = True
    d = MemDocs()
    d.put("salid_media_usage", 1, "2026-10", {"music_clips": S_MUSIC.video_ads_per_month})
    res = run_ad(tmp_path, settings=S_MUSIC, music=True, usage=videoad.MediaUsage(d))
    assert pipeline.music_calls == 0 and res.path.exists()
    assert pipeline.assembled[0]["music"] is None
    assert any("music budget" in r.getMessage() for r in caplog.records)


def test_with_both_models_empty_nothing_is_called(tmp_path, pipeline, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no media add-on call while its model setting is empty")

    for name in ("start_hero", "wait_hero", "music_clip"):
        monkeypatch.setattr(videoad.videogen, name, boom)
    monkeypatch.setattr(videoad, "_hero_clip", REAL_HERO)
    monkeypatch.setattr(videoad, "_music_track", REAL_MUSIC)
    usage = videoad.MediaUsage(d := MemDocs())
    res = run_ad(tmp_path, settings=S_OFF, hero_shots=2, music=True, usage=usage)
    assert res.path.exists() and d.get("salid_media_usage", 1, "2026-10") == {"ads": 1}


# ------------------------------------------------------------------ the probe script
SCRIPT = Path(__file__).parents[2] / "scripts" / "salid_media_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("salid_media_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_key(tmp_path, project: str | None) -> Path:
    key = {"type": "service_account", "private_key": "FAKE-PRIVATE-KEY-DO-NOT-PRINT",
           "client_email": "probe@fake.iam.gserviceaccount.com"}  # fmt: skip
    if project is not None:
        key["project_id"] = project
    path = tmp_path / "key.json"
    path.write_text(json.dumps(key))
    return path


def test_probe_guard_reads_only_the_key_project(tmp_path):
    mod = load_probe()
    other = fake_key(tmp_path, "another-org-project")
    problem = mod.credential_problem(str(other), "meeting-scribe-hackathon")
    assert "gcloud auth application-default login" in problem and "SALID_GCP_PROJECT" in problem
    assert "another-org-project" not in problem and "FAKE-PRIVATE-KEY" not in problem
    same = fake_key(tmp_path, "meeting-scribe-hackathon")
    assert mod.credential_problem(str(same), "meeting-scribe-hackathon") is None
    assert mod.credential_problem(None, "meeting-scribe-hackathon") is None
    assert mod.credential_problem("", "meeting-scribe-hackathon") is None
    user = fake_key(tmp_path, None)  # an authorized-user file has no project_id
    assert mod.credential_problem(str(user), "meeting-scribe-hackathon") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert "application-default login" in mod.credential_problem(str(broken), "p")
    assert "application-default login" in mod.credential_problem(str(tmp_path / "gone.json"), "p")


def test_probe_refuses_another_projects_key_before_any_client(tmp_path, monkeypatch, capsys):
    mod = load_probe()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key(tmp_path, "other-org-x")))
    monkeypatch.setenv("SALID_GCP_PROJECT", "meeting-scribe-hackathon")

    def no_client(settings):
        raise AssertionError("no client may be made")

    monkeypatch.setattr(mod, "make_client", no_client)
    monkeypatch.setattr(mod, "make_media_client", no_client)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    assert mod.main(["--image", "--video"]) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "gcloud auth application-default login" in text
    assert "other-org-x" not in text and "FAKE-PRIVATE-KEY" not in text
    assert str(tmp_path) not in text  # not even the key's path
    assert not (tmp_path / "out").exists()


def test_probe_needs_something_to_probe(monkeypatch, capsys):
    mod = load_probe()
    monkeypatch.setattr(mod, "make_client", lambda s: pytest.fail("no client"))
    assert mod.main([]) == 2


def test_probe_writes_only_to_its_tmp_folder():
    assert str(load_probe().OUT_DIR) == "/tmp/salid-media-probe"


def test_probe_runs_offline_with_a_fake_client(tmp_path, monkeypatch, capsys):
    """The whole probe with fake clients (image, Omni and music): paid creates on the media
    client, polls on the main one, files in OUT_DIR, latency / status / usage printed."""
    mod = load_probe()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key(tmp_path, "proj-ok")))
    monkeypatch.setenv("SALID_GCP_PROJECT", "proj-ok")
    monkeypatch.setenv("SALID_VIDEO_HERO_MODEL", "setting-omni")
    monkeypatch.setenv("SALID_VIDEO_MUSIC_MODEL", "setting-lyria")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    fake = FakeInteractions(["in_progress", "completed"], b"MP4")  # the main client
    media = FakeInteractions(["completed"], audio=b"MP3")

    def image_call(model, contents, config):
        assert config.image_config.image_size == "1K"
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[
                types.Part.from_bytes(data=png_bytes((576, 1024)), mime_type="image/png")
            ]), finish_reason=types.FinishReason.STOP)],
            usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=7),
        )  # fmt: skip

    fake.models = type("M", (), {"generate_content": staticmethod(
        lambda *, model, contents, config: image_call(model, contents, config))})()  # fmt: skip
    media.models = fake.models
    monkeypatch.setattr(mod, "make_client", lambda s: fake)
    monkeypatch.setattr(mod, "make_media_client", lambda s: media)
    monkeypatch.setattr(mod, "POLL_S", 0.0)
    assert mod.main(["--image", "--video", "--music", "--video-model", "flag-omni"]) == 0
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["image.png", "music.mp3", "omni.mp4"]
    assert (tmp_path / "out" / "omni.mp4").read_bytes() == b"MP4"
    assert Image.open(tmp_path / "out" / "image.png").size == (576, 1024)
    video, music = media.created
    assert video["model"] == "flag-omni" and music["model"] == "setting-lyria"  # flag > setting
    assert fake.created == [] and fake.got == ["i", "i"] and media.got == []
    assert "--video-model, else SALID_VIDEO_HERO_MODEL" in " ".join(mod.__doc__.split())
    assert "--music-model, else SALID_VIDEO_MUSIC_MODEL" in " ".join(mod.__doc__.split())
    assert video["response_format"][0]["resolution"] == "360p"
    assert video["response_format"][0]["duration"] == "4s"
    text = capsys.readouterr().out
    assert "ms=" in text and "status=" in text and "usage=" in text
    assert "FAKE-PRIVATE-KEY" not in text
