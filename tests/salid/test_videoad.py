import inspect
import io
import re
import shutil
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FixedClockSettings
from PIL import Image, ImageChops
from promo_helpers import (
    fake_client,
    json_response,
    png_bytes,
    probe,
    silent_wav,
    text_response,
    tiny_scene_assets,
    video_frame,
)
from pydantic import ValidationError

from salid import cards, promo, videoad
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.i18n import t as tr

FONTS = cards.card_fonts()
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg"
)


def sb(*secs, src="ai", prompt="a latte on a table", voice="หอมกรุ่น"):
    return videoad.Storyboard(
        title="t",
        scenes=[
            videoad.Scene(
                image_prompt_en=prompt, caption_th="ลาเต้", voice_th=voice, seconds=s, source=src
            )
            for s in secs
        ],
    )


def sources(board, *srcs):
    for scene, src in zip(board.scenes, srcs, strict=False):
        scene.source = src
    return board


def photos(n: int) -> list[bytes]:
    return [png_bytes((900, 900)) for _ in range(n)]


def image_parts(contents) -> tuple[list, list[int]]:
    parts = contents[0].parts
    return parts, [i for i, p in enumerate(parts) if p.inline_data is not None]


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


def test_check_storyboard_edges():
    ok = dict(seconds=20, photos=0, hero_shots=0)
    assert videoad.check_storyboard(sb(5, 5, 5, 6), **ok) == []  # 21 s is within 1 s
    assert videoad.check_storyboard(sb(5, 5, 5, 4), **ok) == []  # 19 s too
    one_photo = dict(seconds=20, photos=1, hero_shots=0)
    assert videoad.check_storyboard(sb(5, 5, 5, 5, src="photo:0"), **one_photo) == []
    assert videoad.check_storyboard(sb(5, 5, 5, 5, src="photo:0"), **ok)  # no photos at all
    assert videoad.check_storyboard(sb(5, 5, 5, 5, src="photo: 0"), **one_photo)
    assert videoad.check_storyboard(sb(5, 5, 5, 5, src="stock"), **ok)  # unknown source
    two_heroes = sources(sb(5, 5, 5, 5), "hero", "hero")
    assert videoad.check_storyboard(two_heroes, seconds=20, photos=0, hero_shots=2) == []


def test_scene_and_storyboard_limits():
    with pytest.raises(ValidationError):
        videoad.Scene(image_prompt_en="x", caption_th="ก" * 41, voice_th="", seconds=4, source="ai")
    with pytest.raises(ValidationError):
        videoad.Storyboard(title="t", scenes=sb(5, 5, 5).scenes)  # 3 scenes


def test_fit_narration_never_cuts_a_syllable():
    out = videoad.fit_narration("ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือน", 2.0, cps=12.0)
    assert len(out) <= 24 and cards.break_ok(out[-1], "ก")
    assert videoad.fit_narration("short", 5.0) == "short"


def test_fit_narration_keeps_most_of_the_line_when_the_last_space_is_early():
    text = "ลด 20% ลาเต้เย็นหอมกรุ่นลดราคาพิเศษทั้งเดือนสั่งเลยวันนี้ที่ร้านใกล้บ้านคุณทุกสาขาทั่วประเทศ"
    out = videoad.fit_narration(text, 5.0)
    assert 30 < len(out) <= 60 and text.startswith(out)
    assert cards.break_ok(out[-1], text[len(out)])  # a syllable-safe cut of the original


def test_fit_narration_prefers_the_last_space():
    assert videoad.fit_narration("ลาเต้เย็น หอมกรุ่น ลดราคาพิเศษ", 2.0, cps=12.0) == "ลาเต้เย็น หอมกรุ่น"
    assert videoad.fit_narration("one two three", 1.0, cps=7.0) == "one two"


def test_plan_storyboard_reasks_once_then_fails():
    bad = json_response(sb(5, 5, 5, 8).model_dump())
    good = json_response(sb(5, 5, 5, 5).model_dump())
    c = fake_client(bad, good)
    out = videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=[], hero_shots=0, lang="th"
    )
    assert sum(s.seconds for s in out.scenes) == 20 and len(c.models.calls) == 2
    assert "total" in str(c.models.calls[1][1])  # problems were sent back
    with pytest.raises(videoad.StoryboardError):
        videoad.plan_storyboard(
            fake_client(bad, bad), "m", brief="x", seconds=20, photos=[], hero_shots=0, lang="th"
        )


def test_storyboard_request_asks_for_json_and_carries_the_rules():
    both = sources(sb(5, 5, 5, 5), "photo:0", "photo:1")
    c = fake_client(json_response(both.model_dump()))
    videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=photos(2), hero_shots=1, lang="th"
    )
    cfg = c.models.calls[0][2]
    assert cfg.response_mime_type == "application/json"
    assert "children" in cfg.system_instruction and "photo:" in cfg.system_instruction


def test_prompt_carries_every_safety_rule_and_the_numbers():
    both = sources(sb(5, 5, 5, 5), "photo:0", "photo:1")
    c = fake_client(json_response(both.model_dump()))
    videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=photos(2), hero_shots=1, lang="en"
    )
    model, contents, cfg = c.models.calls[0]
    rules = cfg.system_instruction
    assert cfg.response_schema is videoad.Storyboard and model == "m"
    for word in ("celebrit", "public figure", "children", "health", "cosmetic", "supplement"):
        assert word in rules
    assert "quot" in rules and "English" in rules and "20 seconds" in rules
    assert "{" not in rules  # every placeholder was filled
    assert "every" in rules and "photo:0" in rules and "photo:1" in rules
    assert "latte" in str(contents)


def test_prompt_without_photos_claims_none():
    c = fake_client(json_response(sb(5, 5, 5, 5).model_dump()))
    videoad.plan_storyboard(c, "m", brief="latte", seconds=20, photos=[], hero_shots=0, lang="th")
    rules = c.models.calls[0][2].system_instruction
    assert "no shop photos" in rules and "follow" not in rules and "attached" not in rules
    assert image_parts(c.models.calls[0][1])[1] == []


def test_the_model_sees_each_photo_after_its_label_and_the_brief_last():
    board = sources(sb(5, 5, 5, 5), "photo:0", "photo:1", "photo:2")
    c = fake_client(json_response(board.model_dump()))
    shots = [png_bytes((2000, 1000)), png_bytes((900, 900)), png_bytes((600, 1200))]
    videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=shots, hero_shots=0, lang="th"
    )
    parts, images = image_parts(c.models.calls[0][1])
    assert [parts[i - 1].text for i in images] == ["photo:0", "photo:1", "photo:2"]
    assert parts[-1].text == "latte" and images[-1] < len(parts) - 1
    first = parts[images[0]].inline_data
    assert first.mime_type == "image/jpeg"
    assert Image.open(io.BytesIO(first.data)).size == (768, 384)
    assert len(c.models.calls) == 1


def test_an_unreadable_photo_is_skipped_but_keeps_its_index():
    board = sources(sb(5, 5, 5, 5), "photo:1")
    c = fake_client(json_response(board.model_dump()))
    shots = [b"not an image", png_bytes((900, 900))]
    out = videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=shots, hero_shots=0, lang="th"
    )
    parts, images = image_parts(c.models.calls[0][1])
    assert len(images) == 1 and parts[images[0] - 1].text == "photo:1"
    assert len(c.models.calls) == 1 and out.scenes[0].source == "photo:1"


def test_photo_labels_are_checked_against_the_photos_the_model_saw():
    """photo:N must be a photo the model was shown: readable, and one of the first MAX_PHOTOS."""
    board = sources(sb(5, 5, 5, 5), "photo:1")
    assert videoad.check_storyboard(board, seconds=20, photos=[1], hero_shots=0) == []
    unseen = sources(sb(5, 5, 5, 5), "photo:0", "photo:1")
    [problem] = videoad.check_storyboard(unseen, seconds=20, photos=[1], hero_shots=0)
    assert "photo:0" in problem and "photo:1" in problem  # the labels it may use
    seventh = sources(sb(5, 5, 5, 5), "photo:6")
    assert videoad.check_storyboard(seventh, seconds=20, photos=range(6), hero_shots=0)
    assert videoad.check_storyboard(board, seconds=20, photos=[], hero_shots=0)  # none shown


def test_a_storyboard_using_an_unreadable_photo_is_reasked_then_fails():
    shots = [b"not an image", png_bytes((900, 900))]  # photo:0 is never shown to the model
    bad = sources(sb(5, 5, 5, 5), "photo:0", "photo:1")
    c = fake_client(json_response(bad.model_dump()), json_response(bad.model_dump()))
    with pytest.raises(videoad.StoryboardError):
        videoad.plan_storyboard(
            c, "m", brief="latte", seconds=20, photos=shots, hero_shots=0, lang="th"
        )
    assert len(c.models.calls) == 2 and "photo:0" in str(c.models.calls[1][1][-1])


def test_a_seventh_photo_is_never_a_scene():
    bad = sources(sb(5, 5, 5, 5), "photo:0", "photo:1", "photo:2", "photo:6")
    good = sources(sb(5, 5, 5, 5), "photo:0", "photo:1", "photo:2", "photo:3")
    c = fake_client(json_response(bad.model_dump()), json_response(good.model_dump()))
    out = videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=photos(7), hero_shots=0, lang="th"
    )
    assert len(c.models.calls) == 2 and "photo:6" in str(c.models.calls[1][1][-1])
    assert [s.source for s in out.scenes] == ["photo:0", "photo:1", "photo:2", "photo:3"]


def test_an_unused_photo_is_reasked_once_then_accepted():
    only_first = sources(sb(5, 5, 5, 5), "photo:0")
    c = fake_client(json_response(only_first.model_dump()), json_response(only_first.model_dump()))
    out = videoad.plan_storyboard(
        c, "m", brief="latte", seconds=20, photos=photos(2), hero_shots=0, lang="th"
    )
    assert len(c.models.calls) == 2 and "photo:1" in str(c.models.calls[1][1][-1])
    assert [s.source for s in out.scenes] == ["photo:0", "ai", "ai", "ai"]


def test_bad_json_and_schema_errors_are_sent_back():
    long_caption = sb(5, 5, 5, 5).model_dump()
    long_caption["scenes"][0]["caption_th"] = "ก" * 41
    c = fake_client(text_response("not json"), json_response(long_caption), text_response("{}"))
    with pytest.raises(videoad.StoryboardError):
        videoad.plan_storyboard(c, "m", brief="x", seconds=20, photos=[], hero_shots=0, lang="th")
    assert len(c.models.calls) == 2
    c = fake_client(json_response(long_caption), json_response(sb(5, 5, 5, 5).model_dump()))
    videoad.plan_storyboard(c, "m", brief="x", seconds=20, photos=[], hero_shots=0, lang="th")
    assert "caption_th" in str(c.models.calls[1][1])
    assert "ก" * 41 not in str(c.models.calls[1][1][-1])  # the fix request echoes no content


def test_long_narration_is_reasked_once_then_trimmed_by_code():
    long = "หอมกรุ่น " * 9  # 81 characters; a 5 s scene fits 60
    c = fake_client(
        json_response(sb(5, 5, 5, 5, voice=long).model_dump()),
        json_response(sb(5, 5, 5, 5, voice=long).model_dump()),
    )
    out = videoad.plan_storyboard(c, "m", brief="x", seconds=20, photos=[], hero_shots=0, lang="th")
    assert len(c.models.calls) == 2 and "voice_th" in str(c.models.calls[1][1])
    assert all(0 < len(s.voice_th) <= 60 for s in out.scenes)


def test_a_worse_reask_keeps_the_first_storyboard():
    long = "หอมกรุ่น " * 9
    c = fake_client(
        json_response(sb(5, 5, 5, 5, voice=long).model_dump()),
        json_response(sb(5, 5, 5, 8).model_dump()),  # the re-ask broke the total
    )
    out = videoad.plan_storyboard(c, "m", brief="x", seconds=20, photos=[], hero_shots=0, lang="th")
    assert sum(s.seconds for s in out.scenes) == 20
    assert all(len(s.voice_th) <= 60 for s in out.scenes)


def test_an_empty_brief_still_sends_text():
    c = fake_client(json_response(sources(sb(5, 5, 5, 5), "photo:0").model_dump()))
    videoad.plan_storyboard(
        c, "m", brief="  ", seconds=20, photos=photos(1), hero_shots=0, lang="th"
    )
    assert c.models.calls[0][1][0].parts[-1].text.strip()


# ------------------------------------------------------------------ stills, captions, motion
@pytest.mark.parametrize("src", [(3000, 1000), (800, 4000), (1080, 1920), (900, 1500)])
def test_frame_photo_is_exact_and_undistorted(src):  # Review Focus 2
    out = videoad.frame_photo(Image.new("RGB", src, (200, 30, 30)))
    assert out.size == (1080, 1920)


def test_wide_photo_sits_on_a_blurred_fill():
    img = Image.new("RGB", (3000, 1000), (220, 20, 20))
    out = videoad.frame_photo(img)
    assert out.getpixel((540, 960))[0] > 180 and out.getpixel((540, 40)) != out.getpixel((540, 960))


def test_extreme_photos_keep_their_aspect_on_the_fill():  # Review Focus 2
    wide = Image.new("RGB", (3000, 1000), (255, 255, 255))
    out = videoad.frame_photo(wide).convert("L")
    rows = [y for y in range(1920) if out.getpixel((540, y)) > 240]
    assert len(rows) == pytest.approx(360, abs=4)  # 1080 wide -> 360 high, not stretched
    tall = Image.new("RGB", (800, 4000), (255, 255, 255))
    out = videoad.frame_photo(tall).convert("L")
    cols = [x for x in range(1080) if out.getpixel((x, 960)) > 240]
    assert len(cols) == pytest.approx(384, abs=4)  # 1920 high -> 384 wide


def test_near_9x16_photo_is_cover_cropped():
    img = Image.new("RGB", (900, 1500), (10, 200, 10))
    out = videoad.frame_photo(img)
    assert out.getpixel((2, 2)) == out.getpixel((540, 960))  # no fill border


@pytest.mark.parametrize("reason", ["error", "timeout"])
def test_failed_ai_still_retries_once_then_uses_a_card(monkeypatch, reason):
    calls = []
    monkeypatch.setattr(
        videoad.imagegen, "generate_images", lambda *a, **k: calls.append(1) or ([], reason)
    )
    stills, billed = videoad.scene_stills(object(), "m", sb(5, 5, 5, 5).scenes[:1], [], "red")
    assert len(calls) == 2 and stills[0].size == (1080, 1920) and billed == 2
    assert stills[0].tobytes() == videoad.fallback_card("red").tobytes()


def test_a_blocked_still_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        videoad.imagegen, "generate_images", lambda *a, **k: calls.append(1) or ([], "blocked")
    )
    stills, billed = videoad.scene_stills(object(), "m", sb(5, 5, 5, 5).scenes[:1], [], "gold")
    assert len(calls) == 1 and billed == 1
    assert stills[0].tobytes() == videoad.fallback_card("gold").tobytes()


def test_no_still_is_started_or_retried_after_a_cancel(monkeypatch):
    cancelled, calls = threading.Event(), []

    def gen(*a, **k):  # the user presses Cancel while the first still is being made
        calls.append(1)
        cancelled.set()
        return [], "error"

    monkeypatch.setattr(videoad.imagegen, "generate_images", gen)
    monkeypatch.setattr(videoad, "STILL_WORKERS", 1)  # the other scenes are queued
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5).scenes, [], "red", cancelled=cancelled
    )
    assert calls == [1] and billed == 1  # no retry, no queued still: nothing more is billed
    assert len(stills) == 4
    calls.clear()
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5).scenes[:2], [], "red", cancelled=cancelled
    )
    assert calls == [] and billed == 0  # already cancelled: no call at all


def test_photo_scenes_use_the_shop_photo_without_a_model_call(monkeypatch):
    monkeypatch.setattr(videoad.imagegen, "generate_images", lambda *a, **k: 1 / 0)
    stills, billed = videoad.scene_stills(
        object(),
        "m",
        sb(5, 5, 5, 5, src="photo:0").scenes[:1],
        [png_bytes((900, 900))],
        "red",
    )
    assert stills[0].size == (1080, 1920) and billed == 0


def test_unused_parameters_are_gone():
    assert "fonts" not in inspect.signature(videoad.scene_stills).parameters
    assert "board" not in inspect.signature(videoad._music_track).parameters


def _gif() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (900, 900), (1, 2, 3)).save(buf, "GIF")
    return buf.getvalue()


@pytest.mark.parametrize(
    "shots", [[b"not an image"], [_gif()], []], ids=["unreadable", "gif", "missing"]
)
def test_an_unusable_shop_photo_falls_back_to_an_ai_still(monkeypatch, shots):
    calls = []

    def gen(client, model, prompt, **kw):
        calls.append(prompt)
        return [Image.new("RGB", (768, 1376), (9, 9, 9))], ""

    monkeypatch.setattr(videoad.imagegen, "generate_images", gen)
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5, src="photo:0").scenes[:1], shots, "red"
    )
    assert stills[0].size == (1080, 1920) and stills[0].getpixel((540, 960)) == (9, 9, 9)
    assert len(calls) == 1 and "a latte on a table" in calls[0] and billed == 1


def test_ai_stills_are_9x16_2k_and_resized_to_the_frame(monkeypatch):
    seen = []

    def gen(client, model, prompt, **kw):
        seen.append((model, kw))
        return [Image.new("RGB", (1536, 2752), (40, 50, 60))], ""

    monkeypatch.setattr(videoad.imagegen, "generate_images", gen)
    stills, billed = videoad.scene_stills(object(), "m", sb(5, 5, 5, 5).scenes, [], "red")
    assert [s.size for s in stills] == [(1080, 1920)] * 4 and billed == 4
    assert seen[0] == ("m", {"aspect": "9:16", "size": "2K"})


def test_stills_run_at_most_three_at_a_time_and_keep_scene_order(monkeypatch):
    lock, running, peak = threading.Lock(), [0], [0]

    def gen(client, model, prompt, **kw):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        shade = int(prompt.split()[1])
        return [Image.new("RGB", (768, 1376), (shade, shade, shade))], ""

    monkeypatch.setattr(videoad.imagegen, "generate_images", gen)
    board = sb(4, 4, 4, 4, 2, 2)
    for i, scene in enumerate(board.scenes):
        scene.image_prompt_en = f"shade {i * 10} cup"
    stills, billed = videoad.scene_stills(object(), "m", board.scenes, [], "red")
    assert peak[0] == 3 and billed == 6
    assert [s.getpixel((540, 960))[0] for s in stills] == [0, 10, 20, 30, 40, 50]


def test_mixed_photo_and_ai_scenes_keep_their_order(monkeypatch):
    monkeypatch.setattr(
        videoad.imagegen,
        "generate_images",
        lambda *a, **k: ([Image.new("RGB", (768, 1376), (9, 9, 9))], ""),
    )
    board = sources(sb(5, 5, 5, 5), "photo:0", "ai", "photo:1", "hero")
    shots = [png_bytes((900, 1600), (250, 0, 0)), png_bytes((900, 1600), (0, 250, 0))]
    stills, billed = videoad.scene_stills(object(), "m", board.scenes, shots, "red")
    centre = [s.getpixel((540, 960)) for s in stills]
    assert centre == [(250, 0, 0), (9, 9, 9), (0, 250, 0), (9, 9, 9)] and billed == 2


def test_no_model_call_without_a_client_or_with_a_thai_prompt(monkeypatch):
    monkeypatch.setattr(videoad.imagegen, "generate_images", lambda *a, **k: 1 / 0)
    stills, billed = videoad.scene_stills(None, "m", sb(5, 5, 5, 5).scenes[:1], [], "red")
    assert stills[0].size == (1080, 1920) and billed == 0
    thai = sb(5, 5, 5, 5, prompt="ลาเต้").scenes[:1]
    assert videoad.scene_stills(object(), "m", thai, [], "red")[1] == 0


@pytest.mark.parametrize("accent", list(promo.ACCENTS))
def test_fallback_card_is_a_textless_gradient_under_a_readable_caption(accent):
    card = videoad.fallback_card(accent)
    assert card.size == (1080, 1920) and card.mode == "RGB"
    r, g, b = card.crop(videoad.SAFE).split()
    lowest = ImageChops.darker(ImageChops.darker(r, g), b)  # per pixel: its darkest channel
    assert lowest.getextrema()[1] < 230  # no near-white (text) pixel inside SAFE
    assert len(card.getcolors(1_000_000)) > 1  # a gradient, not one flat colour
    cap, box = videoad.caption_png("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", FONTS)
    comp = card.convert("RGBA")
    comp.alpha_composite(cap)
    assert (255, 255, 255) in {c for _, c in comp.crop(box).convert("RGB").getcolors(1_000_000)}
    brightest = "#" + "".join(f"{hi:02X}" for _, hi in card.crop(box).getextrema())
    assert cards.contrast("#FFFFFF", brightest) >= 3.0


def test_fallback_card_with_an_unknown_accent_is_red():
    assert (
        videoad.fallback_card("no-such-accent").tobytes() == videoad.fallback_card("red").tobytes()
    )


def test_caption_stays_in_the_safe_area_with_the_label():
    png, box = videoad.caption_png("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", FONTS)
    x0, y0, x1, y1 = box
    assert png.mode == "RGBA" and png.size == (1080, 1920)
    assert x0 >= 65 and x1 <= 1015 and y0 >= 270 and y1 <= 1250
    assert png.getpixel((5, 1900))[3] == 0  # outside: transparent


def test_caption_png_draws_only_inside_the_safe_area_and_the_label_is_optional():
    png, box = videoad.caption_png("ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20% ทั้งเดือน", FONTS)
    alpha = png.getchannel("A")
    text = alpha.point(lambda v: 255 if v == 255 else 0)  # fully opaque = text, not shadow
    tx0, ty0, tx1, ty1 = text.getbbox()
    assert tx0 >= 65 and tx1 <= 1015 and ty0 >= 270 and ty1 <= 1250
    label = alpha.crop((65, 270, 540, 360))
    assert label.getextrema()[1] > 0  # the AI label sits top-left inside SAFE
    assert box[1] > 360  # the caption does not cover it
    bare, _ = videoad.caption_png("ลาเต้เย็น", FONTS, label=False)
    assert bare.getchannel("A").crop((65, 270, 540, 360)).getextrema()[1] == 0
    empty, _ = videoad.caption_png("", FONTS, label=False)
    assert empty.getchannel("A").getextrema() == (0, 0)


def test_zoompan_args_frames_and_direction():
    a, b = videoad.zoompan_args(4.0, 0), videoad.zoompan_args(4.0, 1)
    assert "d=120" in " ".join(a) and "s=1080x1920" in " ".join(a) and a != b
    assert "fps=30" in " ".join(a) and videoad.zoompan_args(4.0, 2) == a


@needs_ffmpeg
def test_motion_clip_renders_with_ffmpeg(tmp_path):
    still = tmp_path / "s.png"
    Image.new("RGB", (1080, 1920), "gray").save(still)
    videoad.motion_clip(still, 1.0, 0, tmp_path / "c.mp4")
    p = probe(tmp_path / "c.mp4")
    assert p["duration"] == pytest.approx(1.0, abs=0.1)
    assert p["size"] == (1080, 1920) and p["codecs"] == ["h264"]


@needs_ffmpeg
def test_motion_clip_failure_carries_ffmpegs_last_line(tmp_path):
    with pytest.raises(RuntimeError, match="No such file"):
        videoad.motion_clip(tmp_path / "missing.png", 1.0, 1, tmp_path / "c.mp4")


def test_motion_clip_runs_ffmpeg_without_a_shell(monkeypatch, tmp_path):
    seen = {}

    class Done:
        returncode = 1
        stderr = b"frame=0\nError opening input: boom\n"

    def run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return Done()

    monkeypatch.setattr(videoad.subprocess, "run", run)
    monkeypatch.setattr(videoad, "_low_priority", lambda: [])  # no `nice` on this machine
    with pytest.raises(RuntimeError, match="Error opening input: boom"):
        videoad.motion_clip(tmp_path / "s.png", 2.0, 0, tmp_path / "c.mp4")
    cmd = seen["cmd"]
    assert isinstance(cmd, list) and cmd[0] == "ffmpeg" and "-loop" in cmd
    assert seen["kw"] == {"check": False, "capture_output": True, "timeout": 120}
    for flag in ("libx264", "yuv420p", "veryfast", "18", str(tmp_path / "c.mp4")):
        assert flag in cmd
    assert cmd[cmd.index("-t") + 1] == "2.000"


def test_encodes_run_at_low_priority_with_two_threads(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [1000] * 4)
    which = {"nice": "/usr/bin/nice"}
    monkeypatch.setattr(videoad.shutil, "which", lambda name: which.get(name))
    videoad.motion_clip(tmp_path / "s.png", 2.0, 0, tmp_path / "c.mp4")
    videoad.conform_clip(tmp_path / "omni.mp4", 2.0, tmp_path / "h.mp4")
    caps = caption_files(tmp_path, 2)
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    videoad.assemble(clips, [2.0, 2.0], caps, None, tmp_path / "o.mp4", budget_mib=9)
    assert len(calls) == 3
    for cmd in calls:
        assert cmd[:4] == ["/usr/bin/nice", "-n", "10", "ffmpeg"]
        assert opt(cmd, "-threads") == "2"  # an output option: the x264 encoder's threads
        assert cmd.index("-threads") > max(i for i, a in enumerate(cmd) if a == "-i")
    which.clear()  # no `nice` installed: plain ffmpeg
    videoad.motion_clip(tmp_path / "s.png", 2.0, 0, tmp_path / "c.mp4")
    assert calls[-1][0] == "ffmpeg" and opt(calls[-1], "-threads") == "2"


# ------------------------------------------------------------------ narration and assembly
def test_bitrate_formula():
    assert videoad.video_kbps(19, 30) == 5184 and videoad.budget_mib(None) == 9.0
    assert videoad.budget_mib(10 * 2**20) == 9.0
    assert videoad.budget_mib(50 * 2**20) == 19.0  # never above 19 MiB
    assert videoad.video_kbps(19, 30, audio_kbps=0) == 5312


def wav_samples(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path)) as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    return rate, [
        int.from_bytes(frames[i : i + 2], "little", signed=True) for i in range(0, len(frames), 2)
    ]


def pcm(value: int, seconds: float) -> bytes:
    """`seconds` of 24 kHz mono s16le PCM holding one constant sample value."""
    return value.to_bytes(2, "little", signed=True) * round(24_000 * seconds)


@needs_ffmpeg
def test_narration_never_runs_past_its_scene(tmp_path):  # Review Focus 1
    long_pcm = b"\x00\x01" * 24_000 * 6  # 6 s of audio
    ok = videoad.narration_wav(
        lambda text: long_pcm, [(0.0, 2.0, "a"), (2.0, 2.0, "b")], 4.0, tmp_path / "v.wav"
    )
    assert ok and probe(tmp_path / "v.wav")["duration"] == pytest.approx(4.0, abs=0.05)


@needs_ffmpeg
def test_narration_is_48k_with_each_line_at_its_scene_start(tmp_path):
    lines = [(0.0, 2.0, "a"), (2.0, 2.0, "b"), (4.0, 2.0, "c")]
    said = []

    def say(text):
        said.append(text)
        return pcm(1000, 6.0 if text == "b" else 0.5)

    assert videoad.narration_wav(say, lines, 6.0, tmp_path / "v.wav")
    rate, s = wav_samples(tmp_path / "v.wav")
    assert rate == 48_000 and len(s) == pytest.approx(6 * 48_000, abs=48)

    def at(t):
        return s[round(t * rate)]

    assert at(0.25) == pytest.approx(1000, abs=20) and at(1.0) == 0  # the line, then silence
    assert at(3.0) == pytest.approx(1000, abs=20)  # the long line fills its scene ...
    assert abs(at(3.995)) < 200  # ... and fades out at the cut instead of clicking
    assert at(4.25) == pytest.approx(1000, abs=20) and at(5.0) == 0
    assert said == ["a", "b", "c"]


def test_narration_failures_degrade(tmp_path):
    assert (
        videoad.narration_wav(lambda t: 1 / 0, [(0.0, 2.0, "a")], 2.0, tmp_path / "v.wav") is False
    )


@needs_ffmpeg
def test_failed_and_empty_lines_leave_silence_and_are_not_logged(tmp_path, caplog):
    said = []

    def say(text):
        said.append(text)
        if text == "secret line":
            raise RuntimeError("tts down")
        return pcm(1000, 1.0)

    lines = [(0.0, 2.0, "secret line"), (2.0, 2.0, "  "), (4.0, 2.0, "ok")]
    assert videoad.narration_wav(say, lines, 6.0, tmp_path / "v.wav")
    assert said == ["secret line", "ok"]  # an empty line is never sent to speech
    rate, s = wav_samples(tmp_path / "v.wav")
    assert s[round(1.0 * rate)] == 0 and s[round(4.5 * rate)] != 0
    assert "secret" not in caplog.text and "RuntimeError" in caplog.text
    empty = videoad.narration_wav(lambda t: b"", [(0.0, 2.0, "a")], 2.0, tmp_path / "e.wav")
    assert empty is False and not (tmp_path / "e.wav").exists()


@needs_ffmpeg
def test_assembled_ad_meets_the_delivery_spec(tmp_path):
    clips, caps = tiny_scene_assets(tmp_path, n=2, seconds=2.0)  # helper: 2 motion clips + captions
    out = videoad.assemble(
        clips, [2.0, 2.0], caps, silent_wav(tmp_path, 4.0), tmp_path / "ad.mp4", budget_mib=19
    )
    p = probe(out)
    assert p["size"] == (1080, 1920) and set(p["codecs"]) == {"h264", "aac"}
    assert p["duration"] == pytest.approx(4.0, abs=0.2) and out.stat().st_size < 19 * 2**20


@needs_ffmpeg
def test_crossfaded_scenes_add_up_and_the_ai_label_never_leaves(tmp_path):
    clips, caps = tiny_scene_assets(tmp_path, n=3, seconds=2.0)
    out = videoad.assemble(
        clips, [2.0, 2.0, 2.0], caps, None, tmp_path / "ad.mp4", budget_mib=19, fonts=FONTS
    )
    p = probe(out)
    assert p["duration"] == pytest.approx(6.0, abs=0.2) and p["codecs"] == ["h264"]
    box = videoad.label_png(FONTS).getchannel("A").getbbox()
    assert box is not None and box[3] <= 400  # top-left, above the captions
    cap_box = videoad.caption_png("ฉากที่ 1", FONTS, label=False)[1]
    end = p["duration"]
    frames = {
        t: video_frame(out, t, tmp_path / f"f{i}.png").convert("L")
        for i, t in enumerate((0.0, 0.1, 2.0, 2.1, 3.0, 4.2, end - 0.1, end - 0.04))
    }
    for t, frame in frames.items():  # the start, both scene changes, the end
        lo, hi = frame.crop(box).getextrema()
        assert lo < 90 and hi > 200, t  # the dark badge and its white text over the gray scene
    caption = {t: frame.crop(cap_box).getextrema()[1] for t, frame in frames.items()}
    assert caption[0.0] > 200 and caption[end - 0.04] > 200  # first and last frame: captions on
    assert caption[2.0] < 200 < caption[3.0]  # captions fade at a scene change


@needs_ffmpeg
def test_small_guild_budget_is_respected(tmp_path):  # Review Focus 3
    clips, caps = tiny_scene_assets(tmp_path, n=2, seconds=2.0, noisy=True)
    out = videoad.assemble(clips, [2.0, 2.0], caps, None, tmp_path / "ad.mp4", budget_mib=0.5)
    assert out.stat().st_size <= 0.5 * 2**20


def fake_ffmpeg(monkeypatch, sizes: list[int]) -> list[list[str]]:
    """Stand in for ffmpeg: each call writes a file of the next size to its output path."""
    calls = []

    class Done:
        returncode = 0
        stderr = b""

    def run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"\0" * sizes.pop(0))
        return Done()

    monkeypatch.setattr(videoad.subprocess, "run", run)
    return calls


def caption_files(tmp_path, n: int) -> list[Path]:
    caps = []
    for i in range(n):
        caps.append(tmp_path / f"cap{i}.png")
        videoad.caption_png(f"ฉาก {i}", FONTS, label=False)[0].save(caps[-1])
    return caps


def opt(cmd: list[str], flag: str) -> str:
    """The value after the first `flag` in an ffmpeg command."""
    return cmd[cmd.index(flag) + 1]


def graph_of(cmd: list[str]) -> str:
    return opt(cmd, "-filter_complex")


def test_assemble_command_follows_the_timeline_and_the_delivery_spec(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [1000])
    clips = [tmp_path / f"c{i}.mp4" for i in range(3)]
    out = videoad.assemble(
        clips,
        [2.0, 3.0, 2.5],
        caption_files(tmp_path, 3),
        tmp_path / "voice.wav",
        tmp_path / "ad.mp4",
        budget_mib=19,
        music=tmp_path / "music.m4a",
        fonts=FONTS,
    )
    assert out == tmp_path / "ad.mp4" and len(calls) == 1
    cmd, graph = calls[0], graph_of(calls[0])
    assert graph.count("tpad=stop_mode=clone:stop_duration=0.4") == 2  # every clip but the last
    assert "duration=0.4:offset=2.000" in graph and "duration=0.4:offset=5.000" in graph
    # captions fade at the two scene changes only; the AI label never fades
    assert graph.count("fade=t=in:st=0:d=0.2:alpha=1") == 2 and graph.count("fade=t=out") == 2
    assert "fade=t=out:st=1.800:d=0.2:alpha=1" in graph  # the first caption, a 2 s scene
    assert "fade=t=out:st=2.800:d=0.2:alpha=1" in graph  # the second, a 3 s scene
    assert "PTS-STARTPTS+5.000/TB" in graph  # the third caption starts with its scene
    assert graph.count("overlay=") == 4  # 3 captions + 1 AI label for the whole video
    assert "volume=-18dB" in graph and "amix=" in graph and "loudnorm=I=-16" in graph
    assert "scale=720:1280" not in graph
    for flag in ("libx264", "high", "yuv420p", "aac", "128k", "48000", "+faststart"):
        assert flag in cmd
    kbps = videoad.video_kbps(19, 7.5)
    assert opt(cmd, "-b:v") == f"{kbps}k" and opt(cmd, "-maxrate") == f"{int(kbps * 1.1)}k"
    assert opt(cmd, "-bufsize") == f"{2 * kbps}k" and cmd[-3:-1] == ["-t", "7.500"]
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"][:3] == [str(c) for c in clips]


def test_assemble_without_voice_or_music_is_video_only(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [1000])
    caps = caption_files(tmp_path, 1)
    videoad.assemble([tmp_path / "a.mp4"], [4.0], caps, None, tmp_path / "o.mp4", budget_mib=9)
    cmd, graph = calls[0], graph_of(calls[0])
    assert "xfade" not in graph and "tpad" not in graph and "loudnorm" not in graph
    assert "aac" not in cmd and opt(cmd, "-b:v") == f"{videoad.video_kbps(9, 4.0, 0)}k"


def test_an_empty_caption_is_not_overlaid(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [1000])
    caps = caption_files(tmp_path, 2)
    videoad.caption_png("", FONTS, label=False)[0].save(caps[1])
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    videoad.assemble(clips, [2.0, 2.0], caps, None, tmp_path / "o.mp4", budget_mib=9)
    assert graph_of(calls[0]).count("overlay=") == 2  # the first caption and the label


def test_over_budget_reencodes_at_80_percent_then_720p(monkeypatch, tmp_path):
    big = 2 * 2**20
    calls = fake_ffmpeg(monkeypatch, [big, big, 1000])
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    caps = caption_files(tmp_path, 2)
    out = videoad.assemble(clips, [2.0, 2.0], caps, None, tmp_path / "o.mp4", budget_mib=1)
    assert out.stat().st_size == 1000 and len(calls) == 3
    rates = [opt(c, "-b:v") for c in calls]
    kbps = videoad.video_kbps(1, 4.0, 0)
    assert rates == [f"{kbps}k", f"{int(kbps * 0.8)}k", f"{int(kbps * 0.8)}k"]
    assert ["scale=720:1280" in graph_of(c) for c in calls] == [False, False, True]


def test_still_over_budget_raises_and_leaves_no_file(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [2 * 2**20] * 3)
    caps = caption_files(tmp_path, 1)
    with pytest.raises(RuntimeError, match="budget"):
        videoad.assemble([tmp_path / "a.mp4"], [4.0], caps, None, tmp_path / "o.mp4", budget_mib=1)
    assert len(calls) == 3 and not (tmp_path / "o.mp4").exists()


def test_a_budget_too_small_for_the_audio_fails_before_encoding(monkeypatch, tmp_path):
    calls = fake_ffmpeg(monkeypatch, [])
    caps = caption_files(tmp_path, 1)
    voice = tmp_path / "v.wav"
    with pytest.raises(RuntimeError, match="budget"):  # 0.4 MiB over 30 s < 128 kbps of audio
        videoad.assemble(
            [tmp_path / "a.mp4"], [30.0], caps, voice, tmp_path / "o.mp4", budget_mib=0.4
        )
    assert calls == []


def test_assemble_needs_one_length_and_caption_per_clip(tmp_path):
    with pytest.raises(ValueError):
        videoad.assemble(
            [tmp_path / "a.mp4"], [2.0, 2.0], [], None, tmp_path / "o.mp4", budget_mib=9
        )


# ------------------------------------------------------------------ the job
S = FixedClockSettings()
S_HERO = FixedClockSettings(video_hero_model="omni-test")
S_MUSIC = FixedClockSettings(video_music_model="music-test")
SILENCE = b"\x00\x00" * 2400
REAL_NARRATION = videoad.narration_wav
BRIEF = "ลาเต้เย็นสูตรลับ ลด 20% ทุกวันศุกร์"


def MemDocs():
    return SQLiteDocStore(":memory:")


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return MemDocs()
    from fake_firestore import FakeFirestore

    return FirestoreDocStore(client=FakeFirestore())


def req(**kw):
    base = dict(
        guild_id=1,
        user_id=7,
        brief=BRIEF,
        seconds=20,
        photos=[],
        hero_shots=0,
        voice=True,
        music=False,
        lang="th",
    )
    return videoad.VideoRequest(**{**base, **kw})


@pytest.fixture
def fake_pipeline(monkeypatch):
    """The paid and slow parts of the job replaced by recorders: a 4 x 5 s storyboard (its first
    `hero_shots` scenes are hero scenes), `billed` (2) billed stills, empty clips, a narration
    file and an MP4 of a few bytes. `hero_ok` makes the hero hook succeed, `music_ok` the music
    hook."""
    fp = SimpleNamespace(
        hero_calls=0, hero_ok=False, on_stills=None, plans=[], stills=[], motion=[],
        narrations=[], assembled=[], labels=[], plan_error=None, music_calls=0, music_ok=False,
        billed=2,
    )  # fmt: skip

    def plan(client, model, *, brief, seconds, photos, hero_shots, lang):
        fp.plans.append(dict(model=model, brief=brief, seconds=seconds, photos=photos,
                             hero_shots=hero_shots, lang=lang))  # fmt: skip
        if fp.plan_error is not None:
            raise fp.plan_error
        board = sb(5, 5, 5, 5)
        for i in range(hero_shots):
            board.scenes[i].source = "hero"
        return board

    def stills(client, model, scenes, photos, accent, *, cancelled=None):
        fp.stills.append(
            dict(model=model, n=len(scenes), photos=photos, accent=accent, cancelled=cancelled)
        )
        if fp.on_stills is not None:
            fp.on_stills()
        return [Image.new("RGB", (108, 192), (90, 90, 90)) for _ in scenes], fp.billed

    def motion(still, seconds, index, out):
        assert still.exists()
        fp.motion.append((seconds, index))
        out.write_bytes(b"clip")

    def narration(say, lines, total_s, out):
        fp.narrations.append((lines, total_s))
        say(lines[0][2])
        out.write_bytes(b"wav")
        return True

    def assemble(
        clips, seconds, captions, audio, out, *, budget_mib, music=None, fonts=None, lang="th"
    ):
        fp.assembled.append(dict(clips=clips, seconds=seconds, captions=captions, audio=audio,
                                 budget=budget_mib, music=music, lang=lang))  # fmt: skip
        assert all(c.exists() for c in [*clips, *captions])
        out.write_bytes(b"mp4")
        return out

    def hero(client, settings, still, scene, handle, out, *, media_client=None):
        fp.hero_calls += 1
        if fp.hero_ok:
            out.write_bytes(b"hero")
        return fp.hero_ok

    def music(client, settings, seconds, handle, out, *, media_client=None):
        fp.music_calls += 1
        if not fp.music_ok:
            return None
        out.write_bytes(b"music")
        return out

    real_caption = videoad.caption_png

    def caption(text, fonts, *, label=True):
        fp.labels.append(label)
        return real_caption(text, fonts, label=label)

    for name, fn in [("plan_storyboard", plan), ("scene_stills", stills), ("motion_clip", motion),
                     ("narration_wav", narration), ("assemble", assemble), ("_hero_clip", hero),
                     ("caption_png", caption), ("_music_track", music)]:  # fmt: skip
        monkeypatch.setattr(videoad, name, fn)
    return fp


def run(tmp_path, *, handle=None, usage=None, settings=S, say=lambda t: SILENCE, **kw):
    return videoad.run_video_ad(
        req(**kw),
        client=None,
        settings=settings,
        say=say,
        workdir=tmp_path,
        budget_mib=19,
        handle=handle or videoad.VideoHandle(),
        usage=usage or videoad.MediaUsage(MemDocs()),
        month="2026-10",
    )


def test_month_usage_refuses_without_writing(docs):
    u = videoad.MediaUsage(docs)
    assert u.take(1, "2026-10", "ads", 1, 2) and u.take(1, "2026-10", "ads", 1, 2)
    assert (
        not u.take(1, "2026-10", "ads", 1, 2)
        and docs.get("salid_media_usage", 1, "2026-10")["ads"] == 2
    )
    assert u.take(1, "2026-11", "ads", 1, 2) and u.take(2, "2026-10", "ads", 1, 2)


def test_month_usage_peek_reads_without_writing(docs):
    u = videoad.MediaUsage(docs)
    assert u.used(1, "2026-10", "ads") == 0 and docs.get("salid_media_usage", 1, "2026-10") is None
    assert u.take(1, "2026-10", "hero_seconds", 5, 60) and u.take(1, "2026-10", "ads", 1, 10)
    assert u.used(1, "2026-10", "hero_seconds") == 5 and u.used(1, "2026-10", "ads") == 1
    assert not u.take(1, "2026-10", "hero_seconds", 56, 60)  # 61 > 60
    assert u.used(1, "2026-10", "hero_seconds") == 5


def test_month_usage_take_is_atomic_across_threads():
    u = videoad.MediaUsage(MemDocs())
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(u.take(1, "2026-10", "ads", 1, 5)))
        for _ in range(20)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert got.count(True) == 5 and u.used(1, "2026-10", "ads") == 5


def test_find_items_returns_each_docs_key(docs):
    docs.put("salid_video_jobs", 1, "j2", {"state": "done"})
    docs.put("salid_video_jobs", 1, "j1", {"state": "stills"})
    docs.put("salid_video_jobs", 2, "j3", {"state": "stills"})
    items = docs.find_items("salid_video_jobs", 1, where=lambda d: d["state"] != "done")
    assert [(k, d["state"]) for k, d in items] == [("j1", "stills")]
    assert [k for k, _ in docs.find_items("salid_video_jobs", 1)] == ["j1", "j2"]


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


def test_runner_wires_every_step(tmp_path, fake_pipeline):
    usage = videoad.MediaUsage(d := MemDocs())
    photos_in = [png_bytes((900, 900))]
    res = run(tmp_path, usage=usage, photos=photos_in, accent="gold")
    assert res.path == tmp_path / "ad-20261014-1200.mp4" and res.path.read_bytes() == b"mp4"
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 1}
    assert res.cost_usd == pytest.approx(2 * videoad.STILL_USD)
    assert set(res.steps_ms) == {"storyboard", "stills", "audio", "render"}
    [plan] = fake_pipeline.plans
    assert plan["model"] == S.gemini_model and plan["photos"] == photos_in
    assert plan["brief"] == BRIEF and plan["seconds"] == 20 and plan["hero_shots"] == 0
    [stills] = fake_pipeline.stills
    assert stills["model"] == S.image_model and stills["photos"] == photos_in
    assert stills["accent"] == "gold" and fake_pipeline.motion == [(5, i) for i in range(4)]
    [(lines, total)] = fake_pipeline.narrations
    assert [(a, b) for a, b, _ in lines] == [(0, 5), (5, 5), (10, 5), (15, 5)] and total == 20
    [asm] = fake_pipeline.assembled
    assert asm["seconds"] == [5, 5, 5, 5] and len(asm["clips"]) == len(asm["captions"]) == 4
    assert asm["audio"] is not None and asm["budget"] == 19 and asm["music"] is None
    assert fake_pipeline.labels == [False] * 4  # assemble adds the one AI label itself
    assert fake_pipeline.music_calls == 0  # no music asked for


def test_music_is_skipped_while_the_model_setting_is_empty(tmp_path, fake_pipeline):
    fake_pipeline.music_ok = True
    run(tmp_path, music=True)
    assert fake_pipeline.music_calls == 0 and fake_pipeline.assembled[0]["music"] is None


def test_music_hook_runs_when_enabled_and_its_track_is_mixed(tmp_path, fake_pipeline):
    seen = []
    fake_pipeline.music_ok = True
    h = videoad.VideoHandle(progress=lambda s, _d: seen.append(s))
    run(tmp_path, settings=S_MUSIC, music=True, voice=False, handle=h)
    assert fake_pipeline.music_calls == 1 and "audio" in seen
    assert fake_pipeline.assembled[0]["music"].read_bytes() == b"music"
    assert fake_pipeline.assembled[0]["audio"] is None  # no voice asked for


def test_no_music_track_means_no_music(tmp_path, fake_pipeline):
    run(tmp_path, settings=S_MUSIC, music=True)  # the hook returns None
    assert fake_pipeline.music_calls == 1 and fake_pipeline.assembled[0]["music"] is None


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
            usage=(u := videoad.MediaUsage(d := MemDocs())),  # noqa: F841 (brief's test as is)
            month="2026-10",
        )
    assert fake_pipeline.hero_calls == 0 and d.get("salid_media_usage", 1, "2026-10") == {"ads": 1}


def test_cancel_during_stills_skips_the_motion_clips(tmp_path, fake_pipeline):
    h = videoad.VideoHandle()
    fake_pipeline.on_stills = h.cancelled.set
    with pytest.raises(videoad.Cancelled):
        run(tmp_path, handle=h)
    assert fake_pipeline.motion == [] and fake_pipeline.assembled == []
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD)  # the stills were billed
    assert fake_pipeline.stills[0]["cancelled"] is h.cancelled  # queued stills stop too


def test_hero_cost_is_counted_as_soon_as_its_seconds_are_taken(
    tmp_path, fake_pipeline, monkeypatch
):
    h = videoad.VideoHandle()

    def hero(client, settings, still, scene, handle, out, *, media_client=None):
        handle.cancelled.set()  # cancelled while the first hero clip is being made
        return False

    monkeypatch.setattr(videoad, "_hero_clip", hero)
    with pytest.raises(videoad.Cancelled):
        run(tmp_path, handle=h, settings=S_HERO, hero_shots=2)
    assert h.cost_usd == pytest.approx(2 * videoad.STILL_USD + 5 * videoad.HERO_USD_PER_S)


def test_cancel_before_the_start_takes_nothing(tmp_path, fake_pipeline):
    h = videoad.VideoHandle()
    h.cancelled.set()
    with pytest.raises(videoad.Cancelled):
        run(tmp_path, handle=h, usage=videoad.MediaUsage(d := MemDocs()))
    assert fake_pipeline.plans == [] and d.get("salid_media_usage", 1, "2026-10") is None


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


def test_hero_seconds_are_taken_before_each_hero_call(tmp_path, fake_pipeline):
    fake_pipeline.hero_ok = True
    seen = []
    usage = videoad.MediaUsage(d := MemDocs())
    h = videoad.VideoHandle(progress=lambda s, _d: seen.append(s))
    res = run(tmp_path, handle=h, usage=usage, settings=S_HERO, hero_shots=2)
    assert fake_pipeline.plans[0]["hero_shots"] == 2 and fake_pipeline.hero_calls == 2
    assert seen == ["storyboard", "stills", "hero", "audio", "render"]
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 1, "hero_seconds": 10}
    assert res.cost_usd == pytest.approx(2 * videoad.STILL_USD + 10 * videoad.HERO_USD_PER_S)
    clips = fake_pipeline.assembled[0]["clips"]
    assert [c.read_bytes() for c in clips] == [b"hero", b"hero", b"clip", b"clip"]


def test_a_failed_hero_keeps_the_motion_clip_and_its_seconds_stay_taken(tmp_path, fake_pipeline):
    usage = videoad.MediaUsage(d := MemDocs())
    run(tmp_path, usage=usage, settings=S_HERO, hero_shots=1)
    assert fake_pipeline.hero_calls == 1
    assert d.get("salid_media_usage", 1, "2026-10")["hero_seconds"] == 5  # no refund
    assert fake_pipeline.assembled[0]["clips"][0].read_bytes() == b"clip"


def test_hero_month_budget_used_up_means_no_hero_call(tmp_path, fake_pipeline, caplog):
    caplog.set_level("INFO", logger="salid.videoad")
    d = MemDocs()
    d.put("salid_media_usage", 1, "2026-10", {"hero_seconds": 58})
    run(tmp_path, usage=videoad.MediaUsage(d), settings=S_HERO, hero_shots=1)
    assert fake_pipeline.hero_calls == 0
    assert d.get("salid_media_usage", 1, "2026-10") == {"hero_seconds": 58, "ads": 1}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("video guild=")]
    assert all(" step=" in ln for ln in lines)  # only step lines use the step prefix
    assert any("hero budget" in r.getMessage() for r in caplog.records)


def say_fails(text):
    raise RuntimeError("tts down")


def ffmpeg_fails(say, lines, total_s, out):
    raise RuntimeError("ffmpeg failed: no encoder")


@pytest.mark.parametrize(
    "say, narration",
    [(say_fails, REAL_NARRATION), (lambda t: SILENCE, ffmpeg_fails)],
    ids=["say raises", "ffmpeg raises"],
)
def test_narration_failure_still_makes_the_video_with_captions_only(
    tmp_path, fake_pipeline, monkeypatch, say, narration
):
    monkeypatch.setattr(videoad, "narration_wav", narration)
    res = run(tmp_path, say=say)
    assert res.path.exists() and fake_pipeline.assembled[0]["audio"] is None


def test_no_voice_means_no_audio_step(tmp_path, fake_pipeline):
    seen = []
    run(tmp_path, voice=False, handle=videoad.VideoHandle(progress=lambda s, d: seen.append(s)))
    assert "audio" not in seen and fake_pipeline.narrations == []
    assert fake_pipeline.assembled[0]["audio"] is None


def test_the_month_limit_is_checked_again_inside_the_job(tmp_path, fake_pipeline):
    d = MemDocs()
    d.put("salid_media_usage", 1, "2026-10", {"ads": S.video_ads_per_month})
    with pytest.raises(RuntimeError, match="month limit"):
        run(tmp_path, usage=videoad.MediaUsage(d))
    assert fake_pipeline.plans == []


def test_storyboard_and_render_errors_propagate(tmp_path, fake_pipeline, monkeypatch):
    fake_pipeline.plan_error = videoad.StoryboardError("no usable storyboard")
    with pytest.raises(videoad.StoryboardError):
        run(tmp_path)

    def broken(*a, **kw):
        raise RuntimeError("over budget")

    fake_pipeline.plan_error = None
    monkeypatch.setattr(videoad, "assemble", broken)
    with pytest.raises(RuntimeError, match="over budget"):
        run(tmp_path)


def test_a_failing_progress_callback_does_not_fail_the_job(tmp_path, fake_pipeline):
    def boom(step, detail):
        raise ValueError("ui gone")

    assert run(tmp_path, handle=videoad.VideoHandle(progress=boom)).path.exists()


def test_steps_are_logged_without_the_brief_or_captions(tmp_path, fake_pipeline, caplog):
    caplog.set_level("INFO", logger="salid.videoad")
    run(tmp_path)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("video guild=")]
    assert [ln.split()[2] for ln in lines] == [
        "step=storyboard", "step=stills", "step=audio", "step=render"
    ]  # fmt: skip
    assert all(re.fullmatch(r"video guild=1 step=\w+ ms=\d+ cost=\d+\.\d\d", ln) for ln in lines)
    assert lines[1].endswith(f"cost={2 * videoad.STILL_USD:.2f}")
    assert BRIEF not in caplog.text and "ลาเต้" not in caplog.text and "หอมกรุ่น" not in caplog.text


# ------------------------------------------------------------------ refunds
def ads_used(d) -> float:
    return (d.get("salid_media_usage", 1, "2026-10") or {}).get("ads", 0)


def test_a_failure_before_any_paid_media_gives_the_ad_back(tmp_path, fake_pipeline):
    fake_pipeline.plan_error = videoad.StoryboardError("no usable storyboard")
    d = MemDocs()
    with pytest.raises(videoad.StoryboardError):
        run(tmp_path, usage=videoad.MediaUsage(d))
    assert ads_used(d) == 0


def test_a_cancel_before_any_paid_media_gives_the_ad_back(tmp_path, fake_pipeline):
    h = videoad.VideoHandle()
    h.progress = lambda step, _d: step == "storyboard" and h.cancelled.set()
    d = MemDocs()
    with pytest.raises(videoad.Cancelled):
        run(tmp_path, handle=h, usage=videoad.MediaUsage(d))
    assert fake_pipeline.stills == [] and ads_used(d) == 0


def test_the_ad_stays_taken_once_paid_media_was_billed(tmp_path, fake_pipeline, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("over budget")

    monkeypatch.setattr(videoad, "assemble", broken)
    d = MemDocs()
    with pytest.raises(RuntimeError):
        run(tmp_path, usage=videoad.MediaUsage(d))  # 2 stills were billed
    assert ads_used(d) == 1


def test_a_failed_ad_made_only_of_shop_photos_gives_the_ad_back(
    tmp_path, fake_pipeline, monkeypatch
):
    def broken(*a, **kw):
        raise RuntimeError("over budget")

    fake_pipeline.billed = 0  # every scene used a shop photo: no image call
    monkeypatch.setattr(videoad, "assemble", broken)
    d = MemDocs()
    with pytest.raises(RuntimeError):
        run(tmp_path, usage=videoad.MediaUsage(d))
    assert ads_used(d) == 0


def test_a_finished_ad_keeps_its_ad_even_without_paid_media(tmp_path, fake_pipeline):
    fake_pipeline.billed = 0
    d = MemDocs()
    run(tmp_path, usage=videoad.MediaUsage(d))
    assert ads_used(d) == 1


def test_a_refused_hero_gives_its_seconds_and_cost_back(tmp_path, fake_pipeline, monkeypatch):
    calls = []

    def refused(client, settings, still, scene, handle, out, *, media_client=None):
        calls.append(handle.cost_usd)
        raise videoad.HeroRefused("PermissionDeniedError", definite=True)

    monkeypatch.setattr(videoad, "_hero_clip", refused)
    d = MemDocs()
    h = videoad.VideoHandle()
    res = run(tmp_path, handle=h, usage=videoad.MediaUsage(d), settings=S_HERO, hero_shots=2)
    assert len(calls) == 1  # no second hero request after a refusal
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 1, "hero_seconds": 0}
    assert res.cost_usd == pytest.approx(2 * videoad.STILL_USD) == h.cost_usd


def test_a_refused_hero_alone_does_not_keep_the_ad_of_a_failed_job(
    tmp_path, fake_pipeline, monkeypatch
):
    def refused(*a, **kw):
        raise videoad.HeroRefused("PermissionDeniedError", definite=True)

    def broken(*a, **kw):
        raise RuntimeError("over budget")

    fake_pipeline.billed = 0
    monkeypatch.setattr(videoad, "_hero_clip", refused)
    monkeypatch.setattr(videoad, "assemble", broken)
    d = MemDocs()
    with pytest.raises(RuntimeError):
        run(tmp_path, usage=videoad.MediaUsage(d), settings=S_HERO, hero_shots=1)
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 0, "hero_seconds": 0}


def test_an_uncertain_hero_refusal_stays_charged_but_stops_the_heroes(
    tmp_path, fake_pipeline, monkeypatch
):
    calls = []

    def unsure(client, settings, still, scene, handle, out, *, media_client=None):
        calls.append(1)
        raise videoad.HeroRefused("ReadTimeout")  # the service may have accepted it

    monkeypatch.setattr(videoad, "_hero_clip", unsure)
    d = MemDocs()
    res = run(tmp_path, usage=videoad.MediaUsage(d), settings=S_HERO, hero_shots=2)
    assert calls == [1]  # no second hero request
    assert d.get("salid_media_usage", 1, "2026-10") == {"ads": 1, "hero_seconds": 5}
    assert res.cost_usd == pytest.approx(2 * videoad.STILL_USD + 5 * videoad.HERO_USD_PER_S)


def test_the_ad_is_taken_before_the_storyboard_step_is_reported(tmp_path, fake_pipeline):
    d, seen = MemDocs(), []
    h = videoad.VideoHandle(progress=lambda step, _d: seen.append((step, ads_used(d))))
    run(tmp_path, handle=h, usage=videoad.MediaUsage(d))
    assert seen[0] == ("storyboard", 1)  # a "storyboard" record always has its ad taken


def test_the_refund_is_reported_before_it_is_made(tmp_path, fake_pipeline):
    """The record leaves "storyboard" ("returned") before the ad goes back, so a sweep after a
    shutdown at that moment never gives it back a second time."""
    fake_pipeline.plan_error = videoad.StoryboardError("no usable storyboard")
    d, seen = MemDocs(), []
    h = videoad.VideoHandle(progress=lambda step, _d: seen.append((step, ads_used(d))))
    with pytest.raises(videoad.StoryboardError):
        run(tmp_path, handle=h, usage=videoad.MediaUsage(d))
    assert seen == [("storyboard", 1), ("returned", 1)] and ads_used(d) == 0


def test_a_shutdown_leaves_the_refund_to_the_sweep(tmp_path, fake_pipeline):
    h = videoad.VideoHandle()

    def stop(step, _d):  # the bot shuts down while the storyboard is planned
        if step == "storyboard":
            h.shutdown.set()
            h.cancelled.set()

    h.progress = stop
    d = MemDocs()
    with pytest.raises(videoad.Cancelled):
        run(tmp_path, handle=h, usage=videoad.MediaUsage(d))
    assert ads_used(d) == 1  # the record stays for the next start's sweep, which decides


def test_give_back_never_goes_below_zero(docs):
    u = videoad.MediaUsage(docs)
    u.give_back(1, "2026-10", "ads", 1)
    assert u.used(1, "2026-10", "ads") == 0
    assert u.take(1, "2026-10", "ads", 1, 2)
    u.give_back(1, "2026-10", "ads", 1)
    assert u.used(1, "2026-10", "ads") == 0


# ------------------------------------------------------------------ the AI label's language
@pytest.mark.parametrize("lang, text", [("th", "สร้างด้วย AI"), ("en", "Made with AI")])
def test_the_ai_label_follows_the_reply_language(lang, text):
    assert tr("video_ai_label", lang) == text
    label = videoad.label_png(FONTS, lang=lang)
    box = label.getchannel("A").getbbox()
    assert label.size == (1080, 1920) and box is not None and box[3] <= 400
    overlay, _ = videoad.caption_png("ลาเต้", FONTS, lang=lang)
    assert overlay.crop(box).tobytes() == label.crop(box).tobytes()


def test_the_two_labels_differ():
    th, en = videoad.label_png(FONTS, lang="th"), videoad.label_png(FONTS, lang="en")
    assert th.tobytes() != en.tobytes()


def test_the_runner_renders_the_label_in_the_requests_language(tmp_path, fake_pipeline):
    run(tmp_path, lang="en")
    run(tmp_path, lang="th")
    assert [a["lang"] for a in fake_pipeline.assembled] == ["en", "th"]


def test_assemble_passes_the_language_to_the_label(monkeypatch, tmp_path):
    seen, real = [], videoad.label_png

    def label(fonts, lang="th"):
        seen.append(lang)
        return real(fonts, lang=lang)

    monkeypatch.setattr(videoad, "label_png", label)
    fake_ffmpeg(monkeypatch, [1000])
    caps = caption_files(tmp_path, 1)
    videoad.assemble(
        [tmp_path / "a.mp4"], [4.0], caps, None, tmp_path / "o.mp4", budget_mib=9, lang="en"
    )
    assert seen == ["en"]
