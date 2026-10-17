import io
import shutil
import threading
import time

import pytest
from PIL import Image, ImageChops
from promo_helpers import fake_client, json_response, png_bytes, probe, text_response
from pydantic import ValidationError

from salid import cards, promo, videoad

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
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5).scenes[:1], [], "red", FONTS
    )
    assert len(calls) == 2 and stills[0].size == (1080, 1920) and billed == 2
    assert stills[0].tobytes() == videoad.fallback_card("red").tobytes()


def test_a_blocked_still_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        videoad.imagegen, "generate_images", lambda *a, **k: calls.append(1) or ([], "blocked")
    )
    stills, billed = videoad.scene_stills(
        object(), "m", sb(5, 5, 5, 5).scenes[:1], [], "gold", FONTS
    )
    assert len(calls) == 1 and billed == 1
    assert stills[0].tobytes() == videoad.fallback_card("gold").tobytes()


def test_photo_scenes_use_the_shop_photo_without_a_model_call(monkeypatch):
    monkeypatch.setattr(videoad.imagegen, "generate_images", lambda *a, **k: 1 / 0)
    stills, billed = videoad.scene_stills(
        object(),
        "m",
        sb(5, 5, 5, 5, src="photo:0").scenes[:1],
        [png_bytes((900, 900))],
        "red",
        FONTS,
    )
    assert stills[0].size == (1080, 1920) and billed == 0


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
        object(), "m", sb(5, 5, 5, 5, src="photo:0").scenes[:1], shots, "red", FONTS
    )
    assert stills[0].size == (1080, 1920) and stills[0].getpixel((540, 960)) == (9, 9, 9)
    assert len(calls) == 1 and "a latte on a table" in calls[0] and billed == 1


def test_ai_stills_are_9x16_2k_and_resized_to_the_frame(monkeypatch):
    seen = []

    def gen(client, model, prompt, **kw):
        seen.append((model, kw))
        return [Image.new("RGB", (1536, 2752), (40, 50, 60))], ""

    monkeypatch.setattr(videoad.imagegen, "generate_images", gen)
    stills, billed = videoad.scene_stills(object(), "m", sb(5, 5, 5, 5).scenes, [], "red", FONTS)
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
    stills, billed = videoad.scene_stills(object(), "m", board.scenes, [], "red", FONTS)
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
    stills, billed = videoad.scene_stills(object(), "m", board.scenes, shots, "red", FONTS)
    centre = [s.getpixel((540, 960)) for s in stills]
    assert centre == [(250, 0, 0), (9, 9, 9), (0, 250, 0), (9, 9, 9)] and billed == 2


def test_no_model_call_without_a_client_or_with_a_thai_prompt(monkeypatch):
    monkeypatch.setattr(videoad.imagegen, "generate_images", lambda *a, **k: 1 / 0)
    stills, billed = videoad.scene_stills(None, "m", sb(5, 5, 5, 5).scenes[:1], [], "red", FONTS)
    assert stills[0].size == (1080, 1920) and billed == 0
    thai = sb(5, 5, 5, 5, prompt="ลาเต้").scenes[:1]
    assert videoad.scene_stills(object(), "m", thai, [], "red", FONTS)[1] == 0


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
    with pytest.raises(RuntimeError, match="Error opening input: boom"):
        videoad.motion_clip(tmp_path / "s.png", 2.0, 0, tmp_path / "c.mp4")
    cmd = seen["cmd"]
    assert isinstance(cmd, list) and cmd[0] == "ffmpeg" and "-loop" in cmd
    assert seen["kw"] == {"check": False, "capture_output": True, "timeout": 120}
    for flag in ("libx264", "yuv420p", "veryfast", "18", str(tmp_path / "c.mp4")):
        assert flag in cmd
    assert cmd[cmd.index("-t") + 1] == "2.000"
