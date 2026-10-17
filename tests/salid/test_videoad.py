import io

import pytest
from PIL import Image
from promo_helpers import fake_client, json_response, png_bytes, text_response
from pydantic import ValidationError

from salid import cards, videoad


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
