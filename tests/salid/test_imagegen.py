import inspect
import io
import logging

import httpx
import pytest
from google.genai import types
from PIL import Image, ImageFile
from promo_helpers import fake_client, image_response, png_bytes, text_response

from salid import cards, imagegen, promo
from salid.media import DailyBudget


def promo_photo() -> promo.Photo:
    return promo.prepare_photo(png_bytes((800, 800)))


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


def test_thai_prompt_is_refused_without_a_call(caplog):
    c = fake_client(image_response(png_bytes((64, 64))))
    assert imagegen.generate_images(c, "m", "ลาเต้ on a table") == ([], "error")
    assert c.models.calls == [] and "ลาเต้" not in caplog.text


def test_open_capped_refuses_big_canvases_before_decoding(monkeypatch):
    data = png_bytes((1200, 1200))

    def no_decode(self):
        raise AssertionError("pixels decoded before the size check")

    # a PNG decodes through ImageFile.load; Image.Image.load alone would never be called
    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decode)
    monkeypatch.setattr(Image.Image, "load", no_decode)
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


# ------------------------------------------------------------------ create_image tool
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


def test_more_bad_fields_are_errors_without_calls():
    c = ictx()
    t = imagegen.image_tools(c)[0]
    for kw in (
        dict(prompt=""),
        dict(prompt="p" * 601),
        dict(prompt="x", size="4K"),
        dict(prompt="x", count=0),
        dict(prompt="x", subtitle="s" * 61),
        dict(prompt="x", headline="h", accent="pink"),
        dict(prompt="x", file_index=3),  # no such photo
        dict(prompt="x", revise=5),
    ):
        assert t(**kw)["status"] == "error", kw
    assert c.client.models.calls == []
    c.memory.put(c.key, [png_bytes((8, 8))])
    assert t(prompt="x", revise=2)["status"] == "error"  # only one remembered
    assert c.client.models.calls == [] and c.user_budget.used(7, c.day) == 0


def test_limit_when_nothing_is_left_for_the_guild_or_the_person():
    gb = DailyBudget(20)
    gb.add(1, "2026-10-17", 20)
    c = ictx(guild_budget=gb)
    out = imagegen.image_tools(c)[0](prompt="a cup")
    assert out["status"] == "limit" and "limit" in out["note"] and c.client.models.calls == []
    out = imagegen.image_tools(ictx(user_budget=DailyBudget(0)))[0](prompt="a cup")
    assert out["status"] == "limit"


def test_budgets_are_taken_per_image_before_the_calls(monkeypatch):
    taken = []

    def gen(client, model, prompt, **kw):  # what both budgets hold while the calls run
        taken.append((c.guild_budget.used(1, c.day), c.user_budget.used(7, c.day)))
        return [Image.new("RGB", (64, 64))] * kw["count"], ""

    monkeypatch.setattr(imagegen, "generate_images", gen)
    c = ictx()
    out = imagegen.image_tools(c)[0](prompt="a cup", count=2)
    assert taken == [(2, 2)] and out["count_made"] == 2 and out["images_left_today"] == 3


def test_units_of_images_that_did_not_come_back_are_given_back():
    c = ictx(client=fake_client(image_response(png_bytes((64, 64))), RuntimeError("x")))
    out = imagegen.image_tools(c)[0](prompt="a cup", count=2)
    assert out["status"] == "ok" and out["count_made"] == 1 and out["files"] == ["image-1.png"]
    assert c.guild_budget.used(1, c.day) == 1 and c.user_budget.used(7, c.day) == 1
    assert "refused or failed" in out["note"] and out["images_left_today"] == 4


def test_nothing_made_is_an_error_and_its_units_are_given_back():
    c = ictx(client=fake_client(text_response("no")))
    out = imagegen.image_tools(c)[0](prompt="a cup")
    assert out["status"] == "error" and out["reason"] == "blocked" and out["count_made"] == 0
    assert c.user_budget.used(7, c.day) == 0 and c.guild_budget.used(1, c.day) == 0
    assert out["images_left_today"] == 5 and c.memory.get(c.key) is None


def test_no_client_gives_every_unit_back():
    c = ictx(client=None)
    out = imagegen.image_tools(c)[0](prompt="a cup", count=3)
    assert out["status"] == "error" and out["reason"] == "error"
    assert c.guild_budget.used(1, c.day) == 0 and c.user_budget.used(7, c.day) == 0


def test_a_crash_after_the_calls_gives_the_units_back(monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("font gone")

    monkeypatch.setattr(imagegen, "overlay", broken)
    c = ictx()
    out = imagegen.image_tools(c)[0](prompt="a cup", count=2, headline="ลดพิเศษ")
    assert out["status"] == "error" and c.outbox == [] and c.memory.get(c.key) is None
    assert c.guild_budget.used(1, c.day) == 0 and c.user_budget.used(7, c.day) == 0


def test_reference_photos_are_never_of_a_real_person():
    """Spec section 5: a reference (attached photo or a revised result) is a product, logo, food
    or place; a photo of a real person is refused with an illustration offered instead."""
    doc = inspect.getdoc(imagegen.image_tools(ictx())[0])
    for text in (doc, imagegen.NOTE):
        assert "products, logos, food or places" in text
        assert "never a photo of a real person" in text and "illustration" in text


def test_note_is_the_ai_reminder_without_extras_when_all_were_made():
    out = imagegen.image_tools(ictx())[0](prompt="a cup", count=2)
    assert out["note"] == imagegen.NOTE and out["count_made"] == 2 and out["reason"] == ""


def test_attached_photo_is_the_reference(monkeypatch):
    seen = {}

    def fake(client, model, prompt, **k):
        seen.update(k, prompt=prompt)
        return [Image.new("RGB", (64, 64))], ""

    monkeypatch.setattr(imagegen, "generate_images", fake)
    photo = png_bytes((300, 200))
    out = imagegen.image_tools(ictx(photos=[(2, photo)]))[0](
        prompt="this bottle on a beach", file_index=2, aspect="4:5", size="2K"
    )
    assert out["status"] == "ok" and (seen["aspect"], seen["size"], seen["count"]) == (
        "4:5",
        "2K",
        1,
    )
    [ref] = seen["refs"]
    assert ref.startswith(b"\x89PNG") and Image.open(io.BytesIO(ref)).size == (300, 200)
    seen.clear()
    imagegen.image_tools(ictx(photos=[(2, photo)]))[0](prompt="a cup")
    assert seen["refs"] == []  # a photo is a reference only when the model picks it


def test_attached_photo_too_large_or_unreadable_is_an_error(monkeypatch):
    monkeypatch.setattr(imagegen, "MAX_DECODE_PIXELS", 1000)
    out = imagegen.image_tools(ictx(photos=[(0, png_bytes((100, 100)))]))[0](
        prompt="x", file_index=0
    )
    assert out["status"] == "error" and "megapixels" in out["error"]
    out = imagegen.image_tools(ictx(photos=[(0, b"not an image")]))[0](prompt="x", file_index=0)
    assert out["status"] == "error" and "read" in out["error"]


def test_files_are_png_with_overlay_and_memory_keeps_the_clean_images():
    c = ictx()
    t = imagegen.image_tools(c)[0]
    t(prompt="a cup", aspect="9:16", count=2, headline="ลดพิเศษ 20%", subtitle="วันนี้")
    name, data = t.outbox[0]
    with Image.open(io.BytesIO(data)) as im:
        assert name == "image-1.png" and im.format == "PNG" and im.size == (768, 1376)
        top = im.convert("RGB").crop((0, 0, 768, 300))
        bottom = im.convert("RGB").crop((0, 1076, 768, 1376))
    plain = Image.new("RGB", (768, 300), (120, 140, 160))
    assert top.tobytes() != plain.tobytes()  # 9:16: the text band is at the top
    assert bottom.tobytes() == plain.tobytes()
    [first, second] = c.memory.get(c.key)
    with Image.open(io.BytesIO(first)) as im:
        assert im.convert("RGB").getpixel((384, 100)) == (120, 140, 160)  # no text in memory
    t(prompt="a mug")
    assert [n for n, _ in t.outbox] == ["image-1.png"]  # a second call replaces the first batch
    assert len(c.memory.get(c.key)) == 1


def test_a_batch_over_the_upload_budget_becomes_jpeg(monkeypatch):
    noisy = [Image.effect_noise((2048, 2048), 120).convert("RGB") for _ in range(4)]
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: (list(noisy), ""))
    c = ictx()
    t = imagegen.image_tools(c)[0]
    out = t(prompt="static on an old tv", count=4, size="2K", headline="ทีวีเก่า")
    names = [n for n, _ in t.outbox]
    assert out["status"] == "ok" and out["files"] == names
    assert names == ["image-1.jpg", "image-2.jpg", "image-3.jpg", "image-4.jpg"]
    assert sum(len(d) for _, d in t.outbox) <= imagegen.OUTBOX_MAX_BYTES
    with Image.open(io.BytesIO(t.outbox[0][1])) as im:
        assert im.format == "JPEG" and im.size == (2048, 2048)
    assert all(d.startswith(b"\x89PNG") for d in c.memory.get(c.key))  # memory stays PNG


def test_the_upload_budget_is_the_contexts_own(monkeypatch):
    noisy = [Image.effect_noise((2048, 2048), 120).convert("RGB") for _ in range(2)]
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: (list(noisy), ""))
    c = ictx(max_bytes=8 * 2**20)
    t = imagegen.image_tools(c)[0]
    out = t(prompt="static on an old tv", count=2, size="2K")
    assert out["files"] == ["image-1.jpg", "image-2.jpg"] == [n for n, _ in t.outbox]
    assert sum(len(d) for _, d in t.outbox) <= 8 * 2**20
    assert ictx().max_bytes == imagegen.OUTBOX_MAX_BYTES


def test_a_jpeg_batch_still_over_the_budget_is_shrunk_until_it_fits(monkeypatch):
    noisy = [Image.effect_noise((1024, 1024), 120).convert("RGB") for _ in range(3)]
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: (list(noisy), ""))
    first = sum(len(cards.to_jpeg(img, max_bytes=10**9)) for img in noisy)
    budget = first // 4  # the first JPEG pass cannot fit this
    c = ictx(max_bytes=budget)
    t = imagegen.image_tools(c)[0]
    out = t(prompt="static on an old tv", count=3)
    assert out["status"] == "ok" and out["files"] == ["image-1.jpg", "image-2.jpg", "image-3.jpg"]
    assert sum(len(d) for _, d in t.outbox) <= budget
    assert c.user_budget.used(7, c.day) == 3


def test_images_that_never_fit_are_dropped_from_the_end_and_said(monkeypatch):
    noisy = [Image.effect_noise((1024, 1024), 120).convert("RGB") for _ in range(3)]
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: (list(noisy), ""))
    small = imagegen._jpeg(noisy[0], *imagegen.SHRINK_STEPS[-1])
    c = ictx(max_bytes=len(small) + 100)  # room for one image at the smallest step
    t = imagegen.image_tools(c)[0]
    out = t(prompt="static on an old tv", count=3)
    assert out["status"] == "ok" and out["files"] == ["image-1.jpg"] == [n for n, _ in t.outbox]
    assert sum(len(d) for _, d in t.outbox) <= c.max_bytes
    assert "upload limit" in out["note"] and out["count_made"] == 1
    assert c.user_budget.used(7, c.day) == 1 and len(c.memory.get(c.key)) == 1  # 2 given back


def test_a_batch_that_cannot_fit_at_all_is_an_error_with_its_units_back(monkeypatch):
    noisy = [Image.effect_noise((512, 512), 120).convert("RGB") for _ in range(2)]
    monkeypatch.setattr(imagegen, "generate_images", lambda *a, **k: (list(noisy), ""))
    c = ictx(max_bytes=100)
    t = imagegen.image_tools(c)[0]
    out = t(prompt="static on an old tv", count=2)
    assert out["status"] == "error" and out["reason"] == "too_large" and t.outbox == []
    assert c.user_budget.used(7, c.day) == 0 and c.memory.get(c.key) is None


def test_a_small_batch_stays_png():
    t = imagegen.image_tools(ictx())[0]
    out = t(prompt="a cup", count=4, size="1K")
    assert out["files"] == [f"image-{i}.png" for i in range(1, 5)]
    assert all(d.startswith(b"\x89PNG") for _, d in t.outbox)


def test_overlay_darkens_the_band_and_draws_thai():
    w, h = 1024, 1024
    out, placed = imagegen.overlay(
        Image.new("RGB", (w, h), "white"), "ลดพิเศษ", "", "red", cards.card_fonts(), top=False
    )
    assert out.getpixel((5, h - 5))[0] < 255 and out.getpixel((5, 5)) == (255, 255, 255)
    [(kind, box)] = placed
    text = out.crop(box).convert("L")
    assert kind == "headline" and text.getextrema()[1] == 255  # white glyphs over the scrim
    _, placed = imagegen.overlay(
        Image.new("RGB", (w, h)), "", "only a subtitle", "blue", cards.card_fonts(), top=True
    )
    assert [k for k, _ in placed] == ["subtitle"]


def test_image_memory_expires_drops_oldest_and_caps_bytes():
    now = [0.0]
    mem = imagegen.ImageMemory(ttl_s=10, max_entries=2, clock=lambda: now[0])
    mem.put((1, 1, 1), [b"a"])
    mem.put((1, 1, 2), [b"b"])
    mem.put((1, 1, 3), [b"c"])
    assert mem.get((1, 1, 1)) is None and mem.get((1, 1, 3)) == [b"c"]
    now[0] = 11
    assert mem.get((1, 1, 3)) is None
    small = imagegen.ImageMemory(max_bytes=10)
    small.put((1, 1, 1), [b"x" * 6])
    small.put((1, 1, 2), [b"y" * 6])
    assert small.get((1, 1, 1)) is None and small.get((1, 1, 2)) == [b"y" * 6]
