import importlib.util
import inspect
import io
import itertools
import logging
import re
from pathlib import Path

import httpx
import pytest
from google.genai import types
from PIL import Image
from promo_helpers import (
    fake_client,
    image_response,
    jpeg_with_orientation,
    photo,
    png_bytes,
    text_response,
    to_png,
)

from salid import cards, promo
from salid.config import Settings
from salid.media import DailyBudget


# ------------------------------------------------------------------ request
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


# ------------------------------------------------------------------ photo
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


# ------------------------------------------------------------------ scene
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
    assert not re.search("[฀-๿]", p)  # no Thai


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


# ------------------------------------------------------------------ layouts
FULL = promo.merge_spec(
    None,
    headline="ลาเต้เย็น หอมกรุ่น ลดพิเศษ 20%",
    scene="cafe",
    price="59.-",
    call_to_action="สั่งเลยวันนี้",
    default_shop="ร้านบ้านกาแฟ",
)
FONTS = cards.card_fonts()


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


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
        assert x0 >= promo.MARGIN and x1 <= w - promo.MARGIN, p
        assert y0 >= fmt.safe_top and y1 <= h - fmt.safe_bottom, p
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


@pytest.mark.parametrize("fmt", promo.FORMATS, ids=lambda f: f.name)
def test_text_over_the_photo_gets_a_shadow(fmt, monkeypatch):
    seen = []
    real = cards.draw_lines

    def spy(img, x, y, f, path, **k):
        seen.append((" ".join(f.lines), k.get("shadow", 0)))
        return real(img, x, y, f, path, **k)

    monkeypatch.setattr(cards, "draw_lines", spy)
    promo.render_format(Image.new("RGB", (1536, 2752), "white"), FULL, fmt, FONTS)
    shadow = {text.split()[0]: s for text, s in seen}
    assert shadow["ร้านบ้านกาแฟ"] > 0 and shadow["ลาเต้เย็น"] > 0  # shop name, headline
    assert shadow["59.-"] == 0 and shadow["สั่งเลยวันนี้"] == 0  # badges sit on their own colour


def test_fallback_scene_contains_the_original_photo():
    red = Image.new("RGB", (800, 600), (220, 20, 20))
    p = promo.prepare_photo(to_png(red))
    s = promo.fallback_scene(p)
    assert s.size == (1536, 2752)
    r, g, b = s.getpixel((768, 1376))  # centre shows the sharp photo
    assert r > 180 and g < 60 and b < 60


# ------------------------------------------------------------------ tool
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


def test_sample_script_renders_offline(tmp_path):
    src = tmp_path / "p.png"
    src.write_bytes(png_bytes((900, 900)))
    script = Path(__file__).parents[2] / "scripts" / "salid_sample_promo.py"
    spec = importlib.util.spec_from_file_location("sample", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main([str(src), "--out", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "promo-1x1.jpg",
        "promo-4x5.jpg",
        "promo-9x16.jpg",
    ]


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
