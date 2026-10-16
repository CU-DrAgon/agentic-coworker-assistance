import io
import re

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

from salid import promo
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
