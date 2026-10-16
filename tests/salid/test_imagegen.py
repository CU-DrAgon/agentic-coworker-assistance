import httpx
import pytest
from google.genai import types
from PIL import Image, ImageFile
from promo_helpers import fake_client, image_response, png_bytes, text_response

from salid import imagegen, promo


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
