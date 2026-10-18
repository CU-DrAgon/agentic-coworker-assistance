# ruff: noqa: F811, F401
"""Concept studio: prompts, the image tool (budget, labels, revisions, storage in the idea, PDF),
price checks (parsing, range, sourcing), the bot glue (flags, voice) and the i18n lines."""

import io
import logging
import os
from datetime import UTC, date, datetime

import httpx
import pytest
from conftest import GUILD, fc_resp, scripted, text_resp
from google.genai import types
from PIL import Image
from promo_helpers import fake_client, image_response, png_bytes, text_response
from test_coordinator import STAFF, world  # fixture: world.g, .pub, .staff
from test_modes import TMsg, bot  # fixture

from salid import cards, concept, qa
from salid.concept import ConceptCtx, ConceptMemory, concept_prompt, concept_tools
from salid.coo_store import SQLiteDocStore
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.i18n import CATALOG, t
from salid.ideas import MAX_IMAGES, IdeaRepo, IdeasCtx, render_markdown
from salid.media import DailyBudget
from salid.pdf_report import find_fonts
from salid.qa import FileInput
from salid.voice_actions import ACTION_TOOL_NAMES

NOW = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)
TODAY = date(2026, 10, 17)


# ------------------------------------------------------------------ prompts
def test_prompt_always_forbids_text_logos_and_faces():
    for kind in concept.IMAGE_KINDS:
        p = concept_prompt("a rounded desk lamp", kind)
        assert "No text, letters, numbers" in p and "brand logos" in p and "faces" in p, kind
        assert "a rounded desk lamp" in p


@pytest.mark.parametrize(
    "kind, aspect",
    [("product", "1:1"), ("logo", "1:1"), ("moodboard", "1:1"), ("ui", "16:9"), ("scene", "16:9")],
)
def test_aspect_per_kind_and_sets_are_wide(kind, aspect):
    assert concept.aspect_for(kind, "front") == aspect
    assert f"Aspect ratio {aspect}." in concept_prompt("x lamp", kind)
    assert concept.aspect_for(kind, "set") == "16:9"
    assert concept.aspect_for(kind, "turnaround") == aspect


def test_kind_specific_wording():
    assert "plain seamless light-grey studio background" in concept_prompt("lamp", "product")
    assert "in use" in concept_prompt("lamp", "scene")
    ui = concept_prompt("job board", "ui")
    assert "grey placeholder bars" in ui and "never as readable words" in ui
    assert "Three different logo symbol ideas" in concept_prompt("tea", "logo")
    assert "moodboard" in concept_prompt("tea", "moodboard")


def test_views_turnaround_grid_set_and_style():
    p = concept_prompt("lamp", "product", "turnaround", "minimal oak")
    assert "four views of the SAME object" in p and "2x2 grid" in p and "front view" in p
    assert "top view" in p and "Style: minimal oak." in p
    s = concept_prompt("lamp", "product", "set")
    assert "three variations" in s and "left, middle, right" in s
    assert "one view only" in concept_prompt("lamp").lower()


def test_turnaround_only_for_products_and_names_are_forgiving():
    assert concept.normalize_views("turnaround", "product") == "turnaround"
    assert concept.normalize_views("turnaround", "ui") == "front"
    assert concept.normalize_views("3D", "product") == "turnaround"
    assert concept.normalize_views("nonsense", "product") == ""
    assert concept.normalize_kind("") == "product" and concept.normalize_kind("mockup") == "ui"
    assert concept.normalize_kind("video") == ""


def test_reference_and_revision_wording():
    ref = concept_prompt("lamp", "product", reference=True)
    assert "keep this object's overall shape" in ref
    assert "current page or sketch" in concept_prompt("job board", "ui", reference=True)
    rev = concept_prompt("lamp", "product", reference=True, change="make it rounder")
    assert (
        "previous version" in rev and "Apply ONLY this change" in rev and "make it rounder" in rev
    )
    assert "keep this object's overall shape" not in rev
    assert "keep this object" not in concept_prompt("lamp", "product")


# ------------------------------------------------------------------ the call
def test_generate_concept_sends_prompt_reference_and_size():
    client = fake_client(image_response(png_bytes((64, 64))))
    img, reason = concept.generate_concept(client, "m", "PROMPT", "16:9", "1K", b"\xff\xd8ref")
    assert reason == "" and img.size == (64, 64)
    model, contents, config = client.models.calls[0]
    assert (
        model == "m" and contents[-1] == "PROMPT" and contents[0].inline_data.data == b"\xff\xd8ref"
    )
    assert (config.image_config.aspect_ratio, config.image_config.image_size) == ("16:9", "1K")
    assert config.response_modalities == ["TEXT", "IMAGE"]
    assert config.temperature is None and config.top_p is None  # an API error on Nano Banana 2.1
    plain = fake_client(image_response(png_bytes((8, 8))))
    concept.generate_concept(plain, "m", "P", "1:1", "1K")
    assert plain.models.calls[0][1] == ["P"]  # no reference part


def test_generate_concept_failures_are_reasons():
    blocked = image_response(png_bytes((9, 9)), finish=types.FinishReason.IMAGE_SAFETY)
    assert concept.generate_concept(fake_client(blocked), "m", "p", "1:1", "1K") == (
        None,
        "blocked",
    )
    assert concept.generate_concept(fake_client(text_response("no")), "m", "p", "1:1", "1K") == (
        None,
        "blocked",
    )
    assert concept.generate_concept(fake_client(httpx.ReadTimeout("t")), "m", "p", "1:1", "1K") == (
        None,
        "timeout",
    )
    assert concept.generate_concept(fake_client(RuntimeError("x")), "m", "p", "1:1", "1K") == (
        None,
        "error",
    )
    assert concept.generate_concept(None, "m", "p", "1:1", "1K") == (None, "error")


# ------------------------------------------------------------------ labels and files
def changed(a: Image.Image, b: Image.Image) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of what differs, per quadrant (2x2) of the image."""
    from PIL import ImageChops

    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    w, h = a.size
    out = []
    for cy in (0, 1):
        for cx in (0, 1):
            box = diff.crop(
                (cx * w // 2, cy * h // 2, (cx + 1) * w // 2, (cy + 1) * h // 2)
            ).getbbox()
            out.append(box)
    return out


def test_labels_are_drawn_on_turnaround_and_set_only():
    fonts = cards.card_fonts()
    base = Image.new("RGB", (1024, 1024), (200, 200, 200))
    assert concept.add_labels(base, "front", "th", fonts).tobytes() == base.tobytes()
    turn = concept.add_labels(base, "turnaround", "th", fonts)
    boxes = changed(turn, base)
    assert all(boxes)  # a label in each of the four cells ...
    for x0, y0, x1, _y1 in boxes:  # ... in the lower part of the cell, centred
        assert y0 > 512 * 0.8 and abs((x0 + x1) / 2 - 256) < 20
    wide = Image.new("RGB", (1024, 576), (200, 200, 200))
    s = concept.add_labels(wide, "set", "en", fonts)
    top_left, top_right, bottom_left, bottom_right = changed(s, wide)
    assert not top_left and not top_right  # the labels sit at the bottom edge
    assert bottom_left and bottom_right and bottom_left[3] > 288 * 0.85
    assert concept.label_texts("turnaround", "en") == ["Front", "Side", "3/4 view", "Top"]
    assert concept.label_texts("set", "th") == ["ตัวเลือก 1", "ตัวเลือก 2", "ตัวเลือก 3"]
    assert concept.label_texts("front", "th") == []


def test_fit_jpeg_stays_under_the_limit():
    noise = Image.effect_noise((1400, 1400), 90).convert("RGB")
    data = concept.fit_jpeg(noise, 300_000)
    assert len(data) <= 300_000 and Image.open(io.BytesIO(data)).format == "JPEG"


# ------------------------------------------------------------------ memory
def test_memory_ttl_lru_and_key_isolation():
    now = [0.0]
    m = ConceptMemory(ttl_s=1800, max_entries=2, clock=lambda: now[0])
    last = concept.LastConcept("lamp", "product", "front", "", "", b"x", 1)
    m.put((1, 1, 1), last)
    m.put((1, 1, 2), last)
    assert m.get((1, 1, 1)) and m.get((2, 1, 1)) is None and m.get((1, 2, 1)) is None
    m.put((1, 1, 3), last)
    assert m.get((1, 1, 2)) is None and m.get((1, 1, 1))
    now[0] = 1801  # 30 minutes
    assert m.get((1, 1, 1)) is None


# ------------------------------------------------------------------ the image tool
def make_ctx(client=None, **kw):
    base = dict(
        guild_id=1,
        day="2026-10-17",
        lang="th",
        asker_id=STAFF,
        asker_name="Somchai",
        key=(1, 10, STAFF),
        image_client=client
        if client is not None
        else fake_client(
            *[image_response(png_bytes((256, 256), (40 * i, 90, 120))) for i in range(6)]
        ),
        image_model="nano",
        image_size="1K",
        budget=DailyBudget(20),
        memory=ConceptMemory(),
        repo=IdeaRepo(SQLiteDocStore(":memory:"), clock=lambda: NOW),
        fonts=cards.card_fonts(),
        today=lambda: TODAY,
    )
    return ConceptCtx(**{**base, **kw})


def tool_of(ctx, name="concept_image"):
    return next(f for f in concept_tools(ctx) if f.__name__ == name)


def test_product_image_is_attached_with_an_ascii_name_and_note():
    c = make_ctx()
    out = tool_of(c)(brief="a rounded desk lamp, brushed brass", kind="product")
    assert out["status"] == "ok" and out["kind"] == "product" and out["views"] == "front"
    assert out["file"] == "concept-a-rounded-desk-lamp-brushed-brass-product-1.jpg"
    [(name, data)] = c.outbox
    assert name == out["file"] and len(data) <= 1_500_000
    assert Image.open(io.BytesIO(data)).format == "JPEG"
    assert "concept render" in out["note"] and "not a 3D model" in out["note"]
    assert out["images_left_today"] == 19 and out["saved_in_idea"] == ""
    model, contents, config = c.image_client.models.calls[0]
    assert model == "nano" and config.image_config.image_size == "1K"


def test_turnaround_note_says_it_is_not_a_3d_model_and_has_labels():
    c = make_ctx(image_client=fake_client(image_response(png_bytes((512, 512), (230, 230, 230)))))
    out = tool_of(c)(brief="a desk lamp", views="turnaround")
    assert out["views"] == "turnaround" and "not a real 3D model" in out["note"]
    prompt = c.image_client.models.calls[0][1][-1]
    assert "2x2 grid" in prompt
    sent = Image.open(io.BytesIO(c.outbox[0][1]))
    assert all(changed(sent, Image.new("RGB", sent.size, (230, 230, 230))))  # four labels


def test_ui_is_wide_and_gets_no_labels():
    c = make_ctx(image_client=fake_client(image_response(png_bytes((640, 360)))))
    out = tool_of(c)(brief="a job board with a team calendar", kind="ui", views="turnaround")
    assert out["views"] == "front"  # a turnaround is for products
    assert c.image_client.models.calls[0][2].image_config.aspect_ratio == "16:9"


def test_input_errors_never_call_the_model():
    c = make_ctx(image_client=fake_client(RuntimeError("must not be called")))
    tool = tool_of(c)
    assert "English" in tool(brief="โคมไฟทรงกลม")["error"]
    assert "English" in tool(brief="a lamp", style="สไตล์มินิมอล")["error"]
    assert "kind must be" in tool(brief="a lamp", kind="video")["error"]
    assert "views must be" in tool(brief="a lamp", views="spin")["error"]
    assert "brief is required" in tool(brief=" ")["error"]
    assert "max 600" in tool(brief="x" * 601)["error"]
    assert c.image_client.models.calls == [] and c.outbox == []


@pytest.mark.parametrize("lang", ["th", "en"])
@pytest.mark.parametrize(
    "setup, reason",
    [
        ("cap", "daily_limit"),
        ("off", "disabled"),
        ("blocked", "blocked"),
        ("timeout", "timeout"),
        ("error", "error"),
    ],
)
def test_no_image_gives_an_honest_line(setup, reason, lang):
    budget = DailyBudget(0 if setup == "off" else 1)
    if setup == "cap":
        budget.add(1, "2026-10-17", 1)
    client = {
        "blocked": fake_client(
            image_response(png_bytes((9, 9)), finish=types.FinishReason.IMAGE_SAFETY)
        ),
        "timeout": fake_client(httpx.ReadTimeout("t")),
        "error": fake_client(RuntimeError("x")),
    }.get(setup, fake_client(RuntimeError("must not be called")))
    c = make_ctx(client, budget=budget, lang=lang)
    out = tool_of(c)(brief="a rounded lamp")
    assert (out["status"], out["reason"]) == ("unavailable", reason)
    assert out["message"] == t(f"concept_img_{reason}", lang) and out["message"]
    assert c.outbox == [] and c.memory.get(c.key) is None
    if setup in ("cap", "off"):
        assert c.image_client.models.calls == []
    else:  # the unit is taken before the call: blocked calls are billed too
        assert budget.used(1, "2026-10-17") == 1


def test_budget_counts_per_image_and_is_separate_from_promo():
    c = make_ctx(budget=DailyBudget(2))
    tool = tool_of(c)
    assert [tool(brief="a lamp")["status"] for _ in range(3)] == ["ok", "ok", "unavailable"]
    assert len(c.outbox) == 2
    from salid import promo

    assert promo.take_unit(DailyBudget(20), 1, "2026-10-17") == ""  # promo has its own counter


def test_reference_photo_of_the_message_is_passed():
    c = make_ctx(photos=[(0, png_bytes((900, 900))), (1, png_bytes((400, 400)))])
    tool_of(c)(brief="a lamp shaped like this")
    _, contents, _ = c.image_client.models.calls[0]
    assert len(contents) == 2 and contents[0].inline_data.mime_type == "image/jpeg"
    assert "keep this object's overall shape" in contents[1]
    c2 = make_ctx(photos=[(0, png_bytes((900, 900))), (1, png_bytes((400, 400)))])
    tool_of(c2)(brief="a lamp", file_index=1)
    assert Image.open(io.BytesIO(c2.image_client.models.calls[0][1][0].inline_data.data)).size == (
        400,
        400,
    )
    assert (
        tool_of(make_ctx(photos=[(0, png_bytes((900, 900)))]))(brief="a lamp", file_index=7)[
            "status"
        ]
        == "ok"
    )
    too_big = make_ctx(photos=[(0, png_bytes((900, 900)))])
    import salid.promo as promo

    old = promo.MAX_PIXELS
    promo.MAX_PIXELS = 100
    try:
        assert "smaller photo" in tool_of(too_big)(brief="a lamp")["error"]
    finally:
        promo.MAX_PIXELS = old


def test_revision_uses_the_previous_brief_the_delta_and_the_previous_image():
    c = make_ctx()
    tool = tool_of(c)
    tool(brief="a desk lamp, brushed brass", style="minimal")
    first_jpeg = c.outbox[0][1]
    out = tool(brief="make it rounder and green", revise=True)
    assert out["status"] == "ok" and out["file"].endswith("-product-2.jpg")
    _, contents, _ = c.image_client.models.calls[1]
    assert contents[0].inline_data.data == first_jpeg  # the previous image is the reference
    prompt = contents[1]
    assert "a desk lamp, brushed brass" in prompt and "Apply ONLY this change" in prompt
    assert "make it rounder and green" in prompt and "Style: minimal." in prompt
    last = c.memory.get(c.key)
    assert "brushed brass; make it rounder and green" in last.brief and last.n == 2
    assert len(c.outbox) == 2


def test_revision_without_a_recent_concept_and_after_30_minutes():
    now = [0.0]
    c = make_ctx(memory=ConceptMemory(clock=lambda: now[0]))
    assert tool_of(c)(brief="rounder", revise=True)["status"] == "no_previous"
    tool_of(c)(brief="a desk lamp")
    now[0] = 1801
    assert tool_of(c)(brief="rounder", revise=True)["status"] == "no_previous"
    other = make_ctx(memory=c.memory, key=(1, 10, STAFF + 1))
    assert tool_of(other)(brief="rounder", revise=True)["status"] == "no_previous"  # per person


def test_image_is_stored_under_the_idea_and_listed_in_markdown():
    c = make_ctx()
    out = tool_of(c)(brief="a desk lamp", idea="Desk Lamp")
    assert out["saved_in_idea"] == "Desk Lamp" and "also saved" in out["note"]
    assert out["file"] == "concept-desk-lamp-product-1.jpg"
    doc = c.repo.get(1, STAFF, "desk-lamp")
    [im] = doc.images
    assert im["name"] == out["file"] and len(im["sha256"]) == 64 and "ภาพสินค้า" in im["caption"]
    assert c.repo.image_bytes(1, doc, im)  # the bytes are in the store
    md = render_markdown(doc, "th")
    assert "## ภาพคอนเซปต์" in md and out["file"] in md
    assert "## Concept images" in render_markdown(doc, "en")
    # the next image of the same idea continues the numbering, even without memory
    c2 = make_ctx(repo=c.repo)
    assert tool_of(c2)(brief="a desk lamp", idea="Desk Lamp")["file"].endswith("-product-2.jpg")


def test_idea_images_are_capped_at_12_oldest_dropped_and_deleted_with_the_idea():
    repo = IdeaRepo(SQLiteDocStore(":memory:"), clock=lambda: NOW)
    for i in range(MAX_IMAGES + 3):
        doc = repo.add_image(
            1, STAFF, "S", "Lamp", f"concept-lamp-product-{i}.jpg", f"c{i}", bytes([i]) * 50
        )
    assert [im["name"] for im in doc.images][0] == "concept-lamp-product-3.jpg" and len(
        doc.images
    ) == 12
    assert len(repo.docs.find("salid_idea_images", 1)) == 12  # dropped bytes are deleted too
    repo.delete(1, doc)
    assert repo.docs.find("salid_idea_images", 1) == []
    with pytest.raises(ValueError):
        repo.add_image(1, STAFF, "S", "Lamp", "n.jpg", "c", b"x" * 700_001)
    with pytest.raises(ValueError):
        repo.add_image(1, STAFF, "S", " ", "n.jpg", "c", b"x")


def test_stored_copy_fits_a_firestore_document():
    c = make_ctx(image_client=fake_client(image_response(_noise_png())))
    tool_of(c)(brief="a desk lamp", idea="Lamp")
    [im] = c.repo.get(1, STAFF, "lamp").images
    raw = c.repo.docs.get(
        "salid_idea_images", 1, c.repo._image_key(c.repo.get(1, STAFF, "lamp"), im["sha256"])
    )
    assert len(raw["b64"]) < 1_000_000  # Firestore documents are limited to 1 MiB
    assert len(c.outbox[0][1]) <= 1_500_000


def _noise_png() -> bytes:
    buf = io.BytesIO()
    Image.effect_noise((1024, 1024), 120).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def test_storing_is_best_effort_and_not_without_an_idea():
    c = make_ctx(repo=None)
    out = tool_of(c)(brief="a desk lamp", idea="Lamp")
    assert (
        out["status"] == "ok" and out["saved_in_idea"] == "" and "could not be saved" in out["note"]
    )
    assert len(c.outbox) == 1


def ideas_ctx(repo, **kw):
    return IdeasCtx(
        guild_id=1, guild_name="บริษัท A", asker_id=STAFF, asker_name="Somchai", is_exec=False,
        repo=repo, client=None, model="m", lang="th", now=lambda: NOW,
        fonts=lambda: find_fonts(os.environ.get("SALID_PDF_FONT", "")), **kw,
    )  # fmt: skip


def test_idea_pdf_embeds_the_concept_images_one_per_page_with_captions():
    from pypdf import PdfReader

    c = make_ctx()
    tool_of(c)(brief="a desk lamp", idea="Desk Lamp")
    tool_of(c)(brief="rounder", revise=True)  # a second image of the same idea
    ictx = ideas_ctx(c.repo)
    ictx.idea_save("Desk Lamp", "## แนวคิด\n- โคมไฟตั้งโต๊ะ", "problem")
    out = ictx.idea_export("Desk Lamp", "pdf")
    assert out["status"] == "ok"
    reader = PdfReader(io.BytesIO(ictx.outbox[0][1]))
    assert len(reader.pages) >= 3  # the text page + one page per image
    for page in reader.pages[1:]:
        assert len(page.images) == 1
    texts = " ".join(p.extract_text() for p in reader.pages)
    assert "ภาพคอนเซปต์" in texts and "a desk lamp" in texts
    assert b"/Subtype /Image" in ictx.outbox[0][1]
    md = ictx.idea_export("Desk Lamp", "md")
    assert md["status"] == "ok"
    body = ictx.outbox[1][1].decode()
    assert "concept-desk-lamp-product-1.jpg" in body and "(concept-desk-lamp-product-1.jpg)" in body


def test_an_idea_with_only_images_can_be_exported_and_a_broken_image_is_skipped():
    repo = IdeaRepo(SQLiteDocStore(":memory:"), clock=lambda: NOW)
    repo.add_image(1, STAFF, "S", "Lamp", "concept-lamp-product-1.jpg", "cap", b"not an image")
    ictx = ideas_ctx(repo)
    assert (
        ictx.idea_export("Lamp", "pdf")["status"] == "ok"
    )  # the broken picture never loses the file


# ------------------------------------------------------------------ price_check
TABLE_TH = """นี่คือราคา
| แหล่ง | รุ่น/สเปก | ราคา | หน่วย | วันที่/หมายเหตุ |
|---|---|---|---|---|
| ร้าน A | ZX-100 | ฿1,290 | ชิ้น | 17 ต.ค. 2569 รวม VAT |
| ร้าน B | ZX-100 | ติดต่อสอบถาม | ชิ้น | ไม่แสดงราคา |
| ร้าน C | ZX-200 | 2,490 บาท | ชิ้น | ส่งฟรี |
| ร้าน D | ZX-100 | - | ชิ้น | |
| ร้าน C | ZX-200 | 2,490 บาท | ชิ้น | ส่งฟรี |
RANGE: ฿1,290-2,490
"""


def price_ctx(text=TABLE_TH, sources=(), **kw):
    resp = text_response(text)
    if sources:
        resp.candidates[0].grounding_metadata = types.GroundingMetadata(
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(uri=u, title=ti))
                for u, ti in sources
            ]
        )
    return make_ctx(web_client=fake_client(resp), web_model="gem", **kw)


def test_price_check_drops_rows_without_a_price_and_keeps_the_sources():
    c = price_ctx(
        sources=[("https://a.example/zx", "a.example"), ("https://b.example/zx", "b.example")]
    )
    out = tool_of(c, "price_check")(item="ZX-100 charger")
    assert out["status"] == "ok" and out["rows"] == 2 and out["date"] == "2026-10-17"
    lines = out["table"].splitlines()
    assert lines[0] == "| แหล่ง | รุ่น/สเปก | ราคา | หน่วย | วันที่/หมายเหตุ |"
    assert len(lines) == 4 and "ร้าน A" in lines[2] and "ร้าน C" in lines[3]
    assert "ร้าน B" not in out["table"] and "ร้าน D" not in out["table"]
    assert out["range"] == "฿1,290-2,490"
    rows = out["lines"].split("\n\n")[0].splitlines()
    assert (
        len(rows) == 2 and rows[0].startswith("1. ร้าน A — ") and rows[1].startswith("2. ร้าน C — ")
    )
    assert "|" not in out["lines"] and "ช่วงราคา: ฿1,290-2,490" in out["lines"]
    assert out["lines"].splitlines()[-2:] == [
        "1. a.example — https://a.example/zx",
        "2. b.example — https://b.example/zx",
    ]
    assert "`lines`" in out["instruction"] and "Markdown tables" in out["instruction"]
    assert out["sources"] == [
        {"n": 1, "title": "a.example", "link": "https://a.example/zx"},
        {"n": 2, "title": "b.example", "link": "https://b.example/zx"},
    ]
    assert "re-check" in out["instruction"] or "ตรวจซ้ำ" in out["instruction"]
    instruction = c.web_client.models.calls[0][2].system_instruction
    assert "Never estimate or invent a price" in instruction and "Thailand" in instruction
    assert "2026-10-17" in instruction and "Thai" in instruction


def test_price_check_empty_result_says_so_and_never_guesses():
    for text in (
        "NONE",
        "ขออภัย ไม่พบ",
        "| แหล่ง | รุ่น/สเปก | ราคา | หน่วย | หมายเหตุ |\n|---|---|---|---|---|\n| A | x | - | | |",
    ):
        c = price_ctx(text)
        out = tool_of(c, "price_check")(item="widget")
        assert out["status"] == "none" and out["message"] == t("concept_price_none", "th")
        assert "table" not in out
    en = price_ctx("NONE", lang="en")
    assert tool_of(en, "price_check")(item="widget")["message"].startswith("No reliable price")


def test_procure_adds_moq_and_lead_time_columns():
    text = (
        "| Source | Model / spec | Price | Unit | MOQ | Lead time | Date / note |\n|---|---|---|---|---|---|---|\n"
        "| Acme Supply | PX-5 | $4.20 | pc | 500 | 14 days | 2026-10-12 |\n"
        "| Beta | PX-5 | $3.90 | pc | 1000 | - | 2026-10-10 |\n"
    )
    c = price_ctx(text, lang="en")
    out = tool_of(c, "price_check")(
        item="PX-5 bracket", country="US", quantity="1000 pcs", purpose="procure"
    )
    assert (
        out["table"].splitlines()[0].count("|") == 8
        and "MOQ" in out["table"]
        and "Lead time" in out["table"]
    )
    assert out["rows"] == 2 and out["range"] == "$3.90-4.20"
    first = out["lines"].splitlines()[0]
    assert first == "1. Acme Supply — PX-5: $4.20 pc (MOQ 500, lead time 14 days, 2026-10-12)"
    assert out["lines"].splitlines()[1].endswith("(MOQ 1000, 2026-10-10)")
    assert "Price range: $3.90-4.20" in out["lines"]
    instruction = c.web_client.models.calls[0][2].system_instruction
    assert (
        "PROCUREMENT" in instruction
        and "1000 pcs" in instruction
        and "United States" in instruction
    )
    buy = price_ctx(
        text.replace("| MOQ | Lead time ", "")
        .replace("| 500 | 14 days ", "")
        .replace("| 1000 | - ", ""),
        lang="en",
    )
    assert "MOQ" not in tool_of(buy, "price_check")(item="PX-5")["table"]


def test_range_needs_one_currency_and_one_unit_else_only_the_model_line():
    rows = [["A", "x", "$10", "pc", "-"], ["B", "x", "฿300", "pc", "-"]]
    assert concept.price_range(rows) == ""
    assert concept.price_range([["A", "x", "฿100", "box", "-"], ["B", "x", "฿90", "pc", "-"]]) == ""
    assert concept.price_range([["A", "x", "฿100", "pc", "-"]]) == "฿100"
    assert (
        concept.price_range([["A", "x", "฿99.5", "pc", "-"], ["B", "x", "฿120", "PC", "-"]])
        == "฿99.50-120"
    )
    mixed = "| a | b | c | d | e |\n|---|---|---|---|---|\n| A | x | $10 | pc | n |\n| B | x | ฿300 | pc | n |\nRANGE: $10-฿300"
    out = tool_of(price_ctx(mixed), "price_check")(item="widget")
    assert out["range"] == "" and out["range_note"] == "$10-฿300"


def test_at_most_eight_rows():
    body = "\n".join(f"| ร้าน {i} | m | ฿{100 + i} | ชิ้น | - |" for i in range(12))
    out = tool_of(
        price_ctx("| a | b | c | d | e |\n|---|---|---|---|---|\n" + body), "price_check"
    )(item="x item")
    assert out["rows"] == 8


def test_price_check_is_saved_in_the_sourcing_section_with_the_date_and_appended():
    c = price_ctx(sources=[("https://a.example/zx", "a.example")])
    out = tool_of(c, "price_check")(item="ZX-100 charger", idea="Charger Hub")
    assert out["saved_in_idea"] == "Charger Hub"
    doc = c.repo.get(1, STAFF, "charger-hub")
    body = doc.sections["sourcing"].body
    assert doc.sections["sourcing"].title == "การจัดหาและราคา"
    assert (
        "### 2026-10-17 · ZX-100 charger" in body
        and "ร้าน A" in body
        and "ช่วงราคา: ฿1,290-2,490" in body
    )
    assert "[1] a.example - https://a.example/zx" in body
    c2 = make_ctx(
        repo=c.repo,
        web_client=fake_client(text_response(TABLE_TH)),
        web_model="g",
        today=lambda: date(2026, 10, 20),
    )
    tool_of(c2, "price_check")(item="ZX-100 charger", idea="Charger Hub")
    body = c.repo.get(1, STAFF, "charger-hub").sections["sourcing"].body
    assert "### 2026-10-17" in body and "### 2026-10-20" in body  # appended, newest last
    assert "## Sourcing & prices" in render_markdown(c.repo.get(1, STAFF, "charger-hub"), "en")


def test_price_check_errors_and_web_off():
    assert tool_of(make_ctx(), "price_check")(item="widget")["error"].startswith(
        "web search is off"
    )
    assert tool_of(price_ctx(), "price_check")(item="")["status"] == "error"
    failing = make_ctx(web_client=fake_client(RuntimeError("down")), web_model="g")
    assert tool_of(failing, "price_check")(item="widget")["status"] == "error"


def test_tools_never_raise_and_log_no_user_text(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="salid.concept")
    c = make_ctx()
    tool_of(c)(brief="secret garden lamp")
    assert (
        "secret garden" not in caplog.text and "concept image guild=1 kind=product" in caplog.text
    )
    monkeypatch.setattr(concept, "fit_jpeg", lambda *a, **k: 1 / 0)
    assert tool_of(make_ctx())(brief="a lamp") == {
        "status": "error",
        "error": "the concept image failed; try again",
    }


def test_tool_schemas():
    import inspect

    img, price = concept_tools(make_ctx())
    assert [img.__name__, price.__name__] == ["concept_image", "price_check"]
    assert list(inspect.signature(img).parameters) == [
        "brief", "kind", "views", "style", "idea", "revise", "file_index",
    ]  # fmt: skip
    assert list(inspect.signature(price).parameters) == [
        "item",
        "country",
        "quantity",
        "purpose",
        "idea",
    ]
    assert "IN ENGLISH" in img.__doc__ and "NOT a 3D model" in img.__doc__
    assert "turnaround" in img.__doc__ and "revise" in img.__doc__
    assert [f.__name__ for f in concept_tools(make_ctx(), images=False)] == ["price_check"]
    c = make_ctx()
    tools = concept_tools(c)
    assert all(f.outbox is c.outbox for f in tools)
    assert qa.tool_files(tools) == c.outbox


# ------------------------------------------------------------------ the bot glue
def names(tools):
    return {f.__name__ for f in tools}


def chat_msg(world, member):
    return TMsg(world.g, world.pub, member, "ช่วยคิดคอนเซปต์")


async def test_tools_for_chat_voice_flags_and_missing_pieces(bot, world):
    m = world.staff
    msg = chat_msg(world, m)
    assert bot._concept_tools(msg, world.g, m, [], "th") == []  # no model client
    bot.llm = fake_client()
    assert names(bot._concept_tools(msg, world.g, m, [], "th")) == {"concept_image", "price_check"}
    assert names(bot._concept_tools(msg, world.g, m, None, "th")) == {
        "concept_image",
        "price_check",
    }
    # voice: the stand-in message has `room`; an image cannot be spoken
    room = bot.room_message(world.g, m, world.pub, room={m.id})
    assert names(bot._concept_tools(room, world.g, m, [], "th")) == {"price_check"}
    for off in ("concept", "ideas"):
        bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={off: False}), "x")
        bot.team._cache.clear()
        assert bot._concept_tools(msg, world.g, m, [], "th") == [], off
        bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={off: True}), "x")
        bot.team._cache.clear()
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"web": False}), "x")
    bot.team._cache.clear()
    assert names(bot._concept_tools(msg, world.g, m, [], "th")) == {
        "concept_image"
    }  # no web: no prices


async def test_the_tools_are_bound_to_guild_channel_author_and_their_own_budget(bot, world):
    m = world.staff
    msg = chat_msg(world, m)
    bot.llm = fake_client(image_response(png_bytes((256, 256))))
    img = next(
        f for f in bot._concept_tools(msg, world.g, m, [], "th") if f.__name__ == "concept_image"
    )
    out = img(brief="a rounded desk lamp", idea="Lamp")
    assert out["status"] == "ok" and qa.tool_files([img]) == img.outbox
    day = bot.s.today().isoformat()
    assert bot.concept_budget.used(GUILD, day) == 1 and bot.promo_budget.used(GUILD, day) == 0
    assert bot.concept_budget is not bot.promo_budget
    assert bot.concept_memory.get((GUILD, world.pub.id, m.id)) is not None
    assert bot.ideas.get(GUILD, m.id, "lamp").images
    assert bot.llm.models.calls[0][0] == "gemini-nano-banana-2.1"


async def test_agent_kwargs_add_the_tools_for_chat_not_interjections_and_voice_has_prices_only(
    bot, world
):
    m = world.staff
    msg = chat_msg(world, m)
    bot.llm = fake_client()
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert {"concept_image", "price_check"} <= names(
        bot._agent_kwargs(msg, world.g, m, False, **kw)["extra_tools"]
    )
    inter = bot._agent_kwargs(msg, world.g, m, False, interjection=True, **kw)
    assert not {"concept_image", "price_check"} & names(inter.get("extra_tools", []))
    room = bot.room_message(world.g, m, world.pub, room={m.id})
    voice = names(bot._agent_kwargs(room, world.g, m, False, files=[], **kw)["extra_tools"])
    assert "price_check" in voice and "concept_image" not in voice
    assert "concept_image" not in ACTION_TOOL_NAMES  # nothing to re-bind: there is no voice tool


async def test_ask_picks_the_prompt_section_from_the_tools_and_attaches_the_image(bot, world):
    m = world.staff
    bot.llm = fake_client(image_response(png_bytes((256, 256))))
    tools = bot._concept_tools(chat_msg(world, m), world.g, m, [], "th")
    seen = {}

    def first(req):
        seen["prompt"] = req.config.system_instruction
        return fc_resp(("concept_image", {"brief": "a rounded desk lamp", "views": "turnaround"}))

    llm = scripted(first, text_resp("แนบภาพให้แล้วค่ะ"))
    ans = await qa.ask_async(
        bot.s, bot.store, "ออกแบบโคมไฟให้หน่อย", guild_id=GUILD, is_exec=False, user_id=m.id,
        model=llm, extra_tools=tools,
    )  # fmt: skip
    assert "Concept studio" in seen["prompt"] and "revise=true" in seen["prompt"]
    assert [n for n, _ in ans.attachments] == ["concept-a-rounded-desk-lamp-product-1.jpg"]
    plain = scripted(lambda req: seen.update(p2=req.config.system_instruction) or text_resp("ok"))
    await qa.ask_async(
        bot.s, bot.store, "สวัสดี", guild_id=GUILD, is_exec=False, user_id=m.id, model=plain
    )
    assert "Concept studio" not in seen["p2"]


def test_flag_defaults_on_and_the_prompt_section_follows_the_tools():
    assert "concept" in FEATURES and Features().concept is True
    from salid.config import Settings

    s = Settings(_env_file=None)
    assert (s.concept_images_per_day, s.concept_image_size) == (20, "1K")
    on = qa.system_prompt(s, scoped=False, concept=True)
    off = qa.system_prompt(s, scoped=False)
    for needle in ("Concept studio", "concept_image(", "price_check(", "turnaround", "revise=true"):
        assert needle in on and needle not in off, needle
    assert "business_framework(kind=product_concept)" in on and "ไม่เกิน 2 คำถาม" in on
    assert "ไม่ใช่โมเดล 3 มิติ" in qa.CAPABILITIES_PROMPT and "price_check" in qa.CAPABILITIES_PROMPT
    assert "Honesty:" in on and "ห้องเสียง" in on


def test_i18n_lines_exist_in_both_languages():
    keys = [k for k in CATALOG if k.startswith("concept_")]
    assert len(keys) >= 17
    for k in keys:
        assert CATALOG[k]["th"] and CATALOG[k]["en"], k
    for reason in ("disabled", "daily_limit", "blocked", "timeout", "error"):
        assert "ค่ะ" in t(f"concept_img_{reason}", "th") or "ค่ะ" in t(f"concept_img_{reason}", "th")
        assert t(f"concept_img_{reason}", "en").isascii()
    assert t("concept_lbl_option", "en", n=2) == "Option 2"
    assert "2026-10-17" in t("concept_price_recheck", "en", date="2026-10-17")
    assert all(f"concept_kind_{k}" in CATALOG for k in concept.IMAGE_KINDS)


def test_concept_prompt_tells_the_model_to_relay_lines_not_the_table():
    from salid.qa import CONCEPT_PROMPT

    assert "lines" in CONCEPT_PROMPT and "Markdown" in CONCEPT_PROMPT
    assert "cheerful" not in CONCEPT_PROMPT.lower() and "ร่าเริง" not in CONCEPT_PROMPT
