import io
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

from salid.config import Settings
from salid.extract import extract_receipt
from salid.llm import make_client

pytestmark = pytest.mark.live


def make_receipt_png() -> bytes:
    img = Image.new("RGB", (520, 560), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default(size=22)
    except TypeError:
        f = ImageFont.load_default()
    lines = [
        "MAKRO SUKHUMVIT BRANCH",
        "TAX INVOICE / RECEIPT",
        "Date: 12/10/2026",
        "-" * 32,
        "Ice 10kg x 2        100.00",
        "Pork belly 3kg      540.00",
        "Cooking oil 5L      325.00",
        "-" * 32,
        "Subtotal           965.00",
        "VAT 7%             67.55",
        "TOTAL THB        1032.55",
        "Paid: CASH",
    ]
    for i, t in enumerate(lines):
        d.text((20, 20 + i * 38), t, fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_live_extract():
    s = Settings.from_env()
    t = time.monotonic()
    ex = extract_receipt(make_client(s), s, make_receipt_png(), "image/png", "ค่าวัตถุดิบ สาขา 2")
    print(f"\n{time.monotonic() - t:.1f}s\n{ex.model_dump_json(indent=2, ensure_ascii=False)}")
    assert ex.is_receipt and "makro" in (ex.vendor or "").lower()
    assert ex.total == pytest.approx(1032.55, abs=0.01)
    assert ex.receipt_date == "2026-10-12"
    assert ex.category == "วัตถุดิบ"
