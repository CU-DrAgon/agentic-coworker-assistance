"""Sample media for the live media checks (`scripts/salid_media_probe.py`, the `live` tests):
drawn in code, so no file and nothing personal is sent to a model."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def cup_png(size: tuple[int, int] = (720, 1280)) -> bytes:
    """A plain drawing of a white cup on a table, as PNG: the still a hero clip starts from."""
    w, h = size
    img = Image.new("RGB", size, (190, 150, 110))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(h * 0.62), w, h), fill=(120, 80, 50))
    d.rounded_rectangle(
        (int(w * 0.33), int(h * 0.40), int(w * 0.62), int(h * 0.64)), radius=24, fill="white"
    )
    d.ellipse(
        (int(w * 0.58), int(h * 0.45), int(w * 0.72), int(h * 0.56)), outline="white", width=14
    )
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
