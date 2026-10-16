"""Attachments as model input: GIFs are decoded in-process (first frame), so their size is capped."""

import io

from PIL import Image

from salid import qa


def gif(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("P", size).save(buf, "GIF")
    return buf.getvalue()


def test_a_gif_becomes_its_first_frame_as_png():
    parts, bad = qa.file_parts([qa.FileInput(0, "a.gif", "image/gif", data=gif((64, 48)))])
    assert bad == [] and parts[1].inline_data.mime_type == "image/png"
    assert Image.open(io.BytesIO(parts[1].inline_data.data)).size == (64, 48)


def test_a_huge_gif_is_refused_before_decoding(monkeypatch):
    # a tiny file can declare a huge canvas; decoding it could take the shared bot's memory
    monkeypatch.setattr(qa, "MAX_DECODE_PIXELS", 1_000_000)
    big = qa.FileInput(0, "big.gif", "image/gif", data=gif((1200, 1200)))
    parts, bad = qa.file_parts([big])
    assert parts == [] and bad == [big]
    assert "megapixels" in big.skip
