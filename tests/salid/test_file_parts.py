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


def test_a_large_gif_frame_is_shrunk_before_it_is_sent():
    parts, bad = qa.file_parts([qa.FileInput(0, "a.gif", "image/gif", data=gif((3000, 2000)))])
    size = Image.open(io.BytesIO(parts[1].inline_data.data)).size
    assert bad == [] and max(size) == qa.MAX_FRAME_SIDE


def test_converted_gifs_respect_the_total_size_cap(monkeypatch):
    monkeypatch.setattr(qa, "MAX_TOTAL_BYTES", 200)  # the PNG is bigger than the original GIF
    f = qa.FileInput(0, "a.gif", "image/gif", data=gif((400, 400)))
    parts, bad = qa.file_parts([f])
    assert parts == [] and bad == [f] and "total size" in f.skip


def test_only_real_gifs_are_decoded():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "JPEG")  # a JPEG named .gif
    f = qa.FileInput(0, "photo.gif", "image/gif", data=buf.getvalue())
    parts, bad = qa.file_parts([f])
    assert parts == [] and bad == [f] and f.skip == "could not be read"


def test_bomb_sized_gifs_get_the_too_large_reason(monkeypatch):
    # Pillow itself refuses canvases over 2 x MAX_IMAGE_PIXELS at open (DecompressionBombError)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100_000)
    f = qa.FileInput(0, "bomb.gif", "image/gif", data=gif((1200, 1200)))
    parts, bad = qa.file_parts([f])
    assert bad == [f] and "megapixels" in f.skip


def test_the_size_check_runs_before_any_pixel_is_decoded(monkeypatch):
    f = qa.FileInput(0, "big.gif", "image/gif", data=gif((1200, 1200)))
    monkeypatch.setattr(qa, "MAX_DECODE_PIXELS", 1_000_000)
    monkeypatch.setattr(Image.Image, "load", lambda self: (_ for _ in ()).throw(AssertionError))
    qa.file_parts([f])
    assert "megapixels" in f.skip  # refused from the header, never decoded


def test_a_huge_gif_is_refused_before_decoding(monkeypatch):
    # a tiny file can declare a huge canvas; decoding it could take the shared bot's memory
    monkeypatch.setattr(qa, "MAX_DECODE_PIXELS", 1_000_000)
    big = qa.FileInput(0, "big.gif", "image/gif", data=gif((1200, 1200)))
    parts, bad = qa.file_parts([big])
    assert parts == [] and bad == [big]
    assert "megapixels" in big.skip
