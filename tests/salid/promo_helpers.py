"""Shared helpers for the promo kit tests: images, Gemini responses and a fake client."""

import io
import json
import subprocess

from conftest import FakeClient
from google.genai import types
from PIL import Image

from salid import promo


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def png_bytes(size: tuple[int, int], color=(120, 140, 160)) -> bytes:
    return to_png(Image.new("RGB", size, color))


def jpeg_with_orientation(size: tuple[int, int], orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 100, 50)).save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def photo() -> promo.Photo:
    return promo.prepare_photo(png_bytes((800, 800)))


def _response(parts, finish) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)
        ]
    )


def image_response(png: bytes, finish=types.FinishReason.STOP) -> types.GenerateContentResponse:
    return _response([types.Part.from_bytes(data=png, mime_type="image/png")], finish)


def text_response(text: str) -> types.GenerateContentResponse:
    return _response([types.Part(text=text)], types.FinishReason.STOP)


def json_response(obj) -> types.GenerateContentResponse:
    return text_response(json.dumps(obj, ensure_ascii=False))


def _raiser(exc: BaseException):
    def call(contents, config):
        raise exc

    return call


def fake_client(*results) -> FakeClient:
    """A client whose generate_content returns (or raises) `results` in order."""
    return FakeClient([_raiser(r) if isinstance(r, BaseException) else r for r in results])


def probe(path) -> dict:
    """ffprobe of a media file: {"duration": seconds, "codecs": [...], "size": (w, h)} (size is
    (None, None) without a video stream)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if "width" in s), {})
    return {
        "duration": float(data["format"]["duration"]),
        "codecs": [s["codec_name"] for s in streams],
        "size": (video.get("width"), video.get("height")),
    }
