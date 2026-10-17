"""Shared helpers for the promo kit tests: images, Gemini responses and a fake client."""

import io
import json
import random
import subprocess
from pathlib import Path

from conftest import FakeClient
from google.genai import types
from PIL import Image

from salid import cards, promo, videoad


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


def tiny_scene_assets(
    tmp_path, n: int, seconds: float, noisy: bool = False
) -> tuple[list[Path], list[Path]]:
    """`n` real motion clips of `seconds` each and their caption overlays (made without the AI
    label, as `assemble` expects). The stills are flat gray, or with `noisy` random colour noise
    that no encoder can squeeze, so a small budget forces the bitrate down."""
    fonts = cards.card_fonts()
    clips, caps = [], []
    for i in range(n):
        if noisy:
            side = (540, 960)
            data = random.Random(i).randbytes(side[0] * side[1] * 3)
            img = Image.frombytes("RGB", side, data).resize((1080, 1920))
        else:
            img = Image.new("RGB", (1080, 1920), (128, 128, 128))
        still, clip, cap = (tmp_path / f"scene{i}.{ext}" for ext in ("png", "mp4", "cap.png"))
        img.save(still)
        videoad.motion_clip(still, seconds, i, clip)
        videoad.caption_png(f"ฉากที่ {i + 1}", fonts, label=False)[0].save(cap)
        clips.append(clip)
        caps.append(cap)
    return clips, caps


def silent_wav(tmp_path, seconds: float) -> Path:
    """A silent 48 kHz mono WAV of `seconds`, made by ffmpeg's anullsrc."""
    out = tmp_path / "silence.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            f"{seconds:.3f}",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return out


def video_frame(path, t: float, out) -> Image.Image:
    """The video frame of `path` at `t` seconds, as RGB."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    with Image.open(out) as im:
        return im.convert("RGB")
