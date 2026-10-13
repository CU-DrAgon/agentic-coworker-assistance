"""Audio helpers: normalise browser recordings (webm/ogg/mp4/...) to 16 kHz mono FLAC."""

import logging
import subprocess

logger = logging.getLogger(__name__)

FFMPEG_ARGS = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
    "-i", "pipe:0",
    "-vn", "-ac", "1", "-ar", "16000",
    "-f", "flac", "pipe:1",
]  # fmt: skip


class AudioConversionError(RuntimeError):
    """ffmpeg could not decode or encode the audio."""


def to_flac(data: bytes) -> bytes:
    """Convert any ffmpeg-readable audio to 16 kHz mono FLAC, using stdin/stdout only."""
    if not data:
        raise AudioConversionError("empty audio payload")
    try:
        proc = subprocess.run(FFMPEG_ARGS, input=data, capture_output=True, timeout=120)
    except FileNotFoundError as e:
        raise AudioConversionError("ffmpeg is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise AudioConversionError("ffmpeg timed out") from e
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip()[-500:]
        raise AudioConversionError(f"ffmpeg failed (exit {proc.returncode}): {err}")
    logger.info("converted %d bytes -> %d bytes FLAC", len(data), len(proc.stdout))
    return proc.stdout
