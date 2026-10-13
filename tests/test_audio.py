import shutil
import subprocess

import pytest

from meeting_scribe.audio import AudioConversionError, to_flac

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone(fmt: str, rate: int = 44100, channels: int = 2) -> bytes:
    cmd = [
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration=1:sample_rate={rate}",
        "-ac", str(channels), "-f", fmt, "pipe:1",
    ]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _probe(data: bytes) -> dict[str, str]:
    cmd = [
        "ffprobe", "-loglevel", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "default=nw=1", "-i", "pipe:0",
    ]  # fmt: skip
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout.decode()
    return dict(line.split("=") for line in out.split())


@pytest.mark.parametrize("fmt", ["wav", "ogg", "matroska"])
def test_to_flac_normalises_format(fmt: str) -> None:
    flac = to_flac(_tone(fmt))
    assert flac[:4] == b"fLaC"
    info = _probe(flac)
    assert info["codec_name"] == "flac"
    assert info["sample_rate"] == "16000"
    assert info["channels"] == "1"


def test_to_flac_rejects_empty() -> None:
    with pytest.raises(AudioConversionError):
        to_flac(b"")


def test_to_flac_rejects_garbage() -> None:
    with pytest.raises(AudioConversionError):
        to_flac(b"this is not audio" * 100)
