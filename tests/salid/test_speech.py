"""Spoken replies: shortening, TTS parsing and the playable source. Fully offline."""

from types import SimpleNamespace

import pytest
from conftest import FakeClient, text_resp
from google.genai import errors

from salid import speech
from salid.speech import (
    SPOKEN_MAX_CHARS,
    clip,
    fallback_text,
    spoken_text,
    strip_markdown,
    synthesize,
)

PCM = b"\x01\x02" * 100
MIME = "audio/L16;codec=pcm;rate=24000"


def audio_resp(data=PCM, mime=MIME):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime), text=None)
    return SimpleNamespace(
        text=None, candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
    )


def api_error(code):
    return errors.APIError(code, {"error": {"message": "x", "status": "X"}})


# ------------------------------------------------------------------ text
def test_strip_markdown_removes_noise():
    raw = "**สรุป**: ดูที่ [รายงาน](https://x.co/a) หรือ https://y.co 🙂\n- ผัก 5,000 บาท\n- เนื้อ `ok`"
    out = strip_markdown(raw)
    assert out == "สรุป : ดูที่ รายงาน หรือ ผัก 5,000 บาท เนื้อ ok"
    assert "http" not in out and "*" not in out and "🙂" not in out


def test_clip_prefers_whitespace_and_respects_the_limit():
    text = "คำ " * 200
    out = clip(text, 50)
    assert len(out) <= 50 and not out.endswith(" ") and out.endswith("คำ")


def test_clip_never_strands_a_thai_combining_mark():
    # No spaces at all: a hard cut must not land right before a vowel/tone mark.
    text = "เดือนนี้จ่ายค่าวัตถุดิบไปทั้งหมดสิบสองพันบาท" * 20
    for n in range(10, 120):
        out = clip(text, n)
        assert len(out) <= n
        nxt = text[len(out)]
        assert not speech.unicodedata.category(nxt).startswith("M"), (n, out[-3:], nxt)


def test_clip_leaves_short_text_alone():
    assert clip("สั้น ๆ", 50) == "สั้น ๆ"


def test_fallback_text_is_stripped_and_bounded():
    out = fallback_text("**ก** " + "ข" * 1000)
    assert out.startswith("ก ข") and len(out) <= SPOKEN_MAX_CHARS


# ------------------------------------------------------------------ spoken_text
def test_spoken_text_uses_the_model_reply_and_the_prompt_has_both_parts():
    client = FakeClient([text_resp("เดือนนี้จ่ายหมื่นสองพันสี่ร้อยห้าสิบบาทครับ")])
    out = spoken_text(client, "flash", "ใช้ไปเท่าไร", "**12,450** บาท\n- ผัก")
    assert out == "เดือนนี้จ่ายหมื่นสองพันสี่ร้อยห้าสิบบาทครับ"
    model, contents, _ = client.models.calls[0]
    assert model == "flash" and "ใช้ไปเท่าไร" in contents[0] and "**12,450** บาท" in contents[0]


def test_spoken_text_clips_an_overlong_reply():
    client = FakeClient([text_resp("ก " * 500)])
    assert len(spoken_text(client, "m", "q", "a")) <= SPOKEN_MAX_CHARS


@pytest.mark.parametrize("script", [[api_error(400)], [text_resp("  ")], [text_resp(None)]])
def test_spoken_text_falls_back_to_the_plain_answer(script):
    def boom(contents, config):
        raise script[0]

    item = boom if isinstance(script[0], Exception) else script[0]
    client = FakeClient([item])
    long = "**ผัก** " + "ข" * 600
    out = spoken_text(client, "m", "q", long)
    assert out == fallback_text(long) and len(out) <= SPOKEN_MAX_CHARS and "*" not in out


# ------------------------------------------------------------------ synthesize
def test_synthesize_returns_raw_pcm_and_asks_for_audio_with_the_voice():
    client = FakeClient([audio_resp()])
    assert synthesize(client, "gemini-2.5-flash-tts", "Kore", "สวัสดีครับ") == PCM
    model, contents, config = client.models.calls[0]
    assert model == "gemini-2.5-flash-tts" and contents == "สวัสดีครับ"
    assert config.response_modalities == ["AUDIO"]
    voice = config.speech_config.voice_config.prebuilt_voice_config.voice_name
    assert voice == "Kore"


@pytest.mark.parametrize(
    "resp",
    [
        text_resp("no audio here"),
        SimpleNamespace(text=None, candidates=[]),
        SimpleNamespace(text=None, candidates=None),
        audio_resp(data=b""),
    ],
)
def test_synthesize_raises_without_audio(resp):
    with pytest.raises(RuntimeError, match="no audio"):
        synthesize(FakeClient([resp]), "m", "Kore", "x")


@pytest.mark.parametrize("mime", ["audio/mp3", "audio/L16;codec=pcm;rate=16000"])
def test_synthesize_rejects_formats_it_cannot_play(mime):
    with pytest.raises(RuntimeError, match="format"):
        synthesize(FakeClient([audio_resp(mime=mime)]), "m", "Kore", "x")


def test_synthesize_retries_a_transient_error(monkeypatch):
    monkeypatch.setattr("salid.llm.time.sleep", lambda s: None)

    def flaky(contents, config):
        raise api_error(503)

    client = FakeClient([flaky, audio_resp()])
    assert synthesize(client, "m", "Kore", "x") == PCM


# ------------------------------------------------------------------ pcm_source
def test_pcm_source_pipes_24k_mono_into_ffmpeg(monkeypatch):
    seen = {}

    def fake(source, **kw):
        seen.update(data=source.read(), **kw)
        return "audio-source"

    monkeypatch.setattr(speech.discord, "FFmpegPCMAudio", fake)
    assert speech.pcm_source(PCM) == "audio-source"
    assert seen["data"] == PCM and seen["pipe"] is True
    assert seen["before_options"] == "-f s16le -ar 24000 -ac 1"


def test_pcm_duration():
    assert speech.pcm_duration_s(bytes(48000)) == 1.0


# ------------------------------------------------------------------ Cloud TTS backend + pitch
def _wav(pcm: bytes, rate: int = 24000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _fake_cloud(monkeypatch, status=200, audio=None):
    import base64

    import google.auth
    import httpx

    creds = SimpleNamespace(token="tok", refresh=lambda req: None)
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (creds, "p"))
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json, headers=headers)
        body = {"audioContent": base64.b64encode(audio).decode()} if audio is not None else {}
        return SimpleNamespace(status_code=status, json=lambda: body)

    monkeypatch.setattr(httpx, "post", post)
    return sent


def test_cloud_tts_returns_raw_pcm_and_uses_the_chirp_voice(monkeypatch):
    import salid.speech as sp

    sent = _fake_cloud(monkeypatch, audio=_wav(PCM))
    assert sp.synthesize_cloud("proj", "Leda", "สวัสดี", "th", 1.1) == PCM
    assert sent["json"]["voice"] == {"languageCode": "th-TH", "name": "th-TH-Chirp3-HD-Leda"}
    assert sent["json"]["audioConfig"]["sampleRateHertz"] == 24000
    assert sent["headers"]["x-goog-user-project"] == "proj"
    sent_en = _fake_cloud(monkeypatch, audio=_wav(PCM))
    sp.synthesize_cloud("proj", "Leda", "hi", "en")
    assert sent_en["json"]["voice"] == {"languageCode": "en-US", "name": "en-US-Chirp3-HD-Leda"}


def test_cloud_tts_errors_raise(monkeypatch):
    import salid.speech as sp

    _fake_cloud(monkeypatch, status=403, audio=b"")
    with pytest.raises(RuntimeError, match="403"):
        sp.synthesize_cloud("p", "Leda", "x")
    _fake_cloud(monkeypatch, audio=_wav(PCM, rate=16000))
    with pytest.raises(RuntimeError, match="format"):
        sp.synthesize_cloud("p", "Leda", "x")


def test_say_falls_back_to_gemini_tts(monkeypatch):
    import salid.speech as sp

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(sp, "synthesize_cloud", boom)
    monkeypatch.setattr(sp, "synthesize", lambda c, m, v, t: b"gemini")
    st = SimpleNamespace(
        tts_backend="cloud", gcp_project="p", cloud_tts_voice="Leda", tts_speaking_rate=1.0,
        tts_model="gemini-2.5-flash-tts", tts_voice="Leda",
    )  # fmt: skip
    assert sp.say(st, None, "x") == b"gemini"
    st.tts_backend = "gemini"
    assert sp.say(st, None, "x") == b"gemini"


def test_pcm_source_pitch_filter(monkeypatch):
    import salid.speech as sp

    seen = []
    monkeypatch.setattr(sp.discord, "FFmpegPCMAudio", lambda src, **kw: seen.append(kw) or "src")
    sp.pcm_source(PCM)
    sp.pcm_source(PCM, pitch=1.06)
    assert seen[0]["options"] is None
    assert "asetrate=25440" in seen[1]["options"] and "atempo=0.9434" in seen[1]["options"]


def test_speakable_drops_symbols_tts_reads_aloud():
    import salid.speech as sp

    assert sp.speakable("เมี้ยว~ แป๊บนึงน้า") == "เมี้ยว แป๊บนึงน้า"
    assert sp.speakable("**12,450 บาท** (8 ใบ) ~ เก่งมาก!") == "12,450 บาท 8 ใบ เก่งมาก!"
    assert sp.speakable("งบโฆษณา 80% ของเดือน 10/2026") == "งบโฆษณา 80% ของเดือน 10/2026"
    acks = "".join(__import__("salid.meeting", fromlist=["ACK_TEXT"]).ACK_TEXT.values())
    assert "~" not in acks and "เมี้ยว" not in acks


def test_say_never_sends_a_tilde_to_the_engine(monkeypatch):
    import salid.speech as sp

    got = []
    monkeypatch.setattr(
        sp, "synthesize_cloud", lambda p, v, t, lang="th", rate=1.0: got.append(t) or b"x"
    )
    st = SimpleNamespace(
        tts_backend="cloud", gcp_project="p", cloud_tts_voice="Leda", tts_speaking_rate=1.0
    )
    sp.say(st, None, "เมี้ยว~ แป๊บนึงน้า~~")
    assert got == ["เมี้ยว แป๊บนึงน้า"]
