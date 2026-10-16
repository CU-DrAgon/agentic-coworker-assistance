"""Live voice evaluation: real Cloud TTS audio -> the real meeting pipeline (Segmenter, Gemini
transcription, wake / engaged / dismiss / leave logic, classic spoken answers) and a REAL Gemini
Live session, all with fake Discord objects (fake VoiceClient that records play / stop calls).

Nothing is posted anywhere: the "text chat" is a list. SQLite only (temp copy of the dev ledger).
Entry point: `run_voice_suite(report, only=None)`; live-call counters are in `LIVE_CALLS`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

LIVE_CALLS = {"gemini": 0, "tts": 0, "live": 0}
FRAME = 3840  # 20 ms of 48 kHz stereo s16le (what MeetingSink delivers)
TTS_DIR = Path(tempfile.gettempdir()) / "salid_eval_tts"
LEDGER_SRC = Path("/home/ubuntu/.salid-dev/ledger.db")
VOICES = {"A": "th-TH-Chirp3-HD-Charon", "B": "th-TH-Chirp3-HD-Kore"}
GUILD_ID, CHAN_ID = 5000, 77
ME = SimpleNamespace(id=999, mention="<@999>")


# ------------------------------------------------------------------ audio synthesis
def _ffmpeg_to_48k_stereo(pcm24k_mono: bytes) -> bytes:
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
         "-f", "s16le", "-ar", "48000", "-ac", "2", "pipe:1"],
        input=pcm24k_mono, capture_output=True, check=True,
    )  # fmt: skip
    return p.stdout


def synth(text: str, voice: str = "A") -> bytes:
    """48 kHz stereo s16le PCM of `text` spoken by a Thai Chirp 3 HD voice (cached on disk)."""
    name = VOICES.get(voice, voice)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    f = TTS_DIR / f"{name}_{hashlib.sha1(text.encode()).hexdigest()[:12]}.pcm"
    if f.exists():
        return f.read_bytes()
    import google.auth
    import google.auth.transport.requests
    import httpx

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    body = {
        "input": {"text": text},
        "voice": {"languageCode": "th-TH", "name": name},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
    }
    r = httpx.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        json=body,
        headers={
            "Authorization": f"Bearer {creds.token}",
            "x-goog-user-project": "meeting-scribe-hackathon",
        },
        timeout=30.0,
    )
    LIVE_CALLS["tts"] += 1
    r.raise_for_status()
    audio = base64.b64decode(r.json()["audioContent"])
    with wave.open(io.BytesIO(audio)) as w:
        raw = w.readframes(w.getnframes())
    pcm = _ffmpeg_to_48k_stereo(raw)
    f.write_bytes(pcm)
    return pcm


# ------------------------------------------------------------------ fakes
class EvalVC:
    """A voice-recv VoiceClient stand-in. `stop()` kills listening too (the real-world bug the
    code must avoid); `stop_playing()` only stops playback. `is_playing` toggles for real: a
    classic clip plays for its duration, a Live source is polled every 20 ms like discord.py."""

    def __init__(self):
        self.sink = object()
        self.listening = self.sink
        self.playing = False
        self.stop_calls = self.stop_playing_calls = 0
        self.plays: list[dict] = []  # {t, kind, first_audio, dur}
        self._halt = threading.Event()
        self.disconnected = False
        self._connection = None

    def is_connected(self):
        return not self.disconnected

    def is_playing(self):
        return self.playing

    def listen(self, sink):
        self.listening = sink

    def stop_listening(self):
        self.listening = None

    def stop(self):
        self.stop_calls += 1
        self.listening = None  # voice_recv: stop() also stops receiving
        self._halt.set()

    def stop_playing(self):
        self.stop_playing_calls += 1
        self._halt.set()

    async def disconnect(self, force=False):
        self.disconnected = True
        self.listening = None
        self._halt.set()

    def play(self, source, after=None):
        if self.playing:
            raise RuntimeError("Already playing audio.")
        self.playing = True
        self._halt.clear()
        rec = {"t": time.monotonic(), "first_audio": None, "dur": 0.0,
               "kind": "live" if hasattr(source, "buffered_bytes") else "clip"}  # fmt: skip
        self.plays.append(rec)

        def run():
            err = None
            try:
                if rec["kind"] == "clip":
                    rec["first_audio"] = time.monotonic()
                    end = rec["t"] + source.duration
                    while time.monotonic() < end and not self._halt.is_set():
                        time.sleep(0.02)
                else:
                    from salid.live_voice import SILENCE

                    while not self._halt.is_set():
                        data = source.read()
                        if data == b"":
                            break
                        if data != SILENCE and rec["first_audio"] is None:
                            rec["first_audio"] = time.monotonic()
                        time.sleep(0.02)
                    with contextlib.suppress(Exception):
                        source.cleanup()
            except Exception as e:  # pragma: no cover
                err = e
            rec["dur"] = time.monotonic() - rec["t"]
            self.playing = False
            if after:
                after(err)

        threading.Thread(target=run, daemon=True).start()


class Chan:
    def __init__(self, id, name="ห้องประชุม"):
        self.id, self.name, self.members, self.sent = id, name, [], []
        self.guild = None

    def permissions_for(self, _):
        return SimpleNamespace(connect=True, view_channel=True, send_messages=True)

    async def send(self, text, **_):
        self.sent.append(text)
        return SimpleNamespace(jump_url="https://example.invalid/x")


class Guild:
    def __init__(self):
        self.id, self.name, self.me, self.owner_id = GUILD_ID, "EvalGuild", ME, 1
        self.members, self.text_channels, self.voice_channels, self.roles = {}, [], [], []

    def get_member(self, uid):
        return self.members.get(uid)

    def get_channel(self, cid):
        return None


def _member(guild, uid, name, channel):
    m = SimpleNamespace(
        id=uid, guild=guild, display_name=name, mention=f"<@{uid}>", bot=False, roles=[],
        voice=SimpleNamespace(channel=channel),
        guild_permissions=SimpleNamespace(administrator=False, manage_guild=False),
    )  # fmt: skip
    guild.members[uid] = m
    return m


# ------------------------------------------------------------------ the rig
class Rig:
    """One meeting in one mode ("classic" or "live") with real models."""

    def __init__(self, mode: str):
        self.mode = mode
        self.events: list[dict] = []  # transcripts, engaged changes, ends
        self._undo: list = []

    def _patch(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self._undo.append((obj, name, old))

    async def open(self):
        from google.genai import models as gm

        from salid import meeting as mt
        from salid.bot import SecretaryBot
        from salid.config import Settings
        from salid.guild_settings import MemoryGuildStore
        from salid.llm import make_client
        from salid.speech import pcm_duration_s
        from salid.store import SQLiteStore

        self.tmp = Path(tempfile.mkdtemp(prefix="salid_voice_eval_"))
        db = self.tmp / "ledger.db"
        if LEDGER_SRC.exists():
            shutil.copy(LEDGER_SRC, db)
        s = Settings(
            store="sqlite", db_path=str(db), voice_tokens=[], voice_reply=True, voice_mode=self.mode,
            meeting_text_echo=True, receipt_bucket="", gmail_client_id="", gmail_client_secret="",
            gmail_refresh_token="", gmail_sender="", exec_user_ids=set(),
        )  # fmt: skip
        self.settings = s
        llm = make_client(s)
        self._patch(SecretaryBot, "user", property(lambda self_: ME))
        self.bot = SecretaryBot(s, SQLiteStore(str(db)), llm, guilds=MemoryGuildStore())
        self.mgr = self.bot.meetings
        self.guild, self.chan, self.vc = Guild(), Chan(CHAN_ID), EvalVC()
        self.chan.guild = self.guild
        self.a = _member(self.guild, 10, "แอน", self.chan)
        self.b = _member(self.guild, 11, "บอย", self.chan)
        self.chan.members = [self.a, self.b]

        async def connect(body, channel):
            return self.vc

        self._patch(self.mgr, "_connect", connect)

        # --- instrumentation
        orig_tr = mt.transcribe_pcm

        def traced(client, model, pcm):
            t0 = time.monotonic()
            text = orig_tr(client, model, pcm)
            self.events.append(
                {
                    "kind": "heard",
                    "t": time.monotonic(),
                    "text": text,
                    "took": time.monotonic() - t0,
                }
            )
            return text

        self._patch(mt, "transcribe_pcm", traced)
        orig_say = mt.say

        def counted_say(*a, **k):
            LIVE_CALLS["tts"] += 1
            return orig_say(*a, **k)

        self._patch(mt, "say", counted_say)

        class Clip:
            def __init__(self, pcm, pitch=1.0):
                self.duration = pcm_duration_s(pcm)

            def cleanup(self):
                pass

            def read(self):
                return b""

        self._patch(mt, "pcm_source", Clip)
        orig_set = self.mgr._set_engaged

        def traced_set(sess, on):
            before = sess.engaged
            orig_set(sess, on)
            if before != on:
                self.events.append({"kind": "engaged", "t": time.monotonic(), "on": on})

        self._patch(self.mgr, "_set_engaged", traced_set)
        orig_end = self.mgr.end

        async def traced_end(key, reason="command"):
            self.events.append({"kind": "end", "t": time.monotonic(), "reason": reason})
            return await orig_end(key, reason)

        self._patch(self.mgr, "end", traced_end)
        for cls, meth in ((gm.Models, "generate_content"), (gm.Models, "generate_content_stream"),
                          (gm.AsyncModels, "generate_content"),
                          (gm.AsyncModels, "generate_content_stream")):  # fmt: skip
            self._patch(cls, meth, self._counting(getattr(cls, meth)))

        res = await self.mgr.join(self.a)
        assert "ห้องประชุม" in res, res
        self.s = self.mgr.sessions[(GUILD_ID, CHAN_ID)]
        if self.mode == "live":
            LIVE_CALLS["live"] += 1
            for _ in range(200):
                if self.s.live is not None and self.s.live._live is not None:
                    break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("Gemini Live did not connect")
            await asyncio.sleep(0.5)

    @staticmethod
    def _counting(fn):
        import inspect

        if inspect.isasyncgenfunction(fn):

            async def agen(*a, **k):
                LIVE_CALLS["gemini"] += 1
                async for x in fn(*a, **k):
                    yield x

            return agen
        if inspect.iscoroutinefunction(fn):

            async def acor(*a, **k):
                LIVE_CALLS["gemini"] += 1
                return await fn(*a, **k)

            return acor

        def sync(*a, **k):
            LIVE_CALLS["gemini"] += 1
            return fn(*a, **k)

        return sync

    async def close(self):
        if self.s.key in self.mgr.sessions:
            self.s.transcript.lines.clear()  # no summary call
            await self.mgr.end(self.s.key, "shutdown")
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- driving
    def reset_listening(self):
        self.mgr._set_engaged(self.s, False)
        self.s.speaking_until = 0.0
        self.s.pending_wake.clear()

    async def speak(self, pcm: bytes, uid: int = 10) -> float:
        """Feed the utterance at real-time pace; returns the monotonic time of its last frame."""
        t0 = time.monotonic()
        n = len(pcm) // FRAME
        for i in range(n):
            self.s.on_pcm(uid, pcm[i * FRAME : (i + 1) * FRAME], time.monotonic())
            await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.02 - time.monotonic()))
        return time.monotonic()

    def snapshot(self):
        return (len(self.events), len(self.vc.plays), len(self.chan.sent))

    async def run_step(self, step: dict) -> dict:
        """Say one utterance and watch what happens. Returns the observation dict."""
        if step.get("reset", True):
            self.reset_listening()
        while self.vc.playing:
            await asyncio.sleep(0.1)
        pcm = synth(step["text"], step.get("voice", "A"))
        ev0, pl0, tx0 = self.snapshot()
        uid = step.get("uid", 10)
        t_end = await self.speak(pcm, uid)
        expect_resp = step.get("respond")
        live = self.mode == "live"
        heard_seen = None
        last_activity = t_end
        deadline = t_end + (45.0 if expect_resp else 25.0)
        no_resp_wait = 9.0 if live else 3.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            evs = self.events[ev0:]
            heard = [e for e in evs if e["kind"] == "heard"]
            if heard and heard_seen is None:
                heard_seen = now
                last_activity = now
            plays = self.vc.plays[pl0:]
            sent = self.chan.sent[tx0:]
            if self.vc.playing or len(plays) + len(sent) + len(evs) != getattr(self, "_n", -1):
                last_activity = now
            self._n = len(plays) + len(sent) + len(evs)
            if step.get("ends") and (self.s.key not in self.mgr.sessions) and self.vc.disconnected:
                break
            if self.s.speech_pending or self.s.speaking:
                last_activity = now
            responded = bool(plays)
            if not live and heard:
                called = _called(heard[-1]["text"])
                quiet = 10.0 if (called or self.s.engaged) else no_resp_wait
            elif live:
                quiet = 4.0 if responded else no_resp_wait
            else:
                quiet = 6.0
            if (heard_seen is not None or live) and now - last_activity > quiet and now - t_end > 2:
                break
        await asyncio.sleep(0.3)
        evs = self.events[ev0:]
        plays = self.vc.plays[pl0:]
        sent = self.chan.sent[tx0:]
        heard = [e["text"] for e in evs if e["kind"] == "heard"]
        firsts = [p["first_audio"] - t_end for p in plays if p["first_audio"]]
        return {
            "t_end": t_end,
            "transcript": " | ".join(heard),
            "transitions": [
                ("engaged" if e["on"] else "listening") for e in evs if e["kind"] == "engaged"
            ],
            "ended": any(e["kind"] == "end" for e in evs),
            "responded": bool(plays),
            "first_audio_s": round(min(firsts), 2) if firsts else None,
            "answer_audio_s": round(firsts[-1], 2) if len(firsts) > 1 else None,
            "plays": [(round(p["t"] - t_end, 2), p["kind"], round(p["dur"], 2)) for p in plays],
            "text_out": [x[:160] for x in sent],
            "engaged_now": self.s.engaged,
            "transcribe_s": round(max([e["took"] for e in evs if e["kind"] == "heard"] or [0]), 2),
        }


def _called(text: str) -> bool:
    from salid.meeting import parse_wake

    return parse_wake(text or "") is not None


# ------------------------------------------------------------------ scenarios
def _weekday_th(offset=0):
    from datetime import timedelta

    from salid.config import Settings

    names = ["จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์"]
    return names[(Settings().now() + timedelta(days=offset)).weekday()]


def steps():
    today, tomorrow = _weekday_th(0), _weekday_th(1)
    return [
        dict(id="v01", text="หวัดดีครับ สลิด", respond=True, state="engaged"),
        dict(
            id="v01b",
            text="หวัดดีครับ สลิด",
            voice="B",
            uid=11,
            respond=True,
            state="engaged",
            alt=True,
        ),
        dict(id="v02", text="สวัสดีครับน้องสลิด", respond=True, state="engaged"),
        dict(id="v03", text="หวัดดีครับ สนิท", respond=True, state="engaged"),
        dict(id="v04", text="เฮ้ Siri", respond=True, state="engaged"),
        dict(id="v05", text="สลิด วันนี้วันอะไร", respond=True, state="engaged", answer_has=today),
        dict(
            id="v05b",
            text="สลิด วันนี้วันอะไร",
            voice="B",
            uid=11,
            respond=True,
            state="engaged",
            alt=True,
        ),
        dict(
            id="v06",
            text="แล้วพรุ่งนี้ล่ะ",
            reset=False,
            respond=True,
            state="engaged",
            answer_has=tomorrow,
        ),
        # a lookup that needs ask_salid (ledger tool) -> the spoken filler comes first
        dict(id="v06b", text="สลิด เดือนนี้ใช้จ่ายไปเท่าไหร่", reset=False, respond=True, state="engaged"),
        dict(id="v07", text="โอเค ขอบคุณครับ", reset=False, respond=None, state="listening"),
        dict(id="v08", text="เมื่อวานกินข้าวที่ไหน", voice="B", uid=11, respond=False, state="listening"),
        dict(id="v09", text="เรากับเขาสนิทกันมาก", respond=False, state="listening"),
        dict(id="v10", kind="bargein"),
        dict(id="v12", kind="sidetalk"),
        dict(id="v11", text="สลิด ออกประชุม", respond=None, ends=True),
    ]


def _judge(step, obs) -> tuple[bool, str, str]:
    exp, got = [], []
    ok = True
    if step.get("respond") is True:
        exp.append("responds")
        if not obs["responded"]:
            ok = False
    elif step.get("respond") is False:
        exp.append("silent")
        if obs["responded"]:
            ok = False
    got.append("responded" if obs["responded"] else "silent")
    if step.get("state"):
        exp.append(f"state={step['state']}")
        now = "engaged" if obs["engaged_now"] else "listening"
        got.append(f"state={now}")
        if now != step["state"]:
            ok = False
    if step.get("ends"):
        exp.append("leaves")
        got.append("left" if obs["ended"] else "stayed")
        ok = ok and obs["ended"]
    if step.get("answer_has"):
        exp.append(f"mentions {step['answer_has']}")
        txt = " ".join(obs["text_out"])
        hit = step["answer_has"] in txt
        got.append("mentions ok" if hit else f"text={txt[:60]!r}")
        if not hit and not obs["text_out"]:
            pass  # no text to check (voice only): not a failure on its own
        elif not hit:
            ok = False
    return ok, ", ".join(exp), ", ".join(got)


async def _bargein(rig: Rig, step) -> dict:
    """Long spoken answer, user talks over it, then check Salid still listens and answers a
    later name call (regression: stop() on a voice-recv client made it deaf)."""
    rig.reset_listening()
    ev0, pl0, _ = rig.snapshot()
    pcm = synth("สลิด เล่าเรื่องตลกสั้นๆ ให้ฟังหน่อย", "A")
    t_end = await rig.speak(pcm, 10)
    # wait until Salid is audibly talking (clip answer or Live audio)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 40:
        await asyncio.sleep(0.1)
        talking = rig.vc.playing and any(p["first_audio"] for p in rig.vc.plays[pl0:])
        if talking and (rig.mode == "live" or len(rig.vc.plays[pl0:]) >= 2):
            break
    started = rig.vc.playing
    over = synth("เดี๋ยวก่อนนะ ขอพูดอะไรหน่อย", "B")
    await rig.speak(over, 11)  # the user talks over Salid
    # let the interruption / playback finish
    t1 = time.monotonic()
    while (rig.vc.playing or rig.s.speaking or rig.s.speech_pending) and time.monotonic() - t1 < 40:
        await asyncio.sleep(0.2)
    await asyncio.sleep(1.5)
    listening = rig.vc.listening is not None
    obs2 = await rig.run_step(dict(text="สลิด วันนี้วันอะไร", respond=True))
    ok = started and listening and rig.vc.stop_calls == 0 and obs2["responded"]
    got = (
        f"talked={started}, listening_after={listening}, stop()={rig.vc.stop_calls}, "
        f"stop_playing()={rig.vc.stop_playing_calls}, later_call_responded={obs2['responded']}"
    )
    return {
        "ok": ok,
        "exp": "keeps listening after playback/barge-in; later name call answered",
        "got": got,
        "obs": {**obs2, "first_audio_s": obs2["first_audio_s"], "t_end": t_end},
        "transcript": obs2["transcript"],
    }


async def _sidetalk(rig: Rig, step) -> dict:
    """Noisy room: while Salid answers Ann, Boy chats about lunch nearby (must NOT cut her off);
    then Ann herself talks over her ("เดี๋ยวก่อน ...") and Salid stops."""
    rig.reset_listening()
    _, pl0, _ = rig.snapshot()
    await rig.speak(synth("สลิด เล่าประวัติกรุงเทพให้ฟังหน่อย ขอยาวๆ ละเอียดๆ เลยนะ", "A"), 10)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 40:
        await asyncio.sleep(0.1)
        if rig.vc.playing and any(p["first_audio"] for p in rig.vc.plays[pl0:]):
            break
    started = rig.vc.playing
    stops0 = rig.vc.stop_playing_calls
    await rig.speak(synth("เที่ยงนี้ไปกินข้าวที่ไหนกันดี ร้านข้างล่างคนเยอะมากเลย", "B"), 11)
    await asyncio.sleep(1.0)
    kept = rig.vc.playing and rig.vc.stop_playing_calls == stops0
    await rig.speak(synth("เดี๋ยวก่อนนะ ขอถามอีกเรื่องหนึ่ง", "A"), 10)
    t1 = time.monotonic()
    while rig.vc.stop_playing_calls == stops0 and time.monotonic() - t1 < 8:
        await asyncio.sleep(0.1)
    stopped = rig.vc.stop_playing_calls > stops0
    ok = started and kept and stopped
    got = f"talked={started}, kept_talking_over_side_chat={kept}, stopped_when_partner_cut_in={stopped}"
    t2 = time.monotonic()
    while (rig.vc.playing or rig.s.speaking) and time.monotonic() - t2 < 30:
        await asyncio.sleep(0.2)
    await asyncio.sleep(1.0)
    return {"ok": ok, "exp": "ignores side chat; stops when the person she answers cuts in",
            "got": got, "obs": {"first_audio_s": None}, "transcript": []}  # fmt: skip


async def _run_mode(mode: str, report: list[dict], only: set[str] | None, alt: bool):
    rig = Rig(mode)
    await rig.open()
    try:
        for step in steps():
            if only and step["id"] not in only and f"{mode}:{step['id']}" not in only:
                continue
            if step.get("alt") and not alt:
                continue
            sid = f"{mode}:{step['id']}"
            try:
                if step.get("kind") in ("bargein", "sidetalk"):
                    r = await (_bargein if step["kind"] == "bargein" else _sidetalk)(rig, step)
                    row = dict(suite="voice", id=sid, scenario=f"[{mode}] barge-in then name call",
                               expected=r["exp"], got=r["got"], passed=bool(r["ok"]),
                               latency_s=r["obs"]["first_audio_s"],
                               detail={"transcript": r["transcript"], **{k: v for k, v in r["obs"].items() if k != "t_end"}})  # fmt: skip
                else:
                    obs = await rig.run_step(step)
                    ok, exp, got = _judge(step, obs)
                    row = dict(suite="voice", id=sid, scenario=f"[{mode}] {step['text']}",
                               expected=exp, got=got, passed=ok, latency_s=obs["first_audio_s"],
                               detail={k: v for k, v in obs.items() if k != "t_end"})  # fmt: skip
            except Exception as e:  # keep going: one broken step must not hide the others
                row = dict(suite="voice", id=sid, scenario=f"[{mode}] {step.get('text', step['id'])}",
                           expected="no error", got=f"{type(e).__name__}: {e}", passed=False,
                           latency_s=None, detail={})  # fmt: skip
            report.append(row)
            if step.get("ends") and rig.s.key not in rig.mgr.sessions:
                break
    finally:
        await rig.close()


async def run_voice_suite(
    report: list[dict], only: set[str] | None = None, modes=("classic", "live"), alt: bool = True
) -> None:
    """Append one dict per utterance to `report`. `only`: step ids ("v05") or "mode:id"."""
    for mode in modes:
        await _run_mode(mode, report, only, alt)
