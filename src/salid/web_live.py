"""Live voice on the web: the SAME Salid as in a Discord voice room, one-to-one in the browser.

Same model (`settings.live_model`, Gemini Live native audio), same voice (`settings.live_voice`),
same persona and tone (`qa.persona`), same turn-taking (`live_voice.build_realtime_input`) and the
same single data tool `ask_salid` (the full text agent with the org's scope). Only the transport
differs: the browser streams 16 kHz mono PCM16 over a WebSocket, Salid's 24 kHz audio streams back,
and the transcripts come along as JSON so the panel can show them.

    ws = WebLive(settings, client, instruction=..., ask=..., out=sender)
    task = ws.start(); ws.feed(pcm16k); ...; await ws.close()

`out(kind, payload)` is awaited for every event: ("audio", bytes) 24 kHz PCM16, ("heard", text)
input transcript, ("said", text) output transcript, ("interrupted", ""), ("turn", "") turn done,
("error", text). Audio is never stored or logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from google.genai import types

from .i18n import lang_name
from .live_voice import build_realtime_input
from .qa import persona

logger = logging.getLogger(__name__)

LIVE_IN_RATE = 16_000
LIVE_OUT_RATE = 24_000
MAX_QUEUE = 200  # ~ 4 s of 20 ms frames: the browser is never allowed to run far ahead
TOOL_TIMEOUT_S = 90
WEB_SILENCE_MS = 2000  # how long the person may pause before Salid takes the turn (API max 2000)

Out = Callable[[str, Any], Awaitable[None]]


def build_web_instruction(
    settings: Any, who: str, lang: str = "en", tone: str = "cheerful", tone_note: str = ""
) -> str:
    """The web session's system instruction: the text persona (identity, tone, language) plus
    how a one-to-one voice chat in the command center works."""
    now = settings.now()
    head = persona(lang, tone, tone_note)
    web = (
        f"Today is {now:%A, %B} {now.day}, {now.year}.\n"
        "You are talking by VOICE, one-to-one, with a member of the company who opened the "
        "Salid command center on the web. Answer right away when they finish speaking; keep "
        "answers short and spoken (no markdown, no lists read aloud, no URLs). They may speak "
        f"Thai or English: understand both, but ALWAYS answer in {lang_name(lang)} only (the "
        "company's language), even when they speak another language or ask you to switch. NEVER talk over them: wait until they have clearly finished their whole "
        "thought, even if they pause mid-sentence. If they say they want to think (ขอคิดก่อน, "
        "wait, hold on, let me think, take your time), reply with at most a short 'take your time' "
        "and then stay silent until they speak again. For anything about the company (receipts, documents, meetings, tasks, "
        "people, settings) call ask_salid(question) and then say the answer in your own words; "
        "for general knowledge and small talk answer directly. If they tell you who they are or "
        "what they do, pass it to ask_salid so it is remembered. EVERY request to DO or CHANGE "
        "something (update a person's profile, title, department or manager, yours or a "
        "colleague's; set a reminder; post on Discord; create an event; change a setting) goes "
        "to ask_salid with their words, as a request: ask_salid has the tools and checks the "
        "person's rights itself. Never refuse or say you cannot do it before calling ask_salid; "
        "repeat what ask_salid reports. If they ask you to stop, stop.\n"
        f"Who you are talking with:\n{who}\n"
    )
    return head + "\n" + web


class WebLive:
    def __init__(
        self,
        settings: Any,
        client: Any,
        *,
        instruction: str,
        ask: Callable[[str], str],
        out: Out,
        connect: Callable[[types.LiveConnectConfig], Any] | None = None,
    ):
        self.s, self.client, self.instruction, self.ask, self.out = (
            settings,
            client,
            instruction,
            ask,
            out,
        )
        self._connect = connect or (
            lambda cfg: client.aio.live.connect(model=settings.live_model, config=cfg)
        )
        self._in: asyncio.Queue[bytes | tuple[str, str] | None] = asyncio.Queue(maxsize=MAX_QUEUE)
        self._task: asyncio.Task | None = None
        self._closing = False
        self.started = time.monotonic()
        self.turns = 0

    # ---- lifecycle
    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run())
        return self._task

    async def close(self) -> None:
        self._closing = True
        with contextlib.suppress(asyncio.QueueFull):
            self._in.put_nowait(None)
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def say(self, text: str) -> None:
        """Hand the model a text turn (typed in the panel, or a cue) to answer out loud."""
        if not self._closing and text.strip():
            with contextlib.suppress(asyncio.QueueFull):
                self._in.put_nowait(("text", text.strip()))

    def greet(self, lang: str = "en") -> None:
        """Salid speaks first when the panel opens, like a friend picking up the phone."""
        cue = (
            "(ผู้ใช้เพิ่งเปิดโหมดเสียงคุยกับคุณ ทักทายสั้น ๆ อย่างอบอุ่น เรียกชื่อถ้ารู้ แล้วถามว่ามีอะไรให้ช่วยไหม "
            "ไม่ต้องอธิบายความสามารถยาว ๆ)"
            if lang == "th"
            else "(The person just opened voice chat with you. Greet them warmly in one or two "
            "short sentences, by name if you know it, and ask what you can help with. Do not "
            "list your features.)"
        )
        self.say(cue)

    def feed(self, pcm16k: bytes) -> None:
        """Mic audio from the browser (PCM16 mono 16 kHz). Drops the oldest when the browser
        runs ahead, so a stalled model never builds up lag."""
        if self._closing or not pcm16k:
            return
        if self._in.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._in.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._in.put_nowait(pcm16k)

    # ---- config: identical knobs to the Discord room
    def config(self) -> types.LiveConnectConfig:
        decl = types.FunctionDeclaration(
            name="ask_salid",
            description=(
                "Salid's text brain with all of her tools. Pass the person's question OR request "
                "in their words: questions about the company (receipts, documents, meetings, "
                "tasks, people, settings) and ACTIONS: update their own or (owners / admins) a "
                "colleague's profile (nickname, title, department, manager), remember who they "
                "are, reminders, tasks, posting or scheduling on Discord, settings. It checks "
                "permissions and returns what it did or found, to say in your own words."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"question": types.Schema(type=types.Type.STRING)},
                required=["question"],
            ),
        )
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.instruction,
            tools=[types.Tool(function_declarations=[decl])],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.s.live_voice)
                )
            ),
            input_audio_transcription={},
            output_audio_transcription={},
            realtime_input_config=build_realtime_input(self._patient_settings()),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        )

    def _patient_settings(self) -> Any:
        """Web calls are one-to-one: give the person room to think. End-of-speech only after a
        longer silence (at least WEB_SILENCE_MS), low end sensitivity; everything else as the
        Discord room."""
        from types import SimpleNamespace

        s = self.s
        return SimpleNamespace(
            live_barge_in="any",  # the PERSON may interrupt Salid by just talking
            live_start_sensitivity=getattr(s, "live_start_sensitivity", "high"),
            live_end_sensitivity="low",
            live_prefix_padding_ms=int(getattr(s, "live_prefix_padding_ms", 300)),
            live_silence_ms=max(int(getattr(s, "live_silence_ms", 1200)), WEB_SILENCE_MS),
        )

    # ---- the session
    async def run(self) -> None:
        try:
            async with self._connect(self.config()) as live:
                await self.out("ready", "")
                sender = asyncio.create_task(self._send(live))
                try:
                    await self._receive(live)
                finally:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("web live: session error: %s: %s", type(e).__name__, e)
            with contextlib.suppress(Exception):
                await self.out("error", type(e).__name__)

    async def _send(self, live: Any) -> None:
        while not self._closing:
            item = await self._in.get()
            if item is None:
                return
            if isinstance(item, tuple):  # ("text", str): a typed or cued turn
                await live.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=item[1])]),
                    turn_complete=True,
                )
                continue
            await live.send_realtime_input(
                audio=types.Blob(data=item, mime_type=f"audio/pcm;rate={LIVE_IN_RATE}")
            )

    async def _receive(self, live: Any) -> None:
        while not self._closing:
            seen = False
            async for msg in live.receive():  # ends after every turn / tool call: loop again
                seen = True
                if await self._on_message(live, msg) == "end":
                    return
            if not seen:
                await asyncio.sleep(0.05)

    async def _on_message(self, live: Any, msg: Any) -> str | None:
        if getattr(msg, "go_away", None) is not None:
            await self.out("error", "go_away")
            return "end"
        if getattr(msg, "tool_call", None) is not None:
            await self._tool_call(live, msg.tool_call)
        sc = getattr(msg, "server_content", None)
        if sc is None:
            return None
        if getattr(sc, "interrupted", False):
            await self.out("interrupted", "")
        if getattr(sc, "model_turn", None) is not None:
            for part in sc.model_turn.parts or []:
                data = part.inline_data.data if part.inline_data is not None else None
                if data:
                    await self.out("audio", data)
        it = getattr(sc, "input_transcription", None)
        if it is not None and it.text:
            await self.out("heard", it.text)
        ot = getattr(sc, "output_transcription", None)
        if ot is not None and ot.text:
            await self.out("said", ot.text)
        if getattr(sc, "turn_complete", False):
            self.turns += 1
            await self.out("turn", "")
        return None

    async def _tool_call(self, live: Any, tool_call: Any) -> None:
        responses = []
        for fc in tool_call.function_calls or []:
            question = str((fc.args or {}).get("question", "")).strip()
            if fc.name != "ask_salid" or not question:
                result = {"error": "unknown tool or empty question"}
            else:
                await self.out("thinking", question)
                try:
                    answer = await asyncio.wait_for(
                        asyncio.to_thread(self.ask, question), TOOL_TIMEOUT_S
                    )
                    result = {"answer": answer}
                except Exception as e:
                    logger.warning("web live: ask_salid failed (%s)", type(e).__name__)
                    result = {"error": "could not get the answer right now"}
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
        if responses:
            await live.send_tool_response(function_responses=responses)
