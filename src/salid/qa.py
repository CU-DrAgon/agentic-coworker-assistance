from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from .config import Settings
from .store import Store

logger = logging.getLogger(__name__)
MAX_ROUNDS = 4
APP_NAME = "salid"
# Vertex 429s (shared quota) and 5xx are retried with backoff by the genai HTTP client
# (same policy as meeting_scribe.agents).
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 500, 502, 503, 504]
)
THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]


def baht(x: float) -> str:
    return "฿" + f"{x:,.2f}".removesuffix(".00")


class LedgerTools:
    """Tools the model may call. All arithmetic happens here (SQL). `forced_submitter_id` is injected into every
    query and is NOT part of any tool schema, so the model cannot widen the scope."""

    def __init__(self, store: Store, forced_submitter_id: int | None = None):
        self.store = store
        self.forced = forced_submitter_id

    def _fmt_summary(self, s: dict) -> dict:
        s["total_fmt"] = baht(s["total"])
        for g in s["groups"]:
            g["total_fmt"] = baht(g["total"])
        return s

    def spending_summary(self, start_date: str, end_date: str, group_by: str = "none") -> dict:
        return self._fmt_summary(
            self.store.summarize(start_date, end_date, group_by, submitter_id=self.forced)
        )

    def list_receipts(
        self,
        start_date: str,
        end_date: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
    ) -> dict:
        rows = self.store.list_receipts(
            start_date,
            end_date,
            category,
            vendor_contains,
            submitter,
            limit,
            submitter_id=self.forced,
        )
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    def compare_periods(
        self, a_start: str, a_end: str, b_start: str, b_end: str, group_by: str = "none"
    ) -> dict:
        a = self.store.summarize(a_start, a_end, group_by, submitter_id=self.forced)
        b = self.store.summarize(b_start, b_end, group_by, submitter_id=self.forced)
        am, bm = (
            {g["key"]: g["total"] for g in a["groups"]},
            {g["key"]: g["total"] for g in b["groups"]},
        )
        rows = []
        for k in sorted(set(am) | set(bm), key=lambda k: -(am.get(k, 0) + bm.get(k, 0))):
            rows.append(self._delta(k, am.get(k, 0.0), bm.get(k, 0.0)))
        return {
            "period_a": {
                "start": a_start,
                "end": a_end,
                "total_fmt": baht(a["total"]),
                "total": a["total"],
            },
            "period_b": {
                "start": b_start,
                "end": b_end,
                "total_fmt": baht(b["total"]),
                "total": b["total"],
            },
            "overall": self._delta("all", a["total"], b["total"]),
            "groups": rows,
        }

    @staticmethod
    def _delta(key, a, b):
        diff = round(a - b, 2)
        return {
            "key": key,
            "a": a,
            "b": b,
            "diff": diff,
            "diff_fmt": ("+" if diff > 0 else "-" if diff < 0 else "") + baht(abs(diff)),
            "pct_change_vs_b": round(diff / b * 100, 1) if b else None,
        }

    def flagged_receipts(self) -> dict:
        rows = self.store.flagged(submitter_id=self.forced)
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    @staticmethod
    def _r(r) -> dict:
        return {
            "id": r.id,
            "date": r.effective_date,
            "vendor": r.vendor,
            "total": r.total,
            "total_fmt": baht(r.total) if r.total is not None else None,
            "category": r.category,
            "submitter": r.submitter_name,
            "confidence": r.confidence,
            "possible_duplicate_of": r.possible_duplicate_of,
            "notes": r.notes,
        }

    def call(self, name: str, args: dict) -> dict:
        allowed = {"spending_summary", "list_receipts", "compare_periods", "flagged_receipts"}
        if name not in allowed:
            return {"error": f"unknown tool {name}"}
        fn = getattr(self, name)
        # only pass parameters the function declares; drops any attempt to inject submitter_id etc.
        params = fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
        clean = {k: v for k, v in (args or {}).items() if k in params and v is not None}
        try:
            return fn(**clean)
        except (TypeError, ValueError) as e:
            return {"error": str(e)}


def system_prompt(settings: Settings, scoped: bool, now=None) -> str:
    now = now or settings.now()
    p = (
        f"คุณชื่อ “สลิด” เป็นเลขาของร้าน ดูแลเรื่องใบเสร็จและบัญชี ตอบเป็นภาษาไทยสั้น กระชับ ตรงประเด็น (เหมาะกับ Discord) ลงท้ายด้วยคำว่าครับเสมอ\n"
        f"วันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        f"สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง 'เดือนนี้', 'สัปดาห์ก่อน' ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD\n"
        "กฎเหล็ก: ห้ามคำนวณตัวเลขเอง (บวก ลบ เปอร์เซ็นต์) ให้เรียกเครื่องมือและใช้ตัวเลขที่เครื่องมือคืนมาเท่านั้น "
        "(ใช้ฟิลด์ *_fmt ได้เลย) ตัวเลขใส่ , คั่นหลักพัน และนำหน้าด้วย ฿. ถ้าไม่มีข้อมูลให้บอกตรงๆ ห้ามเดา. "
        "ถ้าคำถามไม่เกี่ยวกับค่าใช้จ่าย/ใบเสร็จ ให้ปฏิเสธสั้นๆ. เรียกเครื่องมือหลายตัวพร้อมกันได้เพื่อความเร็ว"
    )
    if scoped:
        p += "\nผู้ถามไม่ใช่ผู้บริหาร: ข้อมูลที่เห็นคือเฉพาะใบเสร็จที่ผู้ถามส่งเองเท่านั้น ให้ระบุในคำตอบว่าเป็นของผู้ถามเอง และห้ามอ้างถึงข้อมูลของคนอื่น"
    return p


@dataclass
class Answer:
    text: str
    rounds: int
    seconds: float
    tool_calls: list[tuple[str, dict]]


def build_tools(ledger: LedgerTools, calls: list[tuple[str, dict]] | None = None) -> list:
    """The four ledger tools as plain functions bound to `ledger` (ADK wraps them as FunctionTools).

    Permission scoping lives in `ledger.forced`: none of these signatures has a submitter
    parameter, so the model cannot pass one (ADK also drops undeclared arguments). Errors are
    returned to the model as data instead of raised, so a bad date never kills the turn.
    """
    log = calls if calls is not None else []

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    def spending_summary(start_date: str, end_date: str, group_by: str = "none") -> dict:
        """Total spending and receipt count for a date range, optionally grouped.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive.
            group_by: one of category, vendor, submitter, day, none.
        """
        a = dict(start_date=start_date, end_date=end_date, group_by=group_by)
        return safe("spending_summary", a, lambda: ledger.spending_summary(**a))

    def list_receipts(
        start_date: str,
        end_date: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
    ) -> dict:
        """List individual receipts in a date range with optional filters.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive.
            category: expense category, one of the Thai categories (e.g. วัตถุดิบ).
            vendor_contains: substring of the vendor name.
            submitter: substring of the submitter display name.
            limit: max rows, default 20.
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            category=category,
            vendor_contains=vendor_contains,
            submitter=submitter,
            limit=limit,
        )
        return safe("list_receipts", a, lambda: ledger.list_receipts(**a))

    def compare_periods(
        a_start: str, a_end: str, b_start: str, b_end: str, group_by: str = "none"
    ) -> dict:
        """Compare spending of period A vs period B (diff = A - B, with percent change).

        Args:
            a_start: period A start, ISO date YYYY-MM-DD, inclusive.
            a_end: period A end, ISO date YYYY-MM-DD, inclusive.
            b_start: period B start, ISO date YYYY-MM-DD, inclusive.
            b_end: period B end, ISO date YYYY-MM-DD, inclusive.
            group_by: one of category, vendor, submitter, day, none.
        """
        a = dict(a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end, group_by=group_by)
        return safe("compare_periods", a, lambda: ledger.compare_periods(**a))

    def flagged_receipts() -> dict:
        """Receipts needing attention: low confidence, missing total, or possible duplicates."""
        return safe("flagged_receipts", {}, ledger.flagged_receipts)

    return [spending_summary, list_receipts, compare_periods, flagged_receipts]


def _ensure_vertex_env(settings: Settings) -> None:
    # ADK reads these to route Gemini calls through Vertex AI (meeting_scribe sets them too).
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.gcp_project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.gemini_location)


def build_agent(
    settings: Settings,
    tools: list,
    *,
    scoped: bool,
    now=None,
    model=None,
    on_model_call=None,
) -> LlmAgent:
    """One ADK LlmAgent per request. `model` is injectable so tests stay offline."""
    _ensure_vertex_env(settings)
    prompt = system_prompt(settings, scoped=scoped, now=now)
    model_calls = 0

    def before_model(callback_context, llm_request):
        # Hard cap: after MAX_ROUNDS tool rounds the next call has no tools, so it must answer.
        nonlocal model_calls
        model_calls += 1
        if on_model_call:
            on_model_call(model_calls)
        if model_calls > MAX_ROUNDS:
            llm_request.config.tools = None
            llm_request.tools_dict.clear()
        return None

    return LlmAgent(
        name="salid_qa",
        model=model
        or Gemini(
            model=settings.gemini_model,
            client_kwargs={"location": settings.gemini_location},
            retry_options=RETRY_OPTIONS,
        ),
        description="Answers expense questions from the receipt ledger using tools.",
        # A callable skips ADK's {placeholder} templating of the instruction string.
        instruction=lambda _ctx: prompt,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0, thinking_config=types.ThinkingConfig(thinking_level="LOW")
        ),
        before_model_callback=before_model,
    )


async def ask_async(
    settings: Settings,
    store: Store,
    question: str,
    *,
    is_exec: bool,
    user_id: int,
    now=None,
    model=None,
) -> Answer:
    """Answer a question with an ADK agent. Scope is decided here in code, before any data
    reaches the model: tools are built per request around a LedgerTools whose
    `forced_submitter_id` is the asker for non-executives."""
    started = time.monotonic()
    calls: list[tuple[str, dict]] = []
    ledger = LedgerTools(store, forced_submitter_id=None if is_exec else user_id)
    agent = build_agent(
        settings, build_tools(ledger, calls), scoped=not is_exec, now=now, model=model
    )
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    adk_user, session_id = str(user_id), uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=adk_user, session_id=session_id
    )
    message = types.Content(role="user", parts=[types.Part(text=question)])
    rounds, text = 0, ""
    async for event in runner.run_async(
        user_id=adk_user,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=MAX_ROUNDS + 2),
    ):
        if event.get_function_calls():
            rounds += 1
        elif event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if not p.thought).strip()
    return Answer(
        text or "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้งนะครับ",
        rounds,
        time.monotonic() - started,
        calls,
    )


def ask(
    settings: Settings,
    store: Store,
    question: str,
    *,
    is_exec: bool,
    user_id: int,
    now=None,
    model=None,
) -> Answer:
    """Blocking wrapper around `ask_async` (call from a worker thread, e.g. asyncio.to_thread)."""
    return asyncio.run(
        ask_async(settings, store, question, is_exec=is_exec, user_id=user_id, now=now, model=model)
    )
