from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from google.genai import types

from .config import Settings
from .llm import generate
from .models import CATEGORIES
from .store import GROUPS, Store

logger = logging.getLogger(__name__)
MAX_ROUNDS = 4
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


def _s(desc="", enum=None):
    return types.Schema(type="STRING", description=desc, enum=enum)


_DATE = "ISO date YYYY-MM-DD, inclusive"
_GROUP = _s("grouping", list(GROUPS))
TOOL_DECLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="spending_summary",
            description="Total spending and receipt count for a date range, optionally grouped.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"start_date": _s(_DATE), "end_date": _s(_DATE), "group_by": _GROUP},
                required=["start_date", "end_date"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_receipts",
            description="List individual receipts in a date range with optional filters.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "start_date": _s(_DATE),
                    "end_date": _s(_DATE),
                    "category": _s("expense category", CATEGORIES),
                    "vendor_contains": _s("substring of vendor name"),
                    "submitter": _s("substring of submitter display name"),
                    "limit": types.Schema(type="INTEGER", description="max rows, default 20"),
                },
                required=["start_date", "end_date"],
            ),
        ),
        types.FunctionDeclaration(
            name="compare_periods",
            description="Compare spending of period A vs period B (diff = A - B, with percent change), optionally per group.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "a_start": _s(_DATE),
                    "a_end": _s(_DATE),
                    "b_start": _s(_DATE),
                    "b_end": _s(_DATE),
                    "group_by": _GROUP,
                },
                required=["a_start", "a_end", "b_start", "b_end"],
            ),
        ),
        types.FunctionDeclaration(
            name="flagged_receipts",
            description="Receipts needing attention: low confidence, missing total, or possible duplicates.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ]
)


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


def ask(
    client,
    settings: Settings,
    store: Store,
    question: str,
    *,
    is_exec: bool,
    user_id: int,
    now=None,
) -> Answer:
    """Answer a question with Gemini function calling. Scope is decided here in code, before any data reaches the model."""
    started = time.monotonic()
    tools = LedgerTools(store, forced_submitter_id=None if is_exec else user_id)
    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt(settings, scoped=not is_exec, now=now),
        tools=[TOOL_DECLS],
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    final_cfg = cfg.model_copy(update={"tools": None})
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    calls: list[tuple[str, dict]] = []
    rounds = 0
    while True:
        last = rounds >= MAX_ROUNDS
        resp = generate(client, settings.gemini_model, contents, final_cfg if last else cfg)
        cand = resp.candidates[0].content if resp.candidates else None
        fcs = [p.function_call for p in (cand.parts or []) if p.function_call] if cand else []
        if not fcs or last:
            text = (resp.text or "").strip() if not (last and fcs) else ""
            return Answer(
                text or "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้งนะครับ",
                rounds,
                time.monotonic() - started,
                calls,
            )
        rounds += 1
        contents.append(cand)  # keep model turn verbatim (thought signatures)
        results = []
        for fc in fcs:
            args = dict(fc.args or {})
            calls.append((fc.name, args))
            results.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": tools.call(fc.name, args)}
                )
            )
        contents.append(types.Content(role="user", parts=results))
