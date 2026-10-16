"""Agent tools for business finance, market benchmarking and security checks.

`bizfin_tools(ctx)` returns the plain functions `_agent_kwargs` appends to `extra_tools` (via
`bizfin_bot.BizfinMixin`), so chat and voice share them. All arithmetic is in `bizfin.py`
(Decimal, deterministic); the model only passes numbers in and relays the `text` out.
Attached files go through the `outbox` attribute (see `qa.tool_files`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from . import bizfin as bf
from . import security_check as sec
from .benchmark import run_benchmark
from .ideas_pdf import markdown_pdf
from .longtext import summary_of
from .pdf_report import FontSet

logger = logging.getLogger(__name__)

BIZFIN_PROMPT = (
    "\nการเงินธุรกิจ: ตัวเลขทุกตัวต้องมาจากเครื่องมือคำนวณเท่านั้น ห้ามคิดเลขเอง. "
    "จุดคุ้มทุน/กำไรต่อหน่วย -> unit_economics; ตั้งราคาจากต้นทุน (มี VAT 7% แยก) -> price_from_cost; "
    "ROI/Payback/NPV/IRR -> investment_metrics; ประมาณการ 12-36 เดือน/LTV/CAC/runway -> financial_projection "
    "(แนบ Excel ให้). ถ้าผู้ใช้ถามว่า 'ต้องใช้ข้อมูลอะไรบ้าง' หรือข้อมูลยังไม่พอ ให้เรียก financial_data_checklist "
    "แล้วถามเฉพาะข้อที่ขาด; ถ้าผู้ใช้ไม่รู้ตัวเลข เสนอสมมติฐานที่สมเหตุสมผลและบอกว่าเป็นสมมติฐาน. "
    "ตอบด้วย text ที่ได้ (ย่อได้) พร้อมสูตรที่แสดง ห้ามเปลี่ยนตัวเลข. ถ้าผู้ใช้อยากเก็บเข้าโปรเจกต์ไอเดีย ใส่ save_to_idea=ชื่อไอเดีย. "
    "เทียบตลาด/หาช่องว่าง/เทียบคู่แข่ง -> market_benchmark(topic) (ค้นเว็บมีแหล่งอ้างอิง ใช้เวลา 20-60 วินาที; ตอบ summary แล้วบอกว่าแนบไฟล์เต็ม). "
    "ผู้ใช้ส่งลิงก์/ข้อความถามว่าหลอกไหม/ปลอดภัยไหม -> security_check (ผลเป็นการประเมิน ไม่ใช่การรับรอง; "
    "ห้ามเปิดหรือบอกว่าเปิดลิงก์แล้ว). ผู้ใช้ส่งสลิดโอนเงินและถามว่าจริงไหม -> อ่านค่าจากภาพให้ครบ "
    "(ธนาคาร ยอด วันที่ เวลา เลขอ้างอิง ผู้โอน ผู้รับ) แล้วเรียก verify_slip; บอกชัดว่าสลิดยืนยันกับธนาคารไม่ได้ "
    "ต้องเช็กยอดเข้าในแอปธนาคารเอง ห้ามฟันธงว่าสลิดจริง.\n"
)


@dataclass
class BizfinCtx:
    lang: str
    now: Callable[[], datetime]
    client: object | None = None  # genai client (benchmark); None = no web research
    model: str = ""
    org: str = ""
    fonts: Callable[[], FontSet] = lambda: FontSet(None, None, False)
    safe_browsing_key: str = ""
    sb_post: Callable[..., dict] | None = None
    # (idea title, section, markdown) -> result dict; None when ideas are off
    save_idea: Callable[[str, str, str], dict] | None = None
    # slip reference (compact) -> ledger receipt id the asker may see, or None
    ledger_ref: Callable[[str], int | None] | None = None
    search: Callable | None = None  # tests inject a fake grounded search
    outbox: list[tuple[str, bytes]] = field(default_factory=list)

    # ---- helpers
    def _saved(self, out: dict, title: str, section: str, md: str) -> None:
        if not title.strip():
            return
        if self.save_idea is None:
            out["saved_to_idea"] = "unavailable"
            return
        try:
            res = self.save_idea(title, section, md)
        except Exception:
            logger.warning("bizfin: saving to idea failed", exc_info=True)
            res = {"status": "error"}
        out["saved_to_idea"] = res.get("status", "error")

    def _file(
        self, base: str, fmt: str, sheets: dict[str, list[dict]], md: str, out: dict, title: str
    ) -> None:
        fmt = (fmt or "").lower().lstrip(".")
        if fmt in ("", "none", "no"):
            return
        try:
            if fmt == "pdf":
                data = markdown_pdf(
                    org=self.org,
                    title=title,
                    markdown=md,
                    lang=self.lang,
                    generated_at=self.now(),
                    fonts=self.fonts(),
                )
                name = f"{base}.pdf"
            elif fmt == "csv":
                first = next(iter(sheets.values()), [])
                name, data = f"{base}.csv", bf.to_csv(first)
            else:
                name, data = bf.table_file(base, sheets)
        except Exception:
            logger.warning("bizfin file failed", exc_info=True)
            out["file_error"] = "could not build the file"
            return
        self.outbox.append((name, data))
        out["attached_file"] = name

    def run(self, fn: Callable[[], dict]) -> dict:
        try:
            return fn()
        except bf.FinError as e:
            return {"status": "error", "error": str(e)}
        except Exception:
            logger.exception("bizfin tool failed")
            return {"status": "error", "error": "the finance tool failed; try again"}

    # ---- tools
    def unit_economics(self, **kw) -> dict:
        save = kw.pop("save_to_idea", "")
        res = bf.unit_economics(lang=self.lang, **kw)
        out = {"status": "ok", **res}
        self._saved(out, save, "Unit economics", "## Unit economics\n" + res["text"])
        return out

    def price_from_cost(self, **kw) -> dict:
        save, fmt = kw.pop("save_to_idea", ""), kw.pop("file_format", "")
        res = bf.price_from_cost(lang=self.lang, **kw)
        out = {"status": "ok", **res}
        rows = [
            {"tier": t["tier"], "price_ex_vat": t["price_ex_vat"], "vat": t["vat"],
             "price_inc_vat": t["price_inc_vat"], "margin_pct": t["margin_pct"]}
            for t in res["tiers"]
        ]  # fmt: skip
        costs = [{"item": c["name"], "amount": c["amount"]} for c in res["cost_items"]]
        self._file("pricing", fmt, {"Tiers": rows, "Costs": costs}, res["text"], out, "Pricing")
        self._saved(out, save, "pricing", "## Pricing from cost\n" + res["text"])
        return out

    def investment_metrics(self, **kw) -> dict:
        save, fmt = kw.pop("save_to_idea", ""), kw.pop("file_format", "")
        res = bf.investment_metrics(lang=self.lang, **kw)
        out = {"status": "ok", **res}
        rows = _cash_rows(
            kw["initial_investment"], kw["cash_flows"], res["discount_rate_period_pct"]
        )
        self._file("investment", fmt, {"Cash flows": rows}, res["text"], out, "Investment")
        self._saved(out, save, "Investment analysis", "## Investment analysis\n" + res["text"])
        return out

    def financial_projection(self, **kw) -> dict:
        save, fmt = kw.pop("save_to_idea", ""), kw.pop("file_format", "xlsx")
        res = bf.financial_projection(lang=self.lang, **kw)
        scen = [{"scenario": s, **v} for s, v in res["scenarios"].items()]
        assum = [{"assumption": k, "value": v} for k, v in res["assumptions"].items()]
        out = {"status": "ok", **{k: v for k, v in res.items() if k != "table"}}
        out["table_rows"] = len(res["table"])
        self._file(
            "projection",
            fmt,
            {"Projection": res["table"], "Scenarios": scen, "Assumptions": assum},
            res["text"],
            out,
            "Financial projection",
        )
        self._saved(out, save, "Financial projection", "## Financial projection\n" + res["text"])
        return out

    def checklist(self, product_type: str, provided) -> dict:
        return {"status": "ok", **bf.financial_data_checklist(product_type, provided, self.lang)}

    def benchmark(self, topic: str, our_product: str) -> dict:
        if self.client is None:
            return {"status": "error", "error": "web research is not available"}
        topic = " ".join((topic or "").split())
        if not topic:
            return {"status": "error", "error": "give the market or topic to benchmark"}
        kw = {"search": self.search} if self.search is not None else {}
        today = self.now().astimezone(bf_tz()).date() if self.now() else date.today()
        res = run_benchmark(
            self.client,
            topic=topic,
            our_product=our_product,
            lang=self.lang,
            today=today,
            search_model=self.model,
            synth_model=self.model,
            **kw,
        )
        if res.status != "ok":
            return {
                "status": "no_results",
                "instruction": "Say that not enough sourced information was found; offer web_search with a narrower query.",
            }
        name = _ascii(topic)
        try:
            pdf = markdown_pdf(
                org=self.org,
                title=res.title,
                markdown=res.markdown,
                lang=self.lang,
                generated_at=self.now(),
                fonts=self.fonts(),
            )
            self.outbox.append((f"benchmark-{name}.pdf", pdf))
            fname = f"benchmark-{name}.pdf"
        except Exception:
            logger.warning("benchmark PDF failed", exc_info=True)
            self.outbox.append((f"benchmark-{name}.md", res.markdown.encode("utf-8")))
            fname = f"benchmark-{name}.md"
        return {
            "status": "ok",
            "summary": summary_of(res.markdown, chars=700, headings=8),
            "competitors": [
                {"name": c["name"], "pricing": c["pricing"], "thai_support": c["thai_support"]}
                for c in res.competitors
            ],
            "gaps": [g["gap"] for g in res.gaps],
            "sources": [{"n": s["n"], "title": s["title"], "link": s["link"]} for s in res.sources],
            "attached_file": fname,
            "report_markdown": res.markdown,
            "instruction": (
                "Reply with a short summary (max 10 lines): top 3 gaps, who the main competitors "
                "are, and where we win/lose, with [n] citations and the source links; say the "
                "full report is attached. Never add facts that are not in the result."
            ),
        }

    def security_check(self, link_or_text: str) -> dict:
        res = sec.check(link_or_text, safe_browsing_key=self.safe_browsing_key, post=self.sb_post)
        if res is None:
            return {"status": "nothing_to_check", "text": sec.render_check(None, self.lang)}
        return {
            "status": "ok",
            "verdict": res.verdict,
            "reasons": [sec.reason_text(r, self.lang) for r in res.reasons],
            "hosts": [r.host for r in res.links if r.host],
            "safe_browsing_checked": res.safe_browsing,
            "text": sec.render_check(res, self.lang),
            "instruction": "Relay the verdict and advice. It is an estimate: the page was not opened.",
        }

    def verify_slip(self, **kw) -> dict:
        res = sec.verify_slip_data(
            ledger_refs=self.ledger_ref, now=self.now().astimezone(bf_tz()), lang=self.lang, **kw
        )
        return res


def bf_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo("Asia/Bangkok")


def _ascii(title: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:40]
    return s or "report"


def _cash_rows(initial, flows, rate_pct) -> list[dict]:
    """Cash-flow table for the spreadsheet: flow, cumulative, discount factor, discounted."""
    from decimal import Decimal

    r = Decimal(str(rate_pct)) / 100
    cum = -bf.D(initial)
    dcum = cum
    rows = [{"period": 0, "cash_flow": float(cum), "cumulative": float(cum),
             "discount_factor": 1.0, "discounted": float(cum), "cumulative_discounted": float(dcum)}]  # fmt: skip
    for t, cf in enumerate(flows, 1):
        d = bf.D(cf)
        f = 1 / (1 + r) ** t
        cum += d
        dcum += d * f
        rows.append(
            {
                "period": t,
                "cash_flow": float(d),
                "cumulative": float(round(cum, 2)),
                "discount_factor": round(float(f), 6),
                "discounted": float(round(d * f, 2)),
                "cumulative_discounted": float(round(dcum, 2)),
            }
        )
    return rows


def bizfin_tools(ctx: BizfinCtx, *, web: bool = True) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions)."""

    def unit_economics(
        fixed_costs_month: float,
        variable_cost_unit: float,
        price: float,
        expected_units_month: float | None = None,
        initial_investment: float = 0,
        currency: str = "THB",
        save_to_idea: str = "",
    ) -> dict:
        """Unit economics and break-even, computed exactly in code (never calculate it yourself):
        contribution margin, gross margin, break-even units and revenue per month, months to
        recoup an initial investment. Prices and costs are ex-VAT.

        Args:
            fixed_costs_month: fixed costs per month (salaries, rent, software...).
            variable_cost_unit: variable cost of one unit / one customer per month.
            price: selling price per unit, ex-VAT.
            expected_units_month: expected units sold per month, if known.
            initial_investment: one-off investment, 0 if none.
            currency: currency label, THB by default.
            save_to_idea: idea title to store the result in (only if the user asks), else empty.
        """
        return ctx.run(
            lambda: ctx.unit_economics(
                fixed_costs_month=fixed_costs_month,
                variable_cost_unit=variable_cost_unit,
                price=price,
                expected_units_month=expected_units_month,
                initial_investment=initial_investment,
                currency=currency,
                save_to_idea=save_to_idea,
            )
        )

    def price_from_cost(
        cost_items: list[str],
        target_margin_pct: float | None = None,
        markup_pct: float | None = None,
        competitor_prices: list[float] | None = None,
        competitors_include_vat: bool = False,
        currency: str = "THB",
        file_format: str = "",
        save_to_idea: str = "",
    ) -> dict:
        """Suggest prices from costs: cost-plus price at a target margin (or markup), entry /
        standard / premium tiers, a value-based band from competitor prices, Thai VAT 7% shown
        separately. All math in code. The tiers are assumptions, say so.

        Args:
            cost_items: cost per unit as "name: amount" strings (or plain numbers), e.g.
                ["วัตถุดิบ: 40", "บรรจุภัณฑ์: 8"].
            target_margin_pct: target margin on price in percent (default 30 if neither is given).
            markup_pct: markup on cost in percent (use instead of a margin).
            competitor_prices: competitor prices for the same thing, if known.
            competitors_include_vat: true when those competitor prices include VAT.
            currency: currency label, THB by default.
            file_format: "xlsx", "csv" or "pdf" to attach a file; empty = none.
            save_to_idea: idea title to store the result in (only if the user asks), else empty.
        """
        return ctx.run(
            lambda: ctx.price_from_cost(
                cost_items=cost_items,
                target_margin_pct=target_margin_pct,
                markup_pct=markup_pct,
                competitor_prices=competitor_prices,
                competitors_include_vat=competitors_include_vat,
                currency=currency,
                file_format=file_format,
                save_to_idea=save_to_idea,
            )
        )

    def investment_metrics(
        initial_investment: float,
        cash_flows: list[float],
        discount_rate_pct: float = 10,
        period: str = "month",
        currency: str = "THB",
        file_format: str = "",
        save_to_idea: str = "",
    ) -> dict:
        """ROI, payback period, discounted payback, NPV and IRR for an investment, exact math in
        code with the formulas shown in `text`. Use it to value a product or project.

        Args:
            initial_investment: money spent at the start (positive number).
            cash_flows: net cash flow of each period after the start (negative allowed).
            discount_rate_pct: ANNUAL discount rate / required return in percent (default 10).
            period: "month" or "year": what each cash flow is.
            currency: currency label, THB by default.
            file_format: "xlsx", "csv" or "pdf" to attach the table; empty = none.
            save_to_idea: idea title to store the result in (only if the user asks), else empty.
        """
        return ctx.run(
            lambda: ctx.investment_metrics(
                initial_investment=initial_investment,
                cash_flows=cash_flows,
                discount_rate_pct=discount_rate_pct,
                period=period,
                currency=currency,
                file_format=file_format,
                save_to_idea=save_to_idea,
            )
        )

    def financial_projection(
        arpu_month: float,
        new_customers_month: float,
        churn_pct_month: float,
        cac: float,
        fixed_costs_month: float,
        variable_cost_customer: float = 0,
        initial_customers: float = 0,
        acquisition_growth_pct_month: float = 0,
        opening_cash: float = 0,
        months: int = 24,
        scenario: str = "base",
        currency: str = "THB",
        file_format: str = "xlsx",
        save_to_idea: str = "",
    ) -> dict:
        """12-36 month projection of customers, revenue, costs, profit and cash with LTV, LTV/CAC,
        CAC payback and runway, and low / base / high scenarios. Exact math in code; the Excel
        file (all months + scenarios + assumptions) is attached. Inputs are the user's
        assumptions: say so in the reply.

        Args:
            arpu_month: average revenue per customer per month (ex-VAT).
            new_customers_month: new customers in the first month.
            churn_pct_month: percent of customers lost each month.
            cac: cost to win one customer.
            fixed_costs_month: fixed costs per month.
            variable_cost_customer: cost to serve one customer per month.
            initial_customers: customers at the start.
            acquisition_growth_pct_month: monthly growth of new customers, percent.
            opening_cash: cash at the start.
            months: 12 to 36.
            scenario: "base", "low" or "high" (the others are summarised too).
            currency: currency label, THB by default.
            file_format: "xlsx" (default), "csv", "pdf" or "none".
            save_to_idea: idea title to store the result in (only if the user asks), else empty.
        """
        return ctx.run(
            lambda: ctx.financial_projection(
                arpu_month=arpu_month,
                new_customers_month=new_customers_month,
                churn_pct_month=churn_pct_month,
                cac=cac,
                fixed_costs_month=fixed_costs_month,
                variable_cost_customer=variable_cost_customer,
                initial_customers=initial_customers,
                acquisition_growth_pct_month=acquisition_growth_pct_month,
                opening_cash=opening_cash,
                months=months,
                scenario=scenario,
                currency=currency,
                file_format=file_format,
                save_to_idea=save_to_idea,
            )
        )

    def financial_data_checklist(product_type: str = "", provided: list[str] | None = None) -> dict:
        """Which inputs are needed for a detailed financial analysis (payback, ROI, NPV, IRR,
        break-even, projection) of this kind of product, and which are still missing. Use it when
        the user asks what data is needed, or before calculating with too little information.

        Args:
            product_type: software/app/AI (saas), product/hardware, service/agency, retail/
                restaurant, marketplace/platform, or empty.
            provided: keys the user already gave, e.g. ["price", "fixed_costs_month", "cac"].
        """
        return ctx.run(lambda: ctx.checklist(product_type, provided))

    def security_check(link_or_text: str) -> dict:
        """Check a link or a message for phishing / scam signs: lookalike or punycode domains,
        link shorteners, IP hosts, suspicious endings, masked links, OTP / urgency / prize / fake
        police / fake parcel / investment language, Thai bank-scam patterns, and Google Safe
        Browsing when configured. Returns a verdict (safe / suspicious / dangerous) with reasons
        and what to do. It is an estimate: the page is never opened.

        Args:
            link_or_text: the link or the whole message text to check (max 6000 characters).
        """
        return ctx.run(lambda: ctx.security_check(link_or_text))

    def verify_slip(
        bank: str = "",
        amount: float | None = None,
        date_iso: str = "",
        time_hhmm: str = "",
        ref_no: str = "",
        sender: str = "",
        receiver: str = "",
        expected_amount: float | None = None,
        expected_receiver: str = "",
    ) -> dict:
        """Payment-slip CHECKLIST for a transfer slip image you can see: pass what is printed on
        it and the code flags inconsistencies (future or very old date, odd reference format,
        amount different from what was requested, same sender and receiver, a reference already
        in the ledger). It can NEVER confirm a slip is genuine; tell the user to check the money
        in their own bank app or scan the slip QR with the bank's official verification.

        Args:
            bank: bank / wallet name on the slip.
            amount: amount transferred.
            date_iso: transfer date as YYYY-MM-DD (convert Buddhist years).
            time_hhmm: transfer time HH:MM.
            ref_no: transaction reference number.
            sender: sender name as printed.
            receiver: receiver name as printed.
            expected_amount: the amount the user expected / invoiced, if they said.
            expected_receiver: the receiver the user expected, if they said.
        """
        return ctx.run(
            lambda: ctx.verify_slip(
                bank=bank or None,
                amount=amount,
                date_iso=date_iso or None,
                time_hhmm=time_hhmm or None,
                ref_no=ref_no or None,
                sender=sender or None,
                receiver=receiver or None,
                expected_amount=expected_amount,
                expected_receiver=expected_receiver or None,
            )
        )

    def market_benchmark(topic: str, our_product: str = "") -> dict:
        """Benchmark a market on the web (Google-grounded, numbered citations): market gaps, a
        competitor matrix (pricing, strengths, weaknesses, Thai-language support, sources) and
        where our product wins and loses. Rows without a real source are dropped. Takes 20-60
        seconds; the full report is attached as PDF. Default 'our product' is Salid itself
        (a factual description in code); pass `our_product` only for the user's own product.
        Never put private team data in the arguments: it goes to Google.

        Args:
            topic: the market / category, e.g. "AI meeting summary tools for Thai teams".
            our_product: short factual description of the user's product, empty = Salid.
        """
        return ctx.run(lambda: ctx.benchmark(topic, our_product))

    tools: list[Callable] = [
        unit_economics,
        price_from_cost,
        investment_metrics,
        financial_projection,
        financial_data_checklist,
        security_check,
        verify_slip,
    ]
    if web:
        tools.append(market_benchmark)
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]
    return tools
