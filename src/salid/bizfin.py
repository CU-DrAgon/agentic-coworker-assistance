"""Business finance calculators: pure, deterministic math. NO model ever does arithmetic.

Money is `Decimal` (THB by default, any currency label). IRR is a numpy-free bisection. Every
function returns plain dicts of numbers plus a compact `text` rendering (Thai or English) with the
formulas shown, so the agent only has to relay it. The agent tools and the Discord glue are in
`bizfin_tools.py`; spreadsheets are built by `to_xlsx` / `to_csv` here.
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext

getcontext().prec = 34

VAT_RATE = Decimal("0.07")  # Thai VAT
CENT = Decimal("0.01")
MAX_PERIODS = 600
MAX_MONTHS = 36
MIN_MONTHS = 12


class FinError(ValueError):
    """A bad or impossible input. The message is safe to show the user."""


# ------------------------------------------------------------------ number helpers
def D(x, name: str = "value") -> Decimal:  # noqa: N802
    """Decimal from an int/float/str ("1,200" and "฿" tolerated). Never from a bool/NaN/inf."""
    if isinstance(x, bool) or x is None:
        raise FinError(f"{name} must be a number")
    if isinstance(x, Decimal):
        d = x
    elif isinstance(x, int):
        d = Decimal(x)
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise FinError(f"{name} must be a finite number")
        d = Decimal(repr(x))
    else:
        s = re.sub(r"[,\s฿$€£]|THB|บาท", "", str(x), flags=re.I)
        try:
            d = Decimal(s)
        except InvalidOperation:
            raise FinError(f"{name} must be a number") from None
    if not d.is_finite():
        raise FinError(f"{name} must be a finite number")
    return d


def q2(d: Decimal) -> Decimal:
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def money(d: Decimal | float | int | None, cur: str = "THB") -> str:
    if d is None:
        return "-"
    return f"{q2(Decimal(d)):,.2f} {cur}"


def num(d: Decimal | None, digits: int = 2) -> float | None:
    """A float for JSON (None stays None)."""
    if d is None:
        return None
    return float(round(d, digits))


def pct(d: Decimal | None, digits: int = 1) -> str:
    return "-" if d is None else f"{float(round(d, digits)):,.{digits}f}%"


def _pos(x, name: str, allow_zero: bool = True) -> Decimal:
    d = D(x, name)
    if d < 0 or (d == 0 and not allow_zero):
        raise FinError(f"{name} must be {'>= 0' if allow_zero else '> 0'}")
    return d


_ITEM = re.compile(
    r"^\s*(?P<name>.*?)\s*[:=\-]?\s*(?P<amt>-?[\d,]+(?:\.\d+)?)\s*(?:THB|บาท|฿)?\s*$"
)


def parse_items(items) -> list[tuple[str, Decimal]]:
    """[(name, amount)] from numbers or strings such as "วัตถุดิบ: 40" / "box=12.5" / "30"."""
    out: list[tuple[str, Decimal]] = []
    for i, it in enumerate(items or [], 1):
        if isinstance(it, int | float | Decimal) and not isinstance(it, bool):
            out.append((f"#{i}", D(it)))
            continue
        if isinstance(it, dict):
            amt = it.get("amount", it.get("cost", it.get("value")))
            out.append((str(it.get("name") or f"#{i}")[:60], D(amt, "cost item")))
            continue
        m = _ITEM.match(str(it))
        if not m:
            raise FinError(f"cannot read the cost item {str(it)[:40]!r}; use 'name: amount'")
        out.append(((m["name"] or f"#{i}")[:60], D(m["amt"].replace(",", ""), "cost item")))
    return out


# ------------------------------------------------------------------ unit economics
def unit_economics(
    fixed_costs_month,
    variable_cost_unit,
    price,
    expected_units_month=None,
    initial_investment=0,
    currency: str = "THB",
    lang: str = "en",
) -> dict:
    """Contribution margin, gross margin, break-even units / revenue / months. `price` and costs
    are ex-VAT per unit; fixed costs per month."""
    fixed = _pos(fixed_costs_month, "fixed costs")
    var = _pos(variable_cost_unit, "variable cost per unit")
    p = _pos(price, "price", allow_zero=False)
    invest = _pos(initial_investment or 0, "initial investment")
    cm = p - var
    out: dict = {
        "currency": currency,
        "price": num(p),
        "variable_cost_unit": num(var),
        "fixed_costs_month": num(fixed),
        "contribution_margin_unit": num(cm),
        "contribution_margin_pct": num(cm / p * 100),
        "gross_margin_pct": num((p - var) / p * 100),
    }
    be_units = be_rev = None
    if cm <= 0:
        out["break_even"] = "never"
        out["warning"] = "price is not above the variable cost: every sale loses money"
    else:
        be_units = (fixed / cm).to_integral_value(rounding="ROUND_CEILING")
        be_rev = be_units * p
        out["break_even_units_month"] = int(be_units)
        out["break_even_revenue_month"] = num(be_rev)
    months = profit = None
    if expected_units_month is not None:
        units = _pos(expected_units_month, "expected units per month")
        profit = cm * units - fixed
        out["expected_units_month"] = num(units, 0)
        out["expected_profit_month"] = num(profit)
        out["expected_revenue_month"] = num(units * p)
        if invest > 0 and profit > 0:
            months = (invest / profit).quantize(Decimal("0.1"), rounding="ROUND_CEILING")
            out["months_to_recoup_investment"] = float(months)
        elif invest > 0:
            out["months_to_recoup_investment"] = None
            out["warning"] = (
                "expected volume does not cover fixed costs: the investment is not recouped"
            )
    if invest > 0 and cm > 0:
        out["break_even_units_total_incl_investment"] = int(
            ((fixed + invest) / cm).to_integral_value(rounding="ROUND_CEILING")
        )
    out["text"] = _ue_text(out, p, var, fixed, cm, invest, currency, lang)
    return out


def _ue_text(out, p, var, fixed, cm, invest, cur, lang) -> str:
    th = lang == "th"
    L = [
        ("หน่วยเศรษฐศาสตร์ (ราคาไม่รวม VAT)" if th else "Unit economics (prices ex-VAT)"),
        f"- {'ราคาต่อหน่วย' if th else 'Price per unit'}: {money(p, cur)}",
        f"- {'ต้นทุนผันแปรต่อหน่วย' if th else 'Variable cost per unit'}: {money(var, cur)}",
        f"- {'ต้นทุนคงที่ต่อเดือน' if th else 'Fixed costs per month'}: {money(fixed, cur)}",
        f"- {'กำไรส่วนเกินต่อหน่วย' if th else 'Contribution margin/unit'} = {'ราคา - ต้นทุนผันแปร' if th else 'price - variable cost'}"
        f" = {money(cm, cur)} ({pct(out['contribution_margin_pct'] and Decimal(str(out['contribution_margin_pct'])))})",
    ]
    if "break_even_units_month" in out:
        L.append(
            f"- {'จุดคุ้มทุน' if th else 'Break-even'} = {'ต้นทุนคงที่ ÷ กำไรส่วนเกินต่อหน่วย' if th else 'fixed ÷ contribution/unit'}"
            f" = {out['break_even_units_month']:,} {'หน่วย/เดือน' if th else 'units/month'}"
            f" ({money(Decimal(str(out['break_even_revenue_month'])), cur)}/{'เดือน' if th else 'month'})"
        )
    else:
        L.append(
            "- "
            + ("ไม่มีจุดคุ้มทุน: ราคาไม่สูงกว่าต้นทุนผันแปร" if th else "No break-even: price <= variable cost")
        )
    if "expected_profit_month" in out:
        L.append(
            f"- {'กำไร/ขาดทุนต่อเดือนที่ปริมาณคาด' if th else 'Monthly profit at expected volume'}: "
            f"{money(Decimal(str(out['expected_profit_month'])), cur)}"
        )
    if out.get("months_to_recoup_investment"):
        L.append(
            f"- {'คืนทุนเงินลงทุนแรกเริ่มใน' if th else 'Initial investment recouped in'} "
            f"{out['months_to_recoup_investment']:g} {'เดือน' if th else 'months'}"
        )
    if out.get("warning"):
        L.append(f"! {out['warning']}")
    return "\n".join(L)


# ------------------------------------------------------------------ pricing
def vat_split(net: Decimal) -> dict:
    vat = net * VAT_RATE
    return {"net": q2(net), "vat": q2(vat), "gross": q2(net + vat)}


def _quartiles(prices: list[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    s = sorted(prices)
    if len(s) == 1:
        return s[0] * Decimal("0.9"), s[0], s[0] * Decimal("1.1")
    qs = statistics.quantiles([float(x) for x in s], n=4, method="inclusive")
    return Decimal(repr(qs[0])), Decimal(repr(qs[1])), Decimal(repr(qs[2]))


def price_from_cost(
    cost_items,
    target_margin_pct=None,
    markup_pct=None,
    competitor_prices=None,
    competitors_include_vat: bool = False,
    currency: str = "THB",
    lang: str = "en",
) -> dict:
    """Cost per unit -> cost-plus price and entry / standard / premium tiers. A margin is on the
    price (price = cost / (1 - m)); a markup is on the cost (price = cost x (1 + k)). Competitor
    prices give a value-based band (quartiles). Tiers never go below cost; VAT 7% is separate."""
    items = parse_items(cost_items)
    if not items:
        raise FinError("give at least one cost item")
    cost = sum((a for _n, a in items), Decimal(0))
    if cost <= 0:
        raise FinError("total cost must be > 0")
    if target_margin_pct is not None and markup_pct is not None:
        raise FinError("give a target margin OR a markup, not both")
    basis = "margin"
    if markup_pct is not None:
        k = _pos(markup_pct, "markup %")
        cost_plus = cost * (1 + k / 100)
        basis = "markup"
    else:
        m = _pos(30 if target_margin_pct is None else target_margin_pct, "target margin %")
        if m >= 100:
            raise FinError("target margin must be below 100%")
        cost_plus = cost / (1 - m / 100)
    comp = [_pos(c, "competitor price", allow_zero=False) for c in competitor_prices or []]
    if comp and competitors_include_vat:
        comp = [c / (1 + VAT_RATE) for c in comp]
    band = None
    if comp:
        lo, mid, hi = _quartiles(comp)
        band = {"low": q2(lo), "median": q2(mid), "high": q2(hi), "n": len(comp)}
        entry = max(lo, cost * Decimal("1.05"))
        standard = max(mid, cost_plus)
        premium = max(hi, standard * Decimal("1.15"))
    else:
        entry = max(cost_plus * Decimal("0.85"), cost * Decimal("1.05"))
        standard = cost_plus
        premium = cost_plus * Decimal("1.30")
    tiers = []
    for name, price in (("entry", entry), ("standard", standard), ("premium", premium)):
        price = q2(price)
        tiers.append(
            {
                "tier": name,
                "price_ex_vat": price,
                "vat": q2(price * VAT_RATE),
                "price_inc_vat": q2(price * (1 + VAT_RATE)),
                "margin_pct": q2((price - cost) / price * 100),
                "markup_pct": q2((price - cost) / cost * 100),
            }
        )
    warn = []
    if comp and band and q2(cost_plus) > band["high"]:
        warn.append("cost-plus price is above every competitor: check costs or the value story")
    if comp and band and q2(cost_plus) < band["low"]:
        warn.append("cost-plus price is below the competitor range: room to price higher")
    out = {
        "currency": currency,
        "cost_total": num(cost),
        "cost_items": [{"name": n, "amount": num(a)} for n, a in items],
        "cost_plus_price": num(q2(cost_plus)),
        "basis": basis,
        "competitor_band": {k: (num(v, 2) if k != "n" else v) for k, v in (band or {}).items()}
        or None,
        "tiers": [
            {k: (num(v) if isinstance(v, Decimal) else v) for k, v in t.items()} for t in tiers
        ],
        "vat_rate_pct": 7,
        "warnings": warn,
    }
    out["text"] = _price_text(
        out,
        tiers,
        cost,
        cost_plus,
        basis,
        band,
        target_margin_pct,
        markup_pct,
        currency,
        lang,
        warn,
    )
    return out


def _price_text(out, tiers, cost, cost_plus, basis, band, margin, markup, cur, lang, warn) -> str:
    th = lang == "th"
    formula = (
        f"{'ราคา = ต้นทุน ÷ (1 - มาร์จิน)' if th else 'price = cost / (1 - margin)'}"
        if basis == "margin"
        else f"{'ราคา = ต้นทุน × (1 + markup)' if th else 'price = cost x (1 + markup)'}"
    )
    L = [
        ("ตั้งราคาจากต้นทุน" if th else "Pricing from cost"),
        f"- {'ต้นทุนรวมต่อหน่วย' if th else 'Total cost per unit'}: {money(cost, cur)}",
        f"- {'ราคา cost-plus' if th else 'Cost-plus price'}: {money(q2(cost_plus), cur)} ({formula})",
    ]
    if band:
        L.append(
            f"- {'ช่วงราคาคู่แข่ง (ไม่รวม VAT)' if th else 'Competitor band (ex-VAT)'}: "
            f"{money(band['low'], cur)} / {money(band['median'], cur)} / {money(band['high'], cur)} (n={band['n']})"
        )
    L.append(
        f"| {'แพ็กเกจ' if th else 'Tier'} | {'ราคา' if th else 'Price'} | VAT 7% | {'รวม VAT' if th else 'Incl. VAT'} | {'มาร์จิน' if th else 'Margin'} |"
    )
    L.append("|---|---:|---:|---:|---:|")
    for t in tiers:
        L.append(
            f"| {t['tier']} | {t['price_ex_vat']:,.2f} | {t['vat']:,.2f} | {t['price_inc_vat']:,.2f} | {t['margin_pct']:.1f}% |"
        )
    L += [f"! {w}" for w in warn]
    L.append(
        "ราคาที่เสนอเป็นสมมติฐานจากต้นทุน/คู่แข่งที่ให้มา"
        if th
        else "Suggested prices are assumptions from the given costs / competitors."
    )
    return "\n".join(L)


# ------------------------------------------------------------------ investment metrics
def _flows(cash_flows) -> list[Decimal]:
    flows = [D(c, "cash flow") for c in cash_flows or []]
    if not flows:
        raise FinError("give at least one cash flow")
    if len(flows) > MAX_PERIODS:
        raise FinError(f"at most {MAX_PERIODS} periods")
    return flows


def npv_at(rate: Decimal | float, initial: Decimal, flows: list[Decimal]) -> Decimal:
    r = Decimal(str(rate)) if not isinstance(rate, Decimal) else rate
    total = -initial
    for t, cf in enumerate(flows, 1):
        total += cf / (1 + r) ** t
    return total


def irr(initial: Decimal, flows: list[Decimal]) -> float | None:
    """Per-period IRR by bisection on [-0.99, 1000]; None when there is no sign change (e.g. all
    flows are positive or the sum never repays the investment)."""
    if initial <= 0 or sum(flows, Decimal(0)) <= 0 and all(f <= 0 for f in flows):
        return None
    fl = [float(f) for f in flows]
    i0 = float(initial)

    def f(r: float) -> float:
        return -i0 + sum(cf / (1 + r) ** t for t, cf in enumerate(fl, 1))

    lo, hi = -0.99, 1000.0
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if flo * fhi > 0:
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-9 or (hi - lo) < 1e-12:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def _payback(initial: Decimal, flows: list[Decimal], rate: Decimal | None = None) -> float | None:
    """Periods until cumulative (optionally discounted) flows repay `initial`, interpolated
    linearly inside the period. None = not within the given flows."""
    if initial <= 0:
        return 0.0
    cum = -initial
    for t, cf in enumerate(flows, 1):
        v = cf if rate is None else cf / (1 + rate) ** t
        if cum + v >= 0 and v > 0:
            return float(Decimal(t - 1) + (-cum) / v)
        cum += v
    return None


def investment_metrics(
    initial_investment,
    cash_flows,
    discount_rate_pct=10,
    period: str = "month",
    currency: str = "THB",
    lang: str = "en",
) -> dict:
    """ROI, payback, discounted payback, NPV, IRR. `discount_rate_pct` is the ANNUAL rate; for
    monthly flows it is converted to a monthly rate ((1+r)^(1/12)-1) and the IRR is shown
    both per month and annualised ((1+irr)^12-1)."""
    initial = _pos(initial_investment, "initial investment", allow_zero=False)
    flows = _flows(cash_flows)
    rate_a = D(discount_rate_pct, "discount rate") / 100
    if rate_a <= Decimal("-0.99"):
        raise FinError("discount rate must be above -99%")
    per = "year" if str(period).lower().startswith(("y", "ปี")) else "month"
    ppy = 1 if per == "year" else 12
    r = rate_a if per == "year" else (1 + rate_a) ** (Decimal(1) / 12) - 1
    total = sum(flows, Decimal(0))
    roi = (total - initial) / initial * 100
    npv = npv_at(r, initial, flows)
    pb = _payback(initial, flows)
    dpb = _payback(initial, flows, r)
    i_p = irr(initial, flows)
    i_ann = None if i_p is None else (1 + i_p) ** ppy - 1
    pi = (npv + initial) / initial  # profitability index
    out = {
        "currency": currency,
        "period": per,
        "periods": len(flows),
        "initial_investment": num(initial),
        "total_cash_in": num(total),
        "roi_pct": num(roi),
        "payback_periods": None if pb is None else round(pb, 2),
        "discounted_payback_periods": None if dpb is None else round(dpb, 2),
        "npv": num(npv),
        "irr_period_pct": None if i_p is None else round(i_p * 100, 2),
        "irr_annual_pct": None if i_ann is None else round(i_ann * 100, 2),
        "discount_rate_annual_pct": num(rate_a * 100),
        "discount_rate_period_pct": round(float(r) * 100, 4),
        "profitability_index": round(float(pi), 3),
        "verdict": "npv_positive" if npv > 0 else "npv_negative",
    }
    out["text"] = _inv_text(out, initial, per, currency, lang)
    return out


def _inv_text(o, initial, per, cur, lang) -> str:
    th = lang == "th"
    unit = ("เดือน" if per == "month" else "ปี") if th else per + "s"

    def pb(v):
        return (
            f"{v:g} {unit}"
            if v is not None
            else ("ไม่คืนทุนภายในช่วงที่ให้" if th else "not within the given period")
        )

    L = [
        ("วิเคราะห์การลงทุน" if th else "Investment analysis"),
        f"- {'เงินลงทุนแรกเริ่ม' if th else 'Initial investment'}: {money(initial, cur)}; {o['periods']} {unit}; {'อัตราคิดลด' if th else 'discount rate'} {o['discount_rate_annual_pct']}%/{'ปี' if th else 'yr'}",
        f"- ROI = (Σ{'กระแสเงินสด' if th else 'cash flows'} - {'ลงทุน' if th else 'investment'}) ÷ {'ลงทุน' if th else 'investment'} = {o['roi_pct']:.1f}%",
        f"- Payback: {pb(o['payback_periods'])}",
        f"- {'Discounted payback' if not th else 'Payback แบบคิดลด'}: {pb(o['discounted_payback_periods'])}",
        f"- NPV = -{'ลงทุน' if th else 'I'} + Σ CFt/(1+r)^t = {money(Decimal(str(o['npv'])), cur)}",
    ]
    if o["irr_period_pct"] is None:
        L.append(
            "- IRR: "
            + ("คำนวณไม่ได้ (กระแสเงินสดไม่เปลี่ยนเครื่องหมาย)" if th else "undefined (no sign change)")
        )
    else:
        extra = (
            f", {'ต่อปี' if th else 'annualised'} {o['irr_annual_pct']}%" if per == "month" else ""
        )
        L.append(f"- IRR ({'NPV = 0'}): {o['irr_period_pct']}%/{unit[:5] if th else per}{extra}")
    L.append(
        "- สรุป: NPV เป็นบวก คุ้มค่าตามสมมติฐาน"
        if th and o["npv"] > 0
        else "- สรุป: NPV ติดลบ ยังไม่คุ้มตามสมมติฐาน"
        if th
        else "- Verdict: NPV positive, worthwhile on these assumptions"
        if o["npv"] > 0
        else "- Verdict: NPV negative, not worthwhile on these assumptions"
    )
    return "\n".join(L)


# ------------------------------------------------------------------ projection
SCENARIOS = {  # (new customers x, churn x, ARPU x)
    "low": (Decimal("0.7"), Decimal("1.3"), Decimal("0.9")),
    "base": (Decimal(1), Decimal(1), Decimal(1)),
    "high": (Decimal("1.3"), Decimal("0.7"), Decimal("1.1")),
}


def _project(a: dict, scen: str) -> dict:
    nx, cx, ax = SCENARIOS[scen]
    months = a["months"]
    cust = a["initial_customers"]
    new = a["new_customers_month"] * nx
    churn = min(a["churn_pct_month"] / 100 * cx, Decimal(1))
    arpu = a["arpu_month"] * ax
    cash = a["opening_cash"]
    rows = []
    min_cash, cash_out_month = cash, None
    cum_profit = Decimal(0)
    for m in range(1, months + 1):
        lost = cust * churn
        cust = cust - lost + new
        revenue = cust * arpu
        var = cust * a["variable_cost_customer"]
        mkt = new * a["cac"]
        fixed = a["fixed_costs_month"]
        profit = revenue - var - mkt - fixed
        cash += profit
        cum_profit += profit
        rows.append(
            {
                "month": m,
                "customers": float(round(cust, 1)),
                "new": float(round(new, 1)),
                "revenue": float(q2(revenue)),
                "variable_cost": float(q2(var)),
                "marketing": float(q2(mkt)),
                "fixed_costs": float(q2(fixed)),
                "profit": float(q2(profit)),
                "cash": float(q2(cash)),
            }
        )
        min_cash = min(min_cash, cash)
        if cash_out_month is None and cash < 0:
            cash_out_month = m
        new = new * (1 + a["acquisition_growth_pct_month"] / 100)
    unit_margin = arpu - a["variable_cost_customer"]
    ltv = unit_margin / churn if churn > 0 else None
    be_month = next((r["month"] for r in rows if r["profit"] >= 0), None)
    return {
        "scenario": scen,
        "rows": rows,
        "arpu": num(arpu),
        "churn_pct_month": num(churn * 100),
        "ltv": None if ltv is None else num(ltv),
        "ltv_cac": None if ltv is None or a["cac"] == 0 else round(float(ltv / a["cac"]), 2),
        "cac_payback_months": (
            None if unit_margin <= 0 or a["cac"] == 0 else round(float(a["cac"] / unit_margin), 1)
        ),
        "first_profitable_month": be_month,
        "cash_out_month": cash_out_month,
        "lowest_cash": num(min_cash),
        "end_cash": rows[-1]["cash"],
        "total_revenue": num(sum((Decimal(str(r["revenue"])) for r in rows), Decimal(0))),
        "total_profit": num(cum_profit),
    }


def financial_projection(
    arpu_month,
    new_customers_month,
    churn_pct_month,
    cac,
    fixed_costs_month,
    variable_cost_customer=0,
    initial_customers=0,
    acquisition_growth_pct_month=0,
    opening_cash=0,
    months: int = 24,
    scenario: str = "base",
    currency: str = "THB",
    lang: str = "en",
) -> dict:
    """12-36 month customer / P&L / cash projection with low / base / high scenarios. Churn is a
    monthly rate on the customer base; acquisition grows by `acquisition_growth_pct_month`. LTV =
    (ARPU - cost to serve) / churn; runway = the month cash first goes below zero (cash_out_month)."""
    try:
        n = int(months)
    except (TypeError, ValueError):
        raise FinError("months must be a whole number") from None
    if not MIN_MONTHS <= n <= MAX_MONTHS:
        raise FinError(f"months must be {MIN_MONTHS}-{MAX_MONTHS}")
    scen = str(scenario or "base").lower()
    if scen not in SCENARIOS:
        raise FinError("scenario must be low, base or high")
    churn = _pos(churn_pct_month, "churn %")
    if churn > 100:
        raise FinError("churn must be 0-100% per month")
    a = {
        "months": n,
        "arpu_month": _pos(arpu_month, "ARPU"),
        "new_customers_month": _pos(new_customers_month, "new customers per month"),
        "churn_pct_month": churn,
        "cac": _pos(cac, "CAC"),
        "fixed_costs_month": _pos(fixed_costs_month, "fixed costs"),
        "variable_cost_customer": _pos(variable_cost_customer or 0, "cost to serve per customer"),
        "initial_customers": _pos(initial_customers or 0, "initial customers"),
        "acquisition_growth_pct_month": D(acquisition_growth_pct_month or 0, "growth %"),
        "opening_cash": D(opening_cash or 0, "opening cash"),
    }
    if a["acquisition_growth_pct_month"] <= -100:
        raise FinError("growth must be above -100%")
    results = {s: _project(a, s) for s in SCENARIOS}
    chosen = results[scen]
    out = {
        "currency": currency,
        "months": n,
        "scenario": scen,
        "table": chosen["rows"],
        "metrics": {k: v for k, v in chosen.items() if k != "rows"},
        "scenarios": {
            s: {k: v for k, v in r.items() if k not in ("rows", "scenario")}
            for s, r in results.items()
        },
        "assumptions": {k: (float(v) if isinstance(v, Decimal) else v) for k, v in a.items()},
        "scenario_rules": "low: new customers x0.7, churn x1.3, ARPU x0.9; high: x1.3, x0.7, x1.1",
    }
    out["text"] = _proj_text(out, lang)
    return out


def _proj_text(o, lang) -> str:
    th = lang == "th"
    cur = o["currency"]
    m = o["metrics"]
    L = [
        f"{'ประมาณการการเงิน' if th else 'Financial projection'} {o['months']} {'เดือน' if th else 'months'} ({o['scenario']})",
        f"- LTV = (ARPU - {'ต้นทุนให้บริการ' if th else 'cost to serve'}) ÷ churn = {money(m['ltv'], cur) if m['ltv'] is not None else '-'}; LTV/CAC = {m['ltv_cac'] if m['ltv_cac'] is not None else '-'}; CAC payback = {m['cac_payback_months'] if m['cac_payback_months'] is not None else '-'} {'เดือน' if th else 'mo'}",
        f"- {'เดือนแรกที่กำไร' if th else 'First profitable month'}: {m['first_profitable_month'] or ('ไม่ถึงในช่วงนี้' if th else 'not in this period')}",
        "- Runway: "
        + (
            (f"{'เงินสดหมดเดือนที่' if th else 'cash runs out in month'} {m['cash_out_month']}")
            if m["cash_out_month"]
            else ("เงินสดไม่ติดลบตลอดช่วง" if th else "cash stays positive throughout")
        )
        + f"; {'เงินสดต่ำสุด' if th else 'lowest cash'} {money(m['lowest_cash'], cur)}",
        f"- {'เงินสดปลายงวด' if th else 'End cash'}: {money(m['end_cash'], cur)}; {'กำไรรวม' if th else 'total profit'} {money(m['total_profit'], cur)}",
        f"- {'สถานการณ์' if th else 'Scenarios'} (end cash): "
        + ", ".join(f"{s} {money(v['end_cash'], cur)}" for s, v in o["scenarios"].items()),
        f"| M | {'ลูกค้า' if th else 'Cust.'} | {'รายได้' if th else 'Revenue'} | {'กำไร' if th else 'Profit'} | {'เงินสด' if th else 'Cash'} |",
        "|--:|--:|--:|--:|--:|",
    ]
    rows = o["table"]
    shown = (
        rows
        if len(rows) <= 12
        else [
            r
            for r in rows
            if r["month"] in (1, 2, 3, 6, 9, 12, 18, 24, 30, 36) or r["month"] == len(rows)
        ]
    )
    for r in shown:
        L.append(
            f"| {r['month']} | {r['customers']:,.0f} | {r['revenue']:,.0f} | {r['profit']:,.0f} | {r['cash']:,.0f} |"
        )
    L.append(
        "ตัวเลขเป็นสมมติฐานของผู้ใช้ ไม่ใช่การคาดการณ์ที่รับประกัน"
        if th
        else "Figures follow the user's assumptions; not a guaranteed forecast."
    )
    return "\n".join(L)


# ------------------------------------------------------------------ data checklist
_CORE = [
    (
        "initial_investment",
        "เงินลงทุนแรกเริ่ม (พัฒนา/อุปกรณ์/จดทะเบียน/การตลาดเปิดตัว)",
        "Initial investment (build, equipment, registration, launch marketing)",
    ),
    (
        "fixed_costs_month",
        "ต้นทุนคงที่ต่อเดือน (เงินเดือน เช่า ซอฟต์แวร์ โฆษณาประจำ)",
        "Fixed costs per month (salaries, rent, software, recurring ads)",
    ),
    ("price", "ราคาขายต่อหน่วย/แพ็กเกจ (ไม่รวม VAT)", "Selling price per unit/plan (ex-VAT)"),
    ("discount_rate", "อัตราผลตอบแทนที่ต้องการ/อัตราคิดลดต่อปี", "Required return / annual discount rate"),
    ("horizon", "ระยะเวลาที่ประเมิน (เดือน/ปี)", "Evaluation horizon (months/years)"),
]
_TYPES: dict[str, list[tuple[str, str, str]]] = {
    "saas": [
        ("arpu_month", "รายได้เฉลี่ยต่อผู้ใช้ต่อเดือน (ARPU)", "Average revenue per user per month (ARPU)"),
        (
            "new_customers_month",
            "ลูกค้าใหม่ต่อเดือน และอัตราโต",
            "New customers per month and growth rate",
        ),
        ("churn_pct_month", "อัตราลูกค้าเลิกใช้ต่อเดือน (churn)", "Monthly churn rate"),
        ("cac", "ต้นทุนหาลูกค้าใหม่ (CAC)", "Customer acquisition cost (CAC)"),
        (
            "variable_cost_customer",
            "ต้นทุนให้บริการต่อลูกค้า (API/คลาวด์/ซัพพอร์ต)",
            "Cost to serve per customer (API/cloud/support)",
        ),
        ("opening_cash", "เงินสดตั้งต้น", "Opening cash"),
    ],
    "product": [
        (
            "variable_cost_unit",
            "ต้นทุนต่อชิ้น (วัตถุดิบ ผลิต บรรจุภัณฑ์ ขนส่ง)",
            "Cost per unit (materials, production, packaging, shipping)",
        ),
        ("units_month", "ปริมาณขายที่คาดต่อเดือน", "Expected units sold per month"),
        ("inventory_days", "สต็อก/เครดิตเทอม (วัน)", "Inventory / credit terms (days)"),
        ("channel_fee_pct", "ค่าธรรมเนียมช่องทางขาย/ค่าคอมมิชชัน (%)", "Channel fee / commission (%)"),
    ],
    "service": [
        ("hourly_cost", "ต้นทุนคน/ชั่วโมงหรือต่อโปรเจกต์", "People cost per hour or per project"),
        ("utilization_pct", "อัตราการใช้เวลาที่เรียกเก็บได้ (%)", "Billable utilisation (%)"),
        ("projects_month", "จำนวนงาน/ลูกค้าต่อเดือน", "Projects/clients per month"),
        ("cac", "ต้นทุนหาลูกค้า", "Cost to win a client"),
    ],
    "retail": [
        ("variable_cost_unit", "ต้นทุนสินค้าต่อชิ้น (COGS)", "Cost of goods per unit (COGS)"),
        (
            "units_month",
            "ยอดขายที่คาดต่อเดือน/ลูกค้าต่อวัน",
            "Expected sales per month / customers per day",
        ),
        ("avg_ticket", "ยอดซื้อเฉลี่ยต่อบิล", "Average ticket per sale"),
        ("rent_utilities", "ค่าเช่า/ค่าสาธารณูปโภค", "Rent / utilities"),
        ("waste_pct", "ของเสีย/สต็อกค้าง (%)", "Waste / dead stock (%)"),
    ],
    "marketplace": [
        ("gmv_month", "มูลค่าธุรกรรมต่อเดือน (GMV) และอัตราโต", "Monthly GMV and growth"),
        ("take_rate_pct", "ค่าธรรมเนียมที่เก็บ (take rate %)", "Take rate (%)"),
        ("cac_supply_demand", "CAC ฝั่งผู้ขายและผู้ซื้อ", "CAC for supply and demand sides"),
        ("payment_cost_pct", "ค่าธรรมเนียมชำระเงิน (%)", "Payment processing cost (%)"),
    ],
}
_ALIASES = {
    "software": "saas",
    "app": "saas",
    "subscription": "saas",
    "ai": "saas",
    "แอป": "saas",
    "ซอฟต์แวร์": "saas",
    "สมาชิก": "saas",
    "hardware": "product",
    "สินค้า": "product",
    "manufacturing": "product",
    "ผลิต": "product",
    "consulting": "service",
    "agency": "service",
    "บริการ": "service",
    "shop": "retail",
    "restaurant": "retail",
    "cafe": "retail",
    "ร้าน": "retail",
    "ร้านอาหาร": "retail",
    "e-commerce": "retail",
    "ecommerce": "retail",
    "platform": "marketplace",
    "แพลตฟอร์ม": "marketplace",
}


def resolve_product_type(product_type: str) -> str:
    low = (product_type or "").strip().lower()
    if low in _TYPES:
        return low
    for key, v in _ALIASES.items():
        if key in low:
            return v
    return "generic"


def financial_data_checklist(product_type: str = "", provided=None, lang: str = "en") -> dict:
    """The inputs needed for payback / ROI / NPV / IRR / break-even / projection for this kind of
    product, split into `have` and `missing` by the keys the user already gave."""
    kind = resolve_product_type(product_type)
    inputs = _CORE + _TYPES.get(kind, [])
    have_keys = {str(p).strip().lower() for p in provided or []}
    th = lang == "th"
    rows = [{"key": k, "label": (a if th else b), "have": k in have_keys} for k, a, b in inputs]
    missing = [r for r in rows if not r["have"]]
    L = [
        ("ข้อมูลที่ต้องใช้วิเคราะห์การเงิน" if th else "Data needed for the financial analysis")
        + f" ({kind})"
    ]
    L += [f"- [{'x' if r['have'] else ' '}] {r['label']}" for r in rows]
    L.append(
        ("ยังขาด " if th else "Still missing: ")
        + str(len(missing))
        + (
            " ข้อ ส่งมาเท่าที่มี ที่เหลือสลิดช่วยตั้งสมมติฐานให้ได้ค่ะ"
            if th
            else " item(s); send what you have and Salid can propose assumptions for the rest."
        )
    )
    return {
        "product_type": kind,
        "inputs": rows,
        "missing": [r["key"] for r in missing],
        "text": "\n".join(L),
        "instruction": "Ask the user ONLY for the missing items (a few at a time); then call the calculators.",
    }


# ------------------------------------------------------------------ files
def to_csv(table: list[dict]) -> bytes:
    if not table:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
    return ("﻿" + buf.getvalue()).encode("utf-8")  # BOM: Thai opens right in Excel


def to_xlsx(sheets: dict[str, list[dict]]) -> bytes:
    """One sheet per name from lists of equal-keyed dicts. Raises ImportError without openpyxl."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(re.sub(r"[\[\]\*\?/\\:]", "-", name)[:31] or "Sheet")
        if not rows:
            continue
        keys = list(rows[0])
        ws.append(keys)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([r.get(k) for k in keys])
        for i, k in enumerate(keys, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = max(12, len(str(k)) + 2)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def table_file(name: str, sheets: dict[str, list[dict]]) -> tuple[str, bytes]:
    """(filename, bytes): xlsx when openpyxl exists, else CSV of the first sheet."""
    try:
        return f"{name}.xlsx", to_xlsx(sheets)
    except ImportError:
        first = next(iter(sheets.values()), [])
        return f"{name}.csv", to_csv(first)
