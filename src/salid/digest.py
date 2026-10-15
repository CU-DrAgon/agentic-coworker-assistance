from __future__ import annotations

from datetime import date

from .qa import baht
from .store import Store


def build_daily_digest(store: Store, guild_id: int, day: date | str) -> str:
    d = day.isoformat() if isinstance(day, date) else day
    total = store.summarize(guild_id, d, d, "none")
    if total["count"] == 0:
        return f"📒 สรุปค่าใช้จ่ายประจำวันที่ {d}\nไม่มีใบเสร็จที่บันทึกในวันนี้"
    cats = store.summarize(guild_id, d, d, "category")["groups"]
    vendors = store.summarize(guild_id, d, d, "vendor")["groups"][:3]
    flagged = store.flagged(guild_id, d, d)
    lines = [
        f"📒 สรุปค่าใช้จ่ายประจำวันที่ {d}",
        f"ยอดรวม: {baht(total['total'])} ({total['count']} ใบเสร็จ)",
        "",
        "แยกตามหมวด:",
    ]
    lines += [f"• {g['key']}: {baht(g['total'])} ({g['count']})" for g in cats]
    lines += ["", "ร้านที่ใช้จ่ายสูงสุด 3 อันดับ:"]
    lines += [f"{i}. {g['key']}: {baht(g['total'])}" for i, g in enumerate(vendors, 1)]
    if flagged:
        lines += ["", f"⚠️ รายการที่ต้องตรวจสอบ ({len(flagged)}):"]
        for r in flagged[:10]:
            why = "อาจซ้ำ" if r.possible_duplicate_of else "ข้อมูลไม่ชัด"
            lines.append(
                f"• #{r.id} {r.vendor or 'ไม่ทราบ'} {baht(r.total) if r.total is not None else 'ไม่มียอด'} ({why}) โดย {r.submitter_name}"
            )
    return "\n".join(lines)
