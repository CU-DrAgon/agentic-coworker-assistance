"""Render a sample Thai slide deck offline (no model, no ledger) for checking the look by hand.

    uv run python scripts/salid_sample_deck.py /tmp/salid-sample-deck.pptx

Open the file in PowerPoint, Keynote and Google Slides: Thai text, layout and charts.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from salid.deck import ChartData, DeckSpec, render_deck

MONTH = {"start_date": "2026-10-01", "end_date": "2026-10-31"}
SPEC = {
    "title": "สรุปค่าใช้จ่ายและการประชุม เดือนตุลาคม 2026",
    "subtitle": "ร้านกาแฟสาขาสยาม · ทีมปฏิบัติการ",
    "slides": [
        {"kind": "section", "title": "ภาพรวมเดือนนี้"},
        {
            "kind": "bullets",
            "title": "สิ่งที่ตกลงกันในที่ประชุม",
            "bullets": [
                "เลื่อนเปิดสาขาใหม่เป็นวันศุกร์ที่ 23 ตุลาคม",
                "แอนรับผิดชอบงบโฆษณาเฟซบุ๊ก ไม่เกิน 20,000 บาท",
                "บอยสั่งน้ำแข็งเพิ่มเป็นวันละ 10 ถุงช่วงเทศกาล",
                "ทบทวนเมนูเครื่องดื่มเย็นก่อนสิ้นเดือน",
            ],
            "notes": "รายละเอียดจากการประชุมวันที่ 14 ตุลาคม\nผู้เข้าร่วม: แอน บอย ต้อย",
        },
        {
            "kind": "two_column",
            "title": "ก่อน / หลัง ปรับรอบสั่งวัตถุดิบ",
            "bullets": ["สั่งทุกวัน", "ของเหลือทิ้งสัปดาห์ละ 3 กิโล", "ค่าส่งรวม 4,200 บาท"],
            "right": ["สั่งวันเว้นวัน", "ของเหลือทิ้งไม่ถึง 1 กิโล", "ค่าส่งรวม 2,100 บาท"],
        },
        {
            "kind": "chart",
            "title": "ค่าใช้จ่ายตามหมวด (บาท)",
            "chart": {"group_by": "category", **MONTH},
        },
        {
            "kind": "chart",
            "title": "สัดส่วนค่าใช้จ่าย",
            "chart": {"group_by": "category", "style": "pie", **MONTH},
        },
    ],
}
CATEGORIES = ChartData(  # 8 slices: checks that every pie slice gets its own colour
    categories=[
        "วัตถุดิบ",
        "ค่าเช่า",
        "ค่าแรง",
        "ค่าเดินทาง",
        "สาธารณูปโภค",
        "การตลาด",
        "อุปกรณ์",
        "อื่นๆ",
    ],
    series={"THB": [48250.0, 30000.0, 18500.0, 6420.5, 5310.0, 4100.0, 2750.0, 2890.0]},
)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/salid-sample-deck.pptx")
    data = render_deck(
        DeckSpec.model_validate(SPEC), {3: CATEGORIES, 4: CATEGORIES}, lang="th", today=date.today()
    )
    out.write_bytes(data)
    print(f"{out} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
