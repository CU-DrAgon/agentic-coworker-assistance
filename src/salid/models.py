from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CATEGORIES = ["วัตถุดิบ", "อุปกรณ์", "ค่าเดินทาง", "สาธารณูปโภค", "ค่าเช่า", "ค่าแรง", "การตลาด", "อื่นๆ"]
Category = Literal["วัตถุดิบ", "อุปกรณ์", "ค่าเดินทาง", "สาธารณูปโภค", "ค่าเช่า", "ค่าแรง", "การตลาด", "อื่นๆ"]


class Item(BaseModel):
    name: str
    qty: float | None = None
    amount: float | None = None


class ReceiptExtraction(BaseModel):
    """Structured output schema for Gemini."""

    is_receipt: bool = Field(description="false if the image/PDF is not a receipt, invoice or bill")
    vendor: str | None = None
    receipt_date: str | None = Field(
        None, description="ISO date YYYY-MM-DD (Gregorian, convert Thai Buddhist years), or null"
    )
    currency: str = "THB"
    subtotal: float | None = None
    vat: float | None = None
    total: float | None = None
    category: Category = "อื่นๆ"
    items: list[Item] = Field(default_factory=list)
    payment_method: str | None = None
    confidence: float = Field(0.0, description="0-1 confidence the extraction is correct")
    notes: str | None = None


class Receipt(BaseModel):
    id: int | None = None
    submitter_id: int
    submitter_name: str
    channel_id: int = 0
    message_id: int = 0
    reply_message_id: int | None = None
    sha256: str
    vendor: str | None = None
    receipt_date: str | None = None
    effective_date: str  # receipt_date or submission date (Bangkok); used for all aggregation
    currency: str = "THB"
    subtotal: float | None = None
    vat: float | None = None
    total: float | None = None
    category: str = "อื่นๆ"
    items: list[Item] = Field(default_factory=list)
    payment_method: str | None = None
    confidence: float = 0.0
    notes: str | None = None
    context: str | None = None
    possible_duplicate_of: int | None = None
    created_at: str = ""
