"""Public privacy policy and terms pages (required by the Google OAuth consent screen)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from meeting_scribe.config import get_settings

STATIC = Path(__file__).parent / "static"
router = APIRouter()


def _page(name: str) -> HTMLResponse:
    html = (STATIC / name).read_text(encoding="utf-8")
    html = html.replace("__CONTACT_EMAIL__", get_settings().legal_contact_email)
    return HTMLResponse(html, headers={"Cache-Control": "public, max-age=3600"})


@router.get("/privacy", include_in_schema=False)
def privacy_page() -> HTMLResponse:
    return _page("privacy.html")


@router.get("/terms", include_in_schema=False)
def terms_page() -> HTMLResponse:
    return _page("terms.html")
