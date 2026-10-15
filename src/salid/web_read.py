"""read_web: read links (web pages, YouTube) and search the internet with Google-native tools.

Gemini cannot mix built-in tools (url_context, google_search) with function tools in one agent, so
this is a plain function tool for the main agent that makes a SEPARATE `generate_content` call
carrying only the built-in tools. Our servers never fetch the URLs themselves (Google does), so
there is no SSRF surface.

Data isolation: the outbound request is built here from (a) the current user's message (text and
links) and (b) the model-supplied `request`, truncated to REQUEST_CHARS. Ledger, memory and chat
history never enter this call; they could only leak through `request`, hence the hard cap.
Only domains are logged, never full URLs or text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from urllib.parse import urlparse

from google.genai import types

from . import llm

logger = logging.getLogger(__name__)
MAX_URLS = 5
REQUEST_CHARS = 500  # the model-written part of the outbound request
USER_TEXT_CHARS = 4000  # the user's own message part
MAX_OUTPUT_TOKENS = 4096  # thinking tokens count against this
MAX_SOURCES = 10
_URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+", re.IGNORECASE)
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}

READ_WEB_INSTRUCTION = (
    "Read the URLs in the request and/or search the web; return a faithful summary in the "
    "request's language with the source URLs; do not invent; say clearly if a page is "
    "unreadable, paywalled or login-only. Never claim to have read something you could not."
)


def extract_urls(text: str) -> list[str]:
    """http(s) URLs in order of appearance, trailing punctuation removed, duplicates dropped."""
    out: list[str] = []
    for m in _URL_RE.findall(text or ""):
        u = m.rstrip(".,;:!?’”")
        if u not in out:
            out.append(u)
    return out


def domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def is_youtube(url: str) -> bool:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return False
    if host == "youtu.be":
        return len(p.path) > 1
    return p.path == "/watch" or p.path.startswith(("/shorts/", "/live/"))


def build_contents(user_text: str, request: str) -> tuple[list[types.Part], list[str], str]:
    """(parts, urls used, note). URLs come from the user's message first, then the request."""
    req = (request or "").strip()[:REQUEST_CHARS]
    user = (user_text or "").strip()[:USER_TEXT_CHARS]
    urls = extract_urls(user)
    urls += [u for u in extract_urls(req) if u not in urls]
    note = (
        f"{len(urls) - MAX_URLS} more link(s) ignored (limit {MAX_URLS} per call)"
        if len(urls) > MAX_URLS
        else ""
    )
    for extra in urls[MAX_URLS:]:  # ignored links must not reach Google through the text either
        user, req = user.replace(extra, "[link omitted]"), req.replace(extra, "[link omitted]")
    urls = urls[:MAX_URLS]
    text = f"Request: {req}\n" if req else ""
    if user and user != req:
        text += f"User message: {user}\n"
    if urls:
        text += "URLs:\n" + "\n".join(urls)
    parts = [types.Part(text=text.strip() or "Search the web for the user's question.")]
    parts += [types.Part.from_uri(file_uri=u, mime_type="video/*") for u in urls if is_youtube(u)]
    return parts, urls, note


def _sources(resp) -> tuple[list[dict], list[str]]:
    sources: list[dict] = []
    unreadable: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        gm = getattr(cand, "grounding_metadata", None)
        for ch in getattr(gm, "grounding_chunks", None) or []:
            web = getattr(ch, "web", None)
            uri = getattr(web, "uri", None)
            if uri and all(s["uri"] != uri for s in sources):
                sources.append({"title": getattr(web, "title", None) or domain(uri), "uri": uri})
        um = getattr(cand, "url_context_metadata", None)
        for meta in getattr(um, "url_metadata", None) or []:
            uri = getattr(meta, "retrieved_url", None)
            status = str(getattr(meta, "url_retrieval_status", "") or "")
            if not uri:
                continue
            if status and "SUCCESS" not in status.upper():
                unreadable.append(uri)
            elif all(s["uri"] != uri for s in sources):
                sources.append({"title": domain(uri), "uri": uri})
    return sources[:MAX_SOURCES], unreadable


def web_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=READ_WEB_INSTRUCTION,
        tools=[
            types.Tool(url_context=types.UrlContext()),
            types.Tool(google_search=types.GoogleSearch()),
        ],
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def read_web_call(client, model: str, user_text: str, request: str) -> dict:
    """The blocking Gemini call. Returns {text, sources[, unreadable, note]} or {error}."""
    parts, urls, note = build_contents(user_text, request)
    try:
        resp = llm.generate(client, model, [types.Content(role="user", parts=parts)], web_config())
    except Exception as e:  # never log request text
        logger.warning("read_web failed (%s)", type(e).__name__)
        return {"error": f"could not read or search the web ({type(e).__name__})"}
    text = (getattr(resp, "text", None) or "").strip()
    sources, unreadable = _sources(resp)
    if not text:
        return {"error": "the web read returned no text", "sources": sources}
    out: dict = {"text": text, "sources": sources}
    if unreadable:
        out["unreadable"] = unreadable
    if note:
        out["note"] = note
    return out


def make_read_web(
    client, model: str, user_text: str, calls: list[tuple[str, dict]]
) -> Callable[[str], dict]:
    """The `read_web` function tool for one request. `user_text` is the current user message."""

    def read_web(request: str) -> dict:
        """Read links (web pages, YouTube videos) and/or search the internet with Google Search.
        Use it when the message has links, or asks about current or online information. Do NOT
        put ledger, chat memory or other team data in `request`: it goes to Google.

        Args:
            request: what to read or find, in the user's language (max 500 characters); the
                links in the user's message are attached automatically.
        """
        urls = extract_urls(user_text) + extract_urls(request)
        calls.append(("read_web", {"domains": sorted({domain(u) for u in urls if domain(u)})}))
        logger.info("read_web: %d url(s), domains=%s", len(urls), calls[-1][1]["domains"])
        return read_web_call(client, model, user_text, request)

    return read_web
