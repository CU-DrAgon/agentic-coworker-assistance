"""Text helpers for the chat platforms: splitting, Slack mrkdwn, LINE plain text, Slack input.

Pure functions, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# ------------------------------------------------------------------ splitting


def utf16_len(text: str) -> int:
    """Length in UTF-16 code units (how LINE counts characters of a text message)."""
    return len(text.encode("utf-16-le")) // 2


def _cut(piece: str, limit: int, measure: Callable[[str], int]) -> list[str]:
    """Hard-split `piece` into parts of at most `limit` (last resort, no good boundary)."""
    out, cur = [], ""
    for ch in piece:
        if cur and measure(cur + ch) > limit:
            out.append(cur)
            cur = ""
        cur += ch
    if cur:
        out.append(cur)
    return out


def split_text(text: str, limit: int, measure: Callable[[str], int] = len) -> list[str]:
    """Split `text` into messages of at most `limit` (by `measure`), at paragraph, line, sentence
    or word boundaries in that order. An open ``` code fence is closed at the end of a part and
    reopened in the next, so every part renders on its own."""
    text = (text or "").strip()
    if not text:
        return []
    if measure(text) <= limit:
        return [text]
    # with code fences: room for a closing and a reopening fence in every part
    limit_in = max(limit - 8, 1) if "```" in text else limit
    units: list[str] = []  # smallest pieces that still keep their boundary character
    for para in re.split(r"(?<=\n\n)", text):
        if measure(para) <= limit_in:
            units.append(para)
            continue
        for line in re.split(r"(?<=\n)", para):
            if measure(line) <= limit_in:
                units.append(line)
                continue
            for word in re.split(r"(?<=[.!?。！？ ])", line):
                units.extend([word] if measure(word) <= limit_in else _cut(word, limit_in, measure))
    parts, cur = [], ""
    for u in units:
        if cur and measure(cur + u) > limit_in:
            parts.append(cur)
            cur = ""
        cur += u
    if cur:
        parts.append(cur)
    parts = [p.strip() for p in parts if p.strip()]
    return _balance_fences(parts, limit, measure)


def _balance_fences(parts: list[str], limit: int, measure: Callable[[str], int]) -> list[str]:
    out, open_fence = [], False
    for p in parts:
        if open_fence:
            p = "```\n" + p
        open_fence = p.count("```") % 2 == 1
        if open_fence:
            p = p + "\n```"
        out.append(p)
    # re-adding fences can push a part over the limit by a few characters: split those again
    fixed: list[str] = []
    for p in out:
        fixed.extend([p] if measure(p) <= limit else _cut(p, limit, measure))
    return fixed


# ------------------------------------------------------------------ Slack output (mrkdwn)

_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
_INLINE = re.compile(r"`([^`\n]+)`")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fence_tables(text: str) -> str:
    """Slack has no tables: consecutive `| a | b |` rows become a code block (keeps columns)."""
    out, rows = [], []

    def flush():
        if rows:
            body = [r for r in rows if not re.fullmatch(r"\s*\|[\s:|-]+\|\s*", r)]
            out.append("```\n" + "\n".join(body) + "\n```")
            rows.clear()

    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            flush()
            in_fence = not in_fence
            out.append(line)
        elif not in_fence and _TABLE_ROW.match(line):
            rows.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def to_slack(text: str) -> str:
    """Markdown from the model -> Slack mrkdwn. Escapes `& < >` (so `<!channel>`, `<@U..>` and
    link syntax from the model are inert), maps bold / italic / strike / headings / bullets /
    links, and keeps code spans and fences verbatim."""
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return f"\x00{len(stash) - 1}\x00"

    s = _fence_tables(text or "")
    s = _FENCE.sub(lambda m: keep("```\n" + _escape(m.group(1).strip("\n")) + "\n```"), s)
    s = _INLINE.sub(lambda m: keep("`" + _escape(m.group(1)) + "`"), s)
    s = _escape(s)
    s = re.sub(
        r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", lambda m: keep(f"*{m[1]}*"), s, flags=re.M
    )  # headings -> bold
    s = re.sub(
        r"\*\*(?!\s)(.+?)(?<!\s)\*\*|__(?!\s)(.+?)(?<!\s)__",
        lambda m: keep(f"*{m[1] or m[2]}*"),
        s,
    )
    s = re.sub(r"^\s*([-*_])(?:\s*\1){2,}\s*$", "", s, flags=re.M)  # horizontal rules
    s = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", s, flags=re.M)  # bullets
    s = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", lambda m: keep(f"<{m[2]}|{m[1]}>"), s)
    s = re.sub(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])", r"_\1_", s)  # *italic*
    s = re.sub(
        r"\*\*(?!\s)(.+?)(?<!\s)\*\*|__(?!\s)(.+?)(?<!\s)__", lambda m: f"*{m[1] or m[2]}*", s
    )
    s = re.sub(r"~~(?!\s)(.+?)(?<!\s)~~", r"~\1~", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], s)


# ------------------------------------------------------------------ LINE output (plain text)


def to_line(text: str) -> str:
    """Markdown from the model -> plain text for LINE (it renders no markup at all)."""
    s = text or ""
    s = _FENCE.sub(lambda m: m.group(1).strip("\n"), s)
    s = _INLINE.sub(r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", r"\1", s, flags=re.M)
    s = re.sub(r"^\s*([-*_])(?:\s*\1){2,}\s*$", "", s, flags=re.M)
    s = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", s, flags=re.M)
    s = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
        lambda m: m[2] if m[1].strip() == m[2] else f"{m[1]} ({m[2]})",
        s,
    )
    s = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m[1] or m[2], s, flags=re.S)
    s = re.sub(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])", r"\1", s)
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


# ------------------------------------------------------------------ Slack input

_SLACK_TOKEN = re.compile(r"<([^<>]+)>")


def slack_to_plain(
    text: str,
    bot_user_id: str = "",
    name_of: Callable[[str], str | None] = lambda uid: None,
) -> str:
    """A Slack message's text as plain text: Salid's own `<@bot>` removed, other people's
    `<@U123>` -> `@name`, `<#C1|general>` -> `#general`, `<url|label>` -> `label (url)`,
    `<!here>` -> `@here`, entities unescaped."""

    def one(m: re.Match) -> str:
        body = m.group(1)
        head, _, label = body.partition("|")
        if head.startswith("@"):
            uid = head[1:]
            if uid == bot_user_id:
                return ""
            who = label or name_of(uid)
            return f"@{who}" if who else f"@{uid}"
        if head.startswith("#"):
            return f"#{label}" if label else head
        if head.startswith("!"):
            word = head[1:].split("^")[0]
            return label if label.startswith("@") else f"@{word}"
        if head.startswith("mailto:"):
            return label or head[7:]
        return head if not label or label == head else f"{label} ({head})"

    out = _SLACK_TOKEN.sub(one, text or "")
    out = out.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def slack_mentions(text: str) -> list[str]:
    """User ids mentioned in a raw Slack message (`<@U123>` / `<@U123|name>`)."""
    return re.findall(r"<@([A-Z0-9]+)(?:\|[^>]*)?>", text or "")
