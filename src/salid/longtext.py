"""Splitting long answers for Discord without cutting a section in half.

`split_sections(text, limit)`: pieces of at most `limit` characters. The text is cut into sections
at Markdown headings; a section that fits is never split, and neither is a paragraph, a table or a
code block unless it is longer than `limit` by itself (then at line, sentence or word boundaries).
`needs_attachment` / `summary_of`: an answer that would take more than `MAX_PARTS` messages is sent
as a short summary plus the full text as a file (see `SecretaryBot._deliver`).
"""

from __future__ import annotations

import re

LIMIT = 1900
MAX_PARTS = 3  # more messages than this: summary + attachment
SUMMARY_CHARS = 700
SUMMARY_HEADINGS = 8

_HEADING = re.compile(r"^ {0,3}#{1,6}\s+\S")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|(?<=[ๆฯ])\s+|(?<=ค่ะ)\s+|(?<=คะ)\s+|(?<=ครับ)\s+")


def _blocks(section: str) -> list[str]:
    """Paragraph-sized blocks (blank-line separated; a fenced code block is one block)."""
    out: list[str] = []
    cur: list[str] = []
    fenced = False
    for line in section.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
        if not line.strip() and not fenced:
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def sections_of(text: str) -> list[str]:
    """The text cut before every heading line (outside code fences)."""
    out: list[list[str]] = [[]]
    fenced = False
    for line in text.strip("\n").split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
        if not fenced and _HEADING.match(line) and any(x.strip() for x in out[-1]):
            out.append([])
        out[-1].append(line)
    return ["\n".join(s).strip("\n") for s in out if any(x.strip() for x in s)]


def _cut_long(block: str, limit: int) -> list[str]:
    """One block longer than `limit`: by lines, then sentences, then words, then characters."""
    fence = _FENCE.match(block.split("\n", 1)[0])
    pieces: list[str] = []
    cur = ""

    def push(seg: str) -> None:
        nonlocal cur
        if cur and len(cur) + 1 + len(seg) > limit:
            pieces.append(cur)
            cur = seg
        else:
            cur = f"{cur}\n{seg}" if cur else seg

    for line in block.split("\n"):
        if len(line) <= limit:
            push(line)
            continue
        for sent in _SENTENCE_END.split(line):
            while len(sent) > limit - (len(cur) + 1 if cur else 0):
                # no sentence end: words, then a hard cut as a last resort; fill what is left
                # of the current piece unless that is very little
                room = limit - (len(cur) + 1 if cur else 0)
                if room < limit // 4:
                    pieces.append(cur)
                    cur, room = "", limit
                cut = sent.rfind(" ", 0, room)
                cut = cut if cut > room // 2 else room
                push(sent[:cut].rstrip())
                sent = sent[cut:].lstrip()
            if sent:
                push(sent)
    if cur:
        pieces.append(cur)
    if fence:  # keep every piece a balanced code block
        mark = fence.group(1)
        fixed = []
        for i, p in enumerate(pieces):
            body = p if i == 0 else f"{mark}\n{p}"
            if body.count(mark) % 2:
                body += f"\n{mark}"
            fixed.append(body)
        return fixed
    return pieces


def _pack(blocks: list[str], limit: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for b in blocks:
        for piece in [b] if len(b) <= limit else _cut_long(b, limit):
            if cur and len(cur) + 2 + len(piece) > limit:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def split_sections(text: str, limit: int = LIMIT) -> list[str]:
    """Messages of at most `limit` characters; a section that fits is never cut."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    cur = ""
    for sec in sections_of(text):
        if len(sec) <= limit:
            if cur and len(cur) + 2 + len(sec) > limit:
                out.append(cur)
                cur = sec
            else:
                cur = f"{cur}\n\n{sec}" if cur else sec
            continue
        if cur:
            out.append(cur)
            cur = ""
        parts = _pack(_blocks(sec), limit)
        out += parts[:-1]
        cur = parts[-1] if parts else ""
    if cur:
        out.append(cur)
    return out


def needs_attachment(parts: list[str], max_parts: int = MAX_PARTS) -> bool:
    return len(parts) > max_parts


def _plain(line: str) -> str:
    line = re.sub(r"^\s*(?:#{1,6}|[-*•]|\d+[.)])\s*", "", line)
    return re.sub(r"[*_`]+", "", line).strip()


def summary_of(text: str, chars: int = SUMMARY_CHARS, headings: int = SUMMARY_HEADINGS) -> str:
    """A short lead for a long answer: its first paragraph (cut at a sentence) and the section
    headings, as plain text."""
    secs = sections_of(text)
    heads = [_plain(s.split("\n", 1)[0]) for s in secs if _HEADING.match(s.split("\n", 1)[0])]
    lead = ""
    for b in _blocks(text):
        if _HEADING.match(b.split("\n", 1)[0]) and "\n" not in b:
            continue
        if b.lstrip().startswith(("|", "```")):
            continue
        lead = b.strip()
        break
    if len(lead) > chars:
        cut = max((m.end() for m in _SENTENCE_END.finditer(lead[:chars])), default=0)
        lead = lead[:cut].strip() if cut > chars // 3 else lead[:chars].rstrip() + "…"
    out = [lead] if lead else []
    shown = [h for h in heads if h][:headings]
    if shown:
        out.append("\n".join(f"• {h}" for h in shown))
        if len(heads) > headings:
            out.append(f"• … +{len(heads) - headings}")
    return "\n\n".join(out)
