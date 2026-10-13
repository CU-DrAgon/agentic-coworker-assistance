"""Post a meeting summary to a Discord webhook, with the full transcript attached."""

import json
import logging
import time

import httpx

from .models import Meeting, Summary

logger = logging.getLogger(__name__)

TITLE_MAX = 256
DESCRIPTION_MAX = 4096
FIELD_VALUE_MAX = 1024
FOOTER_MAX = 2048
EMBED_TOTAL_MAX = 6000
MAX_RETRY_WAIT_S = 10.0
EMBED_COLOR = 0x4285F4


class DiscordError(RuntimeError):
    """Webhook call failed. Never includes the webhook URL (it is a secret)."""


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def bullet_list(lines: list[str], limit: int = FIELD_VALUE_MAX) -> str:
    """Join lines as bullets, dropping whole items (with a "+N more" note) to fit `limit`."""
    items = [truncate(f"• {line}", limit) for line in lines if line.strip()]
    out: list[str] = []
    used = 0
    for i, item in enumerate(items):
        remaining = len(items) - i
        note = f"\n… +{remaining} more"
        # Keep room for the "more" note unless this is the last item.
        reserve = len(note) if remaining > 1 else 0
        if used + len(item) + (1 if out else 0) + reserve > limit:
            out.append(f"… +{remaining} more")
            break
        used += len(item) + (1 if out else 0)
        out.append(item)
    return "\n".join(out)


def format_action_item(owner: str, task: str, due: str | None) -> str:
    line = f"**{owner}** — {task}"
    return f"{line} ({due})" if due else line


def _speaker(name: str, names: dict[str, str]) -> str:
    return names.get(name) or name


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"[{total // 60:02d}:{total % 60:02d}]"


def format_transcript(meeting: Meeting) -> str:
    names = meeting.summary.speaker_names if meeting.summary else {}
    return "\n".join(
        f"{format_timestamp(u.start_s)} {_speaker(u.speaker, names)}: {u.text}"
        for u in meeting.utterances()
    )


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


def build_embed(meeting: Meeting) -> dict:
    summary = meeting.summary or Summary(title=meeting.title, overview="No summary available.")
    utterances = meeting.utterances()
    speakers = {_speaker(u.speaker, summary.speaker_names) for u in utterances}
    duration = max((u.end_s for u in utterances), default=0.0)
    footer = truncate(
        f"Duration {format_duration(duration)} · {len(speakers)} speaker"
        f"{'' if len(speakers) == 1 else 's'}",
        FOOTER_MAX,
    )
    title = truncate(summary.title or meeting.title, TITLE_MAX)
    description = truncate(summary.overview, DESCRIPTION_MAX)

    sections = [
        ("Key points", bullet_list(summary.key_points)),
        ("Decisions", bullet_list(summary.decisions)),
        (
            "Action items",
            bullet_list(
                [
                    format_action_item(_speaker(a.owner, summary.speaker_names), a.task, a.due)
                    for a in summary.action_items
                ]
            ),
        ),
        ("Open questions", bullet_list(summary.open_questions)),
    ]

    # Stay under the 6000-character embed total; later fields shrink first.
    budget = EMBED_TOTAL_MAX - len(title) - len(description) - len(footer)
    fields = []
    for name, value in sections:
        if not value:
            continue
        room = min(FIELD_VALUE_MAX, budget - len(name))
        if room < 40:
            break
        value = value if len(value) <= room else truncate(value, room)
        fields.append({"name": name, "value": value, "inline": False})
        budget -= len(name) + len(value)

    embed: dict = {
        "title": title,
        "color": EMBED_COLOR,
        "fields": fields,
        "footer": {"text": footer},
    }
    if description:
        embed["description"] = description
    return embed


def _retry_after(response: httpx.Response) -> float:
    try:
        wait = float(response.json().get("retry_after"))
    except (ValueError, TypeError, AttributeError):
        try:
            wait = float(response.headers.get("Retry-After", 1))
        except ValueError:
            wait = 1.0
    return min(max(wait, 0.0), MAX_RETRY_WAIT_S)


def post_summary(meeting: Meeting, webhook_url: str, *, client: httpx.Client | None = None) -> None:
    """Post one embed plus `transcript-{id}.txt`. Retries once on HTTP 429."""
    filename = f"transcript-{meeting.id}.txt"
    transcript = format_transcript(meeting).encode("utf-8")
    payload = {
        "embeds": [build_embed(meeting)],
        "attachments": [{"id": 0, "filename": filename}],
    }

    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        for attempt in (1, 2):
            response = http.post(
                webhook_url,
                data={"payload_json": json.dumps(payload, ensure_ascii=False)},
                files={"files[0]": (filename, transcript, "text/plain; charset=utf-8")},
            )
            if response.status_code == 429 and attempt == 1:
                wait = _retry_after(response)
                logger.warning("Discord rate limited; retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            break
    finally:
        if owns_client:
            http.close()

    if not response.is_success:
        # Do not use raise_for_status(): its message contains the webhook URL.
        raise DiscordError(
            f"Discord webhook returned {response.status_code}: {response.text[:200]}"
        )
    logger.info("Posted summary of meeting %s to Discord", meeting.id)
