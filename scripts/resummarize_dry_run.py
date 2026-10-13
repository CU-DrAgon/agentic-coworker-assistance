"""Re-run the summarizer on a stored meeting and print a before/after comparison.

Read-only: nothing is written to Firestore and nothing is posted to Discord.
Calls Vertex AI (the extractors + synthesizer), so it costs a little quota.

    uv run python scripts/resummarize_dry_run.py e6379be86dfd
"""

import asyncio
import sys
from collections import Counter

from meeting_scribe.agents import summarize_meeting, unresolved_labels
from meeting_scribe.models import Meeting, Summary
from meeting_scribe.store import MeetingStore


def label_counts(meeting: Meeting, names: dict[str, str]) -> Counter[str]:
    return Counter(names.get(u.speaker, u.speaker) for u in meeting.utterances())


def show(title: str, meeting: Meeting, summary: Summary | None) -> None:
    names = summary.speaker_names if summary else {}
    print(f"\n=== {title} ===")
    print("label counts:", dict(label_counts(meeting, names).most_common()))
    print("speaker_names:", names)
    if summary:
        print("action items:")
        for a in summary.action_items:
            print(f"  - ({a.owner!r}, {a.task!r}, due={a.due!r})")


def main(meeting_id: str) -> None:
    meeting = MeetingStore().get(meeting_id)
    if meeting is None:
        sys.exit(f"meeting {meeting_id} not found")
    print(
        f"meeting {meeting.id} {meeting.title!r}: {len(meeting.chunks)} chunks, "
        f"{len(meeting.utterances())} utterances, participants={meeting.participants}"
    )
    print("raw labels:", dict(Counter(u.speaker for u in meeting.utterances()).most_common()))
    print("labels to resolve:", unresolved_labels(meeting))
    show("BEFORE (stored summary)", meeting, meeting.summary)
    new = asyncio.run(summarize_meeting(meeting))
    show("AFTER (re-run locally, not saved)", meeting, new)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
