"""Real smoke test: upload the synthetic Thai sample, transcribe it whole and as 2 chunks, score it.

Usage: uv run python scripts/smoke_transcribe.py [--model MODEL]...
Needs ADC + network. Ground truth comes from samples/synthetic_th.truth.tsv.
"""

import argparse
import difflib
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from google.cloud import storage

from meeting_scribe.config import get_settings
from meeting_scribe.models import Utterance
from meeting_scribe.transcribe import transcribe_chunk

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "synthetic_th.flac"
TRUTH = ROOT / "samples" / "synthetic_th.truth.tsv"
SPLIT_S = 45.0


def load_truth() -> list[Utterance]:
    rows = [line.split("\t") for line in TRUTH.read_text().splitlines()]
    return [Utterance(speaker=s, start_s=float(a), end_s=float(b), text=t) for s, a, b, t in rows]


def cut(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src)]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    subprocess.run([*cmd, "-c:a", "flac", str(dst)], check=True)


def upload(path: Path, name: str) -> str:
    bucket = get_settings().audio_bucket
    storage.Client(project=get_settings().gcp_project).bucket(bucket).blob(
        name
    ).upload_from_filename(str(path), content_type="audio/flac")
    return f"gs://{bucket}/{name}"


def overlap(a: Utterance, b: Utterance) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def score(truth: list[Utterance], hyp: list[Utterance]) -> None:
    """Diarization accuracy = share of truth speech time whose hyp speaker maps to the right person."""
    # confusion[truth_spk][hyp_label] = seconds of overlap
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for t in truth:
        for h in hyp:
            confusion[t.speaker][h.speaker] += overlap(t, h)
    # best hyp label per truth speaker (labels may legitimately be real names)
    mapping = {t: c.most_common(1)[0][0] for t, c in confusion.items() if c}
    total = sum(sum(c.values()) for c in confusion.values())
    right = sum(confusion[t][h] for t, h in mapping.items())
    one_to_one = len(set(mapping.values())) == len(mapping)
    print(f"  speaker map (truth->hyp): {mapping}  one-to-one={one_to_one}")
    print(
        f"  diarization accuracy (time-weighted): {right / total:.1%}" if total else "  no overlap"
    )

    # timestamps are coarse, so also score per-turn: match each truth turn to its most similar
    # hyp utterance by text, then check that utterance carries the mapped speaker label
    def sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(
            None, "".join(a.split()), "".join(b.split()), autojunk=False
        ).ratio()

    good = (
        sum(
            mapping[t.speaker] == max(hyp, key=lambda h: sim(t.text, h.text)).speaker for t in truth
        )
        if hyp
        else 0
    )
    print(f"  per-turn speaker accuracy (text-matched): {good}/{len(truth)}")
    # text similarity on whitespace-free strings
    norm = lambda us: "".join("".join(u.text.split()) for u in us)  # noqa: E731
    ratio = difflib.SequenceMatcher(None, norm(truth), norm(hyp), autojunk=False).ratio()
    print(f"  text similarity (char ratio): {ratio:.1%}")
    # boundary timing error: start of each truth turn vs nearest hyp start
    errs = [min(abs(t.start_s - h.start_s) for h in hyp) for t in truth if hyp]
    print(f"  mean start-time error to nearest hyp turn: {sum(errs) / len(errs):.1f}s")


def show(us: list[Utterance]) -> None:
    for u in us:
        print(f"  [{u.start_s:6.1f}-{u.end_s:6.1f}] {u.speaker}: {u.text}")


def run(model: str, uris: dict[str, str], truth: list[Utterance]) -> None:
    print(f"\n===== {model} =====")
    t0 = time.monotonic()
    whole = transcribe_chunk(uris["full"], 0, 0.0, [], model=model)
    print(f"-- full file ({time.monotonic() - t0:.1f}s, {len(whole.utterances)} utterances)")
    show(whole.utterances)
    score(truth, whole.utterances)

    t0 = time.monotonic()
    c0 = transcribe_chunk(uris["c0"], 0, 0.0, [], model=model)
    l0 = time.monotonic() - t0
    t0 = time.monotonic()
    c1 = transcribe_chunk(uris["c1"], 1, SPLIT_S, c0.utterances[-12:], model=model)
    l1 = time.monotonic() - t0
    print(f"-- 2 chunks split at {SPLIT_S}s (latency {l0:.1f}s + {l1:.1f}s)")
    show(c0.utterances + c1.utterances)
    score(truth, c0.utterances + c1.utterances)
    print(
        f"  labels chunk0={sorted({u.speaker for u in c0.utterances})} chunk1={sorted({u.speaker for u in c1.utterances})}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    models = ap.parse_args().model or [get_settings().transcribe_model]
    tmp = ROOT / "samples" / ".tmp"
    tmp.mkdir(exist_ok=True)
    cut(SAMPLE, tmp / "c0.flac", None, SPLIT_S)
    cut(SAMPLE, tmp / "c1.flac", SPLIT_S, None)
    uris = {
        "full": upload(SAMPLE, "smoke/synthetic_th.flac"),
        "c0": upload(tmp / "c0.flac", "smoke/synthetic_th-chunk-0000.flac"),
        "c1": upload(tmp / "c1.flac", "smoke/synthetic_th-chunk-0001.flac"),
    }
    print(uris)
    truth = load_truth()
    for m in models:
        run(m, uris, truth)


if __name__ == "__main__":
    main()
