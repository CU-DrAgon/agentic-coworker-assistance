"""Salid offline testing: extract <file>, ask "<q>" --as-exec|--as-user ID, deep "<task>" --as-exec, digest [date]."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .config import Settings
from .deep_task import NOT_EXEC_MESSAGE, DeepTaskError, format_stats, run_deep_task
from .digest import build_daily_digest
from .extract import extract_receipt, guess_mime
from .llm import make_client
from .qa import ask
from .store import make_store


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="salid.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("file")
    e.add_argument("--note", default="")
    a = sub.add_parser("ask")
    a.add_argument("question")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--as-exec", action="store_true")
    g.add_argument("--as-user", type=int, metavar="ID")
    dp = sub.add_parser("deep", help="multi-step task via Gemini CLI (executives only)")
    dp.add_argument("task")
    g2 = dp.add_mutually_exclusive_group(required=True)
    g2.add_argument("--as-exec", action="store_true")
    g2.add_argument("--as-user", type=int, metavar="ID")
    d = sub.add_parser("digest")
    d.add_argument("date", nargs="?")
    args = ap.parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "extract":
        data = Path(args.file).read_bytes()
        mime = guess_mime(args.file) or "image/png"
        print(
            extract_receipt(make_client(s), s, data, mime, args.note).model_dump_json(
                indent=2, ensure_ascii=False
            )
        )
    elif args.cmd == "ask":
        ans = ask(s, make_store(s), args.question, is_exec=args.as_exec, user_id=args.as_user or 0)
        print(ans.text)
        print(f"\n[{ans.seconds:.1f}s, {ans.rounds} tool rounds, calls={ans.tool_calls}]")
    elif args.cmd == "deep":
        if not args.as_exec:
            print(NOT_EXEC_MESSAGE)
            return
        try:
            res = run_deep_task(s, make_store(s), args.task)
        except DeepTaskError as e:
            print(e)
            raise SystemExit(1) from None
        print(res.text)
        print(f"\n[{format_stats(res)}; pinned_ok={res.pinned_ok}]")
    elif args.cmd == "digest":
        day = date.fromisoformat(args.date) if args.date else s.today()
        print(build_daily_digest(make_store(s), day))


if __name__ == "__main__":
    main()
