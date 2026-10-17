"""Translate Salid's fixed-string catalog (the English text of every `salid.i18n.CATALOG` key) into
the second-tier languages with Gemini, into `src/salid/locales/<lang>.json`.

    uv run python scripts/salid_translate_catalog.py                  # all seven languages
    uv run python scripts/salid_translate_catalog.py --langs zh,ja    # some
    uv run python scripts/salid_translate_catalog.py --only-missing   # only keys with no entry yet

Offline tool: it calls Gemini on Vertex via ADC (project `meeting-scribe-hackathon`, location
`global`, the cheap model of `Settings.gate_model`) and is NOT part of the bot. File format:

    {"_meta": {"source_hash": {key: sha1(english text)}}, "strings": {key: translated text}}

A re-run translates only keys whose English text changed (hash differs) or that are missing.
Every translation must keep the exact set of `{placeholders}`; one retry for a key that does not,
then the key is left out (the bot then falls back to English for it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types

import salid
from salid.i18n import CATALOG, EXTRA, LANG_NAMES, has_thai


def import_all_modules() -> None:
    """Feature modules add their strings to CATALOG when imported: import them all first."""
    import importlib
    import pkgutil

    for m in pkgutil.iter_modules(salid.__path__):
        if m.name != "__main__":
            try:
                importlib.import_module(f"salid.{m.name}")
            except Exception as e:  # a module that cannot import here has no strings to give
                print(f"skip salid.{m.name}: {e}", file=sys.stderr)


LOCALES = Path(__file__).resolve().parent.parent / "src" / "salid" / "locales"
PROJECT = "meeting-scribe-hackathon"
LOCATION = "global"
MODEL = "gemini-3.5-flash-lite"  # = Settings.gate_model (the cheap model memory summaries use)
BATCH = 40
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*(?:[:!][^}]*)?\}")  # {n}, {x:.1f}

STYLE = {
    "zh": "Simplified Chinese (简体中文). Polite, natural, concise; use 您 only where a formal tone is clearly needed, otherwise 你.",
    "hi": "Hindi in Devanagari. Polite register: address the user as आप.",
    "ms": "Malay (Bahasa Melayu, Malaysia). Polite, natural, standard register.",
    "id": "Indonesian (Bahasa Indonesia). Polite, natural, standard register (Anda / kamu as fits a friendly workplace).",
    "vi": "Vietnamese. Polite, natural, friendly workplace register.",
    "ko": "Korean. Polite register: 해요체 / 합니다체 endings (요/습니다).",
    "ja": "Japanese. Polite register: です/ます.",
}

PROMPT = """You translate the fixed user-facing messages of "Salid", an AI teammate that lives in a company's
Discord server (a warm, concise female assistant), from English into {language}.

Style: {style}
Rules:
- Keep every {{placeholder}} EXACTLY as written (same names, same braces), e.g. {{n}}, {{label}}.
- Keep the name "Salid" in Latin letters. Keep emoji, Markdown (**bold**, `code`, lists), line breaks,
  numbers, URLs, slash-command names (like /language) and product or file names unchanged.
- Natural, short, like a friendly colleague; not word-for-word. Keep the meaning and any warnings.
- Do not add or drop information.
Input: a JSON object {{key: english text}}. Output: ONLY a JSON object with the same keys and the
translated texts as values."""


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def placeholders(text: str) -> list[str]:
    return sorted(PLACEHOLDER.findall(text))


def valid(src: str, out: object) -> bool:
    return (
        isinstance(out, str)
        and bool(out.strip())
        and placeholders(src) == placeholders(out)
        and ("Salid" in out) == ("Salid" in src)
    )


def source_texts() -> dict[str, str]:
    """English text per key (keys whose English text contains Thai, the bilingual labels, are
    skipped: they are the same in every language)."""
    return {k: v["en"] for k, v in CATALOG.items() if v["en"].strip() and not has_thai(v["en"])}


def load(lang: str) -> dict:
    path = LOCALES / f"{lang}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("_meta", {}).setdefault("source_hash", {})
        data.setdefault("strings", {})
        return data
    return {"_meta": {"source_hash": {}}, "strings": {}}


def translate_batch(client, lang: str, batch: dict[str, str]) -> dict[str, str]:
    system = PROMPT.format(language=LANG_NAMES[lang][1], style=STYLE[lang])
    resp = client.models.generate_content(
        model=MODEL,
        contents=[json.dumps(batch, ensure_ascii=False)],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=16384,
        ),
    )
    out = json.loads(resp.text)
    return out if isinstance(out, dict) else {}


def run_lang(client, lang: str, only_missing: bool) -> tuple[str, int, int]:
    src = source_texts()
    data = load(lang)
    hashes, strings = data["_meta"]["source_hash"], data["strings"]
    for k in [k for k, v in strings.items() if k in src and not valid(src[k], v)]:
        strings.pop(k)  # an old entry that no longer passes the checks is translated again
        hashes.pop(k, None)
    for k in [k for k in strings if k not in src]:  # keys removed from the catalog
        strings.pop(k, None)
        hashes.pop(k, None)
    todo = [
        k
        for k, v in src.items()
        if k not in strings or (not only_missing and hashes.get(k) != sha(v))
    ]
    failed: list[str] = []
    for i in range(0, len(todo), BATCH):
        keys = todo[i : i + BATCH]
        got: dict[str, str] = {}
        for _ in range(2):
            ask = {k: src[k] for k in keys if k not in got}
            if not ask:
                break
            try:
                out = translate_batch(client, lang, ask)
            except Exception as e:  # network / bad JSON: try the batch once more
                print(f"[{lang}] batch error ({e.__class__.__name__}): {e}", file=sys.stderr)
                continue
            for k in ask:
                if valid(src[k], out.get(k)):
                    got[k] = out[k].strip() if src[k] == src[k].strip() else out[k]
        for k in keys:
            if k in got:
                strings[k] = got[k]
                hashes[k] = sha(src[k])
            else:
                failed.append(k)
                strings.pop(k, None)  # stale text must not stay for changed English
                hashes.pop(k, None)
    data["strings"] = dict(sorted(strings.items()))
    data["_meta"]["source_hash"] = dict(sorted(hashes.items()))
    LOCALES.mkdir(parents=True, exist_ok=True)
    (LOCALES / f"{lang}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    if failed:
        print(f"[{lang}] left out ({len(failed)}): {', '.join(failed[:12])}...", file=sys.stderr)
    return lang, len(strings), len(src)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", default=",".join(EXTRA), help="comma separated, default all")
    ap.add_argument("--only-missing", action="store_true", help="skip keys that already have text")
    args = ap.parse_args(argv)
    import_all_modules()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    bad = [x for x in langs if x not in EXTRA]
    if bad:
        ap.error(f"unknown language(s): {bad}; choose from {list(EXTRA)}")
    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=types.HttpOptions(timeout=120_000),
    )
    with ThreadPoolExecutor(max_workers=len(langs)) as pool:
        for lang, have, total in pool.map(lambda x: run_lang(client, x, args.only_missing), langs):
            print(f"{lang}: {have}/{total} keys ({100 * have / max(total, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
