"""Gate item 1 of roadmap item 7: every Word and PowerPoint file under outputs/ renders.

    python eval/office/check_renders.py RENDERS_DIR [--out eval/results/office-renders.json]

Renders each file through app/office.py into RENDERS_DIR (never data/), counts its pages,
and checks that the source's words (python-docx / python-pptx, read in the container) are
in the render's text layer: coverage = matched words / source words, as multisets of
alphanumeric words after NFKC and case folding. Words, not whitespace tokens: the first run
counted "lower‐triangular" (U+2010) as missing where the render has "lower-triangular", and
LaTeX source text that wraps differently on the page. Slide notes don't count; equations a tool
drew as pictures have no source words. Page counts from real Word and PowerPoint are
compared separately (office_counts.ps1), on copies in Windows' temp folder.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pymupdf  # noqa: E402

from app import office  # noqa: E402
from app.config import load_settings  # noqa: E402

WORD = re.compile(r"\w+")


def normal(words: list[str]) -> collections.Counter:
    return collections.Counter(WORD.findall(unicodedata.normalize("NFKC", " ".join(words)).casefold()))


def pdf_words(path: Path) -> list[str]:
    with pymupdf.open(path) as pdf:
        return [w[4] for page in pdf for w in page.get_text("words")]


async def check(settings, source: Path, renders: Path, reuse: bool = False) -> dict:
    relative = source.relative_to(settings.outputs_dir)
    target = renders / (str(relative).replace("/", "__") + ".pdf")
    row = {"file": str(relative), "kind": source.suffix.lstrip(".")}
    started = time.monotonic()
    try:
        if reuse and target.is_file():
            with pymupdf.open(target) as pdf:
                row["pages"] = pdf.page_count
        else:
            row["pages"] = await office.render_pdf(settings, source, target)
    except office.OfficeError as error:
        row["error"] = str(error)
        return row
    if not reuse:
        row["seconds"] = round(time.monotonic() - started, 1)
    want = normal(await office.source_words(settings, source))
    have = normal(pdf_words(target))
    matched = sum(min(n, have[w]) for w, n in want.items())
    total = sum(want.values())
    row["source_words"] = total
    row["coverage"] = round(matched / total, 4) if total else None
    row["missing"] = sorted((want - have).elements())[:30]
    row["render"] = str(target)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("renders", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/office-renders.json")
    parser.add_argument("--reuse", action="store_true", help="score renders already in RENDERS_DIR")
    args = parser.parse_args()
    settings = load_settings()
    args.renders.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in settings.outputs_dir.rglob("*")
                   if p.suffix.lower() in office.SUFFIXES and p.is_file() and not p.is_symlink()
                   and not p.name.startswith("~$"))
    rows = []
    for source in files:
        row = await check(settings, source, args.renders, args.reuse)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("missing", "render")}), flush=True)
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    rendered = [r for r in rows if "error" not in r]
    low = [r for r in rendered if r["coverage"] is not None and r["coverage"] < 0.98]
    print(f"\n{len(rendered)}/{len(rows)} rendered; coverage < 98%: {len(low)}")
    for r in low:
        print(f"  {r['file']}: {r['coverage']:.1%} missing {r['missing'][:12]}")
    for r in rows:
        if "error" in r:
            print(f"  FAILED {r['file']}: {r['error']}")


if __name__ == "__main__":
    asyncio.run(main())
