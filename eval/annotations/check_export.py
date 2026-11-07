"""Check an Obsidian export of the notes (roadmap item 4, gate item 7).

    .venv/bin/python eval/annotations/check_export.py study-notes.zip

For every file: valid YAML frontmatter with a title, tags and the number of notes, which matches
the notes in the file; one file per document (no title twice); every note has a heading with its
page (or "No page"); maths only in $…$ or $$…$$ (no \\( or \\[ left), with every $ paired. Files
are read as data only.
"""
from __future__ import annotations

import re
import sys
import zipfile

import yaml


def check(path: str) -> int:
    problems = []
    titles = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            text = archive.read(name).decode("utf-8")
            if not text.startswith("---\n"):
                problems.append(f"{name}: no frontmatter")
                continue
            head, body = text[4:].split("\n---\n", 1)
            front = yaml.safe_load(head)
            titles.append(front.get("title"))
            headings = re.findall(r"^## (p\. \d+|No page) · note \d+$", body, re.M)
            if not front.get("title") or "study-notes" not in front.get("tags", []):
                problems.append(f"{name}: frontmatter lacks a title or tags: {front}")
            if front.get("notes") != len(headings):
                problems.append(f"{name}: frontmatter says {front.get('notes')} notes, {len(headings)} headings")
            outside_code = re.sub(r"```.*?```|`[^`\n]*`", "", body, flags=re.S)
            if "\\(" in outside_code or "\\[" in outside_code:
                problems.append(f"{name}: \\( or \\[ left in the text")
            if outside_code.replace("$$", "").count("$") % 2:
                problems.append(f"{name}: an unpaired $")
            print(f"{name}: {len(headings)} notes, title {front['title']!r}, tags {front.get('tags')}, "
                  f"{outside_code.count('$$') // 2} display and {outside_code.replace('$$', '').count('$') // 2} inline formulas")
    if len(set(titles)) != len(titles):
        problems.append(f"a title twice: {titles}")
    for problem in problems:
        print("PROBLEM", problem)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1]))
