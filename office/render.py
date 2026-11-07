"""Runs inside the office image. Renders one Word or PowerPoint file to PDF and reads a deck.

    python /opt/office/render.py render   in/NAME.pptx|.docx  ->  out/render.pdf
    python /opt/office/render.py slides   in/NAME.pptx        ->  out/slides.json
    python /opt/office/render.py words    in/NAME.pptx|.docx  ->  out/words.json

Prints one JSON line with the result. The file is untrusted, so this never trusts it either:
LibreOffice runs with a fresh profile under /tmp where no macro can run and links are never
updated, and the container has no network. Hidden slides are exported too, so page n of the
PDF is always slide n.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(os.environ.get("OFFICE_WORK", "/work"))
OUT = WORK / "out"
PROFILE = Path("/tmp/lo-profile")
SETTINGS = Path("/opt/office/registrymodifications.xcu")

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
FILTERS = {
    ".pptx": 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"},'
             '"ExportNotesPages":{"type":"boolean","value":"false"}}',
    ".docx": "pdf:writer_pdf_Export",
}


def finish(**result) -> None:
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)


BAR = re.compile(r"(<m:t(?: [^>]*)?>)\|(</m:t>)")


def word_bars(source: Path) -> Path:
    """LibreOffice turns a Word equation's bare "|" (pandoc writes |x| that way) into a
    syntax error and shows its marker "¿" (found Nov 6, 2026). Of the look-alikes tried,
    U+01C0 (ǀ) renders as Word shows |x|: upright and tight; U+2223 and U+FF5C come out
    slanted, U+2502 spaced. Only the copy rendered here changes, never the file."""
    with zipfile.ZipFile(source) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
        fixed = BAR.sub(lambda m: m.group(1) + "\u01c0" + m.group(2), document)
        if fixed == document:
            return source
        copy = source.with_name("bars-" + source.name)
        with zipfile.ZipFile(copy, "w", zipfile.ZIP_DEFLATED) as out:
            for info in archive.infolist():
                data = fixed.encode("utf-8") if info.filename == "word/document.xml" else archive.read(info)
                out.writestr(info, data)
    return copy


def render(source: Path) -> None:
    suffix = source.suffix.lower()
    if suffix not in FILTERS:
        finish(ok=False, error=f"can't render {suffix} files")
    if suffix == ".docx":
        source = word_bars(source)
    (PROFILE / "user").mkdir(parents=True, exist_ok=True)
    shutil.copy(SETTINGS, PROFILE / "user" / "registrymodifications.xcu")
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    done = subprocess.run(
        ["soffice", f"-env:UserInstallation=file://{PROFILE}", "--headless", "--invisible",
         "--norestore", "--nolockcheck", "--nodefault", "--nologo",
         "--convert-to", FILTERS[suffix], "--outdir", str(OUT), str(source)],
        capture_output=True, text=True, errors="replace",
    )
    made = OUT / f"{source.stem}.pdf"
    if done.returncode != 0 or not made.is_file():
        finish(ok=False, error="LibreOffice couldn't render the file",
               detail=(done.stderr or done.stdout).strip()[-400:])
    made.rename(OUT / "render.pdf")
    finish(ok=True, seconds=round(time.monotonic() - started, 2))


def paragraphs(element) -> list[str]:
    """Every paragraph's text, in document order, groups and tables included, with each
    equation (OMML) as its plain characters. shape.text_frame alone misses text in groups."""
    lines = []
    for para in element.iter(A + "p"):
        if any(parent.tag.startswith(M) for parent in para.iterancestors()):
            continue
        parts = []
        for node in para.iter():
            if node.tag == M + "oMath" and not any(p.tag == M + "oMath" for p in node.iterancestors()):
                parts.append(" " + "".join(t.text or "" for t in node.iter(M + "t")) + " ")
            elif node.tag == A + "t" and not any(p.tag.startswith(M) for p in node.iterancestors()):
                parts.append(node.text or "")
        line = " ".join("".join(parts).split())
        if line:
            lines.append(line)
    return lines


MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
ROW_TOLERANCE = 0.02        # of the slide's height: boxes this close in height read as one row


def _position(element, shapes_by_element) -> tuple[int, int]:
    """(top, left) in EMU: python-pptx's, which inherits a placeholder's from its layout, else
    the element's first offset (an equation box in mc:AlternateContent has no shape object)."""
    shape = shapes_by_element.get(element)
    if shape is not None and shape.top is not None and shape.left is not None:
        return int(shape.top), int(shape.left)
    offset = element.find(".//" + A + "off")
    if offset is not None:
        return int(offset.get("y", 0)), int(offset.get("x", 0))
    return 0, 0


def _reading_order(container, shapes, height: int) -> list:
    """The container's shapes as a reader takes them: rows by top edge (within ROW_TOLERANCE
    of the slide's height), each row left to right. Not the drawing order the file keeps them
    in: on a slide with "2 L-Shaped" over "15-20 km arm length" and "Triangular" over "10 km",
    drawing order paired the labels with the wrong lengths (item 7's gate, Nov 6, 2026)."""
    by_element = {shape._element: shape for shape in shapes}
    kids = [child for child in container if child.tag not in (P + "nvGrpSpPr", P + "grpSpPr")]
    placed = sorted(((*_position(child, by_element), index, child) for index, child in enumerate(kids)),
                    key=lambda item: (item[0], item[2]))
    rows, tolerance = [], ROW_TOLERANCE * height
    for item in placed:
        if rows and item[0] - rows[-1][0][0] <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    ordered = []
    for row in rows:
        for _, _, _, child in sorted(row, key=lambda item: (item[1], item[2])):
            if child.tag == P + "grpSp":
                group = by_element.get(child)
                ordered.extend(_reading_order(child, group.shapes if group is not None else [], height))
            else:
                ordered.append(child)
    return ordered


def slide_lines(slide, height: int) -> list[str]:
    """The slide's paragraphs, shape by shape in reading order. For an equation box
    (mc:AlternateContent), only its real content: PowerPoint's fallback copy is a picture."""
    lines = []
    for element in _reading_order(slide.shapes._spTree, slide.shapes, height):
        if element.tag == MC + "AlternateContent":
            choice = element.find(MC + "Choice")
            element = choice if choice is not None else element
        lines.extend(paragraphs(element))
    return lines


def equations(element) -> list[str]:
    return [" ".join("".join(t.text or "" for t in node.iter(M + "t")).split())
            for node in element.iter(M + "oMath")
            if not any(p.tag == M + "oMath" for p in node.iterancestors())]


def slide_title(slide) -> str:
    """The title placeholder's text, else the text set in the largest font on the slide."""
    shape = slide.shapes.title
    if shape is not None and shape.has_text_frame and shape.text_frame.text.strip():
        return " ".join(shape.text_frame.text.split())
    best, size = "", 0
    for para in slide._element.iter(A + "p"):
        text = " ".join("".join(t.text or "" for t in para.iter(A + "t")).split())
        sizes = [int(r.get("sz")) for r in para.iter(A + "rPr") if r.get("sz", "").isdigit()]
        if text and sizes and max(sizes) > size and len(text) <= 200:
            best, size = text, max(sizes)
    return best


def slides(source: Path) -> None:
    from pptx import Presentation

    deck = Presentation(source)
    core = deck.core_properties
    found = []
    for number, slide in enumerate(deck.slides, 1):
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = "\n".join(paragraphs(slide.notes_slide.notes_text_frame._txBody))
        found.append({
            "slide": number,
            "hidden": slide._element.get("show") == "0",
            "title": slide_title(slide),
            "lines": slide_lines(slide, deck.slide_height or 6858000),
            "equations": equations(slide._element),
            "notes": notes,
        })
    OUT.mkdir(exist_ok=True)
    (OUT / "slides.json").write_text(json.dumps({
        "title": core.title or "", "author": core.author or "",
        "created": core.created.isoformat() if core.created else None,
        "slides": found,
    }, ensure_ascii=False), encoding="utf-8")
    finish(ok=True, slides=len(found))


def words(source: Path) -> None:
    """The source's words, for checking a render: slides' text (not notes), or a Word body."""
    if source.suffix.lower() == ".pptx":
        from pptx import Presentation
        deck = Presentation(source)
        text = [line for slide in deck.slides for line in slide_lines(slide, deck.slide_height or 6858000)]
    else:
        from docx import Document
        document = Document(source)
        body = document.element.body
        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        text = ["".join(t.text or "" for t in p.iter(W + "t")) for p in body.iter(W + "p")]
    OUT.mkdir(exist_ok=True)
    (OUT / "words.json").write_text(json.dumps(" ".join(text).split(), ensure_ascii=False), encoding="utf-8")
    finish(ok=True)


def main() -> None:
    command, name = sys.argv[1], sys.argv[2]
    source = (WORK / name).resolve()
    if WORK.resolve() not in source.parents or not source.is_file():
        finish(ok=False, error="no such file")
    try:
        {"render": render, "slides": slides, "words": words}[command](source)
    except Exception as error:      # a broken file is a result, not a crash
        finish(ok=False, error=f"couldn't read the file: {type(error).__name__}: {str(error)[:300]}")


if __name__ == "__main__":
    main()
