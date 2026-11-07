"""Two readers, one page: Docling's structure with Granite's formulas.

Docling segments the page and keeps the prose exact, but it reads each formula
from a tight crop. Granite reads the whole page, which makes its formulas much
better and its prose less reliable. So Docling keeps the document and Granite
supplies the display LaTeX.

The readers don't divide a page the same way: one may see a three-line block
where the other sees three formulas, and either may miss one. Boxes that overlap
are linked, and each connected group becomes one unit of the merge:

    Granite has it       Granite's reading, with Docling's as a fallback if the
                         group is a single formula.
    only Docling has it  Docling's reading, in case Granite missed it.
    only Granite has it  Inserted at its place on the page.

Every candidate gets the same treatment: the printed number and the fonts come
from the text layer. A reading can compile and still be wrong, so it must also
pass the glyph check: its letters, digits and Greek, and then its operators,
accents and brackets, must be the ones the PDF prints in that box. The first
candidate that is not junk, compiles and passes wins. If none does, the best one
stays but is marked `unverified`, so nobody relies on its LaTeX, and `save_crops`
renders that formula from the page. A group with no acceptable reading stays as
`equation_dropped`.

The check counts, so it can't see a symbol in the wrong place, such as two matrix
entries swapped, or a lost subscript: those stay unflagged.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.ingest.chunker import DROPPED_EQUATION
from app.ingest.docling_structured import clean_latex
from app.ingest.formula_layer import (
    Glyph,
    assign_numbers,
    glyph_marks,
    glyph_symbols,
    glyphs_in,
    has_unnamed_glyphs,
    junk_reason,
    latex_marks,
    latex_symbols,
    page_trace,
    page_words,
    printed_numbers,
    restore_styles,
    strip_number,
    text_layer,
    tag_for,
    wrap_alignment,
)

log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / ".venv" / "bin" / "node"
KATEX_CHECK = ROOT / "eval" / "katex_check.mjs"
KATEX_PACKAGE = ROOT / "frontend" / "node_modules" / "katex"       # what the app renders with

MIN_VERTICAL_OVERLAP = 0.5     # of the shorter box
ABSORBED_AREA = 0.7            # a paragraph this far inside a formula box is part of it
PROSE_WORDS = 5                # words of 3+ letters that mark a line as real prose
CROP_DPI = 200
CROP_PADDING = 6.0             # points around the box, so a crop never clips an edge

_TRANSPOSE = re.compile(r"\\(?:top|intercal)(?![a-zA-Z])")

CompileCheck = Callable[[list[str]], list[bool]]
FormulaReader = Callable[[Path, int], list[dict]]


def katex_compiles(formulas: list[str]) -> list[bool]:
    """KaTeX's verdict on each formula, or all True if KaTeX isn't installed."""
    if not formulas:
        return []
    if not (NODE.exists() and KATEX_PACKAGE.exists()):
        log.warning("KaTeX is not installed (run npm ci in frontend/), so compile checks are skipped")
        return [True] * len(formulas)
    result = subprocess.run(
        [str(NODE), str(KATEX_CHECK)],
        input=json.dumps(formulas), capture_output=True, text=True, check=True,
    )
    return [outcome["ok"] for outcome in json.loads(result.stdout)]


class GraniteReader:
    """Reads one page at a time, so a long book can stop and resume at any page."""

    def __init__(self) -> None:
        self._converter = None

    def __call__(self, pdf: Path, page: int) -> list[dict]:
        from docling.datamodel.base_models import ConversionStatus

        from app.ingest.docling_structured import build_vlm_converter

        if self._converter is None:
            self._converter = build_vlm_converter("granite")
        result = self._converter.convert(pdf, page_range=(page, page), raises_on_error=False)
        if result.status != ConversionStatus.SUCCESS:
            log.warning("Granite could not read page %d of %s", page, pdf.name)
            return []
        document = result.document
        formulas = []
        for item in document.texts:
            if item.label != "formula" or not item.text.strip() or not item.prov:
                continue
            prov = item.prov[0]
            height = document.pages[prov.page_no].size.height
            box = prov.bbox.to_top_left_origin(page_height=height)
            formulas.append({"latex": item.text, "bbox": [box.l, box.t, box.r, box.b]})
        return formulas


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def linked(a: list[float], b: list[float]) -> bool:
    across = min(a[2], b[2]) - max(a[0], b[0])
    down = min(a[3], b[3]) - max(a[1], b[1])
    if across <= 0 or down <= 0:
        return False
    return down >= MIN_VERTICAL_OVERLAP * min(a[3] - a[1], b[3] - b[1])


@dataclass
class Group:
    docling: list[int] = field(default_factory=list)      # indices into the records
    granite: list[int] = field(default_factory=list)      # indices into the page's formulas


def group_boxes(
    docling: dict[int, list[float]], granite: list[list[float]]
) -> list[Group]:
    """Connected components of the overlap graph between the two readers' boxes."""
    keys = [("d", index) for index in docling] + [("g", i) for i in range(len(granite))]
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index, box in docling.items():
        for i, other in enumerate(granite):
            if linked(box, other):
                parent[find(("d", index))] = find(("g", i))

    groups: dict[tuple, Group] = {}
    for key in keys:
        group = groups.setdefault(find(key), Group())
        (group.docling if key[0] == "d" else group.granite).append(key[1])
    return list(groups.values())


def _union(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _strip_delimiters(text: str) -> str:
    return re.sub(r"^\$\$\s*|\s*\$\$$", "", text.strip())


def _docling_latex(record: dict) -> str:
    return "" if record.get("formula_unrecognised") else _strip_delimiters(record["text"])


def _inside_share(inner: list[float], outer: list[float]) -> float:
    across = min(inner[2], outer[2]) - max(inner[0], outer[0])
    down = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if across <= 0 or down <= 0:
        return 0.0
    return across * down / area if area > 0 else 1.0


def _is_prose(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]{3,}", text)) >= PROSE_WORDS


@dataclass
class _Unit:
    """One formula in the merged document, and the readings competing for it."""
    page: int
    bbox: list[float]
    candidates: list[tuple[str, str]]         # (source, raw LaTeX), best first
    records: list[dict]                       # the Docling blocks it replaces
    numbers: list[str] = field(default_factory=list)


@dataclass
class _Plan:
    units: list[_Unit] = field(default_factory=list)
    at: dict[int, list[_Unit]] = field(default_factory=dict)     # record index -> what goes there
    skip: set[int] = field(default_factory=set)                  # records replaced elsewhere
    extras: dict[int, list[_Unit]] = field(default_factory=dict)  # page -> Granite-only formulas
    absorbed: set[int] = field(default_factory=set)              # fragments inside a formula box


def _plan_page(
    plan: _Plan, page: int, records: list[tuple[int, dict]], formulas: list[dict]
) -> list[_Unit]:
    """Add one page's groups to the plan, and return the page's units."""
    by_index = dict(records)
    page_units: list[_Unit] = []
    for group in group_boxes({i: r["bbox"] for i, r in records}, [f["bbox"] for f in formulas]):
        members = [by_index[i] for i in sorted(group.docling)]
        fallback = [("docling", latex) for r in members if (latex := _docling_latex(r))]
        if group.granite:
            chosen = sorted((formulas[i] for i in group.granite), key=lambda f: f["bbox"][1])
            alone = len(chosen) == 1 and len(fallback) == 1
            units = [
                _Unit(page, f["bbox"], [("granite", f["latex"])] + (fallback if alone else []),
                      members if position == 0 else [])
                for position, f in enumerate(chosen)
            ]
        else:
            units = [_Unit(page, members[0]["bbox"], fallback, members)]
        page_units.extend(units)
        if group.docling:
            first, *rest = sorted(group.docling)
            plan.at[first] = units
            plan.skip.update(rest)
        else:
            plan.extras.setdefault(page, []).extend(units)
    plan.units.extend(page_units)
    return page_units


def glyph_mismatch(latex: str, glyphs: list[Glyph], numbers: list[str]) -> str | None:
    """How a formula differs from what its box prints: "glyphs", "signs", or None.

    "glyphs" is a difference in letters, digits or Greek, "signs" one in operators,
    accents or brackets. The box may or may not reach the printed equation number,
    so both cases pass. A transpose is `\\top` in the LaTeX, but a book may print it
    as the letter T, which is the same mathematics, so that passes too. Marks are
    skipped when a glyph in the box can't be identified.
    """
    written = Counter(latex_symbols(latex))
    printed = Counter(glyph_symbols(glyphs))
    number = Counter(symbol for text in numbers for symbol in text if symbol.isalnum())
    as_letters = written + Counter({"T": len(_TRANSPOSE.findall(latex))})
    if not any(reading == page for reading in (written, as_letters)
               for page in (printed, printed - number)):
        return "glyphs"
    if has_unnamed_glyphs(glyphs):
        return None
    marks = Counter(latex_marks(latex))
    printed_marks = Counter(glyph_marks(glyphs))
    brackets = Counter({"(": len(numbers), ")": len(numbers)})
    return None if marks in (printed_marks, printed_marks - brackets) else "signs"


@dataclass
class _Reading:
    """One candidate, cleaned, with what each check said about it."""
    source: str
    latex: str
    junk: str | None
    mismatch: str | None
    compiles: bool = False

    @property
    def verified(self) -> bool:
        return self.compiles and self.mismatch is None


def _tidied(source: str, raw: str, numbers: list[str], glyphs: list[Glyph]) -> _Reading:
    latex = re.sub(r"\\tag\s*\{[^{}]*\}", "", clean_latex(raw))
    for number in numbers:
        latex = strip_number(latex, number)
    reason = junk_reason(latex)
    latex = restore_styles(wrap_alignment(latex), glyphs) + tag_for(numbers)
    return _Reading(source, latex, reason, glyph_mismatch(latex, glyphs, numbers))


def _choose(readings: list[_Reading]) -> _Reading | None:
    """The best reading: verified if there is one, then compiling, then any clean one."""
    clean = [reading for reading in readings if reading.junk is None]
    return (next((r for r in clean if r.verified), None)
            or next((r for r in clean if r.compiles), None)
            or (clean[0] if clean else None))


def _equation_record(unit: _Unit, latex: str, source: str, base: dict) -> dict:
    return {
        **base,
        "kind": "equation",
        "text": f"$$\n{latex}\n$$",
        "page": unit.page,
        "bbox": unit.bbox,
        "bboxes": [{"page": unit.page, "bbox": unit.bbox}],
        # Both readers' boxes: if one clips the formula, the crop still shows it.
        "crop_bbox": _union([unit.bbox, *(r["bbox"] for r in unit.records)]),
        "equation_number": ", ".join(unit.numbers) or None,
        "equation_numbers": unit.numbers,
        "formula_source": source,
    }


def _mark_lone_numbers(records: list[dict]) -> tuple[list[dict], int]:
    """Set aside blocks holding only "(2.28)", split from their equation by layout.

    They aren't formulas, and left in the merge they would claim the printed
    number that belongs to the equation beside them.
    """
    marked, count = [], 0
    for record in records:
        latex = _docling_latex(record) if record["kind"] == "equation" else ""
        if latex and junk_reason(latex) == "number":
            record = {**record, "kind": DROPPED_EQUATION, "drop_reason": "number"}
            count += 1
        marked.append(record)
    return marked, count


def merge_document(
    records: list[dict],
    open_page: Callable[[int], object] | None,
    page_formulas: dict[int, list[dict]],
    compile_check: CompileCheck = katex_compiles,
) -> tuple[list[dict], dict]:
    """The sidecar with display formulas merged, and counts of what happened.

    `open_page` returns a PyMuPDF page by number, or is None when the PDF has no
    usable text layer, in which case fonts and numbers stay as Docling left them.
    """
    records, lone_numbers = _mark_lone_numbers(records)
    equations: dict[int, list[tuple[int, dict]]] = {}
    for index, record in enumerate(records):
        if record["kind"] == "equation" and record.get("page") is not None and record.get("bbox"):
            equations.setdefault(record["page"], []).append((index, record))

    plan = _Plan()
    unplaced_numbers = 0
    trace_by_page: dict[int, list] = {}

    for page in sorted(set(equations) | set(page_formulas)):
        formulas = page_formulas.get(page, [])
        page_units = _plan_page(plan, page, equations.get(page, []), formulas)
        plan.absorbed.update(
            index for index, record in enumerate(records)
            if record.get("page") == page and record["kind"] == "paragraph" and record.get("bbox")
            and not _is_prose(record["text"])
            and any(_inside_share(record["bbox"], f["bbox"]) >= ABSORBED_AREA for f in formulas)
        )
        if open_page is None:
            continue
        pdf_page = open_page(page)
        trace_by_page[page] = page_trace(pdf_page)
        numbers = printed_numbers(page_words(pdf_page), pdf_page.rect.width)
        assigned = assign_numbers([tuple(unit.bbox) for unit in page_units], numbers)
        unplaced_numbers += len(numbers) - sum(len(found) for found in assigned)
        for unit, found in zip(page_units, assigned):
            unit.numbers = found

    units = plan.units
    for unit in units:
        if not unit.numbers:
            unit.numbers = [r["equation_number"] for r in unit.records if r.get("equation_number")]

    prepared: list[list[_Reading]] = []
    signs_unchecked = 0
    for unit in units:
        glyphs = glyphs_in(trace_by_page[unit.page], unit.bbox) if unit.page in trace_by_page else []
        signs_unchecked += has_unnamed_glyphs(glyphs)
        prepared.append([_tidied(source, raw, unit.numbers, glyphs) for source, raw in unit.candidates])
    every_reading = [reading for readings in prepared for reading in readings]
    for reading, compiled in zip(every_reading, compile_check([r.latex for r in every_reading])):
        reading.compiles = compiled

    stats = {"formulas": 0, "granite": 0, "docling": 0, "dropped": 0, "drop_reasons": {},
             "lone_numbers": lone_numbers,
             "compile_failed": 0, "unverified": 0, "unverified_reasons": {}, "swapped": 0,
             "signs_unchecked": signs_unchecked,
             "absorbed": len(plan.absorbed),
             "granite_only": sum(len(found) for found in plan.extras.values()),
             "docling_only": sum(all(s == "docling" for s, _ in u.candidates) for u in units),
             "numbers_placed": sum(len(u.numbers) for u in units), "numbers_unplaced": unplaced_numbers}
    finished: dict[int, dict] = {}
    for unit, readings in zip(units, prepared):
        base = unit.records[0] if unit.records else {"level": 0, "docling_ref": None}
        winner = _choose(readings)
        stats["formulas"] += 1
        if winner is None:
            first = readings[0] if readings else None
            reason = first.junk if first else "unrecognised"
            record = {**_equation_record(unit, first.latex if first else "",
                                         first.source if first else "none", base),
                      "kind": DROPPED_EQUATION, "drop_reason": reason}
            stats["dropped"] += 1
            stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
        else:
            record = _equation_record(unit, winner.latex, winner.source, base)
            record["compile_ok"] = winner.compiles
            alternatives = [reading.latex for reading in readings if reading is not winner]
            if alternatives:
                record["alt_latex"] = alternatives[0]
            stats[winner.source] += 1
            stats["compile_failed"] += not winner.compiles
            stats["swapped"] += winner is not readings[0] and readings[0].junk is None and readings[0].compiles
            if not winner.verified:
                reason = winner.mismatch if winner.compiles else "compile"
                record["unverified"] = True
                record["unverified_reason"] = reason
                stats["unverified"] += 1
                stats["unverified_reasons"][reason] = stats["unverified_reasons"].get(reason, 0) + 1
        finished[id(unit)] = record

    return _place(records, plan, finished), stats


def save_crops(records: list[dict], open_page: Callable[[int], object], directory: Path) -> None:
    """Save each unverified formula's page crop and the text-layer symbols in its box.

    The crop is for the reader and the symbols are for the model, so both use one box.
    Earlier crops go first: numbering follows reading order, which a rerun can change.
    """
    import fitz

    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    traces: dict[int, list] = {}
    for record in records:
        if not record.get("unverified"):
            continue
        page = open_page(record["page"])
        left, top, right, bottom = record["crop_bbox"]
        clip = fitz.Rect(left - CROP_PADDING, top - CROP_PADDING,
                         right + CROP_PADDING, bottom + CROP_PADDING) & page.rect
        name = f"{record['order']:06d}.png"
        page.get_pixmap(clip=page.rect if clip.is_empty else clip, dpi=CROP_DPI).save(directory / name)
        record["crop"] = f"{directory.name}/{name}"
        trace = traces.setdefault(record["page"], page_trace(page))
        record["text_layer"] = text_layer(glyphs_in(trace, record["crop_bbox"]))


def _place(records: list[dict], plan: _Plan, finished: dict[int, dict]) -> list[dict]:
    """Rebuild the reading order, with Granite-only formulas placed by height."""
    pending = {page: sorted(units, key=lambda u: u.bbox[1]) for page, units in plan.extras.items()}
    merged: list[dict] = []

    def flush(page: int | None, above: float | None = None) -> None:
        waiting = pending.get(page, [])
        while waiting and (above is None or waiting[0].bbox[1] < above):
            merged.append(finished[id(waiting.pop(0))])

    current = None
    for index, record in enumerate(records):
        page = record.get("page")
        if page is not None and page != current:
            flush(current)
            current = page
        if page is not None and record.get("bbox"):
            flush(page, record["bbox"][1])
        if index in plan.at:
            merged.extend(finished[id(unit)] for unit in plan.at[index])
        elif index not in plan.skip and index not in plan.absorbed:
            merged.append(record)
    flush(current)
    for page in sorted(pending):                 # a page with no Docling block at all
        flush(page)

    return [{**record, "order": order} for order, record in enumerate(merged)]
