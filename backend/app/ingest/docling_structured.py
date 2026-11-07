"""
docling_structured_v2.py
========================

An improved PDF -> structured-markdown converter built on Docling.

Key improvements over the original `docling_structured.py`:

  1. FORMULA RECOVERY  ......  Enables `do_formula_enrichment`, which runs the
     CodeFormula model over detected equation regions and emits LaTeX. The
     original pipeline left these regions with empty `.text`, so every numbered
     display equation was silently dropped by the `if text:` guard. We now
     render them as `$$ ... $$` blocks.

  2. CODE BLOCKS  ..........  Enables `do_code_enrichment` and renders detected
     code as fenced blocks with the language label.

  3. NOISE REMOVAL  ........  Skips running page headers / footers
     (DocItemLabel.PAGE_HEADER / PAGE_FOOTER) such as "16  Special relativity".

  4. NO DUPLICATE CAPTIONS  .  Captions are rendered *with* their figure/table.
     The original also re-emitted them as body text (3x per figure). We skip
     CAPTION-labelled body items.

  5. FOOTNOTE HANDLING  .....  Margin footnotes (very common in textbooks) are
     interleaved into the main flow by Docling's reading order, which shreds
     paragraphs and equations. With `--footnotes endnotes` we collect them and
     append a single Footnotes section instead.

  6. LATEX CLEANUP  ........  Works around the known enrichment bug (issue #2374)
     where LaTeX comes back space-separated character-by-character.

  7. INLINE-MATH CLEANUP  ...  A light, conservative pass that de-tokenises the
     worst inline-math artefacts ("d s 2" -> "ds^2"). This is heuristic.

  8. NO WHOLE-PAGE VLM  ....  Routing the whole page through Granite-Docling was
     tried (Phase 0): 6 to 8 times slower, it flattened nested scripts, dropped
     equation numbers, rewrote words and turned reference lists into tables. The
     option is gone (review R2-X2, Nov 2026); Granite reads single formula crops
     instead (formula_merge.py, through build_vlm_converter).

Usage (a debug tool for one PDF; the worker calls convert_pdf_to_structured_output)
-----
    python -m app.ingest.docling_structured book.pdf -o out/
    python -m app.ingest.docling_structured book.pdf -o out/ --footnotes endnotes   # as the worker does
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

# ── Docling element types ────────────────────────────────────────────────────
# Canonical location is docling_core.types.doc; we import defensively so the
# script works across a range of docling versions.
from docling_core.types.doc import (
    DocItemLabel,
    ImageRefMode,
    TextItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
    ListItem,
)

try:
    from docling_core.types.doc import CodeItem
except ImportError:  # older docling
    CodeItem = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("docling_structured_v2")


# ─────────────────────────────────────────────────────────────────────────────
# Text / LaTeX helpers
# ─────────────────────────────────────────────────────────────────────────────

_CONTROL_SPACE = "\x00"
_EQUATION_NUMBER = re.compile(r"\(\s*([A-Z]?\s*\.?\s*\d+(?:\s*[.\-]\s*\d+)*\s*[a-z]?)\s*\)\s*$")

_SECTION_NUMBER = re.compile(r"^(\d{1,3}(?:\.\d+)*)\.?\s+\S")
_APPENDIX_SECTION = re.compile(r"^([A-Z](?:\.\d+)+)\.?\s+\S")
_APPENDIX_LETTER = re.compile(r"^([A-Z])\.?\s+\S")
_CHAPTER_WORD = re.compile(r"^(chapter|kapitel|part|teil|appendix|anhang)\s+\S", re.IGNORECASE)
_TOP_LEVEL_NAME = re.compile(
    r"^(abstract|acknowledge?ments?|appendix|bibliography|contents|foreword|index"
    r"|introduction|preface|references|anhang|danksagung|einleitung"
    r"|inhaltsverzeichnis|literatur|literaturverzeichnis|vorwort"
    r"|zusammenfassung)\.?\s*$",
    re.IGNORECASE,
)


def _page(element) -> str | int:
    """Safe page-number extraction."""
    return element.prov[0].page_no if getattr(element, "prov", None) else "unknown"


def clean_latex(s: str) -> str:
    """
    Mitigate the space-separated-LaTeX enrichment bug (docling issue #2374),
    where output looks like '\\frac { 1 } { 2 }'. Conservative: we only tighten
    around structural tokens, we do NOT strip every space (some are meaningful).
    """
    if not s:
        return s
    s = s.strip()
    s = re.sub(r"\s+", " ", s)                 # collapse runs of whitespace
    # "\ frac" -> "\frac". Two letters or more: CodeFormula separates letters
    # with spaces, so "\ W" is a control space followed by W, not a command.
    s = re.sub(r"(?<!\\)\\\s+([a-zA-Z]{2,})", r"\\\1", s)
    # A control space must survive tightening, or "\ {" would become "\{".
    s = re.sub(r"(?<!\\)\\ ", _CONTROL_SPACE, s)
    s = re.sub(r"\s*([{}^_])\s*", r"\1", s)      # tighten braces / sub / superscript
    s = re.sub(r"\s*([()\[\]])\s*", r"\1", s)    # tighten brackets
    # Granite's tokenizer splits the sign into a triangle and a stray letter.
    s = re.sub(r"\\triangle\s*q(?![a-zA-Z])", r"\\triangleq", s)
    return s.replace(_CONTROL_SPACE, "\\ ").strip()


def equation_tag(orig: str, latex: str) -> str | None:
    """
    The printed equation number, recovered from the raw text.

    CodeFormula leaves the number out of the LaTeX, but `orig` still ends with
    it, as in "E = mc 2 (2.14)". If the LaTeX ends the same way, the bracket
    belongs to the formula, like the end of "f(2)", so it isn't a number.
    """
    match = _EQUATION_NUMBER.search(orig or "")
    if not match:
        return None
    number = re.sub(r"\s+", "", match.group(1))
    if re.sub(r"\s+", "", latex).endswith(f"({number})"):
        return None
    return number


def infer_heading_levels(headings: list[str]) -> list[int]:
    """
    Levels for headings that Docling reports as one flat level.

    Numbering decides ("4.4.1" is level 3). An unnumbered heading sits one level
    below the last numbered one, except names like "References", which are
    always top level. A bare "A ..." counts as an appendix only when a "B ..."
    heading exists too, because "A Simple Example" is an ordinary title.
    """
    letters = {m.group(1) for h in headings if (m := _APPENDIX_LETTER.match(h))}
    has_appendices = {"A", "B"} <= letters

    levels = []
    numbered_level = 0
    for heading in headings:
        text = heading.strip()
        number = _SECTION_NUMBER.match(text) or _APPENDIX_SECTION.match(text)
        if number:
            level = number.group(1).count(".") + 1
        elif _CHAPTER_WORD.match(text) or _TOP_LEVEL_NAME.match(text):
            level = 1
        elif has_appendices and _APPENDIX_LETTER.match(text):
            level = 1
        else:
            levels.append(min(numbered_level + 1, 6))
            continue
        numbered_level = level
        levels.append(min(level, 6))
    return levels


# The line element only. A general "x 2" -> "x^2" rule must not come back: the
# text layer has lost the height of the 2, and on the books about 60% of the
# cases it could check were subscripts, so it silently changed the mathematics.
_INLINE_FIXES = [
    (re.compile(r"\bd\s+s\s+2\b"), "ds^2"),
    (re.compile(r"\bd\s+s\s+([\u2032'])\s+2\b"), r"ds'^2"),
]


def clean_inline_math(text: str, enabled: bool) -> str:
    """Optional, conservative de-tokenisation of inline math artefacts."""
    if not enabled or not text:
        return text
    for pattern, replacement in _INLINE_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _regions(element, doc) -> list[dict]:
    """
    Every page region of an element, in PDF points with y measured from the top.

    Docling boxes use a bottom-left origin; PyMuPDF and PDF.js use top-left, and
    SQLite stores what they use. An element split across a page break has one
    region per page.
    """
    regions = []
    for prov in getattr(element, "prov", None) or []:
        page = doc.pages.get(prov.page_no)
        if page is None or page.size is None:
            continue
        box = prov.bbox.to_top_left_origin(page_height=page.size.height)
        regions.append({
            "page": prov.page_no,
            "bbox": [round(box.l, 2), round(box.t, 2), round(box.r, 2), round(box.b, 2)],
        })
    return regions


def _heading_levels(doc) -> dict[str, int]:
    """Map each section header's self_ref to the level it should render at."""
    headers = [el for el, _ in doc.iterate_items() if isinstance(el, SectionHeaderItem)]
    reported = [max(1, min(getattr(el, "level", 1) or 1, 6)) for el in headers]
    if len(set(reported)) > 1:      # Docling found a hierarchy, so trust it
        levels = reported
    else:
        levels = infer_heading_levels([el.text for el in headers])
    return {el.self_ref: level for el, level in zip(headers, levels)}


def _table_to_md(element: TableItem, doc) -> str:
    """Export a TableItem to a markdown string with graceful fallbacks."""
    try:
        return element.export_to_markdown(doc=doc)
    except (AttributeError, TypeError):
        df = element.export_to_dataframe(doc=doc)
        try:
            return df.to_markdown(index=False)   # needs: pip install tabulate
        except Exception:
            return df.to_string(index=False)


def describe_figure(
    image,
    caption: str,
    page: int,
    figure_label: str,
) -> str:
    """
    Produce a semantic description for a figure.

    Stub for now — returns the caption or a page fallback. Later this can be
    swapped for a VLM backend without changing the rest of the pipeline.
    """
    # Future:
    # description = vlm_backend.describe(
    #     image=image,
    #     caption=caption,
    # )
    if caption:
        return caption
    return f"Figure located on page {page}."


def _save_figure_description_file(
    desc_dir: Path,
    desc_basename: str,
    figure_label: str,
    caption: str,
    description: str,
    relative_image_path: str,
) -> str:
    """Write figure_descriptions/<basename>.md and return its relative path."""
    desc_dir.mkdir(parents=True, exist_ok=True)
    rel_desc = f"figure_descriptions/{desc_basename}.md"
    body = f"# {figure_label}\n\n"
    if caption:
        body += f"Caption:\n\n{caption}\n\n"
    body += (
        f"Description:\n\n{description}\n\n"
        f"Original image:\n\n{relative_image_path}\n"
    )
    (desc_dir / f"{desc_basename}.md").write_text(body, encoding="utf-8")
    return rel_desc


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Enriched markdown builder (single reading-order pass)
# ─────────────────────────────────────────────────────────────────────────────

def build_enriched_markdown(
    doc,
    output_dir: Path,
    *,
    footnotes: str = "inline",        # "inline" | "endnotes" | "drop"
    fix_inline_math: bool = True,
    skip_running_heads: bool = True,
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """
    Returns the Markdown, the figure and table summaries, and the sidecar: one
    record per emitted block with its page, bounding boxes and Docling reference.
    All body output goes through `emit`, so the two can't drift apart. The Asset
    Reference Index is left out of the sidecar on purpose: it is a file listing
    for human readers and would only pollute retrieval.
    """
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    desc_dir = output_dir / "figure_descriptions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = []
    blocks: list[dict] = []
    collected_footnotes: list[tuple[str, object]] = []
    heading_levels = _heading_levels(doc)

    def emit(kind: str, markdown: str | list[str], element=None, *,
             text: str | None = None, level: int = 0, **extra) -> None:
        lines = [markdown] if isinstance(markdown, str) else markdown
        md_lines.extend(lines)
        regions = _regions(element, doc) if element is not None else []
        blocks.append({
            "order": len(blocks),
            "kind": kind,
            "text": text if text is not None else "\n".join(lines).strip(),
            "level": level,
            "page": regions[0]["page"] if regions else None,
            "bbox": regions[0]["bbox"] if regions else None,
            "bboxes": regions,
            "docling_ref": getattr(element, "self_ref", None),
            **extra,
        })

    figure_counter = 0
    table_counter = 0
    figure_registry: dict[int, dict] = {}
    table_registry: dict[int, dict] = {}
    figure_summary: list[dict] = []
    table_summary: list[dict] = []

    # Pre-gather caption strings so we can skip body items that merely repeat them
    # (in addition to skipping CAPTION-labelled items).
    caption_texts: set[str] = set()
    for element, _ in doc.iterate_items():
        if isinstance(element, (PictureItem, TableItem)):
            cap = element.caption_text(doc) or ""
            if cap:
                caption_texts.add(_norm(cap))

    for elem_idx, (element, _level) in enumerate(doc.iterate_items()):
        label = getattr(element, "label", None)

        # ── Code blocks (check before generic text) ──────────────────────────
        is_code = (CodeItem is not None and isinstance(element, CodeItem)) or (
            label == DocItemLabel.CODE
        )
        if is_code:
            code = (getattr(element, "text", "") or "").strip()
            lang = getattr(element, "code_language", None) or ""
            lang = "" if lang in ("unknown", None) else str(lang)
            if code:
                emit("code", f"\n```{lang}\n{code}\n```\n", element, language=lang or None)
            continue

        # ── Formulas (FORMULA-labelled TextItem) ─────────────────────────────
        if label == DocItemLabel.FORMULA:
            latex = clean_latex(getattr(element, "text", "") or "")
            if latex:
                number = equation_tag(getattr(element, "orig", "") or "", latex)
                tag = f" \\tag{{{number}}}" if number else ""
                emit("equation", f"\n$$\n{latex}{tag}\n$$\n", element, equation_number=number)
            else:
                # enrichment found a formula region but produced no LaTeX
                emit("equation", f"\n$$\n% [formula not recognised — page {_page(element)}]\n$$\n",
                     element, formula_unrecognised=True)
            continue

        # ── Document title ───────────────────────────────────────────────────
        if label == DocItemLabel.TITLE:
            title = (getattr(element, "text", "") or "").strip()
            if title:
                emit("heading", f"\n# {title}\n", element, text=title, level=1)
            continue

        # ── Section headers ──────────────────────────────────────────────────
        if isinstance(element, SectionHeaderItem):
            level = heading_levels[element.self_ref]
            emit("heading", f"\n{'#' * level} {element.text}\n", element,
                 text=element.text, level=level)
            continue

        # ── Running page headers / footers: drop as noise ────────────────────
        if skip_running_heads and label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            continue

        # ── Captions: rendered with their asset, skip the standalone copy ────
        if label == DocItemLabel.CAPTION:
            continue

        # ── Footnotes: inline / collect / drop ───────────────────────────────
        if label == DocItemLabel.FOOTNOTE:
            txt = (getattr(element, "text", "") or "").strip()
            if not txt:
                continue
            if footnotes == "drop":
                continue
            if footnotes == "endnotes":
                collected_footnotes.append((txt, element))
                continue
            # inline (default)
            emit("footnote", f"\n> [footnote] {txt}\n", element, text=txt)
            continue

        # ── List items (ListItem subclasses TextItem, so this must come first) ──
        if isinstance(element, ListItem):
            marker = (getattr(element, "marker", "") or "").strip()
            prefix = marker if getattr(element, "enumerated", False) and marker else "-"
            txt = clean_inline_math(element.text or "", fix_inline_math)
            emit("list_item", f"{prefix} {txt}", element)
            continue

        # ── Body text ────────────────────────────────────────────────────────
        if isinstance(element, TextItem):
            text = (element.text or "").strip()
            if not text:
                continue
            if _norm(text) in caption_texts:   # duplicated caption as body text
                continue
            text = clean_inline_math(text, fix_inline_math)
            emit("paragraph", f"\n{text}\n", element, text=text)
            continue

        # ── Figures / Pictures ───────────────────────────────────────────────
        if isinstance(element, PictureItem):
            figure_counter += 1
            page = _page(element)
            caption = element.caption_text(doc) or ""
            img = element.get_image(doc)  # PIL Image | None

            fig_filename = f"figure_{elem_idx + 1:03d}_page{page}.png"
            desc_basename = f"figure_{elem_idx + 1:03d}"
            relative_path = f"figures/{fig_filename}"
            if img:
                img.save(figures_dir / fig_filename, "PNG")

            lbl = f"Figure {figure_counter}"
            description = describe_figure(img, caption, page, lbl)
            description_file = _save_figure_description_file(
                desc_dir, desc_basename, lbl, caption, description, relative_path,
            )

            alt = f"{lbl}: {caption}" if caption else lbl
            cap_line = f"_{lbl}: {caption}_" if caption else f"_{lbl} — Page {page}_"
            figure_md = [f"\n![{alt}]({relative_path})\n", f"{cap_line}\n"]
            # The chunk text is the caption alone. The stub's page-number fallback
            # says nothing worth retrieving; Phase 5 supplies real descriptions.
            emit("figure", figure_md, element, text=caption,
                 image_path=relative_path if img else None)

            info = {
                "label": lbl, "file": fig_filename, "relative_path": relative_path,
                "caption": caption, "description_file": description_file, "page": page,
            }
            figure_registry[figure_counter] = info
            figure_summary.append({**info, "figure_num": figure_counter,
                                   "element_index": elem_idx + 1,
                                   "size": list(img.size) if img else None})
            continue

        # ── Tables ───────────────────────────────────────────────────────────
        if isinstance(element, TableItem):
            table_counter += 1
            page = _page(element)
            caption = element.caption_text(doc) or ""
            df = element.export_to_dataframe(doc=doc)

            base = f"table_{elem_idx + 1:03d}_page{page}"
            rel_csv = f"tables/{base}.csv"
            df.to_csv(tables_dir / f"{base}.csv", index=False)
            try:
                df.to_html(tables_dir / f"{base}.html", index=False)
            except Exception:
                pass

            lbl = f"Table {table_counter}"
            title_line = f"**{lbl}: {caption}**" if caption else f"**{lbl}** *(Page {page})*"
            table_md = _table_to_md(element, doc)
            emit("table",
                 [f"\n{title_line}\n", table_md, f"\n> 📄 Full data: [`{base}.csv`]({rel_csv})\n"],
                 element, text=f"{title_line}\n\n{table_md}", csv_path=rel_csv)

            info = {
                "label": lbl, "file": f"{base}.csv", "relative_path": rel_csv,
                "caption": caption, "page": page,
            }
            table_registry[table_counter] = info
            table_summary.append({**info, "table_num": table_counter,
                                  "element_index": elem_idx + 1,
                                  "rows": df.shape[0], "columns": df.shape[1]})
            continue

    # ── Endnotes ─────────────────────────────────────────────────────────────
    if footnotes == "endnotes" and collected_footnotes:
        md_lines.append("\n\n---\n")
        emit("heading", "## Footnotes\n", text="Footnotes", level=1)
        for i, (fn, element) in enumerate(collected_footnotes, 1):
            fn = clean_inline_math(fn, fix_inline_math)
            emit("footnote", f"{i}. {fn}", element, text=fn)

    # ── Asset Reference Index ────────────────────────────────────────────────
    if figure_registry or table_registry:
        md_lines.append("\n\n---\n")
        md_lines.append("## Asset Reference Index\n")
        md_lines.append("> Every **Figure** and **Table** label is mapped to its "
                        "file on disk below.\n")
        if figure_registry:
            md_lines.append("\n### Figures\n")
            md_lines.append("| Label | Page | Caption | File |")
            md_lines.append("|-------|------|---------|------|")
            for info in figure_registry.values():
                md_lines.append(f"| **{info['label']}** | {info['page']} | "
                                f"{info['caption'] or '—'} | "
                                f"[`{info['file']}`]({info['relative_path']}) |")
        if table_registry:
            md_lines.append("\n### Tables\n")
            md_lines.append("| Label | Page | Caption | File |")
            md_lines.append("|-------|------|---------|------|")
            for info in table_registry.values():
                md_lines.append(f"| **{info['label']}** | {info['page']} | "
                                f"{info['caption'] or '—'} | "
                                f"[`{info['file']}`]({info['relative_path']}) |")

    return "\n".join(md_lines), figure_summary, table_summary, blocks


# ─────────────────────────────────────────────────────────────────────────────
# Converter construction
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_converter(formula: bool = True, code: bool = True):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,
        generate_picture_images=True,
        images_scale=2.0,
    )
    # set enrichment flags defensively (names stable since docling 2.x)
    if formula:
        opts.do_formula_enrichment = True
    if code:
        opts.do_code_enrichment = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def build_vlm_converter(model: str = "granite"):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel import vlm_model_specs
    from docling.pipeline.vlm_pipeline import VlmPipeline

    spec = {
        "granite": getattr(vlm_model_specs, "GRANITEDOCLING_TRANSFORMERS", None),
        "granite_mlx": getattr(vlm_model_specs, "GRANITEDOCLING_MLX", None),
        "smoldocling": getattr(vlm_model_specs, "SMOLDOCLING_TRANSFORMERS", None),
    }.get(model)
    if spec is None:
        raise ValueError(f"Unknown / unavailable VLM model: {model}")

    opts = VlmPipelineOptions(vlm_options=spec)
    # force_backend_text must stay off: it replaces the model's LaTeX with the
    # PDF text layer, which is the flattened math this pipeline exists to avoid.
    for attr, val in (("force_backend_text", False),
                      ("generate_page_images", True),
                      ("generate_picture_images", True)):
        if hasattr(opts, attr):
            setattr(opts, attr, val)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, pipeline_options=opts
            )
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def convert_pdf_to_structured_output(
    pdf_path: str,
    output_dir: str,
    *,
    footnotes: str = "inline",
    fix_inline_math: bool = True,
) -> dict:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building standard converter (formula + code enrichment).")
    converter = build_standard_converter(formula=True, code=True)

    log.info("Converting: %s", pdf_path.name)
    t0 = time.time()
    result = converter.convert(str(pdf_path))
    doc = result.document
    log.info("Converted in %.1fs", time.time() - t0)

    markdown, figure_summary, table_summary, blocks = build_enriched_markdown(
        doc, output_dir,
        footnotes=footnotes,
        fix_inline_math=fix_inline_math,
        skip_running_heads=True,
    )
    (output_dir / "document.md").write_text(markdown, encoding="utf-8")
    with open(output_dir / "blocks.jsonl", "w", encoding="utf-8") as sidecar:
        for block in blocks:
            sidecar.write(json.dumps(block, ensure_ascii=False) + "\n")
    # Figures are already PNG files, so the JSON only needs placeholders.
    doc.save_as_json(output_dir / "docling.json", image_mode=ImageRefMode.PLACEHOLDER)

    n_formulas = sum(
        1 for el, _ in doc.iterate_items()
        if getattr(el, "label", None) == DocItemLabel.FORMULA
    )
    log.info("Markdown   -> %s", output_dir / "document.md")
    log.info("  %d figures, %d tables, %d formulas",
             len(figure_summary), len(table_summary), n_formulas)

    outline = [
        {"level": el.level, "title": el.text,
         "page": el.prov[0].page_no if el.prov else None}
        for el, _ in doc.iterate_items()
        if isinstance(el, SectionHeaderItem)
    ]

    title = next(
        (el.text for el, _ in doc.iterate_items()
         if getattr(el, "label", None) == DocItemLabel.TITLE),
        None,
    )
    summary = {
        "source": str(pdf_path),
        "title": title,
        "pages": doc.num_pages(),
        "pipeline": "standard",
        "counts": {"figures": len(figure_summary),
                   "tables": len(table_summary),
                   "formulas": n_formulas},
        "outline": outline,
        "figures": figure_summary,
        "tables": table_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("Sidecar    -> %s (%d blocks)", output_dir / "blocks.jsonl", len(blocks))
    log.info("Summary    -> %s", output_dir / "summary.json")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert a PDF to structured markdown with equations, "
                    "figures, tables, and an asset index."
    )
    p.add_argument("pdf_path", help="Path to the input PDF")
    p.add_argument("-o", "--output-dir", default="structured_output")
    p.add_argument("--footnotes", default="inline",
                   choices=["inline", "endnotes", "drop"],
                   help="how to handle margin footnotes (default: inline)")
    p.add_argument("--no-inline-math-fix", action="store_true",
                   help="disable the heuristic inline-math cleanup pass")
    args = p.parse_args()

    convert_pdf_to_structured_output(
        args.pdf_path, args.output_dir,
        footnotes=args.footnotes,
        fix_inline_math=not args.no_inline_math_fix,
    )