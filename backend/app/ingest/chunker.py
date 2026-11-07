"""Structure-aware chunking.

Three layers:

1. blocks_from_markdown  Markdown -> atomic blocks. In production the blocks
                         come from blocks.jsonl instead; this scanner serves
                         sources without a sidecar, and the tests.
2. _units                blocks -> units that must not be separated: an equation
                         plus the sentence before it and its "where" sentence.
3. _pack                 units -> chunks of 300-600 tokens, overlapping by one
                         sentence.

Priority when the rules collide: never splitting inside a block beats the size
limit, and the size limit beats gluing. So an oversized table becomes one large
chunk, while a long run of equations may break between two equations.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

MIN_TOKENS = 300
MAX_TOKENS = 600
OVERLAP_MAX_TOKENS = 80   # a very long sentence is not worth repeating
SPLIT_HEADING_LEVEL = 3   # H1-H3 start a new section; deeper headings stay inline
PATH_SEPARATOR = " › "

# True bge-m3 tokens per word-or-punctuation token. 1.5, fitted on one paper,
# ran 25% high on the books: real counts were a median 0.8 of that estimate.
TOKEN_SCALE = 1.2

PROSE_KINDS = {"paragraph", "list_item", "text", "caption", "footnote"}

# Kept in the blocks table so the formula gate can still sample it, but never
# chunked or embedded, because it isn't mathematics a reader should retrieve.
DROPPED_EQUATION = "equation_dropped"


def unverified_note(page: int | str | None) -> str:
    return f"[unverified formula, see page {page}]"


def estimate_tokens(text: str) -> int:
    """An estimate of the bge-m3 token count.

    Counts words and punctuation marks separately, then scales up. It needs no
    tokenizer model. Backslashes, braces and subscripts each count, which is
    what makes LaTeX cost more than prose of the same length.
    """
    return math.ceil(TOKEN_SCALE * len(re.findall(r"\w+|[^\w\s]", text)))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Block:
    order: int
    kind: str
    text: str
    page: int | None = None
    level: int = 0            # heading level, 1-6
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    kind: str
    seq: int
    heading_path: list[str]
    text: str
    token_count: int
    block_orders: list[int]
    page_start: int | None
    page_end: int | None
    meta: dict = field(default_factory=dict)


@dataclass
class _Piece:
    """One sentence, or one whole atomic block."""
    text: str
    block: Block
    tokens: int
    is_sentence: bool


# --------------------------------------------------------------------------
# Layer 1: Markdown -> blocks
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"^\s*!\[(.*?)\]\((.*?)\)\s*$")


def _closing_math_line(lines: list[str], start: int, closer: str) -> int | None:
    """Index of the first line after `start` that ends with `closer`."""
    for j in range(start + 1, len(lines)):
        if lines[j].rstrip().endswith(closer):
            return j
    return None


def _is_table_start(lines: list[str], i: int) -> bool:
    if "|" not in lines[i]:
        return False
    if lines[i].lstrip().startswith("|"):
        return True
    next_line = lines[i + 1] if i + 1 < len(lines) else ""
    return "-" in next_line and "|" in next_line and bool(_TABLE_DELIMITER.match(next_line))


def _starts_block(lines: list[str], i: int) -> bool:
    line = lines[i]
    stripped = line.strip()
    return bool(
        not stripped
        or _HEADING.match(line)
        or _FENCE.match(line)
        or stripped.startswith(("$$", "\\["))
        or _is_table_start(lines, i)
        or _LIST_ITEM.match(line)
        or _IMAGE.match(line)
    )


def blocks_from_markdown(markdown: str) -> list[Block]:
    """Split Markdown into blocks, never breaking inside math, code or tables.

    An unterminated code fence runs to the end of the file, as in CommonMark.
    An unterminated `$$` does not: one stray delimiter from the parser would
    otherwise swallow the rest of the book, so it stays an ordinary paragraph.
    """
    lines = markdown.splitlines()
    blocks: list[Block] = []
    i = 0

    def add(kind: str, text: str, **kwargs) -> None:
        blocks.append(Block(order=len(blocks), kind=kind, text=text, **kwargs))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip()
                if (candidate
                        and set(candidate) == {marker[0]}
                        and len(candidate) >= len(marker)):
                    break
                j += 1
            end = min(j, len(lines) - 1)
            add("code", "\n".join(lines[i:end + 1]))
            i = end + 1
            continue

        for opener, closer in (("$$", "$$"), ("\\[", "\\]")):
            if not stripped.startswith(opener):
                continue
            if stripped.endswith(closer) and len(stripped) >= len(opener) + len(closer):
                end = i
            else:
                end = _closing_math_line(lines, i, closer)
            if end is not None:
                add("equation", "\n".join(lines[i:end + 1]))
                i = end + 1
                break
        else:
            end = None
        if end is not None:
            continue

        heading = _HEADING.match(line)
        if heading:
            add("heading", heading.group(2), level=len(heading.group(1)))
            i += 1
            continue

        image = _IMAGE.match(line)
        if image:
            add("figure", image.group(1), meta={"image_path": image.group(2)})
            i += 1
            continue

        if _is_table_start(lines, i):
            j = i
            while j < len(lines) and lines[j].strip() and "|" in lines[j]:
                j += 1
            add("table", "\n".join(lines[i:j]))
            i = j
            continue

        kind = "list_item" if _LIST_ITEM.match(line) else "paragraph"
        j = i + 1
        while j < len(lines) and not _starts_block(lines, j):
            j += 1
        add(kind, "\n".join(lines[i:j]).strip())
        i = j

    return blocks


def load_blocks_jsonl(path: Path) -> list[Block]:
    """Read the sidecar written by docling_structured.py."""
    blocks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            known = {"order", "kind", "text", "page", "level"}
            blocks.append(Block(
                order=record["order"],
                kind=record["kind"],
                text=record["text"],
                page=record.get("page"),
                level=record.get("level", 0),
                meta={k: v for k, v in record.items() if k not in known},
            ))
    return blocks


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "al.", "fig.", "figs.", "eq.", "eqs.",
    "ref.", "refs.", "sec.", "secs.", "ch.", "chap.", "dr.", "prof.", "no.",
    "vol.", "approx.", "resp.", "thm.", "def.", "prop.", "lem.",
    # German
    "z.b.", "d.h.", "u.a.", "bzw.", "abb.", "gl.", "s.", "nr.", "vgl.", "bsp.",
}
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[$\\]*[A-ZÄÖÜ0-9$\\])")
_PROTECTED = re.compile(r"\$[^$]+\$|`[^`]+`")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, leaving inline math and abbreviations alone."""
    normalised = " ".join(text.split())
    protected = [m.span() for m in _PROTECTED.finditer(normalised)]
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(normalised):
        if any(a <= match.start() < b for a, b in protected):
            continue
        before = normalised[start:match.start()].rstrip()
        last_word = before.rsplit(None, 1)[-1].lower() if before else ""
        if last_word.lstrip("(['\"") in _ABBREVIATIONS:
            continue
        if re.fullmatch(r"\(?[a-z]\.", last_word):   # an initial, "A. Einstein"
            continue
        sentences.append(normalised[start:match.start()].strip())
        start = match.end()
    tail = normalised[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


_CONTINUATION = re.compile(
    r"(where|whereby|with|here|in which|for which|such that|wobei|dabei|hierbei|mit)\b",
    re.IGNORECASE,
)


def continues_equation(text: str) -> bool:
    """True when prose after a display equation reads as a continuation of it."""
    stripped = text.lstrip()
    return bool(stripped) and (stripped[0].islower() or bool(_CONTINUATION.match(stripped)))


# --------------------------------------------------------------------------
# Layer 2: blocks -> units
# --------------------------------------------------------------------------


def _unit_tokens(unit: list[_Piece]) -> int:
    return sum(piece.tokens for piece in unit)


def _units(blocks: list[Block], budget: int, count: Callable[[str], int]) -> list[list[_Piece]]:
    units: list[list[_Piece]] = []
    glue_next = False   # the last unit must stay with whatever comes next

    def sentence(text: str, block: Block) -> _Piece:
        return _Piece(text, block, count(text), True)

    def attach(pieces: list[_Piece]) -> None:
        """Extend the last unit, unless that would overflow the budget."""
        fits = units and _unit_tokens(units[-1]) + _unit_tokens(pieces) <= budget
        if fits:
            units[-1].extend(pieces)
        else:
            units.append(pieces)

    for i, block in enumerate(blocks):
        previous = blocks[i - 1] if i else None
        following = blocks[i + 1] if i + 1 < len(blocks) else None
        after_equation = previous is not None and previous.kind == "equation"
        before_equation = following is not None and following.kind == "equation"

        if block.kind == "equation":
            piece = _Piece(block.text, block, count(block.text), False)
            if glue_next or after_equation:
                attach([piece])
            else:
                units.append([piece])
            glue_next = False
            continue

        if block.kind not in PROSE_KINDS:     # code, table, inline heading
            piece = _Piece(block.text, block, count(block.text), False)
            if glue_next:
                attach([piece])
            else:
                units.append([piece])
            glue_next = block.kind == "heading"
            continue

        sentences = [sentence(s, block) for s in split_sentences(block.text)]
        if not sentences:
            continue

        where = None
        if after_equation and continues_equation(block.text):
            where = sentences.pop(0)
        lead_in = sentences.pop() if before_equation and sentences else None

        if where is not None:
            attach([where])
        if sentences:
            whole = _unit_tokens(sentences)
            if glue_next:
                attach(sentences if whole <= budget else sentences[:1])
                if whole > budget:
                    sentences = sentences[1:]
                else:
                    sentences = []
                glue_next = False
            if sentences:
                if _unit_tokens(sentences) <= budget:
                    units.append(sentences)
                else:
                    units.extend([piece] for piece in sentences)
        if lead_in is not None:
            if glue_next:
                attach([lead_in])
            else:
                units.append([lead_in])
            glue_next = True
        elif where is not None and before_equation and not sentences:
            glue_next = True      # one sentence: "where ... and then"
        else:
            glue_next = False

    return units


# --------------------------------------------------------------------------
# Layer 3: units -> chunks
# --------------------------------------------------------------------------


def _overlap_piece(group: list[list[_Piece]], next_group_tokens: int, budget: int) -> _Piece | None:
    """The sentence to repeat at the start of the next group, if there is one."""
    last = group[-1][-1]
    if (last.is_sentence
            and last.tokens <= OVERLAP_MAX_TOKENS
            and last.tokens + next_group_tokens <= budget):
        return last
    return None


def _pack(units: list[list[_Piece]], budget: int, minimum: int) -> list[list[_Piece]]:
    """Group units greedily, rebalance a short tail, then add the overlap."""
    groups: list[list[list[_Piece]]] = []
    current: list[list[_Piece]] = []
    current_tokens = 0

    for unit in units:
        size = _unit_tokens(unit)
        if current and current_tokens + size > budget:
            groups.append(current)
            current, current_tokens = [], 0
            # Budget for the sentence the overlap pass will prepend.
            overlap = _overlap_piece(groups[-1], size, budget)
            current_tokens = overlap.tokens if overlap else 0
        current.append(unit)
        current_tokens += size
    if current:
        groups.append(current)

    def tokens(group: list[list[_Piece]]) -> int:
        return sum(_unit_tokens(unit) for unit in group)

    if len(groups) >= 2 and tokens(groups[-1]) < minimum:
        previous, tail = groups[-2], groups[-1]
        if tokens(previous) + tokens(tail) <= budget:
            groups[-2:] = [previous + tail]
        else:
            # Too full to merge: shift whole units back while both stay viable.
            while (len(previous) > 1
                   and tokens(tail) < minimum
                   and tokens(previous) - _unit_tokens(previous[-1]) >= minimum):
                tail.insert(0, previous.pop())

    packed: list[list[_Piece]] = []
    for i, group in enumerate(groups):
        pieces = [piece for unit in group for piece in unit]
        overlap = _overlap_piece(groups[i - 1], tokens(group), budget) if i else None
        packed.append([overlap, *pieces] if overlap else pieces)
    return packed


def _render(pieces: Iterable[_Piece]) -> str:
    out = ""
    previous = None
    for piece in pieces:
        if previous is None:
            out = piece.text
        elif piece.block is previous.block:
            out += " " + piece.text
        else:
            out += "\n\n" + piece.text
        previous = piece
    return out


def _with_note(block: Block) -> Block:
    """An unverified formula as a chunk shows it: the LaTeX stays, under a warning.

    Only the chunk gets the note. The block keeps its bare LaTeX for compiling.
    """
    if block.kind == "equation" and block.meta.get("unverified"):
        return replace(block, text=f"{unverified_note(block.page)}\n{block.text}")
    return block


def _crops(members: list[Block]) -> dict:
    """What stands in for a chunk's unverified LaTeX: a crop for the reader, symbols for the model."""
    found = [{"page": block.page, "crop": block.meta.get("crop"),
              "text_layer": block.meta.get("text_layer", "")}
             for block in members if block.meta.get("unverified")]
    return {"unverified_formulas": found} if found else {}


def _prefix(path: list[str]) -> str:
    return PATH_SEPARATOR.join(path) + " ›\n\n" if path else ""


def chunk_blocks(
    blocks: list[Block],
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    count: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    blocks = [_with_note(block) for block in blocks if block.kind != DROPPED_EQUATION]
    chunks: list[Chunk] = []
    path_by_level: dict[int, str] = {}
    section: list[Block] = []
    figures: list[Block] = []

    def emit(kind: str, path: list[str], text: str, members: list[Block], meta: dict) -> None:
        pages = [b.page for b in members if b.page is not None]
        pages += [region["page"] for b in members for region in b.meta.get("bboxes") or []]
        chunks.append(Chunk(
            kind=kind,
            seq=len(chunks),
            heading_path=list(path),
            text=text,
            token_count=count(text),
            block_orders=sorted({b.order for b in members}),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            meta=meta,
        ))

    def close_section() -> None:
        path = [path_by_level[level] for level in sorted(path_by_level)]
        prefix = _prefix(path)
        if section:
            budget = max_tokens - count(prefix)
            units = _units(section, budget, count)
            packed = _pack(units, budget, min_tokens)
            for index, pieces in enumerate(packed):
                # The repeated sentence is read for context but belongs to the chunk before: its
                # block would set the page and focus one page early (review R2-C4, Nov 2026).
                repeated = index > 0 and pieces[0] is packed[index - 1][-1]
                members = [piece.block for piece in (pieces[1:] if repeated else pieces)]
                emit("text", path, prefix + _render(pieces), members, _crops(members))
        for figure in figures:
            if figure.text.strip():
                emit("figure", path, prefix + figure.text, [figure], dict(figure.meta))
        section.clear()
        figures.clear()

    for block in blocks:
        if block.kind == "heading" and block.level <= SPLIT_HEADING_LEVEL:
            close_section()
            for level in [lv for lv in path_by_level if lv >= block.level]:
                del path_by_level[level]
            path_by_level[block.level] = block.text
        elif block.kind == "figure":
            figures.append(block)    # text flows around it, so glue survives
        else:
            section.append(block)
    close_section()
    return chunks


def chunk_markdown(markdown: str, **kwargs) -> list[Chunk]:
    return chunk_blocks(blocks_from_markdown(markdown), **kwargs)
