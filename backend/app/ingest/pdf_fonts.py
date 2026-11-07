"""Glyph names from fonts embedded in a PDF.

TeX's big delimiters and sums live in CMEX, whose glyphs have no Unicode value, so
PyMuPDF reports them as U+FFFD. The font still names each glyph ("bracketleftbigg",
"summationdisplay"), and the glyph id PyMuPDF gives is an index into that list.
"""
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache

log = logging.getLogger("ingest")

_EEXEC_KEY = 55665
_EEXEC_C1, _EEXEC_C2 = 52845, 22719
_EEXEC_RANDOM_BYTES = 4
_CHARSTRING = re.compile(rb"/([A-Za-z0-9._]+)\s+(\d+)\s+(?:RD|-\|)\s")
_TRACE_NAME_LENGTH = 24     # PyMuPDF's text trace cuts a font's name to this


def _eexec(cipher: bytes) -> bytes:
    key = _EEXEC_KEY
    plain = bytearray()
    for byte in cipher:
        plain.append(byte ^ (key >> 8))
        key = ((byte + key) * _EEXEC_C1 + _EEXEC_C2) & 0xFFFF
    return bytes(plain[_EEXEC_RANDOM_BYTES:])


def type1_glyph_names(program: bytes) -> list[str]:
    """A Type 1 font's glyphs in file order, with .notdef first as FreeType has it."""
    start = program.find(b"eexec")
    if start < 0:
        return []
    start += len(b"eexec")
    while program[start:start + 1] in (b" ", b"\r", b"\n", b"\t"):
        start += 1
    private = _eexec(program[start:])
    position = private.find(b"/CharStrings")
    if position < 0:
        return []
    names = []
    while (match := _CHARSTRING.search(private, position)) is not None:
        names.append(match.group(1).decode("ascii"))
        position = match.end() + int(match.group(2))        # skip the binary outline
    if ".notdef" in names:
        first = names.index(".notdef")
        names[0], names[first] = names[first], names[0]
    return names


def cff_glyph_names(program: bytes) -> list[str]:
    """A CFF font's charset, which is its glyphs in id order."""
    try:
        from fontTools.cffLib import CFFFontSet
    except ImportError:
        log.warning("fontTools is not installed, so CFF glyph names are unavailable")
        return []
    font_set = CFFFontSet()
    font_set.decompile(io.BytesIO(program), None)
    return list(font_set[font_set.fontNames[0]].charset)


@lru_cache(maxsize=256)
def _names_from(extension: str, program: bytes) -> tuple[str, ...]:
    reader = {"pfa": type1_glyph_names, "pfb": type1_glyph_names, "cff": cff_glyph_names}.get(extension)
    if reader is None:
        return ()
    try:
        return tuple(reader(program))
    except Exception as error:      # a damaged font must not stop a book
        log.warning("Could not read glyph names from a %s font: %s", extension, error)
        return ()


class PageGlyphNames:
    """Looks up glyph names on one page, opening each font only when it's needed."""

    def __init__(self, page) -> None:
        self._page = page
        self._tables: dict[str, tuple[str, ...]] = {}
        self._xrefs: dict[str, int] | None = None

    def name(self, font: str, gid: int) -> str | None:
        table = self._table(font.split("+")[-1])
        return table[gid] if 0 <= gid < len(table) else None

    def _xref(self, font: str) -> int | None:
        if self._xrefs is None:
            self._xrefs = {entry[3].split("+")[-1]: entry[0] for entry in self._page.get_fonts()}
        if font in self._xrefs or len(font) < _TRACE_NAME_LENGTH:
            return self._xrefs.get(font)
        fitting = [xref for name, xref in self._xrefs.items() if name.startswith(font)]
        return fitting[0] if len(fitting) == 1 else None

    def _table(self, font: str) -> tuple[str, ...]:
        if font not in self._tables:
            xref = self._xref(font)
            names: tuple[str, ...] = ()
            if xref is not None:
                _base, extension, _subtype, program = self._page.parent.extract_font(xref)
                names = _names_from(extension, program)
            self._tables[font] = names
        return self._tables[font]
