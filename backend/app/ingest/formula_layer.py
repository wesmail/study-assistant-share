"""Facts about a display formula that the PDF text layer states exactly.

A vision model reads shapes, so it can miss what is obvious to the file itself:
which font drew a glyph, and which number is printed beside an equation. This
module takes the model's LaTeX and restores both from the text layer.

Fonts. The glyphs of a formula come out of the PDF in nearly the order TeX wrote
them, which is nearly the order of the LaTeX source. So the LaTeX symbols and the
glyphs are two versions of one sequence, and a longest-common-subsequence
alignment pairs them. That keeps a bold y and a plain y_i apart inside one
formula. A symbol that doesn't align falls back to its letter, but only if every
instance of that letter in the formula has the same style.

Numbers. A printed equation number is a word such as "(4.20)" with a clear gap
before it, which separates it from the "(2)" that ends "f(2)".

Everything here takes plain data, so it needs no model and no GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.ingest.pdf_fonts import PageGlyphNames

BOLD_ITALIC = "boldsymbol"
BOLD_UPRIGHT = "mathbf"
BLACKBOARD = "mathbb"
CALLIGRAPHIC = "mathcal"
SCRIPT = "mathscr"
BOLD_CALLIGRAPHIC = "boldcal"

# The order matters: CMBSY is calligraphic before it is bold.
_FONT_STYLES = [
    (re.compile(r"CMBSY", re.IGNORECASE), BOLD_CALLIGRAPHIC),
    (re.compile(r"MSBM|dsrom|bbm|bbold|blackboard", re.IGNORECASE), BLACKBOARD),
    (re.compile(r"rsfs|mathscr", re.IGNORECASE), SCRIPT),
    (re.compile(r"^(CMSY|LMMathSymbols)", re.IGNORECASE), CALLIGRAPHIC),
    (re.compile(r"CMMIB|LMMathItalic\d*-Bold|BoldItalic|Bold-?Oblique", re.IGNORECASE), BOLD_ITALIC),
    (re.compile(r"CMBX|CMB\d|Bold", re.IGNORECASE), BOLD_UPRIGHT),
]

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "varpi": "π", "rho": "ρ",
    "varrho": "ρ", "sigma": "σ", "varsigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
    "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
# Variant glyphs of one letter, and the Greek letters that share a code point
# with a mathematical sign in some fonts.
_GLYPH_ALIASES = {
    "ϕ": "φ", "ϵ": "ε", "ϑ": "θ", "ϖ": "π", "ϱ": "ρ", "ς": "σ",
    "\u00b5": "μ", "\u2126": "Ω", "\u2206": "Δ",
}
# Commands that TeX prints as letters. They stay as they are, but their glyphs
# are in the PDF, so the alignment has to expect them.
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_WORD_OPERATORS = {
    "log", "ln", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh",
    "tanh", "det", "dim", "ker", "min", "max", "arg", "lim", "sup", "inf", "gcd", "Pr",
    "arcsin", "arccos", "arctan", "argmin", "argmax", "mod",
}
_STYLED_COMMANDS = {
    "mathbf", "boldsymbol", "bm", "pmb", "mathbb", "mathcal", "mathscr", "mathfrak", "mathbbm",
}
_TEXT_COMMANDS = {
    "text", "textrm", "textbf", "textit", "mathrm", "mathsf", "mathtt", "mathit",
    "operatorname", "operatorname*", "rm", "tt", "bf",
}
_ARGUMENT_COMMANDS = {"tag", "label", "hspace", "vspace", "rule", "color"}


@dataclass(frozen=True)
class Glyph:
    char: str
    style: str | None
    x: float = 0.0
    y: float = 0.0
    middle: float | None = None     # a big delimiter's other height; see page_trace


@dataclass
class _Atom:
    """A symbol in the LaTeX and the slice of source that spells it."""
    start: int
    end: int
    symbol: str
    source: str
    locked: bool = False            # already styled, or part of a word operator
    group: int | None = None        # the \text{...} it sits in
    style: str | None = field(default=None)


@dataclass
class _TextGroup:
    start: int                      # at the backslash
    end: int                        # just past the closing brace
    body_start: int
    body_end: int
    operator: bool = False          # a command like \\max, not a \\text{...} group


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


UNNAMED = "\ufffd"

# A tall bracket is drawn from pieces. The top piece stands for the whole bracket
# and the others stand for nothing, however many the height needs.
_BRACKET_PIECES = {
    **dict.fromkeys("\uf8eb\u239b", "("), **dict.fromkeys("\uf8ee\u23a1", "["),
    **dict.fromkeys("\uf8f1\u23a7", "{"), **dict.fromkeys("\uf8f6\u239e", ")"),
    **dict.fromkeys("\uf8f9\u23a4", "]"), **dict.fromkeys("\uf8fc\u23ab", "}"),
    **dict.fromkeys("\uf8ec\uf8ed\uf8ef\uf8f0\uf8f2\uf8f3\uf8f4\uf8f5\uf8f7\uf8f8"
                    "\uf8fa\uf8fb\uf8fd\uf8fe"
                    "\u239c\u239d\u239f\u23a0\u23a2\u23a3\u23a5\u23a6"
                    "\u23a8\u23a9\u23aa\u23ac\u23ad", ""),
}
_BRACKET_NAME = re.compile(r"(paren|bracket|brace|angbracket)(left|right)(big|Big|bigg|Bigg|tp)$")
_BRACKET_SIGNS = {
    ("paren", "left"): "(", ("paren", "right"): ")", ("bracket", "left"): "[",
    ("bracket", "right"): "]", ("brace", "left"): "{", ("brace", "right"): "}",
    ("angbracket", "left"): "⟨", ("angbracket", "right"): "⟩",
}
_NAMED_SIGNS = [
    (re.compile(r"summation(display|text)$"), "∑"), (re.compile(r"product(display|text)$"), "∏"),
    (re.compile(r"(cont)?integral(display|text)$"), "∫"), (re.compile(r"radical(big|Big|bigg|Bigg|bt)$"), "√"),
    (re.compile(r"hatwide(r|st)?$"), "ˆ"), (re.compile(r"tildewide(r|st)?$"), "˜"),
    (re.compile(r"bracehtipupleft$"), "⏟"),      # one of the four tips of an under- or overbrace
]


def sign_of_glyph_name(name: str) -> str:
    """The sign a named glyph draws, or "" if it's a piece or something not compared."""
    bracket = _BRACKET_NAME.match(name)
    if bracket:
        return _BRACKET_SIGNS[bracket.group(1), bracket.group(2)]
    return next((sign for pattern, sign in _NAMED_SIGNS if pattern.match(name)), "")


def _decoded(code: int, gid: int, font: str, names: PageGlyphNames) -> str:
    """What a glyph draws, as text. "" is a piece of a sign already counted."""
    char = chr(code)
    if char in _BRACKET_PIECES:
        return _BRACKET_PIECES[char]
    if char != UNNAMED:
        return char
    name = names.name(font, gid)
    return UNNAMED if name is None else sign_of_glyph_name(name)


def page_trace(page) -> list[Glyph]:
    """Every visible glyph on a PyMuPDF page, in the order the PDF drew them."""
    found = []
    names = PageGlyphNames(page)
    for span in page.get_texttrace():
        if span["type"] == 3:                # invisible OCR layer
            continue
        style = font_style(span["font"])
        for code, gid, (x, y), box in span["chars"]:
            char = _decoded(code, gid, span["font"], names) if code > 0 else ""
            if not char:
                continue
            # A big delimiter hangs below its origin, unlike a letter, which sits
            # on it, so its origin can fall outside a formula's box. Its middle is
            # usually inside, but font formats differ on that box, so either will do.
            middle = None if char == chr(code) else (box[1] + box[3]) / 2
            found.append(Glyph(char, style, x, y, middle))
    return found


def glyphs_in(trace: list[Glyph], bbox: list[float], pad: float = 1.5) -> list[Glyph]:
    top, bottom = bbox[1] - pad, bbox[3] + pad
    return [glyph for glyph in trace
            if bbox[0] - pad <= glyph.x <= bbox[2] + pad
            and (top <= glyph.y <= bottom
                 or (glyph.middle is not None and top <= glyph.middle <= bottom))]


_SLASH = "\u0338"              # TeX draws a Feynman slash before its letter


def text_layer(glyphs: list[Glyph]) -> str:
    """The symbols a PDF box prints, in drawing order: right symbols, layout lost.

    A jump backwards is a new row or the start of a denominator, so it becomes a space.
    """
    out = ""
    previous = None
    for glyph in glyphs:
        if previous is not None and glyph.x < previous.x - 1:
            out += " "
        out += glyph.char
        previous = glyph
    return re.sub(f"{_SLASH}(.)", rf"\1{_SLASH}", out).strip()


def page_words(page) -> list[Word]:
    return [Word(w[4], w[0], w[1], w[2], w[3]) for w in page.get_text("words")]


def font_style(font: str) -> str | None:
    """The mathematical alphabet a font draws, or None for an ordinary one."""
    name = font.split("+")[-1]
    for pattern, style in _FONT_STYLES:
        if pattern.search(name):
            return style
    return None


def canonical(char: str) -> str:
    return _GLYPH_ALIASES.get(char, char)


def _is_symbol(char: str) -> bool:
    return char.isalnum() and (char.isascii() or canonical(char) in GREEK.values())


def glyph_symbols(glyphs: list[Glyph]) -> list[str]:
    """The letters, digits and Greek that a PDF box actually prints."""
    return [symbol for glyph in glyphs
            for symbol in _LIGATURES.get(glyph.char, canonical(glyph.char))
            if _is_symbol(symbol)]


def _argument_span(latex: str, start: int) -> tuple[int, int] | None:
    """The span of a {group} or single token at `start`, past any spaces."""
    i = start
    while i < len(latex) and latex[i] == " ":
        i += 1
    if i >= len(latex):
        return None
    if latex[i] == "{":
        depth, j = 0, i
        while j < len(latex):
            if latex[j] == "\\":
                j += 2
                continue
            if latex[j] == "{":
                depth += 1
            elif latex[j] == "}":
                depth -= 1
                if depth == 0:
                    return i, j + 1
            j += 1
        return None
    if latex[i] == "\\":
        match = re.compile(r"\\[a-zA-Z]+|\\.").match(latex, i)
        return (i, match.end()) if match else None
    return i, i + 1


def _tokenize(latex: str) -> tuple[list[_Atom], list[_TextGroup]]:
    atoms: list[_Atom] = []
    groups: list[_TextGroup] = []
    i = 0
    while i < len(latex):
        char = latex[i]
        if char == "\\":
            match = re.compile(r"\\([a-zA-Z]+\*?)").match(latex, i)
            if match is None:
                i += 2                       # \{ or \, or a control space
                continue
            name, after = match.group(1), match.end()
            if name in GREEK:
                atoms.append(_Atom(i, after, canonical(GREEK[name]), latex[i:after]))
                i = after
            elif name in _STYLED_COMMANDS:
                span = _argument_span(latex, after)
                if span is None:
                    i = after
                    continue
                for symbol in re.findall(r"\\[a-zA-Z]+|[a-zA-Z0-9]", latex[span[0]:span[1]]):
                    if symbol.startswith("\\"):
                        symbol = GREEK.get(symbol[1:], "")
                    if symbol:
                        atoms.append(_Atom(span[0], span[1], canonical(symbol), "", locked=True))
                i = span[1]
            elif name in _TEXT_COMMANDS:
                span = _argument_span(latex, after)
                if span is None or latex[span[0]] != "{":
                    i = after
                    continue
                group = len(groups)
                groups.append(_TextGroup(i, span[1], span[0] + 1, span[1] - 1))
                for j in range(span[0] + 1, span[1] - 1):
                    if _is_symbol(latex[j]):
                        atoms.append(_Atom(j, j + 1, canonical(latex[j]), latex[j], group=group))
                i = span[1]
            elif name in ("begin", "end"):
                span = _argument_span(latex, after)
                i = span[1] if span else after
                if name == "begin" and span and latex[span[0] + 1:span[1] - 1] in ("array", "tabular"):
                    columns = _argument_span(latex, i)
                    i = columns[1] if columns else i
            elif name in _ARGUMENT_COMMANDS:
                span = _argument_span(latex, after)
                i = span[1] if span else after
            elif name in _WORD_OPERATORS:
                group = len(groups)
                groups.append(_TextGroup(i, after, i + 1, after, operator=True))
                atoms.extend(_Atom(i, after, letter, "", group=group) for letter in name)
                i = after
            else:
                i = after
            continue
        if _is_symbol(char):
            atoms.append(_Atom(i, i + 1, canonical(char), char))
        i += 1
    return atoms, groups


def latex_symbols(latex: str) -> list[str]:
    """The letters, digits and Greek a formula spells, as a PDF would print them.

    Commands that print words count as their letters, and \\tag is left out.
    """
    return [atom.symbol for atom in _tokenize(latex)[0]]


# --------------------------------------------------------------------------
# Marks: the operators, accents and brackets of a formula, in one vocabulary for
# the LaTeX and for the page. Vertical bars and dots are left out, because they
# have too many spellings to count, and so is a bar accent, which may be a drawn
# rule with no glyph. `\\top` is compared as a letter, in formula_merge.
# --------------------------------------------------------------------------

_NOT = "\u0338"
_MARK_COMMANDS = {
    "sum": "∑", "prod": "∏", "int": "∫", "iint": "∫∫", "iiint": "∫∫∫", "oint": "∫", "sqrt": "√",
    "hat": "ˆ", "widehat": "ˆ", "tilde": "˜", "widetilde": "˜", "dot": "˙", "ddot": "¨",
    "check": "ˇ", "breve": "˘", "widecheck": "˘", "vec": "→", "overrightarrow": "→",
    "underbrace": "⏟", "overbrace": "⏟",
    "leq": "≤", "le": "≤", "leqslant": "≤", "geq": "≥", "ge": "≥", "geqslant": "≥",
    "lt": "<", "gt": ">", "neq": "=" + _NOT, "ne": "=" + _NOT, "approx": "≈", "sim": "∼", "simeq": "≃",
    "equiv": "≡", "propto": "∝", "triangleq": "≜", "coloneqq": ":=", "colon": ":",
    "in": "∈", "notin": "∈" + _NOT, "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅", "setminus": "\\", "backslash": "\\",
    "to": "→", "rightarrow": "→", "mapsto": "→", "longrightarrow": "→", "longmapsto": "→",
    "leftarrow": "←", "longleftarrow": "←", "gets": "←", "Rightarrow": "⇒", "Longrightarrow": "⇒",
    "implies": "⇒", "Leftarrow": "⇐", "Longleftarrow": "⇐", "impliedby": "⇐", "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔", "iff": "⇔",
    "forall": "∀", "exists": "∃", "nabla": "∇", "partial": "∂", "infty": "∞",
    "pm": "±", "mp": "∓", "times": "×", "otimes": "⊗", "oplus": "⊕", "odot": "⊙",
    "ast": "∗", "prime": "′", "langle": "⟨", "rangle": "⟩",
    "lbrace": "{", "rbrace": "}", "lbrack": "[", "rbrack": "]",
    "binom": "()", "dbinom": "()", "tbinom": "()",
}
_ESCAPED_MARKS = {"\\{": "{", "\\}": "}"}
_PLAIN_MARKS = {"=": "=", "+": "+", "-": "−", "<": "<", ">": ">", "/": "/", "!": "!", ":": ":",
                "(": "(", ")": ")", "[": "[", "]": "]", "*": "∗", "'": "′"}
_ENVIRONMENT_FENCES = {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"), "Bmatrix": (r"\{", r"\}"),
                       "cases": (r"\{", ""), "dcases": (r"\{", "")}
_GLYPH_MARKS = {
    **{mark: mark for marks in _MARK_COMMANDS.values() for mark in marks},
    **{mark: mark for mark in _PLAIN_MARKS.values()},
    "-": "−", "\u2013": "−", "⩽": "≤", "⩾": "≥", "≠": "=" + _NOT, "∉": "∈" + _NOT, "*": "∗", "'": "′",
    "^": "ˆ", "⌢": "ˆ", "~": "˜", "⌣": "˘", "\u20d7": "→", "↦": "→",
}
# TeX draws a long arrow from a short one and a line.
_LONG_ARROWS = (("⇐=⇒", "⇔"), ("⇐⇒", "⇔"), ("=⇒", "⇒"), ("⇐=", "⇐"), ("−→", "→"), ("←−", "←"))
_NOT_MATHEMATICS = re.compile(
    r"\\(?:tag\*?|hspace\*?|vspace\*?|label|rule)\s*\{[^{}]*\}|\\\\\s*\[[^\]]*\]")
_COLUMN_SPEC = re.compile(r"(\\begin\{(?:array|tabular)\})\{[^{}]*\}")
_ENVIRONMENT = re.compile(r"\\(begin|end)\{([a-zA-Z]+\*?)\}")
_MARK_TOKEN = re.compile(r"\\[a-zA-Z]+\*?|\\.|\S")


def _fence(match: re.Match) -> str:
    opening, closing = _ENVIRONMENT_FENCES.get(match.group(2), ("", ""))
    return f" {opening if match.group(1) == 'begin' else closing} "


def latex_marks(latex: str) -> list[str]:
    """The operators, accents and brackets a formula would print."""
    body = _COLUMN_SPEC.sub(r"\1", _NOT_MATHEMATICS.sub(" ", latex))
    marks: list[str] = []
    for token in _MARK_TOKEN.findall(_ENVIRONMENT.sub(_fence, body)):
        if token in _ESCAPED_MARKS:
            marks.append(_ESCAPED_MARKS[token])
        elif token.startswith("\\"):
            marks.extend(_MARK_COMMANDS.get(token[1:].rstrip("*"), ""))
        elif token in _PLAIN_MARKS:
            marks.append(_PLAIN_MARKS[token])
    return marks


def glyph_marks(glyphs: list[Glyph]) -> list[str]:
    """The operators, accents and brackets a PDF box prints."""
    printed = "".join(glyph.char for glyph in glyphs)
    for pieces, arrow in _LONG_ARROWS:
        printed = printed.replace(pieces, arrow)
    return [mark for char in printed for mark in _GLYPH_MARKS.get(char, "")]


def has_unnamed_glyphs(glyphs: list[Glyph]) -> bool:
    """True if a glyph in the box can't be identified, so marks can't be compared."""
    return any(glyph.char == UNNAMED for glyph in glyphs)


def _wrap(atom: _Atom, style: str) -> str:
    source = atom.source
    greek = source.startswith("\\")
    if style == BOLD_CALLIGRAPHIC:
        return f"\\boldsymbol{{\\mathcal{{{source}}}}}" if atom.symbol.isupper() else source
    if style in (CALLIGRAPHIC, SCRIPT):
        return f"\\{style}{{{source}}}" if atom.symbol.isascii() and atom.symbol.isupper() else source
    if style == BLACKBOARD:
        return f"\\mathbb{{{source}}}" if atom.symbol.isascii() and atom.symbol.isupper() else source
    if style == BOLD_UPRIGHT and greek:
        return f"\\boldsymbol{{{source}}}"
    return f"\\{style}{{{source}}}"


def _align(atoms: list[_Atom], glyphs: list[Glyph]) -> int:
    """Copy styles from the glyphs onto the atoms. Returns how many atoms aligned."""
    wanted = [atom.symbol for atom in atoms]
    printed = [canonical(glyph.char) for glyph in glyphs]
    aligned = 0
    placed = [False] * len(atoms)
    for block in SequenceMatcher(None, wanted, printed, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            atoms[block.a + offset].style = glyphs[block.b + offset].style
            placed[block.a + offset] = True
            aligned += 1

    by_symbol: dict[str, set[str | None]] = {}
    for symbol, glyph in zip(printed, glyphs):
        by_symbol.setdefault(symbol, set()).add(glyph.style)
    for atom, done in zip(atoms, placed):
        styles = by_symbol.get(atom.symbol, set())
        if not done and len(styles) == 1:
            atom.style = next(iter(styles))
    return aligned


def restore_styles(latex: str, glyphs: list[Glyph]) -> str:
    """The LaTeX with the alphabet of each symbol taken from the PDF's fonts.

    A formula whose glyphs barely overlap with it is left alone: that means the
    region or the reading is off, and copying styles would only add noise.
    """
    glyphs = [glyph for glyph in glyphs if _is_symbol(glyph.char)]
    atoms, groups = _tokenize(latex)
    if not atoms or not glyphs:
        return latex
    if _align(atoms, glyphs) < 0.5 * min(len(atoms), len(glyphs)):
        return latex

    edits: list[tuple[int, int, str]] = []
    for index, group in enumerate(groups):
        members = [atom for atom in atoms if atom.group == index]
        styles = {atom.style for atom in members}
        if len(styles) == 1 and next(iter(styles)) in (BOLD_UPRIGHT, BOLD_ITALIC):
            bold = f"\\mathbf{{{latex[group.body_start:group.body_end]}}}"
            edits.append((group.start, group.end, f"\\mathop{{{bold}}}" if group.operator else bold))
    for atom in atoms:
        if atom.locked or atom.group is not None or atom.style is None:
            continue
        edits.append((atom.start, atom.end, _wrap(atom, atom.style)))

    for start, end, replacement in sorted(edits, reverse=True):
        latex = latex[:start] + replacement + latex[end:]
    return latex


_NUMBER_WORD = re.compile(r"^\((\d+(?:\.\d+)*[a-z]?|[A-Z]\.\d+[a-z]?)\)[.,;]?$")
NUMBER_MIN_GAP = 8.0
NUMBER_MIN_POSITION = 0.5
NUMBER_MIN_HEIGHT = 0.85        # of the page's tallest: a smaller one is a margin note citing it


def printed_numbers(words: list[Word], page_width: float) -> list[Word]:
    """Words on a page that stand alone at the end of a line as equation numbers."""
    found = []
    for word in words:
        if not _NUMBER_WORD.match(word.text) or word.x0 < NUMBER_MIN_POSITION * page_width:
            continue
        middle = (word.y0 + word.y1) / 2
        before = [w.x1 for w in words if w.x1 <= word.x0 + 0.5 and w.y0 < middle < w.y1]
        if not before or word.x0 - max(before) >= NUMBER_MIN_GAP:
            found.append(word)
    tallest = max((word.y1 - word.y0 for word in found), default=0.0)
    return [word for word in found if word.y1 - word.y0 >= NUMBER_MIN_HEIGHT * tallest]


def number_of(word: Word) -> str:
    return _NUMBER_WORD.match(word.text).group(1)


def assign_numbers(
    boxes: list[tuple[float, float, float, float]], numbers: list[Word]
) -> list[list[str]]:
    """Give each printed number to the formula box around it.

    Boxes of neighbouring equations can overlap, so a number inside two boxes
    goes to the one whose centre is nearest.
    """
    assigned: list[list[str]] = [[] for _ in boxes]
    for word in sorted(numbers, key=lambda w: w.y0):
        middle = (word.y0 + word.y1) / 2
        holders = [i for i, box in enumerate(boxes) if box[1] - 1 <= middle <= box[3] + 1]
        if holders:
            best = min(holders, key=lambda i: abs((boxes[i][1] + boxes[i][3]) / 2 - middle))
            assigned[best].append(number_of(word))
    return assigned


def strip_number(latex: str, number: str) -> str:
    """Remove a number the model copied into the formula, however it spaced it."""
    spaced = r"\s*".join(re.escape(char) for char in number)
    bracketed = rf"\(\s*{spaced}\s*\)"
    pattern = (rf"(?:\s|&|\\quad|\\qquad|\\,|\\;|\\ )*"
               rf"(?:\\text\s*\{{\s*{bracketed}\s*\}}|\\tag\s*\{{\s*{spaced}\s*\}}|{bracketed})"
               rf"(?:\^\{{[^{{}}]*\}})?")
    return re.sub(pattern, "", latex)


def tag_for(numbers: list[str]) -> str:
    """One \\tag. A block printing (4.67a) and (4.67b) carries both."""
    return f" \\tag{{{', '.join(numbers)}}}" if numbers else ""


def _has_bare_ampersand(latex: str) -> bool:
    """True if an unescaped & sits outside every \\begin...\\end environment."""
    depth, i = 0, 0
    while i < len(latex):
        if latex.startswith("\\begin", i):
            depth += 1
            i += 6
        elif latex.startswith("\\end", i):
            depth -= 1
            i += 4
        elif latex[i] == "\\":
            i += 2
        elif latex[i] == "&" and depth <= 0:
            return True
        else:
            i += 1
    return False


def wrap_alignment(latex: str) -> str:
    """Tidy a line break at the end, and give a bare & the environment it needs."""
    latex = re.sub(r"(?:\s|\\\\|\\,|\\;|\\quad)+$", "", latex.strip())
    if _has_bare_ampersand(latex):
        return f"\\begin{{aligned}}{latex}\\end{{aligned}}"
    return latex


_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\S")
_ONLY_A_NUMBER = re.compile(r"\(?[\s\d.]+[a-z]?\)?")
_SPACED_LETTERS = re.compile(r"(?:\b[A-Za-z] ){3,}[A-Za-z]\b")
_TEXT_RUN = re.compile(r"\\(?:text|mathrm|textrm)\s*\{([^{}]*)\}")
JUNK_MAX_LENGTH = 2500          # the longest real formula in the books is well below
JUNK_MIN_REPEATS = 8            # a zero row of a 9-column matrix is the worst real case
JUNK_MAX_PERIOD = 14
JUNK_SPACED_LETTERS = 12        # "o p t i m i z a t i o n": a word spelled out
JUNK_PROSE_WORDS = 6            # a sentence in \text{}, not "for all x"


def _longest_repeat(tokens: list[str]) -> int:
    """How many times the most-repeated run of tokens occurs in a row."""
    best = 1
    for period in range(1, min(JUNK_MAX_PERIOD, len(tokens) // 2) + 1):
        i = 0
        while i + period < len(tokens):
            j = i
            while j + period < len(tokens) and tokens[j] == tokens[j + period]:
                j += 1
            best = max(best, (j - i) // period + 1)
            i = max(j, i + 1)
    return best


def junk_reason(latex: str) -> str | None:
    """Why a formula isn't one, or None if it may be.

    These catch what a model produces when it reads past the equation: prose
    from the next line, a word spelled out letter by letter, or a loop.
    """
    body = re.sub(r"\\tag\s*\{[^{}]*\}", "", latex).strip()
    if not body or _ONLY_A_NUMBER.fullmatch(body):
        return "number"
    if "\\intertext" in body:
        return "prose"
    if len(body) > JUNK_MAX_LENGTH:
        return "length"
    longest_spelling = max((len(m.group(0)) // 2 + 1 for m in _SPACED_LETTERS.finditer(body)), default=0)
    if longest_spelling >= JUNK_SPACED_LETTERS:
        return "prose"
    if any(len(m.group(1).split()) >= JUNK_PROSE_WORDS for m in _TEXT_RUN.finditer(body)):
        return "prose"
    if _longest_repeat(_LATEX_TOKEN.findall(body)) >= JUNK_MIN_REPEATS:
        return "repeat"
    return None
