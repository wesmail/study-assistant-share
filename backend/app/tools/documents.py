"""Turning what the model wrote into what the builders take.

Nothing here reads a file the text asks for. An image reference is checked against
outputs/ by name, and only a file the app itself made gets copied into the work folder.
The sandbox would refuse anything else anyway; this gives the model a clear error.
"""
from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, Field

from app.config import ROOT
from app.tools.sandbox import safe_copy

# The limits and layout names live with the design, in the folder the image is built from,
# so this side and the builder can't disagree.
_spec = importlib.util.spec_from_file_location("theme", ROOT / "sandbox" / "theme.py")
theme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(theme)

CITED = re.compile(r"\[S(\d+)\]|\\tag\*?\{S(\d+)")
CITATION = re.compile(r"\s*\[S\d+\]")
TAGGED_SOURCE = re.compile(r"\s*\\tag\*?\{S\d+[^{}]*\}")
FIELD = re.compile(r"^\**(subtitle|caption|left|right|animate)\**\s*:\**\s*(.*)$", re.I)
LAYOUT_PREFIX = re.compile(r"^\[?([a-z_]+)\]?\s*[:\-\u2013]\s*(.+)$")
CLAUSE_END = (",", ";", ":", "\u2013", "\u2014")
SUMMARY_WORDS = re.compile(r"\b(summary|takeaways?|conclusions?|recap|key points)\b", re.I)
YES = ("yes", "true", "on", "1")
SLIDE_COUNT = re.compile(r"\b(\d{1,2})[- ]slides?\b|\b(?:with|in)\s+(\d{1,2})\s+slides\b", re.I)
INDEX_LETTER_WORDS = {"min", "in", "init", "lin", "mix"}
SCRIPT_WORD = re.compile(r"([_^])\{([A-Za-z]{2,})\}")
PATH_MARK = re.compile(r"[/\\:]|\.\.")
# Typography gpt-oss likes and these fonts lack: a glyph that's missing simply vanishes.
TYPOGRAPHY = str.maketrans({"\u2011": "-", "\u202f": " ", "\u2009": " ", "\u200a": " ", "\u200b": ""})
MAX_FOOTER_SOURCES = 2
MIN_CLAUSE_WORDS = 5
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*(<[^>]*>|[^)\s]+)([^)]*)\)")
OUTPUT_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|jpe?g)")
FENCE = re.compile(r"^```")
IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)\s*$")
NOTES = re.compile(r"^\**(?:speaker\s+)?notes\**\s*:\**\s*(.*)$", re.I)
BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
FORMULA_IN_LINE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]")
RAW_MATH = re.compile(r"\$|\\[A-Za-z(\[]")
MAX_TITLE_CHARS = 70
INLINE_FORMULA = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)\$|\\\((.+?)\\\)")
SCRIPT_ARGUMENT = r"(\{[^{}]*\}|\\?[A-Za-z0-9+\-=()*⊤])"
UNICODE_MACROS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "sim": "∼", "equiv": "≡", "cdot": "·", "times": "×", "pm": "±",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩", "sum": "Σ", "prod": "Π",
    "int": "∫", "infty": "∞", "partial": "∂", "nabla": "∇", "to": "→", "rightarrow": "→", "mapsto": "↦",
    "Rightarrow": "⇒", "ldots": "…", "dots": "…", "cdots": "⋯", "top": "⊤", "intercal": "⊤", "mid": "|",
    "forall": "∀", "exists": "∃", "circ": "∘", "ell": "ℓ", "hbar": "ℏ", "langle": "⟨", "rangle": "⟩",
    "det": "det", "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos", "tan": "tan",
    "max": "max", "min": "min", "lim": "lim", "tr": "tr", "dim": "dim", "rank": "rank",
}
BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼"}
SUPERSCRIPTS = str.maketrans("0123456789+-=()ni⊤T", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀᵀ")
SUBSCRIPTS = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")


class ContentError(ValueError):
    """Something the model can fix, in words it can act on."""


class Slide(BaseModel):
    layout: str | None = None
    title: str = Field(min_length=1)
    subtitle: str = ""
    bullets: list[str] = []
    equations: list[str] = []
    image: str | None = None
    caption: str = ""
    left_heading: str = ""
    right_heading: str = ""
    left_bullets: list[str] = []
    right_bullets: list[str] = []
    animate: bool = False
    notes: str = ""
    footer: str = ""
    cited: list[int] = []


class Deck(BaseModel):
    slides: list[Slide] = Field(min_length=1)


def page_label(source: dict) -> str:
    first, last = source.get("page_start"), source.get("page_end")
    if first is None:
        return "page unknown"
    return f"p. {first}" if last in (None, first) else f"pp. {first}-{last}"


def cited_numbers(text: str) -> list[int]:
    found = (int(left or right) for left, right in CITED.findall(text))
    return list(dict.fromkeys(found))


def surname(name: str) -> str:
    return name.split()[-1] if name.split() else name


def author_names(source: dict) -> list[str]:
    return [name.strip() for name in re.split(r",|;|\band\b", source.get("author") or "") if name.strip()]


def short_reference(source: dict) -> str:
    """Deisenroth et al. 2020: for one line at the foot of a slide. Falls back to the title."""
    names = author_names(source)
    if not names:
        who = source["doc_title"]
    elif len(names) == 1:
        who = surname(names[0])
    elif len(names) == 2:
        who = f"{surname(names[0])} and {surname(names[1])}"
    else:
        who = f"{surname(names[0])} et al."
    return f"{who} {source['year']}" if source.get("year") and names else who


def full_reference(source: dict) -> str:
    """Authors (year). Title. Publisher, pages: for a source list or a speaker's notes."""
    names = author_names(source)
    authors = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)
    head = f"{authors} ({source['year']}). " if authors and source.get("year") else (f"{authors}. " if authors else "")
    publisher = f" {source['publisher']}," if source.get("publisher") else ""
    title = source["doc_title"] if source["doc_title"].endswith(("?", "!", ".")) else f"{source['doc_title']}."
    return f"{head}{title}{publisher} {page_label(source)}"


def source_lines(text: str, sources: dict[int, dict]) -> list[str]:
    return [f"S{number}: {full_reference(sources[number])}" for number in cited_numbers(text) if number in sources]


def unverified_note(sources: dict[int, dict]) -> str:
    """The warning the chat gives with page crops. A file has no such panel."""
    places = list(dict.fromkeys(
        f"{source['doc_title'][:MAX_TITLE_CHARS]}, p. {page}"
        for source in sources.values() for page in source.get("unverified_pages", [])
    ))
    if not places:
        return ""
    return ("Unverified formulas: the passages used for this file contain formulas that could "
            f"not be verified automatically. Check any such formula on the page: {'; '.join(places)}.")


def with_sources(text: str, sources: dict[int, dict]) -> str:
    """The document, then the list its [S n] refer to, then any warning."""
    lines = source_lines(text, sources)
    parts = [text.rstrip()]
    if lines:
        parts.append("## Sources\n\n" + "\n".join(f"- {line}" for line in lines))
    if note := unverified_note(sources):
        parts.append(f"> {note}")
    return "\n\n".join(parts) + "\n"


def resolve_images(deck: Deck, saved_as: dict[str, str], figures: Mapping[str, Path] = MappingProxyType({})) -> None:
    """The app renames what a run saves, and models remember the name they wrote. An original
    figure's name is never guessed into a plot's."""
    made = set(saved_as.values())
    for slide in deck.slides:
        if slide.image in figures:
            continue
        if slide.image in saved_as:
            slide.image = saved_as[slide.image]
        elif slide.image and slide.image not in made and len(made) == 1 and OUTPUT_IMAGE.fullmatch(slide.image):
            slide.image = next(iter(made))      # only one picture exists, so that's the one


def requested_slides(question: str) -> int | None:
    """"A 5-slide deck", "5 slides": the count the reader asked for, if they gave one."""
    found = SLIDE_COUNT.search(question)
    return int(found.group(1) or found.group(2)) if found else None


def keep_slide_count(deck: Deck, count: int | None) -> None:
    """Exactly what was asked for: a title slide and a summary are no excuse for more.

    The last slide stays, since it's usually the summary, and what's taken out goes to its
    notes.
    """
    if not count or len(deck.slides) <= count:
        return
    last = deck.slides[-1]
    extra = deck.slides[count - 1:-1]
    lines = [line for slide in extra for line in [
        slide.title, *slide.bullets, *slide.left_bullets, *slide.right_bullets, slide.notes] if line]
    last.notes = "\n".join(filter(None, [last.notes, *lines]))
    deck.slides = deck.slides[:count - 1] + [last]


def drop_missing_images(deck: Deck, outputs: Path, figures: Mapping[str, Path] = MappingProxyType({})) -> list[str]:
    """A plain file name that doesn't exist loses its picture, not the deck.

    Models invent a name now and then. Anything else, such as a path, is left for
    bring_image to refuse.
    """
    missing = []
    for slide in deck.slides:
        if not slide.image:
            continue
        inline_data = slide.image.lower().startswith("data:")       # never read, so safe to drop
        bare_name = not PATH_MARK.search(slide.image)
        known = slide.image in figures or (OUTPUT_IMAGE.fullmatch(slide.image) and (outputs / slide.image).is_file())
        if inline_data or (bare_name and not known):
            missing.append("an inline image" if inline_data else slide.image)
            slide.image = None
    return missing


def bring_image(name: str, outputs: Path, work: Path, max_bytes: int,
                figures: Mapping[str, Path] = MappingProxyType({})) -> None:
    """Copy one picture into the work folder, or refuse: a plot the app made, or an original
    figure from the sources. A figure's name is the app's own, mapped to its crop here, so the
    model never names a path."""
    if name in figures:
        source = figures[name]
    else:
        source = outputs / name
        if not OUTPUT_IMAGE.fullmatch(name) or not source.is_file():
            raise ContentError(f"The image {name!r} isn't a plot this app made or an original figure "
                               "from the sources. Use a name the app gave you.")
    folder = work / "images"
    folder.mkdir(exist_ok=True)
    target = folder / name
    if not target.exists():
        safe_copy(source, target, max_bytes)


def localize_images(markdown: str, outputs: Path, work: Path, max_bytes: int,
                    figures: Mapping[str, Path] = MappingProxyType({})) -> str:
    """Point each image at its copy in the work folder. Any other path stops the build."""
    def localize(match: re.Match) -> str:
        name = match.group(2).strip("<>")
        bring_image(name, outputs, work, max_bytes, figures)
        return f"![{match.group(1)}](images/{name})"
    return MARKDOWN_IMAGE.sub(localize, markdown)


def dollar_math(text: str) -> str:
    """\\( \\) and \\[ \\] become $ and $$, outside code. Other Markdown tools read only those."""
    def convert(part: str) -> str:
        part = re.sub(r"\\\[\s*(.+?)\s*\\\]", lambda found: f"$${found.group(1)}$$", part, flags=re.S)
        return re.sub(r"\\\(\s*(.+?)\s*\\\)", lambda found: f"${found.group(1)}$", part, flags=re.S)
    pieces = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    return "".join(piece if index % 2 else convert(piece) for index, piece in enumerate(pieces))


def script(text: str, table: dict, mark: str) -> str:
    """x^{2} as a real superscript, or x^(a+b) if some character has no such form."""
    body = text.strip("{}")
    translated = body.translate(table)
    if all(ord(character) in table for character in body):
        return translated
    return f"{mark}({body})" if len(body) > 1 else f"{mark}{body}"


def plain_math(latex: str) -> str | None:
    """Simple LaTeX as Unicode, for places that can't show an equation: A^\\top becomes Aᵀ.

    Returns None if anything would be left over, rather than show broken math. Scripts go
    first, so that \\sqrt{a_{11}} has no nested braces left when its turn comes.
    """
    text = re.sub(r"\\[,;:!]", " ", latex)
    text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg|displaystyle|quad|qquad)(?![A-Za-z])", " ", text)
    text = re.sub(r"\\(?:mathbf|mathrm|mathit|boldsymbol|bm|text|mathcal|operatorname|hat|bar|tilde|vec)\s*"
                  r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbb\s*\{(\w)\}", lambda found: BLACKBOARD.get(found.group(1), found.group(1)), text)
    text = re.sub(r"\\(?:\{|\})", lambda found: found.group()[1], text)
    text = re.sub(r"\\([A-Za-z]+)", lambda found: UNICODE_MACROS.get(found.group(1), found.group()), text)
    for _ in range(4):
        text = re.sub(rf"\^{SCRIPT_ARGUMENT}", lambda found: script(found.group(1), SUPERSCRIPTS, "^"), text)
        text = re.sub(rf"_{SCRIPT_ARGUMENT}", lambda found: script(found.group(1), SUBSCRIPTS, "_"), text)
        text, fractions = re.subn(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda found: (
            f"{found.group(1)}/{found.group(2)}" if len(found.group(1)) + len(found.group(2)) <= 4
            else f"({found.group(1)})/({found.group(2)})"), text)
        text, roots = re.subn(r"\\sqrt\s*\{([^{}]*)\}", lambda found: (
            f"√{found.group(1)}" if len(found.group(1)) <= 2 else f"√({found.group(1)})"), text)
        if not fractions and not roots:
            break
    text = " ".join(text.replace("{", "").replace("}", "").split())
    return None if "\\" in text or "^" in text and "^(" not in text else text


def inline_to_unicode(text: str) -> str:
    """Each inline formula that can be written in Unicode, written in Unicode."""
    def convert(found: re.Match) -> str:
        return plain_math(found.group(1) or found.group(2)) or found.group()
    return INLINE_FORMULA.sub(convert, text)


def upright_word(word: str) -> bool:
    """ISR, had, max: a word, set upright. ij, ii or ijk: indices, left as maths.

    i is left out of the vowels because it's the commonest index; min is the word that
    rule would miss.
    """
    if re.fullmatch(r"[A-Z]{2,}[a-z]*", word) or word in INDEX_LETTER_WORDS:
        return True
    return len(word) >= 3 and word.islower() and re.search(r"[aeouy]", word) is not None


def upright_words(latex: str) -> str:
    """\\sigma_{had} becomes \\sigma_{\\text{had}}, which is how a subscript word is set."""
    return SCRIPT_WORD.sub(lambda found: f"{found.group(1)}{{\\text{{{found.group(2)}}}}}"
                           if upright_word(found.group(2)) else found.group(), latex)


def upright_words_in_math(markdown: str) -> str:
    """The same, for every formula in a document, and nowhere else."""
    pieces = re.split(r"(```.*?```|`[^`\n]*`)", markdown, flags=re.S)
    math = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$|\\\(.+?\\\)|\\\[.+?\\\]", re.S)
    return "".join(piece if index % 2 else math.sub(lambda found: upright_words(found.group()), piece)
                   for index, piece in enumerate(pieces))


def plain_typography(text: str) -> str:
    return text.translate(TYPOGRAPHY)


LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,2}[.)])\s+\S")


def lists_after_paragraphs(markdown: str) -> str:
    """A blank line before a list that follows a line of text. Models write lists straight under
    a sentence, and pandoc then reads "* item" as text: the PDF showed literal asterisks. Code
    and display maths are left alone, and so is a list's own next item."""
    out: list[str] = []
    fenced = in_math = False
    for line in markdown.split("\n"):
        stripped = line.strip()
        if FENCE.match(stripped):
            fenced = not fenced
        elif not fenced and stripped.startswith("$$"):
            if not (stripped.endswith("$$") and len(stripped) > 2):
                in_math = not in_math
        elif not fenced and not in_math and LIST_ITEM.match(line) and out:
            before = out[-1]
            if before.strip() and not LIST_ITEM.match(before) and not before.startswith((" ", "\t")):
                out.append("")
        out.append(line)
    return "\n".join(out)


def without_labels(text: str) -> str:
    """\\label does nothing in these files, and KaTeX refuses a formula that has one."""
    return re.sub(r"\\label\{[^{}]*\}[ \t]*", "", text)


def clean_bullet(text: str) -> str:
    text = re.sub(r"<sub>(.*?)</sub>", lambda found: script(found.group(1), SUBSCRIPTS, "_"), text, flags=re.I)
    text = re.sub(r"<sup>(.*?)</sup>", lambda found: script(found.group(1), SUPERSCRIPTS, "^"), text, flags=re.I)
    return inline_to_unicode(re.sub(r"</?[a-z]+\s*/?>", "", text)).replace("**", "").replace("__", "").strip()


def closing_formula(lines: list[str], start: int) -> tuple[str, int]:
    """A formula that runs over several lines. Returns it and the line after it.

    If it's never closed, it ends at the next blank line: TeX then has the last word,
    which is better than losing the deck over a missing $$.
    """
    closer = "$$" if lines[start].strip().startswith("$$") else r"\]"
    opener = "$$" if closer == "$$" else r"\["
    body = [lines[start].strip().removeprefix(opener)]
    for index in range(start + 1, len(lines)):
        line = lines[index].strip()
        if closer in line:
            before, _, after = line.partition(closer)
            body.append(before)
            return "\n".join(part for part in body if part.strip()), index + 1
        if not line:
            return "\n".join(part for part in body if part.strip()), index
        body.append(line)
    return "\n".join(part for part in body if part.strip()), len(lines)


def split_slides(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Slides start at ## headings, or at # headings if there are no ## ones."""
    level = "## " if any(line.startswith("## ") for line in lines) else "# "
    slides: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith(level):
            slides.append((line[len(level):].strip(), []))
        elif slides:
            slides[-1][1].append(line)
    return slides


def bullet_text(line: str, slide: Slide) -> str:
    """One bullet in plain text. Math too complex for Unicode becomes one of the slide's equations."""
    bullet = BULLET.match(line)
    text = clean_bullet(bullet.group(1) if bullet else line)

    def lift(found: re.Match) -> str:
        slide.equations.append((found.group(1) or found.group(2)).strip())
        return ""
    text = " ".join(INLINE_FORMULA.sub(lift, text).split()).rstrip(":,; ")
    if RAW_MATH.search(text.replace(r"\$", "")):
        text = plain_math(text) or text
    if RAW_MATH.search(text.replace(r"\$", "")):
        raise ContentError(f"The slide {slide.title!r} has LaTeX in a bullet that I can't read: {text!r}. "
                           "Put it on its own line between $$.")
    return text


def add_bullet(line: str, slide: Slide, column: list[str] | None) -> None:
    text = bullet_text(line, slide)
    if CITATION.sub("", text).strip(" .,:;-*"):
        (slide.bullets if column is None else column).append(text)
    elif text:                                  # only a citation was left
        slide.notes = f"{slide.notes} {text}".strip()


def lift_formulas(line: str, slide: Slide) -> str:
    """Display formulas become the slide's equations. Returns the rest of the line, if any.

    A citation left on its own still counts: the notes keep it for the footer.
    """
    def lift(found: re.Match) -> str:
        slide.equations.append((found.group(1) or found.group(2)).strip())
        return " "
    rest = FORMULA_IN_LINE.sub(lift, line).strip()
    if not CITATION.sub("", rest).strip(" .,:;-*"):
        slide.notes = f"{slide.notes} {rest}".strip(" .,:;-*")
        return ""
    return rest.rstrip(":, ")


def parse_slide(heading: str, lines: list[str]) -> Slide:
    """One slide of the outline. The layout is the word before the colon in its heading."""
    prefix = LAYOUT_PREFIX.match(heading)
    layout = prefix.group(1) if prefix and prefix.group(1) in theme.LAYOUTS else None
    slide = Slide(title=clean_bullet(prefix.group(2) if layout else heading) or "Untitled", layout=layout)
    column: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        image = IMAGE_LINE.match(line)
        notes = NOTES.match(line)
        field = FIELD.match(line)
        if not line or RULE.match(line) or FENCE.match(line):
            continue
        if FORMULA_IN_LINE.search(line):
            line = lift_formulas(line, slide)
            if not line:
                continue
            add_bullet(line, slide, column)
        elif line.startswith(("$$", r"\[")):
            formula, index = closing_formula(lines, index - 1)
            slide.equations.append(formula.strip())
        elif image:
            slide.image = image.group(1)
            slide.caption = slide.caption or clean_bullet(re.match(r"!\[([^\]]*)\]", line).group(1))
        elif notes:
            note = " ".join([notes.group(1)] + [rest.strip() for rest in lines[index:]])
            slide.notes = f"{slide.notes} {inline_to_unicode(note)}".strip()
            break
        elif field:
            name, value = field.group(1).lower(), clean_bullet(field.group(2))
            if name == "animate":
                slide.animate = value.lower() in YES
            elif name in ("left", "right"):
                setattr(slide, f"{name}_heading", value)
                column = slide.left_bullets if name == "left" else slide.right_bullets
            else:
                setattr(slide, name, value)
        else:
            add_bullet(line, slide, column)
    return slide


def inferred_layout(slide: Slide, first: bool) -> str:
    """A layout for a slide that didn't name one, from what it holds."""
    if slide.image:
        return "figure_text"
    if slide.left_bullets or slide.right_bullets:
        return "two_columns"
    if first and not slide.bullets and not slide.equations:
        return "title"
    if slide.equations and not slide.bullets:
        return "big_equation"
    if SUMMARY_WORDS.search(slide.title) and not slide.equations:
        return "summary"
    return "bullets_equation"


def parse_deck(text: str) -> Deck:
    """The outline the model writes. Markdown, not JSON, so LaTeX needs no escaping."""
    lines = [line for line in text.splitlines() if not FENCE.match(line.strip())]
    slides = [parse_slide(title, body) for title, body in split_slides(lines)]
    if not slides:
        raise ContentError("I found no slides. Start each slide with a line like: ## bullets_equation: Slide title")
    for number, slide in enumerate(slides):
        slide.layout = slide.layout or inferred_layout(slide, number == 0)
    return Deck(slides=slides)


def shorten(text: str, limit: int) -> str:
    """At most `limit` words, cut at the end of a clause if there is one."""
    words = text.split()
    if len(words) <= limit:
        return text
    kept = " ".join(words[:limit])
    if kept.count("(") > kept.count(")"):       # never stop inside a bracket
        before = kept[:kept.rindex("(")].rstrip(",;:\u2013\u2014 ")
        if len(before.split()) >= MIN_CLAUSE_WORDS:
            return before
    kept_words = kept.split()
    for end in range(len(kept_words), MIN_CLAUSE_WORDS - 1, -1):
        if kept_words[end - 1].endswith(CLAUSE_END):
            return " ".join(kept_words[:end]).rstrip(",;:\u2013\u2014 ")
    return kept.rstrip(",;:\u2013\u2014 ") + "\u2026"


def word_count(slide: Slide) -> int:
    """Every word the audience sees, footer included."""
    parts = [slide.title, slide.subtitle, slide.caption, slide.left_heading, slide.right_heading, slide.footer,
             *slide.bullets, *slide.left_bullets, *slide.right_bullets]
    return sum(len(part.split()) for part in parts)


def strip_citations(slide: Slide) -> None:
    """Take [S n] out of everything visible, and remember them for the footer."""
    texts = [slide.title, slide.subtitle, slide.caption, *slide.bullets, *slide.left_bullets,
             *slide.right_bullets, *slide.equations, slide.notes]
    slide.cited = cited_numbers("\n".join(texts))
    clean = lambda text: CITATION.sub("", text).strip()     # noqa: E731
    slide.title, slide.subtitle, slide.caption = clean(slide.title), clean(slide.subtitle), clean(slide.caption)
    slide.bullets = [clean(text) for text in slide.bullets]
    slide.left_bullets = [clean(text) for text in slide.left_bullets]
    slide.right_bullets = [clean(text) for text in slide.right_bullets]
    slide.notes = clean(slide.notes)
    slide.equations = [TAGGED_SOURCE.sub("", formula) for formula in slide.equations]


def fit_layout(slide: Slide, moved: list[str]) -> None:
    """Make the content match the layout. What doesn't belong goes to `moved`, for the notes."""
    def to_notes(equations: list[str]) -> None:
        moved.extend(plain_math(formula) or formula for formula in equations)

    if slide.layout == "figure_text" and not slide.image:
        slide.layout = "bullets_equation"
    if slide.layout == "two_columns" and not (slide.left_bullets or slide.right_bullets):
        middle = (len(slide.bullets) + 1) // 2
        slide.left_bullets, slide.right_bullets, slide.bullets = slide.bullets[:middle], slide.bullets[middle:], []
    if slide.layout == "bullets_equation" and len(slide.equations) > theme.MAX_EQUATIONS and not slide.bullets:
        slide.layout = "big_equation"
    if slide.layout == "title":
        if slide.bullets and not slide.subtitle:
            slide.subtitle = slide.bullets.pop(0)
        moved.extend(slide.bullets)
        to_notes(slide.equations)
        slide.bullets, slide.equations = [], []
    elif slide.layout == "big_equation":
        if len(slide.bullets) == 1 and not slide.caption:
            slide.caption = slide.bullets.pop()
        moved.extend(slide.bullets)
        slide.bullets = []
        to_notes(slide.equations[theme.MAX_DERIVATION_LINES:])
        slide.equations = slide.equations[:theme.MAX_DERIVATION_LINES]
    elif slide.layout in ("two_columns", "summary"):
        to_notes(slide.equations)
        slide.equations = []
    else:
        limit = 1 if slide.layout == "figure_text" else theme.MAX_EQUATIONS
        to_notes(slide.equations[limit:])
        slide.equations = slide.equations[:limit]
    if slide.layout != "figure_text":
        slide.image = None


def apply_limits(slide: Slide, moved: list[str]) -> list[str]:
    """The hard limits. Nothing fails or is lost: what's cut goes to `moved`, in full."""
    trimmed: list[str] = []

    def short(text: str) -> str:
        cut = shorten(text, theme.MAX_BULLET_WORDS)
        if cut != text:
            moved.append(text)
            trimmed.append(cut)
        return cut

    for name in ("title", "caption", "subtitle"):       # the full text goes to the notes as well
        text = getattr(slide, name)
        cut = shorten(text, theme.MAX_BULLET_WORDS)
        if cut != text:
            moved.append(text)
            setattr(slide, name, cut)
    slide.left_heading = shorten(slide.left_heading, 4)
    slide.right_heading = shorten(slide.right_heading, 4)
    if slide.layout == "two_columns":
        while len(slide.left_bullets) + len(slide.right_bullets) > theme.MAX_BULLETS:
            longer = slide.left_bullets if len(slide.left_bullets) >= len(slide.right_bullets) else slide.right_bullets
            moved.append(longer.pop())
        slide.left_bullets = [short(text) for text in slide.left_bullets]
        slide.right_bullets = [short(text) for text in slide.right_bullets]
    else:
        moved.extend(slide.bullets[theme.MAX_BULLETS:])
        slide.bullets = [short(text) for text in slide.bullets[:theme.MAX_BULLETS]]
    while word_count(slide) > theme.MAX_SLIDE_WORDS:
        column = max((slide.bullets, slide.left_bullets, slide.right_bullets), key=len)
        if not column:
            break
        moved.append(column.pop())
    return trimmed


def footer_line(cited: list[int], sources: dict[int, dict]) -> str:
    """One line for the slide: each work and its pages, once. The notes have the full reference."""
    places: dict[str, list[str]] = {}
    for number in cited:
        if number in sources:
            label = page_label(sources[number])
            work = places.setdefault(short_reference(sources[number]), [])
            if label not in work:
                work.append(label)
    parts = [f"{work}, {', '.join(labels)}" for work, labels in places.items()]
    if not parts:
        return ""
    extra = f"; and {len(parts) - MAX_FOOTER_SOURCES} more" if len(parts) > MAX_FOOTER_SOURCES else ""
    return "Sources: " + "; ".join(parts[:MAX_FOOTER_SOURCES]) + extra


def finish_deck(deck: Deck, sources: dict[int, dict]) -> tuple[Deck, list[str]]:
    """Fit each slide to its layout and limits, and move citations to footers and notes.

    Returns the deck and the bullets that were shortened, for the model to hear about.
    The slide count never changes, so a requested number of slides stays right.
    """
    trimmed: list[str] = []
    for slide in deck.slides:
        strip_citations(slide)
        slide.equations = [upright_words(formula) for formula in slide.equations]
        moved: list[str] = []
        fit_layout(slide, moved)
        slide.footer = footer_line(slide.cited, sources)
        trimmed += apply_limits(slide, moved)
        references = [f"S{number}: {full_reference(sources[number])}" for number in slide.cited if number in sources]
        parts = [slide.notes]
        if moved:
            parts.append("More detail:\n" + "\n".join(moved))
        if references:
            parts.append("References:\n" + "\n".join(references))
        slide.notes = "\n\n".join(part for part in parts if part)
    if note := unverified_note(sources):
        last = deck.slides[-1]
        last.notes = f"{last.notes}\n\n{note}".strip()
    return deck, trimmed
