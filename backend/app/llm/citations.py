"""[S n] citations: keep the real ones and remove invented ones as the text streams."""
from __future__ import annotations

import re

# gpt-oss writes [*S6*], 【S1】, (S1, S3), See S1 and (S3, p. 197, ...) as well as [S6],
# and once [S1] with a zero-width space. It also labels formulas with \tag{S1}, \tag{S1, S5}
# or \tag{S1:4.106}. Underscores stay out of the gap, and the round forms take plain digits
# only, so math like E[S_1] and f(S_1, S_2) is never a citation.
_GAP = "[\\s*\u200b]*"
_BOX_OPEN, _BOX_CLOSE = "[\\[【]", "[\\]】]"
_LIST_JOIN = r"(?:,\s*|\s+and\s+)"
_DETAIL = r"p{1,2}\.|[Ee]q|[Ss]ec|[Ff]ig|[Cc]h"     # (S3, p. 197, ...): a source and where in it
_CITATION = re.compile(
    rf"{_BOX_OPEN}{_GAP}S{_GAP}\d+{_GAP}(?:[,;]{_GAP}S?{_GAP}\d+{_GAP})*{_BOX_CLOSE}"
    r"|\(S\d+(?:,\s*S\d+)*\)"
)
_SEE = re.compile(rf"([Ss]ee\s+)(S\d+(?:{_LIST_JOIN}S\d+)*)")
_LOCATED = re.compile(rf"(\()(S\d+)(?=,\s*(?:{_DETAIL}))")
_UNFINISHED = re.compile(
    rf"{_BOX_OPEN}{_GAP}(?:S{_GAP}(?:\d+{_GAP}(?:[,;]{_GAP}S?{_GAP}\d*{_GAP})*)?)?$"
    r"|\((?:S(?:\d+(?:,\s*(?:S\d*|[A-Za-z.]{0,3})?)*)?)?$"
)
# A prefix of "See S1, S2 and S3". Each repetition starts with "," or "and", so there is one
# way to read the text: a group whose parts were all optional let a run of spaces or digits
# split many ways, and a near miss tried them all (review R2-C1, Nov 2026).
_UNFINISHED_SEE = re.compile(
    rf"[Ss](?:e(?:e(?:\s+(?:S(?:\d+(?:{_LIST_JOIN}S\d+)*"
    r"(?:,\s*(?:S\d*)?|\s+(?:a(?:n(?:d(?:\s+(?:S\d*)?)?)?)?)?)?)?)?)?)?)?$"
)
_TAG = re.compile(r"\\tag\*?\{\s*(S\s*\d+(?:\s*[,;]\s*S\s*\d+)*)\s*(?::\s*([^{}]*?))?\s*\}")
_UNFINISHED_TAG = re.compile(r"\\(?:t(?:a(?:g(?:\*?(?:\{\s*(?:S[^{}]{0,80})?)?)?)?)?)?")
_OPENER = re.compile(r"[\[【(]|(?<!\w)[Ss](?=ee\b|e\Z|\Z)|\\(?=tag|ta?\Z|\Z)")
_NUMBER = re.compile(r"\d+")


class _HeldBack:
    """Streamed text where a citation may still be arriving: what is held, and the spaces before
    it, which go too if the citation is dropped. Shared by both filters (review R2-X3)."""

    def __init__(self) -> None:
        self.cited: list[int] = []
        self.dropped: list[int] = []
        self._held = ""
        self._spaces = ""

    def feed(self, text: str, final: bool = False) -> str:
        raise NotImplementedError

    def flush(self) -> str:
        """The end of the answer: decide what was held back, then release the rest."""
        rest = self.feed("", final=True) + self._spaces
        self._spaces = ""
        return rest

    def _clear(self, text: str) -> str:
        """Pass text on, keeping its trailing spaces in case a dropped citation follows."""
        text = self._spaces + text
        body = text.rstrip(" ")
        self._spaces = text[len(body):]
        return body


class CitationFilter(_HeldBack):
    """Feed it the model's text in pieces; it returns text that is safe to display.

    A citation can arrive split across tokens ("[S", "1", "2]"), so text that could
    still become one is held back until it can be decided. Ids that were never
    provided are removed before anything is shown, along with the space before them.
    """

    def __init__(self, provided: set[int]) -> None:
        super().__init__()
        self.provided = provided

    def feed(self, text: str, final: bool = False) -> str:
        self._held += text
        out = ""
        while self._held:
            opener = _OPENER.search(self._held)
            if opener is None:
                out += self._clear(self._held)
                self._held = ""
                break
            out += self._clear(self._held[:opener.start()])
            self._held = self._held[opener.start():]
            if not final and _UNFINISHED_SEE.fullmatch(self._held):
                break
            tag = _TAG.match(self._held)
            if tag:
                out += self._cite_tag(tag)
                self._held = self._held[tag.end():]
                continue
            if not final and _UNFINISHED_TAG.fullmatch(self._held):
                break
            prefixed = _SEE.match(self._held) or _LOCATED.match(self._held)
            found = _CITATION.match(self._held)
            if prefixed:
                out += self._cite(prefixed.group(2), prefixed.group(1))
                self._held = self._held[prefixed.end():]
            elif found:
                out += self._cite(found.group())
                self._held = self._held[found.end():]
            elif not final and _UNFINISHED.fullmatch(self._held):
                break
            else:                           # an ordinary bracket or word
                out += self._clear(self._held[0])
                self._held = self._held[1:]
        return out

    def _keep(self, citation: str) -> list[int]:
        """Note each number as cited or dropped, and return the real ones."""
        kept = []
        for number in map(int, _NUMBER.findall(citation)):
            if number in self.provided:
                kept.append(number)
                if number not in self.cited:
                    self.cited.append(number)
            elif number not in self.dropped:
                self.dropped.append(number)
        return kept

    def _cite(self, citation: str, lead: str = "") -> str:
        kept = "".join(f"[S{number}]" for number in self._keep(citation))
        if not kept:
            self._spaces = ""
            return ""
        return self._clear(lead + kept)

    def _cite_tag(self, tag: re.Match) -> str:
        """The tag stays in the formula, in one form, for the frontend to turn into a link."""
        kept = ", ".join(f"S{number}" for number in self._keep(tag.group(1)))
        detail = tag.group(2)
        if kept:
            return self._clear(f"\\tag{{{kept}: {detail}}}" if detail else f"\\tag{{{kept}}}")
        return self._clear(f"\\tag{{{detail}}}") if detail else ""


def strip_citations(text: str) -> str:
    """Remove every [S n] and [Web n], for old answers whose numbering no longer applies."""
    text = _TAG.sub(lambda tag: f"\\tag{{{tag.group(2)}}}" if tag.group(2) else "", text)
    return strip_web_citations(re.sub(r" ?" + _CITATION.pattern, "", text))


# --- [Web n]: web pages fetched during this answer (Phase 5d) -------------------------------------

_WEB_CITATION = re.compile(r"[\[【(]\s*Web\s*\d+(?:\s*[,;]\s*(?:Web\s*)?\d+)*\s*[\]】)]")
# A prefix of "[Web 1, Web 2; 3]", read one way only, as _UNFINISHED_SEE: the old pattern took
# 36 s on "(Web 1, ..., Web 6 and" and froze the server (review R2-C1).
_WEB_UNFINISHED = re.compile(
    r"[\[【(]\s*(?:W(?:e(?:b(?:\s*(?:\d+(?:\s*[,;]\s*(?:Web\s*)?\d+)*"
    r"(?:\s*(?:[,;]\s*(?:W(?:e(?:b\s*)?)?)?)?)?)?)?)?)?)?$"
)
_WEB_STRIP = re.compile(r" ?" + _WEB_CITATION.pattern)


class WebCitationFilter(_HeldBack):
    """As CitationFilter, for [Web n]: a number that matches no page fetched in this answer is
    removed. `pages` is asked each time, because pages are fetched while the answer runs."""

    def __init__(self, pages=lambda: set()) -> None:
        super().__init__()
        self.pages = pages

    def feed(self, text: str, final: bool = False) -> str:
        self._held += text
        out = ""
        while self._held:
            start = min((i for i in (self._held.find("["), self._held.find("【"), self._held.find("(")) if i >= 0),
                        default=-1)
            if start < 0:
                out += self._clear(self._held)
                self._held = ""
                break
            out += self._clear(self._held[:start])
            self._held = self._held[start:]
            found = _WEB_CITATION.match(self._held)
            if found:
                kept = self._cite(found.group())
                out += self._clear(kept) if kept else ""
                if not kept:
                    self._spaces = ""       # a dropped citation takes the space before it along
                self._held = self._held[found.end():]
            elif not final and _WEB_UNFINISHED.fullmatch(self._held):
                break
            else:
                out += self._clear(self._held[0])
                self._held = self._held[1:]
        return out

    def _cite(self, citation: str) -> str:
        known = self.pages()
        kept = []
        for number in map(int, _NUMBER.findall(citation)):
            if number in known:
                kept.append(number)
                if number not in self.cited:
                    self.cited.append(number)
            elif number not in self.dropped:
                self.dropped.append(number)
        return "".join(f"[Web {number}]" for number in kept)


class Citations:
    """[S n] and [Web n] together: the text goes through the library filter, then the web one."""

    def __init__(self, provided: set[int], web_pages=lambda: set()) -> None:
        self.library = CitationFilter(provided)
        self.web = WebCitationFilter(web_pages)

    @property
    def provided(self) -> set[int]:
        return self.library.provided

    @property
    def cited(self) -> list[int]:
        return self.library.cited

    @property
    def dropped(self) -> list[int]:
        return self.library.dropped

    def feed(self, text: str) -> str:
        return self.web.feed(self.library.feed(text))

    def flush(self) -> str:
        return self.web.feed(self.library.flush()) + self.web.flush()


def strip_web_citations(text: str) -> str:
    return _WEB_STRIP.sub("", text)
