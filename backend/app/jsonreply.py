r"""JSON from a model's reply, where its LaTeX may have broken the escapes.

Models write LaTeX inside JSON strings, and often with one backslash: "\\phi" as `\\phi` is
fine, but `\\phi` written as `\phi` is an invalid escape, and `\boldsymbol` turns into a
backspace and "oldsymbol". So every lone backslash that starts a LaTeX command, or a LaTeX
symbol that is no JSON escape (`\,` `\%` `\{` `\ `), is doubled before parsing (Balanced's
Explain replies, Nov 6: 5 of 10 unreadable). After `\b` or `\f` letters are always LaTeX; a
command that begins with `\n`, `\r` or `\t` (`\theta`, `\nabla`) counts only if KaTeX knows
it, so that a real `\n` before a word stays a newline. The reply as given is the fallback:
valid JSON with `\frac` in it would parse, but as a form feed and "rac".
"""
from __future__ import annotations

import json
import re

OBJECT = re.compile(r"\{.*\}", re.S)
ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|[A-Za-z]+|.)", re.S)
# `\b` and `\f` (backspace, form feed) are never what a model means here, so a letter after
# them is always LaTeX: `\begin`, `\bigoplus`, `\frac` (review R2-C6, Nov 2026: `\begin` and
# `\bigoplus` became control characters). After `\n` `\r` `\t` it is LaTeX only for a command
# KaTeX renders, so that `\nabla` is LaTeX and `\nprint` a newline and then "print". The list
# is KaTeX 0.16.47's commands that start with n, r or t, without the escapes `\r` and `\t`.
ESCAPE_LIKE = set("""
    nLeftarrow nLeftrightarrow nRightarrow nVDash nVdash nabla natnums natural ncong ne nearrow neg
    negmedspace negthickspace negthinspace neq newcommand newline nexists ngeq ngeqq ngeqslant ngtr ni
    nleftarrow nleftrightarrow nleq nleqq nleqslant nless nmid nobreak nobreakspace noexpand nolimits
    nonumber normalsize not notag notin notni nparallel nprec npreceq nrightarrow nshortmid
    nshortparallel nsim nsubseteq nsubseteqq nsucc nsucceq nsupseteq nsupseteqq ntriangleleft
    ntrianglelefteq ntriangleright ntrianglerighteq nu nvDash nvdash nwarrow rArr rBrace rVert raisebox
    rang rangle rarr ratio rbrace rbrack rceil real reals red redA redB redC redD redE relax
    renewcommand restriction rfloor rgroup rhd rho right rightarrow rightarrowtail rightharpoondown
    rightharpoonup rightleftarrows rightleftharpoons rightrightarrows rightsquigarrow rightthreetimes
    risingdotseq rlap rm rmoustache rparen rq rrbracket rtimes rule rvert tag tan tanh tau tbinom tealA
    tealB tealC tealD tealE text textasciicircum textasciitilde textbackslash textbar textbardbl textbf
    textbraceleft textbraceright textcircled textcolor textcopyright textdagger textdaggerdbl textdegree
    textdollar textellipsis textemdash textendash textgreater textit textless textmd textnormal
    textquotedblleft textquotedblright textquoteleft textquoteright textregistered textrm textsf
    textsterling textstyle texttt textunderscore textup tfrac tg th therefore theta thetasym thickapprox
    thicksim thickspace thinspace tilde times tiny tmspace to top triangle triangledown triangleleft
    trianglelefteq triangleq triangleright trianglerighteq tt twoheadleftarrow twoheadrightarrow
""".split())


def repaired(text: str) -> str:
    """Backslashes taken left to right, so that an escaped backslash pair is never split: a
    valid JSON escape is kept; a LaTeX command or symbol gets its backslash doubled."""
    def fix(match: re.Match) -> str:
        after = match.group(1)
        if after in ("\\", '"', "/") or re.fullmatch(r"u[0-9A-Fa-f]{4}", after):
            return match.group(0)
        if after[0] in "nrt" and after not in ESCAPE_LIKE:
            return match.group(0)
        return "\\\\" + after
    return ESCAPE.sub(fix, text)


def loads(reply: str):
    """The JSON object in a reply (a fence or words around it are tolerated), or None."""
    match = OBJECT.search(reply)
    if not match:
        return None
    text = match.group(0)
    for attempt in dict.fromkeys((repaired(text), text)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None
