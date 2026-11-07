"""Builds files from text the model wrote. Runs only inside the sandbox.

    python /opt/sandbox/build.py pdf|docx|deck

Works in /work, reads doc.md or deck.json, writes out/out.<kind>, and prints one line of
JSON for the app. Everything it reads is untrusted, which is why it runs in a container
with no network and only /work mounted. `--sandbox` adds a second wall around pandoc's
own reading, but it doesn't reach TeX: `\\input` in a formula goes straight through, so
for PDFs the container is the protection.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

from common import OUT, WORK, finish, missing_characters, run, tectonic, tex_errors
from deck import build_deck
import theme

HERE = Path(__file__).resolve().parent
REFERENCE_DOC = HERE / "reference.docx"

# Off: every way of getting raw TeX or HTML past the writer, and metadata, which goes
# straight into the template.
MARKDOWN = ("markdown-raw_tex-raw_html-raw_attribute-yaml_metadata_block-pandoc_title_block"
            "-citations+tex_math_single_backslash")
# Fonts by file name, from TeX's own bundle, so they're baked into the cache like any package.
OTF_FAMILY = "Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic"
TTF_FAMILY = "Extension=.ttf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic"
TTF_MONO = "Extension=.ttf,UprightFont=*,BoldFont=*-Bold,ItalicFont=*-Oblique,BoldItalicFont=*-BoldOblique"
PDF_OPTIONS = ["-V", "mainfont=LibertinusSerif", "-V", f"mainfontoptions={OTF_FAMILY}",
               "-V", "sansfont=Carlito", "-V", f"sansfontoptions={TTF_FAMILY}",
               "-V", "monofont=DejaVuSansMono", "-V", f"monofontoptions={TTF_MONO}",
               "-V", "mathfont=LibertinusMath-Regular.otf",
               "-V", "geometry:margin=2.3cm", "-V", "fontsize=11pt", "-V", "colorlinks=true",
               "--lua-filter", str(HERE / "table.lua")]
# The Word template's design, for TeX: the same colours, heading font, rules and page numbers.
PDF_THEME = rf"""
\usepackage{{mathtools}}
\renewcommand{{\boldsymbol}}[1]{{\symbfit{{#1}}}}
\providecommand{{\bm}}[1]{{\symbfit{{#1}}}}
\usepackage{{fvextra}}
\DefineVerbatimEnvironment{{Highlighting}}{{Verbatim}}{{breaklines,commandchars=\\\{{\}}}}
\definecolor{{accent}}{{HTML}}{{{theme.ACCENT}}}
\definecolor{{accentdark}}{{HTML}}{{{theme.ACCENT_DARK}}}
\definecolor{{rulegrey}}{{HTML}}{{{theme.RULE}}}
\definecolor{{muted}}{{HTML}}{{{theme.MUTED}}}
\AtBeginDocument{{\hypersetup{{linkcolor=accent,urlcolor=accent,citecolor=accent}}}}
\usepackage{{titlesec}}
\titleformat{{\section}}{{\sffamily\LARGE\bfseries\color{{accent}}}}{{\thesection}}{{0.6em}}{{}}
\titleformat{{\subsection}}{{\sffamily\Large\bfseries\color{{accent}}}}{{\thesubsection}}{{0.6em}}{{}}
\titleformat{{\subsubsection}}{{\sffamily\large\bfseries\color{{accent}}}}{{\thesubsubsection}}{{0.6em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{16pt}}{{5pt}}
\titlespacing*{{\subsection}}{{0pt}}{{12pt}}{{4pt}}
\usepackage{{longtable,booktabs,array}}
\usepackage{{colortbl}}
\arrayrulecolor{{accent}}
\renewcommand{{\arraystretch}}{{1.25}}
\AtBeginDocument{{\let\plaintoprule\toprule\renewcommand{{\toprule}}{{\plaintoprule\rowcolor{{accent}}}}}}
\setlength{{\belowrulesep}}{{0pt}}
\setlength{{\aboverulesep}}{{0pt}}
\setlength{{\LTleft}}{{\tabcolsep}}
\setlength{{\LTright}}{{\fill}}
\usepackage{{fancyhdr}}
\pagestyle{{fancy}}
\fancyhf{{}}
\renewcommand{{\headrulewidth}}{{0pt}}
\fancyfoot[R]{{\small\color{{muted}}\thepage}}
\usepackage{{framed}}
% A figure's caption keeps the source's own label ("Figure 5: ..."), so LaTeX adds no number of
% its own: article's \@makecaption with "#1: #2" as "#2". Tables are longtables, with their own.
\makeatletter
\long\def\@makecaption#1#2{{\vskip\abovecaptionskip\sbox\@tempboxa{{#2}}\ifdim\wd\@tempboxa>\hsize #2\par\else\global\@minipagefalse\hb@xt@\hsize{{\hfil\box\@tempboxa\hfil}}\fi\vskip\belowcaptionskip}}
\makeatother
\renewenvironment{{quote}}{{\color{{muted}}\def\FrameCommand{{{{\color{{accent}}\vrule width 2pt}}\hspace{{10pt}}}}\MakeFramed{{\advance\hsize-\width\FrameRestore}}}}{{\endMakeFramed}}
"""
IMAGE = re.compile(r"(!\[[^\]]*\]\()(images/[A-Za-z0-9._-]+)(\))")
def pandoc(*args: str) -> subprocess.CompletedProcess:
    return run(["pandoc", "--sandbox", "-f", MARKDOWN, *args])


def build_pdf() -> None:
    (WORK / "theme.tex").write_text(PDF_THEME, encoding="utf-8")
    done = pandoc("-t", "latex", "--standalone", *PDF_OPTIONS, "-H", "theme.tex", "-o", "doc.tex", "doc.md")
    if done.returncode != 0:
        finish(ok=False, error=f"pandoc: {done.stderr.strip()[-400:]}")
    done, log = tectonic("doc.tex")
    if done.returncode != 0 or not (WORK / "doc.pdf").is_file():
        finish(ok=False, error=f"TeX: {tex_errors(done, log)}")
    os.replace(WORK / "doc.pdf", OUT / "out.pdf")
    info = run(["pdfinfo", str(OUT / "out.pdf")]).stdout
    pages = re.search(r"^Pages:\s+(\d+)", info, re.M)
    finish(ok=True, file="out.pdf", pages=int(pages.group(1)) if pages else None,
           missing_characters=missing_characters(log))


def inline_images(markdown: str) -> str:
    """`--sandbox` reads only the named input, so pictures travel inside it as data URIs."""
    def inline(match: re.Match) -> str:
        path = WORK / match.group(2)
        if not path.is_file() or path.is_symlink():
            return match.group()        # pandoc then refuses it
        kind = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f"{match.group(1)}data:image/{kind};base64,{data}{match.group(3)}"
    return IMAGE.sub(inline, markdown)


def build_docx() -> None:
    (WORK / "inlined.md").write_text(inline_images((WORK / "doc.md").read_text()))
    # A warning here is a picture left out or a formula pandoc couldn't convert, so a
    # finished file would be missing something: better to fail and have it fixed.
    done = pandoc("-t", "docx", "--fail-if-warnings", f"--reference-doc={REFERENCE_DOC}",
                  "-o", str(OUT / "out.docx"), "inlined.md")
    if done.returncode != 0 or not (OUT / "out.docx").is_file():
        finish(ok=False, error=f"pandoc: {done.stderr.strip()[-600:]}")
    keep_rows_whole(OUT / "out.docx")
    finish(ok=True, file="out.docx")


def keep_rows_whole(path: Path) -> None:
    """No table row splits across a page break. pandoc has no option for it."""
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = docx.Document(path)
    for row in document.element.body.iter(qn("w:tr")):
        properties = row.find(qn("w:trPr"))
        if properties is None:
            properties = OxmlElement("w:trPr")
            row.insert(0 if row.find(qn("w:tblPrEx")) is None else 1, properties)
        if properties.find(qn("w:cantSplit")) is None:
            properties.insert(0, OxmlElement("w:cantSplit"))
    document.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    {"pdf": build_pdf, "docx": build_docx, "deck": build_deck}[sys.argv[1]]()


if __name__ == "__main__":
    main()
