"""PowerPoint decks: a real theme, six named layouts, equations, and click-by-click builds.

The design is here and in theme.py, not in what the model writes. It picks a layout and
fills its fields, and the app has already enforced the limits by the time this runs
(documents.py on the host), so everything here can assume a slide fits its layout.

The theme is built into the file: its colours and fonts are the deck's theme colours and
fonts, and the layouts are named master layouts. So a slide edited in PowerPoint can be
reset to its layout, and changing the theme there changes the whole deck.

python-pptx has no API for themes, animations or transitions, so those are written as
XML. PowerPoint is strict about this XML, so the shapes of it stay close to what
PowerPoint itself writes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import metrics
import theme
from common import OUT, WORK, finish, missing_characters, run, tectonic, tex_errors, LINE_ERROR

WIDTH, HEIGHT = 13.333, 7.5                 # inches, 16:9
MARGIN = 0.7
CONTENT = WIDTH - 2 * MARGIN
TITLE_TOP, TITLE_HEIGHT = 0.4, 0.95
RULE_TOP = 1.38
BODY_TOP, BODY_BOTTOM = 1.7, 6.62           # between the title's rule and the footer
FOOTER_TOP, FOOTER_HEIGHT = 6.82, 0.36
NUMBER_WIDTH = 0.7
PANEL_PADDING = 0.22
GAP = 0.2
COLUMN_GAP = 0.45
COLUMN = (CONTENT - COLUMN_GAP) / 2
FIGURE_TEXT_WIDTH = 5.3
CARD_PADDING = 0.3
CARD_HEADING = 0.6

TITLE_SIZES = (32, 28, 24)
COVER_SIZES = (48, 44, 40, 36)
SUBTITLE_SIZE = 24
BULLET_SIZES = (28, 26, 24, 22, 20)
COLUMN_HEADING_SIZE = 24
EQUATION_BULLET_SIZES = (24, 22, 20)
BULLET_GAP = 0.14
BULLET_INDENT = 0.3                         # inches taken by the bullet itself
FOOTER_SIZE = 11
CAPTION_SIZE = 16
NOTES_SIZE = 12
SUMMARY_PADDING = 0.35

EQUATION_DPI = 300                          # set at 14 pt
BIG_SCALE = 2.6                             # about 36 pt: big_equation's whole point
PANEL_SCALE = 2.4                           # about 34 pt beside bullets
SMALL_SCALE = 0.8                           # below this it's reported, though still built
ROW_PADDING = 10                            # pixels around a cropped equation

WIDE_FIGURE = 0.62                          # height / width, below which a figure takes the full width
WIDE_EQUATION_ROOM = 1.2                    # inches kept under a wide figure's text for its equations
CAPTION_SAFETY = 0.97                       # of the width: wrap a little early, not a little late
SUBSCRIPT, SUPERSCRIPT = -25000, 30000      # PowerPoint's baselines, in thousandths of a percent
EMU = 914400
TRANSITION = '<p:transition xmlns:p="{p}" spd="med"><p:fade/></p:transition>'
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"

EQUATION_PAGE = r"""\documentclass[14pt,fleqn]{extarticle}
\usepackage[paperwidth=60cm,paperheight=30cm,margin=1cm]{geometry}
\usepackage{amsmath,mathtools}
\usepackage{unicode-math}
\setmainfont{LibertinusSerif}[Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic]
\setsansfont{LibertinusSans}[Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-Italic]
\setmonofont{DejaVuSansMono}[Extension=.ttf,BoldFont=*-Bold,ItalicFont=*-Oblique]
\setmathfont{LibertinusMath-Regular.otf}
\renewcommand{\boldsymbol}[1]{\symbfit{#1}}
\providecommand{\bm}[1]{\symbfit{#1}}
\pagestyle{empty}
\begin{document}
"""
DISPLAY_ENVIRONMENT = re.compile(r"\s*\\begin\{(align|alignat|gather|multline|equation|flalign)\*?\}")
ROWS = re.compile(r"^\s*\\begin\{(aligned|align\*?)\}(.*)\\end\{\1\}\s*$", re.S)
OPEN_OR_CLOSE = re.compile(r"\\begin\{|\\end\{|\\\\|&|\{|\}")
# M_Z or Γ_Z, which Unicode has no subscript for: one letter, so run_python stays as it is.
SCRIPTED = re.compile(r"(?<!\w)([^\W\d_])([_^])(?:\(([^()]{1,16})\)|([^\W_]{1,3})(?!\w))")


def split_top_level(body: str, separator: str) -> list[str]:
    """Split at `\\\\` or `&`, but not inside braces or a nested matrix."""
    parts, start, depth, position = [], 0, 0, 0
    while match := OPEN_OR_CLOSE.search(body, position):
        token = match.group()
        position = match.end()
        if token == "\\begin{":
            depth += 1
        elif token == "\\end{":
            depth -= 1
        elif token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        elif token == separator and depth == 0:
            parts.append(body[start:match.start()])
            start = match.end()
    parts.append(body[start:])
    return parts


def derivation_rows(latex: str) -> list[str]:
    """One aligned block per row, showing only that row.

    The other rows stay as invisible \\phantom text, so every row keeps its exact place
    in the block and the = signs still line up once the rows are cropped and stacked.
    """
    found = ROWS.match(latex)
    if not found:
        return [latex]
    rows = [row for row in split_top_level(found.group(2), "\\\\") if row.strip()]
    if not 2 <= len(rows) <= theme.MAX_DERIVATION_LINES:
        return [latex]

    def hidden(row: str) -> str:
        return " & ".join(f"\\phantom{{{cell.strip()}}}" for cell in split_top_level(row, "&"))

    return ["\\begin{aligned}\n" + " \\\\\n".join(row if index == shown else hidden(row)
                                                  for index, row in enumerate(rows)) + "\n\\end{aligned}"
            for shown in range(len(rows))]


def equation_page(latex: str) -> str:
    body = " ".join(latex.split())
    return body if DISPLAY_ENVIRONMENT.match(body) else f"\\[ {body} \\]"


def equation_source(pages: list[str]) -> str:
    """One page per equation, each on one line, so an error's line number names it."""
    return EQUATION_PAGE + "\n\\clearpage\n".join(equation_page(page) for page in pages) + "\n\\end{document}\n"


def failing_page(done, log: str, source: str) -> int | None:
    """The 1-based page an error's line number falls on."""
    found = LINE_ERROR.search(done.stderr) or LINE_ERROR.search(log)
    if not found:
        return None
    before = "\n".join(source.split("\n")[:int(found.group(1)) - 1])
    return before.count("\\clearpage") + 1


def equation_groups(slides: list[dict]) -> list[tuple[int, list[str]]]:
    """Each equation as a group of pages: several for a derivation that builds row by row."""
    groups = []
    for number, slide in enumerate(slides):
        for latex in slide.get("equations", []):
            groups.append((number, derivation_rows(latex) if slide.get("animate") else [latex]))
    return groups


def transparent(image, box: tuple[int, int, int, int]):
    """The ink colour with the page's darkness as alpha, so it sits on any panel."""
    from PIL import Image, ImageOps

    cropped = ImageOps.invert(image.crop(box).convert("L"))
    red, green, blue = (int(theme.INK[index:index + 2], 16) for index in (0, 2, 4))
    result = Image.new("RGBA", cropped.size, (red, green, blue, 0))
    result.putalpha(cropped)
    return result


def render_equations(slides: list[dict]) -> dict[int, list[list[Path]]]:
    """For each slide, its equations, each as one picture per row."""
    from PIL import Image, ImageChops

    groups = equation_groups(slides)
    pages = [page for _, group in groups for page in group]
    if not pages:
        return {}
    source = equation_source(pages)
    (WORK / "equations.tex").write_text(source, encoding="utf-8")
    done, log = tectonic("equations.tex")
    if done.returncode != 0 or not (WORK / "equations.pdf").is_file():
        number = failing_page(done, log, source)
        where = f"equation {number}: {pages[number - 1]!r}: " if number else ""
        finish(ok=False, error=f"TeX: {where}{tex_errors(done, log)}")
    if missing_characters(log):
        finish(ok=False, error=f"TeX: no glyph for {missing_characters(log)}")
    run(["pdftoppm", "-r", str(EQUATION_DPI), "-gray", "-png", str(WORK / "equations.pdf"), str(WORK / "equation")])
    files = sorted(WORK.glob("equation-*.png"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))
    if len(files) != len(pages):
        finish(ok=False, error=f"TeX made {len(files)} pages for {len(pages)} equations")

    rendered: dict[int, list[list[Path]]] = {}
    position = 0
    for slide, group in groups:
        images = [Image.open(path).convert("L") for path in files[position:position + len(group)]]
        boxes = [ImageChops.invert(image).getbbox() for image in images]
        if any(box is None for box in boxes):
            finish(ok=False, error=f"TeX: an equation on slide {slide + 1} rendered empty")
        left = max(min(box[0] for box in boxes) - ROW_PADDING, 0)    # rows share a left edge
        right = max(box[2] for box in boxes) + ROW_PADDING
        rows = []
        for index, (image, box) in enumerate(zip(images, boxes)):
            path = WORK / f"row-{position + index}.png"
            transparent(image, (left, max(box[1] - ROW_PADDING, 0), right, box[3] + ROW_PADDING)).save(path)
            rows.append(path)
        rendered.setdefault(slide, []).append(rows)
        position += len(group)
    return rendered


def inches(value: float) -> int:
    return int(value * EMU)


def set_box(shape, left: float, top: float, width: float, height: float) -> None:
    shape.left, shape.top, shape.width, shape.height = inches(left), inches(top), inches(width), inches(height)


def style_theme(master) -> None:
    """Put the palette and font pair into the file's theme part."""
    from lxml import etree
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT

    part = master.part.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    root.set("name", "Study Assistant")
    scheme = root.find(f".//{a}clrScheme")
    scheme.set("name", "Study Assistant")
    colours = {"dk1": theme.INK, "lt1": theme.WHITE, "dk2": theme.ACCENT_DARK, "lt2": theme.SURFACE,
               "accent1": theme.ACCENT, "accent2": theme.HIGHLIGHT, "accent3": theme.MUTED, "accent4": theme.RULE,
               "accent5": theme.ACCENT_DARK, "accent6": theme.MUTED, "hlink": theme.ACCENT, "folHlink": theme.MUTED}
    for name, value in colours.items():
        slot = scheme.find(f"{a}{name}")
        for child in list(slot):
            slot.remove(child)
        etree.SubElement(slot, f"{a}srgbClr").set("val", value)
    fonts = root.find(f".//{a}fontScheme")
    fonts.set("name", "Study Assistant")
    fonts.find(f"{a}majorFont/{a}latin").set("typeface", theme.HEADING_FONT)
    fonts.find(f"{a}minorFont/{a}latin").set("typeface", theme.BODY_FONT)
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def text_style(level, size: int, bold: bool, scheme: str, major: bool) -> None:
    """Set one level of the master's text styles."""
    from pptx.oxml import parse_xml

    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    old = level.find(f"{a}defRPr")
    new = parse_xml(
        f'<a:defRPr {NS} sz="{size * 100}" b="{int(bold)}" kern="1200">'
        f'<a:solidFill><a:schemeClr val="{scheme}"/></a:solidFill>'
        f'<a:latin typeface="{"+mj-lt" if major else "+mn-lt"}"/></a:defRPr>')
    level.replace(old, new)


def rule_shape(left: float, top: float, scheme: str):
    """The short accent line under a title."""
    from pptx.oxml import parse_xml

    return parse_xml(
        f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="90" name="accent rule"/><p:cNvSpPr/><p:nvPr userDrawn="1"/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{inches(left)}" y="{inches(top)}"/>'
        f'<a:ext cx="{inches(1.1)}" cy="{inches(0.045)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:schemeClr val="{scheme}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>')


def style_master(master) -> None:
    from pptx.oxml import parse_xml

    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    p = "{%s}" % NS_P
    styles = master._element.find(f"{p}txStyles")
    title = styles.find(f"{p}titleStyle/{a}lvl1pPr")
    title.set("algn", "l")
    text_style(title, TITLE_SIZES[0], True, "accent1", True)
    body = styles.find(f"{p}bodyStyle/{a}lvl1pPr")
    body.set("marL", str(inches(BULLET_INDENT)))
    body.set("indent", str(-inches(BULLET_INDENT)))
    for child in list(body):
        if child.tag != f"{a}defRPr":
            body.remove(child)
    body.insert(0, parse_xml(f'<a:spcBef {NS}><a:spcPts val="{int(BULLET_GAP * 72 * 100)}"/></a:spcBef>'))
    body.insert(1, parse_xml(f'<a:buClr {NS}><a:schemeClr val="accent1"/></a:buClr>'))
    body.insert(2, parse_xml(f'<a:buFont {NS} typeface="Arial"/>'))
    body.insert(3, parse_xml(f'<a:buChar {NS} char="&#8226;"/>'))
    text_style(body, BULLET_SIZES[0], False, "tx1", False)

    tree = master.shapes._spTree
    for shape in list(master.placeholders):
        kind = str(shape.placeholder_format.type)
        if "TITLE" not in kind and "BODY" not in kind:
            tree.remove(shape._element)
            continue
        set_box(shape, MARGIN, *((TITLE_TOP, CONTENT, TITLE_HEIGHT) if "TITLE" in kind
                                 else (BODY_TOP, CONTENT, BODY_BOTTOM - BODY_TOP)))
        properties = shape._element.txBody.find(f"{a}bodyPr")
        properties.set("lIns", "0")     # so text lines up with the rule and panels
        properties.set("rIns", "0")
    tree.append(rule_shape(MARGIN, RULE_TOP, "accent1"))


def own_style(shape, size: int, scheme: str, bold: bool, major: bool, anchor: str | None = None) -> None:
    """A layout's text with no bullet, for what isn't body text: headings and subtitles."""
    from pptx.oxml import parse_xml

    body = shape._element.txBody
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    body.replace(body.find(f"{a}lstStyle"), parse_xml(
        f'<a:lstStyle {NS}><a:lvl1pPr marL="0" indent="0" algn="l"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:buNone/>'
        f'<a:defRPr sz="{size * 100}" b="{int(bold)}"><a:solidFill><a:schemeClr val="{scheme}"/></a:solidFill>'
        f'<a:latin typeface="{"+mj-lt" if major else "+mn-lt"}"/></a:defRPr></a:lvl1pPr></a:lstStyle>'))
    if anchor:
        shape._element.txBody.find(f"{a}bodyPr").set("anchor", anchor)


def body_placeholders(owner) -> list:
    """Everything but the title, in order."""
    return sorted((shape for shape in owner.placeholders if shape.placeholder_format.idx != 0),
                  key=lambda shape: shape.placeholder_format.idx)


def shape_layouts(deck) -> dict:
    """Turn six of the template's layouts into the theme's six, and drop the rest."""
    from pptx.dml.color import RGBColor

    chosen = {0: "title", 1: "bullets_equation", 5: "big_equation", 3: "figure_text", 4: "two_columns", 2: "summary"}
    layouts = list(deck.slide_layouts)
    for index, layout in reversed(list(enumerate(layouts))):
        if index not in chosen:
            deck.slide_layouts.remove(layout)
    named = {chosen[index]: layouts[index] for index in chosen}
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for name, layout in named.items():
        layout._element.cSld.set("name", name)
        for shape in list(layout.placeholders):
            if shape.placeholder_format.idx >= 10:
                shape._element.getparent().remove(shape._element)
                continue
            # The template's layouts restyle their own text, which would beat the theme.
            style_list = shape._element.txBody.find(f"{a}lstStyle")
            for child in list(style_list):
                style_list.remove(child)
            properties = shape._element.txBody.find(f"{a}bodyPr")
            for attribute in list(properties.attrib):
                del properties.attrib[attribute]
            if shape.placeholder_format.idx == 0 and name != "title":
                set_box(shape, MARGIN, TITLE_TOP, CONTENT, TITLE_HEIGHT)

    cover = named["title"]
    cover._element.set("showMasterSp", "0")
    cover.background.fill.solid()
    cover.background.fill.fore_color.rgb = RGBColor.from_string(theme.ACCENT_DARK)
    set_box(cover.placeholders[0], 0.9, 1.7, WIDTH - 1.8, 2.3)
    set_box(cover.placeholders[1], 0.9, 4.35, WIDTH - 1.8, 1.1)
    own_style(cover.placeholders[0], COVER_SIZES[0], "bg1", True, True, "b")
    own_style(cover.placeholders[1], SUBTITLE_SIZE, "bg2", False, False)
    cover.shapes._spTree.append(rule_shape(0.9, 4.16, "accent2"))

    set_box(body_placeholders(named["bullets_equation"])[0], MARGIN, BODY_TOP, CONTENT, BODY_BOTTOM - BODY_TOP)
    summary = body_placeholders(named["summary"])[0]
    set_box(summary, MARGIN + 0.4, BODY_TOP + 0.3, CONTENT - 0.8, BODY_BOTTOM - BODY_TOP - 0.6)
    summary._element.txBody.find(f"{a}bodyPr").set("anchor", "ctr")
    figure = body_placeholders(named["figure_text"])
    set_box(figure[0], MARGIN, BODY_TOP, FIGURE_TEXT_WIDTH, BODY_BOTTOM - BODY_TOP)
    figure[1]._element.getparent().remove(figure[1]._element)
    columns = body_placeholders(named["two_columns"])
    for index, shape in enumerate(columns):
        left = MARGIN + (index // 2) * (COLUMN + COLUMN_GAP) + CARD_PADDING
        if index % 2 == 0:
            set_box(shape, left, BODY_TOP + CARD_PADDING, COLUMN - 2 * CARD_PADDING, CARD_HEADING)
            own_style(shape, COLUMN_HEADING_SIZE, "accent1", True, True)
        else:
            set_box(shape, left, BODY_TOP + CARD_PADDING + CARD_HEADING, COLUMN - 2 * CARD_PADDING,
                    BODY_BOTTOM - BODY_TOP - CARD_HEADING - 2 * CARD_PADDING)
    return named


def themed_deck():
    from pptx import Presentation

    deck = Presentation()
    deck.slide_width, deck.slide_height = inches(WIDTH), inches(HEIGHT)
    style_theme(deck.slide_master)
    style_master(deck.slide_master)
    return deck, shape_layouts(deck)


def script_parts(text: str) -> list[tuple[str, int]]:
    """Text split where a real subscript or superscript goes, with each part's baseline."""
    parts, position = [], 0
    for found in SCRIPTED.finditer(text):
        parts.append((text[position:found.end(1)], 0))
        parts.append((found.group(3) or found.group(4), SUBSCRIPT if found.group(2) == "_" else SUPERSCRIPT))
        position = found.end()
    parts.append((text[position:], 0))
    return [part for part in parts if part[0]]


def write_runs(paragraph, text: str, points: int) -> None:
    from pptx.util import Pt

    for part, baseline in script_parts(text):
        run_ = paragraph.add_run()
        run_.text = part
        run_.font.size = Pt(points)
        if baseline:
            run_._r.get_or_add_rPr().set("baseline", str(baseline))


def write_notes(frame, notes: str) -> None:
    """Speaker notes get real subscripts too: M_Z, never a lowercase letter standing in."""
    for index, line in enumerate(notes.split("\n")):
        write_runs(frame.paragraphs[0] if index == 0 else frame.add_paragraph(), line, NOTES_SIZE)


def write_title(page, text: str, sizes: tuple[int, ...] = TITLE_SIZES, width: float = CONTENT) -> None:
    size = next((size for size in sizes if metrics.width_of(text, "Calibri bold", size) <= width - 0.2), sizes[-1])
    write_runs(page.shapes.title.text_frame.paragraphs[0], text, size)


def write_bullets(shape, bullets: list[str], points: int) -> None:
    frame = shape.text_frame
    frame.word_wrap = True
    for index, line in enumerate(bullets):
        write_runs(frame.paragraphs[0] if index == 0 else frame.add_paragraph(), line, points)


def bullet_height(bullets: list[str], points: int, width: float) -> float:
    return metrics.height_of(bullets, "Cambria", points, width - BULLET_INDENT - 0.2, BULLET_GAP) + 0.2


def fitting_size(bullets: list[str], sizes: tuple[int, ...], width: float, room: float) -> tuple[int, float]:
    """The largest size whose text fits. The limits mean the smallest always should."""
    for points in sizes:
        height = bullet_height(bullets, points, width)
        if height <= room:
            return points, height
    return sizes[-1], min(bullet_height(bullets, sizes[-1], width), room)


def add_panel(page, left: float, top: float, width: float, height: float):
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE

    panel = page.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, inches(left), inches(top), inches(width), inches(height))
    panel.name = "panel"
    panel.adjustments[0] = 0.04
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor.from_string(theme.SURFACE)
    panel.line.fill.background()
    panel.shadow.inherit = False
    return panel


def natural_size(path: Path) -> tuple[float, float]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size[0] / EQUATION_DPI, image.size[1] / EQUATION_DPI


def place_equations(page, equations: list[list[Path]], sources: list[str], left: float, top: float,
                    width: float, height: float, largest: float, warnings: list[str], number: int) -> list:
    """Fill the frame: as large as fits, up to `largest`, centred. Returns the pictures in order."""
    if not equations:
        return []
    rows = [(index, path, *natural_size(path)) for index, group in enumerate(equations) for path in group]
    gaps = GAP * (len(equations) - 1)
    scale = min(largest, min(width / wide for _, _, wide, _ in rows),
                (height - gaps) / sum(tall for *_, tall in rows))
    if scale < SMALL_SCALE:
        warnings.append(f"An equation on slide {number} had to be drawn small. Consider splitting it.")
    total = sum(tall * scale for *_, tall in rows) + gaps
    y = top + (height - total) / 2
    pictures = []
    previous = rows[0][0]
    for index, path, wide, tall in rows:
        if index != previous:
            y += GAP
            previous = index
        x = left + (width - wide * scale) / 2
        picture = page.shapes.add_picture(str(path), inches(x), inches(y), inches(wide * scale), inches(tall * scale))
        picture.name = f"equation {len(pictures) + 1}"
        picture._element.nvPicPr.cNvPr.set("descr", sources[min(index, len(sources) - 1)])
        pictures.append(picture)
        y += tall * scale
    return pictures


def add_text(page, name: str, text: str, left: float, top: float, width: float, height: float,
             points: int, colour: str, italic: bool = False, align=None) -> None:
    from pptx.dml.color import RGBColor
    from pptx.util import Pt

    box = page.shapes.add_textbox(inches(left), inches(top), inches(width), inches(height))
    box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    paragraph = frame.paragraphs[0]
    write_runs(paragraph, text, points)
    if align is not None:
        paragraph.alignment = align
    for run_ in paragraph.runs:
        run_.font.color.rgb = RGBColor.from_string(colour)
        run_.font.italic = italic


def add_footer(page, slide: dict, number: int) -> None:
    """One line of sources, and the slide number as a live field."""
    from pptx.enum.text import PP_ALIGN
    from pptx.oxml import parse_xml

    if slide.get("footer"):
        add_text(page, "sources", slide["footer"], MARGIN, FOOTER_TOP, CONTENT - NUMBER_WIDTH, FOOTER_HEIGHT,
                 FOOTER_SIZE, theme.MUTED)
    add_text(page, "slide number", str(number), WIDTH - MARGIN - NUMBER_WIDTH, FOOTER_TOP, NUMBER_WIDTH,
             FOOTER_HEIGHT, FOOTER_SIZE, theme.MUTED, align=PP_ALIGN.RIGHT)
    shape = page.shapes[-1]
    run_ = shape.text_frame.paragraphs[0].runs[0]._r
    field = parse_xml(f'<a:fld {NS} id="{{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}}" type="slidenum">'
                      f'<a:t>{number}</a:t></a:fld>')
    field.insert(0, run_.find("{http://schemas.openxmlformats.org/drawingml/2006/main}rPr"))
    run_.getparent().replace(run_, field)


def timing_xml(steps: list[tuple[int, int | None]]) -> str:
    """Each step appears on a click: a (shape id, paragraph) pair, or a whole shape."""
    next_id = iter(range(3, 10000))
    clicks = []
    for shape_id, paragraph in steps:
        outer, middle, effect, setter = (next(next_id) for _ in range(4))
        target = (f'<p:spTgt spid="{shape_id}"><p:txEl><p:pRg st="{paragraph}" end="{paragraph}"/></p:txEl></p:spTgt>'
                  if paragraph is not None else f'<p:spTgt spid="{shape_id}"/>')
        group = ' grpId="0"' if paragraph is not None else ""
        clicks.append(
            f'<p:par><p:cTn id="{outer}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
            f'<p:childTnLst><p:par><p:cTn id="{middle}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'<p:childTnLst><p:par><p:cTn id="{effect}" presetID="1" presetClass="entr" presetSubtype="0" '
            f'fill="hold"{group} nodeType="clickEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{setter}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'</p:cTn><p:tgtEl>{target}</p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName>'
            '</p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
            '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
    text_shapes = list(dict.fromkeys(shape_id for shape_id, paragraph in steps if paragraph is not None))
    builds = "".join(f'<p:bldP spid="{shape_id}" grpId="0" build="p"/>' for shape_id in text_shapes)
    return (
        f'<p:timing {NS}><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        '<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
        f'<p:childTnLst>{"".join(clicks)}</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0">'
        '<p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0">'
        '<p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
        f'{f"<p:bldLst>{builds}</p:bldLst>" if builds else ""}</p:timing>')


def add_motion(page, steps: list[tuple[int, int | None]]) -> None:
    """A fade on every slide, and click-by-click steps where asked. The order in the XML matters."""
    from pptx.oxml import parse_xml

    root = page._element
    root.append(parse_xml(TRANSITION.format(p=NS_P)))
    if steps:
        root.append(parse_xml(timing_xml(steps)))


def caption_height(text: str, width: float, points: int) -> float:
    return metrics.height_of([text], "Cambria italic", points, width * CAPTION_SAFETY) + 0.14


def bring_to_front(page, shape) -> None:
    """Placeholders are created first, so anything drawn later would cover them."""
    page.shapes._spTree.remove(shape._element)
    page.shapes._spTree.append(shape._element)


def fill_title(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    write_title(page, slide["title"], COVER_SIZES, WIDTH - 1.8)
    subtitle = page.placeholders[1]
    if slide.get("subtitle"):
        write_runs(subtitle.text_frame.paragraphs[0], slide["subtitle"], SUBTITLE_SIZE)
    else:
        subtitle._element.getparent().remove(subtitle._element)
    return []


def fill_bullets_equation(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    body = body_placeholders(page)[0]
    bullets = slide.get("bullets", [])
    steps: list = []
    if not equations:
        points, _ = fitting_size(bullets, BULLET_SIZES, CONTENT, BODY_BOTTOM - BODY_TOP)
        write_bullets(body, bullets, points)
        return [(body.shape_id, index) for index in range(len(bullets))]
    if bullets:
        points, height = fitting_size(bullets, EQUATION_BULLET_SIZES, CONTENT, 2.6)
        set_box(body, MARGIN, BODY_TOP, CONTENT, height)
        write_bullets(body, bullets, points)
        steps = [(body.shape_id, index) for index in range(len(bullets))]
        top = BODY_TOP + height + GAP
    else:
        body._element.getparent().remove(body._element)
        top = BODY_TOP
    add_panel(page, MARGIN, top, CONTENT, BODY_BOTTOM - top)
    pictures = place_equations(page, equations, slide["equations"], MARGIN + PANEL_PADDING, top + PANEL_PADDING,
                               CONTENT - 2 * PANEL_PADDING, BODY_BOTTOM - top - 2 * PANEL_PADDING, PANEL_SCALE,
                               warnings, number)
    return steps + [(picture.shape_id, None) for picture in pictures]


def fill_big_equation(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    room = caption_height(slide["caption"], CONTENT, CAPTION_SIZE + 2) if slide.get("caption") else 0
    bottom = BODY_BOTTOM - room - (0.08 if room else 0)
    add_panel(page, MARGIN, BODY_TOP, CONTENT, bottom - BODY_TOP)
    pictures = place_equations(page, equations, slide["equations"], MARGIN + PANEL_PADDING, BODY_TOP + PANEL_PADDING,
                               CONTENT - 2 * PANEL_PADDING, bottom - BODY_TOP - 2 * PANEL_PADDING, BIG_SCALE,
                               warnings, number)
    if slide.get("caption"):
        add_text(page, "caption", slide["caption"], MARGIN, bottom + 0.08, CONTENT, room, CAPTION_SIZE + 2,
                 theme.MUTED, italic=True)
    return [(picture.shape_id, None) for picture in pictures]


def fill_figure_text(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    """The picture beside the text, or above it if it's wide enough to waste a column."""
    from PIL import Image
    from pptx.enum.text import PP_ALIGN

    path = WORK / "images" / Path(slide.get("image") or "missing").name
    if not path.is_file():
        finish(ok=False, error=f"slide {number} needs a picture, and there's none called {slide.get('image')!r}")
    with Image.open(path) as opened:
        ratio = opened.size[1] / opened.size[0]
    body = body_placeholders(page)[0]
    bullets = slide.get("bullets", [])
    caption = slide.get("caption", "")
    steps: list = []
    pictures: list = []

    if ratio < WIDE_FIGURE:
        text_height = 0.0
        if bullets:
            points, text_height = fitting_size(bullets, EQUATION_BULLET_SIZES, CONTENT, 1.9)
        # The equations get a panel under the text, as beside a tall figure; this branch used to
        # drop them without a word (review R2-C8, Nov 2026).
        panel = WIDE_EQUATION_ROOM if equations else 0.0
        room = BODY_BOTTOM - BODY_TOP - text_height - GAP - (panel + GAP if equations else 0)
        room -= caption_height(caption, CONTENT, CAPTION_SIZE) if caption else 0
        width = min(CONTENT, room / ratio)
        picture = page.shapes.add_picture(str(path), inches(MARGIN + (CONTENT - width) / 2), inches(BODY_TOP),
                                          inches(width), inches(width * ratio))
        picture.name = "image"
        y = BODY_TOP + width * ratio
        if caption:
            height = caption_height(caption, CONTENT, CAPTION_SIZE)
            add_text(page, "caption", caption, MARGIN, y + 0.04, CONTENT, height, CAPTION_SIZE, theme.MUTED,
                     italic=True, align=PP_ALIGN.CENTER)
            y += height
        if bullets:
            set_box(body, MARGIN, y + 0.1, CONTENT, text_height)
            write_bullets(body, bullets, points)
            steps = [(body.shape_id, index) for index in range(len(bullets))]
            y += 0.1 + text_height
        else:
            body._element.getparent().remove(body._element)
        if equations:
            top = y + GAP
            add_panel(page, MARGIN, top, CONTENT, BODY_BOTTOM - top)
            pictures = place_equations(page, equations, slide["equations"], MARGIN + PANEL_PADDING, top + PANEL_PADDING,
                                       CONTENT - 2 * PANEL_PADDING, BODY_BOTTOM - top - 2 * PANEL_PADDING, PANEL_SCALE,
                                       warnings, number)
    else:
        room = BODY_BOTTOM - BODY_TOP - (1.4 if equations else 0)
        if bullets:
            points, height = fitting_size(bullets, EQUATION_BULLET_SIZES, FIGURE_TEXT_WIDTH, room)
            set_box(body, MARGIN, BODY_TOP, FIGURE_TEXT_WIDTH, height)
            write_bullets(body, bullets, points)
            steps = [(body.shape_id, index) for index in range(len(bullets))]
        else:
            height = 0.0
            body._element.getparent().remove(body._element)
        if equations:
            top = BODY_TOP + height + GAP
            add_panel(page, MARGIN, top, FIGURE_TEXT_WIDTH, BODY_BOTTOM - top)
            pictures = place_equations(page, equations, slide["equations"], MARGIN + 0.15, top + 0.15,
                                       FIGURE_TEXT_WIDTH - 0.3, BODY_BOTTOM - top - 0.3, PANEL_SCALE, warnings, number)
        left = MARGIN + FIGURE_TEXT_WIDTH + COLUMN_GAP
        width = WIDTH - MARGIN - left
        below = caption_height(caption, width, CAPTION_SIZE) if caption else 0
        shown = min(width, (BODY_BOTTOM - BODY_TOP - below) / ratio)
        picture = page.shapes.add_picture(str(path), inches(left + (width - shown) / 2), inches(BODY_TOP),
                                          inches(shown), inches(shown * ratio))
        picture.name = "image"
        if caption:
            add_text(page, "caption", caption, left, BODY_TOP + shown * ratio + 0.04, width, below, CAPTION_SIZE,
                     theme.MUTED, italic=True, align=PP_ALIGN.CENTER)
    return steps + [(item.shape_id, None) for item in pictures]


def fill_two_columns(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    heads = [slide.get("left_heading", ""), slide.get("right_heading", "")]
    columns = [slide.get("left_bullets", []), slide.get("right_bullets", [])]
    shapes = body_placeholders(page)
    inner = COLUMN - 2 * CARD_PADDING
    room = BODY_BOTTOM - BODY_TOP - CARD_HEADING - 2 * CARD_PADDING
    points = min((fitting_size(column, BULLET_SIZES, inner, room)[0] for column in columns if column),
                 default=BULLET_SIZES[0])
    tallest = max(bullet_height(column, points, inner) for column in columns)
    height = min(CARD_HEADING + tallest + 2 * CARD_PADDING, BODY_BOTTOM - BODY_TOP)
    top = BODY_TOP + (BODY_BOTTOM - BODY_TOP - height) / 2      # short cards sit in the middle
    steps: list = []
    for index, (head, column) in enumerate(zip(heads, columns)):
        left = MARGIN + index * (COLUMN + COLUMN_GAP)
        add_panel(page, left, top, COLUMN, height)
        head_shape, body = shapes[2 * index], shapes[2 * index + 1]
        set_box(head_shape, left + CARD_PADDING, top + CARD_PADDING, inner, CARD_HEADING)
        set_box(body, left + CARD_PADDING, top + CARD_PADDING + CARD_HEADING, inner, height - CARD_HEADING - 2 * CARD_PADDING)
        write_runs(head_shape.text_frame.paragraphs[0], head, COLUMN_HEADING_SIZE)
        bring_to_front(page, head_shape)
        if column:
            write_bullets(body, column, points)
            bring_to_front(page, body)
            steps += [(body.shape_id, line) for line in range(len(column))]
        else:
            body._element.getparent().remove(body._element)
    return steps


def fill_summary(page, slide: dict, equations, warnings: list[str], number: int) -> list:
    """The panel is as tall as its points, not the whole slide."""
    bullets = slide.get("bullets", [])
    width = CONTENT - 2 * SUMMARY_PADDING
    points, height = fitting_size(bullets, BULLET_SIZES, width, BODY_BOTTOM - BODY_TOP - 2 * SUMMARY_PADDING)
    add_panel(page, MARGIN, BODY_TOP, CONTENT, height + 2 * SUMMARY_PADDING)
    body = body_placeholders(page)[0]
    set_box(body, MARGIN + SUMMARY_PADDING, BODY_TOP + SUMMARY_PADDING, width, height)
    bring_to_front(page, body)
    write_bullets(body, bullets, points)
    return [(body.shape_id, index) for index in range(len(bullets))]


FILL = {"title": fill_title, "bullets_equation": fill_bullets_equation, "big_equation": fill_big_equation,
        "figure_text": fill_figure_text, "two_columns": fill_two_columns, "summary": fill_summary}


def add_slide(deck, layouts: dict, number: int, slide: dict, equations: list, warnings: list[str]) -> None:
    page = deck.slides.add_slide(layouts[slide["layout"]])
    if slide["layout"] != "title":
        write_title(page, slide["title"])
    steps = FILL[slide["layout"]](page, slide, equations, warnings, number)
    if slide["layout"] != "title":
        add_footer(page, slide, number)
    if slide.get("notes"):
        write_notes(page.notes_slide.notes_text_frame, slide["notes"])
    add_motion(page, steps if slide.get("animate") else [])


def build_deck() -> None:
    slides = json.loads((WORK / "deck.json").read_text(encoding="utf-8"))["slides"]
    rendered = render_equations(slides)
    deck, layouts = themed_deck()
    warnings: list[str] = []
    for index, slide in enumerate(slides):
        add_slide(deck, layouts, index + 1, slide, rendered.get(index, []), warnings)
    deck.save(OUT / "out.pptx")
    finish(ok=True, file="out.pptx", slides=len(slides), warnings=warnings)
