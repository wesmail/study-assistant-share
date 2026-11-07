"""Builds the Word template pandoc applies to every document, once, as the image is built.

pandoc copies styles, margins and footers from a "reference" document, so the styling
lives in this file's output and not in each generated document.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

import theme

MARGIN_CM = 2.3
HEADINGS = {"Heading 1": (20, 18, 6), "Heading 2": (15, 14, 4), "Heading 3": (12.5, 10, 3)}   # size, before, after


def set_font(style, name: str, size: float | None = None, bold: bool | None = None, colour: str | None = None) -> None:
    """Name the font outright: the defaults point at theme fonts, which would win."""
    properties = style.element.get_or_add_rPr()
    fonts = properties.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        properties.insert(0, fonts)
    for attribute in list(fonts.attrib):
        del fonts.attrib[attribute]
    for slot in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(slot), name)
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if colour:
        style.font.color.rgb = RGBColor.from_string(colour)


def style_table(document) -> None:
    """Light rules, room in the cells, and a header row in the accent colour."""
    table = document.styles["Table"].element
    properties = table.find(qn("w:tblPr"))
    if properties is None:
        properties = OxmlElement("w:tblPr")
        table.append(properties)
    for child in list(properties):
        properties.remove(child)
    properties.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="{theme.ACCENT}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{theme.ACCENT}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{theme.RULE}"/>'
        '</w:tblBorders>'))
    properties.append(parse_xml(
        f'<w:tblCellMar {nsdecls("w")}><w:top w:w="70" w:type="dxa"/><w:left w:w="110" w:type="dxa"/>'
        '<w:bottom w:w="70" w:type="dxa"/><w:right w:w="110" w:type="dxa"/></w:tblCellMar>'))
    table.append(parse_xml(f'<w:trPr {nsdecls("w")}><w:cantSplit/></w:trPr>'))    # rows stay whole
    for old in table.findall(qn("w:tblStylePr")):
        table.remove(old)
    table.append(parse_xml(
        f'<w:tblStylePr {nsdecls("w")} w:type="firstRow"><w:rPr><w:b/><w:color w:val="{theme.WHITE}"/></w:rPr>'
        f'<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="{theme.ACCENT}"/></w:tcPr></w:tblStylePr>'))


def add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(theme.MUTED)
    for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
        if kind:
            mark = OxmlElement("w:fldChar")
            mark.set(qn("w:fldCharType"), kind)
            run._r.append(mark)
        else:
            instruction = OxmlElement("w:instrText")
            instruction.text = text
            run._r.append(instruction)


def main(target: str) -> None:
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                             capture_output=True, check=True).stdout
    path = Path(target)
    path.write_bytes(default)
    document = docx.Document(path)
    styles = document.styles

    # No colour on these: table cells inherit them, and it would beat the white table head.
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        set_font(styles[name], theme.BODY_FONT, 11)
    for name in ("Body Text", "First Paragraph"):
        styles[name].paragraph_format.space_before = Pt(0)
        styles[name].paragraph_format.space_after = Pt(7)
        styles[name].paragraph_format.line_spacing = 1.15
    for name, (size, before, after) in HEADINGS.items():
        set_font(styles[name], theme.HEADING_FONT, size, True, theme.ACCENT)
        styles[name].paragraph_format.space_before = Pt(before)
        styles[name].paragraph_format.space_after = Pt(after)
        styles[name].paragraph_format.keep_with_next = True
    set_font(styles["Title"], theme.HEADING_FONT, 26, True, theme.ACCENT_DARK)
    set_font(styles["Block Text"], theme.BODY_FONT, 10.5, colour=theme.MUTED)
    borders = parse_xml(f'<w:pBdr {nsdecls("w")}><w:left w:val="single" w:sz="18" w:space="10" '
                        f'w:color="{theme.ACCENT}"/></w:pBdr>')
    styles["Block Text"].element.get_or_add_pPr().append(borders)
    set_font(styles["Image Caption"], theme.BODY_FONT, 9.5, colour=theme.MUTED)
    set_font(styles["Table Caption"], theme.BODY_FONT, 9.5, colour=theme.MUTED)
    for name in ("Source Code", "Verbatim Char"):       # pandoc adds these when a document needs them
        if name in styles:
            set_font(styles[name], theme.MONO_FONT, 9.5)
    style_table(document)

    for section in document.sections:
        section.left_margin = section.right_margin = Cm(MARGIN_CM)
        section.top_margin = section.bottom_margin = Cm(MARGIN_CM)
        add_page_number(section)
    document.save(path)


if __name__ == "__main__":
    main(sys.argv[1])
