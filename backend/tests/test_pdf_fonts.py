"""Reading glyph names out of embedded fonts, with small fonts built here."""
import pytest

from app.ingest.pdf_fonts import PageGlyphNames, _names_from, cff_glyph_names, type1_glyph_names


def eexec_encrypt(plain: bytes) -> bytes:
    key = 55665
    cipher = bytearray()
    for byte in b"\0\0\0\0" + plain:
        code = byte ^ (key >> 8)
        cipher.append(code)
        key = ((code + key) * 52845 + 22719) & 0xFFFF
    return bytes(cipher)


def type1(body: bytes) -> bytes:
    return b"%!PS-AdobeFont-1.0: CMEX10\ncurrentfile eexec\n" + eexec_encrypt(body)


def test_type1_names_come_in_file_order_with_notdef_first():
    font = type1(b"/CharStrings 3 dict dup begin\n"
                 b"/bracketleftbig 2 RD ab ND\n/.notdef 1 RD c ND\n/summationdisplay 3 RD def ND\n")
    assert type1_glyph_names(font) == [".notdef", "bracketleftbig", "summationdisplay"]


def test_binary_outlines_are_skipped_not_searched():
    outline = b"/fake 1 RD x"                     # looks like a glyph, but it's inside one
    font = type1(b"/CharStrings 1 dict dup begin\n/real %d RD " % len(outline) + outline + b" ND\n")
    assert type1_glyph_names(font) == ["real"]


def test_the_other_charstring_spelling_is_read_too():
    assert type1_glyph_names(type1(b"/CharStrings\n/a 1 -| x |-\n")) == ["a"]


@pytest.mark.parametrize("program", [b"", b"no encrypted part", type1(b"nothing useful")])
def test_a_font_without_glyphs_gives_no_names(program):
    assert type1_glyph_names(program) == []


def test_cff_names_come_in_id_order():
    fontbuilder = pytest.importorskip("fontTools.fontBuilder")
    from fontTools.pens.t2CharStringPen import T2CharStringPen

    names = [".notdef", "bracketleftbig", "summationdisplay"]
    pen = T2CharStringPen(600, None)
    builder = fontbuilder.FontBuilder(1000, isTTF=False)
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({})
    builder.setupCFF("CMEX10", {}, {name: pen.getCharString() for name in names}, {})
    table = builder.font["CFF "]
    assert cff_glyph_names(table.compile(builder.font)) == names


def test_a_damaged_font_gives_no_names_and_no_error():
    assert _names_from("cff", b"not a font") == ()


class FakeDocument:
    def __init__(self, program):
        self.program = program
        self.opened = 0

    def extract_font(self, xref):
        self.opened += 1
        return ("ABCDEF+CMEX10", "pfa", "Type1", self.program)


class FakePage:
    def __init__(self, parent, fonts=("ABCDEF+CMEX10",)):
        self.parent = parent
        self.fonts = fonts

    def get_fonts(self):
        return [(9 + i, "pfa", "Type1", name, "F1", "") for i, name in enumerate(self.fonts)]


@pytest.fixture
def document():
    return FakeDocument(type1(b"/CharStrings\n/.notdef 1 RD a ND\n/parenleftbig 1 RD b ND\n"))


def test_a_name_is_found_by_font_and_glyph_id(document):
    assert PageGlyphNames(FakePage(document)).name("ABCDEF+CMEX10", 1) == "parenleftbig"


def test_a_font_is_found_without_its_subset_prefix(document):
    assert PageGlyphNames(FakePage(document)).name("CMEX10", 1) == "parenleftbig"


def test_a_font_name_the_trace_cut_short_is_still_found(document):
    page = FakePage(document, ("ABCDEF+LMMathExtension10-Regular",))
    assert PageGlyphNames(page).name("LMMathExtension10-Regula", 1) == "parenleftbig"


def test_a_cut_name_that_fits_two_fonts_finds_neither(document):
    page = FakePage(document, ("A+LongFontFamilyNameHere-Bold", "B+LongFontFamilyNameHere-Book"))
    assert PageGlyphNames(page).name("LongFontFamilyNameHere-B", 1) is None


def test_a_short_name_is_never_taken_for_the_start_of_a_longer_one(document):
    assert PageGlyphNames(FakePage(document, ("A+CMEX10",))).name("CMEX", 1) is None


def test_a_page_opens_each_font_once(document):
    names = PageGlyphNames(FakePage(document))
    names.name("CMEX10", 0)
    names.name("CMEX10", 1)
    assert document.opened == 1


@pytest.mark.parametrize("font, gid", [("CMEX10", 99), ("CMEX10", -1), ("NoSuchFont", 1)])
def test_an_unknown_glyph_has_no_name(document, font, gid):
    assert PageGlyphNames(FakePage(document)).name(font, gid) is None
