"""LibreOffice in its container (app/office.py, office/). Roadmap item 7, gate items 1 and 2.

The unit tests need nothing. The rest need Docker and scripts/build_office.sh, and are
skipped otherwise. Hostile files are built here at test time, none is kept in the repo.
As in test_sandbox_docker.py, where a test shows something can't happen, a control beside
it shows the same thing happening when it's allowed, so a pass can't mean a broken attempt.
"""
import asyncio
import shutil
import subprocess
import zipfile
from pathlib import Path

import pymupdf
import pytest
from pptx import Presentation
from pptx.util import Inches, Pt

from app import office
from app.config import Settings
from app.tools.sandbox import container_command, run_container, work_folder

IMAGE = Settings().office_image


def image_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


needs_image = pytest.mark.skipif(not image_ready(), reason=f"needs Docker and the {IMAGE} image")


@pytest.fixture
def settings(tmp_path):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    return Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library",
                    outputs_dir=tmp_path / "outputs", office_fonts_dir=fonts)


def make_deck(path: Path, slides: int = 3, notes: bool = True) -> Path:
    deck = Presentation()
    for number in range(1, slides + 1):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = f"Slide title {number}"
        slide.placeholders[1].text = f"Body words for slide {number}"
        group = slide.shapes.add_group_shape()
        box = group.shapes.add_textbox(Inches(1), Inches(5), Inches(4), Inches(1))
        box.text_frame.text = f"Grouped text {number}"
        box.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        if notes:
            slide.notes_slide.notes_text_frame.text = f"Speaker notes for slide {number}"
    deck.save(path)
    return path


def zip_with(path: Path, entries: dict[str, bytes], declared: dict[str, int] | None = None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    if declared:        # rewrite the sizes the directory claims, as a hostile file would
        raw = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as archive:
            infos = {info.filename: info for info in archive.infolist()}
        start = raw.rfind(b"PK\x01\x02", 0)
        while start != -1:
            name_length = int.from_bytes(raw[start + 28:start + 30], "little")
            name = raw[start + 46:start + 46 + name_length].decode()
            if name in declared:
                raw[start + 24:start + 28] = declared[name].to_bytes(4, "little")
            start = raw.rfind(b"PK\x01\x02", 0, start)
        path.write_bytes(bytes(raw))
        assert infos
    return path


# A deck whose Basic macro writes a file the moment it opens. The layout is what LibreOffice
# itself writes for a macro bound to "Open Document" (checked Nov 6, 2026). The event name
# and the language are QNames: without the dom and ooo namespaces declared, LibreOffice
# silently drops the binding.
MACRO_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" xmlns:script="urn:oasis:names:tc:opendocument:xmlns:script:1.0" xmlns:dom="http://www.w3.org/2001/xml-events" xmlns:ooo="http://openoffice.org/2004/office" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" office:version="1.3">
<office:scripts><office:event-listeners><script:event-listener script:language="ooo:script" script:event-name="dom:load" xlink:href="vnd.sun.star.script:Standard.Module1.Pwn?language=Basic&amp;location=document" xlink:type="simple"/></office:event-listeners></office:scripts>
<office:body><office:presentation><draw:page draw:name="p1"><draw:frame svg:x="1cm" svg:y="1cm" svg:width="10cm" svg:height="2cm"><draw:text-box><text:p>Macro test slide</text:p></draw:text-box></draw:frame></draw:page></office:presentation></office:body></office:document-content>"""
MACRO_MODULE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">Sub Pwn
 Open &quot;/work/out/pwned&quot; For Output As #1
 Print #1, &quot;ran&quot;
 Close #1
End Sub
</script:module>"""
MACRO_LIBRARIES = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE library:libraries PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "libraries.dtd">
<library:libraries xmlns:library="http://openoffice.org/2000/library" xmlns:xlink="http://www.w3.org/1999/xlink">
 <library:library library:name="Standard" library:link="false"/>
</library:libraries>"""
MACRO_LIBRARY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE library:library PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "library.dtd">
<library:library xmlns:library="http://openoffice.org/2000/library" library:name="Standard" library:readonly="false" library:passwordprotected="false">
 <library:element library:name="Module1"/>
</library:library>"""
MACRO_MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">
 <manifest:file-entry manifest:full-path="/" manifest:version="1.3" manifest:media-type="application/vnd.oasis.opendocument.presentation"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="Basic/Standard/Module1.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="Basic/Standard/script-lb.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="Basic/script-lc.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""


def make_macro_deck(path: Path) -> Path:
    """An ODF presentation named .pptx: LibreOffice goes by content, not by name."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.oasis.opendocument.presentation")
        for name, text in [("content.xml", MACRO_CONTENT), ("Basic/Standard/Module1.xml", MACRO_MODULE),
                           ("Basic/Standard/script-lb.xml", MACRO_LIBRARY),
                           ("Basic/script-lc.xml", MACRO_LIBRARIES), ("META-INF/manifest.xml", MACRO_MANIFEST)]:
            archive.writestr(name, text, compress_type=zipfile.ZIP_DEFLATED)
    return path


def open_in_container(settings, deck: Path, profile: str, seconds: int = 40) -> set[str]:
    """Open the deck as a reader would (not a conversion) with the given profile settings,
    in the office image under the same walls; return what's in out/ afterwards."""
    script = f"""mkdir -p /tmp/p/user
cat > /tmp/p/user/registrymodifications.xcu <<'X'
{profile}
X
timeout {seconds - 10} soffice -env:UserInstallation=file:///tmp/p --headless --norestore /work/in.pptx
true
"""
    with work_folder() as work:
        (work / "out").mkdir()
        shutil.copy(deck, work / "in.pptx")
        (work / "open.sh").write_text(script)
        asyncio.run(run_container(settings, work, ["sh", "/work/open.sh"], seconds, office.office_image(settings)))
        return {path.name for path in (work / "out").iterdir()}


MACROS_ALLOWED = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>false</value></prop></item>
</oor:items>"""
OUR_PROFILE = (Path(__file__).parents[2] / "office" / "registrymodifications.xcu").read_text()


class TestPrecheck:
    def test_refuses_other_kinds(self, settings, tmp_path):
        path = tmp_path / "a.ppt"
        path.write_bytes(b"x")
        with pytest.raises(office.OfficeError, match="Only .pptx and .docx"):
            office.precheck(path, settings)

    def test_refuses_a_file_that_isnt_a_zip(self, settings, tmp_path):
        path = tmp_path / "a.pptx"
        path.write_bytes(b"not a zip at all")
        with pytest.raises(office.OfficeError, match="isn't a valid Office file"):
            office.precheck(path, settings)

    def test_refuses_a_big_file(self, tmp_path):
        settings = Settings(office_max_bytes=1000)
        path = make_deck(tmp_path / "a.pptx")
        with pytest.raises(office.OfficeError, match="the limit is"):
            office.precheck(path, settings)

    def test_refuses_a_deck_of_2000_slides_before_any_container(self, settings, tmp_path):
        entries = {f"ppt/slides/slide{n}.xml": b"<x/>" for n in range(1, 2001)}
        path = zip_with(tmp_path / "many.pptx", entries)
        with pytest.raises(office.OfficeError, match="2000 slides"):
            office.precheck(path, settings)

    def test_refuses_a_zip_that_says_it_unpacks_to_too_much(self, settings, tmp_path):
        path = zip_with(tmp_path / "bomb.pptx", {"ppt/slides/slide1.xml": b"\0" * (20 * 2**20)})
        small = Settings(office_max_unpacked_bytes=10 * 2**20)
        with pytest.raises(office.OfficeError, match="unpacks to"):
            office.precheck(path, small)
        office.precheck(path, settings)     # under the real limit, it passes this check

    def test_a_normal_deck_passes(self, settings, tmp_path):
        office.precheck(make_deck(tmp_path / "a.pptx"), settings)


class TestContainer:
    def test_same_walls_as_the_sandbox_and_fonts_read_only(self, settings, tmp_path):
        command = container_command(settings, tmp_path, "n", ["x"], office.office_image(settings))
        joined = " ".join(command)
        for wall in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--pids-limit"):
            assert wall in joined
        mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
        assert mounts[0] == f"type=bind,source={tmp_path},target=/work"
        assert mounts[1] == f"type=bind,source={settings.office_fonts_dir},target=/opt/fonts/windows,readonly"
        assert len(mounts) == 2
        assert command[command.index("--memory") + 1] == settings.office_memory
        assert settings.office_image in command

    def test_no_fonts_mount_when_unset_or_missing(self, tmp_path):
        for fonts in (None, tmp_path / "nowhere"):
            image = office.office_image(Settings(office_fonts_dir=fonts))
            assert image.read_only == ()

    def test_the_sandbox_itself_is_unchanged(self, settings, tmp_path):
        command = container_command(settings, tmp_path, "n", ["x"])
        assert settings.sandbox_image in command and settings.office_image not in command
        assert sum(part == "--mount" for part in command) == 1
        assert command[command.index("--memory") + 1] == settings.sandbox_memory


@needs_image
class TestRendering:
    def test_a_deck_renders_one_page_per_slide(self, settings, tmp_path):
        deck = make_deck(tmp_path / "a.pptx", slides=4)
        target = tmp_path / "out" / "a.pdf"
        assert asyncio.run(office.render_pdf(settings, deck, target)) == 4
        with pymupdf.open(target) as pdf:
            text = pdf[2].get_text()
        assert "Slide title 3" in text and "Grouped text 3" in text
        assert not list(target.parent.glob(".*.part"))

    def test_hidden_slides_still_get_their_page(self, settings, tmp_path):
        deck = make_deck(tmp_path / "a.pptx", slides=3)
        loaded = Presentation(deck)
        loaded.slides[1]._element.set("show", "0")
        loaded.save(deck)
        assert asyncio.run(office.render_pdf(settings, deck, tmp_path / "a.pdf")) == 3
        slides = asyncio.run(office.read_slides(settings, deck))["slides"]
        assert [s["hidden"] for s in slides] == [False, True, False]

    def test_slides_text_groups_and_notes(self, settings, tmp_path):
        deck = make_deck(tmp_path / "a.pptx", slides=2)
        read = asyncio.run(office.read_slides(settings, deck))
        assert len(read["slides"]) == 2
        second = read["slides"][1]
        assert second["slide"] == 2 and second["title"] == "Slide title 2"
        assert "Grouped text 2" in second["lines"]
        assert second["notes"] == "Speaker notes for slide 2"

    def test_a_word_file_renders(self, settings, tmp_path):
        from docx import Document
        document = Document()
        document.add_heading("A heading", 1)
        document.add_paragraph("Some body text.")
        document.save(tmp_path / "a.docx")
        assert asyncio.run(office.render_pdf(settings, tmp_path / "a.docx", tmp_path / "a.pdf")) == 1
        words = asyncio.run(office.source_words(settings, tmp_path / "a.docx"))
        assert words == ["A", "heading", "Some", "body", "text."]

    def test_a_broken_file_is_an_error_not_a_crash(self, settings, tmp_path):
        path = zip_with(tmp_path / "broken.pptx", {"[Content_Types].xml": b"<nonsense"})
        with pytest.raises(office.OfficeError):
            asyncio.run(office.render_pdf(settings, path, tmp_path / "b.pdf"))
        with pytest.raises(office.OfficeError):
            asyncio.run(office.read_slides(settings, path))
        assert not (tmp_path / "b.pdf").exists()


@needs_image
class TestHostileFiles:
    def test_a_macro_runs_when_allowed(self, settings, tmp_path):
        """The control: this deck's macro is live, and LibreOffice runs it if let."""
        deck = make_macro_deck(tmp_path / "macro.pptx")
        assert "pwned" in open_in_container(settings, deck, MACROS_ALLOWED)

    def test_our_profile_stops_the_macro_even_on_a_plain_open(self, settings, tmp_path):
        deck = make_macro_deck(tmp_path / "macro.pptx")
        assert "pwned" not in open_in_container(settings, deck, OUR_PROFILE)

    def test_rendering_runs_no_macro(self, settings, tmp_path):
        deck = make_macro_deck(tmp_path / "macro.pptx")
        with work_folder() as work:
            (work / "out").mkdir()
            shutil.copy(deck, work / "in.pptx")
            done = asyncio.run(run_container(settings, work, ["python", office.SCRIPT, "render", "in.pptx"],
                                             60, office.office_image(settings)))
            made = {path.name for path in (work / "out").iterdir()}
        assert '"ok": true' in done.stdout
        assert made == {"render.pdf"}

    def test_no_network_from_the_office_image(self, settings):
        code = ("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
                "    print('CONNECTED')\nexcept OSError as e:\n    print('BLOCKED', e)\n")
        with work_folder() as work:
            (work / "net.py").write_text(code)
            done = asyncio.run(run_container(settings, work, ["python", "/work/net.py"], 20,
                                             office.office_image(settings)))
        assert "BLOCKED" in done.stdout and "CONNECTED" not in done.stdout

    def test_linked_images_and_objects_fetch_nothing_and_finish(self, settings, tmp_path):
        """A picture linked to a web address and one to a local file: the render finishes
        quickly, with nothing fetched (no network) and no file outside the container read."""
        deck = make_deck(tmp_path / "linked.pptx", slides=1)
        loaded = Presentation(deck)
        slide = loaded.slides[0]
        rels = slide.part.rels
        for target in ("http://203.0.113.7/secret.png", "file:///etc/hostname"):
            rid = rels.get_or_add_ext_rel(
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", target)
            picture = slide.shapes._spTree.add_pic(len(slide.shapes) + 10, "linked", "", rid, 0, 0, 914400, 914400)
            blip = picture.blipFill.find("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
            blip.attrib.pop("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", None)
            blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}link", rid)
        loaded.save(deck)
        target = tmp_path / "linked.pdf"
        assert asyncio.run(office.render_pdf(settings, deck, target)) == 1
        with pymupdf.open(target) as pdf:
            assert pdf[0].get_images() == []

    def test_fonts_folder_is_read_only_inside(self, settings):
        (settings.office_fonts_dir / "f.ttf").write_bytes(b"font")
        code = ("import pathlib\np = pathlib.Path('/opt/fonts/windows')\nprint('SEEN', sorted(x.name for x in p.iterdir()))\n"
                "try:\n    (p / 'new').write_text('x')\n    print('WROTE')\nexcept OSError:\n    print('REFUSED')\n"
                "pathlib.Path('/work/ok').write_text('x')\nprint('WORK OK')\n")
        with work_folder() as work:
            (work / "f.py").write_text(code)
            done = asyncio.run(run_container(settings, work, ["python", "/work/f.py"], 20,
                                             office.office_image(settings)))
        assert "SEEN ['f.ttf']" in done.stdout
        assert "REFUSED" in done.stdout and "WROTE" not in done.stdout and "WORK OK" in done.stdout
        assert not (settings.office_fonts_dir / "new").exists()

    def test_a_bomb_that_lies_about_its_size_is_stopped(self, settings, tmp_path):
        """1.5 GB of zeros in one slide, the directory claiming 1 KB: the size check passes,
        so the container's limits have to hold. Either way it ends as an error, in time."""
        zeros = b"<p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main'>" + b"\0" * (1536 * 2**20)
        deck = make_deck(tmp_path / "base.pptx", slides=1)
        with zipfile.ZipFile(deck) as archive:
            entries = {info.filename: archive.read(info) for info in archive.infolist()}
        entries["ppt/slides/slide1.xml"] = zeros
        path = zip_with(tmp_path / "bomb.pptx", entries, declared={"ppt/slides/slide1.xml": 1024})
        assert path.stat().st_size < 10 * 2**20
        office.precheck(path, settings)
        quick = Settings(**{**settings.__dict__, "office_seconds": 60})
        with pytest.raises(office.OfficeError):
            asyncio.run(office.render_pdf(quick, path, tmp_path / "bomb.pdf"))
        assert not (tmp_path / "bomb.pdf").exists()
        assert not running_containers()

    def test_the_time_limit_kills_a_long_render(self, settings, tmp_path, monkeypatch):
        """LibreOffice alone takes over a second to start, so a 1 s limit with no grace
        always runs out: the container is killed and nothing is left behind."""
        from app.tools import sandbox
        monkeypatch.setattr(sandbox, "GRACE_SECONDS", 0)
        deck = make_deck(tmp_path / "a.pptx", slides=40)
        slow = Settings(**{**settings.__dict__, "office_seconds": 1})
        with pytest.raises(office.OfficeError, match="took longer than 1 s"):
            asyncio.run(office.render_pdf(slow, deck, tmp_path / "a.pdf"))
        assert not (tmp_path / "a.pdf").exists()
        assert not running_containers()


def running_containers() -> list[str]:
    import time
    for _ in range(20):         # docker kill is started, not awaited
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                               text=True).stdout.split()
        left = [name for name in names if name.startswith("study-sandbox-")]
        if not left:
            return []
        time.sleep(0.5)
    return left


@needs_image
class TestPreviewEndpoint:
    @pytest.fixture
    def client(self, settings, tmp_path):
        from fastapi.testclient import TestClient
        from app.main import create_app
        settings.outputs_dir.mkdir(parents=True)
        settings.library_dir.mkdir(parents=True)
        return TestClient(create_app(settings, frontend=tmp_path / "no-frontend"), base_url="http://localhost")

    def test_renders_once_and_keeps_it(self, client, settings):
        make_deck(settings.outputs_dir / "deck.pptx", slides=2)
        first = client.get("/api/outputs/deck.pptx/preview")
        assert first.status_code == 200
        assert first.headers["content-type"] == "application/pdf"
        assert first.headers["x-content-type-options"] == "nosniff"
        kept = list(settings.renders_dir.glob("*.pdf"))
        assert len(kept) == 1
        with pymupdf.open(stream=first.content, filetype="pdf") as pdf:
            assert pdf.page_count == 2
        stamp = kept[0].stat().st_mtime_ns
        assert client.get("/api/outputs/deck.pptx/preview").content == first.content
        assert kept[0].stat().st_mtime_ns == stamp         # served from the cache

    def test_a_changed_file_gets_a_new_render(self, client, settings):
        make_deck(settings.outputs_dir / "deck.pptx", slides=2)
        client.get("/api/outputs/deck.pptx/preview")
        make_deck(settings.outputs_dir / "deck.pptx", slides=3)
        again = client.get("/api/outputs/deck.pptx/preview")
        with pymupdf.open(stream=again.content, filetype="pdf") as pdf:
            assert pdf.page_count == 3
        assert len(list(settings.renders_dir.glob("*.pdf"))) == 2

    def test_only_word_and_powerpoint_outputs(self, client, settings):
        (settings.outputs_dir / "a.pdf").write_bytes(b"%PDF-1.7")
        (settings.outputs_dir / "a.md").write_text("# x")
        for name in ("a.pdf", "a.md", "missing.pptx", "..%2Fsecret.pptx", ".hidden.pptx"):
            assert client.get(f"/api/outputs/{name}/preview").status_code == 404, name
        library_deck = make_deck(settings.library_dir / "lib.pptx")
        (settings.outputs_dir / "link.pptx").symlink_to(library_deck)
        assert client.get("/api/outputs/link.pptx/preview").status_code == 404

    def test_a_file_that_wont_render_is_a_422_with_the_reason(self, client, settings):
        zip_with(settings.outputs_dir / "broken.pptx", {"[Content_Types].xml": b"<nonsense"})
        response = client.get("/api/outputs/broken.pptx/preview")
        assert response.status_code == 422
        assert "broken.pptx" in response.json()["detail"]
        assert not list(settings.renders_dir.glob("*.pdf"))


OMML_ABS = ('<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath>'
            '<m:r><m:t>|</m:t></m:r><m:r><m:t>x</m:t></m:r><m:r><m:t>|</m:t></m:r>'
            '<m:r><m:t>=</m:t></m:r><m:r><m:t>y</m:t></m:r></m:oMath></m:oMathPara>')


def make_word_with_equation(path: Path) -> Path:
    """A Word file with |x| = y as pandoc writes it: bare bars in OMML runs."""
    from docx import Document
    from lxml import etree
    document = Document()
    document.add_paragraph("Before the equation.")
    paragraph = document.add_paragraph()
    paragraph._p.append(etree.fromstring(OMML_ABS))
    document.add_paragraph("After the equation.")
    document.save(path)
    return path


@needs_image
class TestWordEquations:
    def test_equations_render_and_bars_are_not_errors(self, settings, tmp_path):
        """Without LibreOffice Math, Word's equations vanished; with it, a bare | became the
        error marker ¿. Both found Nov 6, 2026 against Word's own PDF."""
        source = make_word_with_equation(tmp_path / "eq.docx")
        before = source.read_bytes()
        target = tmp_path / "eq.pdf"
        assert asyncio.run(office.render_pdf(settings, source, target)) == 1
        with pymupdf.open(target) as pdf:
            text = pdf[0].get_text()
        assert "Before the equation." in text and "After the equation." in text
        assert "¿" not in text
        assert "x" in text and "y" in text and "=" in text
        assert source.read_bytes() == before        # only the copy in the container changed


@needs_image
def test_slide_text_comes_in_reading_order_not_drawing_order(settings, tmp_path):
    """Labels over values, drawn values first and right to left (as on WE-DPG26's slide 2,
    whose drawing order paired the labels with the wrong lengths): read row by row, left to
    right, with boxes a little out of line still one row."""
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for text, left, top in [("10 km arm length", 6, 3.05), ("15-20 km arm length", 2, 3.0),
                            ("Triangular", 6, 2.0), ("2 L-Shaped", 2, 2.04), ("Heading", 1, 0.5)]:
        slide.shapes.add_textbox(Inches(left), Inches(top), Inches(3), Inches(0.5)).text_frame.text = text
    deck.save(tmp_path / "order.pptx")
    lines = asyncio.run(office.read_slides(settings, tmp_path / "order.pptx"))["slides"][0]["lines"]
    assert lines == ["Heading", "2 L-Shaped", "Triangular", "15-20 km arm length", "10 km arm length"]
