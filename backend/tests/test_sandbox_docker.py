"""The sandbox for real. Needs Docker and scripts/build_sandbox.sh; skipped otherwise.

Each wall is tested from inside a container. Where a test shows something fails, a
control beside it shows the same thing working when it's allowed, so a pass can't just
mean the attempt was broken.
"""
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import replace

import fitz
import pytest

from app.config import Settings
from app.tools.executor import Pending, ToolBox
from app.tools.sandbox import run_container, work_folder

IMAGE = Settings().sandbox_image
CANARY = "CANARY-sk-not-a-real-key-0123456789"

# This checkout's own .env, wherever it lives: what an attack would try to read.
ENV_FILE = str(Path(__file__).resolve().parents[2] / ".env")
HOME = str(Path.home())


def image_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    done = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return done.returncode == 0


pytestmark = pytest.mark.skipif(not image_ready(), reason=f"needs Docker and the {IMAGE} image")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library",
                    outputs_dir=tmp_path / "outputs")


def load_run_tools():
    """eval/run_tools.py, whose checks the tool tasks use."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_tools", Path(__file__).resolve().parents[2] / "eval" / "run_tools.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_tools", module)     # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def box_for(settings):
    return ToolBox(settings, lambda query: [], {}, [])


def inside(settings, code, seconds=20):
    """Run Python in the container, the same way run_python does."""
    with work_folder() as work:
        (work / "main.py").write_text(code)
        return asyncio.run(run_container(settings, work, ["python", "-u", "/work/main.py"], seconds))


def run(settings, code):
    return asyncio.run(box_for(settings).call("run_python", {"code": code}))


def build(settings, tool, content, name="out"):
    return asyncio.run(box_for(settings).build(Pending(tool, name), content))


class TestWalls:
    def test_there_is_no_network(self, settings):
        done = inside(settings, """
import socket
for address in (("1.1.1.1", 53), ("127.0.0.1", 11434), ("host.docker.internal", 80)):
    try:
        socket.create_connection(address, timeout=2)
        print("OPEN", address)
    except OSError:
        print("blocked")
""")
        assert done.stdout.count("blocked") == 3 and "OPEN" not in done.stdout

    def test_only_loopback_exists(self, settings):
        assert inside(settings, "import os; print(sorted(os.listdir('/sys/class/net')))").stdout.strip() == "['lo']"

    def test_the_project_and_windows_arent_visible(self, settings):
        done = inside(settings, f"""
import pathlib
for name in ({HOME!r}, {ENV_FILE!r}, "/mnt/c", "/mnt/wsl", "/var/run/docker.sock"):
    print("VISIBLE" if pathlib.Path(name).exists() else "gone", name)
""")
        assert "VISIBLE" not in done.stdout and done.stdout.count("gone") == 5

    def test_only_work_and_tmp_are_writable(self, settings):
        done = inside(settings, """
for folder in ("/", "/usr", "/opt/sandbox", "/etc", "/var"):
    try:
        open(folder + "/x", "w")
        print("WROTE", folder)
    except OSError:
        print("refused")
open("/work/ok", "w"); open("/tmp/ok", "w"); print("fine")
""")
        assert "WROTE" not in done.stdout and "fine" in done.stdout

    def test_it_runs_unprivileged(self, settings):
        done = inside(settings, """
import os
print(os.getuid() != 0, open("/proc/self/status").read().split("CapEff:")[1].split()[0], flush=True)
""")
        assert done.stdout.split() == ["True", "0000000000000000"]

    def test_the_environment_holds_no_secret(self, settings, monkeypatch):
        monkeypatch.setenv("STUDY_ASSISTANT_ANTHROPIC_KEY", CANARY)
        assert CANARY not in inside(settings, "import os; print(dict(os.environ))").stdout

    def test_memory_is_limited(self, settings):
        settings.sandbox_memory = "256m"
        outcome = run(settings, "x = bytearray(1024 ** 3)\nprint('allocated', len(x), x[-1])")
        assert "allocated" not in outcome.detail["stdout"] and outcome.detail["exit_code"] != 0

    def test_a_fork_bomb_stops_at_the_process_limit(self, settings):
        settings.sandbox_pids = 32
        outcome = run(settings, """
import os, time
count = 0
try:
    for _ in range(500):
        if os.fork() == 0:
            time.sleep(5)
            os._exit(0)
        count += 1
except OSError:
    pass
print("forked", count)
""")
        count = int(outcome.detail["stdout"].split()[-1])
        assert count < 32

    def test_a_file_cant_outgrow_the_limit(self, settings):
        settings.sandbox_file_bytes = 1024 * 1024
        outcome = run(settings, """
try:
    with open("big.txt", "wb") as out:
        for _ in range(10):
            out.write(b"x" * 1024 * 512)
    print("WROTE ALL")
except OSError as error:
    print("stopped")
""")
        assert "WROTE ALL" not in outcome.detail["stdout"]


class TestWorkFolderSize:
    """Review S6: each file was capped, the folder (on the host's /tmp) was not. Run for real."""

    WRITER = """
import os
os.makedirs("/work/fill", exist_ok=True)
for n in range(100):
    with open(f"/work/fill/{n}.bin", "wb") as f:
        f.write(os.urandom(10 * 2**20))      # 10 MB each, well under the per-file limit
print("wrote it all")
"""

    def test_many_files_under_the_per_file_limit_are_stopped(self, settings):
        small = replace(settings, sandbox_work_bytes=50 * 2**20)
        with work_folder() as work:
            (work / "main.py").write_text(self.WRITER)
            done = asyncio.run(run_container(small, work, ["python", "-u", "/work/main.py"], 60))
            left = sum(path.stat().st_size for path in work.rglob("*") if path.is_file())
        assert done.too_big and "wrote it all" not in done.stdout and "stopped: the run wrote more" in done.stderr
        assert left < 400 * 2**20         # killed early, not after the whole 1 GB

    def test_control_a_small_run_isnt_stopped(self, settings):
        small = replace(settings, sandbox_work_bytes=50 * 2**20)
        done = inside(small, "open('/work/a.bin', 'wb').write(b'x' * 2**20); print('ok')")
        assert not done.too_big and done.stdout.strip() == "ok"


class TestCopyOut:
    """Attacks on the way files leave the container (review S1, S2), run for real."""

    def test_out_swapped_for_a_link_to_a_host_folder_leaks_nothing(self, settings, tmp_path):
        private = tmp_path / "private"
        private.mkdir()
        (private / "loot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + CANARY.encode())
        settings.outputs_dir.mkdir()
        outcome = run(settings, f"""
import os, shutil
os.chdir("/work")
shutil.rmtree("out")
os.symlink({str(private)!r}, "out")
print("swapped")
""")
        assert outcome.detail["stdout"].strip() == "swapped"
        assert outcome.detail["refused"] == ["out"] and outcome.files == []
        assert "loot.png" not in outcome.text
        assert not any(CANARY.encode() in path.read_bytes() for path in settings.outputs_dir.rglob("*") if path.is_file())

    def test_control_a_real_plot_in_out_is_copied(self, settings):
        settings.outputs_dir.mkdir()
        outcome = run(settings, "open('/work/out/plot.png', 'wb').write(b'\\x89PNG\\r\\n\\x1a\\n')")
        assert [item["kind"] for item in outcome.files] == ["png"]

    def test_a_pipe_left_in_out_is_refused_and_nothing_hangs(self, settings):
        settings.outputs_dir.mkdir()

        async def bounded():
            return await asyncio.wait_for(box_for(settings).call("run_python", {"code": (
                "import os\nos.mkfifo('/work/out/plot.png')\nprint('made')")}), 60)
        outcome = asyncio.run(bounded())
        assert outcome.detail["refused"] == ["plot.png"] and outcome.files == []


class TestRunPython:
    def test_output_comes_back(self, settings):
        outcome = run(settings, "print('hello', 6 * 7)")
        assert outcome.ok and outcome.detail["stdout"].strip() == "hello 42"

    def test_the_libraries_are_there(self, settings):
        outcome = run(settings, "import numpy, scipy, sympy, matplotlib; print('ok')")
        assert outcome.ok

    def test_a_traceback_shows_only_the_readers_code(self, settings):
        outcome = run(settings, "1 / 0")
        assert not outcome.ok and "ZeroDivisionError" in outcome.text and "run_user" not in outcome.text

    def test_plt_show_is_caught(self, settings):
        outcome = run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()")
        assert outcome.ok and [item["kind"] for item in outcome.files] == ["png"]

    def test_a_figure_never_shown_is_caught(self, settings):
        assert len(run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])").files) == 1

    def test_savefig_is_caught_once(self, settings):
        code = "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.savefig('mine.png')\nplt.close()"
        assert [item["name"].split("-")[0] for item in run(settings, code).files] == ["mine"]

    def test_savefig_then_show_makes_one_file_not_two(self, settings):
        code = "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.savefig('mine.png')\nplt.show()"
        assert [item["name"].split("-")[0] for item in run(settings, code).files] == ["mine"]

    def test_the_time_limit_keeps_the_output_so_far(self, settings):
        settings.run_seconds = 3
        outcome = run(settings, "import time\nprint('started', flush=True)\nwhile True:\n    time.sleep(0.1)")
        assert outcome.detail["timed_out"] and "started" in outcome.detail["stdout"]
        assert outcome.detail["seconds"] < 3 + 5

    def test_a_container_that_ignores_its_timer_is_still_stopped(self, settings):
        with work_folder() as work:
            (work / "main.py").write_text("import time\ntime.sleep(60)")
            done = asyncio.run(run_container(settings, work, ["python", "/work/main.py"], 1))
        assert done.timed_out and done.seconds < 10

    def test_endless_printing_is_capped(self, settings):
        settings.run_seconds = 2
        outcome = run(settings, "while True:\n    print('x' * 1000)")
        assert len(outcome.detail["stdout"]) < 2 * settings.sandbox_output_bytes
        assert "bytes cut" in outcome.detail["stdout"]


class TestFiles:
    MATH = "# Title\n\nInline $x^2$ and\n\n$$\\boldsymbol{A} = \\begin{bmatrix} a & b \\end{bmatrix}$$\n"

    def test_a_pdf_with_typeset_math(self, settings):
        outcome = build(settings, "make_pdf", self.MATH)
        assert outcome.ok, outcome.text
        assert outcome.files[0]["pages"] == 1
        with fitz.open(settings.outputs_dir / "out.pdf") as document:
            assert "$$" not in document[0].get_text()

    def test_greek_and_umlauts_in_prose(self, settings):
        outcome = build(settings, "make_pdf", "Größe λ äöü")
        assert outcome.ok and outcome.detail["missing_characters"] == []

    def test_a_tex_error_comes_back_to_the_model(self, settings):
        outcome = build(settings, "make_pdf", "$$\\undefinedmacro{x}$$")
        assert not outcome.ok and "Undefined" in outcome.text

    def test_a_word_file_has_native_equations(self, settings):
        import docx

        outcome = build(settings, "make_docx", self.MATH)
        assert outcome.ok, outcome.text
        body = docx.Document(settings.outputs_dir / "out.docx").element.body
        assert len(body.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")) == 2

    def test_a_plot_goes_into_word_under_pandocs_sandbox(self, settings):
        plot = run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()").files[0]["name"]
        outcome = build(settings, "make_docx", f"# T\n\n![A plot]({plot})\n")
        assert outcome.ok, outcome.text

    def test_a_deck_with_equations_and_a_plot(self, settings):
        from pptx import Presentation

        plot = run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()").files[0]["name"]
        outcome = build(settings, "make_pptx",
                        f"## One\n- First\n$$ A = LL^\\top $$\n## Two\n![Plot]({plot})\nNotes: Say this\n")
        assert outcome.ok, outcome.text
        deck = Presentation(settings.outputs_dir / "out.pptx")
        assert len(deck.slides) == 2

    def test_words_and_every_math_alphabet_work_in_slide_equations(self, settings):
        """The first deck to use \\text failed: its font wasn't in the offline cache."""
        outcome = build(settings, "make_pptx", (
            "## One\n$$ \\text{update: } x_{\\text{new}} = \\mathbf{b} + \\mathsf{s} + \\mathtt{t} "
            "+ \\mathfrak{f} + \\mathscr{c} $$\n"))
        assert outcome.ok, outcome.text

    def test_the_same_alphabets_work_in_a_pdf(self, settings):
        outcome = build(settings, "make_pdf", (
            "$$ \\text{update: } x_{\\text{new}} = \\mathbf{b} + \\mathsf{s} + \\mathtt{t} "
            "+ \\mathfrak{f} + \\mathscr{c} $$\n"))
        assert outcome.ok, outcome.text

    def test_math_works_wherever_the_font_size_changes(self, settings):
        """A formula in a heading failed: TeX wanted a size of a font the cache didn't have."""
        headings = "\n\n".join(f"{'#' * level} Level $x^2_i$ and **bold $y$**" for level in range(1, 6))
        outcome = build(settings, "make_pdf", (
            f"{headings}\n\nA note.[^1]\n\n[^1]: With $\\sum_i x_i$ inside.\n\n"
            "| $a$ | **$b$** |\n|-----|---------|\n| $c$ | $d$ |\n"))
        assert outcome.ok, outcome.text

    def test_a_themed_deck_uses_its_layouts_and_animates_a_derivation(self, settings):
        from pptx import Presentation

        outcome = build(settings, "make_pptx", (
            "## title: A deck\nSubtitle: A line\n"
            "## big_equation: Steps\n$$\n\\begin{aligned} a &= b \\\\ &= c \\\\ &= d \\end{aligned}\n$$\nAnimate: yes\n"
            "## summary: End\n- One\n"))
        assert outcome.ok, outcome.text
        slides = Presentation(settings.outputs_dir / "out.pptx").slides
        assert [slide.slide_layout.name for slide in slides] == ["title", "big_equation", "summary"]
        rows = [shape for shape in slides[1].shapes if shape.name.startswith("equation")]
        assert len(rows) == 3
        assert len({shape.left for shape in rows}) == 1             # the = signs stay lined up
        assert [shape.top for shape in rows] == sorted(shape.top for shape in rows)

    def test_equation_pictures_are_transparent(self, settings):
        from io import BytesIO

        from PIL import Image
        from pptx import Presentation

        build(settings, "make_pptx", "## big_equation: T\n$$ a = b $$\n")
        shape = next(s for s in Presentation(settings.outputs_dir / "out.pptx").slides[0].shapes
                     if s.name == "equation 1")
        with Image.open(BytesIO(shape.image.blob)) as image:
            assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0

    def test_a_capital_subscript_reaches_the_slide_as_real_subscript(self, settings):
        from pptx import Presentation

        outcome = build(settings, "make_pptx", "## summary: The Z peak\n- Near $M_Z$, the width $\\Gamma_Z$ matters\n")
        assert outcome.ok, outcome.text
        slide = Presentation(settings.outputs_dir / "out.pptx").slides[0]
        body = next(shape for shape in slide.placeholders if shape.placeholder_format.idx != 0)
        runs = body.text_frame.paragraphs[0].runs
        scripted = [run.text for run in runs if run._r.get_or_add_rPr().get("baseline") == "-25000"]
        assert scripted == ["Z", "Z"]
        assert "z" not in "".join(scripted)

    def test_a_bad_equation_is_named(self, settings):
        outcome = build(settings, "make_pptx", "## One\n$$ a $$\n$$ \\undefinedmacro $$\n")
        assert not outcome.ok and "equation 2" in outcome.text


class TestTemplates:
    TABLE = "# Heading\n\nText.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"

    def test_word_files_carry_the_theme(self, settings):
        import docx

        outcome = build(settings, "make_docx", self.TABLE)
        assert outcome.ok, outcome.text
        document = docx.Document(settings.outputs_dir / "out.docx")
        heading = document.styles["Heading 1"]
        assert heading.font.name == "Calibri" and str(heading.font.color.rgb) == "0F766E"
        assert document.styles["Body Text"].font.name == "Cambria"
        assert round(document.sections[0].left_margin.cm, 1) == 2.3
        xml = document.styles["Table"].element.xml
        assert "firstRow" in xml and "0F766E" in xml

    def test_a_pdf_builds_with_no_table_too(self, settings):
        """pandoc loads table packages only for tables, and the theme once relied on them."""
        outcome = build(settings, "make_pdf", "# Heading\n\nJust text.\n")
        assert outcome.ok, outcome.text

    def test_the_colour_of_a_quote_doesnt_leak_into_what_follows(self, settings):
        build(settings, "make_pdf", "> A quote.\n\nAfter it.\n")
        with fitz.open(settings.outputs_dir / "out.pdf") as document:
            spans = [span for block in document[0].get_text("dict")["blocks"] for line in block.get("lines", [])
                     for span in line["spans"]]
        after = next(span for span in spans if "After" in span["text"])
        assert after["color"] == 0

    def test_no_table_row_can_split_across_pages(self, settings):
        import docx

        rows = "\n".join(f"| {n} | {'word ' * 40} |" for n in range(30))
        outcome = build(settings, "make_docx", f"| A | B |\n|---|---|\n{rows}\n")
        assert outcome.ok, outcome.text
        table = docx.Document(settings.outputs_dir / "out.docx").tables[0]
        cant_split = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cantSplit"
        assert all(row._tr.find(f".//{cant_split}") is not None for row in table.rows)

    def test_word_subscripts_in_a_document_are_upright(self, settings):
        import docx

        build(settings, "make_docx", "$$\\sigma_{had}$$\n")
        xml = docx.Document(settings.outputs_dir / "out.docx").element.body.xml
        assert "had" in xml and "m:sty" in xml          # Word's mark for upright text in a formula

    def test_a_table_in_word_uses_that_style(self, settings):
        import docx

        build(settings, "make_docx", self.TABLE)
        table = docx.Document(settings.outputs_dir / "out.docx").tables[0]
        assert table.style.name == "Table"

    def test_pdf_headings_use_the_theme_font_and_tables_get_a_bold_head(self, settings):
        outcome = build(settings, "make_pdf", self.TABLE)
        assert outcome.ok, outcome.text
        with fitz.open(settings.outputs_dir / "out.pdf") as document:
            fonts = {font[3].split("+")[-1] for font in document[0].get_fonts()}
        assert any("Carlito-Bold" in font for font in fonts), fonts

    def test_the_template_doesnt_open_the_door_to_files(self, settings):
        """pandoc reads the reference document itself, but a document still can't name others."""
        outcome = build(settings, "make_docx", "![x](/opt/sandbox/reference.docx)\n")
        assert not outcome.ok


class TestReadingFiles:
    """The attack works when the target is reachable, and fails only because it isn't."""

    def test_control_tex_really_can_read_a_file_through_math(self, settings):
        with work_folder() as work:
            (work / "secret.tex").write_text(CANARY)
            (work / "doc.md").write_text("$$\\text{\\input{/work/secret.tex}}$$\n")
            done = asyncio.run(run_container(settings, work, ["python", "/opt/sandbox/build.py", "pdf"], 60))
            assert json.loads(done.stdout.strip().splitlines()[-1])["ok"] is True
            with fitz.open(work / "out" / "out.pdf") as document:
                assert CANARY in "".join(page.get_text() for page in document).replace("\n", "")

    def test_control_a_picture_in_the_work_folder_really_does_reach_a_pdf(self, settings):
        plot = run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()").files[0]["name"]
        outcome = build(settings, "make_pdf", f"# T\n\n![A plot]({plot})\n")
        assert outcome.ok, outcome.text
        with fitz.open(settings.outputs_dir / "out.pdf") as document:
            assert document[0].get_images()

    def test_a_list_under_a_sentence_is_a_list_in_the_pdf_not_asterisks(self, settings):
        """The bug report's PDF read "...ablations [S1][S2]: * Multi-token prediction (MTP): ..."."""
        outcome = build(settings, "make_pdf", "# S\n\nThree components:\n* **MTP:** first\n* **Head:** second\n")
        assert outcome.ok, outcome.text
        with fitz.open(settings.outputs_dir / "out.pdf") as document:
            text = document[0].get_text()
        assert "* MTP" not in text and "*MTP" not in text and "MTP:" in text
        assert "•" in text      # LaTeX's bullet

    def test_a_source_figure_reaches_the_pdf_as_its_original_pixels(self, settings):
        """The bug report of Nov 7: a figure from the paper goes in as its crop, not a redraw."""
        from app.retrieval.search import Hit

        crop = settings.parsed_dir / "abc" / "figures" / "p13-1.png"
        crop.parent.mkdir(parents=True)
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), False)
        for x in range(120):                # a pattern a redraw wouldn't reproduce
            for y in range(80):
                pixmap.set_pixel(x, y, ((x * 7) % 256, (y * 11) % 256, (x * y) % 256))
        pixmap.save(str(crop))
        figure = {"figure_id": 176, "chunk_id": 1, "page": 13, "caption": "Figure 5: Phase error.",
                  "image_path": "figures/p13-1.png", "sha256": "abc", "url": "", "score": None}
        hit = Hit(chunk_id=1, doc_id=1, doc_path="paper.pdf", doc_title="Paper", heading_path=[], text="Text",
                  page_start=13, page_end=13, block_ids=[], meta={}, score=1.0, figures=[figure])
        box = ToolBox(settings, lambda query: [], {}, [hit])
        outcome = asyncio.run(box.build(Pending("make_pdf", "summary"), "# S\n\n![Figure 5](source-figure-176.png)\n"))
        assert outcome.ok, outcome.text
        checks = load_run_tools()
        assert checks.embeds_original_figure(settings.outputs_dir / "summary.pdf", [crop])
        plot = run(settings, "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.show()").files[0]["name"]
        assert build(settings, "make_pdf", f"# T\n\n![A redraw]({plot})\n", name="redraw").ok
        assert not checks.embeds_original_figure(settings.outputs_dir / "redraw.pdf", [crop])     # the control

    @pytest.mark.parametrize("target", [
        ENV_FILE,
        ".." + ENV_FILE,
        "/proc/1/environ",
    ])
    def test_a_file_outside_the_work_folder_cant_be_included(self, settings, target):
        outcome = build(settings, "make_pdf", f"$$\\text{{\\input{{{target}}}}}$$\n")
        assert not outcome.ok
        assert not list(settings.outputs_dir.glob("*")) if settings.outputs_dir.exists() else True

    def test_tex_cant_run_commands(self, settings):
        outcome = build(settings, "make_pdf", "$$\\text{\\immediate\\write18{touch /work/pwned}}$$\n")
        assert "pwned" not in outcome.text

    def test_yaml_metadata_cant_reach_the_template(self, settings):
        with work_folder() as work:
            (work / "doc.md").write_text("---\nheader-includes: '\\input{/nonexistent}'\n---\n\nText\n")
            done = asyncio.run(run_container(settings, work, ["python", "/opt/sandbox/build.py", "pdf"], 60))
            assert json.loads(done.stdout.strip().splitlines()[-1])["ok"] is True


class TestVerifyClaims:
    """Roadmap item 3: the claims block is read and checked only inside the container."""

    def test_a_block_is_checked_in_the_container(self, settings):
        from app.verify import check_block
        block = ("let sig = 1/(1 + exp(-f))\ndiff(sig, f) = sig*(1 - sig)\n"
                 "let M_W = 80.38 [GeV]\nlet G_F = 1.1664e-5 [GeV^-2]\n"
                 "G_F*M_W**3/(6*sqrt(2)*pi) ≈ 0.227 [GeV]\n3*G_F*M_W**3/(6*sqrt(2)*pi) = 2.05 [GeV]\n")
        report = asyncio.run(check_block(settings, block))
        assert report.failure is None
        assert [claim.verdict for claim in report.claims] == ["passed", "passed", "failed"]
        assert report.badge() == "2 of 3 checked claims passed"

    FORGED = ("1 = __import__('os')._exit(print('{\"claims\": ' + chr(91) + '{\"line\": \"2 + 2 = 5\", "
              "\"verdict\": \"passed\"}' + chr(93) + '}', flush=True) or 0)")

    def test_a_claim_cant_forge_the_report(self, settings):
        """Review S3's attack: print a report saying a false claim passed, then exit."""
        from app.verify import check_block
        report = asyncio.run(check_block(settings, f"2 + 2 = 5\n{self.FORGED}\n"))
        assert report.failure is None
        assert [claim.verdict for claim in report.claims] == ["failed", "error"]
        assert "double underscores" in report.claims[1].note
        assert report.badge() != "1 of 1 checked claim passed" and report.passed == 0

    @pytest.mark.parametrize("claim", [
        "print(srepr(x)) = 1",                         # printers aren't there: an unknown function
        "sin(latex(Symbol(x))) = 0",
        "exit(0) = 1",
        "x.func.mro() = 1",
        "sympify(x) = x",
    ])
    def test_without_underscores_nothing_runs_either(self, settings, claim):
        with work_folder() as work:
            (work / "checks.txt").write_text(f"{claim}\n2 + 2 = 5\n")
            (work / "nonce").write_text("n0")
            done = asyncio.run(run_container(settings, work, ["python", "/opt/sandbox/verify_claims.py"], 60))
        lines = done.stdout.strip().splitlines()
        assert len(lines) == 1, done.stdout                 # nothing printed but the report
        result = json.loads(lines[0])
        assert result["nonce"] == "n0"
        assert result["claims"][-1]["verdict"] == "failed"
        assert result["claims"][0]["verdict"] != "passed"

    def test_sympy_check_cant_be_forged_either(self, settings):
        outcome = asyncio.run(box_for(settings).call("sympy_check", {"lhs": "2 + 2", "rhs": self.FORGED[4:]}))
        assert outcome.detail["verdict"] == "error" and "underscores" in outcome.detail["note"]

    def test_control_symbols_in_quotes_still_work(self, settings):
        from app.verify import check_block
        report = asyncio.run(check_block(settings, "let x = symbols('x')\ndiff(x**2, x) = 2*x\n"))
        assert [claim.verdict for claim in report.claims] == ["passed"]

    def test_code_in_a_claim_cant_reach_the_host(self, settings):
        from app.verify import check_block
        block = f"__import__('pathlib').Path({ENV_FILE!r}).read_text() = '{CANARY}'\n"
        report = asyncio.run(check_block(settings, block))
        assert report.failure is None
        assert report.claims[0].verdict != "passed"
        assert CANARY not in (report.claims[0].note or "") + (report.claims[0].left or "")
