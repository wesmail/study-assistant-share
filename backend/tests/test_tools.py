"""The tools without Docker: what reaches a container, what leaves it, and what the model wrote."""
import asyncio
import contextlib
import os
import signal
from pathlib import Path

import pytest

# This checkout's own .env, wherever it lives: what an attack would try to read.
ENV_FILE = str(Path(__file__).resolve().parents[2] / ".env")

from app.config import Settings
from app.retrieval.search import Hit
from app.tools import documents
from app.tools.documents import ContentError, finish_deck, parse_deck, theme, with_sources
from app.tools.executor import Pending, ToolBox, last_json_line, stem_of
from app.tools.sandbox import (
    CappedOutput, SandboxUnavailable, container_command, copy_out, docker_environment, safe_copy,
)
from app.tools.schemas import FILE_TOOLS, TOOLS, anthropic_tools, ollama_tools

PNG = b"\x89PNG\r\n\x1a\n"
SOURCE = {"id": 1, "doc_title": "Book A", "page_start": 118, "page_end": 119, "unverified_pages": []}
MML = {"id": 1, "doc_title": "Mathematics for Machine Learning", "author": "Marc Peter Deisenroth, A. Aldo Faisal, "
       "Cheng Soon Ong", "year": 2020, "publisher": "Cambridge University Press", "page_start": 117, "page_end": 120,
       "unverified_pages": []}
UDL = {"id": 2, "doc_title": "Understanding Deep Learning", "author": "Simon J. D. Prince", "year": 2024,
       "publisher": "MIT Press", "page_start": 290, "page_end": 293, "unverified_pages": []}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library",
                    outputs_dir=tmp_path / "outputs")


@contextlib.contextmanager
def no_longer_than(seconds: int):
    """Fail instead of hanging: the alarm interrupts a blocked system call."""
    def give_up(*_):
        raise TimeoutError(f"still blocked after {seconds} s")
    previous = signal.signal(signal.SIGALRM, give_up)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def box_for(settings, sources=None, previous=None, previous_sources=None):
    return ToolBox(settings, lambda query: [], sources or {}, [], previous, previous_sources)


def hit(number: int, text: str = "Words") -> Hit:
    return Hit(chunk_id=number, doc_id=1, doc_path="a.pdf", doc_title="Book A", heading_path=[], text=text,
               page_start=number, page_end=number, block_ids=[], meta={}, score=1.0, chunk_ids=[number])


def flag(command: list[str], name: str) -> list[str]:
    return [command[index + 1] for index, part in enumerate(command) if part == name]


class TestTheContainer:
    def command(self, settings, tmp_path):
        return container_command(settings, tmp_path, "name", ["python", "x.py"])

    def test_it_has_no_network(self, settings, tmp_path):
        assert flag(self.command(settings, tmp_path), "--network") == ["none"]

    def test_the_only_mount_is_the_work_folder(self, settings, tmp_path):
        command = self.command(settings, tmp_path)
        assert flag(command, "--mount") == [f"type=bind,source={tmp_path},target=/work"]
        assert "-v" not in command and "--volume" not in command and "--volumes-from" not in command

    def test_nothing_else_is_writable_or_privileged(self, settings, tmp_path):
        command = self.command(settings, tmp_path)
        assert "--read-only" in command
        assert flag(command, "--cap-drop") == ["ALL"]
        assert flag(command, "--security-opt") == ["no-new-privileges"]
        assert "--privileged" not in command and "--cap-add" not in command
        assert flag(command, "--user") == [f"{os.getuid()}:{os.getgid()}"]

    def test_it_has_cpu_memory_and_process_limits_and_no_swap(self, settings, tmp_path):
        command = self.command(settings, tmp_path)
        assert flag(command, "--memory") == flag(command, "--memory-swap") == [settings.sandbox_memory]
        assert flag(command, "--cpus") == [str(settings.sandbox_cpus)]
        assert flag(command, "--pids-limit") == [str(settings.sandbox_pids)]

    def test_no_environment_reaches_it_except_a_scratch_home(self, settings, tmp_path, monkeypatch):
        monkeypatch.setenv("STUDY_ASSISTANT_ANTHROPIC_KEY", "secret-value")
        command = self.command(settings, tmp_path)
        assert flag(command, "--env") == ["HOME=/tmp"]
        assert "--env-file" not in command
        assert "STUDY_ASSISTANT_ANTHROPIC_KEY" not in docker_environment()

    def test_the_command_comes_last_so_it_cant_add_docker_flags(self, settings, tmp_path):
        command = container_command(settings, tmp_path, "name", ["--privileged"])
        assert command.index(settings.sandbox_image) == len(command) - 2


def test_a_missing_docker_is_reported_and_runs_nothing(settings, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    box = box_for(settings)
    outcome = asyncio.run(box.call("run_python", {"code": "print(1)"}))
    assert (outcome.ok, outcome.summary) == (False, "Sandbox not available")
    assert "isn't installed" in outcome.text


class TestLeavingTheWorkFolder:
    def test_a_regular_file_is_copied(self, tmp_path):
        (tmp_path / "a.png").write_bytes(PNG)
        assert safe_copy(tmp_path / "a.png", tmp_path / "b.png", 100) == len(PNG)
        assert (tmp_path / "b.png").read_bytes() == PNG

    def test_a_link_to_a_secret_is_refused_not_followed(self, tmp_path):
        secret = tmp_path / "secret.env"
        secret.write_text("KEY=value")
        (tmp_path / "plot.png").symlink_to(secret)
        with pytest.raises(ValueError, match="not a regular file"):
            safe_copy(tmp_path / "plot.png", tmp_path / "copy.png", 100)
        assert not (tmp_path / "copy.png").exists()

    def test_an_ordinary_copy_would_have_leaked_it(self, tmp_path):
        """What the refusal above is protecting against."""
        import shutil

        secret = tmp_path / "secret.env"
        secret.write_text("KEY=value")
        (tmp_path / "plot.png").symlink_to(secret)
        shutil.copy(tmp_path / "plot.png", tmp_path / "copy.png")
        assert (tmp_path / "copy.png").read_text() == "KEY=value"

    def test_a_fifo_is_refused_rather_than_hanging(self, tmp_path):
        os.mkfifo(tmp_path / "pipe")
        handle = os.open(tmp_path / "pipe", os.O_RDONLY | os.O_NONBLOCK)
        writer = os.open(tmp_path / "pipe", os.O_WRONLY | os.O_NONBLOCK)
        try:
            with pytest.raises(ValueError, match="not a regular file"):
                safe_copy(tmp_path / "pipe", tmp_path / "out", 100)
        finally:
            os.close(writer)
            os.close(handle)

    def test_a_fifo_with_no_writer_is_refused_without_waiting(self, tmp_path):
        """Opening a pipe for reading waits for a writer; nobody will write (review S2)."""
        os.mkfifo(tmp_path / "pipe")
        with no_longer_than(5), pytest.raises(ValueError, match="not a regular file"):
            safe_copy(tmp_path / "pipe", tmp_path / "out", 100)

    def test_a_name_with_a_folder_is_refused_with_a_dir_fd(self, tmp_path):
        handle = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(ValueError, match="plain file name"):
                safe_copy("../x", tmp_path / "copy", 100, dir_fd=handle)
        finally:
            os.close(handle)

    def test_an_out_folder_replaced_by_a_link_is_never_read(self, settings, tmp_path):
        """The run swapped out/ for a link to a folder on this machine (review S1)."""
        secrets_dir = tmp_path / "private"
        secrets_dir.mkdir()
        (secrets_dir / "loot.png").write_bytes(PNG + b"KEY=value")
        work = tmp_path / "work"
        work.mkdir()
        (work / "out").symlink_to(secrets_dir)
        settings.outputs_dir.mkdir()
        box = box_for(settings)
        cards, refused = box.run_outputs(work)
        assert (cards, refused) == ([], ["out"])
        assert list(settings.outputs_dir.iterdir()) == []
        with pytest.raises(ValueError):
            copy_out(work, "loot.png", tmp_path / "copy.png", 1000)
        assert not (tmp_path / "copy.png").exists()

    def test_an_ordinary_folder_walk_would_have_leaked_it(self, tmp_path):
        """What the refusal above protects against: O_NOFOLLOW guards only the last name."""
        secrets_dir = tmp_path / "private"
        secrets_dir.mkdir()
        (secrets_dir / "loot.png").write_bytes(PNG)
        (tmp_path / "out").symlink_to(secrets_dir)
        assert safe_copy(tmp_path / "out" / "loot.png", tmp_path / "copy.png", 100) == len(PNG)

    def test_a_huge_file_is_refused(self, tmp_path):
        (tmp_path / "big").write_bytes(b"x" * 200)
        with pytest.raises(ValueError, match="too large"):
            safe_copy(tmp_path / "big", tmp_path / "out", 100)

    def test_it_never_overwrites(self, tmp_path):
        (tmp_path / "a").write_bytes(b"new")
        (tmp_path / "b").write_bytes(b"old")
        with pytest.raises(FileExistsError):
            safe_copy(tmp_path / "a", tmp_path / "b", 100)

    def test_a_linked_plot_never_reaches_outputs(self, settings, tmp_path):
        work = tmp_path / "work"
        (work / "out").mkdir(parents=True)
        secret = tmp_path / "secret.env"
        secret.write_text("KEY=value")
        (work / "out" / "plot.png").symlink_to(secret)
        (work / "out" / "real.png").write_bytes(PNG)
        settings.outputs_dir.mkdir()
        cards, refused = box_for(settings).run_outputs(work)
        assert refused == ["plot.png"]
        assert [item["kind"] for item in cards] == ["png"]
        assert not any(b"KEY=value" in path.read_bytes() for path in settings.outputs_dir.iterdir())


class TestFileNames:
    @pytest.mark.parametrize("given, stem", [
        ("notes.pdf", "notes"),
        ("../../.env", "env"),
        (ENV_FILE, "env"),
        ("..\\..\\windows", "windows"),
        ("my notes (final).md", "my-notes-final"),
        ("....", "file"),
        ("", "file"),
    ])
    def test_a_name_cant_leave_outputs(self, given, stem):
        assert stem_of(given) == stem


class TestRetries:
    def test_a_retry_in_the_same_turn_replaces_its_own_file(self, settings):
        box = box_for(settings)
        asyncio.run(box.build(Pending("make_markdown", "a"), "# Long"))
        outcome = asyncio.run(box.build(Pending("make_markdown", "a"), "# Short"))
        assert outcome.files[0]["name"] == "a.md"
        assert [path.name for path in settings.outputs_dir.iterdir()] == ["a.md"]
        assert (settings.outputs_dir / "a.md").read_text().startswith("# Short")

    def test_a_file_from_an_earlier_turn_is_never_replaced(self, settings):
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), "# First turn"))
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), "# Second turn"))
        assert sorted(path.name for path in settings.outputs_dir.iterdir()) == ["a-2.md", "a.md"]


class TestSourcesInFiles:
    def test_cited_sources_are_listed_with_their_pages(self):
        text = with_sources("A fact [S1].", {1: SOURCE, 2: {**SOURCE, "doc_title": "Unused"}})
        assert "S1: Book A. pp. 118-119" in text
        assert "Unused" not in text

    def test_a_source_list_gives_authors_year_title_and_publisher(self):
        text = with_sources("A fact [S1].", {1: MML})
        assert ("S1: Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong (2020). Mathematics for Machine "
                "Learning. Cambridge University Press, pp. 117-120") in text

    def test_a_tagged_citation_counts(self):
        assert "S1: Book A" in with_sources("$$ x \\tag{S1} $$", {1: SOURCE})

    def test_a_document_citing_nothing_gets_no_empty_list(self):
        assert "Sources" not in with_sources("Words.", {1: SOURCE})

    def test_unverified_formulas_are_flagged_even_if_uncited(self):
        text = with_sources("Words.", {1: {**SOURCE, "unverified_pages": [34]}})
        assert "could not be verified" in text and "p. 34" in text


class TestImagesInMarkdown:
    @pytest.fixture
    def outputs(self, tmp_path):
        folder = tmp_path / "outputs"
        folder.mkdir()
        (folder / "plot-1.png").write_bytes(PNG)
        return folder

    def localize(self, text, outputs, tmp_path):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        return documents.localize_images(text, outputs, work, 1000), work

    def test_a_plot_the_app_made_is_copied_in(self, outputs, tmp_path):
        text, work = self.localize("![A plot](plot-1.png)", outputs, tmp_path)
        assert text == "![A plot](images/plot-1.png)"
        assert (work / "images" / "plot-1.png").read_bytes() == PNG

    @pytest.mark.parametrize("target", [
        ENV_FILE,
        "../../.env",
        "<../.env>",
        "file:///etc/passwd",
        "https://example.org/x.png",
        "../outputs/plot-1.png",
        "missing.png",
        "plot-1.png/../../.env",
    ])
    def test_any_other_path_stops_the_build(self, outputs, tmp_path, target):
        with pytest.raises(ContentError, match="isn't a plot this app made"):
            self.localize(f"![x]({target})", outputs, tmp_path)
        assert not (tmp_path / "work" / "images").exists()

    def test_a_deck_image_path_is_refused_the_same_way(self, outputs, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(ContentError):
            documents.bring_image(ENV_FILE, outputs, work, 1000)

    def test_a_link_in_outputs_isnt_brought_in(self, outputs, tmp_path):
        secret = tmp_path / "secret.env"
        secret.write_text("KEY=value")
        (outputs / "linked.png").symlink_to(secret)
        with pytest.raises(ValueError, match="not a regular file"):
            self.localize("![x](linked.png)", outputs, tmp_path)


class TestDeckOutline:
    def test_a_full_slide(self):
        deck = parse_deck("## bullets_equation: Title [S1]\n- One\n- Two [S1]\n$$ a = b $$\nAnimate: yes\nNotes: Say this.\n")
        slide = deck.slides[0]
        assert (slide.layout, slide.bullets, slide.equations) == ("bullets_equation", ["One", "Two [S1]"], ["a = b"])
        assert (slide.animate, slide.notes) == (True, "Say this.")

    @pytest.mark.parametrize("layout", theme.LAYOUTS)
    def test_every_layout_name_is_read(self, layout):
        assert parse_deck(f"## {layout}: A title\n- a").slides[0].layout == layout

    def test_a_word_before_a_colon_that_isnt_a_layout_stays_in_the_title(self):
        slide = parse_deck("## Theorem: a result\n- a").slides[0]
        assert (slide.title, slide.layout) == ("Theorem: a result", "bullets_equation")

    @pytest.mark.parametrize("outline, layout", [
        ("## One\n- a", "bullets_equation"),
        ("## One\n$$ a $$", "big_equation"),
        ("## One\n![p](plot.png)", "figure_text"),
        ("## One\nLeft: L\n- a\nRight: R\n- b", "two_columns"),
        ("## One", "title"),
        ("## First\n- a\n## Key takeaways\n- b", "summary"),
    ])
    def test_a_slide_with_no_layout_gets_one_from_what_it_holds(self, outline, layout):
        assert parse_deck(outline).slides[-1].layout == layout

    def test_columns_keep_their_own_bullets(self):
        slide = parse_deck("## two_columns: T\nLeft: One\n- a\n- b\nRight: Two\n- c").slides[0]
        assert (slide.left_heading, slide.left_bullets) == ("One", ["a", "b"])
        assert (slide.right_heading, slide.right_bullets) == ("Two", ["c"])

    def test_a_formula_over_several_lines(self):
        deck = parse_deck("## T\n$$\na = b\n$$\n\\[\nc = d\n\\]\n")
        assert deck.slides[0].equations == ["a = b", "c = d"]

    def test_a_deck_title_above_the_slides_isnt_a_slide(self):
        assert [s.title for s in parse_deck("# Deck\n## One\n- a\n## Two\n- b").slides] == ["One", "Two"]

    def test_hash_headings_work_if_there_are_no_double_hashes(self):
        assert len(parse_deck("# One\n- a\n# Two\n- b").slides) == 2

    def test_text_before_the_first_slide_is_dropped(self):
        assert parse_deck("Here is your deck:\n\n## One\n- a").slides[0].bullets == ["a"]

    def test_a_fence_around_the_whole_outline_is_ignored(self):
        assert parse_deck("```markdown\n## One\n- a\n```").slides[0].bullets == ["a"]

    def test_html_scripts_become_unicode(self):
        assert parse_deck("## The Z<sub>0</sub> line\n- a").slides[0].title == "The Z₀ line"

    def test_bold_is_dropped_because_bullets_are_plain_text(self):
        assert parse_deck("## One\n- **Key** idea").slides[0].bullets == ["Key idea"]

    def test_bold_notes_label_and_lines_after_it(self):
        assert parse_deck("## One\n**Speaker notes:** first\nsecond").slides[0].notes == "first second"

    @pytest.mark.parametrize("bullet, shown", [
        ("The matrix $A$ is big", "The matrix A is big"),
        (r"So \(A = LL^\top\) holds", "So A = LLᵀ holds"),
        (r"Then $x_{k+1} = 0.8 x_k$ [S1]", "Then xₖ₊₁ = 0.8 xₖ [S1]"),
        (r"Iterates follow $x_{i+1} = 0.8\,x_i$", "Iterates follow xᵢ₊₁ = 0.8 xᵢ"),
        (r"Diagonal: \(l_{11} = \sqrt{a_{11}}\)", "Diagonal: l₁₁ = √(a₁₁)"),
    ])
    def test_simple_math_in_a_bullet_becomes_unicode(self, bullet, shown):
        assert parse_deck(f"## One\n- {bullet}").slides[0].bullets == [shown]

    def test_math_too_complex_for_a_bullet_becomes_an_equation(self):
        slide = parse_deck(r"## One" + "\n" + r"- Log-likelihood: $\log p(x) = -\frac12 \begin{pmatrix} a \end{pmatrix}$ [S2]").slides[0]
        assert slide.bullets == ["Log-likelihood [S2]"] or slide.bullets == ["Log-likelihood: [S2]"]
        assert len(slide.equations) == 1 and "pmatrix" in slide.equations[0]

    def test_a_bare_command_becomes_unicode(self):
        assert parse_deck(r"## One" + "\n" + r"- We use \lambda here").slides[0].bullets == ["We use λ here"]

    def test_only_unreadable_latex_is_refused(self):
        with pytest.raises(ContentError, match="can't read"):
            parse_deck(r"## One" + "\n" + r"- See \unknowncommand{x} here")

    def test_formulas_in_notes_become_unicode_too(self):
        assert parse_deck(r"## One" + "\n" + r"Notes: The value $\theta$ here").slides[0].notes == "The value θ here"

    @pytest.mark.parametrize("bullet", ["Costs \\$5", "λ is an eigenvalue of Aᵀ"])
    def test_ordinary_text_isnt_mistaken_for_math(self, bullet):
        assert parse_deck(f"## One\n- {bullet}").slides[0].bullets

    def test_an_unclosed_formula_ends_at_the_blank_line_and_the_deck_survives(self):
        slide = parse_deck("## One\n$$\na = b\n\n- A bullet\n").slides[0]
        assert slide.equations == ["a = b"] and slide.bullets == ["A bullet"]

    def test_a_citation_after_a_formula_doesnt_unbalance_it(self):
        """gpt-oss wrote this three times in a row, and the old pattern called it never closed."""
        deck, _ = finished("## One\n$$ a = b $$ [S1]\n- A bullet", {1: SOURCE})
        slide = deck.slides[0]
        assert slide.equations == ["a = b"] and "Book A" in slide.footer

    def test_a_formula_inside_a_bullet_is_lifted_out(self):
        slide = parse_deck("## One\n- The rule is: $$a = b$$\n").slides[0]
        assert slide.equations == ["a = b"] and slide.bullets == ["The rule is"]

    def test_a_bullet_that_is_only_a_formula_leaves_no_empty_bullet(self):
        slide = parse_deck("## One\n- $$a = b$$\n").slides[0]
        assert slide.equations == ["a = b"] and slide.bullets == []

    def test_no_headings_is_refused(self):
        with pytest.raises(ContentError, match="no slides"):
            parse_deck("Just some text.")


LONG = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"


def finished(outline: str, sources=None):
    return finish_deck(parse_deck(outline), sources or {})


class TestPlotNames:
    def test_the_name_the_model_remembers_finds_the_saved_file(self):
        deck = parse_deck("## figure_text: T\n![p](gd_iterates.png)")
        documents.resolve_images(deck, {"gd_iterates.png": "gd_iterates-03751e.png"})
        assert deck.slides[0].image == "gd_iterates-03751e.png"

    def test_when_only_one_picture_exists_it_is_the_one(self):
        deck = parse_deck("## figure_text: T\n![p](plot.png)")
        documents.resolve_images(deck, {"figure-1.png": "figure-1-abc123.png"})
        assert deck.slides[0].image == "figure-1-abc123.png"

    def test_with_two_pictures_it_doesnt_guess(self):
        deck = parse_deck("## figure_text: T\n![p](plot.png)")
        documents.resolve_images(deck, {"a.png": "a-1.png", "b.png": "b-1.png"})
        assert deck.slides[0].image == "plot.png"

    def test_a_path_is_never_replaced_by_a_guess(self):
        deck = parse_deck(f"## figure_text: T\n![p]({ENV_FILE})")
        documents.resolve_images(deck, {"a.png": "a-1.png"})
        assert deck.slides[0].image == ENV_FILE


class TestSlideCount:
    @pytest.mark.parametrize("question, count", [
        ("Make a 5-slide deck on Cholesky.", 5), ("A 3 slide deck", 3), ("a deck with 4 slides", 4),
        ("Make a deck on Cholesky.", None), ("Slide 3 should show the plot", None),
    ])
    def test_the_count_comes_from_the_readers_words(self, question, count):
        assert documents.requested_slides(question) == count

    def test_extra_slides_go_to_the_last_slides_notes(self):
        deck = parse_deck("## title: A\n## B\n- b\n## C\n- c\n## D\n- d\n## summary: E\n- e")
        documents.keep_slide_count(deck, 3)
        assert [slide.title for slide in deck.slides] == ["A", "B", "E"]
        assert "c" in deck.slides[-1].notes and "D" in deck.slides[-1].notes

    def test_a_short_deck_is_left_alone(self):
        deck = parse_deck("## One\n- a")
        documents.keep_slide_count(deck, 5)
        assert len(deck.slides) == 1

    def test_no_count_asked_changes_nothing(self):
        deck = parse_deck("## One\n- a\n## Two\n- b")
        documents.keep_slide_count(deck, None)
        assert len(deck.slides) == 2


class TestMissingPictures:
    def test_an_invented_name_costs_the_picture_not_the_deck(self, tmp_path):
        deck = parse_deck("## figure_text: T\n![p](example.png)\n- a")
        assert documents.drop_missing_images(deck, tmp_path) == ["example.png"]
        finish_deck(deck, {})
        assert deck.slides[0].layout == "bullets_equation" and deck.slides[0].bullets == ["a"]

    def test_an_image_written_out_inline_is_dropped_unread(self, tmp_path):
        deck = parse_deck("## figure_text: T\n![p](data:image/png;base64,iVBORw0KGgo)\n- a")
        assert documents.drop_missing_images(deck, tmp_path) == ["an inline image"]

    def test_a_real_picture_stays(self, tmp_path):
        (tmp_path / "plot.png").write_bytes(PNG)
        deck = parse_deck("## figure_text: T\n![p](plot.png)")
        assert documents.drop_missing_images(deck, tmp_path) == []
        assert deck.slides[0].image == "plot.png"

    def test_a_bare_name_with_no_extension_is_dropped_too(self, tmp_path):
        deck = parse_deck("## figure_text: T\n![p](filename_placeholder)")
        assert documents.drop_missing_images(deck, tmp_path) == ["filename_placeholder"]

    @pytest.mark.parametrize("target", [ENV_FILE, "../../.env", "../x.png"])
    def test_a_path_is_not_quietly_dropped_it_stays_for_the_refusal(self, tmp_path, target):
        deck = parse_deck(f"## figure_text: T\n![p]({target})")
        assert documents.drop_missing_images(deck, tmp_path) == []
        assert deck.slides[0].image == target


class TestDeckLimits:
    def test_a_fifth_bullet_goes_to_the_notes(self):
        deck, _ = finished("## bullets_equation: T\n- a\n- b\n- c\n- d\n- e")
        slide = deck.slides[0]
        assert slide.bullets == ["a", "b", "c", "d"] and "e" in slide.notes

    def test_a_long_bullet_is_cut_to_twelve_words_and_kept_in_full_in_the_notes(self):
        deck, trimmed = finished(f"## bullets_equation: T\n- {LONG}")
        slide = deck.slides[0]
        assert len(slide.bullets[0].split()) == 12 and slide.bullets[0].endswith("…")
        assert LONG in slide.notes and trimmed == slide.bullets

    def test_it_cuts_at_a_clause_when_it_can(self):
        deck, _ = finished("## bullets_equation: T\n- one two three four five six, seven eight nine ten eleven twelve thirteen")
        assert deck.slides[0].bullets == ["one two three four five six"]

    def test_a_cut_never_stops_inside_a_bracket(self):
        text = "From Breit-Wigner theory to LEP precision measurements (based on Rubbia, Phenomenology of Particle Physics)"
        assert documents.shorten(text, 12) == "From Breit-Wigner theory to LEP precision measurements"

    def test_a_long_caption_keeps_its_full_text_in_the_notes(self):
        deck, _ = finished(f"## big_equation: T\n$$ a $$\nCaption: {LONG}")
        slide = deck.slides[0]
        assert len(slide.caption.split()) == 12 and LONG in slide.notes

    def test_a_short_bullet_is_left_alone(self):
        deck, trimmed = finished("## bullets_equation: T\n- A short bullet")
        assert deck.slides[0].bullets == ["A short bullet"] and trimmed == []

    def test_two_columns_share_the_limit(self):
        deck, _ = finished("## two_columns: T\nLeft: L\n- a\n- b\n- c\nRight: R\n- d\n- e\n- f")
        slide = deck.slides[0]
        assert len(slide.left_bullets) + len(slide.right_bullets) == theme.MAX_BULLETS
        assert "c" in slide.notes and "f" in slide.notes

    def test_the_slide_count_never_changes(self):
        outline = "\n".join(f"## bullets_equation: S{n}\n" + "\n".join(f"- {LONG}" for _ in range(9)) for n in range(5))
        assert len(finished(outline)[0].slides) == 5

    def test_no_slide_ever_goes_over_the_word_limit(self):
        sources = {1: {**SOURCE, "doc_title": "A rather long book title indeed for this particular test"}}
        deck, _ = finished(f"## bullets_equation: {LONG}\n" + "\n".join(f"- {LONG} [S1]" for _ in range(9)), sources)
        assert documents.word_count(deck.slides[0]) <= theme.MAX_SLIDE_WORDS

    def test_nothing_cut_from_a_slide_is_lost(self):
        lines = [f"bullet number {n} {LONG}" for n in range(8)]
        deck, _ = finished("## bullets_equation: T\n" + "\n".join(f"- {line}" for line in lines))
        for line in lines:
            assert line in deck.slides[0].notes


class TestDeckLayoutFit:
    def test_too_many_equations_beside_bullets_go_to_the_notes(self):
        deck, _ = finished("## bullets_equation: T\n- a\n$$ x^2 $$\n$$ y^2 $$\n$$ z^2 $$")
        slide = deck.slides[0]
        assert slide.equations == ["x^2", "y^2"] and "z²" in slide.notes

    def test_three_equations_and_no_bullets_become_a_big_equation_slide(self):
        assert finished("## bullets_equation: T\n$$ a $$\n$$ b $$\n$$ c $$")[0].slides[0].layout == "big_equation"

    def test_a_figure_slide_with_no_figure_falls_back(self):
        assert finished("## figure_text: T\n- a")[0].slides[0].layout == "bullets_equation"

    def test_a_title_slides_first_bullet_becomes_its_subtitle(self):
        slide = finished("## title: T\n- A line\n- Another")[0].slides[0]
        assert slide.subtitle == "A line" and slide.bullets == [] and "Another" in slide.notes

    def test_a_lone_bullet_on_a_big_equation_becomes_its_caption(self):
        slide = finished("## big_equation: T\n$$ a $$\n- Why it matters")[0].slides[0]
        assert (slide.caption, slide.bullets) == ("Why it matters", [])

    def test_two_columns_with_plain_bullets_split_in_half(self):
        slide = finished("## two_columns: T\n- a\n- b\n- c\n- d")[0].slides[0]
        assert (slide.left_bullets, slide.right_bullets) == (["a", "b"], ["c", "d"])


class TestDeckCitations:
    SOURCES = {1: MML, 2: UDL}

    def test_no_citation_stays_in_a_bullet(self):
        deck, _ = finished("## bullets_equation: T\n- A fact [S1]\n- Another [S2]", self.SOURCES)
        assert deck.slides[0].bullets == ["A fact", "Another"]

    def test_the_footer_is_one_line_of_authors_and_years(self):
        deck, _ = finished("## bullets_equation: T\n- A fact [S1]\n- Another [S2]", self.SOURCES)
        assert deck.slides[0].footer == "Sources: Deisenroth et al. 2020, pp. 117-120; Prince 2024, pp. 290-293"

    def test_full_references_go_in_the_notes(self):
        deck, _ = finished("## bullets_equation: T\n- A fact [S2]", self.SOURCES)
        assert "S2: Simon J. D. Prince (2024). Understanding Deep Learning. MIT Press, pp. 290-293" in deck.slides[0].notes

    def test_a_book_cited_twice_is_named_once(self):
        deck, _ = finished("## bullets_equation: T\n- A [S1]\n- B [S1]", self.SOURCES)
        assert deck.slides[0].footer.count("Deisenroth") == 1

    def test_many_sources_still_fit_one_line(self):
        sources = {n: {**SOURCE, "doc_title": f"Book {n}"} for n in range(1, 6)}
        deck, _ = finished("## bullets_equation: T\n- " + " ".join(f"[S{n}]" for n in range(1, 6)), sources)
        assert deck.slides[0].footer.endswith("and 3 more")

    def test_a_citation_in_an_equations_tag_counts_and_leaves_the_equation(self):
        deck, _ = finished("## big_equation: T\n$$ a = b \\tag{S1} $$", self.SOURCES)
        assert deck.slides[0].equations == ["a = b"] and "Deisenroth" in deck.slides[0].footer

    def test_unverified_formulas_are_flagged_on_the_last_slide(self):
        deck, _ = finished("## One\n- a\n## Two\n- b", {1: {**SOURCE, "unverified_pages": [34]}})
        assert "could not be verified" in deck.slides[-1].notes and "p. 34" in deck.slides[-1].notes


class TestReferences:
    @pytest.mark.parametrize("author, short", [
        ("Kevin P. Murphy", "Murphy 2022"),
        ("A One, B Two", "One and Two 2022"),
        ("Waleed Esmail, Stuart Russell, Jana Klinge, Alexander Kappes, Christine Thomas", "Esmail et al. 2022"),
        ("Waleed Esmail; Stuart Russell", "Esmail and Russell 2022"),
    ])
    def test_a_footer_names_authors_by_surname(self, author, short):
        assert documents.short_reference({"doc_title": "T", "author": author, "year": 2022}) == short

    def test_with_no_author_a_footer_falls_back_to_the_title_not_the_file(self):
        assert documents.short_reference({"doc_title": "Untitled document", "year": 2022}) == "Untitled document"

    def test_a_title_ending_in_a_question_mark_gets_no_extra_full_stop(self):
        source = {"doc_title": "When Do Models Forecast?", "author": "A One", "year": 2026, "page_start": 1, "page_end": 1}
        assert documents.full_reference(source) == "A One (2026). When Do Models Forecast? p. 1"

    def test_nothing_in_a_reference_comes_from_a_file_name(self):
        """The file name can't reach a reference: the record has no field for it."""
        source = {"doc_title": "Untitled document", "page_start": 4, "page_end": 4,
                  "doc_path": "Simon J. D. Prince - Understanding Deep Learning-55-386.pdf"}
        for text in (documents.full_reference(source), documents.short_reference(source)):
            assert "Prince" not in text and ".pdf" not in text


class TestSchemas:
    def test_there_are_seven_tools_and_annotate_pdf_is_not_one(self):
        assert len(TOOLS) == 7 and "annotate_pdf" not in TOOLS

    def test_both_providers_get_every_tool(self):
        assert [t["name"] for t in anthropic_tools()] == [t["function"]["name"] for t in ollama_tools()] == list(TOOLS)

    def test_file_tools_take_no_content(self):
        for tool in anthropic_tools():
            if tool["name"] in FILE_TOOLS:
                assert set(tool["input_schema"]["properties"]) == {"filename", "export_previous_answer"}

    def test_a_bad_call_is_refused_and_says_why(self, settings):
        outcome = asyncio.run(box_for(settings).call("run_python", {"cod": "print(1)"}))
        assert not outcome.ok and "code" in outcome.text

    def test_an_unknown_tool_is_refused(self, settings):
        assert not asyncio.run(box_for(settings).call("annotate_pdf", {})).ok


class TestFilesFromTheAnswer:
    def test_a_new_file_waits_for_the_document(self, settings):
        outcome = asyncio.run(box_for(settings).call("make_markdown", {"filename": "a"}))
        assert outcome.pending == Pending("make_markdown", "a")
        assert not settings.outputs_dir.exists()

    def test_the_flag_exports_the_stored_answer_with_its_own_sources(self, settings):
        box = box_for(settings, previous="An answer [S1].", previous_sources={1: SOURCE})
        outcome = asyncio.run(box.call("make_markdown", {"filename": "a", "export_previous_answer": True}))
        text = (settings.outputs_dir / "a.md").read_text()
        assert outcome.ok and text.startswith("An answer [S1].") and "S1: Book A" in text

    @pytest.mark.parametrize("question", [
        "Export this answer as a PDF.", "Export this answer as a Word document.", "save that answer to a pdf",
        "Turn your explanation into a deck", "this answer as a PDF please", "Export it", "Exportiere diese Antwort",
    ])
    def test_the_readers_words_export_even_if_the_model_forgets_the_flag(self, settings, question):
        box = box_for(settings, previous="An answer.")
        box.question = question
        outcome = asyncio.run(box.call("make_markdown", {"filename": "a"}))
        assert outcome.pending is None and outcome.files

    @pytest.mark.parametrize("question", [
        "Make a 5-slide deck on the Cholesky decomposition from MML.",
        "Save study notes on the evidence lower bound as a Markdown file.",
        "Make a one-page cheat sheet on the SVD, as a PDF.",
        "Write out the derivation in a PDF.",
    ])
    def test_a_new_document_isnt_mistaken_for_an_export(self, settings, question):
        box = box_for(settings, previous="An answer.")
        box.question = question
        assert asyncio.run(box.call("make_markdown", {"filename": "a"})).pending is not None

    def test_asking_to_export_with_nothing_to_export_writes_something_new(self, settings):
        """gpt-oss often sets the flag on a first question, where it can't mean anything."""
        box = box_for(settings)
        outcome = asyncio.run(box.call("make_pdf", {"filename": "a", "export_previous_answer": True}))
        assert outcome.pending is not None


class TestPlainMath:
    @pytest.mark.parametrize("latex, plain", [
        (r"A = LL^{\top}", "A = LLᵀ"),
        (r"\lambda_1 \leq \lambda_2", "λ₁ ≤ λ₂"),
        (r"x^2 + y^2", "x² + y²"),
        (r"a_{ij}", "aᵢⱼ"),
        (r"x_{k+1}", "xₖ₊₁"),
        (r"\frac{a}{b}", "a/b"),
        (r"\frac{x+1}{y+2}", "(x+1)/(y+2)"),
        (r"\sqrt{2}", "√2"),
        (r"\sqrt{a+b}", "√(a+b)"),
        (r"\mathbf{x} \in \mathbb{R}^n", "x ∈ ℝⁿ"),
        (r"\sum_{i=1}^{n} x_i", "Σᵢ₌₁ⁿ xᵢ"),
        (r"\sigma^{2}", "σ²"),
        (r"e^{-x}", "e^(-x)"),
        (r"\left( a \right)", "( a )"),
        (r"0.8\,x_i", "0.8 xᵢ"),
        (r"\sqrt{a_{11}}", "√(a₁₁)"),
        (r"\frac{a_{21}}{l_{11}}", "(a₂₁)/(l₁₁)"),
    ])
    def test_it_writes_common_math_in_unicode(self, latex, plain):
        assert documents.plain_math(latex) == plain

    @pytest.mark.parametrize("latex", [r"\begin{bmatrix} a \end{bmatrix}", r"\unknownmacro{x}"])
    def test_anything_it_cant_express_comes_back_as_none(self, latex):
        assert documents.plain_math(latex) is None


class TestTypography:
    def test_characters_the_fonts_lack_become_plain_ones(self):
        assert documents.plain_typography("non\u2011breaking 2\u202f000") == "non-breaking 2 000"

    def test_every_file_gets_it(self, settings):
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), "non\u2011breaking"))
        assert "non-breaking" in (settings.outputs_dir / "a.md").read_text()


class TestLabels:
    def test_a_label_is_dropped(self):
        assert documents.without_labels("$$ a \\tag{1}\n\\label{eq:a}\n$$") == "$$ a \\tag{1}\n\n$$"

    def test_a_file_has_none(self, settings):
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), "$$a \\label{x}$$"))
        assert "label" not in (settings.outputs_dir / "a.md").read_text()


CAPITALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZΓΔΘΛΞΠΣΦΨΩ"


class TestCapitalSubscripts:
    """A capital is never written as a lowercase subscript. Unicode has almost no capital
    subscripts, so a capital stays a capital and the deck sets it as real subscript text."""

    @pytest.mark.parametrize("capital", list(CAPITALS))
    def test_no_capital_is_ever_lowercased(self, capital):
        plain = documents.plain_math(f"M_{capital}")
        assert plain == f"M_{capital}"
        assert capital.lower() not in plain and not any("\u2090" <= c <= "\u209c" or c in "ᵢⱼᵣᵤᵥ" for c in plain)

    @pytest.mark.parametrize("latex, plain", [
        (r"\Gamma_Z", "Γ_Z"), (r"M_Z^2", "M_Z²"), (r"x_{N}", "x_N"), (r"A_{ij}", "Aᵢⱼ"), (r"T_{\mu\nu}", "T_(μν)"),
    ])
    def test_lowercase_ones_still_become_unicode(self, latex, plain):
        assert documents.plain_math(latex) == plain

    def test_a_bullet_keeps_the_capital_for_the_deck_to_set(self):
        assert parse_deck(r"## One" + "\n" + r"- Near $M_Z$ and $\Gamma_Z$").slides[0].bullets == ["Near M_Z and Γ_Z"]


class TestUprightWords:
    @pytest.mark.parametrize("latex, upright", [
        (r"\sigma^0_{had}", r"\sigma^0_{\text{had}}"),
        (r"p_{ISR}(s', s)", r"p_{\text{ISR}}(s', s)"),
        (r"\Gamma_{had} + E_{max}", r"\Gamma_{\text{had}} + E_{\text{max}}"),
        (r"l_{ii} + a_{ij} + x_{nk}", r"l_{ii} + a_{ij} + x_{nk}"),
        (r"T_{ijk}", r"T_{ijk}"),
        (r"E_{min} < E_{tot}", r"E_{\text{min}} < E_{\text{tot}}"),
        (r"\sigma_{\text{had}}", r"\sigma_{\text{had}}"),
    ])
    def test_words_are_upright_and_indices_stay_maths(self, latex, upright):
        assert documents.upright_words(latex) == upright

    def test_only_inside_formulas(self):
        text = "The word_{had} in text, and $\\sigma_{had}$ in maths. `code_{had}`"
        assert documents.upright_words_in_math(text) == (
            "The word_{had} in text, and $\\sigma_{\\text{had}}$ in maths. `code_{had}`")

    def test_every_file_gets_it(self, settings):
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), "$$\\sigma_{had}$$"))
        assert "\\sigma_{\\text{had}}" in (settings.outputs_dir / "a.md").read_text()

    def test_deck_equations_get_it(self):
        deck, _ = finished("## big_equation: T\n$$ p_{ISR} $$")
        assert deck.slides[0].equations == [r"p_{\text{ISR}}"]


class TestDollarMath:
    def test_both_kinds_of_backslash_math_convert(self):
        assert documents.dollar_math(r"a \(x\) b \[ y \] c") == "a $x$ b $$y$$ c"

    def test_a_formula_over_several_lines(self):
        assert documents.dollar_math("\\[\na = b\n\\]") == "$$a = b$$"

    def test_code_is_left_alone(self):
        text = "`\\(x\\)` and\n```\n\\[ y \\]\n```"
        assert documents.dollar_math(text) == text

    def test_a_markdown_file_gets_dollars(self, settings):
        asyncio.run(box_for(settings).build(Pending("make_markdown", "a"), r"Math \(x\)"))
        assert "$x$" in (settings.outputs_dir / "a.md").read_text()


class TestSearchLibrary:
    def box(self, settings, found):
        box = ToolBox(settings, lambda query: found, {1: SOURCE}, [hit(1)], record=lambda n, h: {"id": n, "page_start": h.page_start})
        return box

    def test_new_passages_continue_the_numbering(self, settings):
        box = self.box(settings, [hit(5, "New words")])
        outcome = asyncio.run(box.call("search_library", {"query": "q"}))
        assert outcome.text.startswith("[S2] Book A, p. 5") and 2 in box.sources

    def test_a_passage_already_read_isnt_sent_again(self, settings):
        outcome = asyncio.run(self.box(settings, [hit(1)]).call("search_library", {"query": "q"}))
        assert outcome.text == "That search found nothing new."

    def test_the_search_uses_the_scope_it_was_given(self, settings):
        """The model passes a query and nothing else, so it can't widen the scope."""
        assert set(TOOLS["search_library"].schema.model_fields) == {"query"}

    def test_a_huge_result_is_cut_to_the_budget(self, settings):
        settings.tool_search_tokens = 50
        box = self.box(settings, [hit(5, "word " * 200), hit(6, "word " * 200)])
        assert len(asyncio.run(box.call("search_library", {"query": "q"})).hits) == 1


def test_capped_output_keeps_both_ends():
    capped = CappedOutput(10)
    for chunk in (b"abcde", b"12345", b"VWXYZ"):
        capped.add(chunk)
    assert capped.text().startswith("abcde") and capped.text().endswith("VWXYZ")
    assert "5 bytes cut" in capped.text()


def test_the_last_json_line_ignores_noise_before_it():
    assert last_json_line('warning\n{"ok": true}\n') == {"ok": True}
    assert last_json_line("no json at all") is None


def test_sandbox_unavailable_is_a_runtime_error_the_chat_can_report():
    assert issubclass(SandboxUnavailable, RuntimeError)


def test_nothing_in_the_sandbox_folder_reads_project_paths():
    """The image is built from this folder alone, so a project path here would be a bug."""
    folder = Path(__file__).resolve().parents[2] / "sandbox"
    for path in folder.iterdir():
        if path.is_file() and path.name != ".dockerignore":
            assert "study-assistant/" not in path.read_text(encoding="utf-8"), path.name


def test_a_tool_that_crashes_ends_the_call_not_the_answer(settings, monkeypatch):
    """Review C4: an exception other than SandboxUnavailable left ToolBox.call and broke the stream."""
    async def crash(self, request):
        raise KeyError("surprise")

    monkeypatch.setattr(ToolBox, "sympy_check", crash)
    outcome = asyncio.run(box_for(settings).call("sympy_check", {"lhs": "x", "rhs": "x"}))
    assert not outcome.ok and "KeyError" in outcome.text


class TestListsAfterParagraphs:
    """The bug report's PDF showed "* Multi-token prediction" as text: a list straight under a sentence."""

    def test_a_list_under_a_sentence_gets_its_blank_line(self):
        text = "The study looks at three parts:\n* **MTP:** first\n* **Head:** second\n\n1. one\n2. two\n"
        assert documents.lists_after_paragraphs(text) == (
            "The study looks at three parts:\n\n* **MTP:** first\n* **Head:** second\n\n1. one\n2. two\n")

    @pytest.mark.parametrize("text", [
        "Intro.\n\n- already apart\n- fine\n",
        "*Figure 5: an italic caption, not a list.*\nNext line.\n",
        "```python\nx = 1\n- y\n```\n",
        "$$\na\n- b\n$$\n",
        "- item\n  continued here\n- next item\n",
        "Text with a - dash and 3. a number inside.\n",
    ], ids=["apart", "italics", "code", "maths", "continuation", "inline"])
    def test_nothing_else_changes(self, text):
        assert documents.lists_after_paragraphs(text) == text
