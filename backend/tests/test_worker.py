"""Queue, stage and restart behaviour, with a fake parser and embedder."""
import json
import math
import multiprocessing
import os
import threading
import time
from pathlib import Path

import fitz
import pytest

from app.config import Settings
from app.db.connection import connect
from app.ingest.watcher import LibraryWatcher, SettleQueue
from app.supervisor import LOCKED
from app.ingest.worker import (
    STAGES,
    Worker,
    is_ingestible,
    reindex,
    scan_library,
    sync_file,
)


class Crash(BaseException):
    """Escapes `except Exception`, like a kill: nothing gets to clean up."""


def unit_vector(seed: int) -> list[float]:
    values = [math.sin(seed + i) + 1.5 for i in range(1024)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def fake_embedder(texts):
    return [unit_vector(len(t)) for t in texts]


def write_sidecar(out_dir: Path, sections: int = 6, marker: str = "word") -> None:
    records = []
    for i in range(sections):
        records.append({"order": len(records), "kind": "heading", "text": f"Section {i}",
                        "level": 1, "page": i + 1})
        records.append({"order": len(records), "kind": "paragraph",
                        "text": f"{marker} number {i} in the body.", "page": i + 1,
                        "bbox": [10, 20, 30, 40], "docling_ref": f"#/texts/{i}"})
    with open(out_dir / "blocks.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def fake_parser(pdf: Path, out_dir: Path) -> dict:
    write_sidecar(out_dir)
    return {"title": "Parsed title", "pages": 6}


def no_formulas(pdf: Path, page: int) -> list[dict]:
    return []


SUMMARY_CALLS: list[str] = []


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """No test reaches Ollama or a hosted describer: the summarize stage gets a fake model."""
    import app.ingest.summarize as tree

    SUMMARY_CALLS.clear()

    def fake_ollama_ask(settings, model, think=False, timeout=900):
        def ask(prompt):
            SUMMARY_CALLS.append(prompt)
            return f"Summary {len(SUMMARY_CALLS)}."
        return ask
    monkeypatch.setattr(tree, "ollama_ask", fake_ollama_ask)
    monkeypatch.setattr(tree, "ollama_stream_ask", lambda settings, model, think=False, busy=None: fake_ollama_ask(settings, model))

    def no_describer(ctx, rows):
        raise AssertionError("a test reached the figure describer")
    monkeypatch.setattr("app.ingest.worker.default_describe_figures", no_describer)


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "library").mkdir()
    return Settings(library_dir=tmp_path / "library", data_dir=tmp_path / "data",
                    embed_batch=2, settle_seconds=0.2)


@pytest.fixture
def conn(settings):
    connection = connect(settings.db_path)
    yield connection
    connection.close()


def add_pdf(settings, name="a.pdf", content=b"%PDF-1.7 one"):
    path = settings.library_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def jobs(conn, doc_id=1):
    rows = conn.execute("SELECT stage, status FROM jobs WHERE doc_id = ? ORDER BY id", (doc_id,))
    return [(r["stage"], r["status"]) for r in rows]


def text_chunks(conn):
    return conn.execute("SELECT count(*) FROM chunks WHERE kind = 'text'").fetchone()[0]


def text_vectors(conn):
    return conn.execute("SELECT count(*) FROM chunk_vec v JOIN chunks c ON c.id = v.chunk_id "
                        "WHERE c.kind = 'text'").fetchone()[0]


def doc(conn, doc_id=1):
    return conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()


def count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# --- Detect ------------------------------------------------------------------


def test_new_file_gets_one_job_per_stage(conn, settings):
    add_pdf(settings)
    assert scan_library(conn, settings) == {"new": 1}
    assert jobs(conn) == [(s, "pending") for s in STAGES]


def test_rescan_is_a_no_op(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    assert scan_library(conn, settings) == {"known": 1}
    assert count(conn, "jobs") == len(STAGES)


def test_changed_file_keeps_its_id_and_requeues(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()

    path.write_bytes(b"%PDF-1.7 two")
    assert sync_file(conn, settings, path) == "changed"
    assert doc(conn)["id"] == 1
    assert doc(conn)["status"] == "queued"
    assert count(conn, "documents") == 1


def test_change_mid_queue_drops_the_stale_pending_jobs(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    path.write_bytes(b"%PDF-1.7 two")
    sync_file(conn, settings, path)
    assert count(conn, "jobs") == len(STAGES)
    new_sha = doc(conn)["sha256"]
    assert {r[0] for r in conn.execute("SELECT sha256 FROM jobs")} == {new_sha}


def test_renamed_file_keeps_its_document(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()

    path.rename(settings.library_dir / "renamed.pdf")
    assert scan_library(conn, settings) == {"moved": 1}
    row = doc(conn)
    assert row["path"] == "renamed.pdf"
    assert row["status"] == "indexed"
    assert count(conn, "documents") == 1


def test_identical_copy_is_skipped(conn, settings):
    add_pdf(settings, "a.pdf")
    add_pdf(settings, "copy.pdf")
    assert scan_library(conn, settings) == {"new": 1, "duplicate": 1}


def test_deleted_file_is_marked_missing_and_recovers(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()
    content = path.read_bytes()

    path.unlink()
    assert scan_library(conn, settings) == {"missing": 1}
    assert doc(conn)["status"] == "missing"
    assert count(conn, "chunks") > 0       # notes depend on it, so nothing is deleted

    path.write_bytes(content)
    assert scan_library(conn, settings) == {"restored": 1}
    assert doc(conn)["status"] == "indexed"


@pytest.mark.parametrize("relative,expected", [
    ("book.pdf", True),
    ("BOOK.PDF", True),
    ("physics/book.pdf", True),
    ("annotated/book.pdf", False),
    ("annotated/sub/book.pdf", False),
    (".hidden.pdf", False),
    (".cache/book.pdf", False),
    ("~$book.pdf", False),
    ("book.pdf.part", False),
    ("notes.txt", False),
    ("deck.pptx", True),
    ("~$deck.pptx", False),
    ("deck.ppt", False),
])
def test_is_ingestible(tmp_path, relative, expected):
    assert is_ingestible(tmp_path / relative, tmp_path) is expected


def test_a_path_outside_the_library_is_not_ingestible(tmp_path):
    assert is_ingestible(tmp_path / "other" / "a.pdf", tmp_path / "library") is False


# --- Stages ------------------------------------------------------------------


def test_full_pipeline_populates_everything(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()

    row = doc(conn)
    assert row["status"] == "indexed"
    assert row["title"] == "Parsed title"
    assert row["pages"] == 6
    assert jobs(conn) == [(s, "done") for s in STAGES]
    assert count(conn, "blocks") == 12
    assert text_chunks(conn) == text_vectors(conn) == 6
    summaries = conn.execute("SELECT count(*) FROM chunks WHERE kind = 'summary'").fetchone()[0]
    assert summaries > 0 and count(conn, "chunk_vec") == count(conn, "chunks") == 6 + summaries
    assert conn.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'number'"
    ).fetchall()


def test_chunk_points_back_to_blocks_with_positions(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()

    chunk = conn.execute("SELECT * FROM chunks ORDER BY seq LIMIT 1").fetchone()
    block_id = json.loads(chunk["block_ids"])[0]
    block = conn.execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
    assert block["page"] == chunk["page_start"] == 1
    assert json.loads(block["bbox"]) == [10, 20, 30, 40]
    assert block["docling_ref"] == "#/texts/0"


def test_stages_run_in_order_per_document(conn, settings):
    add_pdf(settings, "a.pdf", b"one")
    add_pdf(settings, "b.pdf", b"two")
    scan_library(conn, settings)
    worker = Worker(conn, settings, fake_parser, fake_embedder)
    assert worker.next_job()["stage"] == "parse"
    worker.run_one()
    assert worker.next_job()["stage"] == "formulas"
    assert worker.next_job()["doc_id"] == 1


def test_reingest_replaces_content_but_keeps_note_chunks(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()
    conn.execute(
        "INSERT INTO chunks (doc_id, kind, seq, text, token_count) VALUES (1, 'note', 0, 'my note', 2)"
    )
    conn.commit()

    def second_parser(pdf, out_dir):
        write_sidecar(out_dir, sections=3, marker="fresh")
        return {"pages": 3}

    path.write_bytes(b"%PDF-1.7 two")
    sync_file(conn, settings, path)
    Worker(conn, settings, second_parser, fake_embedder).drain()

    kinds = [r[0] for r in conn.execute("SELECT kind FROM chunks")]
    assert kinds.count("text") == 3
    assert kinds.count("note") == 1
    assert not conn.execute("SELECT 1 FROM chunk_fts WHERE chunk_fts MATCH 'word'").fetchall()
    conn.execute("INSERT INTO chunk_fts (chunk_fts) VALUES ('integrity-check')")


# --- Failure and restart -----------------------------------------------------


def test_exception_retries_then_fails_and_the_queue_moves_on(conn, settings):
    add_pdf(settings, "bad.pdf", b"bad")
    add_pdf(settings, "good.pdf", b"good")
    scan_library(conn, settings)
    calls = []

    def parser(pdf, out_dir):
        calls.append(pdf.name)
        if pdf.name == "bad.pdf":
            raise ValueError("cannot parse this")
        return fake_parser(pdf, out_dir)

    Worker(conn, settings, parser, fake_embedder).drain()

    assert calls.count("bad.pdf") == settings.max_attempts
    bad = doc(conn, 1)
    assert bad["status"] == "failed"
    job = conn.execute("SELECT * FROM jobs WHERE doc_id = 1 AND stage = 'parse'").fetchone()
    assert "cannot parse this" in job["error"]
    assert doc(conn, 2)["status"] == "indexed"


def test_transient_failure_recovers(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    failures = [ConnectionError("ollama restarting")]

    def flaky(texts):
        if failures:
            raise failures.pop()
        return fake_embedder(texts)

    Worker(conn, settings, fake_parser, flaky).drain()
    assert doc(conn)["status"] == "indexed"


def test_killed_parse_leaves_no_finished_folder_and_restarts_clean(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)

    def dying(pdf, out_dir):
        (out_dir / "blocks.jsonl").write_text("half a file")
        raise Crash

    with pytest.raises(Crash):
        Worker(conn, settings, dying, fake_embedder).drain()
    sha = doc(conn)["sha256"]
    assert not (settings.parsed_dir / sha).exists()

    worker = Worker(conn, settings, fake_parser, fake_embedder)
    assert worker.recover() == 1
    worker.drain()
    assert doc(conn)["status"] == "indexed"
    assert not (settings.parsed_dir / f"{sha}.tmp").exists()


def test_finished_parse_is_not_repeated_after_a_later_crash(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    calls = []

    def counting(pdf, out_dir):
        calls.append(1)
        return fake_parser(pdf, out_dir)

    def dying(texts):
        raise Crash

    with pytest.raises(Crash):
        Worker(conn, settings, counting, dying).drain()
    worker = Worker(conn, settings, counting, fake_embedder)
    worker.recover()
    worker.drain()
    assert len(calls) == 1
    assert doc(conn)["status"] == "indexed"


def test_index_resumes_at_the_last_committed_batch(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    seen: list[str] = []

    def dies_on_third_batch(texts):
        if len(seen) >= 4:
            raise Crash
        seen.extend(texts)
        return fake_embedder(texts)

    with pytest.raises(Crash):
        Worker(conn, settings, fake_parser, dies_on_third_batch).drain()
    assert count(conn, "chunk_vec") == 4

    resumed: list[str] = []

    def remainder(texts):
        resumed.extend(texts)
        return fake_embedder(texts)

    worker = Worker(conn, settings, fake_parser, remainder)
    worker.recover()
    worker.drain()

    resumed = [text for text in resumed if "AI-written summary" not in text]     # the summarize stage's
    assert len(resumed) == 2
    assert not set(seen) & set(resumed)
    assert text_vectors(conn) == text_chunks(conn) == 6
    job = conn.execute("SELECT * FROM jobs WHERE stage = 'index'").fetchone()
    assert (job["progress"], job["total"]) == (6, 6)


def test_failed_chunk_stage_rolls_back_to_the_previous_version(conn, settings):
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder).drain()
    before = count(conn, "chunks")

    def broken_sidecar(pdf, out_dir):
        (out_dir / "blocks.jsonl").write_text('{"order": 0, "kind": "paragraph"}\n')
        return {"pages": 1}

    path.write_bytes(b"%PDF-1.7 two")
    sync_file(conn, settings, path)
    Worker(conn, settings, broken_sidecar, fake_embedder).drain()

    assert doc(conn)["status"] == "failed"
    assert count(conn, "chunks") == before


def test_job_that_keeps_killing_the_process_gives_up(conn, settings):
    add_pdf(settings, "poison.pdf", b"poison")
    add_pdf(settings, "good.pdf", b"good")
    scan_library(conn, settings)

    def killer(pdf, out_dir):
        if pdf.name == "poison.pdf":
            raise Crash
        return fake_parser(pdf, out_dir)

    for _ in range(settings.max_attempts):
        with pytest.raises(Crash):
            Worker(conn, settings, killer, fake_embedder).drain()
        Worker(conn, settings).recover()

    Worker(conn, settings, killer, fake_embedder).drain()
    status = dict(conn.execute("SELECT path, status FROM documents").fetchall())
    assert status == {"poison.pdf": "failed", "good.pdf": "indexed"}
    job = conn.execute(
        "SELECT j.error FROM jobs j JOIN documents d ON d.id = j.doc_id "
        "WHERE d.path = 'poison.pdf' AND j.stage = 'parse'"
    ).fetchone()
    assert "Gave up" in job["error"]


def test_a_stopped_job_goes_back_in_the_queue_without_using_an_attempt(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)

    def stopped(pdf, out_dir):
        raise KeyboardInterrupt

    for _ in range(settings.max_attempts + 2):      # more restarts than attempts
        with pytest.raises(KeyboardInterrupt):
            Worker(conn, settings, stopped, fake_embedder).drain()
    job = conn.execute("SELECT status, attempts FROM jobs WHERE stage = 'parse'").fetchone()
    assert (job["status"], job["attempts"]) == ("pending", 0)

    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    assert doc(conn)["status"] == "indexed"


def _die_mid_index(db_path, library_dir, data_dir):
    """A real process that dies with no Python cleanup at all."""
    settings = Settings(library_dir=Path(library_dir), data_dir=Path(data_dir), embed_batch=2)
    connection = connect(db_path)
    batches = []

    def embedder(texts):
        batches.append(texts)
        if len(batches) == 2:
            os._exit(9)
        return fake_embedder(texts)

    Worker(connection, settings, fake_parser, embedder).drain()


def test_hard_process_death_mid_index_then_resume(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_die_mid_index,
        args=(str(settings.db_path), str(settings.library_dir), str(settings.data_dir)),
    )
    process.start()
    process.join(60)
    assert process.exitcode == 9
    assert count(conn, "chunk_vec") == 2

    worker = Worker(conn, settings, fake_parser, fake_embedder)
    assert worker.recover() == 1
    worker.drain()
    assert doc(conn)["status"] == "indexed"
    assert text_vectors(conn) == 6


# --- Watcher -----------------------------------------------------------------


def test_settle_queue_holds_a_path_until_quiet(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"x")
    queue = SettleQueue(settle_seconds=5)
    queue.touch(path, now=0)
    assert queue.settled(now=3) == []
    queue.touch(path, now=4)
    assert queue.settled(now=6) == []
    assert queue.settled(now=9) == [path]
    assert queue.settled(now=20) == []


def test_settle_queue_waits_for_a_file_still_growing(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"x")
    queue = SettleQueue(settle_seconds=5)
    queue.touch(path, now=0)
    path.write_bytes(b"xx")       # grew, but no event reached the queue
    assert queue.settled(now=6) == []
    assert queue.settled(now=12) == [path]


def test_settle_queue_releases_a_deleted_path(tmp_path):
    path = tmp_path / "gone.pdf"
    queue = SettleQueue(settle_seconds=1)
    queue.touch(path, now=0)
    assert queue.settled(now=2) == [path]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return None


def test_watcher_sees_real_events_and_skips_annotated_copies(settings):
    (settings.library_dir / "annotated").mkdir()
    watcher = LibraryWatcher(settings)
    watcher.start()
    try:
        add_pdf(settings, "annotated/copy.pdf", b"copy")
        add_pdf(settings, "notes.txt", b"text")
        target = add_pdf(settings, "book.pdf", b"content")
        assert wait_for(watcher.settled) == [target]
        time.sleep(0.4)
        assert watcher.settled() == []
    finally:
        watcher.stop()


def test_worker_loop_ingests_a_file_dropped_in_while_running(settings, monkeypatch):
    """End to end: scan, watch, settle, queue, stages, clean stop."""
    import app.ingest.worker as worker_module

    original_init = Worker.__init__

    def patched_init(self, conn, cfg, parser=None, embedder=None):
        original_init(self, conn, cfg, fake_parser, fake_embedder)

    monkeypatch.setattr(Worker, "__init__", patched_init)
    stop = threading.Event()
    thread = threading.Thread(target=worker_module.run, args=(settings, True, stop))
    thread.start()
    try:
        time.sleep(0.3)
        add_pdf(settings, "dropped.pdf", b"content")
        check = connect(settings.db_path)

        def indexed():
            row = check.execute("SELECT status FROM documents WHERE path = 'dropped.pdf'").fetchone()
            return row is not None and row["status"] == "indexed"

        assert wait_for(indexed, timeout=10)
        check.close()
    finally:
        stop.set()
        thread.join(10)
    assert not thread.is_alive()


def test_second_worker_refuses_to_start(settings):
    stop = threading.Event()
    first = threading.Thread(target=__import__("app.ingest.worker", fromlist=["run"]).run,
                             args=(settings, True, stop))
    first.start()
    try:
        assert wait_for((settings.data_dir / "worker.lock").exists)
        time.sleep(0.2)
        from app.ingest.worker import run
        with pytest.raises(SystemExit) as refused:
            run(settings, watch=False)
        assert refused.value.code == LOCKED          # the supervisor waits on this code, not a crash
    finally:
        stop.set()
        first.join(10)


# --- The formula stage -------------------------------------------------------


def real_pdf(settings, name="real.pdf", pages=2):
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=600, height=800)
        page.insert_text((110, 100), "x", fontname="hebo", fontsize=12)
        page.insert_text((500, 100), "(1.1)", fontname="helv", fontsize=12)
    path = settings.library_dir / name
    document.save(path)
    document.close()
    return path


def equation_parser(pdf: Path, out_dir: Path) -> dict:
    box = [100.0, 85.0, 400.0, 105.0]
    records = [
        {"order": 0, "kind": "heading", "text": "1 Start", "level": 1, "page": 1},
        {"order": 1, "kind": "paragraph", "text": "An opening sentence.", "page": 1,
         "bbox": [100.0, 40.0, 400.0, 60.0]},
        {"order": 2, "kind": "equation", "text": "$$\nx\n$$", "page": 1, "bbox": box,
         "bboxes": [{"page": 1, "bbox": box}]},
        {"order": 3, "kind": "equation", "text": "$$\nx\n$$", "page": 2, "bbox": box,
         "bboxes": [{"page": 2, "bbox": box}]},
    ]
    (out_dir / "blocks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return {"title": "Equations", "pages": 2}


def granite_reader(pdf: Path, page: int) -> list[dict]:
    return [{"latex": "x", "bbox": [100.0, 85.0, 500.0, 105.0]}]


def formula_worker(conn, settings, reader):
    return Worker(conn, settings, parser=equation_parser, embedder=fake_embedder,
                  formula_reader=reader, compile_check=lambda f: [True] * len(f))


def test_formula_stage_restores_fonts_and_numbers(conn, settings):
    scan_library(conn, settings)
    real_pdf(settings)
    scan_library(conn, settings)
    formula_worker(conn, settings, granite_reader).drain()
    assert doc(conn)["status"] == "indexed"
    rows = conn.execute("SELECT text FROM blocks WHERE kind = 'equation' ORDER BY id").fetchall()
    assert [r["text"] for r in rows] == ["$$\n\\mathbf{x} \\tag{1.1}\n$$"] * 2
    chunk = conn.execute("SELECT text FROM chunks").fetchone()["text"]
    assert "\\mathbf{x}" in chunk


def test_a_crash_mid_book_resumes_at_the_page_it_reached(conn, settings):
    real_pdf(settings, pages=2)
    scan_library(conn, settings)
    pages = []

    def crashes_on_page_two(pdf, page):
        pages.append(page)
        if page == 2:
            raise Crash()
        return granite_reader(pdf, page)

    worker = formula_worker(conn, settings, crashes_on_page_two)
    with pytest.raises(Crash):
        worker.drain()
    assert pages == [1, 2]

    pages.clear()
    worker = formula_worker(conn, settings, lambda pdf, page: pages.append(page) or granite_reader(pdf, page))
    worker.recover()
    worker.drain()
    assert pages == [2]
    assert doc(conn)["status"] == "indexed"


def test_rerunning_the_merge_reads_no_pages(conn, settings):
    real_pdf(settings)
    scan_library(conn, settings)
    formula_worker(conn, settings, granite_reader).drain()

    def never(pdf, page):
        raise AssertionError("the checkpoint should have this page")

    parse_jobs = conn.execute("SELECT id FROM jobs WHERE stage = 'parse'").fetchall()
    assert reindex(conn, settings, "formulas") == 1
    assert conn.execute("SELECT id FROM jobs WHERE stage = 'parse'").fetchall() == parse_jobs
    assert count(conn, "chunks") == 0 and count(conn, "chunk_vec") == 0
    formula_worker(conn, settings, never).drain()
    assert doc(conn)["status"] == "indexed"
    assert count(conn, "chunk_vec") > 0


def test_a_full_reindex_keeps_the_document_and_its_notes(conn, settings):
    real_pdf(settings)
    scan_library(conn, settings)
    formula_worker(conn, settings, granite_reader).drain()
    with conn:
        conn.execute("INSERT INTO notes (doc_id, body_md) VALUES (1, 'keep me')")
    parsed = settings.parsed_dir / doc(conn)["sha256"]
    assert parsed.exists()

    reindex(conn, settings)
    assert not parsed.exists()
    assert doc(conn)["status"] == "queued"
    assert jobs(conn) == [(s, "pending") for s in STAGES]
    formula_worker(conn, settings, granite_reader).drain()
    assert doc(conn)["status"] == "indexed"
    assert count(conn, "notes") == 1


def test_a_dropped_formula_is_stored_but_not_chunked(conn, settings):
    real_pdf(settings)
    scan_library(conn, settings)
    junk = lambda pdf, page: [{"latex": r"\\vdots \\, , \\, " * 9, "bbox": [100.0, 85.0, 500.0, 105.0]}]

    def junk_parser(pdf, out_dir):
        info = equation_parser(pdf, out_dir)
        text = (out_dir / "blocks.jsonl").read_text()
        (out_dir / "blocks.jsonl").write_text(text.replace("$$\\nx\\n$$", r"x \\intertext{a}"))
        return info

    Worker(conn, settings, parser=junk_parser, embedder=fake_embedder, formula_reader=junk,
           compile_check=lambda f: [True] * len(f)).drain()
    assert count(conn, "blocks") > 0
    stored = conn.execute("SELECT count(*) FROM blocks WHERE kind = 'equation_dropped'").fetchone()[0]
    assert stored == 2
    assert all("vdots" not in r["text"] for r in conn.execute("SELECT text FROM chunks"))


def misreading_parser(pdf, out_dir):
    info = equation_parser(pdf, out_dir)
    text = (out_dir / "blocks.jsonl").read_text()
    (out_dir / "blocks.jsonl").write_text(text.replace("$$\\nx\\n$$", "$$\\nw\\n$$"))
    return info


def misreading_worker(conn, settings, reader):
    return Worker(conn, settings, parser=misreading_parser, embedder=fake_embedder,
                  formula_reader=reader, compile_check=lambda f: [True] * len(f))


def misread(pdf, page):
    return [{"latex": "z", "bbox": [100.0, 85.0, 500.0, 105.0]}]


def test_an_unverified_formula_is_warned_about_and_cropped(conn, settings):
    real_pdf(settings)
    scan_library(conn, settings)
    misreading_worker(conn, settings, misread).drain()
    assert doc(conn)["status"] == "indexed"

    parsed = settings.parsed_dir / doc(conn)["sha256"]
    records = [json.loads(line) for line in (parsed / "blocks.jsonl").read_text().splitlines()]
    flagged = [r for r in records if r.get("unverified")]
    assert len(flagged) == 2
    assert all((parsed / r["crop"]).stat().st_size > 0 for r in flagged)

    blocks = conn.execute("SELECT text FROM blocks WHERE kind = 'equation'").fetchall()
    assert all("unverified" not in r["text"] for r in blocks)

    chunk = conn.execute("SELECT text, meta FROM chunks").fetchone()
    assert "[unverified formula, see page 1]" in chunk["text"]
    assert "[unverified formula, see page 2]" in chunk["text"]
    assert [item["page"] for item in json.loads(chunk["meta"])["unverified_formulas"]] == [1, 2]

    stats = json.loads((parsed / "formula_stats.json").read_text())
    assert stats["unverified"] == 2


def test_a_verified_book_saves_no_crops(conn, settings):
    real_pdf(settings)
    scan_library(conn, settings)
    formula_worker(conn, settings, granite_reader).drain()
    parsed = settings.parsed_dir / doc(conn)["sha256"]
    assert list((parsed / "formula_crops").iterdir()) == []
    assert "unverified" not in conn.execute("SELECT text FROM chunks").fetchone()["text"]


# --- The full flow: describe and summarize stages, approval, pausing (Phase 5c) ----------

from app.config import FigureDescriber  # noqa: E402
from app.ingest import figures as figure_module  # noqa: E402
from app.ingest.worker import add_late_stages  # noqa: E402

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


def parser_with_figures(count: int = 2):
    def parse(pdf: Path, out_dir: Path) -> dict:
        write_sidecar(out_dir)
        (out_dir / "figures").mkdir()
        with open(out_dir / "blocks.jsonl", "a", encoding="utf-8") as handle:
            for i in range(count):
                name = f"figures/figure_{i:03d}_page{i + 1}.png"
                (out_dir / name).write_bytes(TINY_PNG + bytes([i]))     # a different hash per figure
                handle.write(json.dumps({"order": 100 + i, "kind": "figure", "text": f"Figure {i}: a plot",
                                         "page": i + 1, "bbox": [1, 2, 3, 4], "image_path": name}) + "\n")
        return {"title": "Parsed title", "pages": 6}
    return parse


def hosted(settings):
    from dataclasses import replace
    return replace(settings, figure_describer=FigureDescriber("anthropic", "claude-sonnet-5-5", effort="high",
                                                              expected_tokens=(1302, 770)))


def local(settings):
    from dataclasses import replace
    return replace(settings, figure_describer=FigureDescriber("ollama", "gemma4:26b-a4b-it-qat"))


def fake_describer(described: list):
    def describe(ctx, rows):
        for row in rows:
            figure_module.store(ctx.conn, row["image_sha"], ctx.settings, f"A plot of figure {row['id']}.", 0.01)
            described.append(row["id"])
        return {"described": len(rows), "failed": 0, "spent_usd": 0.01 * len(rows)}
    return describe


def job(conn, stage, doc_id=1):
    return conn.execute("SELECT * FROM jobs WHERE doc_id = ? AND stage = ?", (doc_id, stage)).fetchone()


def approve(conn, usd, doc_id=1):
    conn.execute("UPDATE jobs SET waiting = NULL, approved_usd = ? WHERE doc_id = ? AND stage = 'describe'",
                 (usd, doc_id))
    conn.commit()


def test_a_hosted_describer_waits_for_an_ok_and_the_summary_doesnt(conn, settings):
    settings = hosted(settings)
    add_pdf(settings)
    scan_library(conn, settings)
    described = []
    Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas, describe_figures=fake_describer(described)).drain()

    waiting = job(conn, "describe")
    detail = json.loads(waiting["detail"])
    assert (waiting["status"], waiting["waiting"], waiting["attempts"]) == ("pending", "approval", 0)
    assert detail["figures"] == 2 and detail["usd"] == pytest.approx(2 * (1302 * 2 + 770 * 10) / 1e6)
    assert described == []
    assert job(conn, "summarize")["status"] == "done"
    assert doc(conn)["status"] == "indexed"


def test_the_ok_lets_the_figures_be_described_and_indexed(conn, settings):
    settings = hosted(settings)
    add_pdf(settings)
    scan_library(conn, settings)
    described = []
    worker = Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas,
                    describe_figures=fake_describer(described))
    worker.drain()
    approve(conn, json.loads(job(conn, "describe")["detail"])["usd"])
    worker.drain()

    assert len(described) == 2 and job(conn, "describe")["status"] == "done"
    texts = [row[0] for row in conn.execute("SELECT text FROM chunks WHERE kind = 'figure'")]
    assert texts and all(figure_module.MARKER in text for text in texts)
    unembedded = conn.execute("SELECT count(*) FROM chunks c LEFT JOIN chunk_vec v ON v.chunk_id = c.id "
                              "WHERE v.chunk_id IS NULL").fetchone()[0]
    assert unembedded == 0


def test_an_ok_for_less_than_the_new_estimate_asks_again(conn, settings):
    settings = hosted(settings)
    add_pdf(settings)
    scan_library(conn, settings)
    described = []
    worker = Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas,
                    describe_figures=fake_describer(described))
    worker.drain()
    approve(conn, 0.001)
    worker.drain()
    assert described == [] and job(conn, "describe")["waiting"] == "approval"


def test_a_local_describer_needs_no_ok(conn, settings):
    settings = local(settings)
    add_pdf(settings)
    scan_library(conn, settings)
    described = []
    Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas, describe_figures=fake_describer(described)).drain()
    assert len(described) == 2 and job(conn, "describe")["status"] == "done"


def test_a_failed_figure_retries_and_leaves_the_document_searchable(conn, settings):
    settings = local(settings)
    add_pdf(settings)
    scan_library(conn, settings)

    def failing(ctx, rows):
        return {"described": 0, "failed": len(rows), "spent_usd": 0}
    Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas, describe_figures=failing).drain()
    failed = job(conn, "describe")
    assert failed["status"] == "failed" and failed["attempts"] == settings.max_attempts
    assert doc(conn)["status"] == "indexed" and job(conn, "summarize")["status"] == "done"


def test_the_summary_tree_is_built_and_stored_by_the_worker(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    summaries = conn.execute("SELECT text, meta FROM chunks WHERE kind = 'summary'").fetchall()
    assert summaries and SUMMARY_CALLS
    assert json.loads(summaries[0]["meta"])["summarizer"] == "ollama:qwen3.8:27b"


def test_a_rebuilt_tree_comes_from_the_cache(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    calls = len(SUMMARY_CALLS)
    reindex(conn, settings, "chunk")
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    assert len(SUMMARY_CALLS) == calls
    assert conn.execute("SELECT count(*) FROM chunks WHERE kind = 'summary'").fetchone()[0] > 0


def test_every_document_is_searchable_before_any_is_summarized(conn, settings):
    add_pdf(settings, "a.pdf", b"one")
    add_pdf(settings, "b.pdf", b"two")
    scan_library(conn, settings)
    order = []
    worker = Worker(conn, settings, fake_parser, fake_embedder, no_formulas)
    while (next_job := worker.next_job()) is not None:
        order.append((next_job["doc_id"], next_job["stage"]))
        worker.run_one()
    assert order.index((2, "index")) < order.index((1, "summarize"))


def test_jobs_wait_while_a_chat_runs(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    marker = figure_module.busy_dir(settings) / "chat"
    marker.parent.mkdir(parents=True)
    marker.touch()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            marker.unlink()         # the chat ends
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas, sleep=sleep).run_one()
    assert len(sleeps) == 3
    assert job(conn, "parse")["status"] == "done"


def test_granite_pauses_between_pages(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    worker = Worker(conn, settings, fake_parser, fake_embedder, no_formulas)
    worker.run_one()                                    # parse
    calls = []
    worker.wait_for_chat = lambda: calls.append("wait")
    from app.ingest import worker as worker_module
    pages = list(range(1, 4))
    original = worker_module.pages_to_read
    worker_module.pages_to_read = lambda records, document: pages
    try:
        pdf = fitz.open()                               # a real PDF this time, so the formula stage opens it
        for _ in pages:
            pdf.new_page()
        pdf.save(settings.library_dir / "a.pdf")
        worker.formula_reader = lambda path, page: (calls.append(page), [])[1]
        worker.run_one()                                # formulas
    finally:
        worker_module.pages_to_read = original
    assert calls == ["wait", "wait", 1, "wait", 2, "wait", 3]   # once before claiming, then before each page


def test_books_indexed_before_the_new_stages_get_them_marked_done(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    worker = Worker(conn, settings, parser_with_figures(), fake_embedder, no_formulas,
                    describe_figures=fake_describer([]))
    worker.settings = local(settings)
    worker.drain()
    # As the five books are: indexed before describe and summarize were stages.
    conn.execute("DELETE FROM jobs WHERE stage IN ('describe', 'summarize')")
    conn.commit()
    assert add_late_stages(conn, local(settings)) == 2
    assert {job(conn, s)["status"] for s in ("describe", "summarize")} == {"done"}
    assert add_late_stages(conn, local(settings)) == 0


def test_a_book_without_summaries_gets_a_pending_summarize_job(conn, settings):
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    conn.execute("DELETE FROM jobs WHERE stage IN ('describe', 'summarize')")
    conn.execute("DELETE FROM chunks WHERE kind = 'summary'")
    conn.commit()
    add_late_stages(conn, settings)
    assert job(conn, "summarize")["status"] == "pending" and job(conn, "describe")["status"] == "done"


def test_a_dropped_summary_call_is_asked_again_after_the_chat(conn, settings):
    from app.ingest.summarize import Preempted
    add_pdf(settings)
    scan_library(conn, settings)
    calls = []

    def ask(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise Preempted
        return "A summary."
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas, summarizer_ask=ask).drain()
    assert calls[0] == calls[1]                 # the same prompt again
    assert job(conn, "summarize")["status"] == "done" and job(conn, "summarize")["attempts"] == 1


def test_summary_cache_rows_belong_to_their_document(conn, settings):
    from app.ingest.worker import link_summary_caches
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    rows = conn.execute("SELECT doc_id FROM summary_cache").fetchall()
    assert rows and {row[0] for row in rows} == {1}

    # As for the five books summarized before rows carried a doc_id: linked again at startup, no model asked.
    conn.execute("UPDATE summary_cache SET doc_id = NULL")
    conn.commit()
    calls = len(SUMMARY_CALLS)
    assert link_summary_caches(conn, settings) == {1: len(rows)}
    assert {row[0] for row in conn.execute("SELECT doc_id FROM summary_cache")} == {1}
    assert len(SUMMARY_CALLS) == calls
    assert link_summary_caches(conn, settings) == {}        # nothing left to link


def test_a_tree_the_cache_cant_rebuild_marks_nothing(conn, settings):
    from app.ingest.summarize import link_cache
    add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    conn.execute("UPDATE summary_cache SET doc_id = NULL")
    conn.execute("DELETE FROM summary_cache WHERE rowid = (SELECT max(rowid) FROM summary_cache)")
    conn.commit()
    assert link_cache(conn, settings, 1, "ollama:qwen3.8:27b") is None
    assert conn.execute("SELECT count(*) FROM summary_cache WHERE doc_id IS NOT NULL").fetchone()[0] == 0


def test_a_document_whose_file_is_gone_isnt_linked(conn, settings):
    from app.ingest.worker import link_summary_caches
    path = add_pdf(settings)
    scan_library(conn, settings)
    Worker(conn, settings, fake_parser, fake_embedder, no_formulas).drain()
    conn.execute("UPDATE summary_cache SET doc_id = NULL")
    conn.commit()
    path.unlink()
    assert link_summary_caches(conn, settings) == {}
