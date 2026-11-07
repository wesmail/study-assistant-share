"""The library pane's states and actions: figure OKs and metadata.yaml entries."""
import json

import pytest
import yaml

from app.config import Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.library import Refused, approve_figures, document_state, library_rows, save_details, skip_figures, suggestion

STAGES = ("parse", "formulas", "chunk", "index", "describe", "summarize")
HEADER = """# What the app shows for each document.
# Keyed by file name in the library folder. Edit freely; the app rereads this file.

a.pdf:
  title: "Book A"   # a comment the app must keep
  year: 2020

"b c.pdf":
  title: "Book B"
"""


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "library").mkdir()
    return Settings(library_dir=tmp_path / "library", data_dir=tmp_path / "data", embed_batch=2)


@pytest.fixture
def conn(settings):
    db = connect(settings.db_path)
    db.execute("INSERT INTO documents (id, path, sha256, title, kind, status) VALUES (1, 'new.pdf', 'v2', 'Parsed', 'pdf', 'indexed')")
    db.commit()
    return db


def states(**statuses):
    """Jobs for document 1, version v2: stage=status, or stage=(status, waiting, detail)."""
    jobs = []
    for order, stage in enumerate(STAGES):
        value = statuses.get(stage, "pending")
        status, waiting, detail = value if isinstance(value, tuple) else (value, None, None)
        jobs.append({"stage": stage, "stage_order": order, "status": status, "waiting": waiting, "detail": detail,
                     "sha256": "v2", "progress": 3, "total": 10, "error": "Traceback\nValueError: bad page" if status == "failed" else None})
    return jobs


DOC = {"status": "indexed", "sha256": "v2"}
EARLY_DONE = {stage: "done" for stage in ("parse", "formulas", "chunk", "index")}


@pytest.mark.parametrize("statuses, expected", [
    ({}, "queued"),
    ({"parse": "running"}, "parsing"),
    ({"parse": "done", "formulas": "running"}, "parsing"),
    ({"parse": "done", "formulas": "done", "chunk": "running"}, "indexing"),
    ({"parse": "done", "formulas": "done"}, "indexing"),                  # between stages
    ({**EARLY_DONE, "describe": "running"}, "describing"),
    ({**EARLY_DONE, "summarize": "running"}, "summarizing"),
    ({**EARLY_DONE, "describe": "done", "summarize": "done"}, "done"),
    ({**EARLY_DONE, "describe": ("pending", "approval", '{"usd": 0.5}'), "summarize": "done"}, "done"),
    ({"parse": "done", "formulas": "failed"}, "failed"),
    ({**EARLY_DONE, "describe": "failed", "summarize": "done"}, "failed"),
])
def test_the_state_follows_the_jobs(statuses, expected):
    assert document_state(DOC, states(**statuses))["state"] == expected


def test_a_removed_file_says_so_whatever_its_jobs():
    assert document_state({"status": "missing", "sha256": "v2"}, states(**EARLY_DONE))["state"] == "removed"


def test_an_old_versions_jobs_dont_count():
    old = [{**job, "sha256": "v1", "status": "failed"} for job in states()]
    assert document_state(DOC, old + states(parse="running"))["state"] == "parsing"


def test_a_wait_for_an_ok_carries_the_estimate():
    shown = document_state(DOC, states(**EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}')))
    assert shown["waiting"] == {"stage": "describe", "figures": 3, "usd": 0.04}


def test_a_failure_shows_its_last_line_and_a_late_one_stays_searchable():
    shown = document_state(DOC, states(**EARLY_DONE, describe="failed", summarize="done"))
    assert shown["error"] == "ValueError: bad page" and shown["failed_stage"] == "describe" and shown["searchable"]


def test_a_running_job_shows_its_progress():
    assert document_state(DOC, states(parse="running"))["progress"] == {"done": 3, "total": 10}


def add_jobs(conn, **statuses):
    for job in states(**statuses):
        conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order, status, waiting, detail) VALUES (1, 'v2', ?, ?, ?, ?, ?)",
                     (job["stage"], job["stage_order"], job["status"], job["waiting"], job["detail"]))
    conn.commit()


def describe_job(conn):
    return conn.execute("SELECT * FROM jobs WHERE stage = 'describe'").fetchone()


def test_an_ok_releases_the_job_with_the_estimate_shown(conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    approve_figures(conn, 1, 0.04)
    job = describe_job(conn)
    assert (job["waiting"], job["approved_usd"], job["status"]) == (None, 0.04, "pending")


def test_an_ok_for_an_estimate_that_changed_is_refused(conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 30, "usd": 0.40}'))
    with pytest.raises(Refused, match=r"\$0\.40"):
        approve_figures(conn, 1, 0.04)
    assert describe_job(conn)["waiting"] == "approval"


def test_an_ok_with_nothing_waiting_is_refused(conn):
    add_jobs(conn, **EARLY_DONE, describe="done")
    with pytest.raises(Refused):
        approve_figures(conn, 1, 0.04)


def test_skipping_finishes_the_job_without_describing(conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    skip_figures(conn, 1)
    job = describe_job(conn)
    assert job["status"] == "done" and json.loads(job["detail"])["skipped"] is True


def test_the_list_asks_for_details_only_once_the_document_is_searchable(conn, settings):
    add_jobs(conn, parse="running")
    assert library_rows(conn, settings)[0]["needs_details"] is False
    conn.execute("DELETE FROM jobs")
    add_jobs(conn, **EARLY_DONE, describe="done", summarize="done")
    assert library_rows(conn, settings)[0]["needs_details"] is True
    (settings.library_dir / "metadata.yaml").write_text('new.pdf:\n  title: "Known"\n', encoding="utf-8")
    assert library_rows(conn, settings)[0]["needs_details"] is False


# --- metadata.yaml ------------------------------------------------------------------------

def fake_embed(texts, settings):
    return [[1.0] + [0.0] * 1023 for _ in texts]


def doc(conn, path="new.pdf"):
    conn.execute("UPDATE documents SET path = ?", (path,))
    return conn.execute("SELECT * FROM documents WHERE id = 1").fetchone()


def test_a_new_entry_is_appended_and_every_comment_kept(conn, settings):
    file = settings.library_dir / "metadata.yaml"
    file.write_text(HEADER, encoding="utf-8")
    save_details(conn, settings, doc(conn), {"title": " Quantum Field Theory ", "author": "A. Zee", "year": "2010",
                                              "publisher": ""}, fake_embed)
    text = file.read_text(encoding="utf-8")
    assert text.startswith(HEADER) and "# a comment the app must keep" in text
    assert yaml.safe_load(text)["new.pdf"] == {"title": "Quantum Field Theory", "author": "A. Zee", "year": 2010}
    row = conn.execute("SELECT title, author, year FROM documents WHERE id = 1").fetchone()
    assert tuple(row) == ("Quantum Field Theory", "A. Zee", 2010)


def test_an_existing_entry_is_replaced_in_place(conn, settings):
    file = settings.library_dir / "metadata.yaml"
    file.write_text(HEADER, encoding="utf-8")
    save_details(conn, settings, doc(conn, "b c.pdf"), {"title": "Book B, 2nd ed.", "year": 2021}, fake_embed)
    text = file.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert loaded["b c.pdf"] == {"title": "Book B, 2nd ed.", "year": 2021}
    assert loaded["a.pdf"] == {"title": "Book A", "year": 2020} and "# a comment the app must keep" in text
    assert text.count("b c.pdf") == 1


def test_a_file_name_with_quotes_and_colons_survives(conn, settings):
    name = 'Odd: "name" #1.pdf'
    save_details(conn, settings, doc(conn, name), {"title": "Odd"}, fake_embed)
    save_details(conn, settings, doc(conn, name), {"title": "Odd, again"}, fake_embed)
    assert yaml.safe_load((settings.library_dir / "metadata.yaml").read_text(encoding="utf-8")) == {name: {"title": "Odd, again"}}


@pytest.mark.parametrize("values, message", [
    ({"title": ""}, "title"), ({"title": "X", "year": "soon"}, "number"), ({"title": "X", "year": 20}, "between"),
])
def test_bad_details_are_refused_and_the_file_left_alone(conn, settings, values, message):
    file = settings.library_dir / "metadata.yaml"
    file.write_text(HEADER, encoding="utf-8")
    with pytest.raises(Refused, match=message):
        save_details(conn, settings, doc(conn), values, fake_embed)
    assert file.read_text(encoding="utf-8") == HEADER


def test_a_new_title_reaches_the_summaries_labels_and_their_vectors(conn, settings):
    label = '[AI-written summary of "Ch. 1", pages 1-9 of Parsed. Check details against the text passages.]'
    chunk = conn.execute("INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids) "
                         "VALUES (1, 'summary', 100000000, '[]', ?, 5, '[]')", (label + "\n\nIt covers X.",)).lastrowid
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk, serialize([0.0, 1.0] + [0.0] * 1022, 1024)))
    conn.commit()
    save_details(conn, settings, doc(conn), {"title": "Real Title"}, fake_embed)
    text = conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk,)).fetchone()[0]
    assert "pages 1-9 of Real Title. Check details" in text
    assert conn.execute("SELECT count(*) FROM chunk_vec WHERE chunk_id = ?", (chunk,)).fetchone()[0] == 1


def test_the_form_starts_from_the_entry_or_the_parsed_title(conn, settings):
    parsed = settings.data_dir / "parsed" / "v2"
    parsed.mkdir(parents=True)
    (parsed / "info.json").write_text('{"title": "From the pages"}', encoding="utf-8")
    assert suggestion(settings, doc(conn))["title"] == "From the pages"
    (settings.library_dir / "metadata.yaml").write_text(HEADER, encoding="utf-8")
    assert suggestion(settings, doc(conn, "a.pdf")) == {"title": "Book A", "author": None, "year": 2020, "publisher": None}


# --- Through the API ------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture
def client(conn, settings, tmp_path):
    return TestClient(create_app(settings, embed=fake_embed, frontend=tmp_path / "no-frontend"), base_url="http://localhost")


def test_the_list_carries_each_documents_state(client, conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'), summarize="running")
    [row] = client.get("/api/documents").json()["documents"]
    assert row["state"] == "summarizing" and row["waiting"]["usd"] == 0.04


def test_an_ok_through_the_api(client, conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    assert client.post("/api/documents/1/figures/approve", json={"usd": 0.04}).status_code == 200
    assert describe_job(conn)["waiting"] is None


def test_a_stale_ok_is_a_conflict_with_the_reason(client, conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 30, "usd": 0.40}'))
    response = client.post("/api/documents/1/figures/approve", json={"usd": 0.04})
    assert response.status_code == 409 and "$0.40" in response.json()["detail"]


def test_skip_and_unknown_documents(client, conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    assert client.post("/api/documents/1/figures/skip").status_code == 200
    assert client.post("/api/documents/1/figures/skip").status_code == 409
    assert client.post("/api/documents/99/figures/skip").status_code == 404


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},   # what a browser sends
    {"Origin": "http://localhost:3000", "Sec-Fetch-Site": "same-site"},   # another local page
    {"Origin": "null"},                                                    # a sandboxed iframe
    {"Origin": "https://evil.example"},                                    # an older browser
])
def test_another_site_cant_skip_or_remove_through_the_browser(client, conn, headers):
    """Review S4: a page elsewhere can POST without a body; CORS doesn't stop the request."""
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    for path in ("/api/documents/1/figures/skip", "/api/documents/1/remove", "/api/annotations/export/1"):
        response = client.post(path, headers=headers)
        assert response.status_code == 403, path
    assert describe_job(conn)["waiting"] is not None          # nothing was skipped


def test_control_the_apps_own_page_and_scripts_can_write(client, conn):
    add_jobs(conn, **EARLY_DONE, describe=("pending", "approval", '{"figures": 3, "usd": 0.04}'))
    page = {"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"}
    assert client.post("/api/documents/1/figures/skip", headers=page).status_code == 200
    assert client.get("/api/documents", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_details_round_trip_through_the_api(client, settings):
    assert client.get("/api/documents/1/details").json()["title"] is None
    saved = client.post("/api/documents/1/details", json={"title": "Real", "author": "A. B", "year": "1999"})
    assert saved.status_code == 200 and saved.json() == {"title": "Real", "author": "A. B", "year": 1999}
    assert client.get("/api/documents/1/details").json()["year"] == 1999
    assert client.get("/api/documents").json()["documents"][0]["title"] == "Real"
    assert client.post("/api/documents/1/details", json={"title": ""}).status_code == 409


# --- Removing a document whose file has gone (step 5) ------------------------------------------

from app.library import remove_document, removal_preview  # noqa: E402

VEC = serialize([1.0] + [0.0] * 1023, 1024)


def build_two_books(conn, settings):
    """Book 1 (gone: two file versions, its own and a shared figure image) beside book 2 (kept)."""
    conn.execute("DELETE FROM documents")
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, status) VALUES (1, 'gone.pdf', 'v2', 'Gone', 'pdf', 'missing')")
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, status) VALUES (2, 'kept.pdf', 'k1', 'Kept', 'pdf', 'indexed')")
    for doc_id in (1, 2):
        for seq, kind in enumerate(("text", "text", "figure", "summary")):
            chunk = conn.execute("INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids) "
                                 "VALUES (?, ?, ?, '[]', ?, 5, '[]')", (doc_id, kind, seq, f"book {doc_id} {kind} {seq} quark")).lastrowid
            conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk, VEC))
        conn.execute("INSERT INTO blocks (doc_id, \"order\", kind, page, text) VALUES (?, 0, 'paragraph', 1, 'x')", (doc_id,))
    conn.execute("INSERT INTO figures (doc_id, page, image_path, image_sha) VALUES (1, 1, 'figures/a.png', 'own')")
    conn.execute("INSERT INTO figures (doc_id, page, image_path, image_sha) VALUES (1, 2, 'figures/b.png', 'shared')")
    conn.execute("INSERT INTO figures (doc_id, page, image_path, image_sha) VALUES (2, 5, 'figures/c.png', 'shared')")
    for sha, cost in (("own", 0.011), ("shared", 0.012)):
        conn.execute("INSERT INTO figure_descriptions (image_sha, describer, prompt_version, text, cost_usd) "
                     "VALUES (?, 'anthropic:claude-sonnet-5-5', 1, 'A plot.', ?)", (sha, cost))
    for key, doc_id in (("s1", 1), ("s2", 1), ("s3", 2), ("s4", None)):
        conn.execute("INSERT INTO summary_cache (input_sha, summarizer, prompt_version, text, seconds, doc_id) "
                     "VALUES (?, 'ollama:qwen3.8:27b', 2, 'S.', 60, ?)", (key, doc_id))
    for sha in ("v1", "v2"):
        for order, stage in enumerate(STAGES):
            conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order, status) VALUES (1, ?, ?, ?, 'done')", (sha, stage, order))
    conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order, status) VALUES (2, 'k1', 'parse', 0, 'done')")
    conn.execute("INSERT INTO notes (doc_id, body_md) VALUES (1, 'my note')")
    conn.execute("INSERT INTO annotations (doc_id, page, bbox, quote, kind, origin, status) "
                 "VALUES (1, 1, '[0,0,1,1]', 'q', 'highlight', 'user', 'accepted')")
    conn.commit()
    for name in ("v1", "v2", "v2.tmp", "k1"):
        folder = settings.parsed_dir / name / "figures"
        folder.mkdir(parents=True)
        (folder / "a.png").write_bytes(b"x" * 1000)


def table_counts(conn, doc_id):
    count = lambda sql: conn.execute(sql, (doc_id,)).fetchone()[0]  # noqa: E731
    return {
        "chunks": count("SELECT count(*) FROM chunks WHERE doc_id = ?"),
        "vectors": count("SELECT count(*) FROM chunk_vec WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)"),
        "blocks": count("SELECT count(*) FROM blocks WHERE doc_id = ?"),
        "figures": count("SELECT count(*) FROM figures WHERE doc_id = ?"),
        "jobs": count("SELECT count(*) FROM jobs WHERE doc_id = ?"),
        "cache": count("SELECT count(*) FROM summary_cache WHERE doc_id = ?"),
    }


def gone(conn):
    return conn.execute("SELECT * FROM documents WHERE id = 1").fetchone()


def test_the_preview_lists_everything_removal_would_delete(conn, settings):
    build_two_books(conn, settings)
    preview = removal_preview(conn, settings, gone(conn))
    assert preview["chunks"] == {"text": 2, "figure": 1, "summary": 1}
    assert (preview["blocks"], preview["figures"], preview["jobs"]) == (1, 2, 12)
    assert (preview["figure_descriptions"], preview["figure_descriptions_usd"]) == (1, 0.011)    # not the shared one
    assert (preview["summary_cache"], preview["summary_cache_seconds"]) == (2, 120)
    assert sorted(preview["folders"]) == ["parsed/v1", "parsed/v2", "parsed/v2.tmp"]
    assert preview["folder_bytes"] == 3000 and preview["annotations"] == 1 and preview["notes_kept"] == 1


def test_removal_deletes_the_document_and_leaves_the_other_alone(conn, settings):
    build_two_books(conn, settings)
    kept_before = table_counts(conn, 2)
    remove_document(conn, settings, gone(conn))

    assert gone(conn) is None
    assert set(table_counts(conn, 1).values()) == {0}
    assert table_counts(conn, 2) == kept_before
    assert conn.execute("SELECT count(*) FROM chunk_vec").fetchone()[0] == kept_before["vectors"]
    keyword = {row[0] for row in conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'quark'")}
    assert keyword == {row[0] for row in conn.execute("SELECT id FROM chunks")}          # only book 2's
    assert [row[0] for row in conn.execute("SELECT image_sha FROM figure_descriptions")] == ["shared"]
    assert {row[0] for row in conn.execute("SELECT input_sha FROM summary_cache")} == {"s3", "s4"}
    assert not any((settings.parsed_dir / name).exists() for name in ("v1", "v2", "v2.tmp"))
    assert (settings.parsed_dir / "k1" / "figures" / "a.png").is_file()
    assert conn.execute("SELECT doc_id, body_md FROM notes").fetchone()[:] == (None, "my note")
    assert conn.execute("SELECT count(*) FROM annotations").fetchone()[0] == 0


@pytest.mark.parametrize("change, message", [
    ("UPDATE documents SET status = 'indexed' WHERE id = 1", "left library"),
    ("UPDATE jobs SET status = 'running' WHERE doc_id = 1 AND stage = 'summarize' AND sha256 = 'v2'", "still running"),
])
def test_removal_is_refused_unless_the_file_is_gone_and_nothing_runs(conn, settings, change, message):
    build_two_books(conn, settings)
    conn.execute(change)
    conn.commit()
    with pytest.raises(Refused, match=message):
        remove_document(conn, settings, gone(conn))
    assert gone(conn) is not None and (settings.parsed_dir / "v2").is_dir()


def test_removal_is_refused_when_the_file_came_back(conn, settings):
    build_two_books(conn, settings)
    (settings.library_dir / "gone.pdf").write_bytes(b"%PDF")
    with pytest.raises(Refused):
        removal_preview(conn, settings, gone(conn))


def test_removal_through_the_api(client, conn, settings):
    build_two_books(conn, settings)
    preview = client.get("/api/documents/1/removal").json()
    assert preview["chunks"]["text"] == 2
    response = client.post("/api/documents/1/remove")
    assert response.status_code == 200 and response.json()["removed"]["figures"] == 2
    assert [row["id"] for row in client.get("/api/documents").json()["documents"]] == [2]
    assert client.post("/api/documents/2/remove").status_code == 409


# --- The test collection (step 6) ----------------------------------------------------------------

def test_the_collection_comes_from_metadata_yaml(conn, settings):
    from app.ingest import metadata
    (settings.library_dir / "metadata.yaml").write_text('new.pdf:\n  title: "T"\n  collection: test\n', encoding="utf-8")
    metadata.refresh(conn, settings)
    assert library_rows(conn, settings)[0]["collection"] == "test"
    (settings.library_dir / "metadata.yaml").write_text('new.pdf:\n  title: "T"\n', encoding="utf-8")
    metadata.refresh(conn, settings)
    assert library_rows(conn, settings)[0]["collection"] is None


def test_saving_details_keeps_the_entrys_collection(conn, settings):
    file = settings.library_dir / "metadata.yaml"
    file.write_text('# keep me\nnew.pdf:\n  title: "Old"\n  collection: test\n', encoding="utf-8")
    save_details(conn, settings, doc(conn), {"title": "New", "year": 2001}, fake_embed)
    assert yaml.safe_load(file.read_text(encoding="utf-8"))["new.pdf"] == {"title": "New", "year": 2001, "collection": "test"}
    assert file.read_text(encoding="utf-8").startswith("# keep me\n")


def test_library_settings_default_and_change(conn):
    from app.library import hidden_collections, library_settings, set_library_settings
    assert library_settings(conn) == {"hide_test_collection": False} and hidden_collections(conn) == ()
    set_library_settings(conn, {"hide_test_collection": True})
    assert hidden_collections(conn) == ("test",)
    with pytest.raises(Refused):
        set_library_settings(conn, {"hide_test_collection": "yes"})
