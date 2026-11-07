"""Feedback on answers (roadmap item 2): ratings, the review list and gold candidates."""
import json
import sqlite3

import pytest
import yaml

from app import feedback
from app.db.connection import connect
from tests.test_chat import FakeOllama, client_for, events_of, post, reply, settings  # noqa: F401 (fixture)


def asked(client, message="Hello") -> dict:
    """Ask a question; returns the ids and events of the turn."""
    events = events_of(post(client, message))
    meta = events[0][1]
    done = dict(events)[events[-1][0]]
    return {"question_id": meta["question_id"], "conversation_id": meta["conversation_id"],
            "last": events[-1][0], "data": done}


def rate(client, question_id, rating="down", note=""):
    return client.put(f"/api/questions/{question_id}/feedback", json={"rating": rating, "note": note})


def test_the_events_carry_the_question_and_answer_ids(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    turn = asked(client)
    assert turn["last"] == "done"
    conn = connect(settings.db_path)
    rows = conn.execute("SELECT id, role, option, details FROM messages ORDER BY id").fetchall()
    assert [row["id"] for row in rows] == [turn["question_id"], turn["data"]["answer_id"]]
    assert [row["option"] for row in rows] == ["fast", "fast"]
    assert json.loads(rows[1]["details"])["sources"] == []


def test_a_rating_copies_the_turn(settings):
    client = client_for(settings, FakeOllama(reply("Proper ", "time.")))
    turn = asked(client, "What is proper time?")
    response = rate(client, turn["question_id"], "up", "  Clear.  ")
    assert response.status_code == 200
    saved = response.json()
    assert saved["rating"] == "up" and saved["note"] == "Clear."
    assert saved["question"] == "What is proper time?" and saved["answer"] == "Proper time."
    assert saved["option"] == "fast" and saved["model"] == settings.chat_model and saved["hosted"] is False
    assert saved["error"] is None and saved["answer_id"] == turn["data"]["answer_id"]
    assert saved["asked_at"] and saved["answered_at"] and saved["status"] == "open"
    listed = client.get("/api/feedback").json()
    assert [item["id"] for item in listed["entries"]] == [saved["id"]]
    assert listed["counts"] == {"total": 1, "up": 1, "down": 0, "open": 1, "candidates": 0}


def test_rating_again_changes_the_one_entry(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    turn = asked(client)
    first = rate(client, turn["question_id"], "up").json()
    second = rate(client, turn["question_id"], "down", "Wrong sign.").json()
    assert second["id"] == first["id"] and second["rating"] == "down" and second["note"] == "Wrong sign."
    assert len(client.get("/api/feedback").json()["entries"]) == 1


def test_a_failed_answer_can_be_rated_with_its_error(settings):
    client = client_for(settings, FakeOllama(status=500))
    turn = asked(client)
    assert turn["last"] == "error"
    saved = rate(client, turn["question_id"], "down", "Ollama was down").json()
    assert saved["answer"] == "" and saved["answer_id"] is None
    assert saved["error"] == turn["data"]["message"] and saved["option"] == "fast"


def test_a_question_without_an_answer_or_an_error_is_refused(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    conn = connect(settings.db_path)
    conversation = conn.execute("INSERT INTO conversations (title) VALUES ('t')").lastrowid
    question = conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', 'Stopped')",
                            (conversation,)).lastrowid
    conn.commit()
    assert rate(client, question).status_code == 409


def test_unknown_questions_answers_and_ratings_are_refused(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    turn = asked(client)
    assert rate(client, 999).status_code == 404
    assert rate(client, turn["data"]["answer_id"]).status_code == 404       # an answer, not a question
    assert rate(client, turn["question_id"], "meh").status_code == 422
    assert client.put(f"/api/questions/{turn['question_id']}/feedback",
                      json={"rating": "up", "note": "x" * 4001}).status_code == 422
    assert client.patch("/api/feedback/999", json={"status": "done"}).status_code == 404
    assert client.get("/api/feedback?rating=sideways").status_code == 422


def test_entries_can_be_marked_done_and_filtered(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    up = rate(client, asked(client)["question_id"], "up").json()
    down = rate(client, asked(client)["question_id"], "down").json()
    assert client.patch(f"/api/feedback/{up['id']}", json={"status": "done"}).json()["status"] == "done"
    assert [item["id"] for item in client.get("/api/feedback").json()["entries"]] == [down["id"], up["id"]]
    assert [item["id"] for item in client.get("/api/feedback?status=open").json()["entries"]] == [down["id"]]
    assert [item["id"] for item in client.get("/api/feedback?rating=up").json()["entries"]] == [up["id"]]
    counts = client.get("/api/feedback").json()["counts"]
    assert counts == {"total": 2, "up": 1, "down": 1, "open": 1, "candidates": 0}


def test_only_a_thumbs_down_becomes_a_candidate(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    up = rate(client, asked(client)["question_id"], "up").json()
    assert client.patch(f"/api/feedback/{up['id']}", json={"candidate": True}).status_code == 409
    assert not (settings.data_dir / feedback.CANDIDATES_FILE).exists()


def stored_turn(conn: sqlite3.Connection, details: dict, question: str = "Derive the geodesic equation") -> int:
    """A question and its answer as the chat stores them, with sources: what search would have given."""
    doc = conn.execute("INSERT INTO documents (path, sha256, title, kind) VALUES ('Carroll.pdf', 'x', 'Spacetime', "
                       "'pdf')").lastrowid
    for item in details["sources"]:
        item["doc_id"] = doc
    conversation = conn.execute("INSERT INTO conversations (title) VALUES ('t')").lastrowid
    question_id = conn.execute("INSERT INTO messages (conversation_id, role, content, option) "
                               "VALUES (?, 'user', ?, 'deep')", (conversation, question)).lastrowid
    conn.execute("INSERT INTO messages (conversation_id, role, content, model, hosted, option, details) "
                 "VALUES (?, 'assistant', 'An answer', 'claude-opus-5-5', 1, 'deep', ?)",
                 (conversation, json.dumps(details)))
    conn.commit()
    return question_id


def source(number: int, page: int) -> dict:
    return {"id": number, "doc_id": 0, "doc_title": "Spacetime", "page_start": page, "page_end": page + 1,
            "focus_page": page, "heading_path": ["3 Curvature"], "regions": [], "chunk_id": 7}


def test_answer_details_mark_what_was_cited():
    finished = {"citations": [2], "web_citations": [1], "served_by": None, "done_reason": "end_turn",
                "cost_usd": 0.05, "spend": {"cost_usd": 0.07},
                "web_sources": [{"number": 1, "title": "PDG", "url": "https://pdg.lbl.gov", "fetched": "2026-11-05"},
                                {"number": 2, "title": "Wiki", "url": "https://en.wikipedia.org", "fetched": None}]}
    details = feedback.answer_details(finished, [source(1, 10), source(2, 40)])
    assert [item["cited"] for item in details["sources"]] == [False, True]
    assert "chunk_id" not in details["sources"][0]
    assert [page["cited"] for page in details["web_sources"]] == [True, False]
    assert details["cost_usd"] == 0.07          # the log's figure, every call included


def test_a_candidate_is_written_as_a_gold_stub_and_removed_again(settings):
    conn = connect(settings.db_path)
    finished = {"citations": [2], "web_citations": [1], "web_sources": [
        {"number": 1, "title": "Notes", "url": "https://example.org/geodesics", "fetched": None}]}
    question_id = stored_turn(conn, feedback.answer_details(finished, [source(1, 10), source(2, 40)]))
    client = client_for(settings, FakeOllama(reply("Hi")))
    down = rate(client, question_id, "down", "Dropped the Christoffel term.").json()
    assert down["sources"][1]["cited"] and down["web_sources"][0]["url"] == "https://example.org/geodesics"

    assert client.patch(f"/api/feedback/{down['id']}", json={"candidate": True}).json()["candidate"] is True
    path = settings.data_dir / feedback.CANDIDATES_FILE
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Gold-question candidates")
    stub = yaml.safe_load(text)["questions"][0]
    assert stub["id"] == f"candidate-{down['id']:03d}"
    assert stub["question"] == "Derive the geodesic equation"
    assert stub["expected_sources"] == [{"doc": "Carroll.pdf", "pages": [40]}]      # cited only, by file name
    assert stub["answer_web_pages"] == ["https://example.org/geodesics"]
    assert stub["key_points"] == [] and stub["category"] == ""
    assert "Dropped the Christoffel term." in stub["notes"] and "deep (claude-opus-5-5)" in stub["notes"]

    client.patch(f"/api/feedback/{down['id']}", json={"candidate": False})
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["questions"] == []


def test_a_thumbs_down_turned_up_stops_being_a_candidate(settings):
    conn = connect(settings.db_path)
    question_id = stored_turn(conn, feedback.answer_details({"citations": [1]}, [source(1, 10)]))
    client = client_for(settings, FakeOllama(reply("Hi")))
    down = rate(client, question_id, "down").json()
    client.patch(f"/api/feedback/{down['id']}", json={"candidate": True})
    up = rate(client, question_id, "up").json()
    assert up["candidate"] is False
    assert yaml.safe_load((settings.data_dir / feedback.CANDIDATES_FILE).read_text())["questions"] == []


def test_the_entry_outlives_its_conversation(settings):
    client = client_for(settings, FakeOllama(reply("Hi")))
    turn = asked(client)
    saved = rate(client, turn["question_id"], "up").json()
    conn = connect(settings.db_path)
    conn.execute("DELETE FROM conversations WHERE id = ?", (turn["conversation_id"],))
    conn.commit()
    kept = client.get("/api/feedback").json()["entries"][0]
    assert kept["id"] == saved["id"] and kept["question"] == "Hello" and kept["answer"] == "Hi"
    assert kept["question_id"] is None and kept["answer_id"] is None


def test_an_old_database_gets_the_new_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role TEXT NOT NULL, "
                "content TEXT NOT NULL, model TEXT, hosted INTEGER NOT NULL DEFAULT 0, citations TEXT, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    old.execute("INSERT INTO messages (conversation_id, role, content) VALUES (1, 'user', 'kept')")
    old.commit()
    old.close()
    conn = connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    assert {"option", "details"} <= columns
    assert conn.execute("SELECT content FROM messages").fetchone()[0] == "kept"
    assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 0
