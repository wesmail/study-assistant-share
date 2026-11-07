"""The reader's feedback on answers (roadmap item 2): thumbs up or down, an optional note,
and a review list. A thumbs-down can be marked as a gold-question candidate.

Each rating keeps a copy of its turn (the question, the answer or its error, the option and
model, the sources and web pages, the times), taken from the database rather than from the
browser, so the review list shows what was really asked and answered, and doesn't depend on
the conversation staying around.

Candidates are written to data/gold_candidates.yaml as stubs in the questions.yaml format.
The app never touches eval/questions.yaml: the 35 gold questions change only by hand.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import yaml

from app.errors import NotFound, Refused  # noqa: F401  (re-exported: feedback.Refused)
from app.config import Settings

RATINGS = ("up", "down")
STATUSES = ("open", "done")
NOTE_CHARS = 4000
CANDIDATES_FILE = "gold_candidates.yaml"
CANDIDATES_HEADER = """\
# Gold-question candidates from the feedback list: answers rated thumbs down and marked as
# candidates. The app rewrites this whole file whenever a candidate is added or removed, so
# don't edit it here. To use one, copy it into eval/questions.yaml by hand, then give it a
# real id, its category (lookup | derivation | overview | code | figure), difficulty,
# language (en | de) and key points, and check its pages.
#
# expected_sources are the pages the rated answer cited (the page each source was found by).
# For a wrong answer they may be the wrong pages. answer_web_pages are the web pages it
# cited; a web question belongs in eval/web_questions.yaml instead.
"""


def source_summary(source: dict, cited: bool) -> dict:
    """What the review list needs of one source: enough to show it and open its page."""
    return {
        "id": source["id"],
        "doc_id": source["doc_id"],
        "doc_title": source.get("doc_title") or "",
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "focus_page": source.get("focus_page"),
        "heading_path": source.get("heading_path") or [],
        "regions": source.get("regions") or [],
        "cited": cited,
    }


def answer_details(finished: dict, sources: list[dict]) -> dict:
    """messages.details for an answer: every source it was given and every web page it read,
    each marked cited or not, and how it was served."""
    cited = set(finished.get("citations") or [])
    web_cited = set(finished.get("web_citations") or [])
    spend = finished.get("spend") or {}
    return {
        "sources": [source_summary(source, source["id"] in cited) for source in sources],
        "web_sources": [{"number": page["number"], "title": page.get("title") or "", "url": page["url"],
                         "fetched": page.get("fetched"), "cited": page["number"] in web_cited}
                        for page in finished.get("web_sources") or []],
        "served_by": finished.get("served_by"),
        "fallback_from": finished.get("fallback_from"),
        "done_reason": finished.get("done_reason"),
        "cost_usd": spend.get("cost_usd", finished.get("cost_usd")),
        **({"checks": {key: checks.get(key) for key in ("badge", "summary", "source", "attempt", "attempts",
                                                       "stopped", "none_passed", "claims")},
            "replaced": [{"attempt": item["attempt"], "badge": item["checks"]["badge"], "failing": item["failing"]}
                         for item in finished.get("replaced") or []]}
           if (checks := finished.get("checks")) else {}),
    }


def failure_details(error: dict, sources: list[dict], web_sources: list[dict]) -> dict:
    """messages.details for a question whose answer failed: the error, and what it had found."""
    spend = error.get("spend") or {}
    return {
        "error": error.get("message") or "The answer failed",
        "sources": [source_summary(source, False) for source in sources],
        "web_sources": [{"number": page["number"], "title": page.get("title") or "", "url": page["url"],
                         "fetched": page.get("fetched"), "cited": False} for page in web_sources],
        "cost_usd": spend.get("cost_usd"),
    }


def answer_row(conn: sqlite3.Connection, question: sqlite3.Row) -> sqlite3.Row | None:
    """The reply to a question: the next message in its conversation, if that's an answer."""
    after = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? AND id > ? ORDER BY id LIMIT 1",
        (question["conversation_id"], question["id"]),
    ).fetchone()
    return after if after is not None and after["role"] == "assistant" else None


def entry(row: sqlite3.Row) -> dict:
    details = json.loads(row["details"] or "{}")
    return {
        "id": row["id"],
        "question_id": row["question_id"],
        "answer_id": row["answer_id"],
        "rating": row["rating"],
        "note": row["note"],
        "status": row["status"],
        "candidate": bool(row["candidate"]),
        "question": row["question"],
        "answer": row["answer"],
        "error": row["error"],
        "option": row["option"],
        "model": row["model"],
        "hosted": bool(row["hosted"]),
        "served_by": details.get("served_by"),
        "sources": details.get("sources", []),
        "web_sources": details.get("web_sources", []),
        "asked_at": row["asked_at"],
        "answered_at": row["answered_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get(conn: sqlite3.Connection, feedback_id: int) -> dict:
    row = conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
    if row is None:
        raise NotFound(f"No feedback {feedback_id}")
    return entry(row)


def rate(conn: sqlite3.Connection, settings: Settings, question_id: int, rating: str, note: str = "") -> dict:
    """Save the reader's rating of the answer to a question, or change it.

    The first rating copies the turn; a later one changes only the rating and the note.
    A thumbs-down turned into a thumbs-up stops being a candidate.
    """
    if rating not in RATINGS:
        raise Refused(f"A rating is up or down, not {rating!r}")
    note = (note or "").strip()[:NOTE_CHARS]
    question = conn.execute("SELECT * FROM messages WHERE id = ? AND role = 'user'", (question_id,)).fetchone()
    if question is None:
        raise NotFound(f"No question {question_id}")
    answer = answer_row(conn, question)
    if answer is not None:
        details = json.loads(answer["details"] or "{}")
        error = None
    else:
        details = json.loads(question["details"] or "{}")
        error = details.pop("error", None)
        if error is None:
            raise Refused("This question has no answer yet, or it was stopped before one came")
    existing = conn.execute("SELECT id, candidate FROM feedback WHERE question_id = ?", (question_id,)).fetchone()
    if existing is None:
        feedback_id = conn.execute(
            "INSERT INTO feedback (question_id, answer_id, rating, note, question, answer, error, option, model, "
            "hosted, details, asked_at, answered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (question_id, answer["id"] if answer else None, rating, note, question["content"],
             answer["content"] if answer else "", error, (answer["option"] if answer else None) or question["option"],
             answer["model"] if answer else None, answer["hosted"] if answer else 0, json.dumps(details),
             question["created_at"], answer["created_at"] if answer else None),
        ).lastrowid
        candidate_changed = False
    else:
        feedback_id = existing["id"]
        candidate_changed = bool(existing["candidate"]) and rating == "up"
        conn.execute(
            "UPDATE feedback SET rating = ?, note = ?, candidate = candidate AND ? = 'down', "
            "updated_at = datetime('now') WHERE id = ?", (rating, note, rating, feedback_id))
    conn.commit()
    if candidate_changed:
        write_candidates(conn, settings)
    return get(conn, feedback_id)


def entries(conn: sqlite3.Connection, rating: str | None = None, status: str | None = None) -> dict:
    """The review list, newest first, with counts for the filters."""
    where, values = [], []
    if rating:
        where.append("rating = ?")
        values.append(rating)
    if status:
        where.append("status = ?")
        values.append(status)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(f"SELECT * FROM feedback {clause} ORDER BY id DESC", values).fetchall()
    counts = conn.execute(
        "SELECT count(*) AS total, "
        "coalesce(sum(rating = 'up'), 0) AS up, coalesce(sum(rating = 'down'), 0) AS down, "
        "coalesce(sum(status = 'open'), 0) AS open, coalesce(sum(candidate), 0) AS candidates FROM feedback"
    ).fetchone()
    return {"entries": [entry(row) for row in rows], "counts": dict(counts)}


def update(conn: sqlite3.Connection, settings: Settings, feedback_id: int,
           status: str | None = None, candidate: bool | None = None) -> dict:
    """Mark an entry done or open again, or make a thumbs-down a gold candidate or not."""
    current = get(conn, feedback_id)
    if status is not None and status not in STATUSES:
        raise Refused(f"A status is open or done, not {status!r}")
    if candidate and current["rating"] != "down":
        raise Refused("Only a thumbs-down can become a gold candidate")
    if status is not None:
        conn.execute("UPDATE feedback SET status = ?, updated_at = datetime('now') WHERE id = ?",
                     (status, feedback_id))
    if candidate is not None:
        conn.execute("UPDATE feedback SET candidate = ?, updated_at = datetime('now') WHERE id = ?",
                     (int(candidate), feedback_id))
    conn.commit()
    if candidate is not None and candidate != current["candidate"]:
        write_candidates(conn, settings)
    return get(conn, feedback_id)


def candidate_stub(conn: sqlite3.Connection, item: dict) -> dict:
    """One candidate in the questions.yaml format, with what's known filled in."""
    pages: dict[str, set[int]] = {}
    for source in item["sources"]:
        page = source.get("focus_page") or source.get("page_start")
        if not source["cited"] or not page:
            continue
        row = conn.execute("SELECT path FROM documents WHERE id = ?", (source["doc_id"],)).fetchone()
        name = row["path"] if row else source["doc_title"]
        pages.setdefault(name, set()).add(page)
    web = [page["url"] for page in item["web_sources"] if page["cited"]]
    said = f"Reader's note: {item['note']}" if item["note"] else "No note."
    failed = f" It failed: {item['error']}" if item["error"] else ""
    stub = {
        "id": f"candidate-{item['id']:03d}",
        "category": "",
        "difficulty": None,
        "language": "",
        "question": item["question"],
        "expected_sources": [{"doc": name, "pages": sorted(found)} for name, found in sorted(pages.items())],
        "key_points": [],
        "notes": (f"From feedback {item['id']}, asked {item['asked_at']} UTC: thumbs down on "
                  f"{item['option'] or 'an unknown option'} ({item['model'] or 'no model'}).{failed} {said}"),
    }
    if web:
        stub["answer_web_pages"] = web
    return stub


def write_candidates(conn: sqlite3.Connection, settings: Settings) -> Path:
    """Rewrite data/gold_candidates.yaml from every candidate, oldest first. Written to a
    temporary file and moved into place, so a reader never sees half a file."""
    rows = conn.execute("SELECT * FROM feedback WHERE candidate = 1 ORDER BY id").fetchall()
    stubs = [candidate_stub(conn, entry(row)) for row in rows]
    path = settings.data_dir / CANDIDATES_FILE
    temporary = path.with_suffix(".yaml.tmp")
    body = yaml.safe_dump({"questions": stubs}, sort_keys=False, allow_unicode=True, width=100)
    temporary.write_text(CANDIDATES_HEADER + "\n" + body, encoding="utf-8")
    os.replace(temporary, path)
    return path
