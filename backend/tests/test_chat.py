"""The chat endpoint, with Ollama replaced by a canned stream."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.connection import connect
from app.ingest.chunker import estimate_tokens
from app.llm import hosted
from app.llm.ollama import Delta, chat_payload
from app.chat import last_citations, last_turn, recent_history
from app.ingest.embed import serialize
from app.llm.prompts import SYSTEM_PROMPT
from app.main import ALLOWED_HOSTS, HOST, create_app, sse


def ndjson(*events: dict) -> bytes:
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


def reply(*words: str, thinking: str = "") -> bytes:
    events = [{"message": {"thinking": thinking}, "done": False}] if thinking else []
    events += [{"message": {"content": word}, "done": False} for word in words]
    events.append({"message": {"content": ""}, "done": True, "done_reason": "stop",
                   "prompt_eval_count": 11, "eval_count": len(words)})
    return ndjson(*events)


class FakeOllama(httpx.MockTransport):
    def __init__(self, body: bytes = b"", status: int = 200):
        self.requests: list[dict] = []
        super().__init__(lambda request: self._answer(request, body, status))

    def _answer(self, request: httpx.Request, body: bytes, status: int) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(status, content=body)


@pytest.fixture
def settings(tmp_path):
    # These tests pin the path without tools; test_tool_loop.py covers the one with them.
    made = Settings(data_dir=tmp_path, library_dir=tmp_path, tools_enabled=False)
    connect(made.db_path).close()
    return made


def embed_nothing(texts, settings):
    return [[1.0] + [0.0] * 1023 for _ in texts]


def client_for(settings, ollama) -> TestClient:
    return TestClient(create_app(settings, transport=ollama, embed=embed_nothing), base_url="http://localhost")


def events_of(response) -> list[tuple[str, dict]]:
    found = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        found.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return found


def post(client, message="Hello", **extra):
    return client.post("/api/chat", json={"message": message, **extra})


def names(events) -> list[str]:
    return [name for name, _ in events]


def test_tokens_arrive_one_event_at_a_time(settings):
    events = events_of(post(client_for(settings, FakeOllama(reply("Pro", "per", " time")))))
    assert names(events) == ["meta", "sources", "token", "token", "token", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == "Proper time"


def test_the_response_is_an_event_stream(settings):
    response = post(client_for(settings, FakeOllama(reply("Hi"))))
    assert response.headers["content-type"].startswith("text/event-stream")


def test_thinking_stays_separate_from_the_answer(settings):
    events = events_of(post(client_for(settings, FakeOllama(reply("Hi", thinking="Let me see")))))
    assert ("thinking", {"text": "Let me see"}) in events
    assert [data for name, data in events if name == "token"] == [{"text": "Hi"}]


def test_done_reports_how_the_model_finished(settings):
    events = events_of(post(client_for(settings, FakeOllama(reply("Hi")))))
    assert (events[-1][0], events[-1][1]["done_reason"]) == ("done", "stop")
    assert (events[-1][1]["prompt_tokens"], events[-1][1]["output_tokens"]) == (11, 1)


def test_every_request_sets_the_context_size_and_thinking_effort():
    payload = chat_payload([], Settings(chat_num_ctx=12345, chat_think="medium"))
    assert payload["options"]["num_ctx"] == 12345
    assert payload["think"] == "medium"


def test_the_model_comes_from_settings():
    assert chat_payload([], Settings(chat_model="qwen3.6:35b-a3b"))["model"] == "qwen3.6:35b-a3b"


def test_defaults_match_phase_0():
    assert (Settings().chat_model, Settings().chat_think) == ("gpt-oss:20b", "medium")


def test_an_answer_with_only_thinking_is_an_error_not_a_blank_reply(settings):
    body = ndjson({"message": {"thinking": "loop"}, "done": False},
                  {"message": {"content": ""}, "done": True, "done_reason": "length"})
    events = events_of(post(client_for(settings, FakeOllama(body))))
    assert names(events)[-1] == "error"
    assert "8,192-token limit" in events[-1][1]["message"]
    assert events[-1][1]["cut_off"] is True


def test_an_empty_answer_is_not_stored(settings):
    body = ndjson({"message": {"content": ""}, "done": True, "done_reason": "stop"})
    post(client_for(settings, FakeOllama(body)))
    conn = connect(settings.db_path)
    assert conn.execute("SELECT count(*) FROM messages WHERE role='assistant'").fetchone()[0] == 0


def test_an_ollama_failure_reaches_the_reader(settings):
    events = events_of(post(client_for(settings, FakeOllama(b'{"error":"boom"}', status=500))))
    assert names(events) == ["meta", "sources", "error"]
    assert "boom" in events[-1][1]["message"]


def test_an_error_inside_the_stream_reaches_the_reader(settings):
    events = events_of(post(client_for(settings, FakeOllama(ndjson({"error": "out of memory"})))))
    assert "out of memory" in events[-1][1]["message"]


def test_a_conversation_is_stored_with_the_model_name(settings):
    conversation = events_of(post(client_for(settings, FakeOllama(reply("Hi")))))[0][1]
    conn = connect(settings.db_path)
    rows = conn.execute(
        "SELECT role, content, model FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation["conversation_id"],),
    ).fetchall()
    assert [tuple(row) for row in rows] == [("user", "Hello", None), ("assistant", "Hi", settings.chat_model)]


def test_a_follow_up_sends_the_earlier_turns(settings):
    ollama = FakeOllama(reply("Hi"))
    client = client_for(settings, ollama)
    conversation = events_of(post(client, "First"))[0][1]["conversation_id"]
    post(client, "Second", conversation_id=conversation)
    sent = ollama.requests[-1]["messages"]
    assert [m["content"] for m in sent[1:3]] == ["First", "Hi"]
    assert "Question: Second" in sent[3]["content"]


def test_an_unknown_conversation_is_refused(settings):
    assert post(client_for(settings, FakeOllama(reply("Hi"))), conversation_id=99).status_code == 404


def test_an_empty_message_is_refused(settings):
    assert post(client_for(settings, FakeOllama(reply("Hi"))), "").status_code == 422


def test_history_keeps_the_newest_turns_that_fit():
    rows = [{"role": "user", "content": word * 400} for word in ("old ", "mid ", "new ")]
    kept = recent_history(rows, 2 * estimate_tokens(rows[0]["content"]))
    assert [row["content"][:3] for row in kept] == ["mid", "new"]


def test_the_server_binds_to_loopback():
    assert HOST == "127.0.0.1"


def test_a_foreign_host_header_is_refused(settings):
    client = TestClient(create_app(settings, transport=FakeOllama(reply("Hi")), embed=embed_nothing),
                        base_url="http://evil.example")
    assert post(client).status_code == 400


@pytest.mark.parametrize("host", ALLOWED_HOSTS)
def test_localhost_names_are_allowed(settings, host):
    client = TestClient(create_app(settings, transport=FakeOllama(reply("Hi")), embed=embed_nothing),
                        base_url=f"http://{host}")
    assert post(client).status_code == 200


def test_sse_framing_ends_each_event_with_a_blank_line():
    assert sse("token", {"text": "é"}) == 'event: token\ndata: {"text": "é"}\n\n'


MARKED_TEXT = "Intro\n\n[unverified formula, see page 7]\n$$ a = b $$"


@pytest.fixture
def indexed(settings, tmp_path):
    """One document with one chunk, one block and an unverified formula."""
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    (settings.library_dir / "metadata.yaml").write_text("book.pdf:\n  title: A Book\n")
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'book.pdf', 'abc', 'A Book', 'pdf', 10, 'indexed')")
    conn.execute("INSERT INTO blocks (id, doc_id, \"order\", kind, page, bbox, bboxes_json, text) "
                 "VALUES (5, 1, 1, 'equation', 7, '[1, 2, 3, 4]', ?, '$$ a = b $$')",
                 ('[{"page": 7, "bbox": [1, 2, 3, 4]}]',))
    conn.execute(
        "INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, "
        "page_start, page_end, meta) VALUES (1, 1, 'text', 1, ?, ?, 10, '[5]', 7, 7, ?)",
        ('["Ch. 1"]', MARKED_TEXT,
         '{"unverified_formulas": [{"page": 7, "crop": "formula_crops/000003.png", "text_layer": "a=b"}]}'),
    )
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (1, ?)",
                 (serialize(embed_nothing(["x"], None)[0], 1024),))
    conn.commit()
    conn.close()
    return settings


def test_the_source_list_comes_before_the_answer(indexed):
    events = events_of(post(client_for(indexed, FakeOllama(reply("Hi")))))
    assert names(events)[:2] == ["meta", "sources"]
    source = events[1][1]["sources"][0]
    assert (source["id"], source["doc_title"], source["page_start"]) == (1, "A Book", 7)
    assert source["regions"] == [{"page": 7, "bbox": [1, 2, 3, 4]}]


def test_the_model_gets_the_system_prompt_and_the_marked_source(indexed):
    ollama = FakeOllama(reply("Hi"))
    post(client_for(indexed, ollama))
    system, user = ollama.requests[0]["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[unverified formula, see page 7]\nRaw PDF symbols, layout lost: a=b" in user["content"]
    assert "$$ a = b $$" not in user["content"]


def test_an_invented_citation_never_reaches_the_reader(indexed):
    events = events_of(post(client_for(indexed, FakeOllama(reply("True [S", "1] false [S", "9].")))))
    text = "".join(data["text"] for name, data in events if name == "token")
    assert text == "True [S1] false."
    assert (events[-1][1]["citations"], events[-1][1]["dropped_citations"]) == ([1], [9])


def test_a_cited_source_shows_its_unverified_crop(indexed):
    events = events_of(post(client_for(indexed, FakeOllama(reply("See [S1].")))))
    assert events[-1][1]["unverified"] == [{
        "source": 1, "doc_id": 1, "page": 7, "url": "/api/documents/1/crops/000003.png", "cited": True}]


def test_an_answer_that_cites_nothing_still_shows_the_crop(indexed):
    """gpt-oss copied flagged LaTeX with no citation, so citations can't be the gate."""
    crops = events_of(post(client_for(indexed, FakeOllama(reply("No citation")))))[-1][1]["unverified"]
    assert [(crop["page"], crop["cited"]) for crop in crops] == [(7, False)]


def test_a_stored_answer_keeps_only_the_cited_sources(indexed):
    post(client_for(indexed, FakeOllama(reply("See [S1] and [S2]."))))
    row = connect(indexed.db_path).execute(
        "SELECT content, citations FROM messages WHERE role = 'assistant'").fetchone()
    assert row["content"] == "See [S1] and."
    assert [source["id"] for source in json.loads(row["citations"])] == [1]


def test_a_failed_search_is_reported_not_answered_without_sources(settings):
    def dead(texts, settings):
        raise OSError("connection refused")

    ollama = FakeOllama(reply("Hi"))
    client = TestClient(create_app(settings, transport=ollama, embed=dead), base_url="http://localhost")
    events = events_of(post(client))
    assert names(events)[-1] == "error" and ollama.requests == []


def test_chat_can_be_scoped_to_a_document(indexed):
    events = events_of(post(client_for(indexed, FakeOllama(reply("Hi"))), doc_ids=[2]))
    assert events[1][1]["sources"] == []


def test_documents_are_listed_with_their_chunk_count(indexed):
    body = client_for(indexed, FakeOllama()).get("/api/documents").json()
    assert body["documents"][0]["chunks"] == 1
    assert (body["documents"][0]["title"], body["documents"][0]["status"]) == ("A Book", "indexed")


def test_a_document_file_is_served(indexed):
    (indexed.library_dir / "book.pdf").write_bytes(b"%PDF-1.4 pretend")
    response = client_for(indexed, FakeOllama()).get("/api/documents/1/file")
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_a_document_file_supports_range_requests(indexed):
    (indexed.library_dir / "book.pdf").write_bytes(b"%PDF-1.4 pretend")
    response = client_for(indexed, FakeOllama()).get("/api/documents/1/file", headers={"Range": "bytes=0-3"})
    assert (response.status_code, response.content) == (206, b"%PDF")


def test_a_path_outside_the_library_is_not_served(indexed, tmp_path_factory):
    secret = tmp_path_factory.mktemp("elsewhere") / "secret.pdf"
    secret.write_bytes(b"secret")
    conn = connect(indexed.db_path)
    conn.execute("UPDATE documents SET path = ? WHERE id = 1", (str(secret),))
    conn.commit()
    assert client_for(indexed, FakeOllama()).get("/api/documents/1/file").status_code == 404


def test_a_missing_document_is_a_404(indexed):
    assert client_for(indexed, FakeOllama()).get("/api/documents/9/file").status_code == 404


def test_a_crop_is_served(indexed):
    folder = indexed.parsed_dir / "abc" / "formula_crops"
    folder.mkdir(parents=True)
    (folder / "000003.png").write_bytes(b"png")
    response = client_for(indexed, FakeOllama()).get("/api/documents/1/crops/000003.png")
    assert (response.status_code, response.headers["content-type"]) == (200, "image/png")


@pytest.mark.parametrize("name", ["..%2F..%2Fsecret.png", "000003.txt", "a.png", "missing1.png"])
def test_a_crop_name_cant_leave_its_folder(indexed, name):
    assert client_for(indexed, FakeOllama()).get(f"/api/documents/1/crops/{name}").status_code == 404


def test_the_built_frontend_is_served_at_the_root(settings, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<title>Study assistant</title>")
    client = TestClient(create_app(settings, embed=embed_nothing, frontend=dist), base_url="http://localhost")
    assert "Study assistant" in client.get("/").text
    assert client.get("/api/documents").status_code == 200


def test_the_api_works_without_a_built_frontend(settings, tmp_path):
    client = TestClient(create_app(settings, embed=embed_nothing, frontend=tmp_path / "none"),
                        base_url="http://localhost")
    assert client.get("/api/documents").status_code == 200


def test_a_first_run_with_no_database_shows_an_empty_library(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library")
    client = TestClient(create_app(settings, embed=embed_nothing, frontend=tmp_path / "none"),
                        base_url="http://localhost")
    answer = client.get("/api/documents")
    assert answer.status_code == 200 and answer.json()["documents"] == []
    assert client.get("/api/search", params={"q": "anything"}).json()["hits"] == []


def deep_reply(seen):
    async def fake(messages, settings, purpose="chat", fallback=True):
        seen.append(messages)
        yield Delta(text="Deep answer.")
        yield Delta(done=True, done_reason="end_turn", model="claude-sonnet-5-5", cost_usd=0.01)
    return fake


def stream_events(response) -> list[tuple[str, dict]]:
    found = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        found.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return found


def test_fast_is_the_default_and_stays_local(settings):
    client = client_for(settings, FakeOllama(reply("Local.")))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hosted, "answer_stream", lambda *a, **k: pytest.fail("fast mode went to the hosted model"))
        events = stream_events(client.post("/api/chat", json={"message": "Hello?"}))
    assert events[0][1]["hosted"] is False and events[0][1]["model"] == settings.chat_model
    assert events[-1][1]["hosted"] is False


def test_deep_mode_says_it_is_hosted_and_stores_who_answered(settings, monkeypatch):
    monkeypatch.setattr(hosted, "answer_stream", deep_reply([]))
    client = client_for(settings, FakeOllama())
    events = stream_events(client.post("/api/chat", json={"message": "Hello?", "mode": "deep"}))
    assert events[0][1]["hosted"] is True and events[0][1]["model"] == settings.hosted_model
    conn = connect(settings.db_path)
    row = conn.execute("SELECT model, hosted FROM messages WHERE role = 'assistant'").fetchone()
    assert (row["model"], row["hosted"]) == ("claude-sonnet-5-5", 1)


def turn(role, content):
    return {"role": role, "content": content}


def test_the_last_turn_is_the_latest_question_and_its_answer():
    history = [turn("user", "q1"), turn("assistant", "a1"), turn("user", "q2"), turn("assistant", "a2")]
    assert last_turn(history) == history[2:]


def test_a_question_with_no_stored_answer_is_skipped_for_the_turn_before_it():
    history = [turn("user", "q1"), turn("assistant", "a1"), turn("user", "failed")]
    assert last_turn(history) == history[:2]


@pytest.mark.parametrize("history", [[], [turn("user", "q")], [turn("assistant", "a")]])
def test_no_complete_turn_means_nothing_is_sent(history):
    assert last_turn(history) == []


def test_deep_mode_sends_the_previous_turn_but_nothing_older(settings, monkeypatch):
    client = client_for(settings, FakeOllama(reply("Local secret.")))
    first = stream_events(client.post("/api/chat", json={"message": "Older question?"}))
    conversation = first[0][1]["conversation_id"]
    assert first[-1][1]["answer"] == "Local secret."
    client = client_for(settings, FakeOllama(reply("Latest answer.")))
    second = stream_events(client.post("/api/chat", json={"message": "Latest question?", "conversation_id": conversation}))
    assert second[-1][1]["answer"] == "Latest answer."
    seen = []
    monkeypatch.setattr(hosted, "answer_stream", deep_reply(seen))
    client.post("/api/chat", json={"message": "Follow-up?", "mode": "deep", "conversation_id": conversation})
    sent = json.dumps(seen[0])
    assert "Latest question?" in sent and "Latest answer." in sent and "Follow-up?" in sent
    assert "Older question" not in sent and "Local secret" not in sent


def test_the_first_deep_question_sends_no_past(settings, monkeypatch):
    seen = []
    monkeypatch.setattr(hosted, "answer_stream", deep_reply(seen))
    client_for(settings, FakeOllama()).post("/api/chat", json={"message": "Hello?", "mode": "deep"})
    assert [message["role"] for message in seen[0]] == ["system", "user"]


def test_an_unknown_mode_is_refused(settings):
    client = client_for(settings, FakeOllama())
    assert client.post("/api/chat", json={"message": "Hello?", "mode": "turbo"}).status_code == 422


def test_the_citations_come_from_the_latest_answer():
    rows = [{"role": "assistant", "citations": '[{"chunk_id": 1}]'},
            {"role": "user", "citations": None},
            {"role": "assistant", "citations": '[{"chunk_id": 2}]'},
            {"role": "user", "citations": None}]
    assert last_citations(rows) == [{"chunk_id": 2}]


@pytest.mark.parametrize("rows", [[], [{"role": "user", "citations": None}],
                                  [{"role": "assistant", "citations": None}]])
def test_no_cited_answer_means_nothing_to_carry(rows):
    assert last_citations(rows) == []



def test_a_bug_mid_answer_ends_in_an_error_event_and_is_stored(settings, monkeypatch):
    """Review C4: an exception inside the answer broke the stream; the reader saw nothing and
    the failure was never stored with the question."""
    async def broken(*args, **kwargs):
        yield "sources", {"sources": []}
        raise LookupError("unknown encoding: x-weird")

    monkeypatch.setattr("app.main.answer_events", broken)
    events = events_of(post(client_for(settings, FakeOllama(reply("Hi")))))
    assert names(events) == ["meta", "sources", "error"]
    assert "LookupError" in events[-1][1]["message"]
    conn = connect(settings.db_path)
    stored = conn.execute("SELECT details FROM messages WHERE role = 'user'").fetchone()[0]
    conn.close()
    assert stored and "LookupError" in stored
