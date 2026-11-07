"""chat.checked_attempts: Check on, best-of-N with early stop (verification layer, step 3).

The model and the sandbox are fakes here: each attempt is a scripted event stream, and
each answer's report is given by the test.
"""
import asyncio

from app import chat
from app.config import Settings
from app.verify import Report

SOURCES = [{"id": 1, "doc_title": "Book"}]


def report(*verdicts, cost=0.0):
    return Report(claims=[{"line": f"claim {i} {v}", "verdict": v} for i, v in enumerate(verdicts)],
                  source="answer", block="...", extraction_cost_usd=cost)


def attempt_events(text, cost=0.0, error=None):
    async def events():
        yield "thinking", {"text": "hm"}
        if error:
            yield "error", {"message": error}
            return
        for piece in (text[:7], text[7:]):
            yield "token", {"text": piece}
        yield "done", {"answer": text, "citations": [1], "cost_usd": cost, "done_reason": "stop"}
    return events()


def run(monkeypatch, answers, reports, samples=3, ceiling=None, errors=None, checks=True):
    errors = errors or {}
    started, verified = [], []

    def generate(number):
        started.append(number)
        return attempt_events(answers[number - 1], cost=0.01, error=errors.get(number))

    async def fake_verify(question, answer, settings, provider, transport=None, purpose="chat", client=None):
        verified.append(answer)
        shown = answer.split("\n```checks")[0]
        return shown, reports[len(verified) - 1]

    monkeypatch.setattr(chat, "verify_answer", fake_verify)
    if ceiling is not None:         # a clock that moves 30 s at every look, so the ceiling is reached on time
        clock = iter(range(0, 10**6, 30))
        monkeypatch.setattr(chat.time, "monotonic", lambda: float(next(clock)))
    settings = Settings(check_samples=samples, check_ceiling_seconds=ceiling)

    async def collect():
        return [item async for item in chat.answered(generate, checks, "q", settings, "ollama", None, "chat",
                                                     None, SOURCES)]

    events = asyncio.run(collect())
    return events, started, verified


def answer(n):
    return f"Answer number {n} [S1].\n```checks\nclaim {n}\n```"


def streamed(events):
    return "".join(data["text"] for event, data in events if event == "token")


def done_of(events):
    return next(data for event, data in events if event == "done")


class TestBestOfN:
    def test_checks_off_is_one_plain_answer(self, monkeypatch):
        events, started, verified = run(monkeypatch, [answer(1)], [], checks=False)
        assert started == [1] and verified == []
        assert done_of(events)["answer"] == answer(1)

    def test_a_first_attempt_that_passes_stops_there(self, monkeypatch):
        events, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("passed", "passed")])
        assert started == [1]
        done = done_of(events)
        assert done["answer"] == "Answer number 1 [S1]."
        assert done["checks"]["badge"] == "2 of 2 checked claims passed"
        assert done["checks"]["attempt"] == 1 and done["checks"]["attempts"] == 1 and done["replaced"] == []
        assert not done["checks"]["none_passed"]

    def test_the_block_never_streams(self, monkeypatch):
        events, _, _ = run(monkeypatch, [answer(1)], [report("passed")])
        assert "checks" not in streamed(events) and "claim 1" not in streamed(events)
        assert ("check_status", {"stage": "checking", "attempt": 1, "of": 3}) in events

    def test_a_failed_claim_tries_again_and_says_which(self, monkeypatch):
        events, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("passed", "failed"), report("passed")])
        assert started == [1, 2]
        replaced = next(data for event, data in events if event == "attempt_replaced")
        assert replaced["attempt"] == 1 and replaced["next"] == 2 and replaced["of"] == 3
        assert replaced["failing"] == ["claim 1 failed"] and replaced["sources"] == SOURCES
        assert replaced["answer"] == "Answer number 1 [S1]."
        done = done_of(events)
        assert done["answer"] == "Answer number 2 [S1]." and done["checks"]["attempt"] == 2
        assert [item["attempt"] for item in done["replaced"]] == [1]

    def test_undecided_and_approximate_claims_dont_trigger_a_retry(self, monkeypatch):
        _, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("undecided", "numerically_equal")])
        assert started == [1]

    def test_when_every_attempt_fails_the_fewest_failures_is_shown(self, monkeypatch):
        reports = [report("failed", "failed"), report("passed", "failed"), report("failed", "failed", "failed")]
        events, started, _ = run(monkeypatch, [answer(1), answer(2), answer(3)], reports)
        assert started == [1, 2, 3]
        done = done_of(events)
        assert done["answer"] == "Answer number 2 [S1]." and done["checks"]["attempt"] == 2
        assert done["checks"]["none_passed"] and done["checks"]["badge"] == "1 of 2 checked claims passed"
        assert [item["attempt"] for item in done["replaced"]] == [1, 3]

    def test_samples_one_is_badges_only(self, monkeypatch):
        events, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("failed")], samples=1)
        assert started == [1]
        assert not any(event == "attempt_replaced" for event, _ in events)
        assert not done_of(events)["checks"]["none_passed"]       # nothing was tried again

    def test_no_attempt_starts_past_the_ceiling(self, monkeypatch):
        events, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("failed")], ceiling=30)
        assert started == [1]
        assert not any(event == "attempt_replaced" for event, _ in events)
        assert done_of(events)["checks"]["stopped"] == "time"

    def test_a_running_attempt_past_the_ceiling_is_given_up(self, monkeypatch):
        # On the fake clock attempt 1 takes 30 s and ends at 90 s, so one more fits in 130 s;
        # attempt 2 passes 130 s at its first event.
        events, started, verified = run(monkeypatch, [answer(1), answer(2)], [report("failed")], ceiling=130)
        assert started == [1, 2] and len(verified) == 1
        done = done_of(events)
        assert done["answer"] == "Answer number 1 [S1]." and done["checks"]["stopped"] == "time"

    def test_a_later_attempt_that_fails_leaves_the_earlier_one(self, monkeypatch):
        events, started, _ = run(monkeypatch, [answer(1), answer(2)], [report("failed")], errors={2: "model gone"})
        assert started == [1, 2]
        done = done_of(events)
        assert done["answer"] == "Answer number 1 [S1]." and done["checks"]["stopped"] == "error"
        assert not any(event == "error" for event, _ in events)

    def test_a_first_attempt_that_fails_is_the_error(self, monkeypatch):
        events, _, verified = run(monkeypatch, [answer(1)], [], errors={1: "model gone"})
        assert ("error", {"message": "model gone"}) in events and verified == []
        assert not any(event == "done" for event, _ in events)

    def test_the_cost_is_every_attempt_and_extraction(self, monkeypatch):
        events, _, _ = run(monkeypatch, [answer(1), answer(2)], [report("failed", cost=0.005), report("passed")])
        assert abs(done_of(events)["cost_usd"] - 0.025) < 1e-9

    def test_the_block_as_written_goes_with_the_report(self, monkeypatch):
        events, _, _ = run(monkeypatch, [answer(1)], [report("passed")])
        assert done_of(events)["checks"]["block"] == "..."
        assert done_of(events)["checks"]["claims"][0]["shown"] == "passed"


class TestChecksStream:
    def stream(self, *pieces):
        from app.verify import ChecksStream
        filtered = ChecksStream()
        return "".join(filtered.feed(piece) for piece in pieces) + filtered.flush()

    def test_a_fence_split_across_pieces_is_held_back(self):
        assert self.stream("Answer.\n\n``", "`chec", "ks\nlet a = 1\n", "a = 1\n```") == "Answer.\n\n"

    def test_other_code_blocks_pass(self):
        assert self.stream("Code:\n```py", "thon\nprint(1)\n```\nend") == "Code:\n```python\nprint(1)\n```\nend"

    def test_inline_backticks_pass_at_once(self):
        from app.verify import ChecksStream
        filtered = ChecksStream()
        assert filtered.feed("x = `y` and ``") == "x = `y` and ``"

    def test_a_block_at_the_start(self):
        assert self.stream("```checks\n1 = 1\n```") == ""


class TestThroughTheApi:
    """POST /api/chat with check: true, Ollama faked and the sandbox replaced by a report."""

    def test_check_on_streams_without_the_block_and_stores_the_report(self, tmp_path, monkeypatch):
        import json

        from app.db.connection import connect
        from app.verify import CHECKS_PROMPT
        from tests.test_chat import FakeOllama, client_for, events_of, reply

        settings = Settings(data_dir=tmp_path, library_dir=tmp_path, tools_enabled=False)
        connect(settings.db_path).close()
        seen = []

        async def fake_check(settings, block, source="answer"):
            seen.append(block)
            return Report(claims=[{"line": "1 = 1", "verdict": "passed", "basis": ["stated_digits"]}],
                          source=source, block=block)

        monkeypatch.setattr(chat, "check_block", fake_check)
        ollama = FakeOllama(reply("It is one.", "\n\n```", "checks\n1 = 1\n```"))
        events = events_of(client_for(settings, ollama).post("/api/chat", json={"message": "Why?", "check": True}))
        assert CHECKS_PROMPT in ollama.requests[0]["messages"][0]["content"]
        assert seen == ["1 = 1"]
        assert "checks" not in "".join(data["text"] for event, data in events if event == "token")
        done = next(data for event, data in events if event == "done")
        assert done["answer"] == "It is one."
        assert done["checks"]["claims"][0]["shown"] == "passed (to stated digits)"
        conn = connect(settings.db_path)
        content, details = conn.execute("SELECT content, details FROM messages WHERE role = 'assistant'").fetchone()
        assert content == "It is one."
        assert json.loads(details)["checks"]["badge"] == "1 of 1 checked claim passed"

    def test_check_off_asks_for_no_block(self, tmp_path):
        from app.db.connection import connect
        from app.verify import CHECKS_PROMPT
        from tests.test_chat import FakeOllama, client_for, events_of, reply

        settings = Settings(data_dir=tmp_path, library_dir=tmp_path, tools_enabled=False)
        connect(settings.db_path).close()
        ollama = FakeOllama(reply("Hi"))
        events = events_of(client_for(settings, ollama).post("/api/chat", json={"message": "Why?"}))
        assert CHECKS_PROMPT not in ollama.requests[0]["messages"][0]["content"]
        assert "checks" not in next(data for event, data in events if event == "done")

    def test_the_options_say_how_many_attempts_check_may_take(self, tmp_path):
        from app.config import load_settings
        from app.db.connection import connect
        from tests.test_chat import FakeOllama, client_for

        configured = load_settings()
        settings = Settings(data_dir=tmp_path, library_dir=tmp_path, answer_options=configured.answer_options)
        connect(settings.db_path).close()
        options = {option["id"]: option for option in
                   client_for(settings, FakeOllama()).get("/api/options").json()["options"]}
        # Badges only since round 19: best-of-N gave no gain, so every option makes one attempt.
        assert all(option["check_samples"] == 1 for option in options.values())
        assert options["fast"]["check_badge"] == options["balanced"]["check_badge"] == "limited"
        assert options["deep"]["check_badge"] == "full"
        assert options["qwen"]["check_badge"] == "unmeasured"  # neutral until it's measured
