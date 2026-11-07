"""app/verify.py: cutting the claims block out of an answer and reading the checker's report.

The claims themselves are never read on the host, so nothing here parses one.
"""
import asyncio
import json

import pytest

from app import verify
from app.config import Settings
from app.llm.prompts import build_messages
from app.tools.sandbox import Completed, SandboxUnavailable
from app.llm.ollama import Delta
from app.verify import CHECKS_PROMPT, Report, check_block, extraction_messages, read_report, split_checks

ANSWER = "The derivative is $\\sigma(1-\\sigma)$ [S1].\n\n"
BLOCK = "let sig = 1/(1 + exp(-f))\ndiff(sig, f) = sig*(1 - sig)"


class TestSplitting:
    def test_the_block_is_cut_out_and_returned_as_text(self):
        shown, block = split_checks(f"{ANSWER}```checks\n{BLOCK}\n```\n")
        assert shown == ANSWER.rstrip()
        assert block == BLOCK

    def test_no_block(self):
        assert split_checks(ANSWER) == (ANSWER, None)

    def test_other_code_blocks_stay(self):
        text = f"{ANSWER}```python\nprint(1)\n```\n\n```checks\n{BLOCK}\n```"
        shown, block = split_checks(text)
        assert "```python\nprint(1)\n```" in shown
        assert block == BLOCK

    def test_the_last_block_counts_and_text_after_it_stays(self):
        text = f"```checks\n1 = 1\n```\n{ANSWER}```checks\n{BLOCK}\n```\nThat's all."
        shown, block = split_checks(text)
        assert block == BLOCK
        assert shown.endswith("That's all.")
        assert shown.count("```checks") == 1      # the earlier block stays as written

    def test_an_unclosed_block_at_the_end_counts(self):
        shown, block = split_checks(f"{ANSWER}```checks\n{BLOCK}")
        assert block == BLOCK
        assert "```" not in shown

    def test_a_space_after_the_fence_is_fine(self):
        assert split_checks(f"{ANSWER}``` checks\n{BLOCK}\n```")[1] == BLOCK


class TestReport:
    def report(self, *verdicts):
        return Report(claims=[{"line": f"c{i}", "verdict": v} for i, v in enumerate(verdicts)])

    def test_the_badge_counts_passed_claims_and_never_says_correct(self):
        report = self.report("passed", "passed", "failed")
        assert report.badge() == "2 of 3 checked claims passed"
        assert "correct" not in report.badge().lower()

    def test_numerically_equal_never_counts_as_passed(self):
        report = self.report("passed", "numerically_equal")
        assert report.passed == 1
        assert not report.all_passed

    def test_all_passed(self):
        assert self.report("passed", "passed").all_passed
        assert not self.report().all_passed

    def test_no_claims_and_a_failed_check_say_so(self):
        assert Report().badge() == "No checkable claims"
        assert Report(failure="no Docker").badge() == "The claims couldn't be checked"

    def test_the_summary_counts_every_verdict(self):
        summary = self.report("passed", "failed", "undecided").summary()
        assert summary["passed"] == 1 and summary["failed"] == 1 and summary["undecided"] == 1
        assert summary["numerically_equal"] == 0 and summary["total"] == 3

    def test_the_checker_output_is_validated(self):
        good = json.dumps({"claims": [{"line": "1 = 1", "verdict": "passed"}], "notes": [], "nonce": "n0"})
        assert read_report(f"noise\n{good}\n", "n0").passed == 1
        with pytest.raises(ValueError):
            read_report(json.dumps({"claims": [{"line": "1 = 1", "verdict": "probably"}], "nonce": "n0"}), "n0")
        with pytest.raises(ValueError):
            read_report("", "n0")

    def test_a_report_without_the_nonce_is_ignored(self):
        """A claim that printed its own report (review S3) can't know the nonce."""
        real = json.dumps({"claims": [{"line": "2 + 2 = 5", "verdict": "failed"}], "nonce": "n0"})
        forged = json.dumps({"claims": [{"line": "2 + 2 = 5", "verdict": "passed"}]})
        guessed = json.dumps({"claims": [{"line": "2 + 2 = 5", "verdict": "passed"}], "nonce": "n1"})
        assert read_report(f"{real}\n{forged}\n{guessed}", "n0").count("failed") == 1
        with pytest.raises(ValueError):
            read_report(f"{forged}\n{guessed}", "n0")


class TestCheckBlock:
    def run(self, monkeypatch, stdout="", timed_out=False, raises=None):
        seen = {}

        async def fake(settings, work, argv, seconds):
            seen["text"] = (work / "checks.txt").read_text(encoding="utf-8")
            seen["argv"] = argv
            if raises:
                raise raises
            nonce = (work / "nonce").read_text(encoding="ascii")
            return Completed(0, json.dumps({**stdout, "nonce": nonce}) if isinstance(stdout, dict) else stdout,
                             "", 1.5, timed_out)

        monkeypatch.setattr(verify, "run_container", fake)
        return seen

    def test_the_block_goes_into_the_sandbox_as_a_file(self, monkeypatch):
        out = {"claims": [{"line": "diff(sig, f) = sig*(1 - sig)", "verdict": "passed"}]}
        seen = self.run(monkeypatch, out)
        report = asyncio.run(check_block(Settings(), BLOCK, source="extracted"))
        assert seen["text"] == BLOCK
        assert seen["argv"][-1] == verify.SCRIPT
        assert report.passed == 1 and report.source == "extracted" and report.seconds == 1.5

    def test_a_long_block_is_cut(self, monkeypatch):
        seen = self.run(monkeypatch, {"claims": []})
        asyncio.run(check_block(Settings(verify_block_chars=10), "1 = 1\n" * 100))
        assert len(seen["text"]) == 10

    def test_none_means_no_claims_and_no_container(self, monkeypatch):
        seen = self.run(monkeypatch)
        report = asyncio.run(check_block(Settings(), "\nnone\n"))
        assert report.claims == [] and not report.failure and "text" not in seen

    def test_no_docker_is_a_failure_not_an_exception(self, monkeypatch):
        self.run(monkeypatch, raises=SandboxUnavailable("Docker isn't installed"))
        report = asyncio.run(check_block(Settings(), BLOCK))
        assert report.failure and report.badge() == "The claims couldn't be checked"

    def test_a_timeout_is_a_failure(self, monkeypatch):
        self.run(monkeypatch, "", timed_out=True)
        assert "timed out" in asyncio.run(check_block(Settings(), BLOCK)).failure


class TestPrompts:
    def test_checks_on_adds_the_instructions(self):
        system = build_messages("q", [], [], checks=True)[0]["content"]
        assert CHECKS_PROMPT in system
        assert CHECKS_PROMPT not in build_messages("q", [], [])[0]["content"]

    def test_checks_work_with_tools_too(self):
        assert CHECKS_PROMPT in build_messages("q", [], [], tools=True, checks=True)[0]["content"]

    def test_extraction_asks_for_the_block_alone_and_carries_the_answer(self):
        messages = extraction_messages("Why?", "Because 1 + 1 = 2.")
        assert "block alone" in messages[0]["content"]
        assert "```checks" in messages[0]["content"]
        assert "Because 1 + 1 = 2." in messages[1]["content"]


class TestVerifyAnswer:
    """chat.verify_answer: a block is used as written; a missing one costs one extra call."""

    def prepare(self, monkeypatch, reply=None, error=None):
        from app import chat
        calls, checked = [], []

        async def fake_reply(messages, settings, provider, transport, purpose, fallback, client=None):
            calls.append({"messages": messages, "purpose": purpose, "provider": provider,
                          "think": settings.chat_think})
            if error:
                raise error
            yield Delta(text=reply)
            yield Delta(done=True, cost_usd=0.02)

        async def fake_check(settings, block, source="answer"):
            checked.append((block, source))
            return Report(claims=[{"line": block.splitlines()[-1], "verdict": "passed"}], source=source, block=block)

        monkeypatch.setattr(chat, "model_reply", fake_reply)
        monkeypatch.setattr(chat, "check_block", fake_check)
        return chat, calls, checked

    def test_an_answer_with_a_block_needs_no_extra_call(self, monkeypatch):
        chat, calls, checked = self.prepare(monkeypatch)
        shown, report = asyncio.run(chat.verify_answer("q", f"{ANSWER}```checks\n{BLOCK}\n```", Settings(), "ollama"))
        assert shown == ANSWER.rstrip() and calls == []
        assert checked == [(BLOCK, "answer")] and report.source == "answer"

    def test_an_answer_without_one_asks_the_same_model_once(self, monkeypatch):
        chat, calls, checked = self.prepare(monkeypatch, reply=f"```checks\n{BLOCK}\n```")
        shown, report = asyncio.run(chat.verify_answer("Why?", ANSWER, Settings(), "anthropic", purpose="eval:x"))
        assert len(calls) == 1 and calls[0]["provider"] == "anthropic" and calls[0]["purpose"] == "eval:x:extract"
        assert ANSWER.strip() in calls[0]["messages"][1]["content"]
        assert checked == [(BLOCK, "extracted")]
        assert report.source == "extracted" and report.extraction_cost_usd == 0.02
        assert report.extraction_seconds is not None

    def test_extraction_thinks_little(self, monkeypatch):
        chat, calls, _ = self.prepare(monkeypatch, reply=BLOCK)
        asyncio.run(chat.verify_answer("q", ANSWER, Settings(chat_think=True), "ollama"))
        asyncio.run(chat.verify_answer("q", ANSWER, Settings(chat_think="medium"), "ollama"))
        assert [call["think"] for call in calls] == [False, "low"]

    def test_a_reply_without_the_fence_is_taken_as_the_block(self, monkeypatch):
        chat, _, checked = self.prepare(monkeypatch, reply=BLOCK)
        asyncio.run(chat.verify_answer("q", ANSWER, Settings(), "ollama"))
        assert checked == [(BLOCK, "extracted")]

    def test_a_failed_extraction_is_reported_not_raised(self, monkeypatch):
        chat, _, checked = self.prepare(monkeypatch, error=RuntimeError("model gone"))
        shown, report = asyncio.run(chat.verify_answer("q", ANSWER, Settings(), "ollama"))
        assert shown == ANSWER and checked == []
        assert "model gone" in report.failure and report.source == "none"

    def test_an_empty_extraction_means_no_claims(self, monkeypatch):
        chat, _, checked = self.prepare(monkeypatch, reply="   ")
        _, report = asyncio.run(chat.verify_answer("q", ANSWER, Settings(), "ollama"))
        assert checked == [] and report.source == "none" and report.badge() == "No checkable claims"


class TestVerdictText:
    def claim(self, verdict, basis=()):
        return Report(claims=[{"line": "x", "verdict": verdict, "basis": list(basis)}]).claims[0]

    def test_how_a_claim_passed_is_shown(self):
        assert self.claim("passed").verdict_text() == "passed"
        assert self.claim("passed", ["stated_digits"]).verdict_text() == "passed (to stated digits)"
        assert self.claim("passed", ["natural_units"]).verdict_text() == "passed (natural units)"
        assert (self.claim("passed", ["natural_units", "stated_digits"]).verdict_text()
                == "passed (natural units, to stated digits)")

    def test_labelled_claims_still_count_as_passed(self):
        report = Report(claims=[{"line": "a", "verdict": "passed", "basis": ["natural_units"]},
                                {"line": "b", "verdict": "passed", "basis": ["stated_digits"]}])
        assert report.all_passed and report.badge() == "2 of 2 checked claims passed"
        summary = report.summary()
        assert summary["passed_natural_units"] == 1 and summary["passed_stated_digits"] == 1

    def test_numerically_equal_reads_as_no_proof(self):
        assert self.claim("numerically_equal").verdict_text() == "≈ (no proof)"

    def test_an_unknown_basis_is_rejected(self):
        with pytest.raises(ValueError):
            Report(claims=[{"line": "a", "verdict": "passed", "basis": ["close enough"]}])
