"""The verification layer: the claims an answer states, checked with SymPy in the sandbox.

With Check on, the answering model ends its answer with a fenced block of claims (the
format is in CHECKS_PROMPT). The host only cuts that block out of the answer, as plain
text: it never parses a claim, because claims are model text and SymPy's reader calls
eval(). sandbox/verify_claims.py reads and checks them in a container and returns one
line of JSON, which is validated here before anything uses it.

If an answer has no block, `extraction_messages` asks the same model, in one short extra
call, to write the block for the answer as it stands.

Only "passed" counts as passed. "numerically_equal" (the sides agreed at the points tried,
without a proof) never does, and the badge says how many claims passed, never "correct".
A passed claim shows how it passed when it wasn't a proof: "passed (to stated digits)" for
a stated number that matched its formula to the digits given, "passed (natural units)" for
units that agree only with hbar = c = 1. Both still count as passed (user's decision, Nov 5).
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.tools.sandbox import SandboxUnavailable, give_nonce, report_line, run_container, work_folder

FENCE = "checks"
# The last ```checks block; an unclosed one at the very end (the model stopped) counts too.
BLOCK = re.compile(r"\n?[ \t]*```[ \t]*checks[ \t]*\n(?P<body>.*?)(?:\n[ \t]*```[ \t]*(?=\n|$)|\Z)", re.S)
SCRIPT = "/opt/sandbox/verify_claims.py"
Verdict = Literal["passed", "failed", "numerically_equal", "undecided", "error"]
Basis = Literal["natural_units", "stated_digits"]
BASIS_TEXT = {"natural_units": "natural units", "stated_digits": "to stated digits"}

CHECKS_PROMPT = """Checks (on for this question)
- The reader turned on checks. After your answer, add one fenced block that starts with ```checks and ends with ```. The app removes it from your answer, checks each claim with SymPy, and shows the reader how many passed.
- List the claims your answer states: its final results first, then the key steps of the derivation. At most 12. Write each exactly as your answer states it, even if you're unsure of it. Never change a claim to make it pass, and never add one your answer doesn't make.
- One per line, in SymPy syntax, never LaTeX: ** for powers, sqrt(), exp(), log(), diff(f, x), integrate(f, (x, a, b)), limit(f, x, oo), series(f, x, 0, n), Sum(f, (k, 1, n)). Use parentheses for calls, never square brackets.
- `assume positive: N, sigma` (also integer, nonnegative, real) declares what the symbols are. Declare it whenever a step relies on it.
- `name = expression` (or `let name = expression`) names something used later, such as the function you differentiate or a given value. Once a name has a value, `name = ...` is a claim about it.
- A claim is `left = right`. Use ≈ when the right side is a rounded number. A chain `a = b = c` checks each step. <, <=, > and >= work too. Anything after # is a label.
- Every claim must be computable: give each quantity in it a value first, with units, so that a stated number can be compared. A line that compares a formula in symbols with no values to a number can't be checked.
- Names are letters, digits and underscores: alpha_post, not alpha'. There is no E[...] or x[n]: write the formula an expectation or a sum leads to instead.
- Numbers with units: put the unit in square brackets after the value, as in `let M_W = 80.38 [GeV]` or `= 3.2e-25 [s]`. Units: eV, keV, MeV, GeV, TeV, s, ms, us, ns, ps, fs, m, cm, mm, um, nm, fm, km, kg, g, J, W, N, Hz, kHz, MHz, GHz, b, mb, nb, pb, fb, K, C, V, T, mol; also [hbar], [c] and [e] for the constants. For a value in natural units, give hbar or (hbar c)^2 its value with a let, as the source does.
- Bare names are symbols, including E, N, I, S and gamma; pi and oo are π and ∞. Write exp(1) for e.
- If nothing in your answer can be checked this way, write the block with the single line: none

Example:
```checks
assume positive: N, sigma
let sig = 1/(1 + exp(-f))
diff(sig, f) = sig*(1 - sig)   # derivative of the sigmoid
let G_F = 1.1664e-5 [GeV^-2]
let M_W = 80.38 [GeV]
G_F*M_W**3/(6*sqrt(2)*pi) ≈ 0.227 [GeV]   # leptonic W width
```"""

CHECKS_REMINDER = "End your answer with the ```checks block of its claims."

EXTRACT_PROMPT = """Below is a question and an answer to it. Write the checks block for the answer as it stands: list the claims it states, its final results first, then the key steps, so that they can be checked with SymPy. Write each claim exactly as the answer states it, even if you think it's wrong; never correct one and never add one the answer doesn't make. Reply with the block alone, nothing before or after it.

The format:

{format}"""


class Claim(BaseModel):
    line: str = Field(max_length=700)
    label: str = Field(default="", max_length=700)
    verdict: Verdict
    note: str = Field(default="", max_length=400)
    left: str | None = Field(default=None, max_length=400)
    right: str | None = Field(default=None, max_length=400)
    difference: str | None = Field(default=None, max_length=400)
    links: list[Verdict] | None = None
    basis: list[Basis] = Field(default_factory=list, max_length=2)     # how a passed claim passed

    def verdict_text(self) -> str:
        """The verdict as the reader sees it: "passed (natural units, to stated digits)"."""
        if self.verdict == "passed" and self.basis:
            return f"passed ({', '.join(BASIS_TEXT[kind] for kind in self.basis)})"
        return {"numerically_equal": "≈ (no proof)"}.get(self.verdict, self.verdict)


class Report(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=40)
    notes: list[str] = Field(default_factory=list, max_length=10)
    source: Literal["answer", "extracted", "none"] = "answer"   # who wrote the block
    block: str | None = None        # the block as written, for the reader and the graders
    seconds: float = 0.0            # the sandbox run
    failure: str | None = None      # the check itself didn't run (no Docker, bad output)
    extraction_seconds: float | None = None     # the extra call, for an answer without a block
    extraction_cost_usd: float = 0.0

    def count(self, verdict: str) -> int:
        return sum(claim.verdict == verdict for claim in self.claims)

    @property
    def passed(self) -> int:
        return self.count("passed")

    @property
    def all_passed(self) -> bool:
        """Every claim passed; ≈, undecided and unreadable claims don't."""
        return bool(self.claims) and self.passed == len(self.claims)

    def badge(self) -> str:
        if self.failure:
            return "The claims couldn't be checked"
        if not self.claims:
            return "No checkable claims"
        total = len(self.claims)
        return f"{self.passed} of {total} checked claim{'' if total == 1 else 's'} passed"

    def summary(self) -> dict:
        return {verdict: self.count(verdict) for verdict in Verdict.__args__} | {
            f"passed_{kind}": sum(c.verdict == "passed" and kind in c.basis for c in self.claims)
            for kind in Basis.__args__} | {"total": len(self.claims), "badge": self.badge()}


def split_checks(answer: str) -> tuple[str, str | None]:
    """The answer without its last ```checks block, and the block's raw text (None if there's none).

    Plain text only: the block's lines are never read here."""
    matches = list(BLOCK.finditer(answer))
    if not matches:
        return answer, None
    last = matches[-1]
    shown = (answer[:last.start()] + answer[last.end():]).rstrip()
    return shown, last.group("body")


def says_none(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return lines == ["none"]


def extraction_messages(question: str, answer: str) -> list[dict]:
    """One extra call for an answer that came without a block. Ask the model that wrote it."""
    format_part = CHECKS_PROMPT.split("\n", 1)[1]           # the rules and the example, not the heading
    return [
        {"role": "system", "content": EXTRACT_PROMPT.format(format=format_part)},
        {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
    ]


def block_from_reply(reply: str) -> str | None:
    """The block in an extraction reply: fenced as asked, or the bare lines if the fence was left off."""
    _, block = split_checks(reply)
    if block is not None:
        return block
    bare = reply.strip().strip("`").strip()
    return bare or None


def read_report(stdout: str, nonce: str) -> Report:
    """The checker's report: only a line with the nonce counts, so a claim can't print one."""
    found = report_line(stdout, nonce)
    if found is None:
        raise ValueError("the checker printed no result")
    return Report.model_validate(found)


async def check_block(settings: Settings, block: str, source: str = "answer") -> Report:
    """Run the claims in the sandbox. Never raises: a check that can't run says so in `failure`."""
    text = block[:settings.verify_block_chars]
    if says_none(block):
        return Report(source=source, block=text)
    limit = settings.verify_seconds
    try:
        with work_folder() as work:
            (work / "checks.txt").write_text(text, encoding="utf-8")
            nonce = give_nonce(work)
            done = await run_container(settings, work, ["timeout", "--signal=KILL", str(limit), "python", SCRIPT],
                                       limit)
    except SandboxUnavailable as error:
        return Report(source=source, block=text, failure=str(error)[:300])
    try:
        report = read_report(done.stdout, nonce)
    except (ValueError, ValidationError) as error:
        why = "timed out" if done.timed_out else f"{type(error).__name__}"
        return Report(source=source, block=text, seconds=done.seconds, failure=f"The check didn't finish ({why}).")
    return report.model_copy(update={"source": source, "block": text, "seconds": round(done.seconds, 2)})


# A line that may still turn into the ```checks fence: held back until it's known.
FENCE_START = re.compile(r"[ \t]*(?:`{0,3}|```[ \t]*(?:c(?:h(?:e(?:c(?:ks?)?)?)?)?)?)")
FENCE_LINE = re.compile(r"(?:^|\n)[ \t]*```[ \t]*checks")


class ChecksStream:
    """Streams an answer without its ```checks block, which is for the checker, not the reader.

    Plain text only, as split_checks: once a line opens the fence, the rest of the stream is
    held back. A line that could still become the fence waits until it can't. Whatever came
    after the block shows when the answer is done, from split_checks on the whole text."""

    def __init__(self):
        self.waiting = ""
        self.inside = False

    def feed(self, text: str) -> str:
        if self.inside:
            return ""
        self.waiting += text
        found = FENCE_LINE.search(self.waiting)
        if found:
            self.inside = True
            shown, self.waiting = self.waiting[:found.start()], ""
            return shown
        start = self.waiting.rfind("\n") + 1
        if FENCE_START.fullmatch(self.waiting[start:]) and self.waiting[start:]:
            shown, self.waiting = self.waiting[:start], self.waiting[start:]
            return shown
        shown, self.waiting = self.waiting, ""
        return shown

    def flush(self) -> str:
        shown, self.waiting = ("" if self.inside else self.waiting), ""
        return shown
