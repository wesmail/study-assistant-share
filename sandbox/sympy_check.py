"""Tests a claimed equality. Runs only inside the sandbox, because sympify calls eval()
(safe_sympify.py keeps that to maths).

    python /opt/sandbox/sympy_check.py     reads /work/claim.json, prints one line of JSON

Three verdicts, and only a proof says "true": the difference must simplify to zero. If it
doesn't, numbers decide "false": a difference that's clearly nonzero at some point rules
the claim out. Anything else is "undecided", including a difference that looks like zero
at every point tried, since that's evidence and not proof.
"""
from __future__ import annotations

import json
import random
import sys

import sympy as sp

import safe_sympify

SAMPLES = 6
TOLERANCE = 1e-9
SEED = 20261102


def parse(text: str, symbols: dict) -> sp.Expr:
    return safe_sympify.sympify(text, symbols)


def evaluated(expression):
    return expression.doit() if hasattr(expression, "doit") else expression


def numeric_difference(difference, free: list) -> float | None:
    """The largest difference seen at random positive points, or None if none could be computed."""
    rng = random.Random(SEED)
    worst = None
    for _ in range(SAMPLES):
        point = {symbol: sp.Rational(rng.randint(1, 40), 7) for symbol in free}
        try:
            value = complex(difference.subs(point).evalf())
        except (TypeError, ValueError, AttributeError):
            continue
        if value != value:          # not a number
            continue
        worst = abs(value) if worst is None else max(worst, abs(value))
    return worst


def check(claim: dict) -> dict:
    symbols = {name: sp.Symbol(name, positive=True) for name in claim.get("positive", [])}
    lhs = evaluated(parse(claim["lhs"], symbols))
    rhs = evaluated(parse(claim["rhs"], symbols))
    difference = sp.simplify(lhs - rhs)
    shown = {"lhs": str(lhs), "rhs": str(rhs), "difference": str(difference)}
    if difference == 0:
        return {"verdict": "true", **shown}
    free = sorted(difference.free_symbols, key=str)
    worst = numeric_difference(difference, free)
    if worst is not None and worst > TOLERANCE:
        return {"verdict": "false", **shown, "note": f"The two sides differ by up to {worst:.6g}."}
    note = "The difference didn't simplify to zero, but no point tried told the sides apart."
    if free:
        note += " Declare symbols positive if the claim assumes that."
    return {"verdict": "undecided", **shown, "note": note}


def main() -> None:
    nonce = safe_sympify.read_nonce()          # first: the claim may not see it
    claim = json.loads(open("/work/claim.json", encoding="utf-8").read())
    try:
        result = check(claim)
    except Exception as error:      # any parse failure is an answer for the model to act on
        result = {"verdict": "error", "note": f"{type(error).__name__}: {error}"[:400]}
    safe_sympify.report(result, nonce)


if __name__ == "__main__":
    sys.exit(main())
