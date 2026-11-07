"""Checks the claims block at the end of an answer. Runs only inside the sandbox.

    python /opt/sandbox/verify_claims.py     reads /work/checks.txt, prints one line of JSON

The host only cuts the block out of the answer; every bit of parsing happens here,
because the text is the model's and sympify calls eval(); safe_sympify.py says how it is
kept from running code or forging this report. The block is lines of:

    assume positive: N, sigma        also integer, nonnegative, real
    let sig = 1/(1 + exp(-f))        a name for later lines; may carry a unit
    let G_F = 1.1664e-5 [GeV^-2]     units go in square brackets after a value
    L = -log(sig)                    also a definition: the first `name = ...` for a new name
    f(x) = x**2                      a function (or let f = lambda x: x**2)
    diff(sig, f) = sig*(1 - sig)     a claim; ==, ≈, <, <=, >, >= work too
    a = b = c                        a chain: each neighbouring pair is checked
    ... # a label                    anything after # is a label, shown with the result

Python habits are read too: sp./np./math. prefixes, import lines (skipped), ceil, alpha'.

Verdicts, per claim, and only the first ever counts as passed:

    passed              proved: the difference simplifies to zero; or, for a stated
                        number such as 0.227 [GeV], the value matches to the digits given
                        (one unit in the last digit, at least 1 ppm), in the same units,
                        directly or with hbar = c = 1
    failed              the sides differ: a number off by more than its digits, a nonzero
                        difference at points that satisfy the assumptions, or units that
                        don't match even with hbar = c = 1 while the numbers differ too
    numerically_equal   no proof, but the sides agreed at every point tried (≈)
    undecided           can't be judged: a formula with no values against a number, a name
                        whose definition couldn't be read, a function with no definition,
                        units that differ while the numbers as written agree (a constant
                        lost its unit), or SymPy ran out of time
    error               the line couldn't be read

A passed claim also says how it passed, in "basis", so the reader can see the kind of check:
"stated_digits" (a stated number matched its formula to the digits given, not exactly) and
"natural_units" (the units agree only with hbar = c = 1). A proof has neither.

A claim the checker can't judge is undecided, never failed: a false "failed" on a true
claim would tell the reader a right answer is wrong.
"""
from __future__ import annotations

import random
import re
import signal
import sys
import time

import sympy as sp
from sympy.physics import units as u
from sympy.physics.units.systems.si import SI, dimsys_SI

import safe_sympify

PATH = "/work/checks.txt"
MAX_LINES = 40
MAX_LINE_CHARS = 600
CLAIM_SECONDS = 10
TOTAL_SECONDS = 90          # the whole block; the host's own limit (verify_seconds) is longer
SAMPLES = 6
TOLERANCE = 1e-9            # relative, for two sides that are both computed
PRECISION_FLOOR = 1e-6      # relative: digits past about the sixth aren't held against a claim
SEED = 20261105
ASSUMPTIONS = {"positive", "integer", "nonnegative", "real"}
VERDICT_ORDER = ("failed", "error", "undecided", "numerically_equal", "passed")   # worst first
BASES = ("natural_units", "stated_digits")     # how a passed claim passed, when not by proof
RELATIONS = ("==", "~=", "<=", ">=", "≈", "≤", "≥", "=", "<", ">")
CONSTANTS = {"pi", "oo"}    # the only bare names that keep SymPy's meaning; E, I, N, S, gamma ... are symbols

GeV = sp.Rational(10**9) * u.electronvolt
BARN = sp.Rational(1, 10**28) * u.meter**2
UNITS = {
    "eV": u.electronvolt, "keV": GeV / 10**6, "MeV": GeV / 1000, "GeV": GeV, "TeV": GeV * 1000,
    "s": u.second, "ms": u.second / 1000, "us": u.second / 10**6, "ns": u.second / 10**9,
    "ps": u.second / 10**12, "fs": u.second / 10**15, "min": u.minute, "h": u.hour,
    "m": u.meter, "km": u.kilometer, "cm": u.centimeter, "mm": u.millimeter, "um": u.micrometer,
    "nm": u.nanometer, "fm": u.meter / 10**15,
    "kg": u.kilogram, "g": u.gram, "J": u.joule, "W": u.watt, "N": u.newton, "Pa": u.pascal,
    "Hz": u.hertz, "kHz": u.hertz * 1000, "MHz": u.hertz * 10**6, "GHz": u.hertz * 10**9,
    "b": BARN, "mb": BARN / 1000, "ub": BARN / 10**6, "nb": BARN / 10**9, "pb": BARN / 10**12,
    "fb": BARN / 10**15,
    "K": u.kelvin, "A": u.ampere, "C": u.coulomb, "V": u.volt, "T": u.tesla, "mol": u.mole,
    "hbar": u.hbar, "c": u.speed_of_light, "e": u.elementary_charge,
}
BASE = [u.meter, u.kilogram, u.second, u.ampere, u.kelvin, u.mole, u.candela]
TYPE_WORDS = {"int", "integer", "float", "real", "number", "dimensionless", "1", "",
              # counts, written as units: [samples] is a number of samples (step 4, gap G4)
              "sample", "samples", "token", "tokens", "bin", "bins", "frame", "frames", "parameter",
              "parameters", "param", "params", "position", "positions", "step", "steps", "count", "counts"}
# Natural units (hbar = c = 1), as particle physics writes them: everything in powers of eV.
NATURAL_EV = sp.Symbol("eV_natural", positive=True)
HBAR_EV_S = sp.Float("6.582119569e-16")      # eV s
HBARC_EV_M = sp.Float("1.973269804e-7")      # eV m
EV_IN_KG = sp.Float("1.78266192e-36")         # kg, for 1 eV/c^2
UNIT_BRACKET = re.compile(r"(?<![\w\])])\s*\[([^\[\]]*)\]")    # "[GeV]" after a space, digit or ")"
STRAY_BRACKET = re.compile(r"[\[\]]")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LAMBDA = re.compile(r"lambda\s+([A-Za-z_][\w\s,]*):\s*(.+)")
FUNCTION = re.compile(r"(?:let\s+)?([A-Za-z_]\w*)\(\s*([A-Za-z_][\w\s,]*)\)\s*=\s*(.+)")
LET = re.compile(r"let\s+([A-Za-z_]\w*)\s*=\s*(.+)")
SYMPY_NAMES = set(dir(sp))      # diff(sig, f) = ... is a claim, never a definition of diff
SKIPPED = re.compile(r"(?:import\s|from\s+\w+\s+import\s)")      # Python habits: nothing to check
PRIME = re.compile(r"(?<!['\w])([A-Za-z_]\w*)('+)")
NUMBER = re.compile(r"^\s*[-+]?(\d+)(?:\.(\d*))?(?:[eE]([-+]?\d+))?\s*$")


class Unreadable(ValueError):
    """The line can't be turned into a claim."""


class UsesBroken(Unreadable):
    """The claim uses a name whose definition couldn't be read: undecided, never failed."""


class OutOfTime(Exception):
    pass


def alarm(_signum, _frame):
    raise OutOfTime


# --- reading ---------------------------------------------------------------------------

def normalised(text: str) -> str:
    """What models write in Python made readable: sp.diff -> diff, np.sqrt -> sqrt, ceil -> ceiling,
    alpha' -> alpha_prime."""
    text = re.sub(r"\b(?:sp|sympy|np|numpy|math)\.", "", text)
    # Python's sum and prod over a range tuple, as SymPy's Sum and Product (step 4, gap G3)
    text = re.sub(r"\bsum\((?!\s*\[)", "Sum(", text)
    text = re.sub(r"\bprod\((?!\s*\[)", "Product(", text)
    return primes(re.sub(r"\bceil\(", "ceiling(", text))


def primes(text: str) -> str:
    """alpha' -> alpha_prime, as models write a posterior's parameters. A quote that closes a
    string, as in symbols('x'), follows a quote-opened name and is left alone."""
    return PRIME.sub(lambda m: m.group(1) + "_prime" * len(m.group(2)), text)


def split_label(line: str) -> tuple[str, str]:
    body, _, label = line.partition("#")
    return body.strip(), label.strip()


def split_relations(text: str) -> tuple[list[str], list[str]]:
    """Split at relation signs outside brackets: 'a = b <= c' -> ['a', 'b', 'c'], ['=', '<=']."""
    parts, signs, depth, start, i = [], [], 0, 0, 0
    while i < len(text):
        char = text[i]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0:
            sign = next((s for s in RELATIONS if text.startswith(s, i)), None)
            if sign:
                parts.append(text[start:i])
                signs.append(sign)
                i += len(sign)
                start = i
                continue
        i += 1
    parts.append(text[start:])
    if any(not part.strip() for part in parts):
        raise Unreadable("a side of the relation is empty")
    return [part.strip() for part in parts], signs


def unit_value(text: str):
    if text.strip().lower() in TYPE_WORDS:          # [int], [float]: a type, not a unit
        return sp.Integer(1)
    names = {name: UNITS[name] for name in NAME.findall(text) if name in UNITS}
    unknown = [name for name in NAME.findall(text) if name not in UNITS]
    if unknown:
        raise Unreadable(f"unknown unit {unknown[0]!r}; known: {', '.join(sorted(UNITS))}")
    text = re.sub(r"(?<=[\w)])\s+(?=[A-Za-z(])", "*", text.strip())        # [mb GeV^2] is mb*GeV^2
    return safe_sympify.sympify(text.replace("^", "**"), names) if text else sp.Integer(1)


def stated_step(text: str, integers: bool = False) -> sp.Rational | None:
    """How far a stated number may be from the computed one. None if `text` isn't a bare number.

    A decimal gets one unit in its last digit ('0.227' -> 0.001), which allows for rounding
    along the way. A whole number is exact ('96'), unless `integers`: a measured quantity
    with a unit, or a claim marked ≈, where it gets half a unit ('227 [MeV]' -> 0.5 MeV),
    and its trailing zeros aren't taken as significant ('≈ 5390 [s]' -> 5 s)."""
    body = UNIT_BRACKET.sub("", text)
    match = NUMBER.match(body)
    if not match:
        return None
    if match.group(2) is None and match.group(3) is None:
        if not integers:
            return None
        digits = match.group(1).lstrip("0") or "0"
        zeros = len(digits) - len(digits.rstrip("0")) if digits != "0" else 0
        return sp.Rational(10) ** zeros / 2
    decimals = len(match.group(2) or "")
    exponent = int(match.group(3) or 0)
    return sp.Rational(10) ** (exponent - decimals)


LOG_ARGUMENT = sp.Dummy("x")


class Reader:
    """Names and values so far. Every value is kept twice: with its units, and `plain`, with
    each [unit] read as 1, the numbers as written, for telling a missing unit from a wrong number."""

    def __init__(self):
        self.assumed: dict[str, dict] = {}
        self.lets: dict[str, sp.Expr] = {}
        self.plain: dict[str, sp.Expr] = {}
        self.broken: set[str] = set()

    def define(self, name: str, text: str, arguments: str | None = None) -> None:
        """name = text, or name(arguments) = text for a function."""
        try:
            if arguments is not None:
                text = f"Lambda(({arguments},), {text})"
            elif match := LAMBDA.fullmatch(text.strip()):
                text = f"Lambda(({match.group(1)},), {match.group(2)})"
            value, plain = self.expression(text), self.expression(text, plain=True)
        except Unreadable:
            self.broken.add(name)
            self.lets.pop(name, None)
            raise
        self.lets[name], self.plain[name] = value, plain
        self.broken.discard(name)

    def symbol(self, name: str) -> sp.Symbol:
        return sp.Symbol(name, **self.assumed.get(name, {}))

    def assume(self, kind: str, names: str) -> None:
        if kind not in ASSUMPTIONS:
            raise Unreadable(f"unknown assumption {kind!r}; use {', '.join(sorted(ASSUMPTIONS))}")
        for name in (n.strip() for n in names.split(",")):
            if not NAME.fullmatch(name):
                raise Unreadable(f"{name!r} isn't a name")
            self.assumed.setdefault(name, {})[kind] = True

    def defines(self, side: str, sign: str) -> bool:
        """`name = ...` with a name that has no value yet defines it, as models write definitions.
        Once it has one, `name = ...` is a claim about it."""
        return sign in ("=", "==") and bool(NAME.fullmatch(side)) and side not in self.lets \
            and side not in CONSTANTS

    def expression(self, text: str, plain: bool = False) -> sp.Expr:
        units: dict[str, sp.Expr] = {}
        lets = self.plain if plain else self.lets

        def unit(match: re.Match) -> str:
            key = f"_unit{len(units)}"
            value = unit_value(match.group(1))
            units[key] = sp.Integer(1) if plain else value
            before = text[:match.start()].rstrip()
            follows_value = bool(before) and (before[-1].isalnum() or before[-1] in "._)")
            return f"*{key}" if follows_value else key

        text = UNIT_BRACKET.sub(unit, text)
        if STRAY_BRACKET.search(text):
            raise Unreadable("square brackets are only for units, after a value: use parentheses for calls")
        text = re.sub(r"\blambda\b", "lamda", text)
        names = dict(units)
        for base in (10, 2):          # log10(x) and log2(x), which SymPy doesn't know (step 4, gap G2)
            if f"log{base}" not in lets:
                names[f"log{base}"] = sp.Lambda(LOG_ARGUMENT, sp.log(LOG_ARGUMENT, base))
        for match in NAME.finditer(text):
            name = match.group()
            if name in names or name in CONSTANTS:
                continue
            called = text[match.end():].lstrip().startswith("(")
            if name in self.broken and name not in lets:
                raise UsesBroken(f"uses {name}, whose definition couldn't be read")
            if name in lets:
                names[name] = lets[name]
            elif not called:
                names[name] = self.symbol(name)
        try:
            value = safe_sympify.sympify(text, names)
        except safe_sympify.Refused as error:
            raise Unreadable(f"{text.strip()!r} isn't read: {error}") from None
        except (sp.SympifyError, SyntaxError, TypeError, AttributeError) as error:
            raise Unreadable(f"SymPy couldn't read {text.strip()!r}: {type(error).__name__}") from None
        if not isinstance(value, sp.Expr):
            raise Unreadable(f"{text.strip()!r} isn't an expression")
        value = self.resolved(value.doit(), lets)
        if value.has(sp.Order):
            value = value.removeO()
        return value

    def resolved(self, value, lets: dict, rounds: int = 6):
        """Names given a value after a formula used them: models often write the formulas first
        and the values after (let Gamma = G_F*M_W**3/... and then G_F = 1.1664e-5 [GeV^-2])."""
        for _ in range(rounds):
            late = {symbol: lets[symbol.name] for symbol in value.free_symbols
                    if isinstance(symbol, sp.Symbol) and symbol.name in lets
                    and not any(inner.name == symbol.name for inner in lets[symbol.name].free_symbols
                                if isinstance(inner, sp.Symbol))}
            if not late:
                return value
            value = value.xreplace(late).doit()
        return value

    def undefined_functions(self, value) -> list[str]:
        from sympy.core.function import AppliedUndef
        return sorted({type(call).__name__ for call in value.atoms(AppliedUndef)})


# --- checking --------------------------------------------------------------------------

def has_units(value) -> bool:
    return bool(value.atoms(u.Quantity))


def dimension_of(value):
    return SI.get_dimensional_expr(value)


def in_base_units(value):
    return sp.simplify(u.convert_to(value, BASE))


def same_dimension(left, right) -> bool:
    dl, dr = dimension_of(left), dimension_of(right)
    try:
        return dimsys_SI.equivalent_dims(u.Dimension(dl), u.Dimension(dr))
    except (TypeError, ValueError):
        return sp.simplify(dl - dr) == 0


def number(value) -> complex | None:
    try:
        result = complex(sp.N(value, 30))
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def sample_points(free: list, rng: random.Random) -> dict:
    """Positive points, as physics quantities mostly are; a third of them below 1, so that
    a claim about fractions such as 1 - beta has points inside its domain."""
    point = {}
    for symbol in free:
        if symbol.is_integer:
            point[symbol] = rng.randint(2, 12)
        elif rng.random() < 1 / 3:
            point[symbol] = sp.Rational(rng.randint(1, 19), 20)
        else:
            point[symbol] = sp.Rational(rng.randint(1, 40), 7)
    return point


def compare_numbers(left: complex, right: complex, sign: str, step) -> tuple[str, str]:
    """Two plain numbers. `step` is one unit in the last stated digit of the stated side."""
    if sign in ("<", "<=", ">", ">=", "≤", "≥"):
        if abs(left.imag) > TOLERANCE or abs(right.imag) > TOLERANCE:
            return "undecided", "complex values can't be ordered"
        a, b = left.real, right.real
        held = {"<": a < b, "<=": a <= b, "≤": a <= b, ">": a > b, ">=": a >= b, "≥": a >= b}[sign]
        return ("passed" if held else "failed"), f"{a:.6g} {sign} {b:.6g} is {'true' if held else 'false'}"
    gap = abs(left - right)
    scale = max(abs(left), abs(right), 1e-300)
    if step is not None:
        allowed = max(float(step), PRECISION_FLOOR * scale)
        if gap <= allowed * (1 + 1e-9):
            return "passed", f"matches to the digits given ({left.real:.6g} vs {right.real:.6g}, allowed ±{allowed:.3g})"
        return "failed", f"{left.real:.6g} vs {right.real:.6g}: off by {gap:.3g}, more than the ±{allowed:.3g} the digits allow"
    if gap <= TOLERANCE * scale:
        return "passed", "equal"
    return "failed", f"{left.real:.10g} vs {right.real:.10g}"


def natural(value):
    """In powers of eV with hbar = c = 1: 1 s = 1/(6.58e-16 eV), 1 m = 1/(1.97e-7 eV), 1 kg = 5.61e35 eV."""
    base = in_base_units(value)
    return sp.simplify(base.subs({u.second: 1 / (HBAR_EV_S * NATURAL_EV),
                                  u.meter: 1 / (HBARC_EV_M * NATURAL_EV),
                                  u.kilogram: NATURAL_EV / EV_IN_KG}))


def check_with_units(left, right, left_text: str, right_text: str, sign: str, shown: dict,
                     plain_left=None, plain_right=None) -> dict:
    """Compared in SI base units; if the dimensions differ there, again with hbar = c = 1, as
    particle physics writes them. Units that still differ mean failed only if the numbers as
    written differ too; if they agree, a constant probably lost its unit: undecided."""
    if left.free_symbols or right.free_symbols:
        return {"verdict": "undecided", **shown,
                "note": "a side has symbols without values; give each quantity a value with let to check units"}
    units_note = "same units"
    if same_dimension(left, right):
        ratio = in_base_units(left / right)
    else:
        ratio = sp.simplify(natural(left) / natural(right))
        units_note = "same units with hbar = c = 1"
        if ratio.free_symbols or has_units(ratio):
            return units_differ(left, right, left_text, right_text, sign, shown, plain_left, plain_right)
    value = number(ratio) if not has_units(ratio) else None
    if value is None:
        return {"verdict": "undecided", **shown, "note": "the two sides couldn't be compared as numbers"}
    natural_basis = ["natural_units"] if units_note != "same units" else []
    if sign in ("<", "<=", ">", ">=", "≤", "≥"):      # left/right against 1 keeps the direction
        verdict, _ = compare_numbers(value, 1 + 0j, sign, None)
        return {"verdict": verdict, **shown, "note": f"{units_note}; left/right = {value.real:.6g}",
                "basis": natural_basis}
    stated_text = right_text if stated_step(right_text, True) is not None else (
        left_text if stated_step(left_text, True) is not None else None)
    if stated_text is None:
        verdict, note = compare_numbers(value, 1 + 0j, sign, None)
        return {"verdict": verdict, **shown, "note": f"{units_note}; left/right = {value.real:.10g}",
                "basis": natural_basis}
    magnitude = abs(float(UNIT_BRACKET.sub("", stated_text)))
    relative = max(float(stated_step(stated_text, True)) / magnitude if magnitude else 0.0, PRECISION_FLOOR)
    other = value if stated_text is right_text else 1 / value     # computed side / stated side
    verdict, _ = compare_numbers(other, 1 + 0j, sign, relative)
    if verdict == "failed" and units_note != "same units":
        # Matched only with hbar = c = 1 and the numbers differ: was a unit dropped instead?
        dropped = units_differ(left, right, left_text, right_text, sign, shown, plain_left, plain_right)
        if dropped["verdict"] == "undecided":
            return dropped
    note = (f"{units_note}; {'within' if verdict == 'passed' else 'outside'} the digits given "
            f"(computed/stated = {other.real:.6g}, allowed 1 ± {relative:.3g})")
    basis = (["natural_units"] if units_note != "same units" else []) + (
        ["stated_digits"] if abs(other - 1) > TOLERANCE else [])
    return {"verdict": verdict, **shown, "note": note, "basis": basis}


def units_differ(left, right, left_text, right_text, sign, shown, plain_left, plain_right) -> dict:
    mismatch = f"units don't match, even with hbar = c = 1: {dimension_of(left)} vs {dimension_of(right)}"
    if plain_left is not None and plain_right is not None and not (plain_left.free_symbols or plain_right.free_symbols):
        lv, rv = number(plain_left), number(plain_right)
        if lv is not None and rv is not None:
            step = stated_step(right_text, True)
            step = step if step is not None else stated_step(left_text, True)
            verdict, _ = compare_numbers(lv, rv, sign, step)
            if verdict == "passed":
                return {"verdict": "undecided", **shown,
                        "note": f"{mismatch}; the numbers as written agree, so a constant such as hbar "
                                f"probably lost its unit: give it one, as in let hbar = 6.582e-25 [GeV*s]"}
    return {"verdict": "failed", **shown, "note": mismatch}


def check_pair(left, right, left_text: str, right_text: str, sign: str,
               plain_left=None, plain_right=None) -> dict:
    approximate = sign in ("≈", "~=")
    shown = {"left": str(left)[:300], "right": str(right)[:300]}
    if has_units(left) or has_units(right):
        return check_with_units(left, right, left_text, right_text, sign, shown, plain_left, plain_right)

    if sign in ("<", "<=", ">", ">=", "≤", "≥"):
        difference = sp.simplify(left - right)
        if not difference.free_symbols:
            value = number(difference)
            if value is None:
                return {"verdict": "undecided", **shown, "note": "couldn't evaluate"}
            verdict, note = compare_numbers(number(left), number(right), sign, None)
            return {"verdict": verdict, **shown, "note": note}
        decided = {"<": difference.is_negative, ">": difference.is_positive,
                   "<=": difference.is_nonpositive, "≤": difference.is_nonpositive,
                   ">=": difference.is_nonnegative, "≥": difference.is_nonnegative}[sign]
        if decided is True:
            return {"verdict": "passed", **shown, "note": "proved from the assumptions"}
        if decided is False:
            return {"verdict": "failed", **shown, "note": "false under the assumptions"}
        return {"verdict": "undecided", **shown, "note": "SymPy can't decide this inequality; declare assumptions"}

    difference = sp.simplify(left - right)
    shown["difference"] = str(difference)[:300]
    if difference == 0:
        return {"verdict": "passed", **shown, "note": "proved: the difference simplifies to zero"}
    if not difference.free_symbols:
        # Symbols can cancel within a side: (y1*(-y1))/(sqrt(y1**2)*sqrt(y1**2)) is -1.
        lv = number(left) if not left.free_symbols else number(sp.simplify(left))
        rv = number(right) if not right.free_symbols else number(sp.simplify(right))
        if lv is None or rv is None:
            gap = number(difference)
            if gap is not None and abs(gap) > TOLERANCE:
                return {"verdict": "failed", **shown, "note": f"the sides differ by a constant, {gap.real:.6g}"}
            return {"verdict": "undecided", **shown, "note": "couldn't evaluate the sides as numbers"}
        step = stated_step(right_text, approximate)
        if step is None:
            step = stated_step(left_text, approximate)
        verdict, note = compare_numbers(lv, rv, sign, step)
        inexact = abs(lv - rv) > TOLERANCE * max(abs(lv), abs(rv), 1e-300)
        return {"verdict": verdict, **shown, "note": note,
                "basis": ["stated_digits"] if step is not None and inexact else []}
    free = sorted(difference.free_symbols, key=str)
    rng = random.Random(SEED)
    worst, tried = None, 0
    for _ in range(SAMPLES * 4):        # points outside the real domain don't count; try more
        gap = relative_gap(difference, left, right, sample_points(free, rng))
        if gap is not None:
            tried += 1
            worst = gap if worst is None else max(worst, gap)
            if tried == SAMPLES:
                break
    if worst is None:
        return {"verdict": "undecided", **shown, "note": "no point tried gave a real number on both sides"}
    undeclared = [s for s in free if not declared(s)]
    if worst > 1e-8 and undeclared and agrees_at_integers(difference, left, right, free, rng):
        names = ", ".join(map(str, undeclared))
        return {"verdict": "numerically_equal", **shown,
                "note": f"false at fractional values but true at the whole numbers tried; if {names} "
                        f"are integers, declare them with 'assume integer: {names}'"}
    if worst > 1e-8 and left.free_symbols and right.free_symbols and left.free_symbols.isdisjoint(right.free_symbols):
        # cV/cA = 1 - 4*s with nothing defining cV, cA or s: a physical relation between
        # quantities the block never defines, not an identity (step 4, gap G1).
        return {"verdict": "undecided", **shown,
                "note": f"the two sides share no symbol ({', '.join(sorted(map(str, left.free_symbols)))} against "
                        f"{', '.join(sorted(map(str, right.free_symbols)))}), and nothing defines them; "
                        f"give definitions or values with let to check it"}
    if worst > 1e-8 and bool(left.free_symbols) != bool(right.free_symbols):
        # A formula in symbols with no values against a number: it can't be checked, so it isn't failed.
        loose = sorted(map(str, (left.free_symbols or right.free_symbols)))
        return {"verdict": "undecided", **shown,
                "note": f"one side is a number and the other depends on {', '.join(loose)}, which "
                        f"{'has' if len(loose) == 1 else 'have'} no value; give values with let to check it"}
    if worst > 1e-8:
        return {"verdict": "failed", **shown,
                "note": f"the sides differ at points that satisfy the assumptions (relative gap up to {worst:.3g})"}
    note = f"no proof, but the sides agreed at all {tried} points tried"
    undeclared = [str(s) for s in free if not declared(s)]
    if undeclared:
        note += f"; declaring {', '.join(undeclared)} positive (or integer) may let SymPy prove it"
    return {"verdict": "numerically_equal", **shown, "note": note}


def declared(symbol) -> bool:
    return any(symbol.assumptions0.get(kind) for kind in ASSUMPTIONS)


def outside_domain(expressions, point: dict) -> bool:
    """A root of a negative number or the log of a non-positive one at this point: outside the
    real domain, as sqrt(1 - beta) for beta > 1, where branch cuts can break an identity that
    holds inside it (sqrt(1 - a)*sqrt(1 - b) is -sqrt((1 - a)*(1 - b)) there, though both are real)."""
    for expression in expressions:
        for node in sp.preorder_traversal(expression):
            if isinstance(node, sp.Pow) and not node.exp.is_integer:
                inner, strict = node.base, False
            elif isinstance(node, sp.log):
                inner, strict = node.args[0], True
            else:
                continue
            if not inner.free_symbols:
                continue
            value = number(inner.subs(point))
            if value is None:
                continue
            if abs(value.imag) > TOLERANCE * max(abs(value), 1) or value.real < 0 or (strict and value.real == 0):
                return True
    return False


def relative_gap(difference, left, right, point: dict) -> float | None:
    """None at a point outside the claim's real domain (see outside_domain), or where a side
    isn't a real number."""
    if outside_domain((left, right), point):
        return None
    value = number(difference.subs(point))
    lv, rv = number(left.subs(point)), number(right.subs(point))
    if value is None or lv is None or rv is None:
        return None
    if abs(lv.imag) > TOLERANCE * max(abs(lv), 1) or abs(rv.imag) > TOLERANCE * max(abs(rv), 1):
        return None
    scale = max(abs(lv), abs(rv))
    return 0.0 if scale == 0 else abs(value) / scale


def agrees_at_integers(difference, left, right, free: list, rng: random.Random) -> bool:
    """For a claim that only holds for whole numbers, such as sin(pi n) = 0 with n undeclared."""
    gaps = [relative_gap(difference, left, right, {s: rng.randint(2, 12) for s in free}) for _ in range(SAMPLES)]
    gaps = [gap for gap in gaps if gap is not None]
    return bool(gaps) and max(gaps) <= 1e-8


def worst(verdicts: list[str]) -> str:
    return min(verdicts, key=VERDICT_ORDER.index)


SUBS = re.compile(r"\bsubs\(([^)]*)")
SUBS_NAME = re.compile(r"(?:^|[{(\[,])\s*([A-Za-z_]\w*)\s*[:,]")


def check_line(reader: Reader, body: str) -> dict | None:
    """The claim's result, or None for a line that only defined a name."""
    for found in SUBS.finditer(body):
        # subs({l: sqrt(n)}) after let l = 128 replaces nothing: l is already 128.
        valued = sorted({name for name in SUBS_NAME.findall(found.group(1))
                         if name in reader.lets and not reader.lets[name].free_symbols})   # a number, not a variable
        if valued:
            return {"verdict": "undecided",
                    "note": f"substitutes for {', '.join(valued)}, which already "
                            f"{'has a value' if len(valued) == 1 else 'have values'}; use a new name for the variable"}
    sides, signs = split_relations(body)
    if not signs:
        raise Unreadable("no relation: a claim needs =, ≈, <, <=, > or >=")
    if reader.defines(sides[0], signs[0]):
        reader.define(sides[0], sides[1])
        sides, signs = sides[1:], signs[1:]
        if not signs:
            return None
    values = [reader.expression(side) for side in sides]
    plains = [reader.expression(side, plain=True) for side in sides]
    missing = sorted({name for value in values for name in reader.undefined_functions(value)})
    if missing:
        return {"verdict": "undecided",
                "note": f"uses {', '.join(missing)}, which {'has' if len(missing) == 1 else 'have'} no definition"}
    pairs = [check_pair(values[i], values[i + 1], sides[i], sides[i + 1], sign, plains[i], plains[i + 1])
             for i, sign in enumerate(signs)]
    verdict = worst([pair["verdict"] for pair in pairs])
    deciding = next(pair for pair in pairs if pair["verdict"] == verdict)
    result = {"verdict": verdict, "note": deciding.get("note", "")}
    if verdict == "passed":
        basis = {kind for pair in pairs for kind in pair.get("basis", [])}
        if basis:
            result["basis"] = [kind for kind in BASES if kind in basis]
    for key in ("left", "right", "difference"):
        if key in deciding:
            result[key] = deciding[key]
    if len(pairs) > 1:
        result["links"] = [pair["verdict"] for pair in pairs]
    return result


def run(text: str) -> dict:
    reader = Reader()
    claims, notes = [], []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > MAX_LINES:
        notes.append(f"only the first {MAX_LINES} lines were read")
        lines = lines[:MAX_LINES]
    deadline = time.monotonic() + TOTAL_SECONDS
    for raw in lines:
        line = raw.strip()[:MAX_LINE_CHARS]
        body, label = split_label(line)
        body = normalised(body)
        if not body or SKIPPED.match(body) or body.strip("` \t").lower() in ("checks", ""):
            continue        # imports, and the fence or its name written again inside the block
        entry = {"line": line, "label": label}
        left = deadline - time.monotonic()
        if left < 1:
            claims.append({**entry, "verdict": "undecided", "note": "not checked: the block ran out of time"})
            continue
        try:
            # One alarm for the whole line: reading a side already evaluates its integrals.
            signal.alarm(max(1, min(CLAIM_SECONDS, int(left))))
            try:
                if match := re.fullmatch(r"assume\s+(\w+)\s*:\s*(.+)", body):
                    reader.assume(match.group(1), match.group(2))
                    continue
                function = FUNCTION.fullmatch(body)
                if function and function.group(1) not in SYMPY_NAMES and (
                        body.startswith("let") or function.group(1) not in reader.lets):
                    reader.define(function.group(1), function.group(3), function.group(2))
                    continue
                if match := LET.fullmatch(body):
                    reader.define(match.group(1), match.group(2))
                    continue
                result = check_line(reader, body)
                if result is not None:
                    claims.append({**entry, **result})
            finally:
                signal.alarm(0)
        except UsesBroken as error:
            claims.append({**entry, "verdict": "undecided", "note": str(error)[:300]})
        except Unreadable as error:
            claims.append({**entry, "verdict": "error", "note": str(error)[:300]})
        except OutOfTime:
            claims.append({**entry, "verdict": "undecided", "note": f"SymPy didn't finish within {CLAIM_SECONDS} s"})
        except Exception as error:      # anything else SymPy raises is the claim's problem, not the run's
            claims.append({**entry, "verdict": "error", "note": f"{type(error).__name__}: {error}"[:300]})
    return {"claims": claims, "notes": notes}


def main() -> None:
    nonce = safe_sympify.read_nonce()          # first: no claim may see it
    signal.signal(signal.SIGALRM, alarm)
    try:
        text = open(PATH, encoding="utf-8").read()
        result = run(text)
    except Exception as error:
        result = {"claims": [], "notes": [f"{type(error).__name__}: {error}"[:300]]}
    safe_sympify.report(result, nonce)


if __name__ == "__main__":
    sys.exit(main())
