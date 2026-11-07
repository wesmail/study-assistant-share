"""sandbox/verify_claims.py, tested here on claims written for these tests.

It runs inside the image in the app; here it's loaded directly, as test_sandbox_scripts.py
does, since it needs only SymPy. The real container is in test_sandbox_docker.py.
"""
import importlib.util
import signal
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "sandbox" / "verify_claims.py"
sys.path.append(str(SCRIPT.parent))            # it imports safe_sympify, as it does in the image
spec = importlib.util.spec_from_file_location("sandbox_verify_claims", SCRIPT)
verify_claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_claims)


@pytest.fixture(autouse=True)
def alarm():
    previous = signal.signal(signal.SIGALRM, verify_claims.alarm)
    yield
    signal.signal(signal.SIGALRM, previous)


def claims(text: str) -> list[dict]:
    return verify_claims.run(text)["claims"]


def verdict(text: str) -> str:
    found = claims(text)
    assert len(found) == 1, found
    return found[-1]["verdict"]


class TestProofs:
    def test_a_true_derivative_is_proved(self):
        assert verdict("let sig = 1/(1 + exp(-f))\ndiff(sig, f) = sig*(1 - sig)") == "passed"

    def test_a_sign_error_fails(self):
        block = "let sig = 1/(1 + exp(-f))\nlet L = -(1 - y)*log(1 - sig) - y*log(sig)\ndiff(L, f) = y - sig"
        assert verdict(block) == "failed"

    def test_a_definite_integral(self):
        block = "assume positive: s\nintegrate(x**2*exp(-x**2/(2*s**2))/sqrt(2*pi*s**2), (x, 0, oo)) = s**2/2"
        assert verdict(block) == "passed"

    def test_a_wrong_integral_fails(self):
        assert verdict("integrate(exp(-x**2), (x, 0, oo)) = pi/2") == "failed"

    def test_a_limit(self):
        assert verdict("assume positive: N\nlimit((N - 1)/N, N, oo) = 1") == "passed"

    def test_a_series_drops_its_order_term(self):
        right = "1 + x - x**2/2 - x**3/6 + x**4/24 + x**5/120"
        assert verdict(f"series(sin(x) + cos(x), x, 0, 6) = {right}") == "passed"
        assert verdict(f"series(sin(x) + cos(x), x, 0, 6) = {right.replace('120', '60')}") == "failed"

    def test_a_sum(self):
        assert verdict("assume positive: n\nassume integer: n\nSum(k, (k, 1, n)) = n*(n + 1)/2") == "passed"

    def test_names_that_are_sympy_functions_are_symbols_when_not_called(self):
        # N, E, S, I and gamma would otherwise be SymPy's N(), Euler's e, S, i and the gamma function.
        assert verdict("assume positive: N\nsqrt(N)**2 = N") == "passed"
        assert verdict("E*gamma/gamma = E") == "passed"
        assert verdict("gamma(5) = 24") == "passed"      # called, it's the function

    def test_lambda_is_a_name(self):
        assert verdict("lambda*2 = 2*lambda") == "passed"


class TestNumbers:
    def test_a_rounded_number_passes_within_its_last_digit(self):
        assert verdict("12288/223 = 55.1") == "passed"
        assert verdict("12288/223 = 55.3") == "failed"

    def test_an_integer_with_equals_is_exact(self):
        assert verdict("floor((12288 - 1)/128) + 1 = 96") == "passed"
        assert verdict("12288/223 = 55") == "failed"

    def test_an_integer_with_approx_allows_rounding(self):
        assert verdict("12288/223 ≈ 55") == "passed"
        assert verdict("12288/223 ~= 56") == "failed"

    def test_scientific_notation_sets_the_last_digit(self):
        assert verdict("1/3000 = 3.33e-4") == "passed"
        assert verdict("1/3000 = 3.36e-4") == "failed"

    def test_inequalities_between_numbers(self):
        assert verdict("sqrt(12288) < 128") == "passed"
        assert verdict("sqrt(12288) >= 128") == "failed"

    def test_an_inequality_from_assumptions(self):
        assert verdict("assume positive: x\nx**2 + 1 > 0") == "passed"
        assert verdict("x**2 - 2*x > 0") == "undecided"


class TestUnits:
    BLOCK = ("let G_F = 1.1664e-5 [GeV^-2]\nlet M_W = 80.38 [GeV]\n"
             "let Gamma_l = G_F*M_W**3/(6*sqrt(2)*pi)\n")

    def test_a_width_in_gev_and_in_mev(self):
        assert verdict(self.BLOCK + "Gamma_l = 0.227 [GeV]") == "passed"
        assert verdict(self.BLOCK + "Gamma_l = 227 [MeV]") == "passed"     # a whole number with a unit is rounded

    def test_a_wrong_factor_fails(self):
        assert verdict(self.BLOCK + "3*Gamma_l = 2.05 [GeV]") == "failed"

    def test_a_lifetime_needs_hbar_and_comes_out_in_seconds(self):
        assert verdict(self.BLOCK + "[hbar]/(9*Gamma_l) = 3.2e-25 [s]") == "passed"

    def test_mismatched_units_fail_and_say_so(self):
        found = claims(self.BLOCK + "[hbar]/(9*Gamma_l) = 3.2e-25 [GeV]")[-1]
        assert found["verdict"] == "failed"
        assert "units don't match" in found["note"]

    def test_a_cross_section_converted_with_hbar_c(self):
        block = "let s0 = 41.541 [nb] / (0.3894 [mb*GeV^2])\ns0 = 1.067e-4 [GeV^-2]"
        assert verdict(block) == "passed"
        assert verdict("let s0 = 41.541 [nb]\ns0 = 2.0e-4 [GeV^-2]") == "failed"

    def test_symbols_without_values_leave_units_undecided_not_failed(self):
        assert verdict("G_F*M_W**3 = 6 [GeV]") == "undecided"

    def test_an_unknown_unit_is_an_error(self):
        found = claims("x = 3 [parsec]")[-1]
        assert found["verdict"] == "error"
        assert "unknown unit" in found["note"]

    def test_an_inequality_with_units(self):
        assert verdict(self.BLOCK + "Gamma_l > 0.2 [GeV]") == "passed"
        assert verdict(self.BLOCK + "0.2 [GeV] > Gamma_l") == "failed"


class TestNoProof:
    def test_agreement_without_proof_is_numerically_equal_never_passed(self):
        found = claims("sqrt(x**2) = x")[-1]
        assert found["verdict"] == "numerically_equal"
        assert "positive" in found["note"]

    def test_a_claim_true_only_for_integers_is_not_called_false(self):
        found = claims("sin(pi*n) = 0")[-1]
        assert found["verdict"] == "numerically_equal"
        assert "assume integer: n" in found["note"]

    def test_declared_integer_proves_it(self):
        assert verdict("assume integer: n\nsin(pi*n) = 0") == "passed"


class TestReading:
    def test_a_chain_checks_each_link_and_reports_the_worst(self):
        found = claims("2*3 = 6 = 7")[-1]
        assert found["verdict"] == "failed"
        assert found["links"] == ["passed", "failed"]

    def test_labels_are_kept(self):
        assert claims("1 + 1 = 2   # arithmetic")[-1]["label"] == "arithmetic"

    def test_assume_and_let_lines_are_not_claims(self):
        assert claims("assume positive: x\nlet y = x**2") == []

    def test_square_brackets_for_a_call_are_an_error_with_a_hint(self):
        found = claims("sig[f] = 1")[-1]
        assert found["verdict"] == "error"
        assert "parentheses" in found["note"]

    def test_a_line_without_a_relation_is_an_error(self):
        assert verdict("x**2 + 1") == "error"

    def test_latex_is_an_error_not_a_crash(self):
        assert verdict(r"\frac{1}{2} = 0.5") == "error"

    def test_an_unknown_assumption_is_an_error(self):
        assert verdict("assume prime: p") == "error"

    def test_only_the_first_forty_lines_are_read(self):
        result = verify_claims.run("\n".join(["1 = 1"] * 45))
        assert len(result["claims"]) == 40
        assert result["notes"]

    def test_a_slow_claim_is_undecided(self, monkeypatch):
        monkeypatch.setattr(verify_claims, "CLAIM_SECONDS", 1)
        found = claims("integrate(exp(-x**4 + sin(x)**3)*log(1 + x**7), (x, 0, oo)) = 1")[-1]
        assert found["verdict"] in ("undecided", "error")

    def test_one_bad_line_doesnt_stop_the_rest(self):
        found = claims("x[ = 1\n1 + 1 = 2")
        assert [claim["verdict"] for claim in found] == ["error", "passed"]

    def test_eval_tricks_stay_text(self):
        # Inside the sandbox anyway; here it must at least not run as Python.
        found = claims("__import__('os').getcwd() = 1")[-1]
        assert found["verdict"] in ("error", "failed", "undecided")


class TestHowModelsWrite:
    """Patterns from the first dev run (dev-checks-fast-1)."""

    def test_the_first_name_equals_defines_and_the_next_checks(self):
        block = ("let sig = 1/(1 + exp(-f))\nL = -(1 - y)*log(1 - sig) - y*log(sig)\n"
                 "diff(L, f) = sig - y\nL = -log(sig)")
        assert [claim["verdict"] for claim in claims(block)] == ["passed", "failed"]

    def test_a_definition_chain_checks_the_rest(self):
        assert [c["verdict"] for c in claims("MAP = (2 + 7 - 1)/(2 + 2 + 10 - 2) = 8/12 = 2/3")] == ["passed"]

    def test_approx_never_defines(self):
        assert verdict("Gamma ≈ 0.227") == "undecided"

    def test_a_formula_without_values_against_a_number_is_undecided_not_failed(self):
        found = claims("Gamma = G_F*M_W**3/(6*sqrt(2)*pi)\nGamma = 0.23")[-1]
        assert found["verdict"] == "undecided"
        assert "G_F, M_W" in found["note"]

    def test_primes_are_names(self):
        block = "alpha' = 2 + 7\nbeta' = 2 + 3\nalpha'/(alpha' + beta') = 9/14"
        assert [c["verdict"] for c in claims(block)] == ["passed"]

    def test_a_quoted_name_is_not_a_prime(self):
        assert verdict("let x = symbols('x')\ndiff(x**2, x) = 2*x") == "passed"

    def test_tiny_values_are_compared_relatively(self):
        # The sides once "agreed" because the relative gap had a floor of 1.
        assert verdict("assume positive: a\n1e-25*a = 2e-25*a") == "failed"


class TestHowModelsWriteRoundTwo:
    """Patterns from dev-checks-fast-2 and dev-checks-balanced-1."""

    def test_python_habits_are_read(self):
        block = ("import sympy as sp\nz = sp.symbols('z')\ny = sp.symbols('y')\nsig = 1 / (1 + sp.exp(-z))\n"
                 "L = -(1 - y) * sp.log(1 - sig) - y * sp.log(sig)\nsp.diff(L, z) == sig - y")
        assert [c["verdict"] for c in claims(block)] == ["passed"]

    def test_a_claim_using_a_broken_definition_is_undecided_not_failed(self):
        found = claims("sig = 1/(1 + foo.exp(-z))\ndiff(sig, z) = sig*(1 - sig)")
        assert [c["verdict"] for c in found] == ["error", "undecided"]
        assert "sig" in found[1]["note"]

    def test_function_definitions(self):
        block = "let f_Y(y) = 3/2*y**(1/2)\nintegrate(f_Y(y), (y, 0, 1)) = 1\nf(x) = x**2\nf(3) = 9"
        assert [c["verdict"] for c in claims(block)] == ["passed", "passed"]

    def test_a_python_lambda(self):
        assert verdict("let f = lambda x: sin(x) + cos(x)\nf(0) = 1") == "passed"

    def test_a_sympy_function_on_the_left_is_a_claim_not_a_definition(self):
        assert verdict("diff(x**2, x) = 2*x") == "passed"
        assert verdict("diff(x**2, x) = 3*x") == "failed"

    def test_an_undefined_function_is_undecided(self):
        found = claims("integral_y = integrate(g(y), (y, 0, 1))\nintegral_y = 1")[-1]
        assert found["verdict"] == "undecided" and "g" in found["note"]

    def test_type_words_in_brackets_are_ignored(self):
        assert verdict("let N = 10 [int]\nN/2 = 5") == "passed"

    def test_natural_units_a_lifetime_from_a_width(self):
        assert verdict("let Gamma = 2.045 [GeV]\n1/Gamma = 3.22e-25 [s]") == "passed"
        assert verdict("let Gamma = 2.045 [GeV]\n1/Gamma = 4.1e-25 [s]") == "failed"

    def test_natural_units_a_cross_section_in_nb_and_gev(self):
        assert verdict("let s0 = 41.541 [nb]\ns0 = 1.067e-4 [GeV^-2]") == "passed"

    def test_a_constant_that_lost_its_unit_is_undecided_not_failed(self):
        block = "let Gamma = 2.045 [GeV]\nlet tau = 6.582e-25/Gamma\ntau ≈ 3.22e-25 [s]"
        found = claims(block)[-1]
        assert found["verdict"] == "undecided" and "lost its unit" in found["note"]

    def test_units_that_differ_and_numbers_that_differ_fail(self):
        assert verdict("let Gamma = 2.045 [GeV]\nlet tau = 6.582e-25/Gamma\ntau ≈ 9.9e-25 [s]") == "failed"

    def test_a_number_given_to_sixteen_digits_isnt_held_to_float_noise(self):
        assert verdict("1/3 ≈ 0.3333333333333333") == "passed"

    def test_digits_past_the_sixth_dont_fail_a_claim_but_a_real_slip_does(self):
        assert verdict("sin(1) + cos(1) ≈ 1.38177337") == "passed"       # true value 1.381773291
        assert verdict("sin(1) + cos(1) ≈ 1.38187337") == "failed"

    def test_numpy_math_and_ceil(self):
        assert verdict("np.sqrt(16) + math.ceil(12288/128) = 100") == "passed"


class TestHowItPassed:
    """A passed claim says how, when it wasn't a proof: the reader sees "passed (to stated
    digits)" or "passed (natural units)" (user's decision, Nov 5); both still count."""

    def found(self, text: str) -> dict:
        result = claims(text)
        assert len(result) == 1, result
        return result[0]

    def test_a_proof_has_no_basis(self):
        assert "basis" not in self.found("let sig = 1/(1 + exp(-f))\ndiff(sig, f) = sig*(1 - sig)")

    def test_a_rounded_number_passes_to_stated_digits(self):
        assert self.found("12288/223 = 55.1")["basis"] == ["stated_digits"]

    def test_an_exact_number_is_not_labelled(self):
        assert "basis" not in self.found("1/2 = 0.5")
        assert "basis" not in self.found("floor((12288 - 1)/128) + 1 = 96")

    def test_si_units_to_stated_digits(self):
        block = TestUnits.BLOCK + "Gamma_l = 0.227 [GeV]"
        assert self.found(block)["basis"] == ["stated_digits"]

    def test_natural_units_and_stated_digits(self):
        found = self.found("let Gamma = 2.045 [GeV]\n1/Gamma = 3.22e-25 [s]")
        assert found["verdict"] == "passed" and found["basis"] == ["natural_units", "stated_digits"]

    def test_natural_units_in_an_inequality(self):
        found = self.found("let Gamma = 2.045 [GeV]\n1/Gamma < 1e-24 [s]")
        assert found["verdict"] == "passed" and found["basis"] == ["natural_units"]

    def test_a_failed_claim_carries_no_basis(self):
        assert "basis" not in self.found("12288/223 = 55.3")

    def test_a_chain_collects_every_link(self):
        found = self.found("let Gamma = 2.045 [GeV]\n1/Gamma = 3.22e-25 [s] = 322 [fs]*1e-12")
        assert found["verdict"] == "passed", found
        assert found["basis"] == ["natural_units", "stated_digits"]


class TestClaimTestSetLessons:
    """What the first run on the claim test set (step 2) showed."""

    def test_trailing_zeros_of_a_rounded_whole_number_arent_significant(self):
        assert verdict("let f_s = 1.9 [Hz]\n2*320*16/f_s ≈ 5390 [s]") == "passed"       # 5389.5
        assert verdict("let f_s = 1.9 [Hz]\n2*320*16/f_s ≈ 5420 [s]") == "failed"
        assert verdict("10240/1.9 ≈ 5390") == "passed"

    def test_an_exact_whole_number_stays_exact(self):
        assert verdict("2*320*16 = 10240") == "passed"
        assert verdict("2*320*16 + 3 = 10240") == "failed"

    def test_points_outside_the_real_domain_dont_fail_an_identity(self):
        block = "assume positive: b1, b2\nsqrt(1 - b2)*sqrt(1 - b1) = sqrt((1 - b1)*(1 - b2))"
        assert verdict(block) == "numerically_equal"        # true for b < 1, never proved, never passed

    def test_a_wrong_claim_inside_the_domain_still_fails(self):
        assert verdict("assume positive: b\nsqrt(1 - b) = 1 - b/3") == "failed"

    def test_sides_that_simplify_to_different_constants_fail(self):
        block = "assume real: y1, y2\n(y1*(-y1) + y2*(-y2))/(sqrt(y1**2 + y2**2)*sqrt(y1**2 + y2**2)) = 1"
        found = claims(block)[-1]
        assert found["verdict"] == "failed"
        assert verdict(block.replace("= 1", "= -1")) == "passed"


class TestHowModelsWriteRoundThree:
    """Best-of-3 dev run in Fast (step 3)."""

    def test_values_given_after_the_formulas(self):
        block = ("assume positive: G_F, M_W, hbar\nlet Gamma_l = G_F*M_W**3/(6*sqrt(2)*pi)\nlet tau = hbar/(9*Gamma_l)\n"
                 "G_F = 1.1664e-5 [GeV^-2]\nM_W = 80.38 [GeV]\nhbar = 6.582e-25 [GeV*s]\n")
        assert verdict(block + "Gamma_l ≈ 0.227 [GeV]") == "passed"
        assert verdict(block + "tau ≈ 3.22e-25 [s]") == "passed"
        assert verdict(block + "Gamma_l ≈ 0.23 [GeV]") == "passed"
        assert verdict(block + "9*Gamma_l ≈ 2.07 [GeV]") == "failed"         # 2.045

    def test_units_separated_by_a_space_multiply(self):
        assert verdict("let hbarc2 = 0.3894 [mb GeV^2]\nlet s0 = 41.541 [nb]\ns0/hbarc2 ≈ 1.067e-4 [GeV^-2]") == "passed"

    def test_the_fence_or_its_name_inside_the_block_is_skipped(self):
        assert claims("checks\n```checks\nlet N = 10\nN/2 = 5\n```") == claims("N = 10\nN/2 = 5")
        assert verdict("checks\nlet N = 10\nN/2 = 5") == "passed"

    def test_subs_for_a_name_with_a_value_is_undecided_not_failed(self):
        block = "assume positive: l, n\nlet n = 12288\nlet l = 128\ncost = l + n/l\n"
        found = claims(block + "cost.subs({l: sqrt(n)}) = 2*sqrt(n)")[-1]
        assert found["verdict"] == "undecided" and "already has a value" in found["note"]
        assert verdict("assume positive: l, n\ncost = l + n/l\ncost.subs({l: sqrt(n)}) = 2*sqrt(n)") == "passed"

    def test_subs_for_a_name_that_is_a_variable_is_checked(self):
        block = "let x = symbols('x')\nlet f = sin(x) + cos(x)\n"
        assert verdict(block + "diff(f, x, 2).subs(x, 0) = -1") == "passed"
        assert verdict(block + "diff(f, x, 3).subs(x, 0) = 1") == "failed"

    def test_a_definition_that_uses_its_own_name_doesnt_loop(self):
        assert verdict("let a = 2\nb = a + 1\nb = 3") == "passed"


class TestStepFourGaps:
    """Gaps logged during the step 4 runs (checker frozen), fixed after grading."""

    def test_g1_a_relation_between_undefined_quantities_is_undecided_not_failed(self):
        found = claims("assume positive: sin2thetaW, cV, cA\ncV_l/cA_l = 1-4*sin2thetaW")[-1]
        assert found["verdict"] == "undecided" and "share no symbol" in found["note"]

    def test_g1_an_identity_with_shared_symbols_still_fails(self):
        assert verdict("assume positive: x\nsin(x)**2 = 1 - cos(x)") == "failed"
        assert verdict("diff(log(cosh(r)), r) = 1/cosh(r)") == "failed"

    def test_g2_log10_and_log2(self):
        assert verdict("10*log10(1/(1 - 0.881**2)) ≈ 6.50") == "passed"
        assert verdict("10*log10(1/(1 - 0.881**2)) ≈ 6.49") == "failed"          # 6.5006
        assert verdict("log2(8) = 3") == "passed"

    def test_g3_python_sum_and_prod_over_a_range(self):
        assert verdict("assume positive: n\nassume integer: n\nsum(k, (k, 1, n)) = n*(n + 1)/2") == "passed"
        assert verdict("prod(1 - b, (s, 1, 3)) = (1 - b)**3") == "passed"

    def test_g4_count_words_as_units(self):
        assert verdict("let L = 128 [samples]\nlet H = 32 [samples]\nL/H = 4") == "passed"
