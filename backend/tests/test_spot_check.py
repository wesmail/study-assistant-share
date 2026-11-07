"""The gate's arithmetic. A bug here passes a book that should fail."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "spot_check", Path(__file__).resolve().parents[2] / "eval" / "spot_check.py")
spot_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot_check)


def sample(compiles=True, flagged=False):
    return {"compile_ok": compiles, "unverified": flagged}


@pytest.mark.parametrize("item, verdict, right, shown, silent", [
    (sample(), {"verdict": "correct"}, True, True, False),
    (sample(), {"verdict": "wrong"}, False, False, True),                       # the dangerous case
    (sample(flagged=True), {"verdict": "wrong", "crop": "ok"}, False, True, False),
    (sample(flagged=True), {"verdict": "wrong", "crop": "bad"}, False, False, False),
    (sample(flagged=True), {"verdict": "correct"}, True, True, False),          # a false alarm costs nothing
    (sample(flagged=True), {"verdict": "correct", "crop": "bad"}, True, True, False),
    (sample(compiles=False, flagged=True), {"verdict": "correct", "crop": "ok"}, False, True, False),
    (sample(compiles=False), {"verdict": "correct"}, False, False, True),       # dropped as junk
])
def test_judge(item, verdict, right, shown, silent):
    assert spot_check.judge(item, verdict) == (right, shown, silent)


def test_a_crop_verdict_cannot_rescue_an_unflagged_formula():
    assert spot_check.judge(sample(), {"verdict": "wrong", "crop": "ok"}) == (False, False, True)


def test_allocation_without_floors_is_proportional():
    assert spot_check.allocate(10, {1: 70, 2: 30}) == {1: 7, 2: 3}


def test_a_floor_lifts_a_small_document_and_the_rest_share_what_remains():
    shares = spot_check.allocate(50, {1: 800, 2: 360, 3: 100}, {3: 15})
    assert shares[3] == 15
    assert sum(shares.values()) == 50
    assert shares[1] > shares[2]


def test_a_floor_already_met_changes_nothing():
    assert spot_check.allocate(50, {1: 50, 2: 50}, {2: 15}) == spot_check.allocate(50, {1: 50, 2: 50})


def test_a_floor_cannot_exceed_the_document():
    assert spot_check.allocate(50, {1: 800, 2: 6}, {2: 15}) == {1: 44, 2: 6}


def test_a_floor_for_a_missing_document_stops_the_draw():
    rows = [{"doc_id": 1, "path": "Mathematics.pdf"}]
    with pytest.raises(SystemExit, match="Tong"):
        spot_check.floors_by_document(rows, {"Tong": 15})


def test_a_floor_finds_its_document_by_file_name():
    rows = [{"doc_id": 1, "path": "a.pdf"}, {"doc_id": 4, "path": "David Tong - Dynamics.pdf"}]
    assert spot_check.floors_by_document(rows, {"tong": 15}) == {4: 15}
