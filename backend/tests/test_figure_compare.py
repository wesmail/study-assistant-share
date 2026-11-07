"""The figure comparison's blinding: no describer is named anywhere a judge looks."""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
spec = importlib.util.spec_from_file_location("figure_compare", EVAL / "figure_compare.py")
figure_compare = importlib.util.module_from_spec(spec)
sys.modules["figure_compare"] = figure_compare
spec.loader.exec_module(figure_compare)

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
NAMES = ("gemma-off", "gemma-on", "opus", "gemma", "Gemma", "Opus", "claude", "thinking")


@pytest.fixture
def pack(tmp_path, monkeypatch):
    image = tmp_path / "f.png"
    image.write_bytes(PNG)
    monkeypatch.setattr(figure_compare, "ROOT", tmp_path)
    monkeypatch.setattr(figure_compare, "RESULTS", tmp_path / "results.jsonl")
    monkeypatch.setattr(figure_compare, "PACK", tmp_path / "pack")
    records = [{"figure": figure, "describer": describer, "document": "Book", "page": 3, "caption": "Figure 1.1 A curve.",
                "page_text": "Text.", "image": "f.png", "text": f"Description of {figure} number {i}."}
               for figure in figure_compare.FIGURES for i, describer in enumerate(figure_compare.DESCRIBERS)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    figure_compare.cmd_pack(None)
    return tmp_path / "pack"


def test_every_figure_gets_all_three_describers_under_a_letter(pack):
    labels = json.loads((pack / "key.json").read_text())["labels"]
    assert set(labels) == {str(figure) for figure in figure_compare.FIGURES}
    for order in labels.values():
        assert sorted(order) == ["A", "B", "C"] and sorted(order.values()) == sorted(figure_compare.FIRST_PACK)


def test_the_order_is_shuffled_per_figure_and_repeatable(pack, tmp_path):
    labels = json.loads((pack / "key.json").read_text())["labels"]
    assert len({tuple(order.values()) for order in labels.values()}) > 1
    first = (pack / "key.json").read_text()
    (tmp_path / "pack").rename(tmp_path / "old")
    figure_compare.cmd_pack(None)
    assert (pack / "key.json").read_text() == first


@pytest.mark.parametrize("name", ["review.html", "batch-1.txt", "batch-2.txt"])
def test_no_describer_is_named_where_a_judge_looks(pack, name):
    text = (pack / name).read_text(encoding="utf-8")
    assert not [word for word in NAMES if word in text]


def test_the_batches_split_the_figures_and_each_names_its_own_output(pack):
    one, two = ((pack / f"batch-{n}.txt").read_text() for n in (1, 2))
    assert one.count("FIGURE ") + two.count("FIGURE ") == len(figure_compare.FIGURES)
    assert str(pack / "grades-1.json") in one and str(pack / "grades-2.json") in two


def test_a_second_pack_is_refused(pack):
    with pytest.raises(SystemExit, match="exists"):
        figure_compare.cmd_pack(None)


def test_an_incomplete_run_is_not_packed(tmp_path, monkeypatch):
    monkeypatch.setattr(figure_compare, "RESULTS", tmp_path / "results.jsonl")
    (tmp_path / "results.jsonl").write_text(json.dumps({"figure": 1, "describer": "opus"}) + "\n")
    with pytest.raises(SystemExit, match="Not every figure"):
        figure_compare.cmd_pack(None)


def test_the_prompt_forbids_values_that_cant_be_read():
    assert "Never estimate a value you can't place" in figure_compare.PROMPT
    assert "never take one from your own knowledge" in figure_compare.PROMPT


@pytest.mark.parametrize("text", ["As Claude, I see a curve.", "Gemma here: the plot", "<think>hmm</think> A curve."])
def test_a_description_that_would_unblind_itself_stops_the_pack(tmp_path, monkeypatch, text):
    monkeypatch.setattr(figure_compare, "RESULTS", tmp_path / "results.jsonl")
    monkeypatch.setattr(figure_compare, "PACK", tmp_path / "pack")
    records = [{"figure": figure, "describer": describer, "text": "A curve."}
               for figure in figure_compare.FIGURES for describer in figure_compare.DESCRIBERS]
    # One of the default pack's own describers, which is what gets checked.
    next(r for r in records if r["describer"] == figure_compare.FIRST_PACK[1])["text"] = text
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(SystemExit, match="unblind"):
        figure_compare.cmd_pack(None)


def test_a_second_pack_compares_only_the_describers_it_names(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(figure_compare, "ROOT", tmp_path)
    monkeypatch.setattr(figure_compare, "GRADING", tmp_path)
    monkeypatch.setattr(figure_compare, "RESULTS", tmp_path / "results.jsonl")
    (tmp_path / "f.png").write_bytes(PNG)
    records = [{"figure": figure, "describer": describer, "document": "Book", "page": 3, "caption": "Figure 1.1",
                "page_text": "Text.", "image": "f.png", "text": f"Description {figure} {describer[-1]}."}
               for figure in figure_compare.FIGURES for describer in figure_compare.DESCRIBERS]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    figure_compare.cmd_pack(SimpleNamespace(name="second", describers=["qwen-vl", "opus"], seed=7))
    key = json.loads((tmp_path / "second" / "key.json").read_text())
    assert key["describers"] == ["qwen-vl", "opus"]
    assert all(sorted(order) == ["A", "B"] and sorted(order.values()) == ["opus", "qwen-vl"] for order in key["labels"].values())
    page = (tmp_path / "second" / "review.html").read_text()
    assert 'value="good"' in page and 'name="f1C"' not in page
    brief = (tmp_path / "second" / "batch-1.txt").read_text()
    assert "two unnamed describers" in brief and '"B": {...}' in brief and "{{" not in brief.split("################")[0]
    assert not [word for word in NAMES + ("qwen", "Qwen") if word in page + brief]
