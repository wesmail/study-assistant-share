"""The tool test set itself: a mistake in it would quietly change what the gate measures."""
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

EVAL = Path(__file__).resolve().parents[2] / "eval"
spec = importlib.util.spec_from_file_location("run_tools", EVAL / "run_tools.py")
run_tools = importlib.util.module_from_spec(spec)
sys.modules["run_tools"] = run_tools        # pydantic resolves annotations through this
spec.loader.exec_module(run_tools)

FILE = run_tools.load_tasks()
BY_ID = {task.id: task for task in FILE.tasks}
ANSWER = FILE.fixtures["cholesky_answer"].answer
RUNS_WITHOUT_LIMITS = ["run-numeric", "run-fix"]


def test_there_are_23_gate_tasks_and_the_security_ones_on_top_with_unique_ids():
    """22 until Nov 7, 2026, when pdf-paper-figure came from a bug report; earlier results are of 22."""
    assert len(run_tools.gate_tasks(FILE.tasks)) == 23
    assert len(BY_ID) == len(FILE.tasks)


def test_every_security_task_must_refuse_and_leak_nothing_and_can_fail():
    security = [task for task in FILE.tasks if task.kind == "security"]
    assert len(security) >= 8
    for task in security:
        assert task.direct, task.id
        assert task.check("secret_absent") is True, task.id
        assert task.check("no_new_file") is True, task.id
        assert task.check("tool_refused") or task.check("files_refused"), task.id


def test_the_attacks_you_asked_for_are_all_there():
    attacks = {task.id: task.direct for task in FILE.tasks if task.direct}
    assert "\\input{" + str(Path(__file__).resolve().parents[2] / ".env") + "}" in attacks["sec-tex-input"].content
    assert attacks["sec-image-pdf"].tool == "make_pdf" and attacks["sec-image-docx"].tool == "make_docx"
    assert attacks["sec-image-pptx"].tool == "make_pptx"
    for kind in ("pdf", "docx", "pptx"):
        assert BY_ID[f"sec-image-{kind}-raw"].raw is True


def test_every_tool_is_tested_and_annotate_pdf_stays_in_phase_6():
    used = {tool for task in run_tools.gate_tasks(FILE.tasks) for tool in task.tools}
    assert used == run_tools.TOOLS
    assert "annotate_pdf" not in used


@pytest.mark.parametrize("task", FILE.tasks, ids=lambda task: task.id)
def test_every_name_in_a_task_exists(task):
    assert set(task.scope) <= set(FILE.docs)
    assert task.after is None or task.after in FILE.fixtures
    for source in task.check("sources_named") or []:
        assert source["doc"] in FILE.docs
    assert task.check("formulas_preserved") in (None, *FILE.fixtures)


@pytest.mark.parametrize("task", FILE.tasks, ids=lambda task: task.id)
def test_a_task_that_makes_a_file_says_which_kind(task):
    makes_file = any(tool.startswith("make_") for tool in task.tools)
    assert makes_file <= (task.check("file") is not None)


def test_tasks_that_call_a_tool_directly_ask_no_model():
    assert all(not task.tools for task in FILE.tasks if task.direct)


def test_the_holdout_has_one_task_of_each_main_kind():
    held_out = Counter(task.kind for task in FILE.tasks if task.split == "holdout")
    assert held_out == {"pptx": 1, "pdf": 1, "docx": 1, "run": 1, "sympy": 1}


def test_an_unknown_check_is_refused():
    with pytest.raises(ValidationError, match="unknown check"):
        run_tools.Task(id="x", split="dev", kind="pdf", prompt="p", tools=[], checks=[{"pdf_page": 1}])


def test_a_tool_outside_this_phase_is_refused():
    with pytest.raises(ValidationError, match="unknown tools"):
        run_tools.Task(id="x", split="dev", kind="pdf", prompt="p", tools=["annotate_pdf"], checks=[])


def test_the_word_export_expects_every_formula_in_the_fixture():
    expected = len(run_tools.display_formulas(ANSWER)) + len(run_tools.inline_formulas(ANSWER))
    assert (len(run_tools.display_formulas(ANSWER)), expected) == (4, 9)
    assert BY_ID["docx-export"].check("docx_equations") == expected


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node on the path")
def test_the_fixture_formulas_compile_in_katex():
    formulas = run_tools.display_formulas(ANSWER)
    done = subprocess.run(["node", str(EVAL / "katex_check.mjs")], input=json.dumps(formulas),
                          capture_output=True, text=True, check=True)
    assert all(result["ok"] for result in json.loads(done.stdout))


@pytest.mark.parametrize("task_id", RUNS_WITHOUT_LIMITS)
def test_the_code_in_a_task_does_what_its_checks_expect(task_id):
    """Run here for now, because these two need only numpy."""
    task = BY_ID[task_id]
    code = task.prompt.split("```python\n")[1].split("```")[0]
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if task.check("failed_then_fixed"):
        assert done.returncode != 0
        fixed = code.replace("np.array([[1., 2.]])", "np.array([1., 2.])")
        done = subprocess.run([sys.executable, "-c", fixed], capture_output=True, text=True)
    assert done.returncode == 0
    for text in task.check("stdout_contains"):
        assert text in done.stdout
