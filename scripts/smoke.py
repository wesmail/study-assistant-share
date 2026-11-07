"""A free check that an install works: scripts/smoke.sh runs it.

No keys and no hosted model are used, and nothing of the reader's is touched: it works in
a scratch folder with its own config (the reader's config, with only the library, data
and outputs folders moved there), on a small PDF and a two-slide deck it writes itself.

    ok    a step that worked
    warn  something optional that doesn't work (web search)
    FAIL  something that does, with the fix

It checks, in order: Docker and the two images, Ollama and the local models, ingesting
both files with the real worker (Docling, the office container for the deck, embeddings),
one Fast answer that must find a fact on page 2 of the PDF and cite that page, Python and
SymPy in the sandbox, and SearXNG (a warning only). Hugging Face is kept offline, so a
missing parser model fails here instead of starting a download.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

INGEST_SECONDS = 900        # parsing loads Docling's models; the deck is rendered in its container
FACT = "The Zorblax constant of the scratch handbook is 42.7 kelvin."
QUESTION = "What is the value of the Zorblax constant in the scratch handbook?"
SLIDE_FACT = "The Quilmore ratio is three to five."

failures: list[str] = []


def ok(text: str) -> None:
    print(f"ok    {text}", flush=True)


def warn(text: str) -> None:
    print(f"warn  {text}", flush=True)


def fail(text: str, fix: str) -> None:
    print(f"FAIL  {text}\n      Fix: {fix}", flush=True)
    failures.append(text)


def stop() -> None:
    print(f"\nThe smoke test failed: {len(failures)} problem{'s' if len(failures) != 1 else ''} above.", flush=True)
    sys.exit(1)


def scratch_config(scratch: Path) -> Path:
    """The reader's config with only the folders moved: models, images and URLs stay theirs."""
    from app.config import CONFIG_ENV

    source = Path(os.environ.get(CONFIG_ENV) or ROOT / "config.yaml")
    values = yaml.safe_load(source.read_text(encoding="utf-8")) if source.is_file() else {}
    values = values or {}
    values.update(library_dir=str(scratch / "library"), data_dir=str(scratch / "data"),
                  outputs_dir=str(scratch / "outputs"))
    target = scratch / "config.yaml"
    target.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.environ[CONFIG_ENV] = str(target)
    return target


def write_pdf(path: Path) -> None:
    import fitz

    pages = [
        ["Scratch handbook", "Chapter 1. Introduction", "This handbook exists only to test the study assistant.",
         "It describes nothing real, so any answer must come from these pages."],
        ["Chapter 2. Constants", FACT, "It was measured on a bench that does not exist.",
         "The constant is used nowhere else in the handbook."],
        ["Chapter 3. Closing remarks", "The handbook ends here. Nothing on this page is a constant."],
    ]
    pdf = fitz.open()
    for lines in pages:
        page = pdf.new_page()
        for number, line in enumerate(lines):
            page.insert_text((72, 90 + 26 * number), line, fontsize=16 if number == 0 else 11)
    pdf.set_metadata({"title": "Scratch handbook", "author": "Smoke test"})
    pdf.save(path)


def write_deck(path: Path) -> None:
    from pptx import Presentation

    deck = Presentation()
    for title, body in (("Scratch talk", "A deck made by the smoke test."), ("Ratios", SLIDE_FACT)):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    deck.core_properties.title = "Scratch talk"
    deck.save(path)


def check_services(settings) -> None:
    import httpx

    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=30)
        ok("Docker is running")
    except (OSError, subprocess.SubprocessError):
        fail("Docker isn't running", "sudo systemctl start docker   (INSTALL.md, Docker)")
        stop()
    for image, script in ((settings.sandbox_image, "scripts/build_sandbox.sh"),
                          (settings.office_image, "scripts/build_office.sh")):
        found = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
        if found:
            ok(f"the image {image} is there")
        else:
            fail(f"there is no {image} image", f"{script}   (INSTALL.md, the images)")
    try:
        tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        fail(f"Ollama isn't answering at {settings.ollama_url}", "sudo systemctl start ollama")
        stop()
    have = {model["name"] for model in tags.get("models", [])}
    have |= {name.removesuffix(":latest") for name in have}
    fast = settings.answer_options[0]
    for model, why in ((fast.model, f"{fast.label} answers"), (settings.embed_model, "search")):
        if model in have:
            ok(f"Ollama has {model} ({why})")
        else:
            fail(f"Ollama has no {model}, which {why} need", f"ollama pull {model}")
    if settings.helper_model not in have:
        warn(f"Ollama has no {settings.helper_model}: follow-up questions aren't rewritten "
             f"(ollama pull {settings.helper_model})")
    if failures:
        stop()


def ingest(settings) -> dict[str, dict]:
    from app.db.connection import connect
    from app.ingest.worker import Worker, scan_library

    conn = connect(settings.db_path)
    try:
        scan_library(conn, settings)
        worker = Worker(conn, settings)
        started = time.monotonic()
        while time.monotonic() - started < INGEST_SECONDS:
            docs = conn.execute("SELECT * FROM documents").fetchall()
            if docs and all(doc["status"] in ("indexed", "failed") for doc in docs):
                break       # searchable: summaries and figure descriptions aren't part of the smoke test
            if not worker.run_one():
                time.sleep(1)
        rows = {}
        for doc in conn.execute("SELECT * FROM documents").fetchall():
            failed = conn.execute("SELECT stage, error FROM jobs WHERE doc_id = ? AND status = 'failed' "
                                  "OR (doc_id = ? AND error IS NOT NULL) ORDER BY stage_order LIMIT 1",
                                  (doc["id"], doc["id"])).fetchone()
            chunks = [row["text"] for row in conn.execute("SELECT text FROM chunks WHERE doc_id = ?", (doc["id"],))]
            rows[doc["path"]] = {"id": doc["id"], "status": doc["status"], "pages": doc["pages"], "chunks": chunks,
                                 "error": (failed["stage"], failed["error"]) if failed else None}
        return rows
    finally:
        conn.close()


def report_ingest(rows: dict[str, dict], seconds: float) -> None:
    for name, kind in (("handbook.pdf", "PDF"), ("talk.pptx", "deck")):
        row = rows.get(name)
        if row and row["status"] == "indexed":
            ok(f"the {kind} was ingested: {row['pages']} pages, {len(row['chunks'])} chunks ({seconds:.0f} s for both)")
            continue
        stage, error = (row or {}).get("error") or ("?", "it never finished")
        last = (error or "").strip().splitlines()[-1:] or [""]
        fix = ("the parser's models aren't in the Hugging Face cache: ingest once with the network on "
               "(INSTALL.md, first run)" if "offline" in (error or "").lower() or "LocalEntryNotFound" in (error or "")
               else "scripts/build_office.sh, then run this again" if kind == "deck" and stage == "parse"
               else "TROUBLESHOOTING.md, ingestion")
        fail(f"the {kind} wasn't ingested (stage {stage}: {last[0][:200]})", fix)
    deck = rows.get("talk.pptx")
    if deck and deck["status"] == "indexed":
        if deck["pages"] == 2 and any("Quilmore" in text for text in deck["chunks"]):
            ok("the deck's render has its 2 slides, and slide 2's text is searchable")
        else:
            fail(f"the deck's render is wrong ({deck['pages']} pages for 2 slides)", "TROUBLESHOOTING.md, decks")


def ask_fast(settings, doc_id: int) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app(settings), base_url="http://localhost")
    for attempt in (1, 2):      # gpt-oss leaves out a citation about 1 answer in 10
        started = time.monotonic()
        response = client.post("/api/chat", json={"message": QUESTION, "mode": settings.answer_options[0].id})
        events = []
        for block in response.text.strip().split("\n\n"):
            name, _, data = block.partition("\n")
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: ") or "null")))
        text = "".join(data["text"] for name, data in events if name == "token")
        done = next((data for name, data in events if name == "done"), None)
        error = next((data for name, data in events if name == "error"), None)
        sources = [source for name, data in events if name in ("sources", "more_sources") for source in data["sources"]]
        if error or done is None:
            fail(f"Fast didn't answer: {(error or {}).get('message', 'no answer')}",
                 "TROUBLESHOOTING.md, answers (is the GPU busy, or the model missing?)")
            return
        cited = set(done.get("citations", []))
        pages = {(source["doc_id"], source.get("focus_page") or source["page_start"])
                 for source in sources if source.get("id") in cited}
        right = "42.7" in (done.get("answer") or text)
        on_page = (doc_id, 2) in pages
        if right and on_page:
            ok(f"Fast answered from page 2 and cited it ({time.monotonic() - started:.0f} s)")
            return
        if attempt == 2 or not right:
            problem = "didn't give 42.7" if not right else f"cited {sorted(pages) or 'nothing'}, not page 2"
            fail(f"Fast's answer {problem}", "TROUBLESHOOTING.md, answers")
            return
        warn("Fast's answer cited nothing this time; asking once more")


def run_sandbox(settings) -> None:
    from app.tools.executor import ToolBox

    box = ToolBox(settings, lambda query: [], {}, [])
    code = asyncio.run(box.call("run_python", {"code": "import numpy as np\nprint(int(np.arange(7).sum() * 2))"}))
    if code.ok and "42" in code.text:
        ok("Python ran in the sandbox")
    else:
        fail(f"Python didn't run in the sandbox: {code.summary}", "scripts/build_sandbox.sh, or TROUBLESHOOTING.md, sandbox")
    check = asyncio.run(box.call("sympy_check", {"lhs": "diff(sin(x)**2, x)", "rhs": "sin(2*x)"}))
    if check.ok and "true" in check.text.lower():
        ok("SymPy checked an identity in the sandbox")
    else:
        fail(f"SymPy didn't check the identity: {check.summary}", "scripts/build_sandbox.sh")


def check_searxng(settings) -> None:
    import httpx

    try:
        response = httpx.get(f"{settings.searxng_url}/search", params={"q": "smoke test", "format": "json"},
                             headers={"X-Real-IP": "127.0.0.1"}, timeout=15)
        response.raise_for_status()
        response.json()
        ok("SearXNG answers (web search can work)")
    except (httpx.HTTPError, ValueError):
        warn(f"SearXNG isn't answering at {settings.searxng_url}: web search won't work (scripts/searxng.sh start)")


def quiet() -> None:
    """Only the smoke test's own lines: the worker's and the parser's logs go to warnings and up."""
    import logging

    logging.basicConfig(level=logging.WARNING, force=True)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass


def main() -> None:
    quiet()
    scratch = Path(sys.argv[1])
    scratch_config(scratch)
    from app.config import load_settings

    settings = load_settings()
    assert str(settings.data_dir).startswith(str(scratch)), "the smoke test must never use the real data"
    print(f"Smoke test in {scratch} (nothing of yours is touched)", flush=True)
    check_services(settings)
    settings.library_dir.mkdir(parents=True)
    write_pdf(settings.library_dir / "handbook.pdf")
    write_deck(settings.library_dir / "talk.pptx")
    started = time.monotonic()
    rows = ingest(settings)
    report_ingest(rows, time.monotonic() - started)
    if rows.get("handbook.pdf", {}).get("status") == "indexed":
        ask_fast(settings, rows["handbook.pdf"]["id"])
    run_sandbox(settings)
    check_searxng(settings)
    if failures:
        stop()
    print("\nThe smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
