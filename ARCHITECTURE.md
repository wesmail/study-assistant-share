# Architecture

How the study assistant is put together, and where its trust boundaries are. How to install it is in [INSTALL.md](INSTALL.md), and what to do when something breaks in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## The parts

```text
 browser (React, three panes)  ──HTTP + SSE, localhost only──▶  FastAPI server (backend/app/main.py)
                                                                   │
            ┌──────────────────────────────┬───────────────────────┼─────────────────────────┐
            ▼                              ▼                       ▼                         ▼
   ingestion worker               chat pipeline            sandbox containers          SQLite + files
   (child process)                (app/chat.py)            (Docker, no network)        data/app.sqlite
   library/ → stages              search → model → SSE     code, SymPy, files,         data/parsed/, data/renders/
                                                           LibreOffice                 outputs/
            │                              │
            ▼                              ▼
     Ollama (GPU): Fast, Balanced, the summarizer, the helper; bge-m3 on the CPU
     Hugging Face (GPU): Docling, Granite-Docling, the reranker
     Hosted, only when chosen: Anthropic (Deep, figure descriptions), OpenRouter (Qwen)
     SearXNG (Docker, 127.0.0.1:8888): web search, only for a question with Web on
```

- **Server** (`backend/app/main.py`): the HTTP API, bound to 127.0.0.1. `scripts/start.sh` starts it, and it starts the ingestion worker as a child process (`app/supervisor.py`). Only `localhost` and `127.0.0.1` are accepted as hosts, and a write coming from another web site is refused.
- **Frontend** (`frontend/src`): React with a zustand store. The panes are the library (`panes/LibraryPane.tsx`), the chat with Feedback and Notes views (`panes/ChatPane.tsx`, `FeedbackPane.tsx`, `NotesPane.tsx`), and the viewer (`panes/ViewerPane.tsx`). Markdown, KaTeX and Shiki are in `render/`; pdf.js highlights, Explain and Suggest are in `viewer/`. The server serves the built files from `frontend/dist`.
- **Storage:** `data/app.sqlite` (schema in `backend/app/db/schema.sql`) holds documents, blocks, chunks with their FTS5 and sqlite-vec indexes, figures, summaries, notes, highlights, conversations, feedback, the hosted-call log and the job queue. `data/parsed/<sha256>/` holds what Docling made of each file version, and `data/renders/<sha256>.pdf` each deck's LibreOffice render. The files the tools make go to `outputs/`. The app writes to `library/` only to add a file from the library pane's Add button (`app/upload.py`): it streams into a hidden temporary file, is checked (name, size, a PDF's or deck's own format, a duplicate), and takes its name through a hard link that fails if the name is taken. A file already there is never changed or replaced.

## Ingestion

The worker (`app/ingest/worker.py`) scans `library/` at start, then a watcher (`watcher.py`) picks up new or changed files. Each file version (by SHA-256) goes through resumable stages, one job per stage in the `jobs` table:

1. **parse:** Docling reads the PDF (`docling_structured.py`) into blocks with page positions. A deck is first rendered to PDF in the office container (`app/office.py`, `office/render.py`), and its slide text and speaker notes are read there too (`ingest/deck.py`).
2. **formulas:** Granite-Docling reads each display formula from its crop, and the PDF's own glyphs check the result (`formula_merge.py`, `formula_layer.py`, `pdf_fonts.py`). A formula that doesn't pass is flagged and shown from the page.
3. **chunk:** structure-aware chunks of 300–600 tokens, with heading paths and one sentence of overlap (`chunker.py`, `outline.py`).
4. **index:** keyword (FTS5) and vector (bge-m3, `embed.py`) indexes. After this stage the document is searchable.
5. **describe:** figure descriptions for search, made by a hosted model only after the reader agrees to the estimated cost in the library pane (`figures.py`). They're cached by image hash.
6. **summarize:** the summary tree of sections, chapters and the whole document, written by a local model (`summarize.py`).

Titles never come from file names (`metadata.py`). `library/metadata.yaml` holds the reader's own titles and `exclude_pages`. Pages listed there are kept out of everything: no block, chunk, figure or summary of them exists, even when they were added after ingestion (`deck.apply_exclusions`). Explain and Suggest refuse to send them to a hosted model. If the file can't be read, the app doesn't treat it as empty: ingestion waits and hosted Explain is refused until it's fixed.

## A question

`app/chat.py` runs one answer, and the evaluation uses the same code:

1. A follow-up is rewritten into a standalone question for the search only, by a small local helper model (`retrieval/rewrite.py`).
2. Hybrid search (`retrieval/search.py`): FTS5 keywords and bge-m3 vectors, merged by reciprocal rank fusion, then reranked by a cross-encoder (`rerank.py`). A few slots are kept for figure descriptions, summaries and the reader's notes.
3. The hits are widened by their neighbouring chunks and joined (`retrieval/context.py`), then numbered as sources in the prompt (`llm/prompts.py`). Flagged formulas reach the model as symbols, not as LaTeX that may be wrong.
4. The chosen option's model answers, through `llm/ollama.py`, `llm/hosted.py` (Anthropic) or `llm/openrouter.py`. Every hosted call is logged with tokens and cost in `hosted_calls`.
5. As the text streams, citations to sources the model wasn't given are removed (`llm/citations.py`). The answer then reaches the browser as Server-Sent Events.
6. With tools on, the model can search again, run Python, check an identity with SymPy, make Markdown, PDF, Word or PowerPoint files, and propose highlights (`tools/`). With Web on, it can search SearXNG and read pages (`tools/web.py`). With Check on, the claims it states are checked in the sandbox (`verify.py`).

## Trust boundaries

**What leaves the machine.** Nothing does unless the reader chooses a hosted option for a question, or agrees to figure descriptions. A hosted option sends:
- the question and the passages found for it;
- the previous question and answer, with the passages that answer cited;
- what tools return.

Older turns, the library's files, notes that weren't found for the question, and kept-out pages are never sent. Figure descriptions send each figure's image and its page's text, after the reader's OK.

**Keys.** They are read only from `.env` (`STUDY_ASSISTANT_ANTHROPIC_KEY`, `STUDY_ASSISTANT_OPENROUTER_KEY`), never from the process environment. They aren't printed, logged, stored in the database, or passed to a container. Backups keep `.env` only under `~/backups`, never in the archives on D:.

**Model-written code and documents** run in a container (`tools/sandbox.py`) with:
- no network and a read-only root;
- one temporary work folder, with a cap on its total size;
- no capabilities, as the reader's own uid;
- CPU, memory, process and time limits.

Files come out of it through a folder that is opened without following links, and only regular files are copied. SymPy reads a claim only as maths: no strings, no builtins, no printers. The report the host trusts carries a one-time nonce that is deleted before any claim is read. Word and PowerPoint files are rendered in the office container, behind the same walls.

**Web pages** are data, never instructions:
- Only links from the question's search results, or ones the reader typed, are opened.
- Every address must be public. It is checked when the connection is made, so a name that resolves to this machine at the last moment is refused.
- Hidden text is dropped, and the page reaches the model between tagged markers.
- Code or a SymPy check copied from a page is refused.
- Web search is off unless the reader turns it on for a question, and only Deep and Qwen offer it.

**Not sandboxed, by choice:** the library's PDFs, and the PDFs LibreOffice makes from decks, are parsed on the host by PyMuPDF and Docling, which are native code. They're files the reader chose to add, by copying them in or with the Add button, which changes how a file arrives, not how it is read. Web searches aren't checked for copied library text: Web is opt-in per question, and checking every search against the library would refuse ordinary ones.

## Models

| Role | Model | Where |
|---|---|---|
| Fast (default) | gpt-oss:20b | Ollama, GPU |
| Balanced | gemma4:26b-a4b-it-qat | Ollama, GPU |
| Qwen | qwen/qwen3.8-27b, DeepInfra only | OpenRouter |
| Deep | claude-opus-5-5, Sonnet 5.5 if it fails | Anthropic |
| Embeddings | bge-m3 | Ollama, pinned to the CPU |
| Follow-up rewrite | qwen3:1.7b | Ollama |
| Summary tree | qwen3.8:27b | Ollama, GPU |
| Reranker | BAAI/bge-reranker-v2-m3 | Hugging Face, GPU |
| Parsing, formulas | Docling (layout, CodeFormula), Granite-Docling 258M | Hugging Face, GPU |
| Figure descriptions | claude-sonnet-5-5, after the reader's OK | Anthropic |

Only one local chat model fits in 16 GB. Switching between Fast and Balanced reloads it, and the selector says when that will happen. While a chat request runs, the worker pauses its GPU work. All of this is configured in `config.yaml`, over the defaults in `backend/app/config.py`.

## Where to look

| Folder | What is there |
|---|---|
| `backend/app/` | the server, chat, Explain, Suggest, notes, highlights, feedback, backups |
| `backend/app/ingest/` | the worker and its stages |
| `backend/app/retrieval/` | search, reranking, context |
| `backend/app/llm/` | model clients, prompts, citations |
| `backend/app/tools/` | tool calls, the sandbox, web search |
| `backend/tests/` | backend tests (`test_sandbox_docker.py` runs the real containers) |
| `frontend/src/` | the React app; `frontend/e2e/` drives the running app in a browser |
| `sandbox/`, `office/` | the two container images, each built from its own folder alone |
| `eval/` | gold questions, evaluation and grading scripts, results |
| `scripts/` | start, smoke test, image builds, SearXNG, backups |
