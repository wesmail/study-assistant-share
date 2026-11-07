"""The HTTP API: chat over Server-Sent Events, search, and the library's files.

    python -m app.main              the server, plus the ingestion worker as a child process
    python -m app.main --no-worker  the server only

The server has no login, so it binds to the loopback address in code rather than
in config, and it refuses Host headers other than localhost, which stops a web
page from reaching it through DNS rebinding.

From Windows, use a browser, or 127.0.0.1 with other tools. With WSL's mirrored
networking ::1 hangs, even with a listener on it, and Windows tries ::1 first for
"localhost". Edge falls back to IPv4 by itself in about 300 ms (measured); PowerShell's
Invoke-WebRequest doesn't.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import re
import threading
import uuid
from dataclasses import replace
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.chat import answer_events, exportable_answer, last_citations, last_turn, recent_history
from app.config import ROOT, AnswerOption, Settings, load_settings
from app.db.connection import connect
from app import anchors, annotated, annotations, errors, explain, feedback, library, notes, office, removed, suggest, upload
from app.ingest import deck, figures, metadata
from app.ingest.embed import embed_texts
from app.llm.hosted import MILLION, month_spend, turn_cost
from app.retrieval.rerank import cross_encoder
from app.retrieval.rewrite import warm_up_helper
from app.retrieval.search import hybrid_search
from app.supervisor import WorkerSupervisor
from app.tools.web import WebSession

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def cross_site(headers) -> bool:
    """True when a browser says the request comes from a page that isn't this app."""
    if headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return True
    origin = headers.get("origin")
    return origin is not None and origin != f"http://{headers.get('host', '')}"
CROP_NAME = re.compile(r"\d+\.png")
FIGURE_NAME = re.compile(r"figure_\d+_page\d+\.png")
OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")
OUTPUT_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}
FRONTEND = ROOT / "frontend" / "dist"


READER_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
READER_IMAGE_BYTES = 5 * 1024 * 1024        # Anthropic's limit per image
READER_IMAGES_MAX = 3


class ReaderImage(BaseModel):
    """A screenshot or picture the reader pasted or uploaded, for this question only."""
    media_type: str
    data: str = Field(min_length=1)             # base64


class RemovedEntry(BaseModel):
    entry: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    images: list[ReaderImage] = Field(default_factory=list, max_length=READER_IMAGES_MAX)
    conversation_id: int | None = None
    doc_ids: list[int] | None = None        # None searches the whole library
    mode: str | None = None                 # an answer option's id; None is the first, the default
    web: bool = False                       # the reader turned web search on for this question
    web_query: str | None = Field(default=None, max_length=300)   # a proposed search the reader agreed to
    check: bool = False                     # the reader turned Check on: claims checked in the sandbox
    notes_only: bool = False                # the Notes scope: search the reader's notes, not the books


def checked_images(request: ChatRequest, option: AnswerOption) -> list[dict]:
    """The reader's images, refused unless the option reads images and each one is a real, small enough image."""
    if not request.images:
        return []
    if not option.vision:
        raise HTTPException(422, f"{option.label} can't read images. Ask with an option that can.")
    images = []
    for image in request.images:
        if image.media_type not in READER_IMAGE_TYPES:
            raise HTTPException(422, f"Images must be PNG, JPEG, WebP or GIF, not {image.media_type}")
        try:
            raw = base64.b64decode(image.data, validate=True)
        except ValueError:
            raise HTTPException(422, "An image wasn't valid base64") from None
        if len(raw) > READER_IMAGE_BYTES:
            raise HTTPException(422, f"An image is {len(raw) / 1e6:.1f} MB; the limit is 5 MB")
        images.append({"media_type": image.media_type, "data": image.data})
    return images


def chosen_option(settings: Settings, request: ChatRequest) -> AnswerOption:
    if request.mode is None:
        return settings.answer_options[0]
    try:
        return settings.option(request.mode)
    except KeyError:
        known = ", ".join(option.id for option in settings.answer_options)
        raise HTTPException(422, f"No answer option {request.mode!r}; the options are {known}") from None


async def loaded_models(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> set[str] | None:
    """What Ollama has in memory now, or None if it can't say."""
    try:
        async with httpx.AsyncClient(timeout=3, transport=transport) as client:
            response = await client.get(f"{settings.ollama_url}/api/ps")
            response.raise_for_status()
            return {model["name"] for model in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


def expected_usd(option: AnswerOption, settings: Settings) -> float | None:
    """The measured mean tokens per question, at today's price."""
    if not option.hosted or not option.expected_tokens:
        return None
    price = settings.hosted_prices[option.model]
    tokens_in, tokens_out = option.expected_tokens
    return (tokens_in * price.input + tokens_out * price.output) / MILLION


def option_record(option: AnswerOption, loaded: set[str] | None, settings: Settings) -> dict:
    price = settings.hosted_prices.get(option.model) if option.hosted else None
    return {
        "id": option.id,
        "label": option.label,
        "provider": option.provider,
        "model": option.model,
        "fallback": option.fallback,
        "route": option.route,
        "hosted": option.hosted,
        "expected_seconds": option.expected_seconds,
        "expected_note": option.expected_note,
        "vision": option.vision,
        "web": option.web,
        "check_samples": option.check_samples,
        "check_ceiling_seconds": option.check_ceiling_seconds,
        "check_badge": option.check_badge,
        "loaded": None if option.hosted or loaded is None else option.model in loaded,
        "expected_usd": expected_usd(option, settings),
        "price": {"input": price.input, "output": price.output, "checked": price.checked} if price else None,
    }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def start_turn(settings: Settings, request: ChatRequest,
               option_id: str | None = None) -> tuple[int, int, list[dict], list[dict], tuple | None]:
    """Store the question. Returns the conversation, the question's id, what came before, the
    last citations, and what an export would save."""
    conn = connect(settings.db_path, apply_schema=False)
    try:
        conversation_id = request.conversation_id
        if conversation_id is None:
            conversation_id = conn.execute(
                "INSERT INTO conversations (title) VALUES (?)", (request.message[:80],)
            ).lastrowid
        elif conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone() is None:
            raise HTTPException(404, f"No conversation {conversation_id}")
        rows = conn.execute(
            "SELECT role, content, citations FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        question_id = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, option) VALUES (?, 'user', ?, ?)",
            (conversation_id, request.message, option_id),
        ).lastrowid
        conn.commit()
        return (conversation_id, question_id, recent_history(rows, settings.chat_history_tokens),
                last_citations(rows), exportable_answer(rows))
    finally:
        conn.close()


def finish_turn(settings: Settings, conversation_id: int, finished: dict, sources: list[dict],
                option_id: str | None = None) -> int:
    """Store the answer, with what the feedback list needs to know about it. Returns its id."""
    used = [source for source in sources if source["id"] in finished["citations"]]
    conn = connect(settings.db_path, apply_schema=False)
    try:
        answer_id = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, model, hosted, citations, option, details) "
            "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
            (conversation_id, finished["answer"], finished["model"], int(finished["hosted"]), json.dumps(used),
             option_id, json.dumps(feedback.answer_details(finished, sources))),
        ).lastrowid
        conn.commit()
        return answer_id
    finally:
        conn.close()


def fail_turn(settings: Settings, question_id: int, error: dict, sources: list[dict], web_sources: list[dict]) -> None:
    """A failed answer stores no reply, so its error goes with the question, for feedback on it."""
    conn = connect(settings.db_path, apply_schema=False)
    try:
        conn.execute("UPDATE messages SET details = ? WHERE id = ?",
                     (json.dumps(feedback.failure_details(error, sources, web_sources)), question_id))
        conn.commit()
    finally:
        conn.close()


class Rating(BaseModel):
    rating: str = Field(pattern="^(up|down)$")
    note: str = Field(default="", max_length=feedback.NOTE_CHARS)


class FeedbackChange(BaseModel):
    status: str | None = Field(default=None, pattern="^(open|done)$")
    candidate: bool | None = None


class NewAnnotation(BaseModel):
    doc_id: int
    page: int = Field(ge=1)
    rects: list[list[float]] = Field(min_length=1)
    quote: str
    color: str = "yellow"
    kind: str = "highlight"


class Passage(BaseModel):
    doc_id: int
    page: int = Field(ge=1)
    rects: list[list[float]] = Field(min_length=1)
    quote: str


class NewNote(BaseModel):
    body_md: str
    tags: list[str] = Field(default_factory=list)
    passage: Passage | None = None


class NoteChange(BaseModel):
    body_md: str | None = None
    tags: list[str] | None = None
    passage: Passage | None = None          # attach the note to this passage instead


class SuggestRequest(BaseModel):
    doc_id: int
    first_page: int = Field(ge=1)
    last_page: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=500)
    option: str | None = None               # one of suggest_options; the first by default


class ExplainRequest(BaseModel):
    text: str = Field(default="", max_length=8000)
    kind: str | None = None                 # equation or code; guessed from the text if left out
    depth: str = "intermediate"
    option: str | None = None               # the reader's mode; used if explain_options allows it for the kind
    passage: Passage | None = None          # a selection in the viewer: the book's LaTeX is used where it has one


class BatchAction(BaseModel):
    action: str = Field(pattern="^(accept|reject)$")


class AnnotationChange(BaseModel):
    color: str | None = None
    body: str | None = None
    status: str | None = None


class FiguresOk(BaseModel):
    usd: float = Field(ge=0)        # the estimate the reader was shown


class Details(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=1000)
    year: int | str | None = None
    publisher: str | None = Field(default=None, max_length=300)


def document_row(settings: Settings, doc_id: int):
    conn = connect(settings.db_path, apply_schema=False)
    try:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, f"No document {doc_id}")
    return row


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    embed=embed_texts,
    frontend: Path = FRONTEND,
) -> FastAPI:
    settings = settings or load_settings()
    conn = connect(settings.db_path)        # a first run has no tables until this
    try:
        metadata.refresh(conn, settings)    # metadata.yaml may have changed since
    except metadata.MetadataError as error:  # start anyway: the library pane says what is wrong
        log.error("%s", error)
    finally:
        conn.close()
    app = FastAPI(title="Study assistant")

    # Every module's NotFound and Refused are app.errors' (review X1): answered here, once.
    @app.exception_handler(errors.NotFound)
    async def not_found(request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(errors.Refused)
    async def refused(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.middleware("http")
    async def same_origin_writes(request, call_next):
        """Another web page open in the browser may send a POST here without a body, and the
        browser lets it (review S4). Writes come only from this app's own page, or from a
        client that isn't a browser (no Origin and no Sec-Fetch-Site, as the eval scripts)."""
        if request.method not in SAFE_METHODS and cross_site(request.headers):
            return JSONResponse({"detail": "Refused: a request from another site"}, status_code=403)
        return await call_next(request)

    async def chat_events(
        request: ChatRequest, option: AnswerOption, conversation_id: int, history: list[dict],
        cited: list[dict], exportable: tuple | None, reader_images: list[dict], hidden: tuple[str, ...] = (),
        question_id: int | None = None,
    ) -> AsyncIterator[str]:
        turn_settings = replace(option.apply(settings), hidden_collections=hidden,
                                search_notes="only" if request.notes_only else settings.search_notes)
        if option.hosted:
            turn_settings = replace(turn_settings, turn_id=uuid.uuid4().hex)
        yield sse("meta", {"conversation_id": conversation_id, "question_id": question_id,
                           "model": option.model, "hosted": option.hosted,
                           "option": option.id, "provider": option.provider, "route": option.route, "vision": option.vision})
        sources: list[dict] = []
        web_sources: list[dict] = []
        web_allowed = settings.tools_enabled and option.web != "off"
        web_on = request.web and web_allowed
        with figures.chat_running(settings):     # the figure job waits while this runs
            try:
                # A hosted option sends the sources, the question and the last turn with what it
                # cited, and nothing older.
                async for event, data in answer_events(
                    request.message, last_turn(history) if option.hosted else history, turn_settings, request.doc_ids,
                    embed, transport, provider=option.provider, cited=cited, tools=settings.tools_enabled,
                    exportable=exportable, vision=option.vision, reader_images=reader_images,
                    web=WebSession(settings, request.message) if web_on else None, web_mode=option.web,
                    propose=settings.web_proposals and web_allowed and not request.web,
                    web_query=request.web_query if web_on else None, checks=request.check,
                ):
                    if event == "sources":
                        sources = data["sources"]
                    if event == "more_sources":
                        sources = sources + data["sources"]
                    if event == "web_source":
                        web_sources = [page for page in web_sources if page["number"] != data["number"]] + [data]
                    if event in ("done", "error") and option.hosted:
                        # What the log says this question cost, every call and failed attempt included.
                        data = {**data, "spend": await run_in_threadpool(turn_cost, settings, turn_settings.turn_id)}
                    if event == "done":
                        answer_id = await run_in_threadpool(finish_turn, settings, conversation_id, data, sources, option.id)
                        data = {**data, "answer_id": answer_id}
                    if event == "error" and question_id is not None:
                        await run_in_threadpool(fail_turn, settings, question_id, data, sources, web_sources)
                    yield sse(event, data)
            except Exception as error:   # a bug mid-answer: the reader sees it and it is stored (review C4)
                log.exception("The answer failed")
                data = {"message": f"The answer failed ({type(error).__name__}: {str(error)[:200]})"}
                if question_id is not None:
                    await run_in_threadpool(fail_turn, settings, question_id, data, sources, web_sources)
                yield sse("error", data)

    def local_model_busy(option: AnswerOption):
        """A local option marks itself as a chat, so the figure job leaves the GPU to it, as chat
        does; Suggest and Explain didn't, and both models spilled to the CPU (review C7)."""
        return contextlib.nullcontext() if option.hosted else figures.chat_running(settings)

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        # Before the response starts, so an unknown option or conversation can still be an error.
        option = chosen_option(settings, request)
        reader_images = checked_images(request, option)
        if request.web and option.web == "off":
            raise HTTPException(422, f"{option.label} doesn't search the web. Ask with an option that does.")
        conversation_id, question_id, history, cited, exportable = await run_in_threadpool(
            start_turn, settings, request, option.id)
        hidden = await run_in_threadpool(hidden_now)
        return StreamingResponse(
            chat_events(request, option, conversation_id, history, cited, exportable, reader_images, hidden,
                        question_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/options")
    async def options() -> dict:
        """The answer selector: each option, and for local ones whether its model is loaded now."""
        loaded = await loaded_models(settings, transport)
        spend = await run_in_threadpool(month_spend, settings)
        return {"options": [option_record(option, loaded, settings) for option in settings.answer_options],
                "default": settings.answer_options[0].id,
                "suggest_options": settings.suggest_options,
                "explain_options": settings.explain_options,
                "switch_seconds": settings.local_switch_seconds,
                "month_spend": [{"provider": provider, "usd": round(usd, 4),
                                 "limit_usd": settings.hosted_monthly_limits.get(provider)}
                                for provider, usd in spend.items()]}

    @app.get("/api/search")
    def search(
        q: str = Query(min_length=1),
        k: int = Query(default=settings.search_top_k, ge=1, le=100),
        doc_id: list[int] | None = Query(default=None),
    ) -> dict:
        """Hybrid search with each list's rank shown, for debugging retrieval."""
        conn = connect(settings.db_path, apply_schema=False)
        try:
            hits = hybrid_search(conn, q, replace(settings, hidden_collections=library.hidden_collections(conn)),
                                 top_k=k, doc_ids=doc_id, embed=embed)
        except OSError as error:        # urllib wraps a refused connection in URLError
            raise HTTPException(503, f"Embedding failed: {error}") from error
        finally:
            conn.close()
        return {"query": q, "hits": [{
            "chunk_id": hit.chunk_id,
            "doc_id": hit.doc_id,
            "doc": hit.doc_path,
            "pages": [hit.page_start, hit.page_end],
            "heading_path": hit.heading_path,
            "score": hit.score,
            "keyword_rank": hit.keyword_rank,
            "vector_rank": hit.vector_rank,
            "text": hit.text,
        } for hit in hits]}

    def feedback_action(action):
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return action(conn)
        finally:
            conn.close()

    @app.put("/api/questions/{question_id}/feedback")
    def rate_answer(question_id: int, body: Rating) -> dict:
        """Thumbs up or down on the answer to a question, with an optional note. Rating again changes it."""
        return feedback_action(lambda conn: feedback.rate(conn, settings, question_id, body.rating, body.note))

    @app.get("/api/feedback")
    def feedback_list(rating: str | None = Query(default=None, pattern="^(up|down)$"),
                      status: str | None = Query(default=None, pattern="^(open|done)$")) -> dict:
        """The review list, newest first, with counts for its filters."""
        return feedback_action(lambda conn: feedback.entries(conn, rating, status))

    @app.patch("/api/feedback/{feedback_id}")
    def change_feedback(feedback_id: int, body: FeedbackChange) -> dict:
        """Mark an entry done or open, or a thumbs-down a gold candidate (data/gold_candidates.yaml) or not."""
        return feedback_action(lambda conn: feedback.update(conn, settings, feedback_id, body.status, body.candidate))

    def annotation_action(action):
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return action(conn)
        finally:
            conn.close()

    @app.get("/api/annotations")
    def annotation_list(doc_id: int) -> dict:
        """A document's highlights, by page."""
        def listed(conn):
            anchors.reanchor_document(conn, settings, doc_id)    # after a changed file, if the worker hasn't yet
            return annotations.listing(conn, doc_id)
        return {"annotations": annotation_action(listed)}

    @app.get("/api/annotations/lost")
    def lost_annotations() -> dict:
        """Highlights that lost their passage when their book's file changed (app/anchors.py)."""
        def listed(conn):
            anchors.reanchor_stale(conn, settings)
            return annotations.lost(conn)
        return {"annotations": annotation_action(listed)}

    @app.post("/api/suggestions")
    async def suggestions(body: SuggestRequest) -> dict:
        """Suggested highlights for an instruction on a few pages (app/suggest.py): saved as
        suggestions, for the reader to accept or reject. Nothing is written to the PDF."""
        chosen = body.option or settings.suggest_options[0]
        if chosen not in settings.suggest_options:
            raise HTTPException(409, f"Suggestions come from {', '.join(settings.suggest_options)}")
        option = next(o for o in settings.answer_options if o.id == chosen)
        conn = connect(settings.db_path, apply_schema=False)
        try:
            refused = deck.refuse_excluded(conn, settings, option, body.doc_id, body.first_page, body.last_page)
            if refused:
                raise HTTPException(409, refused)
            with local_model_busy(option):
                return await suggest.suggest(conn, option.apply(settings), option, body.doc_id, body.first_page,
                                             body.last_page, body.instruction, transport)
        except (httpx.HTTPError, RuntimeError) as error:
            raise HTTPException(502, f"The model failed: {error}"[:300]) from None
        finally:
            conn.close()

    @app.post("/api/explain")
    async def explain_text(body: ExplainRequest) -> dict:
        """A line-by-line explanation of an equation or a code block, as spans and explanations (app/explain.py)."""
        if body.option and not any(o.id == body.option for o in settings.answer_options):
            raise HTTPException(404, f"No answer option {body.option}")
        text, kind, context = body.text, body.kind, ""
        if body.passage:
            conn = connect(settings.db_path, apply_schema=False)
            try:
                latex = explain.book_latex(conn, body.passage.doc_id, body.passage.page, body.passage.rects)
                path = annotations.document_path(conn, settings, body.passage.doc_id)
            finally:
                conn.close()
            text = latex or body.passage.quote
            kind = "equation" if latex else kind
            context = await run_in_threadpool(suggest.page_text, path, body.passage.page, body.passage.page)
        kind = kind or explain.kind_of(text)
        allowed = settings.explain_options.get(kind) or [o.id for o in settings.answer_options]
        chosen = body.option if body.option in allowed else allowed[0]
        option = next((o for o in settings.answer_options if o.id == chosen), None)
        if option is None:
            raise HTTPException(409, f"Explain's option {chosen} isn't configured")
        if body.passage:
            conn = connect(settings.db_path, apply_schema=False)
            try:
                refused = deck.refuse_excluded(conn, settings, option, body.passage.doc_id,
                                               body.passage.page, body.passage.page)
            finally:
                conn.close()
            if refused:
                raise HTTPException(409, refused)
        try:
            with local_model_busy(option):
                explained = await explain.explain(option.apply(settings), option, text, kind, body.depth, context,
                                                  transport)
            return {**explained, "from_book": bool(body.passage and kind == "equation" and text != body.passage.quote)}
        except (httpx.HTTPError, RuntimeError) as error:
            raise HTTPException(502, f"The model failed: {error}"[:300]) from None

    @app.post("/api/annotations/batch/{batch}")
    def batch_action(batch: str, body: BatchAction) -> dict:
        """Accept or reject every suggestion still waiting in one batch."""
        def act(conn):
            if body.action == "accept":
                count = conn.execute("UPDATE annotations SET status = 'accepted' WHERE batch = ? AND status = 'suggested'",
                                     (batch,)).rowcount
            else:
                count = conn.execute("DELETE FROM annotations WHERE batch = ? AND status = 'suggested'", (batch,)).rowcount
            conn.commit()
            return {"changed": count}
        return annotation_action(act)

    @app.post("/api/annotations/export/{doc_id}")
    def export_annotated(doc_id: int) -> dict:
        """Write the annotated copy to outputs/annotated/ (app/annotated.py); the original is only read."""
        return annotation_action(lambda conn: annotated.write_copy(conn, settings, doc_id))

    @app.get("/api/annotated/{doc_id}")
    def annotated_file(doc_id: int, download: bool = False) -> FileResponse:
        path = annotated.copy_path(settings, doc_id)
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "No annotated copy yet: make one first")
        row = document_row(settings, doc_id)
        name = annotated.download_name(row["title"] or Path(row["path"]).stem)
        return FileResponse(path, media_type="application/pdf", filename=name,
                            content_disposition_type="attachment" if download else "inline")

    @app.post("/api/annotations")
    def new_annotation(body: NewAnnotation) -> dict:
        """Save a highlight from a selection in the viewer, snapped to the page's words (app/annotations.py)."""
        return annotation_action(lambda conn: annotations.create(
            conn, settings, body.doc_id, body.page, body.rects, body.quote, body.color, body.kind))

    @app.patch("/api/annotations/{annotation_id}")
    def change_annotation(annotation_id: int, body: AnnotationChange) -> dict:
        return annotation_action(lambda conn: annotations.update(conn, annotation_id, body.color, body.body, body.status))

    @app.delete("/api/annotations/{annotation_id}")
    def remove_annotation(annotation_id: int) -> dict:
        annotation_action(lambda conn: annotations.delete(conn, annotation_id))
        return {"ok": True}

    def note_action(action):
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return action(conn)
        finally:
            conn.close()

    @app.get("/api/notes")
    def note_list(doc_id: int | None = None, q: str = "") -> dict:
        """Notes, newest first; of one document, or with every word of q, if asked."""
        def listed(conn):
            anchors.reanchor_stale(conn, settings)               # after a changed file, if the worker hasn't yet
            return notes.listing(conn, doc_id, q)
        return {"notes": note_action(listed)}

    @app.post("/api/notes")
    def new_note(body: NewNote) -> dict:
        """A note, anchored to a passage selected in the viewer (snapped like a highlight) or free."""
        passage = body.passage.model_dump() if body.passage else None
        return note_action(lambda conn: notes.create(conn, settings, body.body_md, body.tags, passage))

    @app.patch("/api/notes/{note_id}")
    def change_note(note_id: int, body: NoteChange) -> dict:
        passage = body.passage.model_dump() if body.passage else None
        return note_action(lambda conn: notes.update(conn, note_id, body.body_md, body.tags, passage, settings))

    @app.delete("/api/notes/{note_id}")
    def remove_note(note_id: int) -> dict:
        note_action(lambda conn: notes.delete(conn, note_id))
        return {"ok": True}

    @app.get("/api/notes/export")
    def export_notes(doc_id: int | None = None) -> Response:
        """The notes as Markdown for Obsidian: a zip with one file per document."""
        data = note_action(lambda conn: notes.export_zip(conn, doc_id))
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="study-notes.zip"'})

    def hidden_now() -> tuple[str, ...]:
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return library.hidden_collections(conn)
        finally:
            conn.close()

    @app.get("/api/documents")
    def documents() -> dict:
        """Each document with where it is in the queued flow (app/library.py), and the library settings."""
        conn = connect(settings.db_path, apply_schema=False)
        try:
            rows = library.library_rows(conn, settings)
            set_aside = removed.removed_files(settings)
            latest = {item["path"]: item["entry"] for item in set_aside}     # sorted by time: the last wins
            for row in rows:
                row["removed_as"] = latest.get(row["path"]) if row["status"] == "missing" else None
            return {"documents": rows, "settings": library.library_settings(conn),
                    "metadata_problem": metadata.problem(settings.library_dir), "removed": set_aside}
        finally:
            conn.close()

    @app.put("/api/library/files/{name}")
    async def add_document(name: str, request: Request) -> dict:
        """Add a PDF or a deck from the browser: it streams into library/ under its own name,
        checked, never overwriting a file (app/upload.py). The watcher takes it from there."""
        try:
            return await upload.save_upload(settings, name, request.stream(), request.headers.get("content-length"))
        except upload.UploadError as error:
            raise HTTPException(error.status, str(error)) from None

    @app.put("/api/library/settings")
    def library_settings(body: dict) -> dict:
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return library.set_library_settings(conn, body)
        finally:
            conn.close()

    def library_action(doc_id: int, action):
        conn = connect(settings.db_path, apply_schema=False)
        try:
            doc = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if doc is None:
                raise HTTPException(404, f"No document {doc_id}")
            return action(conn, doc)
        finally:
            conn.close()

    @app.post("/api/documents/{doc_id}/figures/approve")
    def approve_figures(doc_id: int, body: FiguresOk) -> dict:
        """The reader agrees to describe this document's figures for the estimate they were shown."""
        library_action(doc_id, lambda conn, doc: library.approve_figures(conn, doc_id, body.usd))
        return {"ok": True}

    @app.post("/api/documents/{doc_id}/figures/skip")
    def skip_figures(doc_id: int) -> dict:
        library_action(doc_id, lambda conn, doc: library.skip_figures(conn, doc_id))
        return {"ok": True}

    @app.post("/api/documents/{doc_id}/set-aside")
    def set_aside(doc_id: int) -> dict:
        """Move a document's file into library/.removed/: out of search, everything kept (app/removed.py)."""
        return library_action(doc_id, lambda conn, doc: removed.set_aside(conn, settings, doc))

    @app.post("/api/library/removed/restore")
    def restore_removed(body: RemovedEntry) -> dict:
        """Move a removed file back; the worker restores its document as it was."""
        return removed.restore(settings, body.entry)

    @app.get("/api/library/removed/empty")
    def empty_removed_preview() -> dict:
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return removed.empty_preview(conn, settings)
        finally:
            conn.close()

    @app.post("/api/library/removed/empty")
    def empty_removed() -> dict:
        """Delete the files in library/.removed/ and the index of their documents: the only deleting step."""
        conn = connect(settings.db_path, apply_schema=False)
        try:
            return removed.empty(conn, settings)
        finally:
            conn.close()

    @app.get("/api/documents/{doc_id}/removal")
    def removal_preview(doc_id: int) -> dict:
        """What removing a document whose file has gone would delete, for the reader to confirm."""
        return library_action(doc_id, lambda conn, doc: library.removal_preview(conn, settings, doc))

    @app.post("/api/documents/{doc_id}/remove")
    def remove_document(doc_id: int) -> dict:
        """Delete a gone document's chunks, figures, summaries, jobs, folders and cache rows."""
        return {"removed": library_action(doc_id, lambda conn, doc: library.remove_document(conn, settings, doc))}

    @app.get("/api/documents/{doc_id}/details")
    def document_details(doc_id: int) -> dict:
        return library_action(doc_id, lambda conn, doc: library.suggestion(settings, doc))

    @app.post("/api/documents/{doc_id}/details")
    def save_document_details(doc_id: int, body: Details) -> dict:
        """Write the document's metadata.yaml entry, keeping the rest of the file as it is."""
        return library_action(doc_id, lambda conn, doc: library.save_details(conn, settings, doc, body.model_dump(), embed))

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: int) -> FileResponse:
        """The PDF the viewer shows: the file itself, or a deck's LibreOffice render."""
        row = document_row(settings, doc_id)
        if deck.is_deck(row["kind"]):
            path = deck.render_path(settings, row["sha256"])
            if not path.is_file():
                raise HTTPException(404, "The deck hasn't been rendered yet")
            return FileResponse(path, media_type="application/pdf", headers={"X-Content-Type-Options": "nosniff"})
        path = (settings.library_dir / row["path"]).resolve()
        if not path.is_relative_to(settings.library_dir.resolve()) or not path.is_file():
            raise HTTPException(404, "The file isn't in the library")
        return FileResponse(path, media_type="application/pdf")

    @app.get("/api/documents/{doc_id}/crops/{name}")
    def formula_crop(doc_id: int, name: str) -> FileResponse:
        if not CROP_NAME.fullmatch(name):
            raise HTTPException(404, "No such crop")
        sha256 = document_row(settings, doc_id)["sha256"]
        path = settings.parsed_dir / sha256 / "formula_crops" / name
        if not path.is_file():
            raise HTTPException(404, "No such crop")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/documents/{doc_id}/figures/{name}")
    def figure_image(doc_id: int, name: str) -> FileResponse:
        """A figure's image as the parser saved it, from that document's own folder only."""
        if not FIGURE_NAME.fullmatch(name):
            raise HTTPException(404, "No such figure")
        path = settings.parsed_dir / document_row(settings, doc_id)["sha256"] / "figures" / name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "No such figure")
        return FileResponse(path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/outputs/{name}")
    def output_file(name: str, download: bool = False) -> FileResponse:
        """A generated file. Only known types, so nothing here can run as a web page."""
        path = settings.outputs_dir / name
        kind = OUTPUT_TYPES.get(path.suffix.lower())
        if not OUTPUT_NAME.fullmatch(name) or kind is None or path.is_symlink() or not path.is_file():
            raise HTTPException(404, "No such file")
        return FileResponse(path, media_type=kind, headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}"',
        })

    @app.get("/api/outputs/{name}/preview")
    async def output_preview(name: str) -> FileResponse:
        """A Word or PowerPoint output as LibreOffice renders it (app/office.py), made on the
        first request and kept. 422 with the reason when it can't be rendered; the chat then
        shows the file's source text, labelled as such."""
        path = settings.outputs_dir / name
        if (not OUTPUT_NAME.fullmatch(name) or path.suffix.lower() not in office.SUFFIXES
                or path.is_symlink() or not path.is_file()):
            raise HTTPException(404, "No such file")
        try:
            rendered = await office.cached_render(settings, path)
        except office.SandboxUnavailable:
            raise HTTPException(422, "LibreOffice isn't set up (scripts/build_office.sh)") from None
        except office.OfficeError as error:
            raise HTTPException(422, str(error)) from None
        return FileResponse(rendered, media_type="application/pdf", headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{path.stem}.pdf"',
        })

    # Last, so the API routes above win. Absent until `npm run build` has run.
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

        @app.middleware("http")
        async def page_caching(request, call_next):
            """The page always revalidates, so a rebuilt frontend reaches the browser at once;
            its scripts are named by their content hash, so they never change and can be kept."""
            response = await call_next(request)
            path = request.url.path
            if not path.startswith("/api/"):
                if path.startswith("/assets/") and response.status_code == 200:
                    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    response.headers["Cache-Control"] = "no-cache"
            return response
    return app


def warm_up(settings: Settings) -> None:
    """Load the helper and the reranker now, so the first question doesn't wait for them."""
    warm_up_helper(settings)
    if settings.rerank_enabled:
        try:
            cross_encoder.scores("warm up", ["warm up"], settings)
        except (RuntimeError, OSError, ValueError, ImportError):
            pass        # search logs this when it happens for real


def main() -> None:
    parser = argparse.ArgumentParser(description="The study assistant: web server and ingestion worker.")
    parser.add_argument("--no-worker", action="store_true",
                        help="the server only; run `python -m app.ingest.worker` yourself")
    args = parser.parse_args()
    settings = load_settings()
    app = create_app(settings)          # applies the schema before the worker opens the database
    supervisor = None if args.no_worker else WorkerSupervisor(settings)
    if supervisor:
        supervisor.start()
    threading.Thread(target=warm_up, args=(settings,), daemon=True).start()
    try:
        uvicorn.run(app, host=HOST, port=settings.port)
    finally:
        if supervisor:
            print("Stopping the worker...", flush=True)
            supervisor.stop()


if __name__ == "__main__":
    main()
