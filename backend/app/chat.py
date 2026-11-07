"""One answer, from question to cited text. The web API and the eval both run this."""
from __future__ import annotations

import base64
import json
import math
import sqlite3
import struct
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from urllib.parse import urlsplit

import httpx
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.db.connection import connect
from app.ingest.chunker import estimate_tokens
from app.ingest.embed import embed_texts
from app.llm import hosted, openrouter
from app.llm.citations import CitationFilter, Citations
from app.llm.ollama import Delta, stream_chat
from app.llm.prompts import build_messages, tools_wanted, with_images
from app.retrieval.context import carried_hits, expand_hits
from app.retrieval.rewrite import adds_context, standalone_question
from app.retrieval.search import Hit, search_chunks
from app.tools.executor import EXPORT_REQUEST, ToolBox
from app.tools.web import WebSession
from app.verify import ChecksStream, Report, block_from_reply, check_block, extraction_messages, split_checks
from app.tools.web_gather import gather, pages_block, search_query
from app.tools.loop import CUT_OFF, Finished, HostedModel, LocalModel, ModelFailed, OpenRouterModel, run_turn

EMPTY_ANSWER = "The model returned no answer"
NO_LIBRARY_NEEDED = {"run_python", "sympy_check"}
Event = tuple[str, dict]


def recent_history(rows: list, budget: int) -> list[dict]:
    """The newest turns that fit in `budget` tokens, oldest first."""
    kept: list[dict] = []
    for row in reversed(rows):
        budget -= estimate_tokens(row["content"])
        if budget < 0:
            break
        kept.append({"role": row["role"], "content": row["content"]})
    return kept[::-1]


def last_turn(history: list[dict]) -> list[dict]:
    """The latest question with its answer, all that a hosted option sends of the past.

    A question whose answer failed has no reply stored, so it's skipped rather than
    sent as half a turn.
    """
    for index in range(len(history) - 1, 0, -1):
        if history[index]["role"] == "assistant" and history[index - 1]["role"] == "user":
            return history[index - 1:index + 1]
    return []


def last_citations(rows: list) -> list[dict]:
    """The sources the latest answer cited, as stored with it."""
    for row in reversed(rows):
        if row["role"] == "assistant":
            return json.loads(row["citations"] or "[]")
    return []


def exportable_answer(rows: list) -> tuple[str, list[dict]] | None:
    """The newest real answer, with what it cited: what "export this answer" means.

    Not simply the last reply: after one export, that's only a note saying it was saved.
    """
    for index in range(len(rows) - 1, 0, -1):
        if (rows[index]["role"] == "assistant" and rows[index - 1]["role"] == "user"
                and not EXPORT_REQUEST.search(rows[index - 1]["content"])):
            return rows[index]["content"], json.loads(rows[index]["citations"] or "[]")
    return None


def needs_no_search(question: str, turn: list[dict]) -> bool:
    """Exports and bare code or SymPy requests: passages would only add noise, and with a
    hosted option they would send book text out for nothing. The model can still search."""
    if turn and EXPORT_REQUEST.search(question):
        return True
    wanted = tools_wanted(question)
    return bool(wanted) and set(wanted) <= NO_LIBRARY_NEEDED


def regions_of(conn: sqlite3.Connection, block_ids: Sequence[int]) -> list[dict]:
    """Every region a chunk's blocks occupy, in reading order: what the viewer highlights."""
    if not block_ids:
        return []
    places = ",".join("?" * len(block_ids))
    rows = conn.execute(
        f'SELECT page, bbox, bboxes_json FROM blocks WHERE id IN ({places}) ORDER BY "order"',
        list(block_ids),
    ).fetchall()
    regions = []
    for row in rows:
        if row["bboxes_json"]:
            regions += json.loads(row["bboxes_json"])
        elif row["bbox"] and row["page"]:
            regions.append({"page": row["page"], "bbox": json.loads(row["bbox"])})
    return regions


def figure_url(doc_id: int, image_path: str) -> str:
    return f"/api/documents/{doc_id}/figures/{image_path.rsplit('/', 1)[-1]}"


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def chunk_vector(conn: sqlite3.Connection, chunk_id: int) -> tuple[float, ...] | None:
    row = conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return struct.unpack(f"{len(row[0]) // 4}f", row[0]) if row else None


def figures_of(conn: sqlite3.Connection, hit: Hit, question_vector: Sequence[float] | None = None) -> list[dict]:
    """The figures in a passage, merged neighbours included, the likeliest one first.

    A passage merged from neighbours can hold many figures. With the question's vector
    each is ranked by how close its chunk (caption and description) is to the question;
    on all ten figure questions that put the figure asked about first, where its distance
    from the chunk the passage matched did not. Without a vector, nearest that chunk first.
    """
    if not hit.chunk_ids:
        return []
    places = ",".join("?" * len(hit.chunk_ids))
    found = conn.execute("SELECT seq FROM chunks WHERE id = ?", (hit.chunk_id,)).fetchone()
    focus = found[0] if found else 0
    rows = conn.execute(
        "SELECT c.id AS chunk_id, c.seq, f.id, f.page, f.caption, f.image_path, d.sha256 FROM chunks c "
        "JOIN figures f ON f.doc_id = c.doc_id AND f.image_path = json_extract(c.meta, '$.image_path') "
        f"JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({places}) AND c.kind = 'figure' "
        "ORDER BY abs(c.seq - ?), c.seq", [*hit.chunk_ids, focus]).fetchall()
    figures = [{"figure_id": row["id"], "chunk_id": row["chunk_id"], "page": row["page"], "caption": row["caption"] or "",
                "image_path": row["image_path"], "sha256": row["sha256"], "url": figure_url(hit.doc_id, row["image_path"]),
                "score": None} for row in rows]
    if question_vector is not None:
        for figure in figures:
            vector = chunk_vector(conn, figure["chunk_id"])
            figure["score"] = similarity(question_vector, vector) if vector else -1.0
        figures.sort(key=lambda figure: -figure["score"])      # stable: ties keep the nearest-chunk order
    return figures


def figure_images(hits: list[Hit], settings: Settings) -> list[dict]:
    """The images to send a vision model, up to the limit: the figures closest to the question
    across all the sources; without scores, each source's first in turn."""
    candidates = [(number, place, figure) for number, hit in enumerate(hits, 1) for place, figure in enumerate(hit.figures)]
    if all(figure.get("score") is not None for _, _, figure in candidates):
        candidates.sort(key=lambda item: (-item[2]["score"], item[0], item[1]))
    else:
        candidates.sort(key=lambda item: (item[1], item[0]))
    images = []
    for number, _, figure in candidates:
        if len(images) >= settings.figure_images_max:
            break
        path = settings.parsed_dir / figure["sha256"] / figure["image_path"]
        if not path.is_file():
            continue
        media = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        images.append({"source": number, "page": figure["page"], "caption": figure["caption"],
                       "media_type": media, "data": base64.b64encode(path.read_bytes()).decode()})
    return images


def images_for(reader_images: Sequence[dict], hits: list[Hit], settings: Settings) -> list[dict]:
    return [{**image, "reader": True} for image in reader_images] + figure_images(hits, settings)


def crop_url(doc_id: int, crop: str) -> str:
    return f"/api/documents/{doc_id}/crops/{crop.rsplit('/', 1)[-1]}"


def source_record(number: int, hit: Hit) -> dict:
    """What the frontend needs to show a source and open it."""
    return {
        "id": number,
        "chunk_id": hit.chunk_id,
        "chunk_ids": hit.chunk_ids,
        "doc_id": hit.doc_id,
        "doc_title": hit.doc_title,
        "author": hit.doc_author,
        "year": hit.doc_year,
        "publisher": hit.doc_publisher,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "focus_page": hit.focus_page,
        "heading_path": hit.heading_path,
        "regions": hit.regions,
        "unverified_pages": sorted({formula["page"] for formula in hit.meta.get("unverified_formulas", [])
                                    if formula.get("page")}),
        "figures": [{"url": figure["url"], "page": figure["page"], "caption": figure["caption"]}
                    for figure in hit.figures],
        "kind": hit.kind,
        "note_id": hit.meta.get("note_id"),
    }


def unverified_crops(cited: list[int], hits: list[Hit]) -> list[dict]:
    """Page crops of the unverified formulas in every source the model was given.

    Not only the cited ones: a model can use a formula without citing it, and then
    the crop is the reader's only warning. Cited sources come first.
    """
    order = cited + [n for n in range(1, len(hits) + 1) if n not in cited]
    crops: dict[str, dict] = {}
    for number in order:
        hit = hits[number - 1]
        for formula in hit.meta.get("unverified_formulas", []):
            if formula.get("crop"):
                url = crop_url(hit.doc_id, formula["crop"])
                crops.setdefault(url, {
                    "source": number,
                    "doc_id": hit.doc_id,
                    "page": formula["page"],
                    "url": url,
                    "cited": number in cited,
                })
    return list(crops.values())


def find_sources(
    settings: Settings,
    question: str,
    doc_ids: Sequence[int] | None,
    embed: Callable,
    cited: Sequence[dict] = (),
) -> list[Hit]:
    conn = connect(settings.db_path, apply_schema=False)
    try:
        hits = search_chunks(conn, question, settings, doc_ids=doc_ids, embed=embed)
        hits = expand_hits(conn, hits, settings.search_neighbour_hits,
                           settings.search_neighbour_radius, settings.search_source_tokens,
                           summary_tokens=settings.search_summary_tokens)
        found = {chunk for hit in hits for chunk in hit.chunk_ids}
        hits = carried_hits(conn, cited, found, settings.followup_carry_tokens, doc_ids, settings.hidden_collections) + hits
        for hit in hits:
            if hit.kind == "note":      # its passage is already its regions; a note has no figures
                continue
            hit.regions = regions_of(conn, hit.block_ids)
            focus = regions_of(conn, hit.focus_block_ids)
            hit.focus_page = focus[0]["page"] if focus else hit.page_start
            hit.figures = figures_of(conn, hit)
        if any(hit.figures for hit in hits):
            # One more embedding of the question, about 50 ms, to rank the figures it may be about.
            question_vector = embed([question], settings)[0]
            for hit in hits:
                if hit.figures:
                    hit.figures = figures_of(conn, hit, question_vector)
        return hits
    finally:
        conn.close()


def model_name(settings: Settings, provider: str) -> str:
    return {"anthropic": settings.hosted_model, "openrouter": settings.openrouter_model}.get(
        provider, settings.chat_model)


def no_answer(reason: str | None, settings: Settings, provider: str) -> dict:
    """The error for a reply with no answer. A cut-off one says the limit, since thinking used it up."""
    if reason in CUT_OFF:
        limit = {"anthropic": settings.hosted_max_tokens, "openrouter": settings.openrouter_max_tokens}.get(
            provider, settings.chat_num_predict)
        return {"message": f"{model_name(settings, provider)} stopped at its {limit:,}-token limit, "
                           "thinking included, before it wrote an answer.", "cut_off": True, "no_answer": True}
    return {"message": f"{EMPTY_ANSWER} ({reason or 'stream ended early'})", "cut_off": False, "no_answer": True}


async def model_reply(
    messages: list[dict],
    settings: Settings,
    provider: str,
    transport: httpx.AsyncBaseTransport | None,
    purpose: str,
    fallback: bool,
    client=None,
) -> AsyncIterator[Delta]:
    """The reply from the provider's model. Only Anthropic has a fallback; OpenRouter never falls back."""
    if provider == "anthropic":
        async for delta in hosted.answer_stream(messages, settings, purpose=purpose, fallback=fallback):
            yield delta
        return
    if provider == "openrouter":
        async for delta in openrouter.stream_chat(messages, settings, client, purpose=purpose):
            yield delta
        return
    timeout = httpx.Timeout(settings.chat_timeout_seconds, connect=10)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        async for delta in stream_chat(messages, settings, client):
            yield delta


async def verify_answer(question: str, answer: str, settings: Settings, provider: str,
                        transport: httpx.AsyncBaseTransport | None = None, purpose: str = "chat",
                        client=None) -> tuple[str, Report]:
    """The answer as shown, without its claims block, and the report on those claims.

    An answer that came without a block gets one extra call to the same model, which
    writes the block for the answer as it stands (source "extracted"). The block itself is
    only cut out here, as text; the sandbox reads it.
    """
    shown, block = split_checks(answer)
    if block is not None:
        return shown, await check_block(settings, block, "answer")
    started = time.monotonic()
    reply, cost = "", 0.0
    # Listing an answer's claims needs no long reasoning: Gemma took 173 s thinking about it.
    # gpt-oss only takes levels, and loops without any; others turn it off.
    light = replace(settings, chat_think="low" if isinstance(settings.chat_think, str) else False)
    try:
        async for delta in model_reply(extraction_messages(question, shown), light, provider, transport,
                                       f"{purpose}:extract", False, client if provider == "openrouter" else None):
            reply += delta.text or ""
            if delta.done:
                cost = delta.cost_usd or 0.0
    except (httpx.HTTPError, RuntimeError) as error:
        return shown, Report(source="none", failure=f"The claims couldn't be extracted: {error}"[:300],
                             extraction_seconds=round(time.monotonic() - started, 2))
    extra = {"extraction_seconds": round(time.monotonic() - started, 2), "extraction_cost_usd": cost}
    block = block_from_reply(reply)
    if block is None:
        return shown, Report(source="none", **extra)
    report = await check_block(settings, block, "extracted")
    return shown, report.model_copy(update=extra)


async def app_web_search(web: WebSession, query: str, settings: Settings, embed: Callable,
                         found: list) -> AsyncIterator[Event]:
    """The app's own web search, for an option with web: app, shown as the tool lines a
    search by the model would be. What it found goes into `found`, for the prompt."""
    shown = search_query(query)
    yield "tool_start", {"id": "app-search", "name": "web_search", "arguments": {"query": shown}}
    gathered = await gather(web, query, lambda texts: embed(texts, settings))
    ok = gathered.search_error is None
    yield "tool_end", {"id": "app-search", "name": "web_search", "ok": ok,
                       "summary": f'The app searched the web for "{shown}"' if ok else "The web search failed",
                       "result": gathered.search_error or f"{gathered.results} results", "detail": {}}
    for number, (url, why) in enumerate(gathered.failed, 1):
        call = f"app-skip-{number}"
        yield "tool_start", {"id": call, "name": "fetch_page", "arguments": {"url": url}}
        yield "tool_end", {"id": call, "name": "fetch_page", "ok": False,
                           "summary": f"Couldn't use {urlsplit(url).hostname or url}: {why}", "result": why, "detail": {}}
    for page in gathered.pages:
        call = f"app-read-{page.number}"
        record = {"number": page.number, "title": page.title, "url": page.final_url, "fetched": page.fetched}
        yield "tool_start", {"id": call, "name": "fetch_page", "arguments": {"url": page.final_url}}
        yield "web_source", record
        yield "tool_end", {"id": call, "name": "fetch_page", "ok": True, "summary": f"Read [Web {page.number}]: {page.title}",
                           "result": "", "detail": {"web": page.number, "url": page.final_url, "title": page.title}}
    found.append(gathered)


def web_fields(box: ToolBox, citations: Citations) -> dict:
    """The web pages this answer read, which it cited, and any search it proposed."""
    pages = box.web.pages if box.web else []
    return {
        "web_sources": [{"number": page.number, "title": page.title, "url": page.final_url, "fetched": page.fetched}
                        for page in pages],
        "web_citations": citations.web.cited,
        "dropped_web_citations": citations.web.dropped,
        "web_proposal": box.proposal,
        "web_searches": box.web.searches if box.web else 0,
    }


def done_event(finished: Finished, citations: Citations, hits: list[Hit], settings: Settings,
               provider: str, fell_back: dict | None, box: ToolBox | None = None) -> dict:
    return {
        **(web_fields(box, citations) if box else {}),
        "answer": finished.answer,
        "citations": citations.cited,
        "dropped_citations": citations.dropped,
        "unverified": unverified_crops(citations.cited, hits),
        "done_reason": finished.done_reason,
        "prompt_tokens": finished.prompt_tokens,
        "output_tokens": finished.output_tokens,
        "cost_usd": finished.cost_usd,
        "model": finished.model or model_name(settings, provider),
        "hosted": provider != "ollama",
        "served_by": finished.provider,
        "fallback_from": fell_back["from"] if fell_back else None,
        "fallback_reason": fell_back["reason"] if fell_back else None,
        "files": finished.files,
        "tools": finished.tools,
    }


async def tool_answer(
    question: str,
    messages: list[dict],
    hits: list[Hit],
    settings: Settings,
    find: Callable,
    previous_answer: str | None,
    cited: Sequence[dict],
    provider: str,
    transport: httpx.AsyncBaseTransport | None,
    purpose: str,
    fallback: bool,
    client=None,
    web: WebSession | None = None,
    propose: bool = False,
    web_tools: bool = True,
) -> AsyncIterator[Event]:
    """Run the turn, and start it again on the fallback model if the first one fails.

    Starting again, not continuing, because no answer is ever part one model and part
    another. A file the first attempt made stays in outputs/, but it isn't announced.
    """
    deep = provider == "anthropic"
    spare = settings.hosted_fallback_model if deep and fallback else None
    attempts = [settings]
    if spare and spare != settings.hosted_model:
        attempts.append(replace(settings, hosted_model=spare))
    fell_back: dict | None = None

    for number, attempt in enumerate(attempts):
        hits_now = list(hits)
        # A fallback starts its web work again too, unless the app did it: its pages are in the prompt.
        session = web.restart() if web and number and web_tools else web
        citations = Citations(set(range(1, len(hits_now) + 1)),
                              (lambda s=session: {page.number for page in s.pages}) if session else (lambda: set()))
        box = ToolBox(attempt, find, {n: source_record(n, hit) for n, hit in enumerate(hits_now, 1)}, hits_now,
                      previous_answer, {source["id"]: source for source in cited}, source_record, question,
                      web=session, propose=propose, web_tools=web_tools)
        if provider == "anthropic":
            model = HostedModel(attempt, purpose, client)
        elif provider == "openrouter":
            model = OpenRouterModel(attempt, purpose, client)
        else:
            model = LocalModel(attempt, transport)
        try:
            async for event, data in run_turn(model, list(messages), box, citations, attempt, strict=deep):
                if event == "finished":
                    finished = data["result"]
                    if not finished.answer.strip():
                        yield "error", no_answer(finished.done_reason, attempt, provider)
                        return
                    yield "done", done_event(finished, citations, box.hits, attempt, provider, fell_back, box)
                    return
                yield event, data
        except (ModelFailed, hosted.HostedFailure) as failure:
            final = number == len(attempts) - 1 or getattr(failure, "key_rejected", False)
            reason = str(failure).removeprefix("Hosted model: ")
            if final:
                yield "error", {"message": f"The model failed: {failure}"}
                return
            fell_back = {"model": spare, "from": settings.hosted_model, "reason": reason}
            yield "fallback", fell_back
        except (httpx.HTTPError, RuntimeError) as error:
            yield "error", {"message": f"The model failed: {error}"}
            return


def again(web: WebSession | None, number: int, app_web: bool) -> WebSession | None:
    """A later attempt starts its web work again, unless the app searched: its pages are in the prompt."""
    if web is None or number == 1 or app_web:
        return web
    return web.restart()


def check_record(report: Report) -> dict:
    """A claims report as the chat shows it and the feedback list keeps it."""
    return {
        "badge": report.badge(),
        "source": report.source,
        "failure": report.failure,
        "summary": {key: value for key, value in report.summary().items() if key != "badge"},
        "claims": [{"line": claim.line, "label": claim.label, "verdict": claim.verdict,
                    "shown": claim.verdict_text(), "note": claim.note} for claim in report.claims],
        "block": report.block,          # as the model wrote it: plain text, shown as text
        "seconds": report.seconds,
        "extraction_seconds": report.extraction_seconds,
    }


def attempt_record(attempt: dict) -> dict:
    """An attempt that isn't the one shown: its answer, its checks and the claims that failed."""
    report: Report = attempt["report"]
    return {"attempt": attempt["number"], "answer": attempt["shown"], "checks": check_record(report),
            "failing": [claim.line for claim in report.claims if claim.verdict == "failed"]}


async def checked_attempts(generate: Callable[[int], AsyncIterator[Event]], question: str, settings: Settings,
                           provider: str, transport: httpx.AsyncBaseTransport | None, purpose: str, client,
                           sources: list[dict]) -> AsyncIterator[Event]:
    """Check on: answers until one's checked claims include no failed one, at most
    `settings.check_samples` of them (best-of-N; 1 is badges only), within the ceiling.

    The first attempt streams as usual, without its claims block. After each, a
    "check_status" while the sandbox checks it; when a claim failed and another attempt
    fits, "attempt_replaced" (with the failing claims) and the next attempt streams in its
    place. No attempt starts if the last one's time would take the question past
    `check_ceiling_seconds`, and one running past it is given up. "done" carries the
    attempt shown, its "checks" and the "replaced" ones. If every attempt has a failed
    claim, the one with the fewest is shown, and the badge says so; it never claims more.
    """
    samples = max(1, settings.check_samples)
    ceiling = settings.check_ceiling_seconds
    started = time.monotonic()
    attempts: list[dict] = []
    stopped: str | None = None
    for number in range(1, samples + 1):
        attempt_started = time.monotonic()
        done = failure = None
        stream = ChecksStream()
        events = generate(number)
        try:
            async for event, data in events:
                if number > 1 and ceiling and time.monotonic() - started > ceiling:
                    stopped = "time"
                    break
                if event == "token":
                    if text := stream.feed(data["text"]):
                        yield "token", {"text": text}
                    continue
                if event == "done":
                    done = data
                    break
                if event == "error":
                    failure = data
                    break
                if event == "fallback" or (event == "withdraw" and "end" not in data):
                    stream = ChecksStream()     # the text before it starts again: so does the filter
                yield event, data
        finally:
            await events.aclose()
        if done is None:
            if attempts:        # a later attempt failed or ran out of time: the earlier ones stand
                stopped = stopped or "error"
                break
            yield "error", failure or {"message": f"{EMPTY_ANSWER} (stream ended early)", "no_answer": True}
            return
        if rest := stream.flush():
            yield "token", {"text": rest}
        yield "check_status", {"stage": "checking", "attempt": number, "of": samples}
        shown, report = await verify_answer(question, done["answer"], settings, provider, transport, purpose, client)
        seconds = time.monotonic() - attempt_started
        attempts.append({"number": number, "done": done, "shown": shown, "report": report, "seconds": seconds})
        if not report.count("failed") or number == samples:
            break
        if ceiling and time.monotonic() - started + seconds > ceiling:
            stopped = "time"
            break
        yield "attempt_replaced", {**attempt_record(attempts[-1]), "next": number + 1, "of": samples,
                                   "sources": sources}

    best = min(attempts, key=lambda a: (a["report"].count("failed"), -a["report"].passed, a["number"]))
    cost = sum((a["done"].get("cost_usd") or 0.0) + a["report"].extraction_cost_usd for a in attempts)
    yield "done", {
        **best["done"],
        "answer": best["shown"],
        "cost_usd": cost,
        "checks": {**check_record(best["report"]), "attempt": best["number"], "attempts": len(attempts),
                   "samples": samples, "stopped": stopped, "seconds": round(time.monotonic() - started, 2),
                   "none_passed": samples > 1 and bool(best["report"].count("failed"))},
        "replaced": [attempt_record(a) for a in attempts if a is not best],
    }


async def answered(generate: Callable[[int], AsyncIterator[Event]], checks: bool, question: str,
                   settings: Settings, provider: str, transport: httpx.AsyncBaseTransport | None, purpose: str,
                   client, sources: list[dict]) -> AsyncIterator[Event]:
    """One answer, or with Check on, the checked attempts."""
    events = (checked_attempts(generate, question, settings, provider, transport, purpose, client, sources)
              if checks else generate(1))
    async for event in events:
        yield event


async def answer_events(
    question: str,
    history: list[dict],
    settings: Settings,
    doc_ids: Sequence[int] | None = None,
    embed: Callable = embed_texts,
    transport: httpx.AsyncBaseTransport | None = None,
    provider: str = "ollama",
    purpose: str = "chat",
    fallback: bool = True,
    cited: Sequence[dict] = (),
    tools: bool = False,
    client=None,
    exportable: tuple[str, list[dict]] | None = None,
    vision: bool = False,
    reader_images: Sequence[dict] = (),
    web: WebSession | None = None,
    propose: bool = False,
    web_query: str | None = None,
    web_mode: str = "tools",
    checks: bool = False,
) -> AsyncIterator[Event]:
    """Yield "sources", then "thinking" and "token" pieces, then "done" or "error".

    `provider` is ollama, openrouter or anthropic, with the model already in `settings`.
    With anthropic a "fallback" event can come along the way. It means drop the answer so
    far: another model has started again, and later events are its answer.

    With `tools`, "tool_start", "tool_end", "file" and "more_sources" come along the way
    too. The gold-set eval leaves them off, so its numbers stay comparable. `client` is
    the hosted client, for tests. `exportable` is what an export would save, if it isn't
    simply the last answer.

    With `vision`, the images of the figures in the passages go to the model too, and
    the descriptions stay in the text, where they found the passage. `reader_images` are
    the reader's own, for this question only, and go first; only a vision option gets them.

    `cited` is what the previous answer cited. It's carried over only for a real
    follow-up, one the helper had to add context to, so a new topic starts clean.

    `web` is this question's web session when the reader turned web search on (Phase 5d):
    then web_search and fetch_page are offered, and "web_source" events come as pages are
    read. Without it, `propose` offers propose_web_search instead, which ends in a
    "web_proposal" event: the reader decides. `web_query` is a search the reader agreed to.
    Web needs tools; the eval's tool-free path never sees any of it. With `web_mode` "app"
    the app searches and reads the pages itself first, and the model gets them in its
    message, with no web tools.

    With `checks` the model is asked to end its answer with a ```checks block of claims
    (app/verify.py), which is checked in the sandbox and never streamed: see
    checked_attempts for the events, and best-of-N with `settings.check_samples`.
    """
    app_web = web is not None and web_mode == "app"
    turn = last_turn(history)
    if tools and needs_no_search(question, turn):
        yield "sources", {"sources": [], "query": None}
        if exportable:
            previous, kept = exportable
        else:
            previous, kept = (turn[-1]["content"] if turn else None), cited

        def find_only(text: str) -> list[Hit]:
            return find_sources(settings, text, doc_ids, embed)

        found: list = []
        if app_web:
            async for event in app_web_search(web, web_query or question, settings, embed, found):
                yield event
        messages = build_messages(question, [], history, tools=True, searched=False, web=web is not None,
                                  propose=propose, web_query=web_query,
                                  web_pages=pages_block(web, found[0]) if found else None, checks=checks)
        if vision:
            messages = with_images(messages, images_for(reader_images, [], settings), provider)

        def generate(number: int) -> AsyncIterator[Event]:
            return tool_answer(question, messages, [], settings, find_only, previous, kept, provider, transport,
                               purpose, fallback, client, again(web, number, app_web), propose, not app_web)

        async for event in answered(generate, checks, question, settings, provider, transport, purpose, client, []):
            yield event
        return

    # With an image attached, "this plot" means the image, not the previous turn: no rewrite.
    query = await standalone_question(question, turn, settings, transport) if turn and not reader_images else question
    try:
        carried = cited if adds_context(query, question) else ()
        hits = await run_in_threadpool(find_sources, settings, query, doc_ids, embed, carried)
    except OSError as error:
        yield "error", {"message": f"Search failed: {error}"}
        return
    shown_sources = [source_record(n, hit) for n, hit in enumerate(hits, 1)]
    yield "sources", {"sources": shown_sources, "query": query if query != question else None}

    if tools:
        def find(text: str) -> list[Hit]:
            return find_sources(settings, text, doc_ids, embed)

        previous = turn[-1]["content"] if turn else None
        found = []
        if app_web:
            async for event in app_web_search(web, web_query or query, settings, embed, found):
                yield event
        messages = build_messages(question, hits, history, tools=True, web=web is not None, propose=propose,
                                  web_query=web_query, web_pages=pages_block(web, found[0]) if found else None,
                                  checks=checks)
        if vision:
            messages = with_images(messages, images_for(reader_images, hits, settings), provider)

        def generate(number: int) -> AsyncIterator[Event]:
            return tool_answer(question, messages, hits, settings, find, previous, cited, provider, transport,
                               purpose, fallback, client, again(web, number, app_web), propose, not app_web)

        async for event in answered(generate, checks, question, settings, provider, transport, purpose, client,
                                    shown_sources):
            yield event
        return

    messages = build_messages(question, hits, history, checks=checks)
    if vision:
        messages = with_images(messages, images_for(reader_images, hits, settings), provider)

    def generate(number: int) -> AsyncIterator[Event]:
        return plain_answer(messages, hits, settings, provider, transport, purpose, fallback, client)

    async for event in answered(generate, checks, question, settings, provider, transport, purpose, client,
                                shown_sources):
        yield event


async def plain_answer(messages: list[dict], hits: list[Hit], settings: Settings, provider: str,
                       transport: httpx.AsyncBaseTransport | None, purpose: str, fallback: bool,
                       client=None) -> AsyncIterator[Event]:
    """One answer without tools, as the gold-set eval asks: "thinking", "token", then "done" or "error"."""
    citations = CitationFilter(set(range(1, len(hits) + 1)))
    answer = ""
    finished = None
    fell_back: dict | None = None
    try:
        async for delta in model_reply(messages, settings, provider, transport, purpose, fallback,
                                       client if provider == "openrouter" else None):
            if delta.restart:
                citations = CitationFilter(set(range(1, len(hits) + 1)))
                answer = ""
                fell_back = {"model": delta.model, "from": settings.hosted_model, "reason": delta.reason}
                yield "fallback", fell_back
                continue
            if delta.thinking:
                yield "thinking", {"text": delta.thinking}
            if delta.text and (text := citations.feed(delta.text)):
                answer += text
                yield "token", {"text": text}
            if delta.done:
                finished = delta
    except (httpx.HTTPError, RuntimeError) as error:
        yield "error", {"message": f"The model failed: {error}"}
        return
    if rest := citations.flush():
        answer += rest
        yield "token", {"text": rest}
    if not answer.strip():
        yield "error", no_answer(finished.done_reason if finished else None, settings, provider)
        return
    yield "done", {
        "answer": answer,
        "citations": citations.cited,
        "dropped_citations": citations.dropped,
        "unverified": unverified_crops(citations.cited, hits),
        "done_reason": finished.done_reason if finished else None,
        "prompt_tokens": finished.prompt_tokens if finished else 0,
        "output_tokens": finished.output_tokens if finished else 0,
        "cost_usd": finished.cost_usd if finished else 0.0,
        "model": (finished and finished.model) or model_name(settings, provider),
        "hosted": provider != "ollama",
        "served_by": finished.provider if finished else None,
        "fallback_from": fell_back["from"] if fell_back else None,
        "fallback_reason": fell_back["reason"] if fell_back else None,
    }
