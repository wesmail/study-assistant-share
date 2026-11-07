"""App-driven web search for the local models.

In the gate run the local models used web_search and fetch_page poorly: Balanced searched
without opening a page and answered from snippets or memory. So for an option with
web: app, the app does the searching and the model only answers:

    1. one search, for the question (or the rewritten follow-up, or the query the reader
       agreed to)
    2. the reader's own links first, then the top results, opened in parallel until
       TOP_PAGES pages could be read
    3. from each page, the passages most relevant to the question, up to EXCERPT_CHARS,
       in page order: the model reads them as [Web n]

All of WebSession's guards still hold, since every search and fetch goes through it: only
public hosts, redirects checked, hidden text dropped, markers, and the copy check on
run_python and sympy_check.

Choosing passages. A page is cut into windows of about WINDOW_CHARS characters, on
paragraph breaks where it can. Each window is ranked twice, by BM25 against the question
and by cosine similarity of its embedding to the question's, and the two ranks are fused
as the library search fuses them:

    score(w) = 1 / (k + rank_bm25(w)) + 1 / (k + rank_cosine(w))

BM25, over the windows of all pages read for this question:

    bm25(w) = sum over query terms t of  idf(t) * f(t, w) * (k1 + 1) / (f(t, w) + k1 * (1 - b + b * |w| / avgdl))
    idf(t)  = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

The first window always stays, since a page's opening usually says what it is (a release,
a paper, a table of values).
"""
from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.tools.web import MAX_FETCHES, Blocked, Page, Result, WebSession, urls_in

TOP_PAGES = 3
EXCERPT_CHARS = 8000            # per page: three pages are about 6K tokens of a 32K context
MIN_PAGE_CHARS = 400            # less is a form, a video or a link list: try the next result
WINDOW_CHARS = 900
QUERY_CHARS = 300               # longer questions are cut for the search engine
BM25_K1 = 1.2
BM25_B = 0.75
GAP = "[…]"
TERM = re.compile(r"[^\W_]+(?:[.'][^\W_]+)*", re.U)
STOPWORDS = frozenset("""
a an and are as at be by did do does for from has have how i in is it its my of on or the this to was
were what when where which who whom why with what's me you your about into than then there these
those their them they we our can could would should will may might must also any all
""".split())


@dataclass
class Gathered:
    """What the app found on the web for one question."""

    query: str
    pages: list[Page] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)     # (url, why)
    search_error: str | None = None
    results: int = 0


NO_TEXT_LINK = re.compile(r"\.pdf(?:$|[?#])|/pdf/|get_pdf|view=pdf|//(?:www\.|m\.)?(?:youtube\.com|youtu\.be|vimeo\.com)/", re.I)


def ranked_results(results: Sequence[Result], question: str, k: int = 60) -> list[str]:
    """The result links in the order to open them: the search's own order fused with how well
    each title, snippet and link matches the question (BM25), as the passages are ranked.
    Links with no readable text (PDFs, videos) go last: each would use up a fetch for nothing.
    """
    described = [f"{result.title} {result.snippet} {' '.join(re.split(r'[/._-]+', result.url))}" for result in results]
    matched = ranks(bm25(question, described))
    score = {result.url: 1 / (k + place) + 1 / (k + matched[place - 1]) for place, result in enumerate(results, 1)}
    order = sorted(score, key=lambda url: (bool(NO_TEXT_LINK.search(url)), -score[url]))
    return order


def search_query(question: str) -> str:
    """The question as one line, cut on a word boundary to what a search engine takes."""
    text = " ".join(question.split())
    if len(text) <= QUERY_CHARS:
        return text
    return text[:QUERY_CHARS].rsplit(" ", 1)[0]


def terms(text: str) -> list[str]:
    return [word for word in TERM.findall(text.lower()) if word not in STOPWORDS]


def windows(text: str, size: int = WINDOW_CHARS) -> list[str]:
    """The text in pieces of about `size` characters: whole paragraphs where they fit, a long
    paragraph cut at sentence ends, and a sentence longer than `size` cut where it must be."""
    pieces: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n|\n", text)):
        if not paragraph:
            continue
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            pieces.extend(sentence[i:i + size] for i in range(0, len(sentence), size))
    out: list[str] = []
    for piece in pieces:
        if out and len(out[-1]) + 1 + len(piece) <= size:
            out[-1] = f"{out[-1]}\n{piece}"
        else:
            out.append(piece)
    return out


def bm25(query: str, docs: Sequence[str]) -> list[float]:
    """BM25 score of each doc for the query; the docs are the whole collection."""
    wanted = set(terms(query))
    counts = [Counter(terms(doc)) for doc in docs]
    if not docs or not wanted:
        return [0.0] * len(docs)
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) or 1.0
    holding = {term: sum(1 for count in counts if term in count) for term in wanted}
    idf = {term: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for term, n in holding.items()}
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for term in wanted:
            f = count.get(term, 0)
            if f:
                score += idf[term] * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * length / average))
        scores.append(score)
    return scores


def ranks(scores: Sequence[float]) -> list[int]:
    """1 for the highest score; ties keep page order."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = [0] * len(scores)
    for place, i in enumerate(order, 1):
        rank[i] = place
    return rank


def fused(bm25_scores: Sequence[float], cosines: Sequence[float] | None, k: int = 60) -> list[float]:
    by_words = ranks(bm25_scores)
    if cosines is None:
        return [1 / (k + r) for r in by_words]
    by_meaning = ranks(cosines)
    return [1 / (k + a) + 1 / (k + b) for a, b in zip(by_words, by_meaning)]


def excerpt(pieces: Sequence[str], scores: Sequence[float], budget: int = EXCERPT_CHARS) -> str:
    """The first piece and then the best ones while they fit, in page order, with [...] for gaps."""
    if sum(len(piece) + 1 for piece in pieces) <= budget:
        return "\n".join(pieces)
    chosen = {0}
    used = len(pieces[0])
    for i in sorted(range(1, len(pieces)), key=lambda i: -scores[i]):
        if used + len(pieces[i]) + 1 > budget:
            continue
        chosen.add(i)
        used += len(pieces[i]) + 1
    out: list[str] = []
    for i in sorted(chosen):
        if out and i - 1 not in chosen:
            out.append(GAP)
        out.append(pieces[i])
    if max(chosen) < len(pieces) - 1:
        out.append(GAP)
    return "\n".join(out)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))        # embeddings come L2-normalised


def choose_passages(pages: list[Page], question: str, embed: Callable[[list[str]], list[list[float]]] | None,
                    budget: int = EXCERPT_CHARS) -> None:
    """Set each page's excerpt from the passages most relevant to the question.

    BM25's statistics come from the windows of every page read, so a word all pages share
    counts for little. Without `embed`, or if embedding fails, BM25 alone ranks them.
    """
    split = [windows(page.text) for page in pages]
    every = [piece for pieces in split for piece in pieces]
    if not every:
        return
    words = bm25(question, every)
    meaning: list[float] | None = None
    if embed is not None:
        try:
            vectors = embed([question, *every])
            meaning = [cosine(vectors[0], vector) for vector in vectors[1:]]
        except (OSError, RuntimeError, ValueError):
            meaning = None
    start = 0
    for page, pieces in zip(pages, split):
        end = start + len(pieces)
        scores = fused(words[start:end], meaning[start:end] if meaning else None)
        page.excerpt = excerpt(pieces, scores, budget) if pieces else ""
        start = end


def pages_block(web: WebSession, found: Gathered) -> str:
    """The pages as the model reads them, after the library passages; or why there are none."""
    if found.pages:
        return (f'Web pages, found by the app\'s web search for "{found.query}" ({web.today}):\n\n'
                + "\n\n".join(web.page_text(page) for page in found.pages))
    why = found.search_error or ("none of the results could be read" if found.results else "it found nothing")
    return (f'Web pages: none. The app searched the web for "{found.query}", but {why}. Tell the reader '
            "that the web search found nothing usable, and answer from the passages if they help.")


async def gather(web: WebSession, query: str, embed: Callable[[list[str]], list[list[float]]] | None = None,
                 top: int = TOP_PAGES) -> Gathered:
    """Search once, read up to `top` pages, and choose their passages for the question."""
    found = Gathered(search_query(query))
    reader_links = sorted(urls_in(web.reader_message))
    candidates = list(reader_links)
    try:
        await web.search(found.query)
    except Blocked as error:
        found.search_error = str(error)
    found.results = len(web.results)
    candidates += [url for url in ranked_results(web.results, query) if url not in candidates]

    rank = {url: place for place, url in enumerate(candidates)}
    position = 0
    while len(web.pages) < top and position < len(candidates) and web.fetches < MAX_FETCHES:
        batch = candidates[position:position + min(top - len(web.pages), MAX_FETCHES - web.fetches)]
        position += len(batch)
        outcomes = await asyncio.gather(*(web.fetch(url) for url in batch), return_exceptions=True)
        for url, outcome in zip(batch, outcomes):
            if isinstance(outcome, Blocked):
                found.failed.append((url, str(outcome)))
            elif isinstance(outcome, BaseException):
                found.failed.append((url, f"the page couldn't be read ({type(outcome).__name__})"))
            elif len(outcome.text) < MIN_PAGE_CHARS and url not in reader_links:
                web.pages.remove(outcome)       # it stays in the copy check
                found.failed.append((url, "too little text to use"))
    # Fetched in parallel, so numbered as they arrived: number them in the order chosen instead.
    web.pages.sort(key=lambda page: rank.get(page.url, len(rank)))
    for number, page in enumerate(web.pages, 1):
        page.number = number
    found.pages = list(web.pages)
    if found.pages:
        await asyncio.to_thread(choose_passages, found.pages, query, embed)
    return found
