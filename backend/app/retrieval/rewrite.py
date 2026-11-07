"""Turns a follow-up into a standalone question, for the search only.

"I don't understand!" matches nothing useful in the library, so the helper model
rewrites it from the previous turn. The answer still gets the reader's own words.
This runs locally in every mode, so it costs nothing and sends nothing out.

A small model sometimes hands back "How can I translate this into code?" with the
"this" still in it, and a search for that finds nothing about the topic. So a message
that points back (this, that, it, these, those...) must come back carrying something
of the previous question; if it doesn't, or the helper fails, the previous question
is added to the search. Any other message the helper can't rewrite is searched as
written.
"""
from __future__ import annotations

import logging
import re

import httpx

from app.config import Settings
from app.llm.citations import strip_citations

log = logging.getLogger("rewrite")
MAX_QUERY_CHARS = 300
SYSTEM_PROMPT = """Rewrite the user's latest message as one standalone question for a search engine, using the previous turn. Replace words like "that", "it" and "this" with the topic they refer to. If the latest message is already about a new topic and makes sense alone, return it exactly as written. Output only the question.

Example
Previous question: What is entropy?
Latest message: what does that mean
Output: What does entropy mean?

Example
Previous question: What is entropy?
Latest message: I'm lost
Output: Explain entropy in simple terms.

Example
Previous question: What loss function does SeismoGPT use?
Latest message: so how do I write this in code?
Output: How do I implement the SeismoGPT loss function in code?

Example
Previous question: What does Figure 4 show about the context ratio?
Latest message: what does that curve show?
Output: What does the curve in Figure 4 show about the context ratio?

Example
Previous question: What is entropy?
Latest message: What is a Fourier transform?
Output: What is a Fourier transform?

Example
Previous question: What is entropy?
Latest message: Prove that the determinant of a product is the product of the determinants.
Output: Prove that the determinant of a product is the product of the determinants."""


# Words that carry no topic, so a rewrite that only adds these brought nothing in.
STOPWORDS = frozenset("""
a an the and or but so to of in into on onto at by for from with without as is are was were be been
being do does did doing can could would should will shall may might must i me my we our you your he
she they them their it its this that these those there here what which who whom whose when where why
how about than then just also very more most some any each all no not only own same such too again
further once please explain show tell give mean means meant really actually now still even like
""".split())
# A word that points back at something earlier. "that" also starts a clause, as in
# "prove that ..." or "such that ...", and then it points at nothing.
POINTING = re.compile(r"\b(this|that|these|those|it|its|they|them|their)\b", re.I)
THAT_STARTS_A_CLAUSE = re.compile(
    r"\b(show|shows|prove|proves|given|such|so|assume|assuming|suppose|note|means|mean|say|says|"
    r"state|states|true|know|verify|check|implies|imply|ensure|ensures)\s+that\b", re.I)


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def topic_words(text: str) -> set[str]:
    return words(text) - STOPWORDS


def points_back(question: str) -> bool:
    """Whether the message leans on something said before, such as "explain this figure"."""
    for match in POINTING.finditer(question):
        if match.group(1).lower() != "that":
            return True
        before = question[:match.end()]
        if not THAT_STARTS_A_CLAUSE.search(before[-40:]):
            return True
    return False


def adds_context(query: str, question: str) -> bool:
    """Whether the rewrite brought a topic in from the previous turn.

    That's what makes a message a follow-up. The helper also tidies new questions,
    dropping "Thanks." or "In one sentence,", and those start clean. Only words that
    carry a topic count: "this to code" coming back as "this into code" brings in nothing.
    """
    return bool(topic_words(query) - topic_words(question))


def resolved(query: str, question: str, previous: str) -> bool:
    """The rewrite carries some of the previous question that the message itself didn't."""
    return bool((topic_words(query) & topic_words(previous)) - topic_words(question))


def with_previous_topic(question: str, previous: str) -> str:
    """The search for a follow-up the helper couldn't resolve: the message and what it points at."""
    return f"{question.strip()} (about: {previous.strip()})"[:MAX_QUERY_CHARS]


def rewrite_payload(question: str, turn: list[dict], settings: Settings) -> dict:
    asked, answered = turn[0]["content"], strip_citations(turn[1]["content"])
    prompt = (f"Previous question: {asked}\n"
              f"Previous answer: {answered[:settings.rewrite_answer_chars]}\n"
              f"Latest message: {question}\nOutput:")
    return {
        "model": settings.helper_model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": settings.helper_keep_alive,
        # On the CPU, because the reranker needs its share of GPU memory.
        "options": {"num_ctx": settings.rewrite_num_ctx, "num_predict": settings.rewrite_num_predict,
                    "temperature": 0, "num_gpu": 0},
    }


def usable(output: str) -> str | None:
    """The first line, without quotes, if it can be a query."""
    lines = output.strip().splitlines()
    line = lines[0].strip().strip("\"'").strip() if lines else ""
    return line if 0 < len(line) <= MAX_QUERY_CHARS else None


def warm_up_helper(settings: Settings) -> None:
    """Load the helper ahead of the first follow-up, which would otherwise wait 15 s."""
    payload = {"model": settings.helper_model, "keep_alive": settings.helper_keep_alive,
               "options": {"num_ctx": settings.rewrite_num_ctx, "num_gpu": 0}}     # or Ollama reloads it
    try:
        httpx.post(f"{settings.ollama_url}/api/generate", json=payload,
                   timeout=settings.helper_timeout_seconds).raise_for_status()
    except httpx.HTTPError as error:
        log.warning("The helper isn't ready (%s)", type(error).__name__)


async def standalone_question(
    question: str,
    turn: list[dict],
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """`turn` is the previous question and answer."""
    query = await helper_rewrite(question, turn, settings, transport)
    previous = turn[0]["content"]
    if points_back(question) and (query is None or not resolved(query, question, previous)):
        log.info("The follow-up points back, and the rewrite didn't say at what: adding the previous question")
        return with_previous_topic(question, previous)
    return query or question


async def helper_rewrite(
    question: str, turn: list[dict], settings: Settings, transport: httpx.AsyncBaseTransport | None,
) -> str | None:
    """What the helper made of the message, or None if it failed or gave nothing usable."""
    timeout = httpx.Timeout(settings.helper_timeout_seconds, connect=5)
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.post(f"{settings.ollama_url}/api/chat",
                                         json=rewrite_payload(question, turn, settings))
        response.raise_for_status()
        return usable(response.json()["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        log.warning("The helper failed (%s)", type(error).__name__)
        return None
