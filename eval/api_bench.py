"""Model comparison bench: the gold questions through OpenRouter, for evaluation only.

    python eval/api_bench.py check                 # key present, models and prices as pinned
    python eval/api_bench.py answers [--limit N]   # all MODELS, saved as answers-RUN.jsonl
    python eval/api_bench.py spent

The app never imports this; it shares the protocol pieces and the key reader of the app's
own OpenRouter client, app/llm/openrouter.py. The key is read only from
STUDY_ASSISTANT_OPENROUTER_KEY in .env, never from the process environment. It is never
printed, logged, stored or put in an error message, and nothing here sets an
OPENAI_API_KEY.

Every question is retrieved once, and the same messages go to every model, built by the
app's own find_sources and build_messages with no history and no tools, as in
run_eval.py's `answers`. The answer passes through the app's CitationFilter. Records have
the same fields as run_eval's, so its blind-pack and blind-score read them unchanged, and
blind-pack checks again that every run had the same chunks.

Each model is pinned to one provider with fallbacks off, because providers serve
different quantizations at different prices. Sampling copies the local Ollama model files.
Every call is logged in hosted_calls with its tokens and billed cost, failures too.
Before each question, the spend so far plus the worst case of every call still to make
for it must stay under the cap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from app.chat import find_sources, source_record, unverified_crops  # noqa: E402
from app.config import ENV_FILE, OPENROUTER_KEY, load_settings, openrouter_api_key  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.llm.citations import CitationFilter  # noqa: E402
from app.llm.hosted import Tokens, record_call  # noqa: E402
from app.llm.openrouter import API, describe, sse_events  # noqa: E402
from app.llm.prompts import build_messages  # noqa: E402

CLIENT_NAME = "study-assistant-eval"
SPEND_CAP_USD = 3.0
MAX_TOKENS = 16000          # as hosted_max_tokens for Opus; a "length" stop is reported
PURPOSE = "eval:api-bench"
MILLION = 1_000_000
CHARS_PER_TOKEN = 3         # for the worst case only: fewer than the true ~4, so it overestimates


@dataclass(frozen=True)
class Model:
    run: str                # answers-RUN.jsonl
    id: str                 # OpenRouter model id
    provider: str           # the one provider allowed, as OpenRouter names it
    quantization: str
    prices: tuple[float, float]     # dollars per million input and output tokens, at that provider
    sampling: dict          # from `ollama show --parameters` of the local model
    reasoning: dict         # thinking on, as in the local runs


# Checked against https://openrouter.ai/api/v1/models/<id>/endpoints on 2026-11-03. `check`
# compares them again before a run. gpt-oss was released in MXFP4, so fp4 is its own format.
MODELS = (
    Model("api-qwen", "qwen/qwen3.8-27b", "DeepInfra", "bf16", (0.15, 1.875),
          {"temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0, "presence_penalty": 0},
          {"enabled": True}),
    Model("api-gemma", "google/gemma-4-26b-a4b-it", "CoreWeave", "bf16", (0.10, 0.30),
          {"temperature": 1, "top_k": 64, "top_p": 0.95},
          {"enabled": True}),
    Model("api-gpt-oss", "openai/gpt-oss-20b", "CoreWeave", "fp4", (0.03, 0.13),
          {"temperature": 1},
          {"effort": "medium"}),      # chat_think for local gpt-oss
)


class BenchFailure(RuntimeError):
    pass


def api_key(path: Path = ENV_FILE) -> str | None:
    """From the .env file and nowhere else, like hosted_api_key. Pass it straight to the client."""
    return openrouter_api_key(path)


def request_for(model: Model, messages: list[dict]) -> dict:
    return {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "reasoning": model.reasoning,
        "usage": {"include": True},
        # One provider, no fallback, and none that keeps or trains on prompts: they hold book text.
        "provider": {"order": [model.provider], "allow_fallbacks": False, "data_collection": "deny",
                     "quantizations": [model.quantization]},
        **model.sampling,
    }


def cost_usd(tokens: Tokens, prices: tuple[float, float]) -> float:
    """Reasoning tokens are billed as output, and OpenRouter counts them in completion_tokens."""
    return (tokens.input * prices[0] + tokens.output * prices[1]) / MILLION


def worst_case_usd(model: Model, messages: list[dict]) -> float:
    characters = sum(len(message["content"]) for message in messages)
    return cost_usd(Tokens(input=characters // CHARS_PER_TOKEN + 1, output=MAX_TOKENS), model.prices)


def spent_usd(db_path: Path) -> float:
    """Everything this bench has cost, across runs, failures included."""
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT SUM(cost_usd) FROM hosted_calls WHERE purpose = ?", (PURPOSE,)).fetchone()[0]
        return total or 0.0
    finally:
        conn.close()


async def answer(model: Model, messages: list[dict], provided: set[int], client: httpx.AsyncClient,
                 key: str, settings) -> dict:
    """One answer, with the fields run_eval's records have. Logged whether it works or not."""
    started = time.monotonic()
    first_output = first_token = None
    tokens = Tokens()
    billed = None
    finish = provider = error = None
    citations = CitationFilter(provided)
    text = ""
    try:
        async with client.stream("POST", f"{API}/chat/completions", json=request_for(model, messages),
                                 headers={"Authorization": f"Bearer {key}", "X-Title": CLIENT_NAME}) as response:
            if response.status_code != 200:
                raise BenchFailure(describe(response.status_code, await response.aread()))
            async for event in sse_events(response.aiter_lines()):
                if "error" in event:
                    raise BenchFailure(describe(event["error"].get("code", 0), json.dumps(event).encode()))
                provider = event.get("provider") or provider
                if usage := event.get("usage"):
                    tokens = Tokens(input=usage.get("prompt_tokens") or 0, output=usage.get("completion_tokens") or 0)
                    billed = usage.get("cost")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if (delta.get("reasoning") or delta.get("reasoning_details")) and first_output is None:
                        first_output = time.monotonic() - started
                    if piece := delta.get("content"):
                        if first_token is None:
                            first_token = time.monotonic() - started
                        text += citations.feed(piece)
                    finish = choice.get("finish_reason") or finish
        text += citations.flush()
        if provider and provider != model.provider:
            raise BenchFailure(f"served by {provider}, not the pinned {model.provider}")
        if not text.strip():
            raise BenchFailure(f"no answer ({finish or 'stream ended early'})")
    except (BenchFailure, httpx.HTTPError, ValueError) as failure:
        error = str(failure) if isinstance(failure, BenchFailure) else type(failure).__name__
    finally:
        cost = billed if billed is not None else cost_usd(tokens, model.prices)
        record_call(replace(settings, hosted_model=model.id), PURPOSE, tokens, cost,
                    time.monotonic() - started, finish, error)
    if error:
        return {"error": error, "cost_usd": cost}
    return {
        "answer": text, "citations": citations.cited, "dropped_citations": citations.dropped,
        "done_reason": finish, "prompt_tokens": tokens.input, "output_tokens": tokens.output,
        "cost_usd": cost, "cost_billed": billed is not None, "model": model.id, "provider": provider,
        "hosted": True, "first_output_seconds": first_output or first_token,
        "first_token_seconds": first_token, "total_seconds": time.monotonic() - started,
    }


async def endpoint_problems(client: httpx.AsyncClient) -> list[str]:
    """Each model's pinned provider is still listed, at the quantization and prices assumed here."""
    problems = []
    for model in MODELS:
        response = await client.get(f"{API}/models/{model.id}/endpoints")
        if response.status_code != 200:
            problems.append(f"{model.id}: HTTP {response.status_code}")
            continue
        found = [e for e in response.json()["data"]["endpoints"] if e.get("provider_name") == model.provider]
        if not found:
            problems.append(f"{model.id}: {model.provider} no longer serves it")
            continue
        endpoint = found[0]
        prices = tuple(round(float(endpoint["pricing"][part]) * MILLION, 6) for part in ("prompt", "completion"))
        if prices != model.prices:
            problems.append(f"{model.id}: {model.provider} now charges {prices}, not {model.prices}")
        if endpoint.get("quantization") != model.quantization:
            problems.append(f"{model.id}: {model.provider} now serves {endpoint.get('quantization')}")
    return problems


async def cmd_check(args) -> None:
    key = api_key()
    print(f"{OPENROUTER_KEY} set in .env: {bool(key)}")
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": CLIENT_NAME}) as client:
        problems = await endpoint_problems(client)
        if key:
            response = await client.get(f"{API}/key", headers={"Authorization": f"Bearer {key}"})
            print(f"Key accepted: {response.status_code == 200}")
    for model in MODELS:
        print(f"{model.run}: {model.id} at {model.provider} ({model.quantization}), "
              f"${model.prices[0]}/${model.prices[1]} per million in/out")
    print("\n".join(problems) or "Endpoints, quantizations and prices as pinned.")
    print(f"Spent so far: ${spent_usd(load_settings().db_path):.4f} of ${SPEND_CAP_USD:.2f}")


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {record["id"]: record for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


async def cmd_answers(args) -> None:
    import run_eval

    key = api_key()
    if not key:
        raise SystemExit(f"{OPENROUTER_KEY} is not set in .env")
    settings = load_settings()
    conn = connect(settings.db_path, apply_schema=False)
    library = {path.name for path in settings.library_dir.glob("*.pdf")}
    gold, _ = run_eval.load_gold(run_eval.QUESTIONS, library)
    paths = {model.run: run_eval.RESULTS / f"answers-{model.run}.jsonl" for model in MODELS}
    done = {run: load_done(path) for run, path in paths.items()}
    asked = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.chat_timeout_seconds, connect=15),
                                 headers={"User-Agent": CLIENT_NAME}) as client:
        if problems := await endpoint_problems(client):
            raise SystemExit("Not run, the endpoints changed:\n" + "\n".join(problems))
        for question in gold:
            waiting = [model for model in MODELS if question["id"] not in done[model.run]]
            if not waiting:
                continue
            if args.limit is not None and asked >= args.limit:
                break
            hits = find_sources(settings, question["question"], None, run_eval.embed_texts)
            messages = build_messages(question["question"], hits, [])
            spent = spent_usd(settings.db_path)
            worst = sum(worst_case_usd(model, messages) for model in waiting)
            if spent + worst > SPEND_CAP_USD:
                print(f"Stopped at ${spent:.4f}: {question['id']} could cost up to ${worst:.4f}, "
                      f"and the cap is ${SPEND_CAP_USD:.2f}", flush=True)
                break
            sources = [source_record(n, hit) for n, hit in enumerate(hits, 1)]
            provided = set(range(1, len(hits) + 1))
            results = await asyncio.gather(*(answer(model, messages, provided, client, key, settings)
                                             for model in waiting))
            asked += 1
            for model, result in zip(waiting, results):
                if result.get("error"):
                    print(f"{question['id']} {model.run}: FAILED, not saved: {result['error']}", flush=True)
                    continue
                record = {"id": question["id"], "question": question["question"], "sources": sources, **result}
                record["unverified"] = unverified_crops(record["citations"], hits)
                record["hit"] = run_eval.first_hit(
                    [(run_eval.document_path(conn, s), s["page_start"], s["page_end"]) for s in sources],
                    question["expected_sources"])
                record["copied_flagged"] = run_eval.flag_report(record, conn)
                with open(paths[model.run], "a", encoding="utf-8") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[model.run][question["id"]] = record
                print(f"{question['id']} {model.run}: {result['done_reason']}, {len(record['citations'])} cited, "
                      f"{result['total_seconds']:.0f}s, ${result['cost_usd']:.4f}", flush=True)
    print(f"Spent so far: ${spent_usd(settings.db_path):.4f} of ${SPEND_CAP_USD:.2f}")


def cmd_spent(args) -> None:
    print(f"${spent_usd(load_settings().db_path):.4f} of ${SPEND_CAP_USD:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check").set_defaults(run=lambda a: asyncio.run(cmd_check(a)))
    answers = commands.add_parser("answers")
    answers.add_argument("--limit", type=int, help="at most this many questions with work left")
    answers.set_defaults(run=lambda a: asyncio.run(cmd_answers(a)))
    commands.add_parser("spent").set_defaults(run=cmd_spent)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
