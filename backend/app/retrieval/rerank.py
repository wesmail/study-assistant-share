"""Reranks search candidates with a cross-encoder, which reads question and passage together.

Keyword and vector search each rank passages without reading them against the question,
so a good passage can sit far down the list. The cross-encoder fixes that, but it's too
slow for the whole library, so it rescores only the top candidates.

It runs on the GPU in half precision: about 0.2 s for 30 passages against 26 s on this
CPU, with the same top 6 on all 35 gold questions. It needs about 1.6 GB, which fits
beside gpt-oss at 32K only because the helper runs on the CPU. If it can't run, search keeps the ranking it had and logs why, so a question
is never lost to it.

The model isn't downloaded on demand. Fetch it once with:

    huggingface-cli download BAAI/bge-reranker-v2-m3
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence

from app.config import Settings
from app.retrieval.search import Hit

log = logging.getLogger("rerank")
Scorer = Callable[[str, list[str], Settings], list[float]]


class CrossEncoder:
    """Loads on first use, because torch alone adds seconds to starting the app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: tuple | None = None

    def _load(self, settings: Settings) -> tuple:
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            options = {"revision": settings.rerank_revision, "local_files_only": True}
            tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model, **options)
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.rerank_model, dtype=torch.float16, **options
            )
            self._loaded = (torch, tokenizer, model.eval().to(settings.rerank_device))
        return self._loaded

    def scores(self, query: str, texts: list[str], settings: Settings) -> list[float]:
        with self._lock:
            torch, tokenizer, model = self._load(settings)
            found: list[float] = []
            with torch.no_grad(), torch.autocast(settings.rerank_device, dtype=torch.float16):
                for start in range(0, len(texts), settings.rerank_batch):
                    part = texts[start:start + settings.rerank_batch]
                    # Passages are cut, never the question.
                    tokens = tokenizer([query] * len(part), part, padding=True, truncation="only_second",
                                       max_length=settings.rerank_max_length, return_tensors="pt")
                    logits = model(**tokens.to(settings.rerank_device)).logits
                    found += logits.view(-1).float().tolist()
            return found


cross_encoder = CrossEncoder()


def rerank(query: str, hits: Sequence[Hit], settings: Settings, scorer: Scorer | None = None) -> list[Hit]:
    """`hits` best first. Equal scores keep their order, and a failure keeps them all."""
    if not hits or not settings.rerank_enabled:
        return list(hits)
    try:
        scores = (scorer or cross_encoder.scores)(query, [hit.text for hit in hits], settings)
    except (RuntimeError, OSError, ValueError, ImportError) as error:
        log.warning("Keeping the search order: the reranker failed (%s: %s)", type(error).__name__, error)
        return list(hits)
    for hit, score in zip(hits, scores):
        hit.score = score
    return sorted(hits, key=lambda hit: hit.score, reverse=True)
