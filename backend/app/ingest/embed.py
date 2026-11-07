"""bge-m3 embeddings through Ollama, pinned to the CPU to leave the GPU to the chat model."""
from __future__ import annotations

import json
import struct
import urllib.request

from app.config import Settings


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Return one vector per text. Ollama L2-normalises them, so cosine = dot."""
    payload = json.dumps({
        "model": settings.embed_model,
        "input": texts,
        "options": {"num_gpu": 0},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.ollama_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=settings.embed_timeout_seconds) as response:
        vectors = json.load(response)["embeddings"]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def serialize(vector: list[float], dimensions: int) -> bytes:
    if len(vector) != dimensions:
        raise ValueError(f"Expected a {dimensions}-dim vector, got {len(vector)}")
    return struct.pack(f"{dimensions}f", *vector)
