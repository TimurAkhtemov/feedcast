"""Ollama client: Qwen for writing (thinking off, JSON-schema output) and embeddinggemma."""

from __future__ import annotations

import json
import logging
from typing import Protocol

import httpx
import numpy as np

log = logging.getLogger("feedcast.llm")


class LLM(Protocol):
    def json(self, system: str, user: str, schema: dict) -> dict: ...


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OllamaError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str, *, url: str = "http://127.0.0.1:11434",
                 temperature: float = 0.4, num_ctx: int = 16384, timeout: float = 600):
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.client = httpx.Client(timeout=timeout)

    def json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        last_error: Exception | None = None
        for _ in range(2):  # one retry: schema-constrained output is almost always valid JSON
            data = _post(self.client, f"{self.url}/api/chat", body)
            content = data.get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                log.warning("Qwen returned invalid JSON, retrying: %s", content[:200])
        raise OllamaError(f"{self.model} did not return valid JSON: {last_error}")


class OllamaEmbedder:
    def __init__(self, model: str = "embeddinggemma", *, url: str = "http://127.0.0.1:11434",
                 batch: int = 32, timeout: float = 300):
        self.name = model
        self.url = url.rstrip("/")
        self.batch = batch
        self.client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            data = _post(self.client, f"{self.url}/api/embed",
                         {"model": self.name, "input": texts[i : i + self.batch]})
            out.extend(data["embeddings"])
        return np.asarray(out, dtype=np.float32).reshape(len(texts), -1)


def _post(client: httpx.Client, url: str, body: dict) -> dict:
    try:
        resp = client.post(url, json=body)
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama is not reachable at {url}: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"Ollama {url} returned HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()
