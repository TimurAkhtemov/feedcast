"""Yes/no judgments from Laya, batched across items, cached, and logged.

Designed around Laya's measured limits:
- Only yes/no (noul) questions, one per topic. Never a choice among topics or feeds,
  because confidence is not calibrated above ~10 options.
- Questions are phrased positively; negation is handled badly.
- Input is the title plus the cleaned opening paragraph, fitted to the 512-token budget
  (question included) here, with real tokenizer counts. Laya would otherwise truncate
  silently.
- English only, and boilerplate is stripped at ingest because unrelated text in the
  input changes answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

from feedcast.db import Item, Store

log = logging.getLogger("feedcast.judge")

# A measured, known-good checkpoint. Do not change without asking.
LAYA_REPO = "aac6fef/laya-mlx"
LAYA_REVISION = "047678560251f28113ee8f5df4be82102c7bf336"
CACHE_LIMIT_BYTES = 512 * 1024**2

AD_QUESTION = "Is this article mainly an advertisement or sponsored post?"

Job = tuple[dict, str]  # (state, yes/no question)


def topic_question(topic: str) -> str:
    return f"Is this article about {topic.strip().rstrip('?.')}?"


class Backend(Protocol):
    name: str

    def noul_many(self, jobs: list[Job]) -> list[float]: ...

    def fit(self, text: str, question: str) -> tuple[str, bool]:
        """Trim `text` so state + question fit the model's input. Returns (text, truncated)."""
        ...


class LayaBackend:
    """Pinned Laya checkpoint on Metal, batching questions across items.

    `Agent.predict` only batches
    questions within one state, so this prepares every (item, question) sequence with the
    agent's own tokenizer, runs them in shared length-sorted batches, and applies the same
    temperature scaling `Agent.predict` uses for noul answers.
    """

    name = "laya"

    def __init__(self, batch_size: int = 16, dtype: str = "float16"):
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        import laya_mlx as laya
        import mlx.core as mx
        import numpy as np
        from laya_mlx.agent import collate_items
        from laya_mlx.common import build_prefix, serialize_state, temp_bucket

        # MLX keeps freed GPU buffers for reuse. Every batch has a different padded shape,
        # so without a cap that cache grew to ~25 GB after 540 questions (model: 0.8 GB).
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        self._np = np
        self._collate = collate_items
        self._temp_bucket = temp_bucket
        self._build_prefix = build_prefix
        self._serialize = serialize_state
        self.agent = laya.load(
            LAYA_REPO, revision=LAYA_REVISION, dtype=dtype, device="gpu", batch_size=batch_size
        )
        self.batch_size = batch_size
        self.max_len = self.agent.cfg.get("max_len", 512)
        self.head_max_len = self.agent.cfg.get("head_max_len", 192)
        self._room: dict[str, int] = {}

    def _state_tokens(self, text: str) -> int:
        tok = self.agent.tok
        serialized = self._serialize(make_state(text)).replace(tok.mask_token, " ")
        return len(tok(serialized)["input_ids"])

    def fit(self, text: str, question: str) -> tuple[str, bool]:
        if question not in self._room:
            q = self.agent._to_internal({"type": "noul", "instructions": question})
            prefix, _ = self._build_prefix(self.agent.tok, q, self.head_max_len)
            self._room[question] = self.max_len - len(prefix) - 1
        room = self._room[question]
        if self._state_tokens(text) <= room:
            return text, False
        words = text.split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._state_tokens(" ".join(words[:mid])) <= room:
                lo = mid
            else:
                hi = mid - 1
        return " ".join(words[:lo]), True

    def noul_many(self, jobs: list[Job]) -> list[float]:
        np, agent = self._np, self.agent
        items = []
        for state, question in jobs:
            prepared, _ = agent.prepare(state, {"q": {"type": "noul", "instructions": question}})
            items.append(prepared[0])
        out: list[float] = [0.0] * len(items)
        for idx in self._group_by_length([len(item["ids"]) for item in items]):
            chunk = [items[i] for i in idx]
            batch = self._collate(
                chunk, agent.tok.pad_token_id, pad_to_multiple=agent.pad_to_multiple,
                max_length=self.max_len,
            )
            logits, _ = agent.forward(batch)
            logits = np.asarray(logits)
            if not np.isfinite(logits).all():
                raise FloatingPointError("Non-finite model outputs")
            for row, (i, item) in enumerate(zip(idx, chunk)):
                k, qt = len(item["markers"]), item["qtype"]
                scale = agent.temperature_by_options.get(self._temp_bucket(qt, k), agent.temperature[qt])
                z = logits[row, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                out[i] = round(float(p[1]), 4)
        return out

    def _group_by_length(self, lengths: list[int]) -> list[list[int]]:
        """Length-sorted batches; a new batch starts when padding would grow past ~25%."""
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        groups: list[list[int]] = []
        for i in order:
            group = groups[-1] if groups else None
            if group and len(group) < self.batch_size and lengths[i] <= lengths[group[0]] * 1.25 + 8:
                group.append(i)
            else:
                groups.append([i])
        return groups


def make_state(text: str) -> dict:
    return {"article": text}


def judge_text(item: Item) -> str:
    """Title + the cleaned opening paragraph: what Laya sees for every question."""
    lead = item.lead.strip()
    if not lead or lead == item.title:
        return item.title
    return f"{item.title}\n\n{lead}"


def answer_key(state: dict, question: str) -> str:
    """Cache by exact input text and exact question, so rewording a question re-asks it."""
    blob = json.dumps(state, sort_keys=True, ensure_ascii=False) + "\x00" + question
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Verdicts:
    question: str
    threshold: float
    p: dict[int, float]

    @property
    def yes(self) -> set[int]:
        return {i for i, p in self.p.items() if p >= self.threshold}


class Judge:
    def __init__(self, store: Store, backend: Backend, *, run_id: str | None = None,
                 chunk: int = 512):
        self.store = store
        self.backend = backend
        self.run_id = run_id
        self.chunk = chunk

    def ask(self, items: list[Item], question: str, threshold: float) -> Verdicts:
        prepared = []
        for item in items:
            text, truncated = self.backend.fit(judge_text(item), question)
            state = make_state(text)
            prepared.append((item, state, answer_key(state, question), truncated))
        cached = self.store.cached_answers([key for _, _, key, _ in prepared])
        todo = [(state, key) for _, state, key, _ in prepared if key not in cached]
        fresh: dict[str, float] = {}
        for start in range(0, len(todo), self.chunk):
            part = todo[start : start + self.chunk]
            ps = self.backend.noul_many([(state, question) for state, _ in part])
            batch = {key: p for (_, key), p in zip(part, ps)}
            self.store.cache_answers(batch)
            fresh.update(batch)
        if todo:
            log.info("laya: %d new answers, %d cached for %r", len(fresh), len(cached), question)

        result: dict[int, float] = {}
        rows = []
        for item, _, key, truncated in prepared:
            p = fresh[key] if key in fresh else cached[key]
            result[item.id] = p
            rows.append({
                "run_id": self.run_id,
                "item_id": item.id,
                "question": question,
                "p": p,
                "threshold": threshold,
                "verdict": int(p >= threshold),
                "cached": int(key not in fresh),
                "truncated": int(truncated),
                "backend": self.backend.name,
            })
        self.store.log_judgments(rows)
        return Verdicts(question, threshold, result)

    def ads(self, items: list[Item], threshold: float) -> Verdicts:
        return self.ask(items, AD_QUESTION, threshold)

    def about(self, items: list[Item], topic: str, threshold: float) -> Verdicts:
        return self.ask(items, topic_question(topic), threshold)
