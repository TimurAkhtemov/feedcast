"""Cluster near-duplicate coverage so ten outlets on one event become one story.

Items are embedded with embeddinggemma (cached per item and text) and grouped by
average-link agglomerative clustering on cosine distance. Average link, unlike single
link, doesn't chain loosely related stories into one blob. A story's importance is the
number of distinct outlets in its cluster.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from feedcast.db import Item, Store
from feedcast.llm import Embedder

EMBED_LEAD_CHARS = 700


def document_text(item: Item) -> str:
    # EmbeddingGemma's document prompt format.
    return f"title: {item.title} | text: {item.lead[:EMBED_LEAD_CHARS]}"


def query_text(topic: str) -> str:
    return f"task: search result | query: {topic}"


def embed_items(store: Store, embedder: Embedder, items: list[Item]) -> dict[int, np.ndarray]:
    """Unit vectors per item id, computing only the ones not cached for this text."""
    texts = {it.id: document_text(it) for it in items}
    hashes = {i: hashlib.sha256(t.encode()).hexdigest()[:16] for i, t in texts.items()}
    cached = store.embeddings(embedder.name, list(texts))
    out = {i: v for i, (h, v) in cached.items() if h == hashes[i]}
    missing = [i for i in texts if i not in out]
    if missing:
        vecs = _unit(embedder.embed([texts[i] for i in missing]))
        store.save_embeddings(embedder.name, [(i, hashes[i], v) for i, v in zip(missing, vecs)])
        out.update(zip(missing, vecs))
    return out


def _unit(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


@dataclass
class Cluster:
    items: list[Item]
    outlets: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Full-text, longer articles first: they're the best material for the writer.
        self.items.sort(key=lambda it: (not it.full_text, -len(it.text)))
        seen: list[str] = []
        for it in self.items:
            if it.outlet not in seen:
                seen.append(it.outlet)
        self.outlets = seen

    @property
    def size(self) -> int:
        return len(self.outlets)

    @property
    def latest(self) -> float:
        return max(it.published for it in self.items)

    @property
    def title(self) -> str:
        return self.items[0].title

    @property
    def ids(self) -> set[int]:
        return {it.id for it in self.items}


def cluster_items(items: list[Item], vecs: dict[int, np.ndarray],
                  similarity: float = 0.8) -> list[Cluster]:
    """Clusters ranked by importance: distinct outlets, then items, then recency."""
    items = [it for it in items if it.id in vecs]
    if not items:
        return []
    if len(items) == 1:
        return [Cluster(items)]
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    matrix = np.stack([vecs[it.id] for it in items])
    dist = np.clip(pdist(matrix, metric="cosine"), 0.0, 2.0)
    labels = fcluster(linkage(dist, method="average"), t=1.0 - similarity, criterion="distance")
    groups: dict[int, list[Item]] = {}
    for it, label in zip(items, labels):
        groups.setdefault(int(label), []).append(it)
    clusters = [Cluster(g) for g in groups.values()]
    clusters.sort(key=lambda c: (c.size, len(c.items), c.latest), reverse=True)
    return clusters


def shortlist(items: list[Item], vecs: dict[int, np.ndarray], topic_vec: np.ndarray,
              k: int) -> list[Item]:
    """The k items closest to a topic: a cheap recall step before Laya judges precision."""
    items = [it for it in items if it.id in vecs]
    if len(items) <= k:
        return items
    q = topic_vec / max(np.linalg.norm(topic_vec), 1e-12)
    scores = np.stack([vecs[it.id] for it in items]) @ q
    order = np.argsort(-scores)[:k]
    return [items[i] for i in order]
