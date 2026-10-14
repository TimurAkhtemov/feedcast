import numpy as np

from feedcast.db import Store
from feedcast.group import Cluster, cluster_items, embed_items, shortlist
from feedcast.write import Dropped, Segment, build_pack, outlet_line, target_words, write_segment

from conftest import FakeQwen, KeywordEmbedder


def _items(store, specs):
    for n, (outlet, title, text) in enumerate(specs):
        store.add_item(outlet=outlet, feed_url="f", url=f"https://e.com/{n}", canonical_url=f"https://e.com/{n}",
                       title=title, lead=text[:200], text=text, content_hash=f"h{n}", published=float(n),
                       full_text=len(text) > 100)
    return store.items_since(0)


def test_clusters_rank_by_distinct_outlets():
    store = Store(":memory:")
    items = _items(store, [
        ("A", "Bridge reopens", "The bridge reopened."),
        ("B", "Bridge open again", "The bridge is open."),
        ("B", "Bridge traffic returns", "Bridge traffic is back."),  # same outlet twice
        ("C", "Rover finds clay", "Tessera found clay."),
        ("D", "Tessera clay find", "The tessera rover found clay."),
        ("E", "Tessera update", "Tessera drives north."),
    ])
    vecs = embed_items(store, KeywordEmbedder(), items)
    clusters = cluster_items(items, vecs, 0.75)
    assert [c.size for c in clusters] == [3, 2]  # tessera: C, D, E; bridge: A, B
    assert len(clusters[1].items) == 3


def test_embeddings_are_cached_per_text():
    store = Store(":memory:")
    items = _items(store, [("A", "Bridge reopens", "x")])

    class Counting(KeywordEmbedder):
        calls = 0

        def embed(self, texts):
            Counting.calls += len(texts)
            return super().embed(texts)

    embed_items(store, Counting(), items)
    embed_items(store, Counting(), items)
    assert Counting.calls == 1


def test_shortlist_keeps_nearest():
    store = Store(":memory:")
    items = _items(store, [("A", "Bridge", "bridge"), ("B", "Volcano", "volcano"), ("C", "Lunch", "lunch")])
    vecs = embed_items(store, KeywordEmbedder(), items)
    topic = KeywordEmbedder().embed(["volcano"])[0]
    assert [it.title for it in shortlist(items, vecs, topic, 1)] == ["Volcano"]


def test_outlet_line():
    assert outlet_line(["NPR"]) == "This story comes from NPR."
    assert outlet_line(["NPR", "BBC News"]) == "This story was reported by NPR and BBC News."
    assert outlet_line(list("ABCDEF")) == "This story was reported by A, B, C, D, and 2 other outlets."


def test_targets_stay_within_60_to_120_seconds():
    store = Store(":memory:")
    big = Cluster(_items(store, [(str(i), f"t{i}", "x" * 3000) for i in range(12)]))
    small = Cluster(_items(Store(":memory:"), [("A", "t", "short")]))
    assert 60 <= target_words(big, 150) / 150 * 60 <= 120
    assert 60 <= target_words(small, 150) / 150 * 60 <= 120


def test_pack_puts_one_article_per_outlet_first():
    store = Store(":memory:")
    items = _items(store, [("A", "a1", "alpha " * 300), ("A", "a2", "alpha two " * 300), ("B", "b1", "beta " * 50)])
    pack = build_pack(Cluster(items), max_chars=3000)
    blocks = [line for line in pack.text.splitlines() if line.startswith("=== From")]
    assert blocks == ["=== From A ===", "=== From B ===", "=== From A ==="]
    assert not any(ch.isdigit() for ch in "".join(blocks))
    assert len(pack.links) == 3


def test_ungrounded_segment_is_rewritten_once_then_dropped():
    store = Store(":memory:")
    text = "HALLUCINATE_ALWAYS. The council met on Tuesday to debate parking rules downtown. A vote is due."
    llm = FakeQwen()
    result = write_segment(llm, Cluster(_items(store, [("A", "Council debates parking", text)])), words=60)
    assert isinstance(result, Dropped) and "Zorblax" in result.reason
    assert len(llm.prompts) == 2 and "Zorblax" in llm.prompts[1]


def test_rewrite_that_fixes_grounding_is_kept_with_outlet_line():
    store = Store(":memory:")
    text = "HALLUCINATE_ONCE. The district will offer free lunch to all students in January. It serves 8,200 students."
    result = write_segment(FakeQwen(), Cluster(_items(store, [("Valley Post", "Lunch expands", text)])), words=40)
    assert isinstance(result, Segment) and result.rewrites == 1
    assert result.first_issues and result.paragraphs[-1] == "This story comes from Valley Post."
    assert not any("Zorblax" in p for p in result.paragraphs)


def test_cluster_requires_vectors():
    store = Store(":memory:")
    items = _items(store, [("A", "t", "x")])
    assert cluster_items(items, {}, 0.75) == []
    assert len(cluster_items(items, {items[0].id: np.ones(3)}, 0.75)) == 1


def test_topic_stories_rank_by_relevance_not_size():
    from feedcast.pipeline import by_relevance

    store = Store(":memory:")
    items = _items(store, [
        ("A", "Bridge reopens", "bridge"), ("B", "Bridge open", "bridge"), ("C", "Bridge news", "bridge"),
        ("D", "Volcano sensors", "volcano"),
    ])
    clusters = cluster_items(items, embed_items(store, KeywordEmbedder(), items), 0.75)
    assert clusters[0].size == 3  # by size, the bridge wins
    by_title = {it.title: it.id for it in items}
    relevance = {by_title["Bridge reopens"]: 0.6, by_title["Volcano sensors"]: 1.7}
    ranked = by_relevance(clusters, relevance)
    assert ranked[0].title == "Volcano sensors"
    assert len(ranked) == 2
