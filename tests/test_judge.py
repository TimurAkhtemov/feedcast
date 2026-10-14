import platform

import pytest

from feedcast.db import Store
from feedcast.judge import AD_QUESTION, Judge, LAYA_REPO, LAYA_REVISION, judge_text, topic_question

from conftest import KeywordLaya


def _item(store, n, title, lead):
    store.add_item(outlet="Example", feed_url="f", url=f"https://e.com/{n}",
                   canonical_url=f"https://e.com/{n}", title=title, lead=lead, text=lead,
                   content_hash=f"h{n}", published=1.0)
    return store.items_since(0)[0]


def test_laya_pin_is_unchanged():
    assert LAYA_REPO == "aac6fef/laya-mlx"
    assert LAYA_REVISION == "047678560251f28113ee8f5df4be82102c7bf336"


def test_questions_are_positive_yes_no():
    assert AD_QUESTION == "Is this article mainly an advertisement or sponsored post?"
    assert topic_question("battery recycling.") == "Is this article about battery recycling?"


def test_judge_caches_and_logs_every_judgment():
    store = Store(":memory:")
    a = _item(store, 1, "Sponsored: big sale", "Shop now and save.")
    b = _item(store, 2, "Senate passes bill", "The Senate voted on Tuesday.")
    backend = KeywordLaya()
    judge = Judge(store, backend, run_id="t1")

    first = judge.ads([a, b], 0.8)
    assert first.yes == {a.id}
    assert backend.calls == 2
    second = judge.ads([a, b], 0.8)
    assert second.p == first.p and backend.calls == 2  # served from cache

    rows = store.db.execute(
        "SELECT item_id, question, threshold, verdict, cached, backend FROM judgments ORDER BY rowid"
    ).fetchall()
    assert len(rows) == 4
    assert rows[0][1:] == (AD_QUESTION, 0.8, 1, 0, "stub-laya")
    assert [r[4] for r in rows] == [0, 0, 1, 1]


def test_rewording_a_question_reasks():
    store = Store(":memory:")
    item = _item(store, 1, "Battery recycling plant opens", "Lithium recovered.")
    backend = KeywordLaya()
    judge = Judge(store, backend)
    judge.about([item], "battery recycling", 0.5)
    judge.about([item], "battery recycling plants", 0.5)
    assert backend.calls == 2


def test_judge_text_is_title_plus_lead():
    store = Store(":memory:")
    item = _item(store, 1, "Title here", "Opening paragraph.")
    assert judge_text(item) == "Title here\n\nOpening paragraph."


@pytest.mark.skipif(platform.machine() != "arm64", reason="needs Apple silicon + MLX")
def test_real_laya_fits_the_512_token_budget():
    pytest.importorskip("laya_mlx")
    import os
    if os.environ.get("FEEDCAST_REAL_LAYA") != "1":
        pytest.skip("set FEEDCAST_REAL_LAYA=1 to load the real model")
    from feedcast.judge import LayaBackend, make_state

    backend = LayaBackend(batch_size=4)
    long_text = "The council met to discuss the budget. " * 200
    fitted, truncated = backend.fit(long_text, AD_QUESTION)
    assert truncated and len(fitted) < len(long_text)
    items, _ = backend.agent.prepare(make_state(fitted), {"q": {"type": "noul", "instructions": AD_QUESTION}})
    assert len(items[0]["ids"]) <= 512
    short, was_cut = backend.fit("Short title", AD_QUESTION)
    assert short == "Short title" and not was_cut
    ps = backend.noul_many([(make_state("Save 50% on headphones! Shop now, sale ends Sunday."), AD_QUESTION),
                            (make_state("The Senate passed the budget bill on Tuesday."), AD_QUESTION)])
    assert ps[0] > ps[1]
