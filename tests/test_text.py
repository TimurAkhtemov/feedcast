from feedcast.text import (
    canonical_url,
    clean_paragraphs,
    content_hash,
    html_to_text,
    lead_text,
    strip_tracking,
)


def test_canonical_url_drops_tracking_and_normalizes():
    a = canonical_url("http://www.Example.com/news/story/?utm_source=rss&utm_medium=x&id=7#top")
    b = canonical_url("https://example.com/news/story?id=7&fbclid=abc")
    assert a == b == "https://example.com/news/story?id=7"


def test_strip_tracking_keeps_the_published_host():
    assert strip_tracking("https://www.example.com/a?utm_source=rss&page=2") == \
        "https://www.example.com/a?page=2"


def test_clean_paragraphs_strips_boilerplate():
    text = "\n".join([
        "By Jane Doe",
        "WASHINGTON (AP) — The Senate voted 52 to 48 on Tuesday.",
        "Sign up for our newsletter today!",
        "Advertisement",
        "Photo: Getty Images",
        "Read more: other story",
        "The bill now goes to the House. See https://example.com/bill for details.",
    ])
    assert clean_paragraphs(text) == [
        "WASHINGTON (AP) — The Senate voted 52 to 48 on Tuesday.",
        "The bill now goes to the House. See for details.",
    ]


def test_lead_text_extends_short_first_paragraphs():
    text = "Breaking.\n\n" + " ".join(["word"] * 50) + "\n\nLater paragraph."
    lead = lead_text(text, min_words=40)
    assert lead.startswith("Breaking. word") and "Later" not in lead


def test_html_to_text_keeps_paragraphs():
    assert html_to_text("<p>One &amp; two</p><p>Three<br>four</p>") == "One & two\n\nThree\n\nfour"


def test_content_hash_ignores_case_and_punctuation():
    body = "The Senate voted on Tuesday. " * 10
    assert content_hash("A", body) == content_hash("B", body.upper().replace(".", "!"))
    assert content_hash("A", "short") != content_hash("B", "short")


def test_canonical_from_page_only_on_the_same_registrable_domain():
    from feedcast.ingest import pick_canonical

    assert pick_canonical("https://news.bbc.co.uk/story/1", "https://evil.co.uk/phish") == \
        "https://news.bbc.co.uk/story/1"
    assert pick_canonical("https://www.bbc.co.uk/news/1", "https://www.bbc.co.uk/news/articles/1") == \
        "https://bbc.co.uk/news/articles/1"
    assert pick_canonical("https://example.com/a?utm_source=x", "https://example.com/") == \
        "https://example.com/a"
