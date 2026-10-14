import pytest

from feedcast.speak import spoken


@pytest.mark.parametrize("written, expected", [
    ("The fine was $1.2bn.", "The fine was one point two billion dollars."),
    ("It cost $5 million.", "It cost five million dollars."),
    ("Prices rose 45%.", "Prices rose forty-five percent."),
    ("Prices rose 45 percent.", "Prices rose forty-five percent."),
    ("About 1,200 people came.", "About one thousand two hundred people came."),
    ("The vote was 52-48.", "The vote was fifty-two to forty-eight."),
    ("It happened in 2026.", "It happened in twenty twenty-six."),
    ("Her 21st birthday.", "Her twenty-first birthday."),
    ("The rate is 4.25 percent.", "The rate is four point two five percent."),
    ("At 3:30 p.m. and 9:05.", "At three thirty p.m. and nine oh five."),
    ("A £10 ticket.", "A ten pounds ticket."),
    ("Read https://example.com/x now.", "Read now."),
    ("Apples & oranges, e.g. fruit.", "Apples and oranges, for example fruit."),
    ("Sold Sept. 28 to Oct. 4.", "Sold September twenty-eighth to October fourth."),
    ("On March 3, 2026, it rained.", "On March third, twenty twenty-six, it rained."),
])
def test_spoken(written, expected):
    assert spoken(written) == expected


def test_words_untouched():
    text = "Mayor Dana Whitfield said traffic would return to normal."
    assert spoken(text) == text
