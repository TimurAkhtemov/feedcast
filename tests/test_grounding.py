from feedcast.grounding import looks_title_case, numbers_in, unsupported

SOURCE = """\
=== From Example News ===
Headline: Port Elm council approves budget
The Port Elm city council approved a $3.24bn budget on Tuesday by a vote of 7 to 2.
Mayor Dana Whitfield said spending on schools would rise 45% next year. About 1,200
teachers will be hired. Gov. Alana Reyes praised the plan. The U.S. Treasury was not
involved. officials expect a final vote in March.
"""


def test_grounded_segment_passes():
    text = (
        "Port Elm's council approved a 3.24 billion dollar budget on Tuesday, by 7 votes to 2. "
        "Mayor Dana Whitfield said school spending will rise 45 percent. "
        "About 1,200 teachers will be hired. Officials expect a final vote in March."
    )
    assert unsupported(text, SOURCE) == []


def test_invented_number_and_name_are_caught():
    text = "The budget passed 8 to 1. Senator Mark Ellison opposed it."
    problems = unsupported(text, SOURCE)
    assert "8" in problems and "1" in problems
    assert "Mark" in problems and "Ellison" in problems


def test_rounding_to_written_precision_is_allowed_but_not_beyond():
    assert unsupported("The budget is 3.2 billion dollars.", SOURCE) == []
    assert unsupported("The budget is 3 billion dollars.", SOURCE) == ["3 billion"]  # 7% off
    assert unsupported("The budget is 3.5 billion dollars.", SOURCE) == ["3.5 billion"]


def test_scale_words_and_digit_forms_match():
    assert unsupported("About 1.2 thousand teachers will be hired.", SOURCE) == []
    assert unsupported("Spending rises forty-five percent.", SOURCE) == []
    assert unsupported("Twelve hundred teachers.", SOURCE) == []


def test_spelled_numbers_are_checked():
    assert unsupported("Seventy people protested.", SOURCE) == ["seventy"]
    # "one" alone is too often a pronoun to check
    assert unsupported("One official said the plan works.", SOURCE) == []


def test_aliases_and_title_abbreviations():
    assert unsupported("The United States Treasury was not involved.", SOURCE) == []
    assert unsupported("Governor Alana Reyes praised the plan.", SOURCE) == []


def test_sentence_initial_ordinary_words_are_not_names():
    text = "Meanwhile, teachers celebrated. Thank you for listening. Stay tuned."
    assert unsupported(text, SOURCE) == []


def test_title_case_headline():
    headline = "Council Approves Record Budget In Port Elm"
    assert looks_title_case(headline)
    assert unsupported(headline, SOURCE, title_case=True) == []
    assert unsupported("Council Approves Ellison Budget", SOURCE, title_case=True) == ["Ellison"]


def test_allowed_terms_pass():
    assert unsupported("This story comes from Valley Post.", SOURCE, allowed=["Valley Post"]) == []


def test_numbers_in_parses_forms():
    values = {(n.raw, n.scaled) for n in numbers_in("$3.24bn, 45%, 1,200 and 5k plus twenty-one")}
    assert ("3.24bn", 3.24e9) in values
    assert ("45", 45) in values
    assert ("1,200", 1200) in values
    assert ("5k", 5000) in values
    assert ("twenty one", 21) in values


def test_date_abbreviations_and_spelled_ordinals():
    src = "Homes sold in the Port Elm area, Sept. 28-Oct. 4, for the third time."
    assert unsupported("Sold from September twenty-eighth to October fourth.", src) == []
    assert unsupported("It was the first time.", src) == []  # lone ordinals aren't figures
    assert unsupported("Sold on September twentieth.", src) == ["twentieth"]
    assert unsupported("Sold on October fifth.", src) == ["fifth"]
