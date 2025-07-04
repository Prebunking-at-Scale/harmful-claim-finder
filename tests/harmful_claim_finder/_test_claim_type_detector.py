from pytest import fixture, mark, param

from harmful_claim_finder.claim_type_detector import production_inference

# "predictions",
# "voting",
# "opinion",
# "support",
# "other",
# "not_claim",

_TEST_CASES = [
    ("GDP is at 1000%.", ["quantity"]),
    ("I really like cake.", ["personal"]),
    ("That means it lead to a recession.", ["correlation"]),
    ("Yes, another layer of restrictions lifts on Monday.", ["rules"]),
    ("But that decision will affect us all.", ["predictions"]),
    (
        "He has no right to overrule the valves are people in Scotland, and we will have the referendum we voted for.",
        ["rules", "voting", "support"],
    ),
    (
        "It is the right project for the Conservatives of how to manage the economy.",
        ["support", "not_claim"],
    ),
]


@fixture(name="typer", scope="module")
def fixture_typer():
    return production_inference.create_claim_typer()


@mark.parametrize(
    "text, expected", [param(text, expected, id=text) for text, expected in _TEST_CASES]
)
def test_should_detect_claims(typer, text, expected):
    # when
    result = typer.label_one(text)

    # then
    assert result.types_detected == expected


def test_should_detect_claims_in_batch(typer):
    # given
    sentences = [text for text, _expected in _TEST_CASES]
    expected = [expected for _text, expected in _TEST_CASES]

    # when
    result = typer.label_batch(sentences)

    # then
    assert [item.types_detected for item in result] == expected
