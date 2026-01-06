from collections import Counter
from unittest.mock import patch

from pytest import mark, param
from test_data.dummy_keywords import test_keywords as big_test_keywords

from harmful_claim_finder.keyword_filter.topic_keyword_filter import (
    AllKeywordsType,
    TopicKeywordFilter,
)
from harmful_claim_finder.utils.models import ParsingError, TopicDetectionError

tiny_test_keywords: AllKeywordsType = {
    "test": {
        "asylum": ["Small boat crossings"],
        "crime": ["Police officers"],
    }
}


test_prompt = """
This is a test prompt.
We will put [KEYWORDS] into it.
We will also add [TEXT].
"""

test_article = [
    "Here's a sentence about crime.",
    "Here's another sentence about something crime.",
    "This could be a sentence about education.",
    "More importantly, let's talk about Severence.",
]


@mark.parametrize(
    "test_keywords,org",
    [
        param(tiny_test_keywords, "test", id="tiny keywords"),
        param(big_test_keywords, "fullfact", id="big keywords"),
    ],
)
def test_make_keyword_prompt(test_keywords, org) -> None:
    filter = TopicKeywordFilter(test_keywords[org], test_prompt)
    prompt = filter.make_keyword_prompt(test_article)
    for _topic, keywords in test_keywords[org].items():
        for keyword in keywords:
            assert keyword in prompt
    assert str(test_article) in prompt


@mark.parametrize(
    "input,expected_output",
    [
        param(
            '```{"A sentence about the number of bobbies on the beat.": ["crime"]}```',
            {"A sentence about the number of bobbies on the beat.": ["crime"]},
            id="backticks",
        ),
        param(
            'json```{"A sentence about the number of bobbies on the beat and tax rises.": ["crime","economy"]}```',
            {
                "A sentence about the number of bobbies on the beat and tax rises.": [
                    "crime",
                    "economy",
                ]
            },
            id="json and backticks",
        ),
        param(
            '{"A sentence about the number of bobbies on the beat'
            ' but also new teacher numbers.": ["crime","education"]}',
            {
                "A sentence about the number of bobbies on the beat but also new teacher numbers.": [
                    "crime",
                    "education",
                ]
            },
            id="newline",
        ),
        param(
            '```{"A sentence about the number of bobbies on the beat.": ["crime"]```',
            {"A sentence about the number of bobbies on the beat.": ["crime"]},
            id="json repair 1",
        ),
        param(
            '{"A sentence about the number of bobbies on the beat.": ["crime"}',
            {"A sentence about the number of bobbies on the beat.": ["crime"]},
            id="json repair 2",
        ),
    ],
)
def test_parsing(input, expected_output) -> None:
    filter = TopicKeywordFilter({}, "")
    assert filter.parse(input) == expected_output


@mark.parametrize(
    "input,article,expected_output",
    [
        param(
            {
                "crime": [
                    "Here's a sentence about crime.",
                    "Here's another sentence about something crime.",
                ],
                "education": ["This could be a sentence about education."],
            },
            test_article,
            {
                "Here's a sentence about crime.": ["crime"],
                "Here's another sentence about something crime.": ["crime"],
                "This could be a sentence about education.": ["education"],
                "More importantly, let's talk about Severence.": [],
            },
            id="One sentence without topic",
        ),
        param(
            {},
            [
                "Declan Rice scored two incredible free kicks against Real Madrid.",
                "That was the same week Bon Iver's new album got rave reviews.",
                "It isn't advised to swim from London to Antarctica.",
            ],
            {
                "Declan Rice scored two incredible free kicks against Real Madrid.": [],
                "That was the same week Bon Iver's new album got rave reviews.": [],
                "It isn't advised to swim from London to Antarctica.": [],
            },
            id="No sentences with topic",
        ),
        param(
            {
                "environment": ["This sentence is about everything."],
                "education": ["This sentence is about everything."],
                "health": ["This sentence is about everything."],
                "economy": ["This sentence is about everything."],
            },
            ["This sentence is about everything.", "But this one isn't."],
            {
                "This sentence is about everything.": [
                    "environment",
                    "economy",
                    "health",
                    "education",
                ],
                "But this one isn't.": [],
            },
            id="sentence about many topics",
        ),
        param(
            {
                "environment": [
                    "This sentence is about everything.",
                    "This sentence is about everything.",
                ],
                "education": [
                    "This sentence is about everything.",
                    "This sentence is about everything.",
                ],
                "health": [
                    "This sentence is about everything.",
                    "This sentence is about everything.",
                ],
                "economy": [
                    "This sentence is about everything.",
                    "This sentence is about everything.",
                ],
            },
            [
                "This sentence is about everything.",
                "This sentence is about everything.",
                "But this one isn't.",
            ],
            {
                "This sentence is about everything.": [
                    "education",
                    "health",
                    "economy",
                    "environment",
                ],
                "But this one isn't.": [],
            },
            id="duplicate sentences",
        ),
    ],
)
def test_format_results_article(input, article, expected_output) -> None:
    filter = TopicKeywordFilter({}, "")
    formatted_results = filter.format_results(input, article)
    for sentence, topics in formatted_results.items():
        assert Counter(expected_output[sentence]) == Counter(topics)

    assert Counter(formatted_results.keys()) == Counter(expected_output.keys())


def test_mapping():
    keywords = big_test_keywords["fullfact"]
    filter = TopicKeywordFilter(keywords)
    assert list(filter.mapped_keywords.keys()) == [
        str(i + 1) for i in range(len(filter.mapped_keywords))
    ]
    for mapped_list, unmapped_list in zip(
        filter.mapped_keywords.values(), filter.keywords.values()
    ):
        assert mapped_list == unmapped_list


def test_unampping():
    keywords = big_test_keywords["fullfact"]
    filter = TopicKeywordFilter(keywords)
    mapped_result = {
        f"sentence {mapped}": [mapped] for mapped in filter.topic_name_map.keys()
    }
    unmapped_result = filter.do_result_unmapping(mapped_result)
    for _, topics in unmapped_result.items():
        for topic in topics:
            assert topic in keywords.keys()


def test_format_results_error():
    """
    Check if we throw a parsing error when we get bad json output from gemini
    """
    filter = TopicKeywordFilter(
        {"health": ["doctor"], "business": ["briefcase"], "travel": ["plane"]}
    )
    article = ["sentence1", "sentence2", "sentence3", "sentence4"]

    correct_result = {
        "1": ["sentence1", "sentence2"],
        "2": ["sentence3", "sentence4"],
        "3": [],
    }

    broken_result = {
        "1": ["sentence1", "sentence2", "2", ["sentence3", "sentence4"], "3", []]
    }

    x = filter.format_results(correct_result, article)
    assert x == {
        "sentence1": ["1"],
        "sentence2": ["1"],
        "sentence3": ["2"],
        "sentence4": ["2"],
    }

    try:
        _ = filter.format_results(broken_result, article)
    except ParsingError:
        assert True
        return

    assert False


@patch(
    "harmful_claim_finder.keyword_filter.topic_keyword_filter.run_prompt_async",
    return_value='{"1": ["sentence1", "sentence2"], "2": ["sentence3"]}',
)
@patch.object(
    TopicKeywordFilter, "format_results", side_effect=ParsingError("Parsing error")
)
async def test_fix_json(mocked_format, mocked_run_prompt):
    """
    Check if we try fixing the json upon a parsing error
    """
    filter = TopicKeywordFilter({"health": ["doctor"], "business": ["briefcase"]})
    article = ["sentence1", "sentence2", "sentence3", "sentence4"]

    try:
        _ = await filter.run_all_for_article(article, 1)
    except TopicDetectionError:
        pass

    # if formatting and prompt running were done twice, we know it retried to fix json
    assert mocked_format.call_count == 2 and mocked_run_prompt.call_count == 2
