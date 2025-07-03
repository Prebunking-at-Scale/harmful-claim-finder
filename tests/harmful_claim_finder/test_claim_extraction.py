import json
from unittest.mock import patch

from pydantic import ValidationError
from pytest import mark, param, raises

from harmful_claim_finder.claim_extraction import (
    TextClaim,
    VideoClaim,
    extract_claims_from_transcript,
    extract_claims_from_video,
)


@patch("harmful_claim_finder.claim_extraction.run_prompt")
def test_text_extraction(mock_run_prompt):
    dummy_claims = [
        {"claim": "this is claim", "original_text": "this is quote"},
        {"claim": "this is also claim", "original_text": "this is also quote"},
        {"claim": "this is third claim", "original_text": "this is third quote"},
    ]
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = extract_claims_from_transcript(["1", "2", "3"])
    expected = [TextClaim(**c) for c in dummy_claims]
    assert claims == expected


@patch("harmful_claim_finder.claim_extraction.run_prompt")
def test_video_extraction(mock_run_prompt):
    dummy_claims = [
        {
            "claim": "this is claim",
            "original_text": "this is quote",
            "timestamp": "00:00:01",
            "duration": 1,
        },
        {
            "claim": "this is also claim",
            "original_text": "this is also quote",
            "timestamp": "00:00:02",
            "duration": 1,
        },
        {
            "claim": "this is third claim",
            "original_text": "this is third quote",
            "timestamp": "00:00:03",
            "duration": 1,
        },
    ]
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = extract_claims_from_video("video_uri")
    expected = [VideoClaim(**c) for c in dummy_claims]
    assert claims == expected


@mark.parametrize(
    "output,error",
    [
        param(
            """[["item1", "item2"], ["item3", "item4"]]""",
            TypeError,
            id="List not dict",
        ),
        param(
            """THIS IS JUST A STRING""",
            ValueError,
            id="String - won't parse",
        ),
        param(
            """{"1": {"claim": "c", "quote": "q"},}""",
            TypeError,
            id="Dictionary",
        ),
        param(
            """[{"claim": "claim", "quote": "quote"}]""",
            ValidationError,
            id="Wrong keys",
        ),
    ],
)
@patch("harmful_claim_finder.claim_extraction.run_prompt")
def test_text_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    mock_run_prompt.return_value = output
    with raises(error):
        extract_claims_from_transcript(["1", "2", "3"])


@mark.parametrize(
    "output,error",
    [
        param(
            """[["item1", "item2"], ["item3", "item4"]]""",
            TypeError,
            id="List not dict",
        ),
        param(
            """THIS IS JUST A STRING""",
            ValueError,
            id="String - won't parse",
        ),
        param(
            """{"1": {"claim": "c", "quote": "q"},}""",
            TypeError,
            id="Dictionary",
        ),
        param(
            """[{"claim": "claim", "quote": "quote"}]""",
            ValidationError,
            id="Wrong keys",
        ),
    ],
)
@patch("harmful_claim_finder.claim_extraction.run_prompt")
def test_video_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    mock_run_prompt.return_value = output
    with raises(error):
        extract_claims_from_video("uri")
