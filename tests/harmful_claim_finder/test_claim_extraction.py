import json
from unittest.mock import patch
from uuid import UUID

from pydantic import ValidationError
from pytest import mark, param, raises

from harmful_claim_finder.claim_extraction import (
    _get_transcript_claims,
    _get_video_claims,
    extract_claims_from_transcript,
    extract_claims_from_video,
)
from harmful_claim_finder.utils.models import (
    ClaimExtractionError,
    TranscriptSentence,
    VideoClaims,
)

fake_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


@patch("harmful_claim_finder.claim_extraction.run_prompt")
async def test_text_extraction(mock_run_prompt):
    dummy_claims = [
        {"claim": "this is claim", "original_text": "this is quote"},
        {"claim": "this is also claim", "original_text": "this is also quote"},
        {"claim": "this is third claim", "original_text": "this is third quote"},
    ]
    dummy_transcript = [
        TranscriptSentence(
            video_id=fake_id, source="", text="this is quote from PM", start_time_s=0
        ),
        TranscriptSentence(
            video_id=fake_id,
            source="",
            text="this is also quote from PM",
            start_time_s=1,
        ),
        TranscriptSentence(
            video_id=fake_id, source="", text="extra sentence", start_time_s=2
        ),
        TranscriptSentence(
            video_id=fake_id,
            source="",
            text="this is third quote from PM",
            start_time_s=3,
        ),
    ]
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = await extract_claims_from_transcript(dummy_transcript)
    expected = [
        VideoClaims(
            video_id=fake_id,
            claim="this is quote",
            start_time_s=0,
            metadata={"paraphrased": "this is claim"},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is also quote",
            start_time_s=1,
            metadata={"paraphrased": "this is also claim"},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is third quote",
            start_time_s=3,
            metadata={"paraphrased": "this is third claim"},
        ),
    ]
    assert claims == expected


@patch("harmful_claim_finder.claim_extraction.run_prompt")
async def test_video_extraction(mock_run_prompt):
    dummy_claims = [
        {
            "claim": "this is claim",
            "original_text": "this is quote",
            "timestamp": 0,
            "duration": 1,
        },
        {
            "claim": "this is also claim",
            "original_text": "this is also quote",
            "timestamp": 1,
            "duration": 1,
        },
        {
            "claim": "this is third claim",
            "original_text": "this is third quote",
            "timestamp": 2,
            "duration": 1,
        },
    ]
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = await extract_claims_from_video(fake_id, "video_uri")
    expected = [
        VideoClaims(
            video_id=fake_id,
            claim="this is claim",
            start_time_s=0,
            metadata={"quote": "this is quote"},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is also claim",
            start_time_s=1,
            metadata={"quote": "this is also quote"},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is third claim",
            start_time_s=2,
            metadata={"quote": "this is third quote"},
        ),
    ]
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
async def test_text_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    mock_run_prompt.return_value = output
    with raises(error):
        await _get_transcript_claims([])


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
async def test_video_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    mock_run_prompt.return_value = output
    with raises(error):
        await _get_video_claims(fake_id, "uri")


@patch("harmful_claim_finder.claim_extraction.run_prompt", return_value="BAD OUTPUT")
async def test_transcript_retries(mock_run_prompt):
    try:
        await extract_claims_from_transcript([], max_attempts=3)
        assert False
    except ClaimExtractionError:
        assert True

    assert mock_run_prompt.call_count == 6


@patch("harmful_claim_finder.claim_extraction.run_prompt", return_value="BAD OUTPUT")
async def test_video_retries(mock_run_prompt):
    try:
        await extract_claims_from_video(fake_id, "video", max_attempts=3)
        assert False
    except ClaimExtractionError:
        assert True

    assert mock_run_prompt.call_count == 6
