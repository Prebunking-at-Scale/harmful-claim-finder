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
        {
            "claim": "this is claim",
            "original_text": "this is quote",
            "topics": ["topic"],
        },
        {
            "claim": "this is also claim",
            "original_text": "this is also quote",
            "topics": ["topic"],
        },
        {
            "claim": "this is third claim",
            "original_text": "this is third quote",
            "topics": ["topic"],
        },
        {
            "claim": "this is fourth claim",
            "original_text": "doesn't appear anywhere",
            "topics": ["topic"],
        },
    ]
    dummy_transcript = [
        TranscriptSentence(
            id=fake_id,
            source="",
            text="this is quote from PM",
            start_time_s=0,
        ),
        TranscriptSentence(
            id=fake_id,
            source="",
            text="this is also quote from PM",
            start_time_s=1,
        ),
        TranscriptSentence(
            id=fake_id,
            source="",
            text="extra sentence",
            start_time_s=2,
        ),
        TranscriptSentence(
            id=fake_id,
            source="",
            text="this is third quote from PM",
            start_time_s=3,
        ),
        TranscriptSentence(
            id=fake_id,
            source="",
            text="fourth sentence now please",
            start_time_s=4,
        ),
    ]
    kw = {"topic": ["keyword"]}
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = await extract_claims_from_transcript(dummy_transcript, kw)
    expected = [
        VideoClaims(
            video_id=fake_id,
            claim="this is claim",
            start_time_s=0,
            metadata={"quote": "this is quote", "topics": ["topic"]},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is also claim",
            start_time_s=1,
            metadata={"quote": "this is also quote", "topics": ["topic"]},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is third claim",
            start_time_s=3,
            metadata={"quote": "this is third quote", "topics": ["topic"]},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is fourth claim",
            start_time_s=3,
            metadata={"quote": "doesn't appear anywhere", "topics": ["topic"]},
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
            "topics": ["topic"],
        },
        {
            "claim": "this is also claim",
            "original_text": "this is also quote",
            "timestamp": 1,
            "duration": 1,
            "topics": ["topic"],
        },
        {
            "claim": "this is third claim",
            "original_text": "this is third quote",
            "timestamp": 2,
            "duration": 1,
            "topics": ["topic"],
        },
    ]
    kw = {"topic": ["keyword"]}
    dummy_output = f"```json{json.dumps(dummy_claims)}```"
    mock_run_prompt.return_value = dummy_output
    claims = await extract_claims_from_video(fake_id, "video_uri", kw)
    expected = [
        VideoClaims(
            video_id=fake_id,
            claim="this is claim",
            start_time_s=0,
            metadata={"quote": "this is quote", "topics": ["topic"]},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is also claim",
            start_time_s=1,
            metadata={"quote": "this is also quote", "topics": ["topic"]},
        ),
        VideoClaims(
            video_id=fake_id,
            claim="this is third claim",
            start_time_s=2,
            metadata={"quote": "this is third quote", "topics": ["topic"]},
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
            """{"1": {"claim": "c", "original_text": "q"},}""",
            TypeError,
            id="Dictionary",
        ),
    ],
)
@patch("harmful_claim_finder.claim_extraction.run_prompt")
async def test_text_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    kw = {"topic": ["keyword"]}
    mock_run_prompt.return_value = output
    with raises(error):
        await _get_transcript_claims([], kw)


@mark.parametrize(
    "output,transcript,expected",
    [
        param(
            """[{"claim": "claim", "original_text": "quote"}]""",
            [
                TranscriptSentence(
                    id=fake_id,
                    source="",
                    text="quote",
                    start_time_s=0,
                    metadata={},
                )
            ],
            [],
            id="All missing",
        ),
        param(
            """
            [{"claim": "claim", "original_text": "quote 1"},
            {"claim": "claim", "original_text": "quote 2", "topics": ["topic"]}]
            """,
            [
                TranscriptSentence(
                    id=fake_id,
                    source="",
                    text="quote 1",
                    start_time_s=0,
                    metadata={},
                ),
                TranscriptSentence(
                    id=fake_id,
                    source="",
                    text="quote 2",
                    start_time_s=1,
                    metadata={},
                ),
            ],
            [
                VideoClaims(
                    video_id=fake_id,
                    claim="claim",
                    start_time_s=1.0,
                    metadata={"quote": "quote 2", "topics": ["topic"]},
                )
            ],
            id="Some missing",
        ),
    ],
)
@patch("harmful_claim_finder.claim_extraction.run_prompt")
async def test_text_extraction_missing_keys(
    mock_run_prompt,
    output: str,
    transcript: list[TranscriptSentence],
    expected: list[VideoClaims],
):
    kw = {"topic": ["keyword"]}
    mock_run_prompt.return_value = output
    claims = await _get_transcript_claims(transcript, kw)
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
async def test_video_extraction_bad_output(
    mock_run_prompt, output: str, error: type[Exception]
):
    kw = {"topic": ["keyword"]}
    mock_run_prompt.return_value = output
    with raises(error):
        await _get_video_claims(fake_id, "uri", kw)


@patch("harmful_claim_finder.claim_extraction.run_prompt", return_value="BAD OUTPUT")
async def test_transcript_retries(mock_run_prompt):
    try:
        kw = {"topic": ["keyword"]}
        await extract_claims_from_transcript([], kw, max_attempts=3)
        assert False
    except ClaimExtractionError:
        assert True

    assert mock_run_prompt.call_count == 6


@patch("harmful_claim_finder.claim_extraction.run_prompt", return_value="BAD OUTPUT")
async def test_video_retries(mock_run_prompt):
    try:
        kw = {"topic": ["keyword"]}
        await extract_claims_from_video(fake_id, "video", kw, max_attempts=3)
        assert False
    except ClaimExtractionError:
        assert True

    assert mock_run_prompt.call_count == 6
