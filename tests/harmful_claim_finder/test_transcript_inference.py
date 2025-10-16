from unittest.mock import Mock, patch
from uuid import UUID

from harmful_claim_finder.utils.models import TranscriptSentence, VideoClaims
from harmful_claim_finder.transcript_inference import (
    get_claims,
    CheckworthyClaimDetector,
    TopicKeywordFilter,
)
from harmful_claim_finder.pastel.pastel import ScoresAndAnswers

fake_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

unscored_claims = [
    TranscriptSentence(
        id=fake_id,
        video_id=fake_id,
        source="test",
        text="claim 1",
        start_time_s=0,
        metadata={"quote": "quote 1"},
    ),
    TranscriptSentence(
        id=fake_id,
        video_id=fake_id,
        source="test",
        text="claim 2",
        start_time_s=1,
        metadata={"quote": "quote 2"},
    ),
    TranscriptSentence(
        id=fake_id,
        video_id=fake_id,
        source="test",
        text="claim 3",
        start_time_s=2,
        metadata={"quote": "quote 3"},
    ),
]

scored_claims = [
    VideoClaims(
        video_id=fake_id,
        claim="claim 1",
        start_time_s=0,
        metadata={
            "quote": "quote 1",
            "score": 0.9,
            "answers": {"q": 0.1},
            "topics": ["topic"],
        },
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 2",
        start_time_s=1,
        metadata={
            "quote": "quote 2",
            "score": 0.2,
            "answers": {"q": 0.2},
            "topics": ["topic"],
        },
    ),
]


@patch("harmful_claim_finder.transcript_inference.CheckworthyClaimDetector")
@patch("harmful_claim_finder.transcript_inference.TopicKeywordFilter")
async def test_output_format(mock_keyword_filter, mock_pastel):
    mock_keyword_class = Mock(TopicKeywordFilter)
    mock_keyword_class.run_all_for_article.return_value = {
        s.text: ["topic"] for s in unscored_claims
    }
    mock_keyword_filter.return_value = mock_keyword_class
    mock_pastel_class = Mock(CheckworthyClaimDetector)
    mock_pastel_class.score_sentences.return_value = {
        "claim 1": ScoresAndAnswers(score=0.9, answers={"q": 0.1}),
        "claim 2": ScoresAndAnswers(score=0.2, answers={"q": 0.2}),
        "claim 3": ScoresAndAnswers(score=0, answers={"q": 0.3}),
    }
    mock_pastel.return_value = mock_pastel_class
    kw = {"topic": ["keyword"]}
    output = await get_claims(kw, unscored_claims)
    assert output == scored_claims
