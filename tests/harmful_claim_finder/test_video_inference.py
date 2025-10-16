from unittest.mock import Mock, patch
from uuid import UUID

from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.pastel.pastel import ScoresAndAnswers
from harmful_claim_finder.utils.models import VideoClaims
from harmful_claim_finder.video_inference import get_claims

fake_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

unscored_claims = [
    VideoClaims(
        video_id=fake_id,
        claim="claim 1",
        start_time_s=0,
        metadata={"quote": "quote 1"},
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 2",
        start_time_s=1,
        metadata={"quote": "quote 2"},
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 3",
        start_time_s=2,
        metadata={"quote": "quote 3"},
    ),
]

scored_claims = [
    VideoClaims(
        video_id=fake_id,
        claim="claim 1",
        start_time_s=0,
        metadata={"quote": "quote 1", "score": 0.9, "answers": {"q": 0.1}},
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 2",
        start_time_s=1,
        metadata={"quote": "quote 2", "score": 0.2, "answers": {"q": 0.2}},
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 3",
        start_time_s=2,
        metadata={"quote": "quote 3", "score": 0, "answers": {"q": 0.3}},
    ),
]


@patch("harmful_claim_finder.video_inference.CheckworthyClaimDetector")
@patch("harmful_claim_finder.video_inference.extract_claims_from_video")
async def test_output_format(mock_extract_claims, mock_pastel):
    mock_extract_claims.return_value = unscored_claims
    mock_pastel_class = Mock(CheckworthyClaimDetector)
    mock_pastel_class.score_sentences.return_value = {
        "claim 1": ScoresAndAnswers(sentence="claim 1", score=0.9, answers={"q": 0.1}),
        "claim 2": ScoresAndAnswers(sentence="claim 2", score=0.2, answers={"q": 0.2}),
        "claim 3": ScoresAndAnswers(sentence="claim 3", score=0, answers={"q": 0.3}),
    }
    mock_pastel.return_value = mock_pastel_class
    kw = {"topic": ["keyword"]}
    output = await get_claims(fake_id, "video_uri", kw)
    assert output == scored_claims
