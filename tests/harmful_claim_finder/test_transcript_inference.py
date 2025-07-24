from unittest.mock import Mock, patch
from uuid import UUID

from harmful_claim_finder.keyword_filter.topic_keyword_filter import TopicKeywordFilter
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.transcript_inference import get_claims
from harmful_claim_finder.utils.models import TranscriptSentence, VideoClaims

fake_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

sentences = [
    TranscriptSentence(video_id=fake_id, source="", text="claim 1", start_time_s=0),
    TranscriptSentence(video_id=fake_id, source="", text="claim 2", start_time_s=1),
    TranscriptSentence(video_id=fake_id, source="", text="claim 3", start_time_s=2),
    TranscriptSentence(video_id=fake_id, source="", text="claim 4", start_time_s=3),
]

topic_dict = {
    "claim 1": ["topic"],
    "claim 2": ["topic"],
    "claim 3": ["topic"],
}

scored_claims = [
    VideoClaims(
        video_id=fake_id,
        claim="claim 1",
        start_time_s=0,
        metadata={"score": 5.0, "topics": ["topic"]},
    ),
    VideoClaims(
        video_id=fake_id,
        claim="claim 2",
        start_time_s=1,
        metadata={"score": 2.5, "topics": ["topic"]},
    ),
]


@patch("harmful_claim_finder.transcript_inference.CheckworthyClaimDetector")
@patch("harmful_claim_finder.transcript_inference.TopicKeywordFilter")
async def test_output_format(mock_keyword_filter, mock_pastel):
    mock_kw_class = Mock(TopicKeywordFilter)
    mock_kw_class.run_all_for_article.return_value = topic_dict
    mock_keyword_filter.return_value = mock_kw_class

    mock_pastel_class = Mock(CheckworthyClaimDetector)
    mock_pastel_class.score_sentences.return_value = [5.0, 2.5, 2.49]
    mock_pastel_class.checkworthy_threshold = 2.5
    mock_pastel.return_value = mock_pastel_class
    output = await get_claims(
        keywords={"topic": ["keyword"]},
        sentences=sentences,
        country_codes=["USA"],
    )
    assert output == scored_claims
