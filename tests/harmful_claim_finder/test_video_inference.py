from unittest.mock import Mock, patch

from harmful_claim_finder.claim_extraction import VideoClaim
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.video_inference import get_claims


@patch("harmful_claim_finder.video_inference.CheckworthyClaimDetector")
@patch("harmful_claim_finder.video_inference.extract_claims_from_video")
def test_output_format(mock_extract_claims, mock_pastel):
    mock_extract_claims.return_value = [
        VideoClaim(
            claim="claim 1", original_text="quote 1", timestamp="00:00:01", duration=1
        ),
        VideoClaim(
            claim="claim 2", original_text="quote 2", timestamp="00:00:02", duration=1
        ),
        VideoClaim(
            claim="claim 3", original_text="quote 3", timestamp="00:00:03", duration=1
        ),
    ]
    mock_pastel_class = Mock(CheckworthyClaimDetector)
    mock_pastel_class.score_sentences.return_value = [0.9, 0.2, 0]
    mock_pastel.return_value = mock_pastel_class
    output = get_claims("video_uri")
    assert output == [
        (
            VideoClaim(
                claim="claim 1",
                original_text="quote 1",
                timestamp="00:00:01",
                duration=1,
            ),
            0.9,
        ),
        (
            VideoClaim(
                claim="claim 2",
                original_text="quote 2",
                timestamp="00:00:02",
                duration=1,
            ),
            0.2,
        ),
        (
            VideoClaim(
                claim="claim 3",
                original_text="quote 3",
                timestamp="00:00:03",
                duration=1,
            ),
            0.0,
        ),
    ]
