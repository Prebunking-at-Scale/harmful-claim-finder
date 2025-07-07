"""
Code for extracting claims from a provided short form video
"""

from harmful_claim_finder.claim_extraction import VideoClaim, extract_claims_from_video
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector


def get_claims(video_uri: str) -> list[tuple[VideoClaim, float]]:
    """
    Retrieve claims from a video directly.

    Args:
        video_uri: str
            A URI to a video in a Google Cloud Bucket.
            The file should be an mp4.

    Returns:
        list[tuple[VideoClaim, float]]
            A list of claim, score tuples.
            Each claim is a claim found in the video.
            Each score was given by the PASTEL model.
    """
    claims = extract_claims_from_video(video_uri)
    pastel = CheckworthyClaimDetector(countries=["GBR"])
    claims_text = [claim.claim for claim in claims]
    scores = pastel.score_sentences(claims_text, max_attempts=2)
    return [(claim, float(score)) for claim, score in zip(claims, scores)]
