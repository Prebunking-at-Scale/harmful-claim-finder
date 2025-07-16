"""
Code for extracting claims from a provided short form video
"""

from uuid import UUID

from harmful_claim_finder.claim_extraction import extract_claims_from_video
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.utils.models import VideoClaims


def get_claims(
    video_id: UUID, video_uri: str, country_codes: list[str]
) -> list[VideoClaims]:
    """
    Retrieve claims from a video directly.

    Args:
        video_id: UUID
            The id of the video being processed.
        video_uri: str
            A URI to a video in a Google Cloud Bucket.
            The file should be an mp4.

        country_codes (list[str]):
            A list of 3-letter ISO country codes for the current sentences.
            e.g. `["GBR", "USA"]`

    Returns:
        list[VideoClaims]
            A list of claims, marked up with scores.
    """
    claims: list[VideoClaims] = extract_claims_from_video(video_id, video_uri)
    pastel = CheckworthyClaimDetector(countries=country_codes)
    claims_text = [claim.claim for claim in claims]
    scores = pastel.score_sentences(claims_text, max_attempts=2)

    for claim, score in zip(claims, scores):
        claim.metadata = (
            {**claim.metadata, "score": score} if claim.metadata else {"score": score}
        )

    return claims
