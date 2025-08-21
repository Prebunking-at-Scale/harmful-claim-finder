"""
Code for extracting claims from a provided short form video
"""

from uuid import UUID

from harmful_claim_finder.claim_extraction import extract_claims_from_video
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.utils.models import VideoClaims


async def get_claims(
    video_id: UUID,
    video_uri: str,
    keywords: dict[str, list[str]],
) -> list[VideoClaims]:
    """
    Retrieve claims from a video directly.

    Args:
        video_id: UUID
            The id of the video being processed.
        video_uri: str
            A URI to a video in a Google Cloud Bucket.
            The file should be an mp4.
        keywords (dict[str, list[str]]):
            A {topic: keywords} dictionary containing the kw for each topic. E.g.
            ```python
            {
                "crime": ["police", "robbers"],
                "health": ["doctor", "hospital"],
            }
            ```

    Returns:
        list[VideoClaims]
            A list of claims, marked up with scores.
    """
    claims: list[VideoClaims] = await extract_claims_from_video(
        video_id, video_uri, keywords
    )
    pastel = CheckworthyClaimDetector()
    claims_text = [claim.claim for claim in claims]
    scores = await pastel.score_sentences(claims_text, max_attempts=2)

    for claim, score in zip(claims, scores):
        claim.metadata = (
            {**claim.metadata, "score": score} if claim.metadata else {"score": score}
        )

    return claims
