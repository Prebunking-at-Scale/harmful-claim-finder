"""
Searches a video transcript for claims.
Does not make the assumption that a sentence is a claim.
Claims can span more than one sentence, or be a span within a sentence.
"""

from harmful_claim_finder.claim_extraction import extract_claims_from_transcript
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.utils.models import TranscriptSentence, VideoClaims


async def get_claims(
    keywords: dict[str, list[str]],
    transcript: list[TranscriptSentence],
) -> list[VideoClaims]:
    """
    Retrieve claims from a video transcript.
    Claims can be more than one sentence, or part of a sentence.

    Args:
        keywords (dict[str, list[str]]):
            A {topic: keywords} dictionary containing the kw for each topic. E.g.
            ```python
            {
                "crime": ["police", "robbers"],
                "health": ["doctor", "hospital"],
            }
            ```
        transcript (list[TranscriptSentence]):
            The transcript you want to search for claims.

    Returns:
        list[VideoClaims]
            A list of claims, marked up with scores.
    """
    claims: list[VideoClaims] = await extract_claims_from_transcript(
        transcript=transcript, keywords=keywords, max_attempts=2
    )
    pastel = CheckworthyClaimDetector()
    claims_text = [claim.claim for claim in claims]
    scores = await pastel.score_sentences(claims_text, max_attempts=2)

    for claim, score in zip(claims, scores):
        claim.metadata = (
            {**claim.metadata, "score": score} if claim.metadata else {"score": score}
        )

    return claims
