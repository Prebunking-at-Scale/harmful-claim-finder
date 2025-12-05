import logging
import time

from harmful_claim_finder.keyword_filter.topic_keyword_filter import TopicKeywordFilter
from harmful_claim_finder.pastel_inference import CheckworthyClaimDetector
from harmful_claim_finder.utils.models import (
    CheckworthyError,
    PastelError,
    TopicDetectionError,
    TranscriptSentence,
    VideoClaims,
)

logger = logging.getLogger(__name__)


async def get_claims(
    keywords: dict[str, list[str]],
    sentences: list[TranscriptSentence],
) -> list[VideoClaims]:
    """
    A wrapper function to run genai checkworthy.
    First finds the topics for each provided sentence.
    Next runs any sentences with topics through PASTEL to get a checkworthy score.

    Args:
        keywords (dict[str, list[str]]):
            A {topic: keywords} dictionary containing the kw for each topic. E.g.
            ```python
            {
                "crime": ["police", "robbers"],
                "health": ["doctor", "hospital"],
            }
            ```
        sentences (list[TranscriptSentence]):
            A list of transcript sentences to run checkworthy on.

    Returns:
        A list of claims contained within the transcript.

    Raises:
        CheckworthyError:
            If something goes wrong during topic detection or
            pastel, the CheckworthyError will say what went wrong.
    """
    try:
        texts = [sentence.text for sentence in sentences]

        topics_start_time = time.time()
        topic_filter = TopicKeywordFilter(keywords=keywords)
        topic_keywords = await topic_filter.run_all_for_article(texts, max_attempts=2)

        have_topic = [sentence for sentence, topics in topic_keywords.items() if topics]
        logger.debug(f"{len(have_topic)} sentences have topics.")

        keywords_runtime = time.time() - topics_start_time
        if not have_topic:
            logger.info(
                f"Topics runtime: {keywords_runtime:.2f}s | "
                "PASTEL runtime: 0.00s | "
                "0 sentences checked by PASTEL | "
                "0 have nonzero score"
            )
            return []

        pastel_start_time = time.time()

        checkworthy_model = CheckworthyClaimDetector()

        all_scores_and_answers = await checkworthy_model.score_sentences(
            have_topic, max_attempts=2
        )

        claims = [
            VideoClaims(
                video_id=sentence.video_id,
                claim=sentence.text,
                start_time_s=sentence.start_time_s,
                metadata=(
                    {
                        **sentence.metadata,
                        "score": float(all_scores_and_answers[sentence.text].score),
                        "topics": topic_keywords[sentence.text],
                        "answers": all_scores_and_answers[sentence.text].answers,
                    }
                ),
            )
            for sentence in sentences
            if sentence.text in all_scores_and_answers.keys()
            and all_scores_and_answers[sentence.text].score > 0
        ]

        pastel_runtime = time.time() - pastel_start_time
        logger.info(
            f"Topics runtime: {keywords_runtime:.2f}s | "
            f"PASTEL runtime: {pastel_runtime:.2f}s | "
            f"{len(have_topic)} sentences checked by PASTEL | "
            f"{len(claims)} have nonzero score"
        )
        return claims

    except (TopicDetectionError, PastelError) as e:
        raise CheckworthyError from e
