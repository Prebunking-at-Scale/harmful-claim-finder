import logging
import time

import pycountry

from harmful_claim_finder.keyword_filter.topic_keyword_filter import TopicKeywordFilter
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector
from harmful_claim_finder.utils.models import (
    CheckworthyError,
    CheckworthyResult,
    PastelError,
    TopicDetectionError,
    TranscriptSentence,
    VideoClaims,
)

logger = logging.getLogger(__name__)


def parse_country_codes(codes: list[str]) -> list[str]:
    """
    Converts a list of ISO 3 letter country codes into a list of country names.
    For some countries that are known by multiple names, we manually expand the
    list accordingly.

    Args:
        codes (list[str]):
            A list of ISO country codes, e.g. `["GBR", "USA"]`.

    Returns:
        A list of country names, e.g. `["United Kingdom", "United States"]`
    """
    try:
        country_names = [pycountry.countries.get(alpha_3=code).name for code in codes]
        if "United Kingdom" in country_names:
            country_names.extend(
                ["England", "Wales", "Scotland", "Northern Ireland", "Britain", "UK"]
            )
        if "United States" in country_names:
            country_names.extend(["America", "USA"])

        return country_names
    except AttributeError as exc:
        logger.error(
            "One of the countries in %s could not be found by pycountry.", codes
        )
        raise CheckworthyError from exc


def get_claims(
    keywords: dict[str, list[str]],
    sentences: list[TranscriptSentence],
    country_codes: list[str],
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

        country_codes (list[str]):
            A list of 3-letter ISO country codes for the current sentences.
            e.g. `["GBR", "USA"]`

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
        topic_keywords = topic_filter.run_all_for_article(texts, max_attempts=2)

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
        country_names = parse_country_codes(country_codes)

        checkworthy_model = CheckworthyClaimDetector(countries=country_names)

        scores = checkworthy_model.score_sentences(have_topic, max_attempts=2)

        scored_sentences = {
            sentence: score for sentence, score in zip(have_topic, scores)
        }

        claims = [
            VideoClaims(
                video_id=sentence.video_id,
                claim=sentence.text,
                start_time_s=sentence.start_time_s,
                metadata={
                    "score": float(scored_sentences[sentence.text]),
                    "topics": topic_keywords[sentence.text],
                },
            )
            for sentence in sentences
            if sentence.text in scored_sentences and scored_sentences[sentence.text] > 0
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
