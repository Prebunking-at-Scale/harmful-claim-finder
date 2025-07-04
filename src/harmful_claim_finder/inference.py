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


def run_checkworthy(
    keywords: dict[str, list[str]],
    sentences: list[str],
    country_codes: list[str],
) -> list[CheckworthyResult]:
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
        sentences (list[str]):
            A list of sentences to run checkworthy on.

        country_codes (list[str]):
            A list of 3-letter ISO country codes for the current sentences.
            e.g. `["GBR", "USA"]`

    Returns:
        A score and list of topics for each sentence.

    Raises:
        CheckworthyError:
            If something goes wrong during topic detection or
            pastel, the CheckworthyError will say what went wrong.
    """
    try:
        topics_start_time = time.time()
        topic_filter = TopicKeywordFilter(keywords=keywords)
        topic_keywords = topic_filter.run_all_for_article(sentences, max_attempts=2)

        have_topic = [sentence for sentence, topics in topic_keywords.items() if topics]
        logger.debug(f"{len(have_topic)} sentences have topics.")

        result_dict = {
            sentence: CheckworthyResult(score=0, topics=[]) for sentence in sentences
        }

        keywords_runtime = time.time() - topics_start_time
        if not have_topic:
            logger.info(
                f"Topics runtime: {keywords_runtime:.2f}s | "
                "PASTEL runtime: 0.00s | "
                "0 sentences checked by PASTEL | "
                "0 have nonzero score"
            )
            return [result_dict[sentence] for sentence in sentences]

        pastel_start_time = time.time()
        country_names = parse_country_codes(country_codes)

        checkworthy_model = CheckworthyClaimDetector(countries=country_names)

        scores = checkworthy_model.score_sentences(have_topic, max_attempts=2)

        count_scored = 0
        for sentence, score in zip(have_topic, scores):
            result_dict[sentence] = CheckworthyResult(
                score=float(score),
                topics=topic_keywords[sentence],
            )
            if score > 0:
                count_scored += 1

        pastel_runtime = time.time() - pastel_start_time
        logger.info(
            f"Topics runtime: {keywords_runtime:.2f}s | "
            f"PASTEL runtime: {pastel_runtime:.2f}s | "
            f"{len(have_topic)} sentences checked by PASTEL | "
            f"{count_scored} have nonzero score"
        )
        return [result_dict[sentence] for sentence in sentences]

    except (TopicDetectionError, PastelError) as e:
        raise CheckworthyError from e
