from typing import Iterable, Tuple, Dict, Any

import ff_streaming
from google import pubsub_v1
from google.pubsub_v1 import SubscriberClient

from article_enrichment_processor import (
    country_classification,
    topic_classification,
    language_detection,
    pre_calculation,
    organisation_checking,
)
import logging

_logger = logging.getLogger(__name__)


class Processor:
    def __init__(
        self,
        country_predictor: country_classification.CountryPredictor,
        topic_classifier: topic_classification.TopicClassifier,
        language_identifier: language_detection.LanguageIdentifier,
        organisation_checker: organisation_checking.OrganisationChecker,
    ) -> None:
        self._country_predictor = country_predictor
        self._topic_classifier = topic_classifier
        self._language_identifier = language_identifier
        self._organisation_checker = organisation_checker

    def __call__(
        self, input_file_messages: Iterable[ff_streaming.FileMessage]
    ) -> Iterable[Tuple[ff_streaming.FileMessage, Dict[str, Any]]]:
        _logger.info("Processing a message...")
        for message in input_file_messages:
            full_article, url = pre_calculation.pre_calculation_messge(message)
            (
                best_guess_country_string,
                all_countries,
            ) = self._country_predictor.get_country_from_article(full_article, url)
            _logger.info("Calculated country.")

            (
                topic,
                all_topics,
            ) = self._topic_classifier.get_topic_predictions_and_output(full_article)
            _logger.info("Calculated topic.")

            best_lang, all_langs = self._language_identifier.predict_language(
                full_article
            )

            organisations = self._organisation_checker.interested_organisations(
                message.publication
            )

            yield message, {
                "country_code": best_guess_country_string,
                "all_countries": all_countries,
                "topic": topic,
                "all_topics": all_topics,
                "language": best_lang,
                "all_languages": all_langs,
                "organisations": organisations["organisations"],
            }
        _logger.info("Message processing complete.")


def run_article_enrichment_processor(
    project_name: str, input_queue_name: str, output_queue_name: str, output_prefix: str
) -> None:
    subscription_path = pubsub_v1.SubscriberClient.subscription_path(
        project_name, f"{input_queue_name}_current"
    )
    output_topic_path = pubsub_v1.PublisherClient.topic_path(
        project_name, f"{output_queue_name}_current"
    )
    media_reader = ff_streaming.create_media_reader()
    media_writer = ff_streaming.create_media_writer()

    subscriber_client = SubscriberClient()

    processor = Processor(
        country_classification.CountryPredictor("GBR"),
        topic_classification.TopicClassifier(topic_classification.create_topic_model()),
        language_detection.LanguageIdentifier(),
        organisation_checking.OrganisationChecker(),
    )

    with subscriber_client:
        ff_streaming.process_files(
            subscriber_client,
            subscription_path,
            "enriched-articles",
            output_prefix,
            media_reader,
            media_writer,
            processor,
            output_topic_path,
        )
