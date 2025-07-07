import csv
import logging
import pathlib
from typing import Dict
from scipy.special import softmax  # type: ignore
from simpletransformers.classification import ClassificationModel

# You should set an environment variable to
# this to specify the sa file that gives permission
# to download the model.
SERVICE_ACCOUNT_FILE_ENV = "TOPIC_MODEL_SERVICE_ACCOUNT_FILE"
# If this env flag exists, the model will always be downloaded.
FORCE_MODEL_DOWNLOAD_FLAG = "TOPIC_MODEL_FORCE_DOWNLOAD"

_logger = logging.getLogger(__name__)

# The location is always relative to the location of this file.
_MODEL_LOCAL_LOC = (
    pathlib.Path(__file__).parent.absolute() / "models" / "news_category_model"
).resolve()

_MODEL_BUCKET = "fullfact-afc-models"
_MODEL_PREFIX = "cm-metaclassifier-topic-model/v2023-10-17/"


def _get_labels() -> Dict[int, str]:
    result = {}
    with open((_MODEL_LOCAL_LOC / "topic_labels.csv").resolve()) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[int(row["index"])] = row["label"]
    return result


def _should_download_new_model() -> bool:
    """
    Check whether the model has already been downloaded
    """
    import os

    return (
        not (_MODEL_LOCAL_LOC / "config.json").exists()
        or FORCE_MODEL_DOWNLOAD_FLAG in os.environ
    )


def _download_model() -> None:
    """
    Downloads all the files under the model in the bucket.
    """
    from google.cloud import storage
    import os

    if SERVICE_ACCOUNT_FILE_ENV in os.environ:
        client = storage.Client.from_service_account_json(
            os.environ[SERVICE_ACCOUNT_FILE_ENV]
        )
    else:
        client = storage.Client()

    for blob in client.list_blobs(
        _MODEL_BUCKET,
        prefix=_MODEL_PREFIX,
        delimiter="/",
        include_trailing_delimiter=True,
    ):
        if blob.name != _MODEL_PREFIX:
            outfile = _MODEL_LOCAL_LOC / blob.name.split("/")[-1]
            blob.download_to_filename(outfile)


def create_topic_model() -> ClassificationModel:
    """
    If use_cosine_similarity=True claim and media sentence are
    embedded and ONE macro similarity score is calculated for each
    claim/media sentence pair. If use_cosine_similarity=False the
    raw topic detection type scores for claim and sentence are
    returned together
    """
    if _should_download_new_model():
        _logger.info("Model directory does not exist. Downloading model...")
        _download_model()
        _logger.info("Model downloaded.")
    else:
        _logger.info("Model directory exists. Skip downloading...")

    model_args = {
        "overwrite_output_dir": True,
        "reprocess_input_data": True,
        "sliding_window": True,
        "do_lower_case": True,
        "eval_batch_size": 32,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "process_count": 1,
    }

    # Create a ClassificationModel
    return ClassificationModel(
        "distilbert",
        str(_MODEL_LOCAL_LOC),
        num_labels=17,
        use_cuda=False,
        args=model_args,
    )


class TopicClassifier:
    def __init__(self, topic_model: ClassificationModel) -> None:
        self._topic_model = topic_model
        self._labels = _get_labels()
        self._topic_threshold: float = 0.1

    def predict(self, text: list[str]) -> list[str]:
        predictions, _embeddings = self._topic_model.predict(text)
        return [self._labels[prediction] for prediction in predictions]

    def predict_and_get_output(
        self, text: list[str]
    ) -> tuple[list[str], list[dict[str, float]]]:
        """
        For a list of texts, returns topic predictions and raw output
        """
        predictions, _embeddings = self._topic_model.predict(text)
        predictions = [self._labels[prediction] for prediction in predictions]
        outputs = [softmax(article_outputs[0]) for article_outputs in _embeddings]
        outputs = [
            {self._labels[i]: float(p) for i, p in enumerate(o)} for o in outputs
        ]
        return predictions, outputs

    def get_topic_predictions_and_output(
        self, text: str
    ) -> tuple[str, dict[str, float]]:
        """
        Gets main topic, all possible topics, and the topic scores, for the inputted text.
        """
        predictions, raw_outputs = self.predict_and_get_output([text])
        main_prediction = predictions[0]
        outputs = raw_outputs[0]
        sorted_outputs = dict(sorted(outputs.items(), key=lambda x: x[1], reverse=True))

        all_topics = {
            t: o for t, o in sorted_outputs.items() if o >= self._topic_threshold
        }
        return main_prediction, all_topics

    def get_topic_and_scores_batch(
        self, text: list[str]
    ) -> tuple[list[str], list[dict[str, float]]]:
        """
        Gets the main topic, and all possible topics with scores, for a list of texts.
        """
        predictions, raw_outputs = self.predict_and_get_output(text)
        all_topics = []
        for output in raw_outputs:
            sorted_scores = dict(
                sorted(output.items(), key=lambda x: x[1], reverse=True)
            )
            current_topics = {
                t: o for t, o in sorted_scores.items() if o >= self._topic_threshold
            }
            all_topics.append(current_topics)
        return predictions, all_topics


if __name__ == "__main__":
    test = "The football is going on forever."
    classifier = TopicClassifier(create_topic_model())
    print(classifier.predict([test]))

    test = "Matt Hancock did a big speech in parliament about how many diseases they try to stop in hospitals."
    main_topic, all_topics = classifier.get_topic_predictions_and_output(test)
    print()
    print(main_topic)
    print(all_topics)
