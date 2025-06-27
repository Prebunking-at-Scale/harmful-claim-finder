# Class to score each of a list of sentences for checkworthiness
import json
import logging
import os
from pathlib import Path

from harmful_claim_finder.models import PastelError
from harmful_claim_finder.pastel import pastel
from harmful_claim_finder.utils.gemini import GeminiError

# flake8: noqa
CHECKWORTHY_MODEL_FILE = (
    Path(os.path.dirname(os.path.abspath(__file__))) / "checkworthy_model.json"
)

_logger = logging.getLogger(__name__)


class CheckworthyClaimDetector:
    """A class for detecting which claims may be worth checking"""

    def __init__(self, countries: list[str]) -> None:
        model_dict: pastel.ModelDict = json.loads(CHECKWORTHY_MODEL_FILE.read_text())

        # Add the country question, which won't be in the file
        country_list = "[" + ", ".join(countries) + "]"
        new_question = (
            "Identify any country named in the sentence, ignoring cities, regions and other places. "
            f"If this sentence mentions any country in this list: {country_list} answer 'no'. If it doesn't name any country at all then also answer 'no'. "
            "If you're not sure, answer 'no'. Only answer 'yes' if it mentions or is clearly about some other country not on that list. "
        )
        model_dict["questions"][new_question] = -1.0
        self.pastel = pastel.Pastel.from_dict(model_dict)

    def score_sentences(
        self, sentences: list[str], max_attempts: int = 3
    ) -> pastel.ARRAY_TYPE:
        """
        Returns a checkworthy score for each of a list of sentences.
        High scores suggest more checkworthy

        Args:
            sentences (list[str])
                The list of sentences for which to run Pastel.

            max_attempts (int):
                The number of retries to attempt if there's an exception.

        Returns:
            pastel.ARRAY_TYPE:
                Checkworthy scores for each sentence.

        Raises:
            PastelError:
                Raises an exception if Pastel fails `max_attempts` times.
        """
        for _ in range(max_attempts):
            try:
                scores = self.pastel.make_predictions(sentences)
                return scores
            except GeminiError as exc:
                _logger.info(f"Error while running Gemini: {repr(exc)}")
            except ValueError as exc:
                _logger.info(
                    f"Error raised while running Pastel (probably parsing): {repr(exc)}"
                )

        raise PastelError(f"Pastel failed {max_attempts} times.")
