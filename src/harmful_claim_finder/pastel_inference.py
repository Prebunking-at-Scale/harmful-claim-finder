# Class to score each of a list of sentences for checkworthiness
import logging
import os
from pathlib import Path

from genai_utils.gemini import GeminiError
from pastel import pastel
from pastel.models import ScoreAndAnswers, Sentence

from harmful_claim_finder.utils.models import PastelError

# flake8: noqa
CHECKWORTHY_MODEL_FILE = (
    Path(os.path.dirname(os.path.abspath(__file__))) / "checkworthy_model.json"
)

_logger = logging.getLogger(__name__)


class CheckworthyClaimDetector:
    """A class for detecting which claims may be worth checking"""

    def __init__(self) -> None:
        self.pastel = pastel.Pastel.load_model(str(CHECKWORTHY_MODEL_FILE))

    async def score_sentences(
        self, sentences: list[str], max_attempts: int = 3
    ) -> dict[str, ScoreAndAnswers]:
        """
        Returns a checkworthy score for each of a list of sentences.
        High scores suggest more checkworthy

        Args:
            sentences (list[str]):
                The list of sentences for which to run Pastel.

            max_attempts (int):
                The number of retries to attempt if there's an exception.

        Returns:
            dict[str, pastel.ScoreAndAnswers]:
                Checkworthy scores for each sentence, with answers for each
                PASTEL question.

        Raises:
            PastelError:
                Raises an exception if Pastel fails `max_attempts` times.
        """
        for _ in range(max_attempts):
            try:
                scores_and_answers = await self.pastel.make_predictions(
                    [Sentence(s, ()) for s in sentences]
                )
                return {
                    sent.sentence_text: scores
                    for sent, scores in scores_and_answers.items()
                }
            except GeminiError as exc:
                _logger.info(f"Error while running Gemini: {repr(exc)}")
            except ValueError as exc:
                _logger.info(
                    f"Error raised while running Pastel (probably parsing): {repr(exc)}"
                )

        raise PastelError(f"Pastel failed {max_attempts} times.")
