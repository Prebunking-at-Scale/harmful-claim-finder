# First attempt at asking a series of yes/no questions for checkworthiness etc., inspired by Sheffield's PASTEL model
# See paper: https://arxiv.org/abs/2309.07601v3 "Weakly Supervised Veracity Classification with LLM-Predicted Credibility Signals"

import asyncio
import enum
import json
import logging
from collections.abc import Callable
from typing import Dict, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import tenacity
from google.api_core import exceptions as core_exceptions

from harmful_claim_finder.pastel import pastel_functions
from harmful_claim_finder.utils.gemini import run_prompt

_logger = logging.getLogger(__name__)

EXAMPLES_TYPE = Tuple[str, float]
ARRAY_TYPE: TypeAlias = npt.NDArray[np.float64]

RETRYABLE_EXCEPTIONS = (
    core_exceptions.ResourceExhausted,
    core_exceptions.InternalServerError,
    core_exceptions.ServiceUnavailable,
    core_exceptions.DeadlineExceeded,
    ValueError,
)


class BiasType(enum.Enum):
    """Used as the key for the bias term in Pastel models"""

    BIAS = "BIAS"


FEATURE_TYPE: TypeAlias = Callable[[str], float] | str | BiasType


def log_retry_attempt(retry_state: tenacity.RetryCallState) -> None:
    """Log the retry attempt number and the exception that occurred."""
    if (not retry_state.outcome) or (not retry_state.next_action):
        return

    _logger.info(
        f"Retrying request due to {retry_state.outcome.exception()}..."
        f"Attempt #{retry_state.attempt_number}, "
        f"waiting {retry_state.next_action.sleep:.2f} seconds."
    )


class Pastel:
    """Uses list of yes/no questions and functions to analyse a piece of text.
    Each of these features has an associated weight which is used to generate
    the final score for the text.
    The main model is a dict mapping features to weights.
    """

    def __init__(self, model: dict[FEATURE_TYPE, float]) -> None:
        """
        Create a new Pastel object from a list of questions and functions.
        A Pastel model is dict of features to weights. Exactly one
        entry should be BiasType.BIAS; zero or more may be features
        that are questions (ie strings) and zero or more may be
        are callable functions defined in the pastel_functions module.
        """
        self.model = model

        # assert bias term exists
        assert isinstance(self.get_bias(), float)

    @staticmethod
    def from_feature_list(feature_names: list[FEATURE_TYPE]) -> "Pastel":
        """Take a list of features without weights. Initialise new
        model with all weights set to zero, ready for training"""
        new_model = dict()
        for feature in feature_names:
            # need to check which are pastel_functions and convert to Callables
            if feature in pastel_functions.__all__:
                new_model[getattr(pastel_functions, str(feature))] = 0.0
            else:
                new_model[feature] = 0.0
        new_model[BiasType.BIAS] = 0.0
        return Pastel(new_model)

    @staticmethod
    def load_model(model_file: str) -> "Pastel":
        """Load model from JSON file. Convert any functions in the model
        from their names to Callable functions."""

        with open(model_file, "rt", encoding="utf-8") as json_in:
            model_json = json.load(json_in)
        # replace function names with function objects found in pastel_functions module
        new_model = {}
        for feature, weight in model_json.items():
            if feature in pastel_functions.__all__:
                new_model[getattr(pastel_functions, feature)] = weight
            elif feature == "bias":
                new_model[BiasType.BIAS] = weight
            else:
                new_model[feature] = weight

        return Pastel(new_model)

    def save_model(self, model_path: str) -> None:
        """
        Save the questions, functions and associated weights to a local JSON file
        Convert callables to their names (strings) first
        """

        # Store the name of each function; all functions are in pastel_functions
        # so we know where to find them after re-loading a model.
        model_json = dict()
        for feature, weight in self.model.items():
            if isinstance(feature, BiasType):
                model_json["bias"] = float(weight)
            if isinstance(feature, str):
                model_json[feature] = float(weight)
            if callable(feature):
                model_json[feature.__name__] = float(weight)
        with open(model_path, "wt", encoding="utf-8") as json_out:
            json.dump(model_json, json_out, indent=2)

    def get_bias(self) -> float:
        """Return just the bias weight"""
        # Every model will have a bias term. If this returns a key error, something's wrong with the model itself
        return self.model[BiasType.BIAS]

    def get_questions(self) -> list[str]:
        """Return just the questions of a model as a list of strings.
        (No weights are returned, nor are the bias term or function components)"""
        questions = []
        for feature in self.model.keys():
            if isinstance(feature, str):
                questions.append(feature)

        return questions

    def get_functions(self) -> list[Callable[[str], float]]:
        """Return just the functions of a model as a list of strings.
        (No weights are returned, nor are the bias term or question components)"""
        functions = []
        for feature in self.model.keys():
            if callable(feature):
                functions.append(feature)

        return functions

    def make_prompt(self, sentence: str) -> str:
        """Makes a prompt for a single given sentence."""

        prompt = """
    Your task is to answer a series of questions about a sentence. Ensure your answers are truthful and reliable.
    You are expected to answer with ‘Yes’ or ‘No’ but you are also allowed to answer with ‘Unsure’ if you do not
    have enough information or context to provide a reliable answer.
    Your response should be limited to the question number and yes/no/unsure.
    Example output:
    0. Yes
    1. Yes
    2. No

    Here are the questions:
    [QUESTIONS]

    Here is the sentence: ```[SENT1]```

    """
        # extract the PastelFeatures whose type is string
        prompt = prompt.replace(
            "[QUESTIONS]",
            "\n".join(
                [f"Question {idx} {q}" for idx, q in enumerate(self.get_questions())]
            ),
        )
        prompt = prompt.replace("[SENT1]", sentence)

        return prompt

    @staticmethod
    def _label_mapping(label: str) -> float:
        """Map yes/no/other response to 1/0/0.5 respectively.
        If model responds 'unsure', 'don't know', 'uncertain' etc. then return 0.5.
        """
        label_map = {"y": 1.0, "n": 0.0}
        return label_map.get(label[0].lower(), 0.5)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before=log_retry_attempt,
    )
    async def _get_answers_for_single_sentence(
        self, sentence: str
    ) -> dict[FEATURE_TYPE, float]:
        sent_answers: Dict[FEATURE_TYPE, float] = {}
        # First, get answers to all the questions from genAI:
        prompt = self.make_prompt(sentence)
        raw_output = await run_prompt(prompt)
        raw_output = raw_output.strip().lower()
        if "question" in raw_output:
            output = raw_output[raw_output.index("0") :]
        else:
            output = raw_output
        answers = output.split("\n")  # e.g. ["1. yes", "2. no"]

        if len(answers) == len(self.get_questions()):
            for q, a in zip(self.get_questions(), answers):
                sent_answers[q] = self._label_mapping(a.split()[1])

        else:
            raise ValueError(
                f"Failed to parse output for the sentence: {sentence}. Output received: {output}"
            )
        # Second, get values from the functions
        for f in self.get_functions():
            sent_answers[f] = f(sentence)

        return sent_answers

    async def get_answers_to_questions(
        self, sentences: list[str]
    ) -> dict[str, dict[FEATURE_TYPE, float]]:
        """Embed each example into the prompt and pass to genAI.
        For each sentence, this Returns a dictionary mapping features to scores."""

        jobs = [
            self._get_answers_for_single_sentence(sentence) for sentence in sentences
        ]
        answers = await asyncio.gather(*jobs, return_exceptions=True)

        # return the answers which didn't cause an exception
        return {
            s: a for s, a in zip(sentences, answers) if not isinstance(a, BaseException)
        }

    def quantify_answers(self, answers: list[dict[FEATURE_TYPE, float]]) -> ARRAY_TYPE:
        """Build numpy array of answers from list of dicts of answers, with one
        dict per sentence.
        Output array will have one row per sentence and one col per feature
        AND the order should match the features in the model, complete with bias column.
        """
        all_answers = []
        for sentence_answers in answers:
            numeric_answers = [0.0] * len(self.model)
            # read through dict of features, getting answer for each one = column
            for idx, feature in enumerate(self.model.keys()):
                if feature == BiasType.BIAS:
                    # We don't get an "answer" for the bias term - it's always 1.0
                    numeric_answers[idx] = 1.0
                else:
                    numeric_answers[idx] = sentence_answers[feature]

            # that's one row done... need to build a whole array!
            all_answers.append(numeric_answers)
        X = np.array(all_answers)
        return X

    def get_scores_from_answers(
        self, answers: list[dict[FEATURE_TYPE, float]]
    ) -> ARRAY_TYPE:
        """Return the predicted score for each sentence.
        This is a linear regression model so the answers are theoretically unbounded,
        but will typically be in the range of the training data.
        answers_num: a numeric vector representing the answers to each question in turn,
        with 1.0 meaning 'yes' and 0.0 meaning 'no'. This will typically be the
        output from get_answers_to_questions()"""

        if sum([abs(w) for w in self.model.values()]) == 0:
            raise ValueError("Must train weights before predicting.")

        X = self.quantify_answers(answers)

        # then calculate & return the dot product, giving one score per sentence:
        weights = np.array(list(self.model.values()))
        scores = X.dot(weights)
        return scores

    async def make_predictions(self, sentences: list[str]) -> ARRAY_TYPE:
        """Use the Pastel questions and weights model to generate
        a score for each of a list of sentences."""
        answers = await self.get_answers_to_questions(sentences)
        if answers:
            scores = self.get_scores_from_answers(list(answers.values()))
        else:
            scores = np.array([])

        scores_dict = {}
        for sentence, score in zip(answers.keys(), scores):
            scores_dict[sentence] = float(score)

        for sentence in sentences:
            if sentence not in scores_dict:
                scores_dict[sentence] = 0.0

        return np.array([scores_dict[sentence] for sentence in sentences])
