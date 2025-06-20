# First attempt at asking a series of yes/no questions for checkworthiness etc., inspired by Sheffield's PASTEL model
# See paper: https://arxiv.org/abs/2309.07601v3 "Weakly Supervised Veracity Classification with LLM-Predicted Credibility Signals"

import json
from typing import Tuple, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

from harmful_claim_finder.utils.gemini import run_prompt

EXAMPLES_TYPE = Tuple[str, float]
ARRAY_TYPE: TypeAlias = npt.NDArray[np.float64]


class ModelDict(TypedDict):
    bias: float
    questions: dict[str, float]


class Pastel:
    """Uses list of yes/no questions to analyse a piece of text. Each question has an
    associated weight used to generate the final score for the text."""

    def __init__(self, questions: list[str]) -> None:
        """
        Create a new Pastel object from a list of questions.
        The initial weights will be set to zero,
        so the model must be trained before use.
        """
        # Create a new Pastel object from a list of questions. The initial
        # weights will be set to zero, so the model must be trained before use.
        self.questions = questions
        # Initialise weights to zero with an extra one for the bias term:
        self.weights = np.zeros(1 + len(questions))

    @staticmethod
    def from_dict(model_dict: ModelDict) -> "Pastel":
        """
        Create a new Pastel model from a dictionary.

        Args:
            model_dict (ModelDict):
                Dictionary in the format:
                ```
                {
                    "bias": 1.513186513,
                    "questions": {
                        "Answer 'yes' if this sentence is interesting": 0.09631786047,
                        "Does this sentence relate to many people?": 0.4356950359,
                        ...
                    }
                }
                ```
        """
        questions: list[str] = list(model_dict["questions"].keys())
        weights = np.array(
            [model_dict["bias"]] + list(model_dict["questions"].values())
        )
        model = Pastel(questions)
        model.weights = weights
        return model

    @staticmethod
    def load_model(model_file: str) -> "Pastel":
        """Load model questions and weights from local file."""
        with open(model_file, "rt", encoding="utf-8") as json_in:
            model_dict = json.load(json_in)

        return Pastel.from_dict(model_dict)

    def save_model(self, model_path: str) -> None:
        """
        Save the questions and associated weights to a local JSON file
        """
        with open(model_path, "wt", encoding="utf-8") as json_out:
            model = {
                "bias": self.weights[0],
                "questions": {k: v for k, v in zip(self.questions, self.weights[1:])},
            }
            json.dump(model, json_out, indent=2)

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
        prompt = prompt.replace(
            "[QUESTIONS]",
            "\n".join([f"Question {idx} {q}" for idx, q in enumerate(self.questions)]),
        )
        prompt = prompt.replace("[SENT1]", sentence)

        return prompt

    def get_answers_to_questions(self, sentences: list[str]) -> ARRAY_TYPE:
        """Embed each example into the prompt and pass to genAI.
        Returns the unweighted numeric scores of the genAI answer
        to each question."""

        all_answers = []

        def label_mapping(label: str) -> float:
            """Map yes/no/other response to 1/0/0.5 respectively.
            If model responds 'unsure', 'don't know', 'uncertain' etc. then return 0.5.
            """
            label_map = {"y": 1.0, "n": 0.0}
            return label_map.get(label[0].lower(), 0.5)

        for ex in sentences:
            prompt = self.make_prompt(ex)
            raw_output = run_prompt(prompt).strip().lower()
            if "question" in raw_output:
                output = raw_output[raw_output.index("0") :]
            else:
                output = raw_output
            answers = output.split("\n")  # e.g. ["1. yes", "2. no"...]

            if len(answers) == len(self.questions):
                all_answers.append([label_mapping(pl.split()[1]) for pl in answers])
            else:
                raise ValueError(
                    f"Failed to parse output for the sentence: {ex}. Output received: {output}"
                )

        all_answers_arr = np.array(all_answers)
        return all_answers_arr

    def get_scores_from_answers(self, answers_num: ARRAY_TYPE) -> ARRAY_TYPE:
        """Return the predicted score for each sentence.
        This is a linear regression model so the answers are theoretically unbounded,
        but will typically be in the range of the training data.
        answers_num: a numeric vector representing the answers to each question in turn,
        with 1.0 meaning 'yes' and 0.0 meaning 'no'. This will typically be the
        output from get_answers_to_questions()"""

        if sum([abs(w) for w in self.weights]) == 0:
            raise ValueError("Must train weights before predicting.")
        # Add a column of ones (on the left) to X for the bias term
        X = np.hstack([np.ones((answers_num.shape[0], 1)), answers_num])
        # ...then calculate & return the dot product
        scores = X.dot(self.weights)
        return scores

    def make_predictions(self, sentences: list[str]) -> ARRAY_TYPE:
        """Use the Pastel questions and weights model to generate
        a score for each of a list of sentences."""
        answers = self.get_answers_to_questions(sentences)
        scores = self.get_scores_from_answers(answers)
        return scores
