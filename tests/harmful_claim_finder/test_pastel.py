import json
import tempfile

import numpy as np
import pytest

from harmful_claim_finder.pastel.pastel import Pastel

# mypy: ignore-errors
# getting "Untyped decorator makes function ... untyped " so ignoring for now:


@pytest.fixture
def pastel_instance() -> Pastel:
    pasteliser = Pastel.from_dict(
        {
            "bias": 1,
            "questions": {
                "Is the statement factual?": -3.1,
                "Does the statement contain bias?": 2.1,
            },
        }
    )
    return pasteliser


def test_load_file(pastel_instance: Pastel) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        model = {
            "bias": 1,
            "questions": {
                "Is the statement factual?": -3.1,
                "Does the statement contain bias?": 2.1,
            },
        }
        json.dump(model, temp_file)
        print(f"Wrote to {temp_file.name}")
    loaded: Pastel = Pastel.load_model(temp_file.name)
    assert loaded.questions == pastel_instance.questions
    assert (loaded.weights == pastel_instance.weights).all()


def test_make_prompt(pastel_instance: Pastel) -> None:
    sentence = "The sky is blue."
    prompt = pastel_instance.make_prompt(sentence)
    assert "[QUESTIONS]" not in prompt
    assert "[SENT1]" not in prompt
    assert "The sky is blue." in prompt
    assert "Is the statement factual?" in prompt
    assert "Does the statement contain bias?" in prompt
    assert "Is this a load of old nonsense" not in prompt


def test_get_scores_from_answers(pastel_instance: Pastel) -> None:
    pastel_instance.weights = np.array([0.5, 1.0, 1.5])
    answers = np.array([[1.0, 0.0]])
    scores = pastel_instance.get_scores_from_answers(answers)
    expected_scores = np.array([1.5])  # 0.5 (=bias) + 1.0*1.0 + 1.5*0.0
    assert np.allclose(scores, expected_scores)


def test_get_scores_from_answers_no_weights(pastel_instance: Pastel) -> None:
    answers = np.array([[1.0, 0.0]])
    pastel_instance.weights = [0.0, 0.0]
    with pytest.raises(ValueError):
        pastel_instance.get_scores_from_answers(answers)
