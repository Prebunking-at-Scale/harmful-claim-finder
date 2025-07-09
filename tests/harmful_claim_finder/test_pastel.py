import json
import tempfile

import numpy as np
import pytest

from harmful_claim_finder.pastel.pastel import BiasType, Pastel

# mypy: ignore-errors
# getting "Untyped decorator makes function ... untyped " so ignoring for now:

Q1 = "Is the statement factual?"
Q2 = "Does the statement contain bias?"


@pytest.fixture
def pastel_instance() -> Pastel:
    pasteliser = Pastel(
        {
            BiasType.BIAS: 1.0,
            Q1: -3.0,
            Q2: 2.0,
        }
    )
    return pasteliser


def test_load_file(pastel_instance: Pastel) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        model = {
            "bias": 1.0,
            Q1: -3.0,
            Q2: 2.0,
        }
        json.dump(model, temp_file)
    loaded: Pastel = Pastel.load_model(temp_file.name)
    assert loaded.model == pastel_instance.model


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
    answers = [{Q1: 1, Q2: 1}, {Q1: 0, Q2: 1}]
    scores = pastel_instance.get_scores_from_answers(answers)
    expected_scores = np.array([0.0, 3.0])
    # [1.0 (=bias) + -3.0*1.0 + 2.0*1 = 0.0 ,
    #  1.0 + -3.0 * 0 + 2.0*1 = 3.0
    assert np.allclose(scores, expected_scores)


def test_get_scores_from_answers_no_weights(pastel_instance: Pastel) -> None:
    for k in pastel_instance.model.keys():
        pastel_instance.model[k] = 0.0
    answers = [{Q1: 1, Q2: 1}, {Q1: 0, Q2: 1}]
    with pytest.raises(ValueError):
        pastel_instance.get_scores_from_answers(answers)


def test_quantify_answers(pastel_instance: Pastel) -> None:
    answers = [{Q1: 1.0, Q2: 0.0}, {Q1: 1.0, Q2: 1.0}]
    numeric_answers = pastel_instance.quantify_answers(answers)
    print(numeric_answers)
    # One row of output per sentence (i.e. input dict):
    assert numeric_answers.shape[0] == len(answers)
    # First column is bias term so should be all 1's:
    # (NB: Model above defines first term is bias)
    assert all(x == 1 for x in numeric_answers[:, 0])
    # Given no sentences, return no answers
    assert pastel_instance.quantify_answers([]).shape[0] == 0
