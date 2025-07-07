"""Performs linear regression to optimise & save model weights"""

import csv
import logging

import numpy as np
from scipy.optimize import least_squares

from harmful_claim_finder.pastel import pastel

_logger = logging.getLogger(__name__)


def lin_reg(X: pastel.ARRAY_TYPE, y: pastel.ARRAY_TYPE) -> pastel.ARRAY_TYPE:
    """Calculates optimum weight vector of linear regression model. This is the
    best-fit line through the X,y data.
    Minimise squared error for y = w.x
    One column of X should be all 1's corresponding to the bias (or intercept
    term) in the model. Without it, the line would always go through the
    origin (0,0) which is an unnecessary constraint."""

    def residuals(ww: pastel.ARRAY_TYPE) -> pastel.ARRAY_TYPE:
        """Define the residual function.
        This calculates the difference between the predicted values (y) and the actual values X.ww
        The smaller the residuals, the better the fit.
        """
        return X @ ww - y

    # Initial guess for the weight vector (including the bias term)
    w0 = np.ones(X.shape[1])

    # Use least squares to minimize the residuals. This calculates the set of weights that
    # produces the smallest sum of squared errors for the training data - i.e. the best fit.
    result = least_squares(residuals, w0)  # type: ignore

    return result.x


def load_examples(filename: str) -> list[pastel.EXAMPLES_TYPE]:
    """Load examples from file. Each row in the CSV file should be a sentence
    followed by its checkworthy score (e.g. in the range 1-5)"""
    examples = []
    with open(filename, "rt", encoding="utf-8") as fin:
        reader = csv.reader(fin, quoting=csv.QUOTE_ALL)
        for row in reader:
            sentence = row[0]
            label = float(row[1])
            examples.append((sentence, label))
    return examples


def learn_weights(
    training_data_filename: str, pasteliser: pastel.Pastel
) -> pastel.ARRAY_TYPE:
    """Minimise sum squared error of labelled data set to find optimal
    set of weights. Note that first weight is for a constant term, so the
    weight vector is one longer than the number of questions in the prompt."""

    examples = load_examples(training_data_filename)
    answers = pasteliser.get_answers_to_questions([e[0] for e in examples])
    predictions = pasteliser.quantify_answers(answers)
    targs = [e[1] for e in examples]
    targs_arr = np.array(targs)
    pred_arr = np.array(predictions)
    weights = lin_reg(pred_arr, targs_arr)

    for idx, k in enumerate(pasteliser.model.keys()):
        pasteliser.model[k] = weights[idx]
    return weights


def evaluate_weights(test_data_filename: str, model_file: str) -> None:
    """Loads a list of sentences with target scores as test data;
    loads a model (i.e. questions with weight scores); generates predictions
    from the model and compares to the target scores."""
    # TODO: is this even used? why would we use RMS error???
    pasteliser = pastel.Pastel.load_model(model_file)

    examples = load_examples(test_data_filename)
    predictions = pasteliser.make_predictions([e[0] for e in examples])
    targs = [e[1] for e in examples]
    targs_arr = np.array(targs)
    pred_arr = np.array(predictions)
    errors = targs_arr - pred_arr @ pasteliser.weights
    rms_error = np.sqrt(np.mean(errors**2))
    _logger.debug(f"RMS Error: {rms_error:.4f}")
