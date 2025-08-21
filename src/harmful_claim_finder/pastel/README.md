# Some notes on Pastel models

## Overview

We use Pastel models for estimating checkworthiness of a sentence. Each model consists of a list of questions and a list of functions and a bias term.

The questions are passed to Gemini along with the sentence; Gemini returns a list of "yes" or "no" responses which are then mapped to floats (yes=1.0, no=0.0).

Any functions in the model are given the sentence as input and generate a float in response.

The bias term is a float that is included to make the model a full linear regression model.

All the responses (from questions and functions) are then multiplied by the corresponding weight in the file and summed to give a final score for that sentence.

### Questions

A set of yes/no questions will be passed to Gemini for each sentence.
The responses are converted to floats and used by the regression model to calculate the score.

You can see the questions by looking in [checkworthy_model.json](/src/harmful_claim_finder/pastel/checkworthy_model.json), alongside their corresponding weights.

### Functions

PASTEL can also take the output of functions as input alongside the questions asked to Gemini. 
Each of these functions return a float.

Currently, the production model does not use any of these functions, but examples would be functions which see if a text is short, or contains a number.

## Saving/loading models

Models are stored a JSON files. To achieve this, Callable functions are converted to strings (i.e. the function names) in `save_model()`. When re-loaded (i.e `load_model()`), any key that matches a function name in `pastel_functions.py` is converted to a Callable. Similarly, the bias term is saved with the key `bias` and that's converted to the special bias type on load.

## Usage

Typically, a list of questions and functions is set up and then the optimise_weights script is used to calculate the optimum weights with respect to a given set of labelled examples. 

## Example model

{
  "Does this sentence relate to many people?": 0.520274384101831,
  "Is this sentence about someone's personal experience?": -0.40700720717983807,
  "Does the sentence contain specific numbers or quantities?": 0.2697015041118784,
  "Could believing this claim harm someone's health?": 0.607968463349725,
  "has_number": 0.3732919809848763,
  "bias": 0.8450754791888597
}