# Simple demos of learning and using a checkworthy-scoring model

import logging
from pathlib import Path

import harmful_claim_finder.pastel.inference as pastel_inference
import harmful_claim_finder.pastel.pastel_functions as pfun
from harmful_claim_finder.pastel import optimise_weights, pastel

logger = logging.getLogger(__name__)
logging.basicConfig(filename="demo.log", encoding="utf-8", level=logging.DEBUG)


print("Inserting demo model file into pastel_inference!")
CHECKWORTHY_MODEL_FILE = "scripts/demos/demo_model_file.json"
pastel_inference.CHECKWORTHY_MODEL_FILE = Path(CHECKWORTHY_MODEL_FILE)
# Note that this model file is not in production so is safe to overwrite or break

MODEL_QUESTIONS = [
    "Answer 'yes' if this sentence is making a specific claim or answer 'no' if it is vague\
     or unclear",
    "Does this sentence relate to many people?",
    "Is this sentence about someone's personal experience?",
    "Does the sentence contain specific numbers or quantities?",
    "Does the sentence contain compare quantities, such as 'more' or 'less'?",
    "Does the sentence discuss superlatives, such as 'biggest ever' or  'fastest growth'?",
    "Does the sentence suggest a course of action?",
    "Does the sentence use emotive language?",
    "Answer 'yes' if this is a general or universal claim or answer 'no' if it is about a \
     single event or individual",
    "Could believing this claim harm someone's health?",
    "Could believing this claim lead to violence",
    "Is this sentence ambiguous?",
    "Is this sentence a joke or satirical?",
    "Is this making a claim that is too good to be true?",
]
MODEL_FUNCTIONS = [pfun.is_short, pfun.has_number]


def demo_inference() -> None:
    """Pass a few examples & see what scores we get"""

    examples = [
        "Over a similar time period, reported mental health problems have also jumped "
        "from 8% to 10% of working-age people to between 13% and 15%, according to "
        "the Institute for Fiscal Studies.",
        "For every 4in increase in height above average, cancer risk increases by 18 "
        "per cent in women and 11 per cent in men, reported researchers at the "
        "Karolinska Institute in Sweden in 2015.",
        "Researchers at Oxford University in 2017 found that every extra 4in of height "
        "above average increases a man's risk of developing aggressive prostate "
        "cancer by 21 per cent and their chance of dying by 17 per cent.",
        "SEVEN in 10 women will experience period pain - often physically and mentally "
        "debilitating - for almost four solid years of their life, according to "
        "research.",
        "It's been 70 years since the Toon celebrated getting their hands on some "
        "silverware, when they beat Manchester City to win the 1955 FA Cup.",
        "Alexander Isak did a very, very cool thing against Virgil van Dijk when they "
        "played in at St James' Park earlier in the season, which finished 3-3..",
        "We've got a fairly similar formation set up for both teams - they're going to "
        "set up as 4-3-3 or 4-2-3-1, fairly similar.",
        "The supplier serves about a quarter of the UK's population, mostly across "
        "London and parts of southern England, and employs 8,000 people.",
        "Environment Secretary Steve Reed has previously said government intervention "
        "in Thames Water would 'cost billions and take years'.",
    ]
    cw_predictor = pastel_inference.CheckworthyClaimDetector(countries=["GBR"])
    scores = cw_predictor.score_sentences(examples)
    _ = [print(f"{s:4.1f} \t{e}") for s, e in sorted(zip(scores, examples))]


def learn_weights(training_examples_file: str, new_model_file: str) -> None:
    """Load annotated sentences; pass them to genAI to get answers to the questions;
    then use the annotations to calculate the optimum weights. Finally save that to
    a file for use as a new model.
    """
    all_features = MODEL_QUESTIONS + MODEL_FUNCTIONS
    pastelizer = pastel.Pastel.from_feature_list(all_features)
    _ = optimise_weights.learn_weights(training_examples_file, pastelizer)
    pastelizer.save_model(new_model_file)


if __name__ == "__main__":
    learn_weights(
        "data/pastel_training/ff_annotations_sample.csv",
        CHECKWORTHY_MODEL_FILE,
    )

    import pprint as pp

    pastelizer = pastel.Pastel.load_model(CHECKWORTHY_MODEL_FILE)
    pp.pprint(pastelizer.model)

    demo_inference()
