from harmful_claim_finder.pastel.pastel_functions import is_short
from harmful_claim_finder.pastel import pastel

if __name__ == "__main__":

    # Simple model:
    qs = {"is this about a health condition?": -1, "is this important?": 0.5}
    fs = {is_short: -1.0}
    md = pastel.ModelDict(bias=1, questions=qs, functions=fs)
    p = pastel.Pastel.from_dict(md)

    sentences = [
        "Measles will kill us all!",
        "this is a long and winding sentence but not entirely serious.",
    ]
    p.make_predictions(sentences)
