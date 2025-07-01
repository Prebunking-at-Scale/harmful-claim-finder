# Functions called by Pastel models should take a string and return a float
# The output will be multiplied by the corresponding weight defined in the Pastel model


def is_short(text: str) -> float:
    """Demo function: is the text short?"""
    return float(len(text) < 30)
