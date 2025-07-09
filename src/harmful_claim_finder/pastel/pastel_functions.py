# Functions called by Pastel models should take a string and return a float.
# The output will be multiplied by the corresponding weight defined in the Pastel model.
# The plan is to enhance these with functions for claim-types and news-categories,
# among others.

__all__ = ["is_short", "has_number"]


def is_short(text: str) -> float:
    """Demo function: is the text short?"""
    return float(len(text) < 30)


def has_number(text: str) -> float:
    """Is there a number (0-9) in this sentence?"""
    return any(char.isdigit() for char in text)
