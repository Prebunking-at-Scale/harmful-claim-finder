from typing import TypedDict


class TopicDetectionError(Exception):
    """
    Raised when topic detection fails.
    """


class PastelError(Exception):
    """
    Raised when PASTEL fails.
    """


class ClaimExtractionError(Exception):
    """
    Raised when claim extraction fails.
    """


class CheckworthyError(Exception):
    """
    Raised if checkworthy fails.
    """


class ParsingError(Exception):
    """
    Raised if parsing fails at some point
    """


class CheckworthyResult(TypedDict):
    score: float
    topics: list[str]
