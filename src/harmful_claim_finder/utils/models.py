from typing import Any, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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


class VideoClaims(BaseModel):
    video_id: UUID
    claim: str  # The claim made in the video
    start_time_s: float  # When in the video the claim starts
    metadata: dict[str, Any] = {}  # Additional metadata about the claim


class TranscriptSentence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: str  # Speech-to-text, OCR, etc
    text: str  # The actual text of the sentence
    start_time_s: float  # Start time in seconds
    metadata: dict[str, Any] = {}
