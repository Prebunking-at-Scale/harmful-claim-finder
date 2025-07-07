import logging
import traceback
from textwrap import dedent
from typing import Any, cast

from pydantic import BaseModel, Field

from harmful_claim_finder.utils.gemini import run_prompt
from harmful_claim_finder.utils.models import ClaimExtractionError
from harmful_claim_finder.utils.parsing import parse_model_json_output

_logger = logging.getLogger(__name__)


class TextClaim(BaseModel):
    claim: str = Field(
        description=(
            "claim being made. "
            "Do not change the meaning of the claim, "
            "but rephrase to make the claim clear without context."
        )
    )
    original_text: str = Field(
        description=(
            "The original sentence containing the claim, "
            "exactly as it appears in the text"
        )
    )


class VideoClaim(BaseModel):
    claim: str = Field(
        description=(
            "claim being made. "
            "Do not change the meaning of the claim, "
            "but rephrase to make the claim clear without context."
        )
    )
    original_text: str = Field(
        description=(
            "The original sentence containing the claim, "
            "exactly as it appears in the input."
            "If the claim is made non-verbally, describe how the claim is made."
        )
    )
    timestamp: str = Field(
        description=(
            "how far through the video was the claim made? Give value in HH:MM:SS"
        )
    )
    duration: int = Field(
        description="How long, in ms, is the claim made for?",
    )


CLAIMS_PROMPT_TEXT = dedent(
    """
    Find the main claims made in the provided text.
    Include any claims that are significant to the overall narrative of the text.
    Include no more than 20 of the most significant claims.

    Here is the text:
    ```
    {TEXT}
    ```
    """
)


CLAIMS_PROMPT_VIDEO = dedent(
    """
    Find all the claims made in this video.
    Include both spoken claims, and claims made visually.
    """
)


FIX_JSON = dedent(
    """
    This JSON is broken. Please fix it, and output valid JSON in the provided format.
    ```
    {TEXT}
    ```
    """
)


def extract_claims_from_transcript(
    transcript: list[str], max_attempts: int = 1
) -> list[TextClaim]:
    """
    Extract claims made in a video transcript.

    Args:
        video_url: list[str]
            A list of sentences in the transcript.
        max_attempts: int
            The number of times the extraction will be attempted upon failure.

    Returns:
        list[VideoClaim]: A list of claims found in the transcript.
    """
    for _ in range(max_attempts):
        try:
            transcript_text = " ".join(transcript)
            prompt = CLAIMS_PROMPT_TEXT.replace("{TEXT}", transcript_text)
            response = run_prompt(prompt, output_schema=list[TextClaim])
            try:
                parsed = parse_model_json_output(response)
                parsed = cast(list[dict[str, Any]], parsed)
                claims = [TextClaim(**claim) for claim in parsed]
            except ValueError:
                _logger.info(f"Parsing error: {traceback.format_exc()}")
                parsed = run_prompt(
                    FIX_JSON.replace("{TEXT}", response),
                    output_schema=list[TextClaim],
                )
                parsed = cast(list[dict[str, Any]], parsed)
                claims = [TextClaim(**claim) for claim in parsed]

            return claims
        except Exception as exc:
            _logger.info(f"Error raised while running claim extraction: {repr(exc)}")
            traceback.print_exc()

    raise ClaimExtractionError(f"Claim extraction failed {max_attempts} times.")


def extract_claims_from_video(
    video_uri: str, max_attempts: int = 1
) -> list[VideoClaim]:
    """
    Extract claims made in a video.
    The claims can be audio or visual.

    Args:
        video_url: str
            A URI to a video in a Google Cloud Bucket.
            The file should be an mp4.
        max_attempts: int
            The number of times the extraction will be attempted upon failure.

    Returns:
        list[VideoClaim]: A list of claims found in the video.
    """
    for _ in range(max_attempts):
        try:
            response = run_prompt(
                CLAIMS_PROMPT_VIDEO,
                video_uri=video_uri,
                output_schema=list[VideoClaim],
            )
            try:
                parsed = parse_model_json_output(response)
                parsed = cast(list[dict[str, Any]], parsed)
                claims = [VideoClaim(**claim) for claim in parsed]
            except ValueError:
                _logger.info(f"Parsing error: {traceback.format_exc()}")
                parsed = run_prompt(
                    FIX_JSON.replace("{TEXT}", response),
                    output_schema=list[VideoClaim],
                )
                parsed = cast(list[dict[str, Any]], parsed)
                claims = [VideoClaim(**claim) for claim in parsed]
            return claims
        except Exception as exc:
            _logger.info(f"Error raised while running claim extraction: {repr(exc)}")
            traceback.print_exc()

    raise ClaimExtractionError(f"Claim extraction failed {max_attempts} times.")
