from textwrap import dedent
from typing import Any, cast

from pydantic import BaseModel, Field

from harmful_claim_finder.utils.gemini import run_prompt
from harmful_claim_finder.utils.parsing import parse_model_json_output


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


def extract_claims_from_transcript(transcript: list[str]) -> list[TextClaim]:
    transcript_text = " ".join(transcript)
    prompt = CLAIMS_PROMPT_TEXT.replace("{TEXT}", transcript_text)
    response = run_prompt(prompt, output_schema=list[TextClaim])
    parsed = parse_model_json_output(response)
    parsed = cast(list[dict[str, Any]], parsed)
    claims = [TextClaim(**claim) for claim in parsed]
    return claims


def extract_claims_from_video(video_uri: str) -> list[VideoClaim]:
    response = run_prompt(
        CLAIMS_PROMPT_VIDEO,
        video_uri=video_uri,
        output_schema=list[VideoClaim],
    )
    parsed = parse_model_json_output(response)
    parsed = cast(list[dict[str, Any]], parsed)
    claims = [VideoClaim(**claim) for claim in parsed]
    return claims
