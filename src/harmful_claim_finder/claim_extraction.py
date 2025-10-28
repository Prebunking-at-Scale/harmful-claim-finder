import json
import logging
import traceback
from textwrap import dedent
from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from harmful_claim_finder.utils.gemini import run_prompt
from harmful_claim_finder.utils.models import (
    ClaimExtractionError,
    TranscriptSentence,
    VideoClaims,
)
from harmful_claim_finder.utils.parsing import parse_model_json_output
from harmful_claim_finder.utils.sentence_linking import link_quotes_and_sentences

_logger = logging.getLogger(__name__)


class TextClaimSchema(BaseModel):
    claim: str = Field(
        description=(
            "The claim being made. "
            "Do not change the meaning of the claim, "
            "but rephrase to make the claim clear without context. "
            "The claim should have enough context to make sense "
            "to a fact checker without seeing the whole transcript. "
            "Must be in the same language as the original transcript."
            "Do not translate into English."
        )
    )
    original_text: str = Field(
        description=(
            "The original chunk of text containing the claim, "
            "exactly as it appears in the provided transcript, "
            "in the original language. "
            "If a claim does not neatly appear in one span of text, "
            "include the most representative span of text available."
        )
    )
    topics: list[str] = Field(
        description=(
            "A list of topics which the claim relates to."
            "The topics should be based on those described in the JSON keywords"
            " you were given."
        )
    )


class VideoClaimSchema(BaseModel):
    claim: str = Field(
        description=(
            "The claim being made. "
            "Do not change the meaning of the claim, "
            "but rephrase to make the claim clear without context. "
            "The claim should have enough context to make sense "
            "to a fact checker without seeing the whole video. "
            "Must be in the same language as the original text."
            "Do not translate into English."
        )
    )
    original_text: str = Field(
        description=(
            "The original sentence containing the claim, "
            "exactly as it appears in the input, in the original language."
            "If the claim is made non-verbally, describe how the claim is made."
        )
    )
    timestamp: float = Field(
        description=(
            "how far through the video does the claim start? Give value in seconds."
        )
    )
    duration: float = Field(
        description="How long, in seconds, is the claim made for?",
    )

    topics: list[str] = Field(
        description=(
            "A list of topics which the claim relates to."
            "The topics should be based on those described by the provided keywords."
            "This should not be []!"
        )
    )


CLAIMS_INSTRUCTION_TEXT = dedent(
    """
    You are assisting a fact checker.
    Ensure your answers are clear and correct.
    Always return output in the requested format.
    Do not translate results into English.

    Find the main claims made in the provided text which relate to the provided topics.
    Include claims that are significant to the overall narrative of the text.
    Include no more than 20 of the most significant claims.
    Do not include claims if they do not relate to any of the provided topics.

    Topics are defined by a set of keywords.
    If a claim relates to one of the keywords, then it counts as being of that topic.
    The claim does not have to contain the exact word, but should contain a very similar
     word, or be on a similar subject.
    A claim can have multiple topics.
    Do not include a claim if it relates to none of the topics.

    The keywords are defined in a dictionary.
    The keys of the dictionary are the topic names.
    The values are the keywords, which define the topics.

    Claims should contain enough context to make sense in isolation.
    A fact checker needs to be able to understand a single claim without necessarily seeing the others.
    For example, if quoting a person or a document, make it clear what that is.
    If something has a causal relationship, make both the alleged claim and effect clear.

    Once you've found all the claims, check the following:
    - Are there any duplicate claims? If there are, only include the most significant instance of the claim.
    - Do all the claims contain enough context to be understood by a fact checker? If not, go back and fill out any missing context.
    - Are all claims in the language they appeared in the original transcript? If not, go back and fix this.
    - Double check that the claims are all the same language as their corresponding quotes.
    """
)

CLAIMS_PROMPT_TEXT = dedent(
    """
    Here are the keywords:
    {KEYWORDS}

    Here is the text:
    ```
    {TEXT}
    ```
    """
)


CLAIMS_INSTRUCTION_VIDEO = dedent(
    """
    You are assisting a fact checker.
    Ensure your answers are clear and correct.
    Always return output in the requested format.
    Do not translate results into English.

    Find all the claims made in this video which relate to the provided topics.
    Include claims that are significant to the overall narrative of the text.
    Include no more than 20 of the most significant claims.
    Do not include claims if they do not relate to any of the provided topics.
    Include both spoken claims, and claims made visually.

    Topics are defined by a set of keywords.
    If a claim relates to one of the keywords, then it counts as being of that topic.
    The claim does not have to contain the exact word, but should contain a very similar
     word, or be on a similar subject.
    A claim can have multiple topics.

    The keywords are defined in a dictionary.
    The keys of the dictionary are the topic names.
    The values are the keywords, which define the topics.

    Claims should contain enough context to make sense in isolation.
    A fact checker needs to be able to understand a single claim without necessarily seeing the others.
    For example, if quoting a person or a document, make it clear what that is.
    If something has a causal relationship, make both the alleged claim and effect clear.

    Once you've found all the claims, check the following:
    - Are there any duplicate claims? If there are, only include the most significant instance of the claim.
    - Do all the claims contain enough context to be understood by a fact checker? If not, go back and fill out any missing context.
    - Are all claims in the language they appeared in the original video? If not, go back and fix this.
    - Double check that the claims are all the same language as their corresponding quotes.
    """
)

CLAIMS_PROMPT_VIDEO = dedent(
    """
    Here are the keywords:
    {KEYWORDS}
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


def _get_timestamps(
    claims: list[TextClaimSchema],
    transcript: list[TranscriptSentence],
) -> dict[str, float]:
    sentences = [s.text for s in transcript]
    quotes = [claim.original_text for claim in claims]
    linked = link_quotes_and_sentences(quotes, sentences, 70)
    quote_timestamps = {
        claims[quote_idx].original_text: transcript[sentence_idx].start_time_s
        for quote_idx, sentence_idx, _ in linked
    }
    return quote_timestamps


def _parse_transcript_claims(
    genai_response: str, transcript: list[TranscriptSentence]
) -> list[VideoClaims]:
    parsed = parse_model_json_output(genai_response)
    parsed = cast(list[dict[str, Any]], parsed)
    genai_claims = []
    for claim_dict in parsed:
        try:
            genai_claims.append(TextClaimSchema(**claim_dict))
        except ValidationError:
            _logger.info(
                f"Skipped malformed json: {json.dumps(claim_dict, ensure_ascii=False)}"
            )
    timestamp_map = _get_timestamps(genai_claims, transcript)

    most_recent_time = 0.0
    output_claims = []
    for claim in genai_claims:
        new_timestamp = timestamp_map.get(claim.original_text, most_recent_time)
        most_recent_time = new_timestamp
        output_claims.append(
            VideoClaims(
                video_id=transcript[0].id,
                claim=claim.claim,
                start_time_s=new_timestamp,
                metadata={"quote": claim.original_text, "topics": claim.topics},
            )
        )
    return output_claims


async def _get_transcript_claims(
    transcript: list[TranscriptSentence],
    keywords: dict[str, list[str]],
) -> list[VideoClaims]:
    transcript_text = " ".join([s.text for s in transcript])
    prompt = CLAIMS_PROMPT_TEXT.replace("{TEXT}", transcript_text)
    prompt = prompt.replace("{KEYWORDS}", json.dumps(keywords))
    response = await run_prompt(
        prompt,
        system_instruction=CLAIMS_INSTRUCTION_TEXT,
        output_schema=list[TextClaimSchema],
    )
    try:
        claims = _parse_transcript_claims(response, transcript)
    except ValueError:
        _logger.info(f"Parsing error: {traceback.format_exc()}")
        fixed_response = await run_prompt(
            FIX_JSON.replace("{TEXT}", response),
            output_schema=list[TextClaimSchema],
        )
        claims = _parse_transcript_claims(fixed_response, transcript)

    return claims


async def extract_claims_from_transcript(
    transcript: list[TranscriptSentence],
    keywords: dict[str, list[str]],
    max_attempts: int = 1,
) -> list[VideoClaims]:
    """
    Extract claims made in a video transcript.

    Args:
        transcript: list[str]
            A list of sentences in the transcript.
        keywords (dict[str, list[str]]):
            A {topic: keywords} dictionary containing the kw for each topic. E.g.
            ```python
            {
                "crime": ["police", "robbers"],
                "health": ["doctor", "hospital"],
            }
            ```
        max_attempts: int
            The number of times the extraction will be attempted upon failure.

    Returns:
        list[VideoClaim]: A list of claims found in the transcript.
    """
    for _ in range(max_attempts):
        try:
            return await _get_transcript_claims(transcript, keywords)
        except Exception as exc:
            _logger.info(f"Error raised while running claim extraction: {repr(exc)}")
            traceback.print_exc()

    raise ClaimExtractionError(f"Claim extraction failed {max_attempts} times.")


def _parse_video_claims(genai_response: str, video_id: UUID) -> list[VideoClaims]:
    parsed = parse_model_json_output(genai_response)
    parsed = cast(list[dict[str, Any]], parsed)
    genai_claims = [VideoClaimSchema(**claim) for claim in parsed]
    output_claims = [
        VideoClaims(
            video_id=video_id,
            claim=claim.claim,
            start_time_s=claim.timestamp,
            metadata={
                "quote": claim.original_text,
                "topics": claim.topics,
            },
        )
        for claim in genai_claims
    ]
    return output_claims


async def _get_video_claims(
    video_id: UUID, video_uri: str, keywords: dict[str, list[str]]
) -> list[VideoClaims]:
    response = await run_prompt(
        CLAIMS_PROMPT_VIDEO.replace("{KEYWORDS}", json.dumps(keywords)),
        video_uri=video_uri,
        system_instruction=CLAIMS_INSTRUCTION_VIDEO,
        output_schema=list[VideoClaimSchema],
    )
    try:
        claims = _parse_video_claims(response, video_id)
    except ValueError:
        _logger.info(f"Parsing error: {traceback.format_exc()}")
        fixed_response = await run_prompt(
            FIX_JSON.replace("{TEXT}", response),
            output_schema=list[VideoClaimSchema],
        )
        claims = _parse_video_claims(fixed_response, video_id)
    return claims


async def extract_claims_from_video(
    video_id: UUID,
    video_uri: str,
    keywords: dict[str, list[str]],
    max_attempts: int = 1,
) -> list[VideoClaims]:
    """
    Extract claims made in a video.
    The claims can be audio or visual.

    Args:
        video_id: UUID
            The id of the video being processed.
        video_url: str
            A URI to a video in a Google Cloud Bucket.
            The file should be an mp4.
        keywords (dict[str, list[str]]):
            A {topic: keywords} dictionary containing the kw for each topic. E.g.
            ```python
            {
                "crime": ["police", "robbers"],
                "health": ["doctor", "hospital"],
            }
            ```
        max_attempts: int
            The number of times the extraction will be attempted upon failure.

    Returns:
        list[VideoClaim]: A list of claims found in the video.
    """
    for _ in range(max_attempts):
        try:
            return await _get_video_claims(video_id, video_uri, keywords)
        except Exception as exc:
            _logger.info(f"Error raised while running claim extraction: {repr(exc)}")
            traceback.print_exc()

    raise ClaimExtractionError(f"Claim extraction failed {max_attempts} times.")
