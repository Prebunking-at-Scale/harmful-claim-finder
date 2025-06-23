from textwrap import dedent

CLAIMS_PROMPT_TEXT = dedent(
    """
    Find the main claims made in the provided text.
    Include any claims that are significant to the overall narrative of the text.
    Include no more than 20 of the most significant claims.

    Return a json list of claims in the following format:
    {
        "claim": <claim being made. Do not change the meaning of the claim, but rephrase to make the claim clear without context.>,
        "original_text": <the original sentence, exactly as it appears in the input, containing the claim>,
    }

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

    Return a json list of claims in the following format:
    {
        "claim": <claim being made. Do not change the meaning of the claim, but rephrase to make the claim clear without context.>,
        "original_text": <the original sentence, exactly as it appears in the input, containing the claim. If the claim is made non-verbally, leave this blank.>,
        "timestamp": "<how far through the video was the claim made? Give value in HH:MM:SS,
        "duration": "How long, in ms, is the claim made for?"
    }
    """
)
