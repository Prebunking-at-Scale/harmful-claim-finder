import os
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]
VIDEO_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

DEFAULT_PARAMETERS = {
    "candidate_count": 1,
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 1,
}

DEFAULT_SYSTEM_INSTRUCTION = """
You are a fact checker.
Ensure your answers are clear and correct.
Always return output in the requested format.
When asked to extract results from non-English text, return results in the original language, and don't translate them into English.
"""


class GeminiError(Exception):
    """
    Exception raised when something goes wrong with Gemini.
    """


class ModelConfig(BaseModel):
    """
    Config for a Gemini model.

    Attributes:
    -----------
    project: str
        The name of the project to run the model in.
    location: str
        The Google Cloud location at which to run the model.
        Different locations are capable of running different models.
        See `location info`_ in docs.
    model_name: str
        The name of the model you want to use.
        For example: `gemini-2.0-flash-lite`.

    .. _location info: https://cloud.google.com/vertex-ai/docs/general/locations
    """

    project: str
    location: str
    model_name: str


def generate_model_config() -> ModelConfig:
    """
    Generates a new model config from environment variables:
    `GEMINI_PROJECT`, `GEMINI_LOCATION`, and `GEMINI_MODEL`.
    """
    try:
        return ModelConfig(
            project=os.environ["GEMINI_PROJECT"],
            location=os.environ["GEMINI_LOCATION"],
            model_name=os.environ["GEMINI_MODEL"],
        )
    except KeyError as exc:
        message = (
            "You need to set the following environment variables: "
            + "GEMINI_PROJECT, "
            + "GEMINI_LOCATION, "
            + "GEMINI_MODEL"
        )
        raise GeminiError(message) from exc


async def run_prompt(
    prompt: str,
    video_uri: str | None = None,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    output_schema: types.SchemaUnion | None = None,
    generation_config: dict[str, Any] = DEFAULT_PARAMETERS,
    safety_settings: list[types.SafetySetting] = DEFAULT_SAFETY_SETTINGS,
    model_config: ModelConfig | None = None,
) -> str:
    """
    Runs a prompt through the model.

    Parameters
    ----------
    prompt: str
        The prompt given to the model
    video_uri: str
        A Google Cloud URI for a video that you want to prompt.
    system_instruction: str
        Optionally provide a system instruction.
    output_schema: types.SchemaUnion
        A valid schema for the model output.
        Generally, we'd recommend this being a pydantic BaseModel inheriting class,
        which defines the desired schema of the model output.
        ```python
        from pydantic import BaseModel, Field

        class Movie(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the film was released in the UK")

        schema = Movie
        # or
        schema = list[Movie]
        ```
        Use this if you want structured JSON output.
    generation_config: dict[str, Any]
        The parameters for the generation. See the docs (`generation config`_).
    safety_settings: dict[generative_models.HarmCategory, generative_models.HarmBlockThreshold]
        The safety settings for generation. Determines what will be blocked.
        See the docs (`safety settings`_)
    model_config: ModelConfig | None
        The config for the Gemini model.
        Specifies project, location, and model name.
        If None, will attempt to use environment variables:
        `GEMINI_PROJECT`, `GEMINI_LOCATION`, and `GEMINI_MODEL`.

    Returns
    -------
    The text output of the Gemini model.

    .. _generation config: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig
    .. _safety settings: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters
    """
    # make a copy of the generation config so it doesn't change between runs
    built_gen_config = {**generation_config}
    if model_config is None:
        model_config = generate_model_config()

    client = genai.Client(
        vertexai=True,
        project=model_config.project,
        location=model_config.location,
    )

    # construct the input, adding the video if provided
    parts = []
    if video_uri:
        parts.append(types.Part.from_uri(file_uri=video_uri, mime_type="video/mp4"))

    parts.append(types.Part.from_text(text=prompt))

    # define the schema for the output of the model
    if output_schema:
        built_gen_config["response_mime_type"] = "application/json"
        built_gen_config["response_schema"] = output_schema

    response = await client.aio.models.generate_content(
        model=model_config.model_name,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            **built_gen_config,
        ),
    )

    if response.candidates and response.text and isinstance(response.text, str):
        return response.text

    raise GeminiError(f"No model output: possible reason: {response.prompt_feedback}")
