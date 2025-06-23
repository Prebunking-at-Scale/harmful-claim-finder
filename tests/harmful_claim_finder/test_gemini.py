import os
from harmful_claim_finder.utils.gemini import (
    GeminiError,
    generate_model_config,
)


def test_generate_model_config():
    os.environ["GEMINI_PROJECT"] = "p"
    os.environ["GEMINI_LOCATION"] = "l"
    os.environ["GEMINI_MODEL"] = "m"

    config = generate_model_config()
    assert config.project == "p"
    assert config.location == "l"
    assert config.model_name == "m"


def test_generate_model_config_no_env_vars():
    if "GEMINI_PROJECT" in os.environ:
        os.environ.pop("GEMINI_PROJECT")
    if "GEMINI_LOCATION" in os.environ:
        os.environ.pop("GEMINI_LOCATION")
    if "GEMINI_MODEL" in os.environ:
        os.environ.pop("GEMINI_MODEL")

    try:
        _ = generate_model_config()
    except GeminiError:
        assert True
        return

    assert False
