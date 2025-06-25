from pathlib import Path
from pprint import pp

from harmful_claim_finder.prompts import (
    CLAIMS_PROMPT_TEXT,
    CLAIMS_PROMPT_VIDEO,
    TextClaimSchema,
    VideoClaimSchema,
)
from harmful_claim_finder.utils.gemini import run_prompt
from harmful_claim_finder.utils.parsing import parse_model_json_output

example_transcript_text = "data/example_transcripts/new_statesman_transcript.txt"
example_video = "gs://raphael-test/tiktok/7304218971153124651.mp4"


def extract_claims_from_text():
    transcript_text = Path(example_transcript_text).read_text()
    prompt = CLAIMS_PROMPT_TEXT.replace("{TEXT}", transcript_text)
    response = run_prompt(prompt, output_schema=list[TextClaimSchema])
    parsed = parse_model_json_output(response)
    print(f"Found {len(parsed)} claims in transcript")
    pp(parsed)


def extract_claims_from_video():
    response = run_prompt(
        CLAIMS_PROMPT_VIDEO,
        video_uri=example_video,
        output_schema=list[VideoClaimSchema],
    )
    parsed = parse_model_json_output(response)
    print(f"Found {len(parsed)} claims in transcript")
    pp(parsed)


if __name__ == "__main__":
    extract_claims_from_text()
