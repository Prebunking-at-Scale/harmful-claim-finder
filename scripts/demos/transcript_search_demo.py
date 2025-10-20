import asyncio
import json
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from harmful_claim_finder.transcript_search import get_claims
from harmful_claim_finder.utils.models import TranscriptSentence


class TranscriptFragment(TypedDict):
    text: str
    start: float
    duration: float


transcripts = [
    "data/example_transcripts/9V1U_hnxEjo.json",
    "data/example_transcripts/8PqlzOVfciQ.json",
    "data/example_transcripts/noNa3WUGpC8.json",
    "data/example_transcripts/kMWlOcjBdMc.json",
    "data/example_transcripts/uIFSvnB1WlQ.json",
]

video_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


def find_checkworthy_claims() -> None:
    output = {}
    kw = {
        "vaccines": ["covid-19", "vaxxer", "vaccine"],
        "miracle_cures": ["miracle cure", "magic herbs", "traditional medicine"],
    }
    output = {}
    for transcript_path in transcripts:
        try:
            transcript: list[TranscriptFragment] = json.loads(
                Path(transcript_path).read_text()
            )
            sentences = [
                TranscriptSentence(
                    video_id=video_id,
                    id=video_id,
                    source="",
                    text=fragment["text"],
                    start_time_s=fragment["start"],
                )
                for fragment in transcript
            ]
            claims = asyncio.run(get_claims(kw, sentences))
            print(f"Found {len(claims)} claims in transcript {transcript_path}")
            jsonable = [claim.model_dump(mode="json") for claim in claims]
            output[transcript_path] = jsonable
        except Exception as exc:
            print(f"Something went wrong with {transcript_path}: {repr(exc)}")
            continue

    Path("output_transcript_search_demo.json").write_text(
        json.dumps(output, indent=4, ensure_ascii=False)
    )


if __name__ == "__main__":
    find_checkworthy_claims()
