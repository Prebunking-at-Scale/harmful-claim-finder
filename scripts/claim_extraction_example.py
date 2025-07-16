import asyncio
import json
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from harmful_claim_finder.claim_extraction import (
    extract_claims_from_transcript,
    extract_claims_from_video,
)
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
    "data/example_transcripts/ClqK7XvfLg0.json",
    "data/example_transcripts/9RFMQEEmU8g.json",
    "data/example_transcripts/EYHPED0r9Uw.json",
    "data/example_transcripts/cExSN3bvSk0.json",
    "data/example_transcripts/paRtNOFG95I.json",
    "data/example_transcripts/Ih5cz-Qwa6U.json",
    "data/example_transcripts/srEZ7V6Z-d0.json",
    "data/example_transcripts/ooYhDNX6Xy0.json",
    "data/example_transcripts/2508ZPcN9PM.json",
    "data/example_transcripts/FCgea5o3ALU.json",
    "data/example_transcripts/Q0dznt5yegc.json",
    "data/example_transcripts/79lSqFzHBtc.json",
    "data/example_transcripts/KMHr18e9Nb8.json",
    "data/example_transcripts/FU7r3yGjjWc.json",
    "data/example_transcripts/BoiwgScsV-c.json",
]
videos = [
    "gs://pas-prototyping-storage/ds-test-videos/7304218971153124651.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7149378297489558830.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7172332152292576558.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7234318005587447083.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7236655201992576282.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7282908798186999083.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7300977537717407022.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7328225789827190059.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7331223830645509419.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7337234721497271598.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7357367790744931630.mp4",
]

video_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


def text_example() -> None:
    output = {}
    for transcript_path in transcripts:
        try:
            transcript: list[TranscriptFragment] = json.loads(
                Path(transcript_path).read_text()
            )
            sentences = [
                TranscriptSentence(
                    video_id=video_id,
                    source="",
                    text=fragment["text"],
                    start_time_s=fragment["start"],
                )
                for fragment in transcript
            ]
            claims = asyncio.run(extract_claims_from_transcript(sentences))
            print(f"Found {len(claims)} claims in transcript {transcript_path}")
            jsonable = [claim.model_dump(mode="json") for claim in claims]
            output[transcript_path] = jsonable
        except Exception as exc:
            print(f"Something went wrong with {transcript_path}: {repr(exc)}")
            continue

    Path("output_transcripts.json").write_text(
        json.dumps(output, indent=4, ensure_ascii=False)
    )


def video_example() -> None:
    kw = {
        "vaccines": ["covid-19", "vaxxer", "vaccine"],
        "miracle_cures": ["miracle cure", "magic herbs", "traditional medicine"],
    }
    output = {}
    for video_uri in videos:
        try:
            claims = asyncio.run(extract_claims_from_video(video_id, video_uri, kw))
            print(f"Found {len(claims)} claims in video from {video_uri}")
            jsonable = [claim.model_dump(mode="json") for claim in claims]
            output[video_uri] = jsonable
        except Exception as exc:
            print(f"Something went wrong with {video_uri}: {repr(exc)}")
            continue

    Path("output_videos.json").write_text(
        json.dumps(output, indent=4, ensure_ascii=False)
    )


if __name__ == "__main__":
    # text_example()
    # print("-" * 100)
    video_example()
