import json
import os
import time
from pathlib import Path
from typing import TypedDict

from youtube_transcript_api import YouTubeTranscriptApi

OUT_DIRECTORY = "data/example_transcripts"


class TranscriptMeta(TypedDict):
    id: str
    lang: str


youtube_ids: list[TranscriptMeta] = [
    {"id": "uIFSvnB1WlQ", "lang": "en"},
    {"id": "8PqlzOVfciQ", "lang": "en"},
    {"id": "kMWlOcjBdMc", "lang": "en"},
    {"id": "EYHPED0r9Uw", "lang": "en"},
    {"id": "Q0dznt5yegc", "lang": "en"},
    {"id": "2508ZPcN9PM", "lang": "en"},
    {"id": "Ih5cz-Qwa6U", "lang": "en"},
    {"id": "FU7r3yGjjWc", "lang": "en"},
    {"id": "ZVkpQKRgHvU", "lang": "en"},
    {"id": "noNa3WUGpC8", "lang": "en"},
    {"id": "cExSN3bvSk0", "lang": "en"},
    {"id": "ooYhDNX6Xy0", "lang": "en"},
    {"id": "paRtNOFG95I", "lang": "en"},
    {"id": "9V1U_hnxEjo", "lang": "en"},
    {"id": "FCgea5o3ALU", "lang": "en"},
    {"id": "srEZ7V6Z-d0", "lang": "en"},
    {"id": "9RFMQEEmU8g", "lang": "en"},
    {"id": "ClqK7XvfLg0", "lang": "en"},
    {"id": "BoiwgScsV-c", "lang": "en"},
    {"id": "KMHr18e9Nb8", "lang": "en"},
    {"id": "5DdQFkCziw4", "lang": "en"},
    {"id": "79lSqFzHBtc", "lang": "en"},
    {"id": "_9KnhUu7Ba4", "lang": "en"},
    {"id": "rfyOihhAD4A", "lang": "en"},
    {"id": "oX9jZ_w_AmQ", "lang": "en"},
]


def download_transcript(yt_meta: TranscriptMeta):
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(yt_meta["id"], languages=[yt_meta["lang"]])
    return transcript.to_raw_data()


def download_everything():
    for yt_id in youtube_ids:
        try:
            transcript = download_transcript(yt_id)
            Path(os.path.join(OUT_DIRECTORY, f"{yt_id['id']}.json")).write_text(
                json.dumps(transcript, ensure_ascii=False, indent=4)
            )
            print(f"Done {yt_id}")
            time.sleep(1)
        except Exception:
            continue


if __name__ == "__main__":
    download_everything()
