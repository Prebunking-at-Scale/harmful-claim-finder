"""
This demo script runs the transcript inference script, which pipes keywords into PASTEL for a transcript.
"""

import asyncio
from pprint import pp
from uuid import UUID

from harmful_claim_finder.transcript_inference import get_claims
from harmful_claim_finder.utils.models import TranscriptSentence

video_id = UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
sentences = [
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=0,
        text="But in words that could further enrage his critics, Starmer insisted that new migrants must “learn the language and integrate” once in the UK.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=1,
        text="He said: “Britain is an inclusive and tolerant country, but the public expect that people who come here should be expected to learn the language and integrate.”",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=2,
        text="Net migration, the difference between the number of people moving to the UK and the number leaving, was 728,000 in the 12 months to June 2024.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=3,
        text="Hamas has released Edan Alexander, a Israeli American hostage held in Gaza who was taken captive while serving in the Israeli army during Hamas's attack on 7 October.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=4,
        text="Israel claimed 20 of its warplanes on Monday had completely destroyed the Houthi-held port of Hodeidah, as well as a nearby cement factory.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=5,
        text="Israel was shocked that its air defences were penetrated by a single Houthi missile, but Iran believes Israel is escalating the crisis in an attempt to disrupt the negotiations between the US and Iran over its nuclear programme.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=6,
        text="Giving obese children weight loss jabs works and could help avoid arguments over mealtimes, according to research.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=7,
        text="The clinicians found that nearly a third of these children dropped enough weight to improve their health, compared with about 27% in earlier treated groups with no access to the drugs.",
    ),
    TranscriptSentence(
        video_id=video_id,
        id=video_id,
        source="video",
        start_time_s=8,
        text="Manchester United intend to retain Ruben Amorim as head coach next season even if they lose the Europa League final to Tottenham.",
    ),
]

if __name__ == "__main__":
    kw = {
        "health": ["doctor", "health", "hospital", "nurses", "obesity", "medicine"],
        "war": ["gun", "bomb", "war", "warplanes", "army", "strike", "attack"],
        "migration": ["crossings", "migrants", "immigration", "migration"],
    }

    result = asyncio.run(get_claims(kw, sentences), debug=True)
    pp([claim.model_dump() for claim in result])
