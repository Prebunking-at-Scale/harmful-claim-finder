# Harmful Claim Finder

This repo is part of the wider Prebunking at Scale project, and contains the functionality for finding claims in short form video.

### What you need to know

In [transcript_inference.py](src/harmful_claim_finder/transcript_inference.py#L49) you'll find a function "[get_claims](src/harmful_claim_finder/transcript_inference.py#L49)".
This is the function that will give you the claims from a transcript.

You can use it like this:
```python
import asyncio
from pprint import pp
from uuid import UUID

from harmful_claim_finder.transcript_inference import get_claims
from harmful_claim_finder.utils.models import TranscriptSentence


sentences = [
    TranscriptSentence(
        video_id=UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        source="video",
        start_time_s=6,
        text="Giving obese children weight loss jabs works and could help avoid arguments over mealtimes, according to research.",
    ),
    TranscriptSentence(
        video_id=UUID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        source="video",
        start_time_s=7,
        text="The clinicians found that nearly a third of these children dropped enough weight to improve their health, compared with about 27% in earlier treated groups with no access to the drugs.",
    ),
]

if __name__ == "__main__":
    kw = {
        "health": ["doctor", "health", "hospital", "nurses", "obesity", "medicine"],
        "war": ["gun", "bomb", "war", "warplanes", "army", "strike", "attack"],
        "migration": ["crossings", "migrants", "immigration", "migration"],
    }

    countries = ["GBR", "USA"]

    result = asyncio.run(get_claims(kw, sentences, countries), debug=True)
    pp([claim.model_dump() for claim in result])

```

### How it works
This is the basic process for finding claims in transcripts.

1. Assign a list of topics to each sentence of the transcript using a Gemini prompt.
Topics are defined by the provided keywords.
Sentences are given a topic if they contain words similar to those listed as keywords for that topic.
This is done by [the topic filter](/src/harmful_claim_finder/keyword_filter/topic_keyword_filter.py).

2. For each sentence which has a topic, calculate a score which says how "checkworthy" it is.
This score is between 0 and 5, with 5 meaning it is definitely worth checking, and 0 meaning definitely not worth checking.
The score is calculated using the PASTEL method, which involves asking a series of yes/no questions to a large language model and learning weights for each question.
This score assignment is handled in [pastel/inference.py](src/harmful_claim_finder/pastel/inference.py).

3. Each claim with a score is considered a claim.
So we return a list of these scored claims for each transcript.