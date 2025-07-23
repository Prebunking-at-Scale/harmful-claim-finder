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
    # These are the keywords that we use to decide topics for each sentence.

    countries = ["GBR", "USA"]
    # These are the countries of interest to an organisation
    # If the transcript refers to other countries and not these, it's PASTEL score will be negatively affected.

    result = asyncio.run(get_claims(kw, sentences, countries), debug=True)
    pp([claim.model_dump() for claim in result])

```

### How it works
This is the basic process for finding claims in transcripts.

1. Assign a list of topics to each sentence of the transcript using a Gemini prompt.
Topics are defined by the provided keywords.
Sentences are given a topic if Gemini decides they contain words semantically similar to those listed as keywords for that topic.
This is done by [the topic filter](/src/harmful_claim_finder/keyword_filter/topic_keyword_filter.py).

2. For each sentence which has a topic, calculate a score which says how "checkworthy" it is.
This score is between 1 and 5, with 5 meaning it is definitely worth checking, and 1 meaning definitely not worth checking.
The score is calculated using the PASTEL method, which involves asking a series of yes/no questions to a large language model and learning weights for each question.
This score assignment is handled in [pastel/inference.py](src/harmful_claim_finder/pastel/inference.py).
More information can be found in the [PASTEL subdirectory](/src/harmful_claim_finder/pastel/).

3. Each claim with a score is considered a claim.
So we return a list of these scored claims for each transcript.
> Currently all claims with a score are returned, but the plan is to introduce a minimum score threshold.


### More detail

There are a couple of scripts that are useful to know about in this repo.

[transcript_inference.py](/src/harmful_claim_finder/transcript_inference.py) is the main script to worry about.
This contains a function for finding claims in a transcript, using the process we described above.
([demo](/scripts/demos/transcript_inference_demo.py))

[video_inference.py](/src/harmful_claim_finder/video_inference.py) also finds claims, but using a video directly instead of a transcript.
Instead of checking each sentence for topics, it extracts claims directly from the video.
These claims are then given to PASTEL for scoring like we described [previously](#how-it-works).
The advantage of this method is that it can find multimodal claims, not just things that are easily represented as text in a transcript.
([demo](/scripts/demos/video_inference_demo.py))

#### Components
There are a few different components to claim detection, which I'll describe below:

##### Claim Type Detection
> Not in the MVP

Categorises sentences into claim types.
These are the possible claim types:
```json
[
    "personal",
    "quantity",
    "correlation",
    "rules",
    "predictions",
    "voting",
    "opinion",
    "support",
    "other",
    "not_claim"
]
```
This information can be used by PASTEL as an input.

##### Keyword Filtering
Returns a list of topics for each sentence provided.
Topics are defined by keywords given to it, e.g. 
```json
{
    "health": ["doctor", "measles"],
    "politics": ["parliament", "senate"]
}
```
Sentences that contain similar words to the keywords will be given the relevant topic.
A Gemini model is used for deciding how to assign topics, so it will decide if a given sentence is semantically similar to the keywords.

([keywords demo](/scripts/demos/keyword_demo.py))

[Keywords for EFCSN orgs](/data/EFCSN_keywords.json), and [translations of topic names](/data/topic_name_translations.json), are provided.

##### Pastel
Gives a checkworthiness score to provided sentences.
A LLM is asked a set of yes/no questions about each sentence.
The answers are turned into a score between 1 and 5 by a regression model.
More information [here](/src/harmful_claim_finder/pastel).

([pastel demo](/scripts/demos/pastel_demo.py))

##### Claim Extraction
> This is not part of the MVP

Finds claims made in a video.
Does not make any checkworthiness judgments about them.

([claim extraction demo](/scripts/demos/claim_extraction_example.py))