import ff_streaming
from typing import Tuple


def pre_calculation_messge(message: ff_streaming.FileMessage) -> Tuple[str, str]:
    """
    Calculate the complete article (as sentence splitting has already happened)
    and pull out the url, which is the url of the first segment, since
    we assume that the country will be the same for all segments in an article.
    """
    full_article = ".".join(
        sentence.sentence_text
        for segment in message.segments
        for sentence in segment.sentences
    )
    url = message.segments[0].url
    return full_article, url
