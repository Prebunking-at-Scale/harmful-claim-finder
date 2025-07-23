"""
Runs demo of both prompt types on Full Fact keywords and a collection of demo articles.
"""

import asyncio
from pprint import pprint

from dummy_text import all_articles, test_keywords_en


from harmful_claim_finder.keyword_filter.prompts import TOPIC_PROMPT
from harmful_claim_finder.keyword_filter.topic_keyword_filter import TopicKeywordFilter

if __name__ == "__main__":

    keywords = test_keywords_en["fullfact"]

    english_article_filter = TopicKeywordFilter(
        keywords=keywords,
        prompt_outline=TOPIC_PROMPT,
    )

    for article in all_articles:

        print("~~~~~~~~~~~~~~~~~~~~~~\n\n### Result for article level prompt:\n\n")
        print("ENGLISH\n")
        pprint(
            asyncio.run(english_article_filter.run_all_for_article(article), debug=True)
        )
