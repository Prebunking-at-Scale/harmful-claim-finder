"""
Runs demo of both prompt types on Full Fact keywords and a collection of demo articles.
"""

from pprint import pprint

from dummy_text import all_articles, arabic_articles, test_keywords_ar, test_keywords_en

from harmful_claim_finder.keyword_filter.keyword_formatter import (
    assert_keyword_type,
    keyword_formatter,
)
from harmful_claim_finder.keyword_filter.prompts import TOPIC_KEYWORD_ARTICLE_PROMPT
from harmful_claim_finder.keyword_filter.topic_keyword_filter import TopicKeywordFilter

if __name__ == "__main__":

    assert_keyword_type(test_keywords_en, "fullfact", "en")

    english_article_filter = TopicKeywordFilter(
        keywords=keyword_formatter(test_keywords_en, "fullfact", "en"),
        prompt_outline=TOPIC_KEYWORD_ARTICLE_PROMPT,
    )

    assert_keyword_type(test_keywords_ar, "factyemen", "ar")

    arabic_article_filter = TopicKeywordFilter(
        keywords=keyword_formatter(test_keywords_ar, "factyemen", "ar"),
        prompt_outline=TOPIC_KEYWORD_ARTICLE_PROMPT,
    )

    for article in all_articles:

        print("~~~~~~~~~~~~~~~~~~~~~~\n\n### Result for article level prompt:\n\n")
        print("ENGLISH\n")
        pprint(english_article_filter.run_all_for_article(article))
        print("\n\nARABIC\n")
        pprint(arabic_article_filter.run_all_for_article(article))

    for article in arabic_articles:

        print("~~~~~~~~~~~~~~~~~~~~~~\n\n### Result for article level prompt:\n\n")
        print("ARABIC\n")
        pprint(arabic_article_filter.run_all_for_article(article))
        print("\n\nENGLISH\n")
        pprint(english_article_filter.run_all_for_article(article))
