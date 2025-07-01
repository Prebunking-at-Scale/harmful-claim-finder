AllKeywordsType = dict[str, dict[str, dict[str, dict[str, list[str]]]]]


def assert_keyword_type(all_keywords: AllKeywordsType, org: str, lang: str) -> None:
    """Function to run a series of assertions on a keyword dictionary to make
    sure its format is exactly correct and it contains keywords for the
    organisation in question.

    Parameters
    ----------
    all_keywords: dict[str, dict[str, dict[str, dict[str, list[str]]]]]
        Dict containing all keywords for all organisations.
        Input type also called `AllKeywordsType`.

        Example:
            ```
            {
                "include_lists": {
                    "test_organisation": {
                        "asylum": {"en": ["Small boat crossings"]},
                        "crime": {"en": ["Police officers"]},
                    }
                }
            }
            ```
        Function will assert if it does indeed have that format.
    org: str
        Organisation we are concerned with.
    lang: str
        Language for organisation we are concerned with.

    Raises
    ------
    AssertionError
        If any of the assertions fail.


    """
    assert len(all_keywords) in [1, 2], (
        "Dict should have length 1 if just include keywords or\n"
        "length 2 if also exclude keywords."
    )
    for inc_exc, orgwords in all_keywords.items():
        assert isinstance(inc_exc, str)
        assert isinstance(orgwords, dict)
        assert (
            org in orgwords.keys()
        ), "Intended organisation is not in list of keywords."
        for org, topicwords in orgwords.items():
            assert isinstance(org, str)
            assert isinstance(topicwords, dict)
            if inc_exc == "include_lists":
                assert len(topicwords) > 0, f"No topics for organisation {org}."
            for topic, langwords in topicwords.items():
                assert isinstance(topic, str)
                assert isinstance(langwords, dict)
                for lang, words in langwords.items():
                    assert isinstance(lang, str)
                    assert isinstance(words, list)
                    if inc_exc == "include_lists":
                        assert (
                            len(words) > 0
                        ), f"Wordlist for topic {topic} in {lang} for {org} empty."
                    for word in words:
                        assert isinstance(word, str)


def keyword_formatter(
    all_keywords: AllKeywordsType, organisation: str, language: str
) -> dict[str, list[str]]:
    """Reformats keywords from `AllKeywordsType` to simpler dictionary
    without `include_lists` key or language. Omits `exclude_lists` as these
    are not used by the prompt.

    Parameters
    ----------
    all_keywords: `AllKeywordsType`
        Dictionary containing all keywords for all organisations.
    organisation: str
        The name of the organisation whose keywords we are extracting.
    language: str
        Organisation's language. It may have more than one; in that case everything
        will be run for each language, so only use one here.

    """
    formatted_keywords = {
        topic: words_for_lang.get(language, [])
        for topic, words_for_lang in all_keywords["include_lists"]
        .get(organisation, {})
        .items()
    }
    return {
        topic: keywords
        for topic, keywords in formatted_keywords.items()
        if len(keywords) > 0
    }
