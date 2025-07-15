import logging
import re
import traceback

from harmful_claim_finder.keyword_filter.prompts import FIX_JSON, TOPIC_PROMPT
from harmful_claim_finder.utils.gemini import GeminiError, run_prompt
from harmful_claim_finder.utils.models import ParsingError, TopicDetectionError
from harmful_claim_finder.utils.parsing import ParsedType, parse_model_json_output

AllKeywordsType = dict[str, dict[str, dict[str, dict[str, list[str]]]]]

logger = logging.getLogger(__name__)


class TopicKeywordFilter:
    """
    A class used to create a GenAI keyword topic filter.

    Each organisation will have one of these filters per language.
    The filter is defined by the dictionary of all keywords for all topics.

    The filter can be given an article, and return each sentence and a list of topics
    the sentence relates to.

    Attributes
    ----------
    keywords: dict[str, list[str]]
        A dictionary of keywords where the keys are topic names and the values are
        lists of keywords for the topic.
    prompt_outline: str
        This is the bulk of the prompt. It should contain the substitutable parts
        [KEYWORDS] and [TEXT]. It will likely be `TOPIC_KEYWORD_ARTICLE_PROMPT`.
    """

    def __init__(
        self,
        keywords: dict[str, list[str]],
        prompt_outline: str = TOPIC_PROMPT,
    ) -> None:
        """
        Parameters
        ----------
        keywords: dict[str, list[str]]
        prompt_outline: str
        """
        self.keywords = keywords
        self.prompt_outline = prompt_outline
        self.mapped_keywords, self.topic_name_map = self.do_topic_name_mapping()

    def do_topic_name_mapping(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """
        Takes a keyword dictionary and replaces all topic names with a number.
        Returns the updated dict, and a name key dict where the new topic
        numbers are keys and the original topic names are values.
        We do this to avoid the topic name being interpreted as a keyword.

        Returns
        -------
        dict[str, list[str]
            Keywords in the same format as the originals provided to the Filter,
            but with topic names replaced with numbers.
        dict[str, str]
            Dict of topic numbers (as strings) mapping back to topic names.
        """
        mapped_keywords = mapped_keywords = {
            f"{i+1}": words for i, words in enumerate(self.keywords.values())
        }
        topic_name_map = {
            mapped: unmapped
            for mapped, unmapped in zip(mapped_keywords.keys(), self.keywords.keys())
        }
        return mapped_keywords, topic_name_map

    def do_result_unmapping(
        self, result_dict: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """
        Takes a results dict where topics are identified by
        mapping numbers and replaces these with topic names.

        Parameters
        ----------
        results_dict: dict[str, list[str]]
            Parsed LLM keyword results dict.
            Key in sentence, value is list of topics for sentence.


        Returns
        -------
        dict[str, list[str]]
            The same data, but with topic numbers replaced with
            topic names.
        """
        unmapped_result = {
            sent: [self.topic_name_map[topic] for topic in topics]
            for sent, topics in result_dict.items()
        }
        return unmapped_result

    def make_keyword_prompt(self, text: list[str]) -> str:
        """Makes prompt by substituting keywords and article
        text into prompt outline.

        Parameters
        ----------
        text: list[str]
            The article to check against the keywords.
            Must be formatted as a list of strings.

        Returns
        -------
        str
            The final prompt to send to the LLM, containing all the keywords
            and the article text.
        """
        prompt = self.prompt_outline
        keywords_for_prompt = self.mapped_keywords

        # This version is deprecated as it encourages Gemini to produce Python as output
        # prompt = prompt.replace("[KEYWORDS]", str(keywords_for_prompt))
        # Instead we make the prompt look more "english" and less "python":
        keyword_prompt = ""
        for topic, keyword_list in keywords_for_prompt.items():
            keyword_prompt += (
                f"Topic '{topic}' is defined by the terms "
                f"[{', '.join(keyword_list)}] \n"
            )

        prompt = prompt.replace("[KEYWORDS]", keyword_prompt)
        prompt = prompt.replace("[TEXT]", str(text))
        return prompt

    @staticmethod
    def parse(response: str) -> ParsedType:
        """Parses an LLM response ro remove extra characters,
        and uses the JSON parse function from `genai_utils`.
        ppips responses it cannot parse.
        TODO: handle these better.

        Parameters
        ----------
        response: str
            Response from an LLM, unformatted and unparsed so it is
            still a string.

        Returns
        -------
        `ParsedType`
            The parsed JSON response. Using ParsedType allows flexibility in
            usage of the function.
        """
        response = re.sub("`", "", response)
        response = re.sub("json", "", response)
        response = re.sub("\n", "", response)
        return parse_model_json_output(response)

    def format_results(
        self, result: ParsedType, article: list[str]
    ) -> dict[str, list[str]]:
        """Formats the output of an article-level prompt. Contains an `invert_dict` function
        only used here.

        Parameters
        ----------
        result: `ParsedType`
            A dictionary where the keys are topics and the values are lists of sentences
            from the article relating to those topics.
        article: list[str]
            The original article, formatted as a list of strings.

        Returns
        -------
        dict[str, list[str]]
            Returns dictionary where the keys are sentences from the article, and values
            are lists of topics associated with those sentences.
        """

        def invert_dict(original: dict[str, list[str]]) -> dict[str, list[str]]:
            """Swaps a dictionary round so keys become values and vice-versa.
            Where values are lists of strings, these are separated.

            Parameters
            ----------
            original: dict[str, list[str]]
                A dictionary where keys are strings and values are lists of strings.

            Returns
            -------
            dict[str, list[str]]
                A dictionary where keys are strings and values are lists of strings.
            """
            inverted: dict[str, list[str]] = {}
            for key, values in original.items():
                for value in values:
                    if value not in inverted:
                        inverted[value] = []
                    inverted[value].append(key)
            return inverted

        if result is not None and isinstance(result, dict):
            try:
                inverted_result = invert_dict(result)
                result_with_all_sentences: dict[str, list[str]] = {
                    sent: list(set(inverted_result.get(sent, []))) for sent in article
                }  # TODO: fuzzy matching to make sure fewer sentences are dropped
            except Exception as exc:
                raise ParsingError(
                    "Topic detection could not parse gemini output"
                ) from exc
        else:
            result_with_all_sentences = {sent: [] for sent in article}

        return result_with_all_sentences

    async def run_all_for_article(
        self, article: list[str], max_attempts: int = 3
    ) -> dict[str, list[str]]:
        """Runs all functions on a new article.
        Makes the prompt, runs the prompt and parses the output, formats the
        result.

        Parameters
        ----------
        article: list[str]
            The article to check, formatted as a list of strings.

        max_attempts: int
            The number of retries to attempt if there's an exception.

        Returns
        -------
        dict[str, list[str]]
            Returns dictionary where the keys are sentences from the article, and values
            are lists of topics associated with those sentences.

        Raises
        ------
        TopicDetectionError:
            If topic detection fails `max_attempts` times, an exception will be raised.
        """
        for _ in range(max_attempts):
            try:
                prompt = self.make_keyword_prompt(article)
                response = await run_prompt(prompt)
                result = self.parse(response)
                try:
                    formatted_result = self.format_results(result, article)
                except ParsingError:
                    logger.info(f"Parsing error: {traceback.format_exc()}")
                    fixed_json = await run_prompt(
                        FIX_JSON.replace("{INPUT_TEXT}", response)
                    )
                    fixed_result = self.parse(fixed_json)
                    formatted_result = self.format_results(fixed_result, article)
                formatted_result = self.do_result_unmapping(formatted_result)
                return formatted_result
            except GeminiError as exc:
                logger.info(f"Error while running Gemini: {repr(exc)}")
            except Exception as exc:
                logger.info(f"Error raised while running topic detection: {repr(exc)}")
                traceback.print_exc()

        raise TopicDetectionError(f"Topic detection failed {max_attempts} times.")
