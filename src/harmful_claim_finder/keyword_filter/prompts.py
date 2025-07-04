TOPIC_PROMPT = """
You are aiding a fact-checking organisation in finding claims in videos which may be worth checking.
You will be given the transcript of a shortform video, and a series of topics and sets of keywords which the fact-checkers use to define those topics,
often particularly relating to current news stories.

Determine if the transcript could be considered to be about any of the numbered topics, using that topic's keywords as a guide for defining the topic.

A transcript can be about more than one topic, or none of the topics.

For each topic, return the sentences which directly make claims about the topics listed.

If a sentence fits a topic, but not specifically the area covered by the keywords, then do not include that sentence.

Each topic may have multiple sentences. If there are no sentences matching a topic, return an empty list ([]) for that topic.

The output will have the format: 

    ```
    {
        '1': ['sentence about topic 1', 'another sentence about topic 1'],
        '2': ['sentence about topic 2'],
        '3': [],
        ...
    }
    ```

(where in this example there are no sentences about topic 3.)    

Make sure the output type is Dict[str, List[str]].
The dict keys are string representations of numbers, and the dict values are lits of sentences.

Return sentences exactly appear, and do not translate anything from the language in which it appears.

If nothing is found, return a dictionary where the keys are the topic numbers provided, and the values are empty lists.

Please use the topics defined in triple backticks below:

```
[KEYWORDS]
```

The transcript will take the form of a list of sentences.
Make sure the sentences returned are exactly the same as those in the original list.
Look for the above topics in the text delineated by triple backticks below:

```
[TEXT]
```
"""


FIX_JSON = """
This JSON string is not quite in the correct format.
The format should be:
{
    '1': ['sentence about topic 1', 'another sentence about topic 1'],
    '2': ['sentence about topic 2'],
    '3': [],
    ...
}
Where each key is a number, and each value is a list of sentences.
Please fix this broken json dictionary, returning only a correctly json formatted string:
{INPUT_TEXT}
""".strip()
