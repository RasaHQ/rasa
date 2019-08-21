import re
from typing import Any, Dict, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData


class WhitespaceTokenizer(Tokenizer, Component):

    provides = ["tokens", "intent_tokens", "response_tokens"]

    defaults = {
        # text will be tokenized with case sensitive as default
        "intent_split_symbol": " ",
        "case_sensitive": True,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super(WhitespaceTokenizer, self).__init__(component_config)
        self.section_split_symbol = self.component_config["intent_split_symbol"]
        self.case_sensitive = self.component_config["case_sensitive"]
        self.is_test_data_featurized = False

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))
            if example.get("intent"):
                example.set(
                    "intent_tokens",
                    self.tokenize(example.get("intent"), self.section_split_symbol),
                )

            if example.get("response"):
                example.set(
                    "response_tokens",
                    self.tokenize(example.get("response"), self.section_split_symbol),
                )

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text: Text, split=" ") -> List[Token]:
        if not self.case_sensitive:
            text = text.lower()
        # remove 'not a word character' if
        words = re.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            split,
            text,
        ).split(split)

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
