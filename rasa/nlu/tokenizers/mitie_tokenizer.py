from typing import Any, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData


class MitieTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set("tokens", self.tokenize(message.text))

    def _token_from_offset(self, text, offset, encoded_sentence):
        return Token(
            text.decode("utf-8"), self._byte_to_char_offset(encoded_sentence, offset)
        )

    def tokenize(self, text: Text) -> List[Token]:
        import mitie

        encoded_sentence = text.encode("utf-8")
        tokenized = mitie.tokenize_with_offsets(encoded_sentence)
        tokens = [
            self._token_from_offset(token, offset, encoded_sentence)
            for token, offset in tokenized
        ]
        return tokens

    @staticmethod
    def _byte_to_char_offset(text: bytes, byte_offset: int) -> int:
        return len(text[:byte_offset].decode("utf-8"))
