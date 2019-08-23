import typing
from typing import Any

from rasa.nlu.config.nlu import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data.training_data import TrainingData
from rasa.nlu.training_data.message import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer):

    provides = ["tokens"]

    requires = ["spacy_doc"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set("tokens", self.tokenize(message.get("spacy_doc")))

    def tokenize(self, doc: "Doc") -> typing.List[Token]:

        return [Token(t.text, t.idx) for t in doc]
