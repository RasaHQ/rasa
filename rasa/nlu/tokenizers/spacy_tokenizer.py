import typing
from typing import Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


from rasa.nlu.constants import (
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_RESPONSE_ATTRIBUTE,
)


class SpacyTokenizer(Tokenizer, Component):

    provides = ["tokens", "intent_tokens", "response_tokens"]

    requires = ["spacy_doc", "intent_spacy_doc", "response_spacy_doc"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))

    def get_doc(self, message, attribute):

        attribute = "" if attribute == MESSAGE_TEXT_ATTRIBUTE else attribute + "_"
        return message.get("{0}spacy_doc".format(attribute))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set(
            "tokens", self.tokenize(self.get_doc(message, MESSAGE_TEXT_ATTRIBUTE))
        )
        intent_doc = self.get_doc(message, MESSAGE_INTENT_ATTRIBUTE)
        if intent_doc:
            message.set("{0}_tokens".format(MESSAGE_INTENT_ATTRIBUTE), intent_doc)
        response_doc = self.get_doc(message, MESSAGE_RESPONSE_ATTRIBUTE)
        if response_doc:
            message.set("{0}_response".format(MESSAGE_RESPONSE_ATTRIBUTE), response_doc)

    def tokenize(self, doc: "Doc") -> typing.List[Token]:

        return [Token(t.text, t.idx) for t in doc]
