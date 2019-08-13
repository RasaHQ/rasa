import typing
from typing import Any, Dict, Text

import requests
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData


class ThirdPartyTokenizer(Tokenizer, Component):
    provides = ["tokens"]

    requires = []

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(ThirdPartyTokenizer, self).__init__(component_config)
        self.third_party_service_endpoint = self.component_config.get(
            "third_party_service_endpoint"
        )

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text: Text) -> typing.List[Token]:
        req = requests.get(self.third_party_service_endpoint + str(text))
        return [Token(v["text"], v["end"]) for v in req.json()]
