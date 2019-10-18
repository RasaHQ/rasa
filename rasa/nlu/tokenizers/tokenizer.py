import logging

from typing import Text, List, Optional, Dict, Any

from rasa.nlu.components import Component
from rasa.nlu.constants import (
    MESSAGE_RESPONSE_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    CLS_TOKEN,
)

logger = logging.getLogger(__name__)


class Token(object):
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)


class Tokenizer(Component):
    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(Tokenizer, self).__init__(component_config)

        try:
            self.use_cls_token = self.component_config["use_cls_token"]
        except KeyError:
            raise KeyError(
                "No default value for 'use_cls_token' was set. Please, "
                "add it to the default dict of the tokenizer."
            )

    def add_cls_token(
        self, tokens: List[Token], attribute: Text = MESSAGE_TEXT_ATTRIBUTE
    ) -> List[Token]:
        if (
            attribute in [MESSAGE_RESPONSE_ATTRIBUTE, MESSAGE_TEXT_ATTRIBUTE]
            and self.use_cls_token
        ):
            # +1 to have a space between the last token and the __cls__ token
            idx = tokens[-1].offset + len(tokens[-1].text) + 1
            tokens.append(Token(CLS_TOKEN, idx))

        return tokens
