from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import re

import os
import requests
from builtins import str

logger = logging.getLogger(__name__)


class NaturalLanguageInterpreter(object):
    def parse(self, text):
        raise NotImplementedError(
                "Interpreter needs to be able to parse "
                "messages into structured output.")

    @staticmethod
    def create(obj):
        if isinstance(obj, NaturalLanguageInterpreter):
            return obj
        if isinstance(obj, str):
            return RasaNLUInterpreter(model_directory=obj)
        else:
            return RegexInterpreter()   # default interpreter


class RegexInterpreter(NaturalLanguageInterpreter):
    @staticmethod
    def extract_intent_and_entities(user_input):
        value_assign_rx = '\s*(.+)\s*=\s*(.+)\s*'
        structed_message_rx = '^_([^\[]+)(\[(.+)\])?'
        m = re.search(structed_message_rx, user_input)
        if m is not None:
            intent = m.group(1).lower()
            offset = m.start(3)
            entities_str = m.group(3)
            entities = []
            if entities_str is not None:
                for entity_str in entities_str.split(','):
                    for match in re.finditer(value_assign_rx, entity_str):
                        start = match.start(2) + offset
                        end = match.end(0) + offset
                        entity = {
                            "entity": match.group(1),
                            "start": start,
                            "end": end,
                            "value": match.group(2)}
                        entities.append(entity)

            return intent, entities
        else:
            return None, []

    def parse(self, text):
        intent, entities = self.extract_intent_and_entities(text)
        return {
            'text': text,
            'intent': {
                'name': intent,
                'confidence': 1.0,
            },
            'intent_ranking': [{
                'name': intent,
                'confidence': 1.0,
            }],
            'entities': entities,
        }


class RasaNLUHttpInterpreter(NaturalLanguageInterpreter):
    def __init__(self, model_name, token, server):
        self.model_name = model_name
        self.token = token
        self.server = server

    def parse(self, text):
        """Parses a text message.

        Returns a default value if the parsing of the text failed."""

        default_return = {"intent": {"name": "", "confidence": 0.0},
                          "entities": [], "text": ""}
        result = self._rasa_http_parse(text)

        return result if result is not None else default_return

    def _rasa_http_parse(self, text):
        """Send a text message to a running rasa NLU http server.

        Returns `None` on failure."""

        if not self.server:
            logger.error(
                    "Failed to parse text '{}' using rasa NLU over http. "
                    "No rasa NLU server specified!".format(text))
            return None

        params = {
            "token": self.token,
            "model": self.model_name,
            "q": text
        }
        url = "{}/parse".format(self.server)
        try:
            result = requests.get(url, params=params)
            if result.status_code == 200:
                return result.json()
            else:
                logger.error(
                        "Failed to parse text '{}' using rasa NLU over http. "
                        "Error: {}".format(text, result.text))
                return None
        except Exception as e:
            logger.error(
                    "Failed to parse text '{}' using rasa NLU over http. "
                    "Error: {}".format(text, e))
            return None


class RasaNLUInterpreter(NaturalLanguageInterpreter):
    def __init__(self, model_directory, config_file=None, lazy_init=False):
        self.model_directory = model_directory
        self.lazy_init = lazy_init
        self.config_file = config_file

        if not lazy_init:
            self._load_interpreter()
        else:
            self.interpreter = None

    def parse(self, text):
        """Parses a text message.

        Returns a default value if the parsing of the text failed."""

        if self.lazy_init and self.interpreter is None:
            self._load_interpreter()
        return self.interpreter.parse(text)

    def _load_interpreter(self):
        from rasa_nlu.model import Interpreter
        from rasa_nlu.config import RasaNLUConfig

        self.interpreter = Interpreter.load(self.model_directory,
                                            RasaNLUConfig(self.config_file,
                                                          os.environ))
