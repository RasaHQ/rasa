import json
import logging
import re

import os
import requests
from typing import Text, List, Dict, Any

from rasa_core import constants
from rasa_core.utils import EndpointConfig
from rasa_core.constants import INTENT_MESSAGE_PREFIX

logger = logging.getLogger(__name__)


class NaturalLanguageInterpreter(object):
    def parse(self, text):
        raise NotImplementedError(
            "Interpreter needs to be able to parse "
            "messages into structured output.")

    @staticmethod
    def create(obj, endpoint=None):
        if isinstance(obj, NaturalLanguageInterpreter):
            return obj
        if isinstance(obj, str):
            name_parts = os.path.split(obj)

            if len(name_parts) == 1:
                if endpoint:
                    # using the default project name
                    return RasaNLUHttpInterpreter(name_parts[0],
                                                  endpoint)
                else:
                    return RasaNLUInterpreter(model_directory=obj)
            elif len(name_parts) == 2:
                if endpoint:
                    return RasaNLUHttpInterpreter(name_parts[1],
                                                  endpoint,
                                                  name_parts[0])
                else:
                    return RasaNLUInterpreter(model_directory=obj)
            else:
                if endpoint:
                    raise Exception(
                        "You have configured an endpoint to use for "
                        "the NLU model. To use it, you need to "
                        "specify the model to use with "
                        "`--nlu project/model`.")
                else:
                    return RasaNLUInterpreter(model_directory=obj)
        else:
            return RegexInterpreter()  # default interpreter


class RegexInterpreter(NaturalLanguageInterpreter):
    @staticmethod
    def allowed_prefixes():
        return INTENT_MESSAGE_PREFIX

    @staticmethod
    def _create_entities(parsed_entities, sidx, eidx):
        entities = []
        for k, vs in parsed_entities.items():
            if not isinstance(vs, list):
                vs = [vs]
            for value in vs:
                entities.append({
                    "entity": k,
                    "start": sidx,
                    "end": eidx,  # can't be more specific
                    "value": value
                })
        return entities

    @staticmethod
    def _parse_parameters(entitiy_str: Text,
                          sidx: int,
                          eidx: int,
                          user_input: Text) -> List[Dict[Text, Any]]:
        if entitiy_str is None or not entitiy_str.strip():
            # if there is nothing to parse we will directly exit
            return []

        try:
            parsed_entities = json.loads(entitiy_str)
            if isinstance(parsed_entities, dict):
                return RegexInterpreter._create_entities(parsed_entities,
                                                         sidx, eidx)
            else:
                raise Exception("Parsed value isn't a json object "
                                "(instead parser found '{}')"
                                ".".format(type(parsed_entities)))
        except Exception as e:
            logger.warning("Invalid to parse arguments in line "
                           "'{}'. Failed to decode parameters"
                           "as a json object. Make sure the intent"
                           "followed by a proper json object. "
                           "Error: {}".format(user_input, e))
            return []

    @staticmethod
    def _parse_confidence(confidence_str: Text) -> float:
        if confidence_str is None:
            return 1.0

        try:
            return float(confidence_str.strip()[1:])
        except Exception as e:
            logger.warning("Invalid to parse confidence value in line "
                           "'{}'. Make sure the intent confidence is an "
                           "@ followed by a decimal number. "
                           "Error: {}".format(confidence_str, e))
            return 0.0

    def _starts_with_intent_prefix(self, text):
        for c in self.allowed_prefixes():
            if text.startswith(c):
                return True
        return False

    @staticmethod
    def extract_intent_and_entities(user_input: Text) -> object:
        """Parse the user input using regexes to extract intent & entities."""

        prefixes = re.escape(RegexInterpreter.allowed_prefixes())
        # the regex matches "slot{"a": 1}"
        m = re.search('^[' + prefixes + ']?([^{@]+)(@[0-9.]+)?([{].+)?',
                      user_input)
        if m is not None:
            event_name = m.group(1).strip()
            confidence = RegexInterpreter._parse_confidence(m.group(2))
            entities = RegexInterpreter._parse_parameters(m.group(3),
                                                          m.start(3),
                                                          m.end(3),
                                                          user_input)

            return event_name, confidence, entities
        else:
            logger.warning("Failed to parse intent end entities from "
                           "'{}'. ".format(user_input))
            return None, 0.0, []

    def parse(self, text):
        """Parse a text message."""

        intent, confidence, entities = self.extract_intent_and_entities(text)

        if self._starts_with_intent_prefix(text):
            message_text = text
        else:
            message_text = INTENT_MESSAGE_PREFIX + text

        return {
            'text': message_text,
            'intent': {
                'name': intent,
                'confidence': confidence,
            },
            'intent_ranking': [{
                'name': intent,
                'confidence': confidence,
            }],
            'entities': entities,
        }


class RasaNLUHttpInterpreter(NaturalLanguageInterpreter):
    def __init__(self,
                 model_name: Text = None,
                 endpoint: EndpointConfig = None,
                 project_name: Text = 'default') -> None:

        self.model_name = model_name
        self.project_name = project_name

        if endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = EndpointConfig(constants.DEFAULT_SERVER_URL)

    def parse(self, text):
        """Parse a text message.

        Return a default value if the parsing of the text failed."""

        default_return = {"intent": {"name": "", "confidence": 0.0},
                          "entities": [], "text": ""}
        result = self._rasa_http_parse(text)

        return result if result is not None else default_return

    def _rasa_http_parse(self, text):
        """Send a text message to a running rasa NLU http server.

        Return `None` on failure."""

        if not self.endpoint:
            logger.error(
                "Failed to parse text '{}' using rasa NLU over http. "
                "No rasa NLU server specified!".format(text))
            return None

        params = {
            "token": self.endpoint.token,
            "model": self.model_name,
            "project": self.project_name,
            "q": text
        }
        url = "{}/parse".format(self.endpoint.url)
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
        """Parse a text message.

        Return a default value if the parsing of the text failed."""

        if self.lazy_init and self.interpreter is None:
            self._load_interpreter()
        return self.interpreter.parse(text)

    def _load_interpreter(self):
        from rasa_nlu.model import Interpreter

        self.interpreter = Interpreter.load(self.model_directory)
