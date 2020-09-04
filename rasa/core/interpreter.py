from json import JSONDecodeError

import aiohttp

import json
import logging
import re

import os
from typing import Text, List, Dict, Any, Union, Optional, Tuple

from rasa.constants import DOCS_URL_STORIES
from rasa.core import constants
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.nlu.constants import INTENT_NAME_KEY
from rasa.utils.common import raise_warning, class_from_module_path
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class NaturalLanguageInterpreter:
    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> Dict[Text, Any]:
        raise NotImplementedError(
            "Interpreter needs to be able to parse messages into structured output."
        )

    @staticmethod
    def create(
        obj: Union["NaturalLanguageInterpreter", EndpointConfig, Text, None]
    ) -> "NaturalLanguageInterpreter":
        """Factory to create an natural language interpreter."""

        if isinstance(obj, NaturalLanguageInterpreter):
            return obj
        elif isinstance(obj, str) and os.path.exists(obj):
            return RasaNLUInterpreter(model_directory=obj)
        elif isinstance(obj, str) and not os.path.exists(obj):
            # user passed in a string, but file does not exist
            logger.warning(
                f"No local NLU model '{obj}' found. Using RegexInterpreter instead."
            )
            return RegexInterpreter()
        else:
            return _create_from_endpoint_config(obj)


class RegexInterpreter(NaturalLanguageInterpreter):
    @staticmethod
    def allowed_prefixes() -> Text:
        return INTENT_MESSAGE_PREFIX

    @staticmethod
    def _create_entities(
        parsed_entities: Dict[Text, Union[Text, List[Text]]], sidx: int, eidx: int
    ) -> List[Dict[Text, Any]]:
        entities = []
        for k, vs in parsed_entities.items():
            if not isinstance(vs, list):
                vs = [vs]
            for value in vs:
                entities.append(
                    {
                        "entity": k,
                        "start": sidx,
                        "end": eidx,  # can't be more specific
                        "value": value,
                    }
                )
        return entities

    @staticmethod
    def _parse_parameters(
        entity_str: Text, sidx: int, eidx: int, user_input: Text
    ) -> List[Dict[Text, Any]]:
        if entity_str is None or not entity_str.strip():
            # if there is nothing to parse we will directly exit
            return []

        try:
            parsed_entities = json.loads(entity_str)
            if isinstance(parsed_entities, dict):
                return RegexInterpreter._create_entities(parsed_entities, sidx, eidx)
            else:
                raise ValueError(
                    f"Parsed value isn't a json object "
                    f"(instead parser found '{type(parsed_entities)}')"
                )
        except (JSONDecodeError, ValueError) as e:
            raise_warning(
                f"Failed to parse arguments in line "
                f"'{user_input}'. Failed to decode parameters "
                f"as a json object. Make sure the intent "
                f"is followed by a proper json object. "
                f"Error: {e}",
                docs=DOCS_URL_STORIES,
            )
            return []

    @staticmethod
    def _parse_confidence(confidence_str: Text) -> float:
        if confidence_str is None:
            return 1.0

        try:
            return float(confidence_str.strip()[1:])
        except ValueError as e:
            raise_warning(
                f"Invalid to parse confidence value in line "
                f"'{confidence_str}'. Make sure the intent confidence is an "
                f"@ followed by a decimal number. "
                f"Error: {e}",
                docs=DOCS_URL_STORIES,
            )
            return 0.0

    def _starts_with_intent_prefix(self, text: Text) -> bool:
        for c in self.allowed_prefixes():
            if text.startswith(c):
                return True
        return False

    @staticmethod
    def extract_intent_and_entities(
        user_input: Text,
    ) -> Tuple[Optional[Text], float, List[Dict[Text, Any]]]:
        """Parse the user input using regexes to extract intent & entities."""

        prefixes = re.escape(RegexInterpreter.allowed_prefixes())
        # the regex matches "slot{"a": 1}"
        m = re.search("^[" + prefixes + "]?([^{@]+)(@[0-9.]+)?([{].+)?", user_input)
        if m is not None:
            event_name = m.group(1).strip()
            confidence = RegexInterpreter._parse_confidence(m.group(2))
            entities = RegexInterpreter._parse_parameters(
                m.group(3), m.start(3), m.end(3), user_input
            )

            return event_name, confidence, entities
        else:
            logger.warning(f"Failed to parse intent end entities from '{user_input}'.")
            return None, 0.0, []

    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> Dict[Text, Any]:
        """Parse a text message."""

        return self.synchronous_parse(text)

    def synchronous_parse(self, text: Text,) -> Dict[Text, Any]:
        """Parse a text message."""

        intent, confidence, entities = self.extract_intent_and_entities(text)

        if self._starts_with_intent_prefix(text):
            message_text = text
        else:
            message_text = INTENT_MESSAGE_PREFIX + text

        return {
            "text": message_text,
            "intent": {INTENT_NAME_KEY: intent, "confidence": confidence},
            "intent_ranking": [{INTENT_NAME_KEY: intent, "confidence": confidence}],
            "entities": entities,
        }


class RasaNLUHttpInterpreter(NaturalLanguageInterpreter):
    def __init__(self, endpoint_config: Optional[EndpointConfig] = None) -> None:
        if endpoint_config:
            self.endpoint_config = endpoint_config
        else:
            self.endpoint_config = EndpointConfig(constants.DEFAULT_SERVER_URL)

    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> Dict[Text, Any]:
        """Parse a text message.

        Return a default value if the parsing of the text failed."""

        default_return = {
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
            "entities": [],
            "text": "",
        }
        result = await self._rasa_http_parse(text, message_id)

        return result if result is not None else default_return

    async def _rasa_http_parse(
        self, text: Text, message_id: Optional[Text] = None
    ) -> Optional[Dict[Text, Any]]:
        """Send a text message to a running rasa NLU http server.
        Return `None` on failure."""

        if not self.endpoint_config:
            logger.error(
                f"Failed to parse text '{text}' using rasa NLU over http. "
                f"No rasa NLU server specified!"
            )
            return None

        params = {
            "token": self.endpoint_config.token,
            "text": text,
            "message_id": message_id,
        }

        if self.endpoint_config.url.endswith("/"):
            url = self.endpoint_config.url + "model/parse"
        else:
            url = self.endpoint_config.url + "/model/parse"

        # noinspection PyBroadException
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        response_text = await resp.text()
                        logger.error(
                            f"Failed to parse text '{text}' using rasa NLU over "
                            f"http. Error: {response_text}"
                        )
                        return None
        except Exception:  # skipcq: PYL-W0703
            # need to catch all possible exceptions when doing http requests
            # (timeouts, value errors, parser errors, ...)
            logger.exception(f"Failed to parse text '{text}' using rasa NLU over http.")
            return None


class RasaNLUInterpreter(NaturalLanguageInterpreter):
    def __init__(
        self,
        model_directory: Text,
        config_file: Optional[Text] = None,
        lazy_init: bool = False,
    ):
        self.model_directory = model_directory
        self.lazy_init = lazy_init
        self.config_file = config_file

        if not lazy_init:
            self._load_interpreter()
        else:
            self.interpreter = None

    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> Dict[Text, Any]:
        """Parse a text message.

        Return a default value if the parsing of the text failed."""

        if self.lazy_init and self.interpreter is None:
            self._load_interpreter()
        result = self.interpreter.parse(text)

        return result

    def _load_interpreter(self) -> None:
        from rasa.nlu.model import Interpreter

        self.interpreter = Interpreter.load(self.model_directory)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig],
) -> "NaturalLanguageInterpreter":
    """Instantiate a natural language interpreter based on its configuration."""

    if endpoint_config is None:
        return RegexInterpreter()
    elif endpoint_config.type is None or endpoint_config.type.lower() == "http":
        return RasaNLUHttpInterpreter(endpoint_config=endpoint_config)
    else:
        return _load_from_module_name_in_endpoint_config(endpoint_config)


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> "NaturalLanguageInterpreter":
    """Instantiate an event channel based on its class name."""

    try:
        nlu_interpreter_class = class_from_module_path(endpoint_config.type)
        return nlu_interpreter_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a "
            f"`NaturalLanguageInterpreter` instance. Error: {e}"
        )
