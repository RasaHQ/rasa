import aiohttp

import logging

import os
from typing import Text, Dict, Any, Union, Optional

from rasa.core import constants
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY
import rasa.shared.utils.io
import rasa.shared.utils.common
import rasa.shared.nlu.interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


def create_interpreter(
    obj: Union[
        rasa.shared.nlu.interpreter.NaturalLanguageInterpreter,
        EndpointConfig,
        Text,
        None,
    ]
) -> "rasa.shared.nlu.interpreter.NaturalLanguageInterpreter":
    """Factory to create a natural language interpreter."""

    if isinstance(obj, rasa.shared.nlu.interpreter.NaturalLanguageInterpreter):
        return obj
    elif isinstance(obj, str) and os.path.exists(obj):
        return RasaNLUInterpreter(model_directory=obj)
    elif isinstance(obj, str):
        # user passed in a string, but file does not exist
        logger.warning(
            f"No local NLU model '{obj}' found. Using RegexInterpreter instead."
        )
        return rasa.shared.nlu.interpreter.RegexInterpreter()
    else:
        return _create_from_endpoint_config(obj)


class RasaNLUHttpInterpreter(rasa.shared.nlu.interpreter.NaturalLanguageInterpreter):
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
        metadata: Optional[Dict] = None,
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


class RasaNLUInterpreter(rasa.shared.nlu.interpreter.NaturalLanguageInterpreter):
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
        metadata: Optional[Dict] = None,
    ) -> Dict[Text, Any]:
        """Parse a text message.

        Return a default value if the parsing of the text failed."""

        if self.lazy_init and self.interpreter is None:
            self._load_interpreter()

        result = self.interpreter.parse(text)

        return result

    def featurize_message(self, message: Message) -> Optional[Message]:
        """Featurize message using a trained NLU pipeline.
        Args:
            message: storing text to process
        Returns:
            message containing tokens and features which are the output of the NLU
            pipeline
        """
        if self.lazy_init and self.interpreter is None:
            self._load_interpreter()
        result = self.interpreter.featurize_message(message)
        return result

    def _load_interpreter(self) -> None:
        from rasa.nlu.model import Interpreter

        self.interpreter = Interpreter.load(self.model_directory)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig],
) -> rasa.shared.nlu.interpreter.NaturalLanguageInterpreter:
    """Instantiate a natural language interpreter based on its configuration."""

    if endpoint_config is None:
        return rasa.shared.nlu.interpreter.RegexInterpreter()
    elif endpoint_config.type is None or endpoint_config.type.lower() == "http":
        return RasaNLUHttpInterpreter(endpoint_config=endpoint_config)
    else:
        return _load_from_module_name_in_endpoint_config(endpoint_config)


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> rasa.shared.nlu.interpreter.NaturalLanguageInterpreter:
    """Instantiate an event channel based on its class name."""

    try:
        nlu_interpreter_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return nlu_interpreter_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a "
            f"`NaturalLanguageInterpreter` instance. Error: {e}"
        )
