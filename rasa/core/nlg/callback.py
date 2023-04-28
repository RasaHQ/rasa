import logging
from typing import Text, Any, Dict, Optional, List

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


def nlg_response_format_spec() -> Dict[Text, Any]:
    """Expected response schema for an NLG endpoint.

    Used for validation of the response returned from the NLG endpoint."""
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "buttons": {"type": ["array", "null"], "items": {"type": "object"}},
            "elements": {"type": ["array", "null"], "items": {"type": "object"}},
            "attachment": {"type": ["object", "null"]},
            "image": {"type": ["string", "null"]},
            "custom": {"type": "object"},
        },
    }


def nlg_request_format(
    utter_action: Text,
    tracker: DialogueStateTracker,
    output_channel: Text,
    message: Text,
    **kwargs: Any,
) -> Dict[Text, Any]:
    """Create the json body for the NLG json body for the request."""
    tracker_state = tracker.current_state(EventVerbosity.ALL)

    return {
        "response": utter_action,
        "message": message,
        "arguments": kwargs,
        "tracker": tracker_state,
        "channel": {"name": output_channel},
    }


class CallbackNaturalLanguageGenerator(NaturalLanguageGenerator):
    """Generate bot utterances by using a remote endpoint for the generation.

    The generator will call the endpoint for each message it wants to
    generate. The endpoint needs to respond with a properly formatted
    json. The generator will use this message to create a response for
    the bot."""

    def __init__(self, endpoint_config: EndpointConfig, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        from rasa.core.nlg import TemplatedNaturalLanguageGenerator

        self.nlg_endpoint = endpoint_config
        self.nlg_templated = None
        if responses is not None:
            self.responses = responses
            self.nlg_templated = TemplatedNaturalLanguageGenerator(
                self.responses
            )

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        message = None
        if self.nlg_templated:
            message = await self.nlg_templated.generate(utter_action, tracker, output_channel)
            print(f"message: {message}")

        """Retrieve a named response from the domain using an endpoint."""
        body = nlg_request_format(utter_action, tracker, output_channel, message, **kwargs)

        logger.debug(
            "Requesting NLG for {} from {}."
            "".format(utter_action, self.nlg_endpoint.url)
        )

        response = await self.nlg_endpoint.request(
            method="post", json=body, timeout=DEFAULT_REQUEST_TIMEOUT
        )

        if isinstance(response, dict) and self.validate_response(response):
            return response
        else:
            raise RasaException("NLG web endpoint returned an invalid response.")

    @staticmethod
    def validate_response(content: Optional[Dict[Text, Any]]) -> bool:
        """Validate the NLG response. Raises exception on failure."""
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            if content is None or content == "":
                # means the endpoint did not want to respond with anything
                return True
            else:
                validate(content, nlg_response_format_spec())
                return True
        except ValidationError as e:
            raise RasaException(
                f"{e.message}. Failed to validate NLG response from API, make sure "
                f"your response from the NLG endpoint is valid. "
                f"For more information about the format please consult the "
                f"`nlg_response_format_spec` function from this same module: "
                f"https://github.com/RasaHQ/rasa/blob/main/rasa/core/nlg/callback.py"
            )
