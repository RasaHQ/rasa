import logging
from typing import Text, Any, Dict, Optional

from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


def nlg_response_format_spec():
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
        },
    }


def nlg_request_format_spec():
    """Expected request schema for requests sent to an NLG endpoint."""

    return {
        "type": "object",
        "properties": {
            "template": {"type": "string"},
            "arguments": {"type": "object"},
            "tracker": {
                "type": "object",
                "properties": {
                    "sender_id": {"type": "string"},
                    "slots": {"type": "object"},
                    "latest_message": {"type": "object"},
                    "latest_event_time": {"type": "number"},
                    "paused": {"type": "boolean"},
                    "events": {"type": "array"},
                },
            },
            "channel": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }


def nlg_request_format(
    template_name: Text,
    tracker: DialogueStateTracker,
    output_channel: Text,
    **kwargs: Any,
) -> Dict[Text, Any]:
    """Create the json body for the NLG json body for the request."""

    tracker_state = tracker.current_state(EventVerbosity.ALL)

    return {
        "template": template_name,
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

    def __init__(self, endpoint_config: EndpointConfig) -> None:

        self.nlg_endpoint = endpoint_config

    async def generate(
        self,
        template_name: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """Retrieve a named template from the domain using an endpoint."""

        body = nlg_request_format(template_name, tracker, output_channel, **kwargs)

        logger.debug(
            "Requesting NLG for {} from {}."
            "".format(template_name, self.nlg_endpoint.url)
        )

        response = await self.nlg_endpoint.request(
            method="post", json=body, timeout=DEFAULT_REQUEST_TIMEOUT
        )

        if self.validate_response(response):
            return response
        else:
            raise Exception("NLG web endpoint returned an invalid response.")

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
            e.message += (
                ". Failed to validate NLG response from API, make sure your "
                "response from the NLG endpoint is valid. "
                "For more information about the format please consult the "
                "`nlg_response_format_spec` function from this same module: "
                "https://github.com/RasaHQ/rasa/blob/master/rasa/core/nlg/callback.py#L12"
            )
            raise e
