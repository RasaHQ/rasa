from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
from rasa_core import utils

from rasa_core.utils import EndpointConfig
from rasa_core.trackers import DialogueStateTracker
from typing import Text, Any, Dict

from rasa_core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


def _nlg_data_schema():
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "buttons": {
                "type": "array",
                "items": {"type": "object"}
            },
            "elements": {
                "type": "array",
                "items": {"type": "object",}
            },
            "attachment": {"type": "object"}
        },
    }


class CallbackNaturalLanguageGenerator(NaturalLanguageGenerator):
    def __init__(self, endpoint_config):
        # type: (EndpointConfig) -> None

        self.nlg_endpoint = endpoint_config

    @staticmethod
    def _nlg_api_format(template_name, tracker, output_channel, kwargs):
        filled_slots = tracker.current_slot_values()
        return {
            "template": template_name,
            "slots": filled_slots,
            "arguments": kwargs,
            "channel": output_channel
        }

    def generate(self, template_name, tracker, output_channel, **kwargs):
        # type: (Text, DialogueStateTracker, Text, **Any) -> Dict[Text, Any]
        """Retrieve a named template from the domain."""

        body = self._nlg_api_format(template_name, tracker, output_channel,
                                    kwargs)

        response = self.nlg_endpoint.request(body, method="post")

        response.raise_for_status()
        content = response.json()
        if self.validate_response(content):
            return content
        else:
            raise Exception("NLG web endpoint returned an invalid response.")

    @staticmethod
    def validate_response(content):
        # type: (content) -> bool
        """Validate rasa training data format to ensure proper training.

        Raises exception on failure."""
        from jsonschema import validate
        from jsonschema import ValidationError

        try:
            validate(content, _nlg_data_schema())
            return True
        except ValidationError as e:
            e.message += (
                ". Failed to validate NLG response from API, make sure your "
                "response from the NLG endpoint is valid. "
                "For more information about the format visit "
                # TODO: TB - DOCU add correct link
                "https://nlu.rasa.com/...")
            raise e
