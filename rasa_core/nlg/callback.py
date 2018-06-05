from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
from typing import Text, Any, Dict

from rasa_core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


class CallbackNaturalLanguageGenerator(NaturalLanguageGenerator):
    def __init__(self, url):
        self.url = url

    @staticmethod
    def _nlg_api_format(template_name, filled_slots, kwargs):
        return {
            "template": template_name,
            "slots": filled_slots,
            "arguments": kwargs
        }

    def generate(self, template_name, filled_slots=None, **kwargs):
        # type: (Text, **Any) -> Dict[Text, Any]
        """Retrieve a named template from the domain."""

        body = self._nlg_api_format(template_name, filled_slots, **kwargs)

        response = requests.post(self.url, json=body)

        response.raise_for_status()
        content = response.json()
        if self.validate_response(content):
            return content
        else:
            raise Exception("NLG web endpoint returned an invalid response.")

    @staticmethod
    def validate_response(content):
        # TODO: do some validation with the response we get from the endpoint
        return True
