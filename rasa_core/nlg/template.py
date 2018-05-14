from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging

import numpy as np
from typing import Text, Any, Dict

from rasa_core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    def __init__(self, templates):
        self.templates = templates

    def _random_template_for(self, utter_action):
        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    def generate(self, template_name, filled_slots=None, **kwargs):
        # type: (Text, **Any) -> Dict[Text, Any]
        """Retrieve a named template from the domain."""

        r = copy.deepcopy(self._random_template_for(template_name))
        if r is not None:
            return self._fill_template_text(r, filled_slots, **kwargs)
        else:
            return {"text": "Undefined utter template <{}>."
                            "".format(template_name)}

    def _fill_template_text(self, template, filled_slots=None, **kwargs):
        template_vars = self._template_variables(filled_slots, kwargs)
        if template_vars:
            try:
                template["text"] = template["text"].format(**template_vars)
            except KeyError as e:
                logger.exception(
                        "Failed to fill utterance template '{}'. "
                        "Tried to replace '{}' but could not find "
                        "a value for it. There is no slot with this "
                        "name nor did you pass the value explicitly "
                        "when calling the template. Return template "
                        "without filling the template. "
                        "".format(template, e.args[0]))
        return template

    @staticmethod
    def _template_variables(filled_slots, kwargs):
        """Combine slot values and key word arguments to fill templates."""

        if filled_slots is None:
            filled_slots = {}
        template_vars = filled_slots.copy()
        template_vars.update(kwargs.items())
        return template_vars
