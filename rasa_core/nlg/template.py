from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging

import numpy as np
from rasa_core.trackers import DialogueStateTracker
from typing import Text, Any, Dict

from rasa_core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    def __init__(self, templates):
        self.templates = templates

    def _random_template_for(self, utter_action, output_channel):
        # TODO: TB - make use of the additional information about the channel
        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    def generate(self, template_name, tracker, output_channel, **kwargs):
        # type: (Text, DialogueStateTracker, Text, **Any) -> Dict[Text, Any]
        """Retrieve a named template from the domain."""

        filled_slots = tracker.current_slot_values()
        # Fetching a random template for the passed template name
        r = copy.deepcopy(self._random_template_for(template_name,
                                                    output_channel))
        # Filling the slots in the template and returning the template
        if r is not None:
            return self._fill_template_text(r, filled_slots, **kwargs)
        else:
            return {"text": "Undefined utter template <{}>."
                            "".format(template_name)}

    def _fill_template_text(self, template, filled_slots=None, **kwargs):
        # type: (Text, **Any) -> Dict[Text, Any]
        """"Combine slot values and key word arguments to fill templates."""

        # Getting the slot values in the template variables
        template_vars = self._template_variables(filled_slots, kwargs)

        # Filling the template variables in the template
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
        # type: (Dict[Text, Any], Dict[Text, Any]) -> Dict[Text, Any]
        """Combine slot values and key word arguments to fill templates."""

        if filled_slots is None:
            filled_slots = {}

        # Copying the filled slots in the template variables.
        template_vars = filled_slots.copy()
        template_vars.update(kwargs.items())
        return template_vars
