import copy
import logging
import re

from rasa_core.trackers import DialogueStateTracker
from typing import Text, Any, Dict, Optional, List

from rasa_core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    """Natural language generator that generates messages based on templates.

    The templates can use variables to customize the utterances based on the
    state of the dialogue."""

    def __init__(self, templates: Dict[Text, List[Dict[Text, Any]]]) -> None:
        self.templates = templates

    # noinspection PyUnusedLocal
    def _random_template_for(self,
                             utter_action: Text,
                             output_channel: Text
                             ) -> Optional[Dict[Text, Any]]:
        """Select random template for the utter action from available ones."""
        import numpy as np

        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    def generate(self,
                 template_name: Text,
                 tracker: DialogueStateTracker,
                 output_channel: Text,
                 **kwargs: Any
                 ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested template."""

        filled_slots = tracker.current_slot_values()
        return self.generate_from_slots(template_name,
                                        filled_slots,
                                        output_channel,
                                        **kwargs)

    def generate_from_slots(self,
                            template_name: Text,
                            filled_slots: Dict[Text, Any],
                            output_channel: Text,
                            **kwargs: Any
                            ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested template."""

        # Fetching a random template for the passed template name
        r = copy.deepcopy(self._random_template_for(template_name,
                                                    output_channel))
        # Filling the slots in the template and returning the template
        if r is not None:
            return self._fill_template_text(r, filled_slots, **kwargs)
        else:
            return None

    def _fill_template_text(
        self,
        template: Dict[Text, Any],
        filled_slots: Optional[Dict[Text, Any]] = None,
        **kwargs: Any
    ) -> Dict[Text, Any]:
        """"Combine slot values and key word arguments to fill templates."""

        # Getting the slot values in the template variables
        template_vars = self._template_variables(filled_slots, kwargs)

        # Filling the template variables in the template
        if template_vars:
            try:
                # transforming template tags from
                # "{tag_name}" to "{0[tag_name]}"
                # as described here:
                # https://stackoverflow.com/questions/7934620/python-dots-in-the-name-of-variable-in-a-format-string#comment9695339_7934969
                # assuming that slot_name do not contain newline character here
                text = re.sub(r'{([^\n]+?)}', r'{0[\1]}', template["text"])
                template["text"] = text.format(template_vars)
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
    def _template_variables(filled_slots: Dict[Text, Any],
                            kwargs: Dict[Text, Any]) -> Dict[Text, Any]:
        """Combine slot values and key word arguments to fill templates."""

        if filled_slots is None:
            filled_slots = {}

        # Copying the filled slots in the template variables.
        template_vars = filled_slots.copy()
        template_vars.update(kwargs)
        return template_vars
