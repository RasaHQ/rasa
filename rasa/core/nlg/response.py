import copy
import logging

from rasa.shared.core.trackers import DialogueStateTracker
from typing import Text, Any, Dict, Optional, List

from rasa.core.nlg import interpolator
from rasa.core.nlg.generator import NaturalLanguageGenerator, ResponseVariationFilter
from rasa.shared.constants import RESPONSE_CONDITION

logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    """Natural language generator that generates messages based on responses.

    The responses can use variables to customize the utterances based on the
    state of the dialogue.
    """

    def __init__(self, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        """Creates a Template Natural Language Generator.

        Args:
            responses: responses that will be used to generate messages.
        """
        self.responses = responses

    # noinspection PyUnusedLocal
    def _random_response_for(
        self, utter_action: Text, output_channel: Text, filled_slots: Dict[Text, Any]
    ) -> Optional[Dict[Text, Any]]:
        """Select random response for the utter action from available ones.

        If channel-specific responses for the current output channel are given,
        only choose from channel-specific ones.
        """
        import numpy as np

        if utter_action in self.responses:
            response_filter = ResponseVariationFilter(self.responses)
            suitable_responses = response_filter.responses_for_utter_action(
                utter_action, output_channel, filled_slots
            )

            if suitable_responses:
                selected_response = np.random.choice(suitable_responses)
                condition = selected_response.get(RESPONSE_CONDITION)
                if condition:
                    formatted_response_conditions = self._format_response_conditions(
                        condition
                    )
                    logger.debug(
                        "Selecting response variation with conditions:"
                        f"{formatted_response_conditions}"
                    )
                return selected_response
            else:
                return None
        else:
            return None

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action."""
        filled_slots = tracker.current_slot_values()
        return self.generate_from_slots(
            utter_action, filled_slots, output_channel, **kwargs
        )

    def generate_from_slots(
        self,
        utter_action: Text,
        filled_slots: Dict[Text, Any],
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action."""
        # Fetching a random response for the passed utter action
        r = copy.deepcopy(
            self._random_response_for(utter_action, output_channel, filled_slots)
        )
        # Filling the slots in the response with placeholders and returning the response
        if r is not None:
            return self._fill_response(r, filled_slots, **kwargs)
        else:
            return None

    def _fill_response(
        self,
        response: Dict[Text, Any],
        filled_slots: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """Combine slot values and key word arguments to fill responses."""
        # Getting the slot values in the response variables
        response_vars = self._response_variables(filled_slots, kwargs)

        keys_to_interpolate = [
            "text",
            "image",
            "custom",
            "buttons",
            "attachment",
            "quick_replies",
        ]
        if response_vars:
            for key in keys_to_interpolate:
                if key in response:
                    response[key] = interpolator.interpolate(
                        response[key], response_vars
                    )
        return response

    @staticmethod
    def _response_variables(
        filled_slots: Dict[Text, Any], kwargs: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        """Combine slot values and key word arguments to fill responses."""
        if filled_slots is None:
            filled_slots = {}

        # Copying the filled slots in the response variables.
        response_vars = filled_slots.copy()
        response_vars.update(kwargs)
        return response_vars

    @staticmethod
    def _format_response_conditions(response_conditions: List[Dict[Text, Any]]) -> Text:
        formatted_response_conditions = [""]
        for index, condition in enumerate(response_conditions):
            constraints = []
            constraints.append(f"type: {str(condition['type'])}")
            constraints.append(f"name: {str(condition['name'])}")
            constraints.append(f"value: {str(condition['value'])}")

            condition_message = " | ".join(constraints)
            formatted_condition = f"[condition {str(index + 1)}] {condition_message}"
            formatted_response_conditions.append(formatted_condition)

        return "\n".join(formatted_response_conditions)
