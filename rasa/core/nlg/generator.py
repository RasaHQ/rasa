import logging
from typing import List, Optional, Union, Text, Any, Dict

import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import CHANNEL, RESPONSE_CONDITION
from rasa.shared.core.domain import Domain
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class NaturalLanguageGenerator:
    """Generate bot utterances based on a dialogue state."""

    async def generate(
        self,
        utter_action: Text,
        tracker: "DialogueStateTracker",
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action.

        There are a lot of different methods to implement this, e.g. the
        generation can be based on responses or be fully ML based by feeding
        the dialogue state into a machine learning NLG model.
        """
        raise NotImplementedError

    @staticmethod
    def create(
        obj: Union["NaturalLanguageGenerator", EndpointConfig, None],
        domain: Optional[Domain],
    ) -> "NaturalLanguageGenerator":
        """Factory to create a generator."""
        if isinstance(obj, NaturalLanguageGenerator):
            return obj
        else:
            return _create_from_endpoint_config(obj, domain)


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None, domain: Optional[Domain] = None
) -> "NaturalLanguageGenerator":
    """Given an endpoint configuration, create a proper NLG object."""
    domain = domain or Domain.empty()

    if endpoint_config is None:
        from rasa.core.nlg import TemplatedNaturalLanguageGenerator

        # this is the default type if no endpoint config is set
        nlg: "NaturalLanguageGenerator" = TemplatedNaturalLanguageGenerator(
            domain.responses
        )
    elif endpoint_config.type is None or endpoint_config.type.lower() == "callback":
        from rasa.core.nlg import CallbackNaturalLanguageGenerator

        # this is the default type if no nlg type is set
        nlg = CallbackNaturalLanguageGenerator(endpoint_config=endpoint_config)
    elif endpoint_config.type.lower() == "response":
        from rasa.core.nlg import TemplatedNaturalLanguageGenerator

        nlg = TemplatedNaturalLanguageGenerator(domain.responses)
    else:
        nlg = _load_from_module_name_in_endpoint_config(endpoint_config, domain)

    logger.debug(f"Instantiated NLG to '{nlg.__class__.__name__}'.")
    return nlg


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig, domain: Domain
) -> "NaturalLanguageGenerator":
    """Initializes a custom natural language generator.

    Args:
        domain: defines the universe in which the assistant operates
        endpoint_config: the specific natural language generator
    """
    try:
        nlg_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return nlg_class(endpoint_config=endpoint_config, domain=domain)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a "
            f"`NaturalLanguageGenerator` instance. Error: {e}"
        )


class ResponseVariationFilter:
    """Filters response variations based on the channel, action and condition."""

    def __init__(self, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        self.responses = responses

    @staticmethod
    def _matches_filled_slots(
        filled_slots: Dict[Text, Any], response: Dict[Text, Any]
    ) -> bool:
        """Checks if the conditional response variation matches the filled slots."""
        constraints = response.get(RESPONSE_CONDITION, [])
        for constraint in constraints:
            name = constraint["name"]
            value = constraint["value"]
            filled_slots_value = filled_slots.get(name)
            if isinstance(filled_slots_value, str) and isinstance(value, str):
                if filled_slots_value.casefold() != value.casefold():
                    return False
            # slot values can be of different data types
            # such as int, float, bool, etc. hence, this check
            # executes when slot values are not strings
            elif filled_slots_value != value:
                return False

        return True

    def responses_for_utter_action(
        self,
        utter_action: Text,
        output_channel: Text,
        filled_slots: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Returns array of responses that fit the channel, action and condition."""
        # filter responses without a condition
        default_responses = list(
            filter(
                lambda x: (x.get(RESPONSE_CONDITION) is None),
                self.responses[utter_action],
            )
        )
        # filter responses with a condition that matches the filled slots
        conditional_responses = list(
            filter(
                lambda x: (
                    x.get(RESPONSE_CONDITION)
                    and self._matches_filled_slots(
                        filled_slots=filled_slots, response=x
                    )
                ),
                self.responses[utter_action],
            )
        )

        # filter conditional responses that match the channel
        conditional_channel = list(
            filter(lambda x: (x.get(CHANNEL) == output_channel), conditional_responses)
        )
        # filter conditional responses that don't match the channel
        conditional_no_channel = list(
            filter(lambda x: (x.get(CHANNEL) is None), conditional_responses)
        )
        # filter default responses that match the channel
        default_channel = list(
            filter(lambda x: (x.get(CHANNEL) == output_channel), default_responses)
        )
        # filter default responses that don't match the channel
        default_no_channel = list(
            filter(lambda x: (x.get(CHANNEL) is None), default_responses)
        )

        if conditional_channel:
            return conditional_channel

        if default_channel:
            return default_channel

        if conditional_no_channel:
            return conditional_no_channel

        return default_no_channel

    def get_response_variation_id(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
    ) -> Optional[Text]:
        """Returns the first matched response variation ID.

        This ID corresponds to the response variation that fits
        the channel, action and condition.
        """
        filled_slots = tracker.current_slot_values()
        if utter_action in self.responses:
            eligible_variations = self.responses_for_utter_action(
                utter_action, output_channel, filled_slots
            )
            response_ids_are_valid = self._validate_response_ids(eligible_variations)

            if eligible_variations and response_ids_are_valid:
                return eligible_variations[0].get("id")

        return None

    @staticmethod
    def _validate_response_ids(response_variations: List[Dict[Text, Any]]) -> bool:
        """Checks that the response IDs of a particular utter_action are unique.

        Args:
            response_variations: The response variations to validate.

        Returns:
            True if the response IDs are unique, False otherwise.
        """
        response_ids = set()
        for response_variation in response_variations:
            response_variation_id = response_variation.get("id")
            if response_variation_id and response_variation_id in response_ids:
                rasa.shared.utils.io.raise_warning(
                    f"Duplicate response id '{response_variation_id}' "
                    f"defined in the domain."
                )
                return False

            response_ids.add(response_variation_id)

        return True
