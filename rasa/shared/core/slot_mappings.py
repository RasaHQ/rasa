import logging
from typing import Text, Dict, Any, List, Optional, TYPE_CHECKING, Tuple, cast

from rasa.shared.constants import DOCS_URL_NLU_BASED_SLOTS, IGNORED_INTENTS
import rasa.shared.utils.io
from rasa.shared.core.slots import ListSlot, Slot
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    NOT_INTENT,
    INTENT_NAME_KEY,
    TEXT,
)
from rasa.shared.core.constants import (
    ACTIVE_FLOW,
    ACTIVE_LOOP,
    REQUESTED_SLOT,
    SLOT_MAPPINGS,
    MAPPING_TYPE,
    SlotMappingType,
    MAPPING_CONDITIONS,
)

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain
    from rasa.shared.nlu.training_data.message import Message
    from rasa.utils.endpoints import EndpointConfig


logger = logging.getLogger(__name__)


class SlotMapping:
    """Defines functionality for the available slot mappings."""

    @staticmethod
    def validate(mapping: Dict[Text, Any], slot_name: Text) -> None:
        """Validates a slot mapping.

        Args:
            mapping: The mapping which is validated.
            slot_name: The name of the slot which is mapped by this mapping.

        Raises:
            InvalidDomain: In case the slot mapping is not valid.
        """
        from rasa.shared.core.domain import InvalidDomain

        if not isinstance(mapping, dict):
            raise InvalidDomain(
                f"Please make sure that the slot mappings for slot '{slot_name}' in "
                f"your domain are valid dictionaries. Please see "
                f"{DOCS_URL_NLU_BASED_SLOTS} for more information."
            )

        try:
            mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))
        except ValueError:
            raise InvalidDomain(
                f"Your domain uses an invalid slot mapping of type "
                f"'{mapping.get(MAPPING_TYPE)}' for slot '{slot_name}'. Please see "
                f"{DOCS_URL_NLU_BASED_SLOTS} for more information."
            )

        validations: Dict[SlotMappingType, List[Text]] = {
            SlotMappingType.FROM_ENTITY: ["entity"],
            SlotMappingType.FROM_INTENT: ["value"],
            SlotMappingType.FROM_TRIGGER_INTENT: ["value"],
            SlotMappingType.FROM_TEXT: [],
            SlotMappingType.CUSTOM: [],
            SlotMappingType.FROM_LLM: [],
        }

        required_keys = validations[mapping_type]
        for required_key in required_keys:
            if mapping.get(required_key) is None:
                raise InvalidDomain(
                    f"You need to specify a value for the key "
                    f"'{required_key}' in the slot mapping of type '{mapping_type}' "
                    f"for slot '{slot_name}'. Please see "
                    f"{DOCS_URL_NLU_BASED_SLOTS} for more information."
                )

    @staticmethod
    def _get_active_loop_ignored_intents(
        mapping: Dict[Text, Any], domain: "Domain", active_loop_name: Text
    ) -> List[Text]:
        from rasa.shared.core.constants import ACTIVE_LOOP

        mapping_conditions = mapping.get(MAPPING_CONDITIONS)
        active_loop_match = True
        ignored_intents = []

        if mapping_conditions:
            match_list = [
                condition.get(ACTIVE_LOOP) == active_loop_name
                for condition in mapping_conditions
            ]
            active_loop_match = any(match_list)

        if active_loop_match:
            form_ignored_intents = domain.forms.get(active_loop_name, {}).get(
                IGNORED_INTENTS, []
            )
            ignored_intents = SlotMapping.to_list(form_ignored_intents)

        return ignored_intents

    @staticmethod
    def intent_is_desired(
        mapping: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        domain: "Domain",
        message: Optional["Message"] = None,
    ) -> bool:
        """Checks whether user intent matches slot mapping intent specifications."""
        mapping_intents = SlotMapping.to_list(mapping.get(INTENT, []))
        mapping_not_intents = SlotMapping.to_list(mapping.get(NOT_INTENT, []))

        active_loop_name = tracker.active_loop_name
        if active_loop_name:
            mapping_not_intents = (
                mapping_not_intents
                + SlotMapping._get_active_loop_ignored_intents(
                    mapping, domain, active_loop_name
                )
            )

        if message is not None:
            intent = message.get(INTENT, {}).get("name")
        elif tracker.latest_message:
            intent = tracker.latest_message.intent.get(INTENT_NAME_KEY)
        else:
            intent = None

        intent_not_blocked = not mapping_intents and intent not in set(
            mapping_not_intents
        )

        return intent_not_blocked or intent in mapping_intents

    # helpers
    @staticmethod
    def to_list(x: Optional[Any]) -> List[Any]:
        """Convert object to a list if it isn't."""
        if x is None:
            x = []
        elif not isinstance(x, list):
            x = [x]

        return x

    @staticmethod
    def entity_is_desired(
        mapping: Dict[Text, Any],
        tracker: "DialogueStateTracker",
        message: Optional["Message"] = None,
    ) -> List[str]:
        """Checks whether slot should be filled by an entity in the input or not.

        Args:
            mapping: Slot mapping.
            tracker: The tracker.
            message: The message being processed.

        Returns:
            A list of matching values.
        """
        if message is not None:
            extracted_entities = message.get(ENTITIES, [])
            matching_values = [
                cast(Text, entity[ENTITY_ATTRIBUTE_VALUE])
                for entity in extracted_entities
                if entity.get(ENTITY_ATTRIBUTE_TYPE)
                == mapping.get(ENTITY_ATTRIBUTE_TYPE)
                and entity.get(ENTITY_ATTRIBUTE_GROUP)
                == mapping.get(ENTITY_ATTRIBUTE_GROUP)
                and entity.get(ENTITY_ATTRIBUTE_ROLE)
                == mapping.get(ENTITY_ATTRIBUTE_ROLE)
            ]
        elif tracker.latest_message and tracker.latest_message.text is not None:
            matching_values = list(
                tracker.get_latest_entity_values(
                    mapping.get(ENTITY_ATTRIBUTE_TYPE),
                    mapping.get(ENTITY_ATTRIBUTE_ROLE),
                    mapping.get(ENTITY_ATTRIBUTE_GROUP),
                )
            )
        else:
            matching_values = []

        return matching_values

    @staticmethod
    def check_mapping_validity(
        slot_name: Text,
        mapping_type: SlotMappingType,
        mapping: Dict[Text, Any],
        domain: "Domain",
    ) -> bool:
        """Checks the mapping for validity.

        Args:
            slot_name: The name of the slot to be validated.
            mapping_type: The type of the slot mapping.
            mapping: Slot mapping.
            domain: The domain to check against.

        Returns:
            True, if intent and entity specified in a mapping exist in domain.
        """
        if (
            mapping_type == SlotMappingType.FROM_ENTITY
            and mapping.get(ENTITY_ATTRIBUTE_TYPE) not in domain.entities
        ):
            rasa.shared.utils.io.raise_warning(
                f"Slot '{slot_name}' uses a 'from_entity' mapping "
                f"for a non-existent entity '{mapping.get(ENTITY_ATTRIBUTE_TYPE)}'. "
                f"Skipping slot extraction because of invalid mapping."
            )
            return False

        if (
            mapping_type == SlotMappingType.FROM_INTENT
            and mapping.get(INTENT) is not None
        ):
            intent_list = SlotMapping.to_list(mapping.get(INTENT))
            for intent in intent_list:
                if intent and intent not in domain.intents:
                    rasa.shared.utils.io.raise_warning(
                        f"Slot '{slot_name}' uses a 'from_intent' mapping for "
                        f"a non-existent intent '{mapping.get('intent')}'. "
                        f"Skipping slot extraction because of invalid mapping."
                    )
                    return False

        return True


def validate_slot_mappings(domain_slots: Dict[Text, Any]) -> None:
    """Raises InvalidDomain exception if slot mappings are invalid."""
    rasa.shared.utils.io.raise_warning(
        f"Slot auto-fill has been removed in 3.0 and replaced with a "
        f"new explicit mechanism to set slots. "
        f"Please refer to {DOCS_URL_NLU_BASED_SLOTS} to learn more.",
        UserWarning,
    )

    for slot_name, properties in domain_slots.items():
        mappings = properties.get(SLOT_MAPPINGS, [])

        for slot_mapping in mappings:
            SlotMapping.validate(slot_mapping, slot_name)


class SlotFillingManager:
    """Manages slot filling based on conversation context."""

    def __init__(
        self,
        domain: "Domain",
        tracker: "DialogueStateTracker",
        message: Optional["Message"] = None,
        action_endpoint: Optional["EndpointConfig"] = None,
    ) -> None:
        self.domain = domain
        self.tracker = tracker
        self.message = message
        self._action_endpoint = action_endpoint

    def is_slot_mapping_valid(
        self,
        slot_name: str,
        mapping_type: SlotMappingType,
        mapping: Dict[str, Any],
    ) -> bool:
        """Check if a slot mapping is valid."""
        return SlotMapping.check_mapping_validity(
            slot_name=slot_name,
            mapping_type=mapping_type,
            mapping=mapping,
            domain=self.domain,
        )

    def is_intent_desired(self, mapping: Dict[str, Any]) -> bool:
        """Check if the intent matches the one indicated in the slot mapping."""
        return SlotMapping.intent_is_desired(
            mapping=mapping,
            tracker=self.tracker,
            domain=self.domain,
            message=self.message,
        )

    def _verify_mapping_conditions(
        self, mapping: Dict[Text, Any], slot_name: Text
    ) -> bool:
        if mapping.get(MAPPING_CONDITIONS) and mapping[MAPPING_TYPE] != str(
            SlotMappingType.FROM_TRIGGER_INTENT
        ):
            if not self._matches_mapping_conditions(mapping, slot_name):
                return False

        return True

    def _matches_mapping_conditions(
        self, mapping: Dict[Text, Any], slot_name: Text
    ) -> bool:
        slot_mapping_conditions = mapping.get(MAPPING_CONDITIONS)

        if not slot_mapping_conditions:
            return True

        active_flow = self.tracker.active_flow

        if active_flow:
            return self._mapping_conditions_match_flow(
                active_flow, slot_mapping_conditions
            )

        # if we are not in a flow, we could be in a form
        return self._mapping_conditions_match_form(slot_name, slot_mapping_conditions)

    @staticmethod
    def _mapping_conditions_match_flow(
        active_flow: str,
        slot_mapping_conditions: List[Dict[str, str]],
    ) -> bool:
        active_flow_conditions = list(
            filter(lambda x: x.get(ACTIVE_FLOW) is not None, slot_mapping_conditions)
        )
        return any(
            [
                condition.get(ACTIVE_FLOW) == active_flow
                for condition in active_flow_conditions
            ]
        )

    def _mapping_conditions_match_form(
        self, slot_name: str, slot_mapping_conditions: List[Dict[str, str]]
    ) -> bool:
        if (
            self.tracker.is_active_loop_rejected
            and self.tracker.get_slot(REQUESTED_SLOT) == slot_name
        ):
            return False

        # check if found mapping conditions matches form
        for condition in slot_mapping_conditions:
            # we allow None as a valid value for active_loop
            # therefore we need to set a different default value
            active_loop = condition.get(ACTIVE_LOOP, "")

            if active_loop and active_loop == self.tracker.active_loop_name:
                condition_requested_slot = condition.get(REQUESTED_SLOT)
                if not condition_requested_slot:
                    return True
                if condition_requested_slot == self.tracker.get_slot(REQUESTED_SLOT):
                    return True

            if active_loop is None and self.tracker.active_loop_name is None:
                return True

        return False

    def _fails_unique_entity_mapping_check(
        self,
        slot_name: Text,
        mapping: Dict[Text, Any],
    ) -> bool:
        from rasa.core.actions.forms import FormAction

        if mapping[MAPPING_TYPE] != str(SlotMappingType.FROM_ENTITY):
            return False

        form_name = self.tracker.active_loop_name

        if not form_name:
            return False

        if self.tracker.get_slot(REQUESTED_SLOT) == slot_name:
            return False

        form = FormAction(form_name, self._action_endpoint)

        if slot_name not in form.required_slots(self.domain):
            return False

        if form.entity_mapping_is_unique(mapping, self.domain):
            return False

        return True

    def _is_trigger_intent_mapping_condition_met(
        self, mapping: Dict[Text, Any]
    ) -> bool:
        active_loops_in_mapping_conditions = [
            condition.get(ACTIVE_LOOP)
            for condition in mapping.get(MAPPING_CONDITIONS, [])
        ]

        trigger_mapping_condition_met = True

        if self.tracker.active_loop_name is None:
            trigger_mapping_condition_met = False
        elif (
            active_loops_in_mapping_conditions
            and self.tracker.active_loop_name is not None
            and (
                self.tracker.active_loop_name not in active_loops_in_mapping_conditions
            )
        ):
            trigger_mapping_condition_met = False

        return trigger_mapping_condition_met

    def extract_slot_value_from_predefined_mapping(
        self,
        mapping_type: SlotMappingType,
        mapping: Dict[Text, Any],
    ) -> List[Any]:
        """Extracts slot value if slot has an applicable predefined mapping."""
        if (
            self.message is None
            and self.tracker.has_bot_message_after_latest_user_message()
        ):
            # TODO: this needs further validation - not sure if this breaks something!!!

            # If the bot sent a message after the user sent a message, we can't
            # extract any slots from the user message. We assume that the user
            # message was already processed by the bot and the slot value was
            # already extracted (e.g. for a prior form slot).
            return []

        should_fill_entity_slot = mapping_type == SlotMappingType.FROM_ENTITY

        should_fill_intent_slot = mapping_type == SlotMappingType.FROM_INTENT

        should_fill_text_slot = mapping_type == SlotMappingType.FROM_TEXT

        trigger_mapping_condition_met = self._is_trigger_intent_mapping_condition_met(
            mapping
        )

        should_fill_trigger_slot = (
            mapping_type == SlotMappingType.FROM_TRIGGER_INTENT
            and trigger_mapping_condition_met
        )

        value: List[Any] = []

        if should_fill_entity_slot:
            value = SlotMapping.entity_is_desired(mapping, self.tracker, self.message)
        elif should_fill_intent_slot or should_fill_trigger_slot:
            value = [mapping.get("value")]
        elif should_fill_text_slot:
            value = [self.message.get(TEXT)] if self.message is not None else []
            if not value:
                value = [
                    self.tracker.latest_message.text
                    if self.tracker.latest_message is not None
                    else None
                ]

        return value

    def should_fill_slot(
        self, slot_name: str, mapping_type: SlotMappingType, mapping: Dict[Text, Any]
    ) -> bool:
        """Checks if a slot should be filled based on the conversation context."""
        if not self.is_slot_mapping_valid(slot_name, mapping_type, mapping):
            return False

        if not self.is_intent_desired(mapping):
            return False

        if not self._verify_mapping_conditions(mapping, slot_name):
            return False

        if self._fails_unique_entity_mapping_check(slot_name, mapping):
            return False

        return True


def extract_slot_value(
    slot: Slot, slot_filling_manager: SlotFillingManager
) -> Tuple[Any, bool]:
    """Extracts the value of a slot based on the conversation context."""
    is_extracted = False

    for mapping in slot.mappings:
        mapping_type = SlotMappingType(
            mapping.get(MAPPING_TYPE, SlotMappingType.FROM_LLM.value)
        )

        if mapping_type in [SlotMappingType.FROM_LLM, SlotMappingType.CUSTOM]:
            continue

        if not slot_filling_manager.should_fill_slot(slot.name, mapping_type, mapping):
            continue

        value: List[Any] = (
            slot_filling_manager.extract_slot_value_from_predefined_mapping(
                mapping_type, mapping
            )
        )

        if value:
            if not isinstance(slot, ListSlot):
                value = value[-1]

            if (
                value is not None
                or slot_filling_manager.tracker.get_slot(slot.name) is not None
            ):
                logger.debug(f"Extracted value '{value}' for slot '{slot.name}'.")

                is_extracted = True
                return value, is_extracted

    return None, is_extracted
