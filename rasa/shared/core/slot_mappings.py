from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Tuple, Union, cast

from pydantic import BaseModel, Field

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_NLU_BASED_SLOTS, IGNORED_INTENTS
from rasa.shared.core.constants import (
    ACTIVE_LOOP,
    KEY_ACTION,
    KEY_MAPPING_TYPE,
    KEY_RUN_ACTION_EVERY_TURN,
    MAPPING_CONDITIONS,
    REQUESTED_SLOT,
    SlotMappingType,
)
from rasa.shared.core.slots import ListSlot, Slot
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    INTENT_NAME_KEY,
    TEXT,
)

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.nlu.training_data.message import Message
    from rasa.utils.endpoints import EndpointConfig


logger = logging.getLogger(__name__)


class SlotMappingCondition(BaseModel):
    """Defines a condition for a slot mapping."""

    active_loop: Optional[str]
    requested_slot: Optional[str] = None
    active_flow: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SlotMappingCondition:
        # we allow None as a valid value for active_loop
        # therefore we need to set a different default value
        active_loop = data.pop(ACTIVE_LOOP, "")

        return SlotMappingCondition(active_loop=active_loop, **data)

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class SlotMapping(BaseModel):
    """Defines functionality for the available slot mappings."""

    type: SlotMappingType
    conditions: List[SlotMappingCondition] = Field(default_factory=list)
    entity: Optional[str] = None
    intent: Optional[Union[str, List[str]]] = None
    role: Optional[str] = None
    group: Optional[str] = None
    not_intent: Optional[Union[str, List[str]]] = None
    value: Optional[Any] = None
    allow_nlu_correction: Optional[bool] = None
    run_action_every_turn: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict[str, Any], slot_name: str) -> SlotMapping:
        data_copy = copy.deepcopy(data)
        mapping_type = SlotMapping.validate_mapping(data_copy, slot_name)
        conditions = [
            SlotMappingCondition.from_dict(condition)
            for condition in data_copy.pop(MAPPING_CONDITIONS, [])
        ]

        deprecated_action = data_copy.pop(KEY_ACTION, None)
        if deprecated_action:
            rasa.shared.utils.io.raise_deprecation_warning(
                f"The `{KEY_ACTION}` key in slot mappings is deprecated and "
                f"will be removed in Rasa Pro 4.0.0. "
                f"Please use the `{KEY_RUN_ACTION_EVERY_TURN}` key instead.",
            )
            data_copy[KEY_RUN_ACTION_EVERY_TURN] = deprecated_action

        run_action_every_turn = data_copy.pop(KEY_RUN_ACTION_EVERY_TURN, None)

        return SlotMapping(
            type=mapping_type,
            conditions=conditions,
            run_action_every_turn=run_action_every_turn,
            **data_copy,
        )

    def as_dict(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        data[KEY_MAPPING_TYPE] = self.type.value

        if self.conditions:
            data[MAPPING_CONDITIONS] = [
                condition.as_dict() for condition in self.conditions
            ]
        else:
            data.pop(MAPPING_CONDITIONS, None)

        return data

    @staticmethod
    def validate_mapping(mapping: Dict[str, Any], slot_name: str) -> SlotMappingType:
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

        mapping_raw = mapping.pop(KEY_MAPPING_TYPE, SlotMappingType.FROM_LLM.value)

        if mapping_raw == "custom":
            rasa.shared.utils.io.raise_deprecation_warning(
                "The `custom` slot mapping type is deprecated and "
                "will be removed in Rasa Pro 4.0.0. "
                "Please use the `controlled` slot mapping type instead.",
            )
            mapping_raw = "controlled"

        try:
            mapping_type = SlotMappingType(mapping_raw)
        except ValueError:
            raise InvalidDomain(
                f"Your domain uses an invalid slot mapping of type "
                f"'{mapping_raw}' for slot '{slot_name}'. Please see "
                f"{DOCS_URL_NLU_BASED_SLOTS} for more information."
            )

        validations: Dict[SlotMappingType, List[Text]] = {
            SlotMappingType.FROM_ENTITY: ["entity"],
            SlotMappingType.FROM_INTENT: ["value"],
            SlotMappingType.FROM_TRIGGER_INTENT: ["value"],
            SlotMappingType.FROM_TEXT: [],
            SlotMappingType.CONTROLLED: [],
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

        return mapping_type

    def _get_active_loop_ignored_intents(
        self, domain: "Domain", active_loop_name: Text
    ) -> List[Text]:
        mapping_conditions = self.conditions
        active_loop_match = True
        ignored_intents = []

        if mapping_conditions:
            match_list = [
                condition.active_loop == active_loop_name
                for condition in mapping_conditions
            ]
            active_loop_match = any(match_list)

        if active_loop_match:
            form_ignored_intents = domain.forms.get(active_loop_name, {}).get(
                IGNORED_INTENTS, []
            )
            ignored_intents = SlotMapping.to_list(form_ignored_intents)

        return ignored_intents

    def intent_is_desired(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        message: Optional["Message"] = None,
    ) -> bool:
        """Checks whether user intent matches slot mapping intent specifications."""
        mapping_intents = SlotMapping.to_list(self.intent)
        mapping_not_intents = SlotMapping.to_list(self.not_intent)

        active_loop_name = tracker.active_loop_name
        if active_loop_name:
            mapping_not_intents = (
                mapping_not_intents
                + self._get_active_loop_ignored_intents(domain, active_loop_name)
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

    def entity_is_desired(
        self,
        tracker: "DialogueStateTracker",
        message: Optional["Message"] = None,
    ) -> List[str]:
        """Checks whether slot should be filled by an entity in the input or not.

        Args:
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
                if entity.get(ENTITY_ATTRIBUTE_TYPE) == self.entity
                and entity.get(ENTITY_ATTRIBUTE_GROUP) == self.group
                and entity.get(ENTITY_ATTRIBUTE_ROLE) == self.role
            ]
        elif tracker.latest_message and tracker.latest_message.text is not None:
            matching_values = list(
                tracker.get_latest_entity_values(
                    self.entity,
                    self.role,
                    self.group,
                )
            )
        else:
            matching_values = []

        return matching_values

    def check_mapping_validity(
        self,
        slot_name: Text,
        domain: "Domain",
    ) -> bool:
        """Checks the mapping for validity.

        Args:
            slot_name: The name of the slot to be validated.
            domain: The domain to check against.

        Returns:
            True, if intent and entity specified in a mapping exist in domain.
        """
        if (
            self.type == SlotMappingType.FROM_ENTITY
            and self.entity not in domain.entities
        ):
            rasa.shared.utils.io.raise_warning(
                f"Slot '{slot_name}' uses a 'from_entity' mapping "
                f"for a non-existent entity '{self.entity}'. "
                f"Skipping slot extraction because of invalid mapping."
            )
            return False

        if self.type == SlotMappingType.FROM_INTENT and self.intent is not None:
            intent_list = SlotMapping.to_list(self.intent)
            for intent in intent_list:
                if intent and intent not in domain.intents:
                    rasa.shared.utils.io.raise_warning(
                        f"Slot '{slot_name}' uses a 'from_intent' mapping for "
                        f"a non-existent intent '{intent}'. "
                        f"Skipping slot extraction because of invalid mapping."
                    )
                    return False

        return True


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
        mapping: SlotMapping,
    ) -> bool:
        """Check if a slot mapping is valid."""
        return mapping.check_mapping_validity(
            slot_name=slot_name,
            domain=self.domain,
        )

    def is_intent_desired(self, mapping: SlotMapping) -> bool:
        """Check if the intent matches the one indicated in the slot mapping."""
        return mapping.intent_is_desired(
            tracker=self.tracker,
            domain=self.domain,
            message=self.message,
        )

    def _verify_mapping_conditions(self, mapping: SlotMapping, slot_name: Text) -> bool:
        if mapping.conditions and mapping.type != str(
            SlotMappingType.FROM_TRIGGER_INTENT
        ):
            return self._matches_mapping_conditions(mapping, slot_name)

        return True

    def _matches_mapping_conditions(
        self, mapping: SlotMapping, slot_name: Text
    ) -> bool:
        slot_mapping_conditions = mapping.conditions

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
        slot_mapping_conditions: List[SlotMappingCondition],
    ) -> bool:
        active_flow_conditions = list(
            filter(lambda x: x.active_flow is not None, slot_mapping_conditions)
        )
        return any(
            [
                condition.active_flow == active_flow
                for condition in active_flow_conditions
            ]
        )

    def _mapping_conditions_match_form(
        self, slot_name: str, slot_mapping_conditions: List[SlotMappingCondition]
    ) -> bool:
        if (
            self.tracker.is_active_loop_rejected
            and self.tracker.get_slot(REQUESTED_SLOT) == slot_name
        ):
            return False

        # check if found mapping conditions matches form
        for condition in slot_mapping_conditions:
            active_loop = condition.active_loop

            if active_loop and active_loop == self.tracker.active_loop_name:
                condition_requested_slot = condition.requested_slot
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
        mapping: SlotMapping,
    ) -> bool:
        from rasa.core.actions.forms import FormAction

        if mapping.type != SlotMappingType.FROM_ENTITY:
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

    def _is_trigger_intent_mapping_condition_met(self, mapping: SlotMapping) -> bool:
        active_loops_in_mapping_conditions = [
            condition.active_loop for condition in mapping.conditions
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
        mapping: SlotMapping,
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
            value = mapping.entity_is_desired(self.tracker, self.message)
        elif should_fill_intent_slot or should_fill_trigger_slot:
            value = [mapping.value]
        elif should_fill_text_slot:
            value = [self.message.get(TEXT)] if self.message is not None else []
            if not value:
                value = [
                    self.tracker.latest_message.text
                    if self.tracker.latest_message is not None
                    else None
                ]

        return value

    def should_fill_slot(self, slot_name: str, mapping: SlotMapping) -> bool:
        """Checks if a slot should be filled based on the conversation context."""
        if not self.is_slot_mapping_valid(slot_name, mapping):
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
        mapping_type = mapping.type

        if mapping_type in [SlotMappingType.FROM_LLM, SlotMappingType.CONTROLLED]:
            continue

        if not slot_filling_manager.should_fill_slot(slot.name, mapping):
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
