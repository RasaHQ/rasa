from typing import Text, Dict, Any, List, Optional, TYPE_CHECKING

from rasa.shared.constants import DOCS_URL_SLOTS, IGNORED_INTENTS
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
    INTENT,
    NOT_INTENT,
    INTENT_NAME_KEY,
)
from rasa.shared.core.constants import (
    SLOT_MAPPINGS,
    MAPPING_TYPE,
    SlotMappingType,
    MAPPING_CONDITIONS,
)

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain


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
                f"{DOCS_URL_SLOTS} for more information."
            )

        try:
            mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))
        except ValueError:
            raise InvalidDomain(
                f"Your domain uses an invalid slot mapping of type "
                f"'{mapping.get(MAPPING_TYPE)}' for slot '{slot_name}'. Please see "
                f"{DOCS_URL_SLOTS} for more information."
            )

        validations: Dict[SlotMappingType, List[Text]] = {
            SlotMappingType.FROM_ENTITY: ["entity"],
            SlotMappingType.FROM_INTENT: ["value"],
            SlotMappingType.FROM_TRIGGER_INTENT: ["value"],
            SlotMappingType.FROM_TEXT: [],
            SlotMappingType.CUSTOM: [],
        }

        required_keys = validations[mapping_type]
        for required_key in required_keys:
            if mapping.get(required_key) is None:
                raise InvalidDomain(
                    f"You need to specify a value for the key "
                    f"'{required_key}' in the slot mapping of type '{mapping_type}' "
                    f"for slot '{slot_name}'. Please see "
                    f"{DOCS_URL_SLOTS} for more information."
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
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker", domain: "Domain"
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

        if tracker.latest_message:
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
        mapping: Dict[Text, Any], tracker: "DialogueStateTracker"
    ) -> bool:
        """Checks whether slot should be filled by an entity in the input or not.

        Args:
            mapping: Slot mapping.
            tracker: The tracker.

        Returns:
            True, if slot should be filled, false otherwise.
        """
        slot_fulfils_entity_mapping = False
        if tracker.latest_message:
            extracted_entities = tracker.latest_message.entities
        else:
            extracted_entities = []

        for entity in extracted_entities:
            if (
                mapping.get(ENTITY_ATTRIBUTE_TYPE) == entity[ENTITY_ATTRIBUTE_TYPE]
                and mapping.get(ENTITY_ATTRIBUTE_ROLE)
                == entity.get(ENTITY_ATTRIBUTE_ROLE)
                and mapping.get(ENTITY_ATTRIBUTE_GROUP)
                == entity.get(ENTITY_ATTRIBUTE_GROUP)
            ):
                matching_values = tracker.get_latest_entity_values(
                    mapping.get(ENTITY_ATTRIBUTE_TYPE),
                    mapping.get(ENTITY_ATTRIBUTE_ROLE),
                    mapping.get(ENTITY_ATTRIBUTE_GROUP),
                )
                slot_fulfils_entity_mapping = matching_values is not None
                break

        return slot_fulfils_entity_mapping

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
            and mapping.get(INTENT) not in domain.intents
        ):
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
        f"Please refer to {DOCS_URL_SLOTS} to learn more.",
        UserWarning,
    )

    for slot_name, properties in domain_slots.items():
        mappings = properties.get(SLOT_MAPPINGS)

        for slot_mapping in mappings:
            SlotMapping.validate(slot_mapping, slot_name)
