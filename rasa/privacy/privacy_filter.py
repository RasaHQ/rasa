import copy
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from rasa.privacy.constants import (
    DEFAULT_PII_MODEL,
    ENTITIES_KEY,
    ENTITY_LABEL_KEY,
    GLINER_LABELS,
    GLINER_MODEL_PATH_ENV_VAR_NAME,
    HUGGINGFACE_CACHE_DIR_ENV_VAR_NAME,
    TEXT_KEY,
    VALUE_KEY,
)
from rasa.privacy.privacy_config import AnonymizationMethod, AnonymizationType
from rasa.shared.core.events import BotUttered, Event, SlotSet, UserUttered

structlogger = structlog.get_logger(__name__)


class PrivacyFilter:
    """A class to anonymise sensitive information."""

    def __init__(self, anonymization_rules: Dict[str, AnonymizationMethod]) -> None:
        """Initialise the PrivacyFilter."""
        self.anonymization_rules = anonymization_rules
        self.labels = GLINER_LABELS
        self.model = self._load_gliner_model()

    def anonymize(
        self, events: List[Event], prior_sensitive_slot_events: List[Event]
    ) -> List[Event]:
        """Anonymize sensitive information in the events of the current turn.

        The order of priority for PII detection is:
        - firstly, the slot-based approach i.e. identify any defined slots in
        the anonymization rules that could have been set in this turn and
        anonymise the plaintext slot values in all 3 event types
        (UserUttered, BotUttered, SlotSet)
        - secondly, the GLiNER model based approach i.e. identify any PII entities
         and anonymise the text in UserUttered events or values of
         SlotSet events that fill from_text slots.
        """
        anonymized_events: List[Event] = []
        anonymized_slots = self._anonymize_sensitive_slots(
            (events + prior_sensitive_slot_events)
        )

        for event in events:
            anonymized_event = self._anonymize_event(event, anonymized_slots)
            anonymized_events.append(anonymized_event)

        return anonymized_events

    @staticmethod
    def _load_gliner_model() -> Optional[Any]:
        """Load the GLiNER model for PII detection."""
        local_model_path = os.getenv(GLINER_MODEL_PATH_ENV_VAR_NAME)
        cache_dir_env_value = os.getenv(HUGGINGFACE_CACHE_DIR_ENV_VAR_NAME)
        cache_dir = Path(cache_dir_env_value).resolve() if cache_dir_env_value else None
        model_path = (
            Path(local_model_path).resolve() if local_model_path else DEFAULT_PII_MODEL
        )
        local_files_only = isinstance(model_path, Path) and model_path.exists()

        structlogger.debug(
            "rasa.privacy.privacy_filter.loading_gliner_model",
            local_files_only=local_files_only,
        )

        try:
            from gliner import GLiNER

            return GLiNER.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except ImportError:
            structlogger.warning(
                "rasa.privacy.privacy_filter.gliner_import_error",
                event_info="Optional GLiNER library is not installed. "
                "Please install it if you wish to use additional "
                "PII detection to the slot based approach.",
            )
            return None

    def _anonymize_sensitive_slots(self, events: List[Event]) -> Dict[str, SlotSet]:
        """Identify and anonymize sensitive slot events.

        Returns a dictionary where the keys represent a concatenation of the slot key
        and its original value, and the values are the anonymized SlotSet events.
        """
        sensitive_slots = self._find_sensitive_slots(events)

        if not sensitive_slots:
            structlogger.debug("rasa.privacy.privacy_filter.no_sensitive_slots_found")
            return {}

        anonymized_slots = {}
        for slot in sensitive_slots:
            slot_value = (
                slot.value if isinstance(slot.value, str) else json.dumps(slot.value)
            )
            anonymized_slots[f"{slot.key}:{slot_value}"] = (
                self._anonymize_sensitive_slot_event(slot)
            )

        return anonymized_slots

    def _find_sensitive_slots(self, processed_events: List[Event]) -> List[SlotSet]:
        """Find all slot events that contain sensitive information.

        These sensitive slots are defined in the anonymization rules and
        have a non-empty value.
        """
        return [
            copy.deepcopy(slot_event)
            for slot_event in processed_events
            if isinstance(slot_event, SlotSet)
            and slot_event.key in self.anonymization_rules
            and bool(slot_event.value)
        ]

    def _anonymize_sensitive_slot_event(
        self,
        slot_event: SlotSet,
    ) -> SlotSet:
        """Anonymize the sensitive slot event if it contains sensitive information.

        A sensitive slot event is defined as a SlotSet event that has a key
        in the anonymization rules and a non-empty value.
        """
        slot_value = slot_event.value
        if not bool(slot_value):
            return slot_event

        anonymized_value = self._anonymize_value(slot_event)
        slot_event.value = anonymized_value

        return slot_event

    def _anonymize_event(
        self, event: Event, anonymized_slots: Dict[str, SlotSet]
    ) -> Event:
        if isinstance(event, SlotSet):
            return self._anonymize_slot_event(event, anonymized_slots)
        elif isinstance(event, UserUttered):
            return self._anonymize_user_event(event, anonymized_slots)
        elif isinstance(event, BotUttered):
            return self._anonymize_bot_event(event, anonymized_slots)
        else:
            return event

    def _anonymize_slot_event(
        self,
        event: SlotSet,
        anonymized_slots: Dict[str, SlotSet],
    ) -> SlotSet:
        """Anonymize the slot event if it contains sensitive information."""
        event_value = (
            event.value if isinstance(event.value, str) else json.dumps(event.value)
        )
        # obtain the anonymized slot event, otherwise return the original event
        slot_event = anonymized_slots.get(f"{event.key}:{event_value}", event)

        # apply the edge case anonymization
        slot_value = (
            slot_event.value
            if isinstance(slot_event.value, str)
            else json.dumps(slot_event.value)
        )
        anonymized_value = self._anonymize_edge_cases(slot_value, anonymized_slots)

        slot_event.value = (
            anonymized_value
            if isinstance(slot_event.value, str)
            else json.loads(anonymized_value)
        )
        slot_event.anonymized_at = datetime.datetime.now(datetime.timezone.utc)
        return slot_event

    def _anonymize_user_event(
        self,
        user_event: UserUttered,
        anonymized_slots: Dict[str, SlotSet],
    ) -> UserUttered:
        """Anonymize the user event if it contains sensitive information."""
        if not user_event.text:
            structlogger.debug(
                "rasa.privacy.privacy_filter.user_event_no_text",
            )
            return user_event

        original_parse_data: Dict[str, Any] = (
            copy.deepcopy(user_event.parse_data) if user_event.parse_data else {}
        )
        anonymized_parse_data: Dict[str, Any] = {}

        for key, slot in anonymized_slots.items():
            original_slot_value = key.split(":", 1)[1]
            anonymized_text = self._smart_replace(
                user_event.text, original_slot_value, slot.value
            )
            user_event.text = anonymized_text

            anonymized_parse_data[TEXT_KEY] = anonymized_text
            for entity in original_parse_data.get(ENTITIES_KEY, []):
                entity_value = entity[VALUE_KEY]
                if entity_value == original_slot_value:
                    anonymized_entities: List[Dict[str, Any]] = (
                        anonymized_parse_data.get(ENTITIES_KEY, [])
                    )
                    anonymized_entities.append({**entity, VALUE_KEY: slot.value})
                    anonymized_parse_data[ENTITIES_KEY] = anonymized_entities

        user_event.parse_data = anonymized_parse_data  # type: ignore[assignment]
        user_event.text = self._anonymize_edge_cases(user_event.text, anonymized_slots)
        # cover the edge case anonymization for the parse data text field
        parse_data_text = user_event.parse_data.get(TEXT_KEY, "")
        user_event.parse_data[TEXT_KEY] = self._anonymize_edge_cases(  # type: ignore[literal-required]
            parse_data_text, anonymized_slots
        )

        user_event.anonymized_at = datetime.datetime.now(datetime.timezone.utc)

        return user_event

    def _anonymize_bot_event(
        self,
        bot_event: BotUttered,
        anonymized_slots: Dict[str, SlotSet],
    ) -> BotUttered:
        """Anonymize the bot event if it contains sensitive information."""
        if not bot_event.text:
            structlogger.debug(
                "rasa.privacy.privacy_filter.bot_event_no_text",
            )
            return bot_event

        for key, slot in anonymized_slots.items():
            original_slot_value = key.split(":", 1)[1]
            anonymized_text = self._smart_replace(
                bot_event.text, original_slot_value, slot.value
            )
            bot_event.text = anonymized_text

        bot_event.text = self._anonymize_edge_cases(bot_event.text, anonymized_slots)
        bot_event.anonymized_at = datetime.datetime.now(datetime.timezone.utc)
        return bot_event

    def _anonymize_value(self, slot: SlotSet) -> str:
        """Anonymize the given slot value using the specified anonymization method."""
        slot_name = slot.key
        slot_value = slot.value
        anonymization_method = self.anonymization_rules[slot_name]

        if anonymization_method.method_type == AnonymizationType.REDACT:
            return self._redact(slot_value, anonymization_method)

        if anonymization_method.method_type == AnonymizationType.MASK:
            return self._mask(slot_name)

        # we won't reach this case, because the json schema specifies
        # the allowed methods, this is to satisfy the type checker
        return ""

    @staticmethod
    def _redact(slot_value: Any, anonymization_method: AnonymizationMethod) -> str:
        """Redact the given slot value using the specified anonymization method."""
        if anonymization_method.keep_left is not None:
            left_part = slot_value[: anonymization_method.keep_left]
        else:
            left_part = ""

        if anonymization_method.keep_right is not None:
            right_part = slot_value[-anonymization_method.keep_right :]
        else:
            right_part = ""

        return (
            left_part
            + anonymization_method.redaction_char
            * (len(slot_value) - len(left_part) - len(right_part))
            + right_part
        )

    @staticmethod
    def _mask(slot_name: str) -> str:
        """Mask the given slot value using the slot name."""
        return f"[{slot_name.upper()}]"

    @staticmethod
    def _strip_square_brackets(string: str) -> str:
        """Strip square brackets from the start and end of the string if present."""
        if len(string) >= 2 and string[0] == "[" and string[-1] == "]":
            return string[1:-1]
        return string

    def _anonymize_edge_cases(
        self, text: str, anonymized_slots: Dict[str, SlotSet]
    ) -> str:
        """Anonymize edge cases in the text using GLiNER model.

        This method is used to detect PII entities in the text that are not
        covered by the slot-based anonymization rules. For example, when
        the user message contains PII entities that are not defined as slots,
        or when the slot is filled from a text input that could contain multiple
        PII entities, such as a from_text slot.

        This method uses the GLiNER model to predict entities in the text
        and replaces them with masked values.
        If the GLiNER model is not loaded, it will skip this step and return
        the original text.
        """
        if self.model is None:
            structlogger.debug(
                "rasa.privacy.privacy_filter.gliner_model_not_loaded",
                event_info="GLiNER model is not loaded, skipping PII detection.",
            )
            return text

        entities = self.model.predict_entities(text, self.labels, threshold=0.85)

        all_anonymized_slot_values = [
            self._strip_square_brackets(str(slot.value))
            for slot in anonymized_slots.values()
        ]

        for entity in entities:
            structlogger.debug(
                "rasa.privacy.privacy_filter.pii_entity_found",
                entity=entity[ENTITY_LABEL_KEY],
            )

            entity_value = entity[TEXT_KEY]

            if entity_value in all_anonymized_slot_values:
                # the entity that was found is already anonymized,
                # we shouldn't override the already anonymized value
                # with a masked value
                structlogger.debug(
                    "rasa.privacy.privacy_filter.pii_entity_already_anonymized",
                    entity=entity[ENTITY_LABEL_KEY],
                )
                continue

            text = text.replace(entity_value, self._mask(entity[ENTITY_LABEL_KEY]))

        return text

    @staticmethod
    def _smart_replace(text: str, original_value: str, replacement: str) -> str:
        """Replace original_value with replacement in text.

        This method performs a string replacement in the text,
        with special handling for floats.
        If original_value is a float string like "24.0",
        also tries replacing the integer version "24".

        Args:
            text (str): The text to perform replacements on
            original_value (str): The value to replace
            replacement (str): The replacement value

        Returns:
            str: The text with replacements applied
        """
        # First try the original replacement
        result = text.replace(original_value, replacement)
        if text != result:
            return result

        # If replacement didn't happen and it's a float,
        # try replacing the integer version
        if "." in original_value:
            try:
                float_val = float(original_value)
                if float_val.is_integer():
                    int_version = str(int(float_val))
                    result = result.replace(int_version, replacement)
            except ValueError:
                structlogger.warning(
                    "rasa.privacy.privacy_filter.smart_replace_float_error",
                    event_info="Unable to anonymize float value.",
                )

        return result
