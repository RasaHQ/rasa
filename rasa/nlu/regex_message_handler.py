from __future__ import annotations
import json
from json.decoder import JSONDecodeError
import logging
import re
from typing import Any, Dict, Match, Optional, Pattern, Text, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain

# FIXME: https://rasa.com/docs/rasa/stories/ doesn't actually explain this format?...
from rasa.shared.constants import DOCS_URL_STORIES, INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class RegexMessageHandler(GraphComponent):
    """Unpacks messages where `TEXT` contains an encoding of attributes.

    The `TEXT` attribute of such messages consists of the following sub-strings:
    1. special symbol "/" (mandatory)
    2. intent name (mandatory)
    3. "@<confidence value>" where the value can be any int or float (optional)
    4. string representation of a dictionary mapping entity types to entity
       values (optional)
    """

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexMessageHandler:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls()

    def __init__(self,) -> None:
        """Creates a new instance."""
        self._prefix = self._get_prefix()
        self._pattern = self._build_pattern()

    @staticmethod
    def _get_prefix() -> Text:
        return INTENT_MESSAGE_PREFIX

    @staticmethod
    def _build_pattern() -> Pattern:
        """Builds the pattern that matches `TEXT`s of messages that need to be unpacked.

        Returns:
            pattern with named groups
        """
        return re.compile(
            f"^{re.escape(RegexMessageHandler._get_prefix())}"
            f"(?P<{INTENT_NAME_KEY}>[^{{@]+)"  # "{{" is a masked "{" in an f-string
            f"(?P<{PREDICTED_CONFIDENCE_KEY}>@[0-9.]+)?"
            f"(?P<{ENTITIES}>{{.+}})?"  # "{{" is a masked "{" in an f-string
            f"(?P<rest>.*)"
        )

    def process(self, messages: List[Message], domain: Domain) -> List[Message]:
        """Unpacks messages where `TEXT` contains an encoding of attributes.

        Note that this method returns a *new* message instance if there is
        something to unpack in the given message (and returns the given message
        otherwise). The new message is created on purpose to get rid of all attributes
        that NLU components might have added based on the `TEXT` attribute which
        does not contain real text but the regex we expect here.

        Args:
            messages: list of messages
            domain: the domain
        Returns:
            list of messages where the i-th message is equal to the i-th input message
            if that message does not need to be unpacked, and a new message with the
            extracted attributes otherwise
        """
        return [self._unpack(message, domain) for message in messages]

    def _unpack(self, message: Message, domain: Domain) -> Message:
        """Unpacks the messsage if `TEXT` contains an encoding of attributes.

        Args:
            message: some message
            domain: the domain
        Returns:
            the given message if that message does not need to be unpacked, and a new
            message with the extracted attributes otherwise
        """
        user_text = message.get(TEXT).strip()

        # If the prefix doesn't match, we don't even need to try to match the pattern.
        if not user_text.startswith(self._prefix):
            return message

        # Try to match the pattern.
        match = self._pattern.match(user_text)

        # If it doesn't match, then (potentially) something went wrong, because the
        # message text did start with the special prefix -- however, a user might
        # just have decided to start their text this way.
        if not match:
            logger.warning(f"Failed to parse intent end entities from '{user_text}'.")
            return message

        # Extract attributes from the match - and validate it via the domain.
        intent_name = RegexMessageHandler._parse_intent_name(match, domain)
        confidence = RegexMessageHandler._parse_optional_confidences(match)
        entities = RegexMessageHandler._parse_optional_entities(match, domain)

        # The intent name is *not* optional, but during parsing we might find out
        # that the given intent is unknown (and warn). In this case, stop here.
        if intent_name is None:
            return message

        if match.group("rest"):
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Failed to interpret some parts. "
                f"Continuing without {match.group('rest')}. ",
                docs=DOCS_URL_STORIES,
            )

        # Add the results to the message.
        message_data = {INTENT: {INTENT_NAME_KEY: intent_name}}
        message_data[INTENT][INTENT_RANKING_KEY] = [
            {INTENT_NAME_KEY: intent_name, PREDICTED_CONFIDENCE_KEY: confidence}
        ]
        message_data[ENTITIES] = entities
        return Message(message_data, output_properties={INTENT, ENTITIES})

    @staticmethod
    def _parse_intent_name(match: Match, domain: Domain) -> Optional[Text]:
        intent_name = match.group(INTENT_NAME_KEY).strip()
        if intent_name not in domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Expected the intent to be one of [{domain.intents}] "
                f"but found {intent_name}."
                f"Continuing with given line as user text.",
                docs=DOCS_URL_STORIES,
            )
            intent_name = None
        return intent_name

    @staticmethod
    def _parse_optional_entities(match: Match, domain: Domain) -> List[Dict[Text, Any]]:
        """Extracts the optional entity information from the given pattern match.

        If no entities are specified or if the extraction fails, then an empty list
        is returned.

        Args:
            match: a match produced by `self.pattern`
            domain: the domain
        Returns:
            some list of entities
        """
        entities_str = match.group(ENTITIES)
        if entities_str is None:
            return []

        try:
            parsed_entities = json.loads(entities_str)
            if not isinstance(parsed_entities, dict):
                raise ValueError(
                    f"Parsed value isn't a json object "
                    f"(instead parser found '{type(parsed_entities)}')"
                )
        except (JSONDecodeError, ValueError) as e:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Failed to decode parameters as a json object (dict). "
                f"Make sure the intent is followed by a proper json object (dict). "
                f"Continuing without entities. "
                f"Error: {e}",
                docs=DOCS_URL_STORIES,
            )
            parsed_entities = dict()

        # validate the given entity types
        entity_types = set(parsed_entities.keys())
        unknown_entity_types = entity_types.difference(domain.entities)
        if unknown_entity_types:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Expected entities from {domain.entities} "
                f"but found {unknown_entity_types}. "
                f"Continuing without unknown entity types. ",
                docs=DOCS_URL_STORIES,
            )
            parsed_entities = {
                key: value
                for key, value in parsed_entities
                if key not in unknown_entity_types
            }

        # convert them into the list of dictionaries that we expect
        entities: List[Dict[Text, Any]] = []
        for entity_type, entity_values in parsed_entities.items():
            if not isinstance(entity_values, list):
                entity_values = [entity_values]
            for entity_value in entity_values:
                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: entity_type,
                        ENTITY_ATTRIBUTE_VALUE: entity_value,
                    }
                )
        return entities

    @staticmethod
    def _parse_optional_confidences(match: Match) -> float:
        """Extracts the optional confidence information from the given pattern match.

        If no confidence is specified, then this method returns the maximum
        confidence `1.0`.
        If a confidence is specified but extraction fails, then this method defaults
        to a confidence of `0.0`.

        Args:
            match: a match produced by `self.pattern`
            domain: the domain

        Returns:
            some confidence value
        """
        confidence_str = match.group(PREDICTED_CONFIDENCE_KEY)
        if confidence_str is None:
            return 1.0
        try:
            confidence_str = confidence_str.strip()[1:]  # remove the "@"
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.0
                raise ValueError(
                    f"Expected confidence to be a non-negative decimal number but "
                    f"found {confidence}. Continuing with 0.0 instead."
                )
            if confidence > 1.0:
                # Due to the pattern we know that this cannot be a negative number.
                original_confidence = confidence
                confidence = min(1.0, confidence)
                raise ValueError(
                    f"Expected confidence to be at most 1.0. "
                    f"but found {original_confidence}. "
                    f"Continuing with {confidence} instead."
                )
            return confidence

        except ValueError as e:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Could not extract confidence value from `{confidence_str}'. "
                f"Make sure the intent confidence is an @ followed "
                f"by a decimal number that not negative and at most 1.0. "
                f"Error: {e}",
                docs=DOCS_URL_STORIES,
            )
            return confidence
