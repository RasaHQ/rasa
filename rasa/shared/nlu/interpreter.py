import json
import logging
import re
from json.decoder import JSONDecodeError
from typing import Text, Optional, Dict, Any, Union, List, Tuple

import rasa.shared
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.nlu.training_data.message import Message


logger = logging.getLogger(__name__)


class NaturalLanguageInterpreter:
    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[Text, Any]:
        raise NotImplementedError(
            "Interpreter needs to be able to parse messages into structured output."
        )

    def featurize_message(self, message: Message) -> Optional[Message]:
        pass


class RegexInterpreter(NaturalLanguageInterpreter):
    @staticmethod
    def allowed_prefixes() -> Text:
        return INTENT_MESSAGE_PREFIX

    @staticmethod
    def _create_entities(
        parsed_entities: Dict[Text, Union[Text, List[Text]]], sidx: int, eidx: int
    ) -> List[Dict[Text, Any]]:
        entities = []
        for k, vs in parsed_entities.items():
            if not isinstance(vs, list):
                vs = [vs]
            for value in vs:
                entities.append(
                    {
                        "entity": k,
                        "start": sidx,
                        "end": eidx,  # can't be more specific
                        "value": value,
                    }
                )
        return entities

    @staticmethod
    def _parse_parameters(
        entity_str: Text, sidx: int, eidx: int, user_input: Text
    ) -> List[Dict[Text, Any]]:
        if entity_str is None or not entity_str.strip():
            # if there is nothing to parse we will directly exit
            return []

        try:
            parsed_entities = json.loads(entity_str)
            if isinstance(parsed_entities, dict):
                return RegexInterpreter._create_entities(parsed_entities, sidx, eidx)
            else:
                raise ValueError(
                    f"Parsed value isn't a json object "
                    f"(instead parser found '{type(parsed_entities)}')"
                )
        except (JSONDecodeError, ValueError) as e:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line "
                f"'{user_input}'. Failed to decode parameters. "
                f"Make sure your regex message is in the format:"
                f"\<intent_name>@<confidence-value><dictionary of entities>"  # noqa:  W505, W605, E501
                f"Error: {e}",
            )
            return []

    @staticmethod
    def _parse_confidence(confidence_str: Text) -> float:
        if confidence_str is None:
            return 1.0

        try:
            return float(confidence_str.strip()[1:])
        except ValueError as e:
            rasa.shared.utils.io.raise_warning(
                f"Invalid to parse confidence value in line "
                f"'{confidence_str}'. Make sure the intent confidence is an "
                f"@ followed by a decimal number. "
                f"Error: {e}",
            )
            return 0.0

    def _starts_with_intent_prefix(self, text: Text) -> bool:
        for c in self.allowed_prefixes():
            if text.startswith(c):
                return True
        return False

    @staticmethod
    def extract_intent_and_entities(
        user_input: Text,
    ) -> Tuple[Optional[Text], float, List[Dict[Text, Any]]]:
        """Parse the user input using regexes to extract intent & entities."""

        prefixes = re.escape(RegexInterpreter.allowed_prefixes())
        # the regex matches "slot{"a": 1}"
        m = re.search("^[" + prefixes + "]?([^{@]+)(@[0-9.]+)?([{].+)?", user_input)
        if m is not None:
            event_name = m.group(1).strip()
            confidence = RegexInterpreter._parse_confidence(m.group(2))
            entities = RegexInterpreter._parse_parameters(
                m.group(3), m.start(3), m.end(3), user_input
            )

            return event_name, confidence, entities
        else:
            logger.warning(f"Failed to parse intent end entities from '{user_input}'.")
            return None, 0.0, []

    async def parse(
        self,
        text: Text,
        message_id: Optional[Text] = None,
        tracker: Optional[DialogueStateTracker] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[Text, Any]:
        """Parse a text message."""

        return self.synchronous_parse(text)

    def synchronous_parse(self, text: Text) -> Dict[Text, Any]:
        """Parse a text message."""

        intent, confidence, entities = self.extract_intent_and_entities(text)

        if self._starts_with_intent_prefix(text):
            message_text = text
        else:
            message_text = INTENT_MESSAGE_PREFIX + text

        return {
            "text": message_text,
            "intent": {INTENT_NAME_KEY: intent, "confidence": confidence},
            "intent_ranking": [{INTENT_NAME_KEY: intent, "confidence": confidence}],
            "entities": entities,
        }
