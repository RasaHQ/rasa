from collections import defaultdict
from typing import Any, Dict, Text

from rasa.nlu.emulators.emulator import Emulator
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    TEXT,
    INTENT,
)


class WitEmulator(Emulator):
    """Emulates the response format of this wit.ai endpoint.

    More information about the endpoint:
    https://wit.ai/docs/http/20200513/#get__message_link
    """

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform response JSON to wit.ai format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
        entities = defaultdict(list)
        for entity in data[ENTITIES]:
            entity_name = entity[ENTITY_ATTRIBUTE_TYPE]
            role = entity.get(ENTITY_ATTRIBUTE_ROLE, entity_name)
            entity_name_including_role = f"{entity[ENTITY_ATTRIBUTE_TYPE]}:{role}"
            normalized_entity: Dict[Text, Any] = {
                "confidence": entity.get("confidence_entity") or 1,
                "name": entity_name,
                "value": entity[ENTITY_ATTRIBUTE_VALUE],
                # Entity value before value was transformed (e.g. by synonym mapper)
                "body": data["text"][
                    entity.get(ENTITY_ATTRIBUTE_START, 0) : entity.get(
                        ENTITY_ATTRIBUTE_END, 0
                    )
                ],
                "start": entity[ENTITY_ATTRIBUTE_START],
                "end": entity[ENTITY_ATTRIBUTE_END],
                "role": role,
                "entities": [],
            }

            entities[entity_name_including_role].append(normalized_entity)

        return {"text": data[TEXT], "intents": [data[INTENT]], "entities": entities}
