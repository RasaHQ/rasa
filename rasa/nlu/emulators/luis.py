from typing import Any, Dict, List, Text

from rasa.nlu.emulators.emulator import Emulator
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    INTENT_RANKING_KEY,
    TEXT,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)


class LUISEmulator(Emulator):
    """Emulates the response format of the LUIS Endpoint API v3.0 /predict endpoint.

    https://westcentralus.dev.cognitive.microsoft.com/docs/services/luis-endpoint-api-v3-0/
    https://docs.microsoft.com/en-us/azure/cognitive-services/LUIS/luis-concept-data-extraction?tabs=V3
    """

    def _intents(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        if data.get(INTENT_RANKING_KEY):
            return {
                intent[INTENT_NAME_KEY]: {"score": intent[PREDICTED_CONFIDENCE_KEY]}
                for intent in data[INTENT_RANKING_KEY]
            }

        top = data.get(INTENT)
        if not top:
            return {}

        return {top[INTENT_NAME_KEY]: {"score": top[PREDICTED_CONFIDENCE_KEY]}}

    def _entities(
        self, data: Dict[Text, Any]
    ) -> Dict[Text, Dict[Text, List[Dict[Text, Any]]]]:
        if ENTITIES not in data:
            return {}

        entities = {"$instance": {}}
        for e in data[ENTITIES]:
            # LUIS API v3 uses entity roles instead of entity names
            # (it's possible because its roles are unique):
            # https://docs.microsoft.com/en-us/azure/cognitive-services/LUIS/luis-migration-api-v3#entity-role-name-instead-of-entity-name
            key = e.get(ENTITY_ATTRIBUTE_ROLE, e[ENTITY_ATTRIBUTE_TYPE])
            entities[key] = [e[ENTITY_ATTRIBUTE_VALUE]]

            entities["$instance"][key] = [
                {
                    "role": e.get(ENTITY_ATTRIBUTE_ROLE),
                    "type": e[ENTITY_ATTRIBUTE_TYPE],
                    "text": e[ENTITY_ATTRIBUTE_VALUE],
                    "startIndex": e.get(ENTITY_ATTRIBUTE_START),
                    "length": (e[ENTITY_ATTRIBUTE_END] - e[ENTITY_ATTRIBUTE_START])
                    if ENTITY_ATTRIBUTE_START in e and ENTITY_ATTRIBUTE_END in e
                    else None,
                    "score": e.get(PREDICTED_CONFIDENCE_KEY),
                    "modelType": e.get(EXTRACTOR),
                }
            ]
        return entities

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform response JSON to LUIS format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
        top = data.get(INTENT)

        return {
            "query": data[TEXT],
            "prediction": {
                "normalizedQuery": data[TEXT],
                "topIntent": top[INTENT_NAME_KEY] if top else None,
                "intents": self._intents(data),
                "entities": self._entities(data),
            },
        }
