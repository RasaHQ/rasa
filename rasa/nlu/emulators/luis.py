from typing import Any, Dict, Text

from rasa.nlu.emulators.emulator import Emulator
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    INTENT_RANKING_KEY,
    TEXT,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from typing import List, Optional


class LUISEmulator(Emulator):
    """Emulates the response format of the LUIS Endpoint API v3.0 /predict endpoint.

    https://westcentralus.dev.cognitive.microsoft.com/docs/services/luis-endpoint-api-v3-0/
    https://docs.microsoft.com/en-us/azure/cognitive-services/LUIS/luis-concept-data-extraction?tabs=V3
    """

    def _top_intent(self, data) -> Optional[Dict[Text, Any]]:
        intent = data.get(INTENT)

        if not intent:
            return None

        return {
            "intent": intent[INTENT_NAME_KEY],
            "score": intent[PREDICTED_CONFIDENCE_KEY],
        }

    def _intents(self, data) -> Dict[Text, Any]:
        if data.get(INTENT_RANKING_KEY):
            return {
                el[INTENT_NAME_KEY]: {"score": el[PREDICTED_CONFIDENCE_KEY]}
                for el in data[INTENT_RANKING_KEY]
            }

        top = self._top_intent(data)
        if not top:
            return {}

        return {top["intent"]: {"score": top["score"]}}

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform response JSON to LUIS format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
        top = self._top_intent(data)

        entities = {"$instance": {}}
        for e in data[ENTITIES]:
            # LUIS API v3 uses entity roles instead of entity names
            # (it's possible because its roles are unique):
            # https://docs.microsoft.com/en-us/azure/cognitive-services/LUIS/luis-migration-api-v3#entity-role-name-instead-of-entity-name
            key = e.get(ENTITY_ATTRIBUTE_ROLE, e[ENTITY_ATTRIBUTE_VALUE])

            entities.update({key: [e[ENTITY_ATTRIBUTE_VALUE]]})
            entities["$instance"].update(
                {
                    "role": e.get(ENTITY_ATTRIBUTE_ROLE),
                    "type": e[ENTITY_ATTRIBUTE_TYPE],
                    "text": e[ENTITY_ATTRIBUTE_VALUE],
                    "startIndex": e.get(ENTITY_ATTRIBUTE_START),
                    "length": len(e[ENTITY_ATTRIBUTE_VALUE]),
                    "score": e.get(PREDICTED_CONFIDENCE_KEY),
                    "modelTypeId": 1,
                    "modelType": "Entity Extractor",
                }
            )

        return {
            "query": data[TEXT],
            "prediction": {
                "normalizedQuery": data[TEXT],
                "topIntent": top[INTENT] if top else None,
                "intents": self._intents(data),
                "entities": entities,
            },
        }
