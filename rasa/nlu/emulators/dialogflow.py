import uuid
from collections import defaultdict
from typing import Any, Dict, Text

from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    TEXT,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.nlu.emulators.emulator import Emulator


class DialogflowEmulator(Emulator):
    """Emulates the response format of the DialogFlow projects.agent.environments.users.sessions.detectIntent

    https://cloud.google.com/dialogflow/es/docs/reference/rest/v2/projects.agent.environments.users.sessions/detectIntent
    https://cloud.google.com/dialogflow/es/docs/reference/rest/v2/DetectIntentResponse
    """

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """"Transform response JSON to DialogFlow format.

        Args:
            data: input JSON data as a dictionary.

        Returns:
            The transformed input data.
        """
        entities = defaultdict(list)
        for entity in data[ENTITIES]:
            entities[entity[ENTITY_ATTRIBUTE_TYPE]].append(
                entity[ENTITY_ATTRIBUTE_VALUE]
            )

        return {
            "responseId": str(uuid.uuid1()),
            "queryResult": {
                "queryText": data[TEXT],
                "action": data[INTENT][INTENT_NAME_KEY],
                "parameters": entities,
                "fulfillmentText": "",
                "fulfillmentMessages": [],
                "outputContexts": [],
                "intent": {
                    "name": data[INTENT][INTENT_NAME_KEY],
                    "displayName": data[INTENT][INTENT_NAME_KEY],
                },
                "intentDetectionConfidence": data[INTENT][PREDICTED_CONFIDENCE_KEY],
            },
        }
