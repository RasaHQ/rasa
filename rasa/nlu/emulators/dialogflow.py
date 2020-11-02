import uuid
from datetime import datetime
from typing import Any, Dict, Text

from rasa.shared.nlu.constants import INTENT_NAME_KEY
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
        entities = {
            entity_type: [] for entity_type in {x["entity"] for x in data["entities"]}
        }

        for entity in data["entities"]:
            entities[entity["entity"]].append(entity["value"])

        return {
            "responseId": str(uuid.uuid1()),
            "queryResult": {
                "resolvedQuery": data["text"],
                "action": data["intent"][INTENT_NAME_KEY],
                "actionIncomplete": False,
                "parameters": entities,
                "contexts": [],
                "metadata": {
                    "intentId": str(uuid.uuid1()),
                    "webhookUsed": "false",
                    "intentName": data["intent"]["name"],
                },
                "fulfillment": {},
                "score": data["intent"]["confidence"],
            }
        }
