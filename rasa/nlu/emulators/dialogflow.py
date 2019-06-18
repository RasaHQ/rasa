import uuid
from datetime import datetime
from typing import Any, Dict, Text

from rasa.nlu.emulators.no_emulator import NoEmulator


class DialogflowEmulator(NoEmulator):
    def __init__(self) -> None:

        super(DialogflowEmulator, self).__init__()
        self.name = "api"

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Transform data to Dialogflow format."""

        # populate entities dict
        entities = {
            entity_type: []
            for entity_type in set([x["entity"] for x in data["entities"]])
        }

        for entity in data["entities"]:
            entities[entity["entity"]].append(entity["value"])

        return {
            "id": str(uuid.uuid1()),
            "timestamp": datetime.now().isoformat(),
            "result": {
                "source": "agent",
                "resolvedQuery": data["text"],
                "action": data["intent"]["name"],
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
            },
            "status": {"code": 200, "errorType": "success"},
            "sessionId": str(uuid.uuid1()),
        }
