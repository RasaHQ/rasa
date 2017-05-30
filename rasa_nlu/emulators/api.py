from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
import uuid
from datetime import datetime

from typing import Any
from typing import Dict
from typing import Text
from typing import List

from rasa_nlu.emulators import NoEmulator


class ApiEmulator(NoEmulator):
    def __init__(self):
        # type: () -> None

        super(ApiEmulator, self).__init__()
        self.name = 'api'

    def normalise_response_json(self, data):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        """Transform data to API.ai format."""

        # populate entities dict
        entities = {
            entity_type: []
            for entity_type in set([x["entity"] for x in data["entities"]])}  # type: Dict[Text, List[Text]]

        for entity in data["entities"]:
            entities[entity["entity"]].append(entity["value"])

        return {
            "id": str(uuid.uuid1()),
            "timestamp": datetime.now().isoformat(),
            "result": {
                "source": "agent",
                "resolvedQuery": data["text"],
                "action": None,
                "actionIncomplete": None,
                "parameters": entities,
                "contexts": [],
                "metadata": {
                    "intentId": str(uuid.uuid1()),
                    "webhookUsed": "false",
                    "intentName": data["intent"]
                },
                "fulfillment": {},
                "score": None,
            },
            "status": {
                "code": 200,
                "errorType": "success"
            },
            "sessionId": str(uuid.uuid1())
        }
