import uuid
from datetime import datetime


class ApiEmulator(object):
    def __init__(self):
        self.name = 'api'

    def normalise_request_json(self, data):
        _data = {}
        # for GET req data["q"] is a list. For POST req data["q"] should be a string
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        return _data

    def normalise_response_json(self, data):
        # populate entities dict
        entities = {entity_type: [] for entity_type in set(map(lambda x: x["entity"], data["entities"]))}
        for entity in data["entities"]:
            entities[entity["entity"]].append(entity["value"])

        return {
            "id": unicode(uuid.uuid1()),
            "timestamp": datetime.now().isoformat("T"),
            "result": {
                "source": "agent",
                "resolvedQuery": data["text"],
                "action": None,
                "actionIncomplete": None,
                "parameters": entities,
                "contexts": [],
                "metadata": {
                    "intentId": unicode(uuid.uuid1()),
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
            "sessionId": unicode(uuid.uuid1())
        }
